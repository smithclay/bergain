"""
DSPy RLM-powered streaming DJ.

The RLM writes its own DJ loop in a sandboxed REPL, driving audio output
through mixer-control tool functions. Server-side mixer state replaces
client-side layer construction — the RLM uses add/fader/pattern/mute
controls instead of building JSON layer arrays.
"""

import copy
import json
import logging
import random
import signal
from collections import defaultdict

import dspy
from dotenv import load_dotenv
from dspy.primitives.python_interpreter import PythonInterpreter

from bergain.indexer import build_index
from bergain.renderer import load_palette, render_bar
from bergain.streamer import AudioStreamer

logger = logging.getLogger(__name__)


class ResilientInterpreter(PythonInterpreter):
    """PythonInterpreter that re-registers tools on every execute() and logs crashes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_code = None
        self._exec_count = 0

    def _ensure_deno_process(self):
        if self.deno_process is not None and self.deno_process.poll() is not None:
            stderr = self.deno_process.stderr.read() if self.deno_process.stderr else ""
            # Try to drain any remaining stdout for clues
            remaining_stdout = ""
            try:
                import select

                if (
                    self.deno_process.stdout
                    and select.select([self.deno_process.stdout], [], [], 0)[0]
                ):
                    remaining_stdout = self.deno_process.stdout.read()
            except Exception:
                pass
            logger.warning(
                "Deno sandbox died (exit %d) after %d executions.\n"
                "  last code: %s\n"
                "  stderr: %s\n"
                "  remaining stdout: %s",
                self.deno_process.returncode,
                self._exec_count,
                (self._last_code or "")[:500],
                stderr[:2000],
                remaining_stdout[:1000],
            )
        super()._ensure_deno_process()

    def execute(self, code, variables=None):
        self._tools_registered = False
        self._last_code = code
        self._exec_count += 1
        return super().execute(code, variables)


DEFAULT_ROLES = ["kick", "hihat", "bassline", "perc", "texture", "clap"]

CRITIQUE_INTERVAL = 8  # bars between trajectory checks
LLM_CRITIC_INTERVAL = 16  # bars between LLM critic calls
GAIN_CAP = {"texture": 0.55, "synth": 0.50}  # auto-enforced gain caps

ENERGY_CURVE = [0.25, 0.50, 0.75, 0.90, 0.90, 0.85, 0.60, 0.30]

NAMED_PATTERNS = {
    "four_on_floor": [0, 1, 2, 3],
    "offbeat_8ths": [0.5, 1.5, 2.5, 3.5],
    "syncopated_a": [0.5, 2.5],
    "syncopated_b": [1.5, 3.5],
    "backbeat": [1, 3],
    "sparse_accent": [1],
    "gallop": [0, 0.5, 2, 2.5],
    "sixteenth_drive": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
}

# Structural arc phases: (threshold, layer_cap, phase_name)
# Thresholds are fractions of total bars; first matching threshold wins.
_ARC_PHASES = [
    (0.125, 3, "intro phase"),
    (0.375, 6, "building phase"),
    (0.8125, 6, "full groove phase"),  # 4 more bars of peak (was 0.75)
    (0.9063, 4, "stripping phase"),  # shorter strip (was 0.875)
    (1.1, 2, "outro phase"),  # shorter outro
]
_DEFAULT_CAP = 6  # infinite mode: flat cap


def _phase_at(bar: int, total: int | None) -> tuple[int, str]:
    """Return (layer_cap, phase_name) at a bar position."""
    if total is None:
        return _DEFAULT_CAP, "infinite"
    pct = bar / max(total, 1)
    for threshold, cap, name in _ARC_PHASES:
        if pct < threshold:
            return cap, name
    return 2, "outro phase"


def _target_energy(bar: int, total: int | None) -> float:
    """Target energy at this bar position (0-1 normalized scale)."""
    if total is None:
        return 0.80
    segment = min(int(bar / max(total, 1) * len(ENERGY_CURVE)), len(ENERGY_CURVE) - 1)
    return ENERGY_CURVE[segment]


def _build_role_map(index: list[dict]) -> dict[str, list[dict]]:
    """Convert flat sample index into a role-keyed map for the RLM.

    Maps category names to DJ roles and groups samples by role.
    """
    category_to_role = {
        "kick": "kick",
        "hihat": "hihat",
        "clap": "clap",
        "clap_ride": "clap",
        "bassline": "bassline",
        "perc": "perc",
        "synth": "synth",
        "texture": "texture",
        "fx": "fx",
        "drum_loop": "drum_loop",
        "noise_hat_loop": "hat_loop",
        "drums_909": None,  # handled by sub_type
    }
    sub_type_to_role = {
        "kick": "kick",
        "hihat": "hihat",
        "clap": "clap",
        "snare": "perc",
        "ride": "hihat",
        "tom": "perc",
        "shaker": "perc",
        "crash": "fx",
        "rimshot": "perc",
    }
    role_map: dict[str, list[dict]] = defaultdict(list)
    for s in index:
        cat = s.get("category", "")
        sub = s.get("sub_type")

        # 909 drums: use sub_type
        if cat == "drums_909" and sub:
            role = sub_type_to_role.get(sub)
        else:
            role = category_to_role.get(cat)

        if role:
            entry = {
                "path": s["path"],
                "name": s["filename"],
                "loop": s.get("is_loop", False),
            }
            if s.get("duration_s"):
                entry["dur"] = round(s["duration_s"], 1)
            role_map[role].append(entry)

    return dict(role_map)


def _pick_random_palette(role_map: dict[str, list[dict]]) -> dict[str, str]:
    """Pick one random sample per default role from the role map."""
    palette = {}
    for role in DEFAULT_ROLES:
        if role in role_map and role_map[role]:
            palette[role] = random.choice(role_map[role])["path"]
    return palette


def _resolve_pattern(name_or_array: str) -> list:
    """Resolve a named pattern or JSON array string to beat positions."""
    lookup = name_or_array.strip().lower()
    if lookup in NAMED_PATTERNS:
        return list(NAMED_PATTERNS[lookup])
    try:
        parsed = json.loads(name_or_array)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    raise ValueError(
        f"Unknown pattern '{name_or_array}'. Known: {', '.join(NAMED_PATTERNS.keys())}"
    )


def _mixer_to_layers(mixer_state: dict[str, dict]) -> list[dict]:
    """Convert mixer_state into a layer list for the renderer."""
    layers = []
    for role, ch in mixer_state.items():
        if ch.get("muted"):
            continue
        layer = {"role": role, "type": ch["type"], "gain": ch["gain"]}
        if ch["type"] == "oneshot":
            layer["beats"] = ch["pattern"]
        layers.append(layer)
    return layers


def _auto_correct_bar(
    bar_spec: dict, bars_played: int = 0, max_bars: int | None = None
) -> tuple[dict, list[str]]:
    """Apply mechanical guardrails to a bar spec before rendering.

    Returns (corrected_spec, list of correction messages).
    """
    corrections: list[str] = []
    layers = bar_spec.get("layers", [])

    # Cap texture/synth gains
    for ly in layers:
        role = ly.get("role", "")
        if role in GAIN_CAP and ly.get("gain", 0) > GAIN_CAP[role]:
            old = ly["gain"]
            ly["gain"] = GAIN_CAP[role]
            corrections.append(f"auto-capped {role} {old:.2f}->{GAIN_CAP[role]}")

    # Enforce position-aware density ceiling: drop lowest-gain non-kick layers
    cap, phase = _phase_at(bars_played, max_bars)
    if len(layers) > cap:
        keep = [ly for ly in layers if ly.get("role") == "kick"]
        rest = sorted(
            [ly for ly in layers if ly.get("role") != "kick"],
            key=lambda ly: ly.get("gain", 0),
            reverse=True,
        )
        layers = keep + rest[: max(0, cap - len(keep))]
        corrections.append(f"auto-trimmed to {len(layers)} layers ({phase})")

    bar_spec["layers"] = layers
    return bar_spec, corrections


def _compute_critique(
    history: list[dict], bar: int = 0, total: int | None = None
) -> str | None:
    """Analyze trajectory and return creative guidance (not mechanical fixes)."""
    window = min(CRITIQUE_INTERVAL, len(history))
    if window < CRITIQUE_INTERVAL:
        return None

    recent = history[-window:]
    observations: list[str] = []

    # Energy trajectory
    energies = [sum(ly.get("gain", 0.5) for ly in b.get("layers", [])) for b in recent]
    mean_e = sum(energies) / len(energies)
    half = len(energies) // 2
    avg_first = sum(energies[:half]) / half
    avg_second = sum(energies[half:]) / (len(energies) - half)
    slope = (avg_second - avg_first) / len(energies)

    # Target energy comparison (normalize sum-of-gains to 0-1 scale)
    normalized_e = mean_e / max(_phase_at(bar, total)[0], 1)
    target = _target_energy(bar, total)
    delta = normalized_e - target
    if abs(delta) > 0.15:
        direction = "strip back" if delta > 0 else "build up"
        observations.append(
            f"energy: {normalized_e:.2f}, target: {target:.2f} — {direction}"
        )
    elif slope > 0.05:
        observations.append(
            f"energy rising (+{slope:.3f}/bar) — consider a plateau or subtractive moment"
        )
    elif slope < -0.05:
        observations.append(f"energy falling ({slope:.3f}/bar) — stabilize or rebuild")

    # Variance
    variance = sum((e - mean_e) ** 2 for e in energies) / len(energies)
    if variance > 0.5:
        observations.append("energy is unstable — hold a steady level longer")

    # Repetition — compare phrase-level snapshots, not individual bars
    # (bars within a single play() call share mixer state, so they're always identical)
    specs = [json.dumps(b, sort_keys=True) for b in recent]
    stride = max(CRITIQUE_INTERVAL, 1)
    phrase_specs = specs[::stride] if len(specs) >= stride else specs
    unique_ratio = len(set(phrase_specs)) / max(len(phrase_specs), 1)
    if len(phrase_specs) >= 2 and unique_ratio < 0.5:
        # Check history further back for persistent stagnation
        if len(history) >= CRITIQUE_INTERVAL * 3:
            older = [
                json.dumps(b, sort_keys=True) for b in history[-CRITIQUE_INTERVAL * 3 :]
            ]
            older_phrases = older[::stride]
            older_unique = len(set(older_phrases)) / max(len(older_phrases), 1)
            if older_unique < 0.5:
                observations.append(
                    "mix STILL stagnant after multiple phrases — STOP tweaking faders. "
                    "Do something structural: swap samples, try a breakdown(), "
                    "drop a layer and rebuild, or diagnose why energy isn't moving"
                )
            else:
                observations.append("mix is stagnant — introduce a variation")
        else:
            observations.append("mix is stagnant — introduce a variation")

    # Kick presence
    kick_pct = sum(
        1 for b in recent if any(ly["role"] == "kick" for ly in b.get("layers", []))
    ) / len(recent)
    if kick_pct < 0.85:
        observations.append(
            f"kick only in {kick_pct:.0%} of bars — keep it more present"
        )

    # Density trend
    densities = [len(b.get("layers", [])) for b in recent]
    avg_density = sum(densities) / len(densities)
    if avg_density > 5:
        observations.append(
            f"avg {avg_density:.1f} layers/bar — consider dropping a layer"
        )

    if not observations:
        return None
    return " | ".join(observations)


def _llm_critique(
    history: list[dict],
    total_bars_played: int,
    lm: dspy.LM,
    total_bars: int | None = None,
) -> str | None:
    """Ask an LM to critique the recent DJ set."""
    window = min(LLM_CRITIC_INTERVAL, len(history))
    if window < LLM_CRITIC_INTERVAL:
        return None

    recent = history[-window:]

    # Build compact summary for the critic
    energies = [
        round(sum(ly.get("gain", 0.5) for ly in b.get("layers", [])), 2) for b in recent
    ]
    # Role usage across the window
    role_counts: dict[str, int] = {}
    for bar in recent:
        for layer in bar.get("layers", []):
            role_counts[layer["role"]] = role_counts.get(layer["role"], 0) + 1
    # Last 4 bars as concrete examples
    last_4 = json.dumps(recent[-4:], indent=1)

    # Count active layers and groove changes for context
    active_layers = len(recent[-1].get("layers", [])) if recent else 0
    bar_snapshots = [json.dumps(b, sort_keys=True) for b in recent]
    unique_snapshots = len(set(bar_snapshots))
    changes_per_phrase = unique_snapshots / max(window // 8, 1)

    # Phase awareness
    _, phase = _phase_at(total_bars_played, total_bars)
    phase_guidance = {
        "intro phase": "We're in the INTRO. Suggest adding one element to start building.",
        "building phase": "We're BUILDING. Suggest adding layers or subtle adjustments to increase density.",
        "full groove phase": "We're at the PEAK. Keep it dense. Suggest holding, a fader ride, or ONE breakdown for tension/release.",
        "stripping phase": "We're STRIPPING DOWN. Suggest removing/muting layers. Do NOT suggest adding anything.",
        "outro phase": "We're in the OUTRO. Suggest continuing to strip toward kick-only. Do NOT suggest adding layers.",
        "infinite": "Open-ended set. Match advice to current density.",
    }
    phase_hint = phase_guidance.get(phase, "")

    prompt = f"""\
You are a Berghain resident DJ reviewing a set in progress (bar {total_bars_played}{f" of {total_bars}" if total_bars else ""}, phase: {phase}, reviewing last {window} bars).

Energy per bar: {energies}
Role usage: {json.dumps(role_counts)}
Active layers right now: {active_layers}
Last 4 bars: {last_4}
Groove changes: {unique_snapshots} distinct states in {window} bars ({changes_per_phrase:.1f} per phrase)

{phase_hint}

Give ONE specific, actionable direction for the next 8-16 bars.
Match your advice to BOTH the phase and density:
- Intro/building + sparse: suggest ADDING a layer.
- Building + 3-4 layers: suggest adding one more OR a fader ride (slow gain change over 8 bars).
- Peak + dense: suggest holding the groove, a fader ride, or a breakdown for tension/release.
- Stripping/outro: suggest MUTING or REMOVING layers (never fade to 0.00). Never add during strip/outro.
- Changing too often (>3/phrase): suggest "hold the current groove."
Fader rides (e.g., "slowly raise hihat from 0.50 to 0.65 over 8 bars") add movement without structural changes.
Express direction as concrete actions with timing (how many bars) and which roles."""

    try:
        response = lm(prompt, temperature=1.0)
        # dspy.LM may return list of strings or list of dicts
        item = response[0] if isinstance(response, list) else response
        if isinstance(item, dict):
            text = item.get("content") or item.get("text") or str(item)
        else:
            text = str(item)
        text = text.strip()
        if len(text) > 300:
            text = text[:297] + "..."
        return text
    except Exception as e:
        print(f"  >> LLM critic error: {e}")
        return None


DJ_SIGNATURE = "sample_menu -> final_status"

DJ_INSTRUCTIONS = """\
You are a Berghain resident DJ. Your job is to create a hypnotic, emotionally
compelling techno set — not to write careful software.

# ── MUSICAL PHILOSOPHY ──
# The dance floor responds to GROOVE, not to cleverness.
#
# GROOVE HIERARCHY (this is your entire job):
#   1. LOCK a groove — kick + one element, let it breathe 8-16 bars
#   2. RIDE it — hold what's working. Resist the urge to touch anything.
#   3. EVOLVE — 1-2 changes per phrase (8 bars). Add OR adjust, not both.
#   4. BREAK — strip to kick only (breakdown). THIS is where tension lives.
#   5. DROP — bring layers back. The return IS the emotional payoff.
#
# HYPNOTIC REPETITION is the point. A groove held for 16 bars is not stagnation
# — it's the thing that makes the breakdown hit. If you're changing 3+ things
# every phrase, you're fidgeting, not DJing.
#
# SUBTRACTIVE MIXING is your secret weapon — but only once you've BUILT
# something worth stripping. You need 4-5 layers before a breakdown hits hard.
# Use breakdown() deliberately — it's your strongest emotional tool, but only
# after the groove is fully developed.
#
# PACING RULES:
#   - 1-2 changes per phrase during BUILD. Fader nudges don't count.
#   - During PEAK: play("16") between actions — let the groove HYPNOTIZE.
#     One fader nudge or pattern change per 16-bar block. No more.
#   - After a breakdown, bring layers back across 2-3 phrases (not all at once)
#
# ENERGY ARC SHAPES (pick one and commit):
#   Hill:  build slowly → peak → strip slowly → end sparse
#   Ramp:  start minimal → build relentlessly → peak at the end
#   Wave:  build → strip → build higher → strip → final peak
#
# FADER DYNAMICS — movement without structural changes:
#   Ride faders for subtle energy shifts within a groove:
#   - Raise hihat from 0.50 → 0.65 over two phrases to build energy
#   - Drop texture from 0.40 → 0.28 to create space before a breakdown
#   - Nudge bassline gain up 0.05 each phrase during the build
#   NEVER fade to 0.00 — that's a silent ghost channel. Use mute() or remove()
#   to actually take a layer out. Fader rides stay WITHIN the gain range.
#   NEVER mute then unmute the same channel within 16 bars — that's indecisive.
#
# WHAT KILLS THE VIBE:
#   ✗ Fidgeting — changing gains/patterns every play() call
#   ✗ Front-loading — adding 4+ layers in the first 16 bars
#   ✗ No breakdowns — all build, no release = no emotional payoff
#   ✗ Mute/unmute oscillation — muting then unmuting within 16 bars sounds broken
#   ✗ Random swaps — changing samples without musical reason
#   ✗ Over-engineering — writing 50 lines of code per iteration instead of DJing

# ── TOOL REFERENCE ──
#   play(bars)              — advance N bars, returns JSON status. YOUR ONLY CLOCK.
#   breakdown(bars)         — mute all but kick for N bars, auto-restore. USE THIS.
#   add(role, gain, pattern) — add channel. Pattern = named pattern string (see below).
#   remove(role)            — drop channel permanently
#   fader(role, gain)       — adjust volume (gain as string, e.g. "0.58")
#   pattern(role, name)     — change beat pattern
#   swap(role, choice)      — "" to LIST, index string to SWAP, "random" for random
#   mute(role)/unmute(role) — toggle without losing settings
#   llm_query(prompt)       — ask sub-LM for creative judgment
#   llm_query_batched(prompts) — parallel sub-LM calls
#
#   Roles: kick, hihat, bassline, perc, texture, clap
#   Patterns: four_on_floor, offbeat_8ths, syncopated_a, syncopated_b,
#             backbeat, sparse_accent, gallop, sixteenth_drive
#   Gains: kick 0.82-1.00 | hihat 0.47-0.67 | bassline 0.57-0.77
#          perc 0.32-0.52 | texture 0.28-0.48 | clap 0.37-0.57

# ── SAFETY ESSENTIALS ──
# Pattern arg in add()/pattern() MUST be a named pattern — NEVER a filename.
#   WRONG: add("kick", "0.90", "909-Bassdrum.wav")  ← CRASHES sandbox
#   RIGHT: add("kick", "0.90", "four_on_floor") then swap("kick", "3")
# ALWAYS store play() return: status = json.loads(play("8"))
# NEVER hardcode mixer state from memory — read it from the status variable.
# NEVER use eval()/exec() on llm_query output — parse JSON, dispatch explicitly.
# If a tool errors, try once simpler, then skip it. Keep playing.
# If tools are missing (NameError), call SUBMIT("done") immediately.
# Keep iterations SHORT: 10-20 lines. You are DJing, not engineering.
# Variables persist across iterations — don't redeclare imports or state.
# swap() with sample index: use clean strings like "3", never filenames.

# ── DJ SET STRUCTURE ──
# FIRST ITERATION: explore sample_menu. Print categories, counts, names.
#   Note interesting sample indices. Ask llm_query for an opening vibe.
#
# INTRO (bars 1-16): Kick alone for 8 bars. Add hihat or perc. Let it breathe.
# BUILD (bars 16-64): ADD a new element every 8 bars. You MUST reach 4-5
#   active layers by bar 48. Add bassline, perc, texture, clap — keep building.
#   The set needs density before breakdowns have any impact.
# PEAK (bars 64-96): 4-6 layers running. HOLD the groove — don't strip yet.
#   Keep layers active, but the peak must EVOLVE, not freeze:
#   - Every 16 bars, change ONE pattern on a non-kick layer. This keeps the
#     groove moving without losing its foundation. Good peak transitions:
#     perc: syncopated_a → gallop (adds urgency)
#     hihat: offbeat_8ths → sixteenth_drive (intensifies)
#     bassline: syncopated_a → sixteenth_drive (drives harder)
#   - Ride faders alongside pattern changes for smooth transitions.
#   - The peak should feel like a JOURNEY, not a static wall of sound.
# BREAKDOWN (bar ~80): NOW strip to kick for 4-8 bars. The impact comes from
#   the contrast with the dense, evolving peak. Bring layers back over 2-3 phrases.
# STRIP (bars 96-112): Remove layers one at a time toward the outro.
# OUTRO (bars 112+): Fade to kick alone. Let the room breathe out.
#
# KEY: build UP during build phase, HOLD during peak, strip LATE.
# A breakdown at bar 24 with only 2 layers has zero impact.
#
# RESPONDING TO FEEDBACK:
#   "energy rising" → good if building, otherwise mute a layer
#   "energy falling" → good if stripping, otherwise check if you dropped too much
#   "stagnant" → you've been holding TOO long. Do something structural.
#   "unstable" → stop changing things. Lock the groove. play("16").
#   CRITIC feedback → read it, but don't obey blindly. You're the DJ.
#
# When the set reaches max_bars or a natural end, call SUBMIT("done").

Input: `sample_menu` — your record bag. Explore it before you play it.
Output: {final_status} via SUBMIT() when the set concludes.
"""


def _dump_lm_history(main_lm, cheap_lm, path: str) -> None:
    """Write full LLM prompt/response history to a JSONL file."""
    entries = []
    seen_lms = set()
    for lm, label in [(main_lm, "main"), (cheap_lm, "critic")]:
        if id(lm) in seen_lms:
            continue
        seen_lms.add(id(lm))
        for item in getattr(lm, "history", []):
            entries.append(
                {
                    "role": label,
                    "model": item.get("model", ""),
                    "timestamp": item.get("timestamp", ""),
                    "prompt": item.get("prompt"),
                    "messages": item.get("messages"),
                    "outputs": item.get("outputs", []),
                    "usage": item.get("usage", {}),
                }
            )
    entries.sort(key=lambda e: e.get("timestamp", ""))
    try:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
        print(f"LLM history ({len(entries)} calls) saved to {path}")
    except Exception as e:
        print(f"Failed to write LLM history: {e}")


class _DJComplete(Exception):
    """Raised when the DJ reaches its max-bars limit."""

    pass


def run_dj(
    sample_dir: str = "sample_pack",
    bpm: int = 128,
    sample_rate: int = 44100,
    lm: str = "openai/gpt-5-mini",
    verbose: bool = False,
    output: str | None = None,
    max_bars: int | None = None,
    critic_lm: str | None = None,
    palette_file: str | None = None,
    prompt_file: str | None = None,
    no_cache: bool = False,
) -> None:
    """Build tools, configure DSPy, invoke RLM, handle shutdown."""
    load_dotenv()
    lm_kwargs: dict = {}
    if no_cache:
        lm_kwargs["cache"] = False
        lm_kwargs["temperature"] = 1.0
    main_lm = dspy.LM(lm, **lm_kwargs)
    cheap_lm = dspy.LM(critic_lm, **lm_kwargs) if critic_lm else main_lm
    dspy.configure(lm=main_lm)

    print(f"Indexing samples from {sample_dir}...")
    index = build_index(sample_dir)
    role_map = _build_role_map(index)
    role_map_json = json.dumps(role_map, indent=2)
    print(f"Indexed {len(index)} samples into {len(role_map)} roles:")
    for role, samples in role_map.items():
        print(f"  {role}: {len(samples)} samples")

    # Always pre-load a palette: from file or randomly generated
    if palette_file:
        with open(palette_file) as f:
            palette = json.load(f)
    else:
        palette = _pick_random_palette(role_map)
    print(f"Palette: { {r: p.split('/')[-1] for r, p in palette.items()} }")

    if output:
        from bergain.streamer import FileWriter

        streamer = FileWriter(output_path=output, sample_rate=sample_rate)
    else:
        streamer = AudioStreamer(sample_rate=sample_rate)
    streamer.start()
    if output:
        print(f"Recording to {output} (BPM={bpm}, SR={sample_rate})")
    else:
        print(f"Audio streamer started (BPM={bpm}, SR={sample_rate})")

    # Shared mutable state for closures
    loaded_samples = load_palette(palette, sample_rate)
    bars_played = [0]
    bar_history: list[dict] = []
    mixer_state: dict[str, dict] = {}
    set_complete = [False]

    _COMPLETE_MSG = json.dumps(
        {
            "set_complete": True,
            "error": "SET COMPLETE — bar limit reached. Call SUBMIT('done') NOW.",
        }
    )

    interpreter = ResilientInterpreter()
    cleanup_done = [False]

    def _cleanup():
        if cleanup_done[0]:
            return
        cleanup_done[0] = True
        interpreter.shutdown()
        if hasattr(streamer, "drain"):
            streamer.drain()
        streamer.stop()
        print(f"Total bars played: {bars_played[0]}")
        if output:
            history_path = output.rsplit(".", 1)[0] + ".llm_history.jsonl"
            _dump_lm_history(main_lm, cheap_lm, history_path)

    # --- Internal helpers (closures over shared state) ---

    def _mixer_snapshot() -> dict:
        """Return a JSON-safe snapshot of mixer_state."""
        return {role: dict(ch) for role, ch in mixer_state.items()}

    def _respond(**data) -> str:
        """JSON response with mixer snapshot appended."""
        return json.dumps({**data, "mixer": _mixer_snapshot()})

    def _render_bars(n: int) -> list[str]:
        """Render N bars from current mixer state. Returns corrections.

        Stops early (without raising) if max_bars is reached.
        """
        all_corrections: list[str] = []
        for _ in range(n):
            if max_bars and bars_played[0] >= max_bars:
                break

            layers = _mixer_to_layers(mixer_state)
            bar_spec = {"layers": copy.deepcopy(layers)}

            bar_spec, corrections = _auto_correct_bar(
                bar_spec, bars_played[0], max_bars
            )

            audio = render_bar(bar_spec, loaded_samples, bpm, sample_rate)
            streamer.enqueue(audio)
            bars_played[0] += 1
            bar_history.append(bar_spec)

            if bars_played[0] % 8 == 0:
                print(f"  Bar {bars_played[0]} (buffer: {streamer.buffer_bars})")
            for c in corrections:
                print(f"  >> {c}")
            all_corrections.extend(corrections)

            if max_bars and bars_played[0] >= max_bars:
                print(f"\nReached {max_bars}-bar limit.")
        return all_corrections

    def _build_status(all_corrections: list[str], count: int) -> str:
        """Build JSON status response after rendering bars."""
        feedback_items = []
        if bars_played[0] % CRITIQUE_INTERVAL == 0:
            critique = _compute_critique(bar_history, bars_played[0], max_bars)
            if critique:
                feedback_items.append(f"TRAJECTORY: {critique}")
                print(f"  >> TRAJECTORY: {critique}")
        if bars_played[0] % LLM_CRITIC_INTERVAL == 0:
            llm_feedback = _llm_critique(
                bar_history, bars_played[0], cheap_lm, max_bars
            )
            if llm_feedback:
                feedback_items.append(f"CRITIC: {llm_feedback}")
                print(f"  >> CRITIC: {llm_feedback}")

        cap, phase = _phase_at(bars_played[0], max_bars)
        recent_energies = [
            sum(ly.get("gain", 0.5) for ly in b.get("layers", []))
            for b in bar_history[-count:]
        ]
        mean_energy = sum(recent_energies) / max(len(recent_energies), 1)
        normalized_energy = round(mean_energy / max(cap, 1), 2)
        target = round(_target_energy(bars_played[0], max_bars), 2)

        return json.dumps(
            {
                "bars_played": bars_played[0],
                "phase": phase,
                "energy": normalized_energy,
                "target_energy": target,
                "corrections": all_corrections,
                "feedback": "\n".join(feedback_items) if feedback_items else None,
                "mixer": _mixer_snapshot(),
                "buffer_depth": streamer.buffer_bars,
            }
        )

    # --- RLM Tool Functions ---

    def play(bars: str = "4") -> str:
        """Advance the mix by N bars. Returns JSON status with bars_played,
        phase, energy, target_energy, corrections, feedback, mixer,
        buffer_depth.  When the set reaches max_bars, the status includes
        set_complete=true — call SUBMIT('done') immediately."""
        if set_complete[0]:
            return _COMPLETE_MSG
        n = int(bars)
        all_corrections = _render_bars(n)
        status_str = _build_status(all_corrections, n)
        if max_bars and bars_played[0] >= max_bars:
            set_complete[0] = True
            status = json.loads(status_str)
            status["set_complete"] = True
            status["feedback"] = (
                "SET COMPLETE — you have reached the bar limit. "
                'Call SUBMIT("done") NOW.'
            )
            return json.dumps(status)
        return status_str

    def breakdown(bars: str = "4") -> str:
        """Mute all channels except kick for N bars, then auto-restore.
        Returns JSON status."""
        if set_complete[0]:
            return _COMPLETE_MSG
        n = int(bars)
        # Save current mute states
        saved = {role: ch.get("muted", False) for role, ch in mixer_state.items()}
        # Mute everything except kick
        for role, ch in mixer_state.items():
            if role != "kick":
                ch["muted"] = True
        print(f"  >> BREAKDOWN: {n} bars (kick only)")
        all_corrections = _render_bars(n)
        # Restore previous mute states
        for role, was_muted in saved.items():
            if role in mixer_state:
                mixer_state[role]["muted"] = was_muted
        print("  >> BREAKDOWN: restored")
        return _build_status(all_corrections, n)

    def add(
        role: str, gain: str, pattern: str = "four_on_floor", type: str = "oneshot"
    ) -> str:
        """Add a channel to the mixer. Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        # Auto-detect: if this role's palette sample is a loop, render as loop
        sample_entry = next(
            (s for s in role_map.get(role, []) if s["path"] == palette.get(role)),
            None,
        )
        if sample_entry and sample_entry.get("loop", False):
            type = "loop"
        else:
            type = "oneshot"
        resolved = _resolve_pattern(pattern)
        mixer_state[role] = {
            "type": type,
            "gain": float(gain),
            "pattern": resolved,
            "muted": False,
        }
        # Ensure sample is loaded
        if role not in loaded_samples:
            if role in palette:
                from pydub import AudioSegment

                seg = (
                    AudioSegment.from_file(palette[role])
                    .set_channels(1)
                    .set_frame_rate(sample_rate)
                )
                loaded_samples[role] = seg
            else:
                del mixer_state[role]
                return json.dumps(
                    {
                        "error": f"No sample loaded for '{role}'. "
                        f"Available roles: {list(palette.keys())}",
                    }
                )
        print(
            f"  >> Added channel: {role} (gain={gain}, pattern={pattern}, type={type})"
        )
        return _respond(added=role)

    def remove(role: str) -> str:
        """Remove a channel from the mixer. Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        removed = mixer_state.pop(role, None)
        if removed is None:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        print(f"  >> Removed channel: {role}")
        return _respond(removed=role)

    def fader(role: str, gain: str) -> str:
        """Adjust a channel's volume. Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        g = float(gain)
        if g <= 0.0:
            return json.dumps(
                {
                    "error": f"Gain must be > 0. Use mute('{role}') or remove('{role}') to silence a channel."
                }
            )
        mixer_state[role]["gain"] = g
        print(f"  >> Fader: {role} -> {gain}")
        return _respond(role=role, gain=g)

    def pattern(role: str, name_or_array: str) -> str:
        """Change a channel's beat pattern. Accepts named patterns or JSON
        arrays. Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        try:
            resolved = _resolve_pattern(name_or_array)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        mixer_state[role]["pattern"] = resolved
        print(f"  >> Pattern: {role} -> {name_or_array} ({resolved})")
        return _respond(role=role, pattern=resolved)

    def mute(role: str) -> str:
        """Mute a channel (preserves all settings). Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        mixer_state[role]["muted"] = True
        print(f"  >> Muted: {role}")
        return _respond(muted=role)

    def unmute(role: str) -> str:
        """Unmute a channel. Returns JSON with mixer snapshot."""
        if set_complete[0]:
            return _COMPLETE_MSG
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        mixer_state[role]["muted"] = False
        print(f"  >> Unmuted: {role}")
        return _respond(unmuted=role)

    def swap(role: str, choice: str = "") -> str:
        """Browse alternatives or swap sample for a role.
        - swap("hihat") → list alternatives
        - swap("hihat", "3") → swap to index 3
        - swap("hihat", "random") → swap randomly
        Returns: JSON string."""
        if set_complete[0]:
            return _COMPLETE_MSG
        alternatives = role_map.get(role, [])
        if not alternatives:
            return json.dumps({"error": f"No alternatives for {role}"})

        # List mode: no choice provided
        if not choice:
            current = palette.get(role, "")
            options = []
            for i, s in enumerate(alternatives):
                entry = {"index": i, "name": s["name"], "loop": s.get("loop", False)}
                if s["path"] == current:
                    entry["current"] = True
                options.append(entry)
            return json.dumps(
                {
                    "role": role,
                    "current": current.split("/")[-1],
                    "alternatives": options,
                }
            )

        # Swap mode
        if choice == "random":
            picked = random.choice(alternatives)
        else:
            try:
                idx = int(choice)
            except ValueError:
                return json.dumps(
                    {"error": f"Invalid choice '{choice}' — use a number or 'random'"}
                )
            if idx < 0 or idx >= len(alternatives):
                return json.dumps(
                    {
                        "error": f"Index {idx} out of range for {role} (0-{len(alternatives) - 1})"
                    }
                )
            picked = alternatives[idx]

        from pydub import AudioSegment

        seg = (
            AudioSegment.from_file(picked["path"])
            .set_channels(1)
            .set_frame_rate(sample_rate)
        )
        loaded_samples[role] = seg
        palette[role] = picked["path"]
        print(f"  Swapped {role} -> {picked['name']}")
        return json.dumps(
            {
                "swapped": role,
                "to": picked["name"],
                "active": list(loaded_samples.keys()),
            }
        )

    # --- Signal handling ---
    def _shutdown(signum, frame):
        print("\nShutting down DJ...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown)

    # --- Invoke RLM ---
    print("Starting DJ set...")
    if max_bars:
        print(f"Will stop after {max_bars} bars.")
    print("Press Ctrl+C to stop.\n")

    if prompt_file:
        with open(prompt_file) as f:
            base_instructions = f.read()
        print(f"Using custom prompt from {prompt_file}")
    else:
        base_instructions = DJ_INSTRUCTIONS
    instructions = base_instructions + f"\nLoaded roles: {list(loaded_samples.keys())}"

    signature = dspy.Signature(DJ_SIGNATURE, instructions=instructions)
    rlm = dspy.RLM(
        signature,
        tools=[play, breakdown, add, remove, fader, pattern, mute, unmute, swap],
        max_iterations=100,
        max_llm_calls=500,
        verbose=verbose,
        sub_lm=cheap_lm,
        interpreter=interpreter,
    )

    try:
        result = rlm(sample_menu=role_map_json)
        print(f"\nDJ set finished: {result.final_status}")
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
