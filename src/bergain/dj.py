"""
DSPy RLM-powered streaming DJ.

The RLM writes its own DJ loop in a sandboxed REPL, driving audio output
through mixer-control tool functions. Server-side mixer state replaces
client-side layer construction — the RLM uses add/fader/pattern/mute
controls instead of building JSON layer arrays.
"""

import copy
import json
import random
import signal
import sys
from collections import defaultdict

import dspy
from dotenv import load_dotenv

from bergain.indexer import build_index
from bergain.renderer import load_palette, render_bar
from bergain.streamer import AudioStreamer

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
    (0.125, 2, "intro phase"),
    (0.25, 4, "building phase"),
    (0.75, 6, "full groove phase"),
    (0.875, 4, "stripping phase"),
    (1.1, 2, "outro phase"),
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

    # Repetition
    specs = [json.dumps(b, sort_keys=True) for b in recent]
    unique_ratio = len(set(specs)) / len(specs)
    if unique_ratio < 0.2:
        observations.append("mix is stagnant — introduce a variation")
    elif unique_ratio > 0.9:
        observations.append("too much variation — repeat patterns longer")

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
    history: list[dict], total_bars_played: int, lm: dspy.LM
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

    prompt = f"""\
You are a Berghain resident DJ reviewing a set in progress (bar {total_bars_played}, reviewing last {window} bars).

Energy per bar: {energies}
Role usage: {json.dumps(role_counts)}
Last 4 bars: {last_4}

Give ONE specific, actionable direction for the next 8-16 bars.
Express it as concrete actions the DJ can take:
- Change a role's gain (e.g., "drop hihat gain to 0.3 for 8 bars")
- Change beat positions (e.g., "move perc from backbeat to syncopated")
- Swap a sample (e.g., "swap hihat for something brighter")
- Add/remove a layer (e.g., "drop texture for 4 bars, then bring it back")
Be specific about timing (how many bars) and which roles to change."""

    try:
        response = lm(prompt, temperature=0.9)
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
You are a Berghain resident DJ. The room breathes — layers enter and leave,
gains shift constantly, the groove evolves every few bars. A flat mix
clears the dance floor.

# Setup — build momentum through rapid layering
add("kick", "0.92", "four_on_floor")
play("4")
add("hihat", "0.55", "offbeat_8ths")
play("4")
add("bassline", "0.65", type="loop")
fader("bassline", "0.72")
play("4")
add("perc", "0.40", "syncopated_a")
add("texture", "0.30", type="loop")
play("4")
add("clap", "0.45", "backbeat")
play("4")                                  # all 6 channels on

# Main loop — MANDATORY: make 2-3 mixer moves between EVERY play() call
while True:
    status = json.loads(play("8"))
    feedback = status.get("feedback")

    if feedback:
        direction = llm_query(
            f"Feedback: {feedback}\\n"
            f"Mixer: {json.dumps(status['mixer'])}\\n"
            f"Phase: {status['phase']}, Energy: {status['energy']}\\n"
            "Respond with precise mixer command. Examples: fader('hihat', '0.58'), pattern('perc', 'gallop'), swap('texture', 'random'), breakdown('4'), mute('clap'). Be surgical."
        )
        # Parse direction and apply it

    # MANDATORY: Execute 2-3 mixer moves every phrase — prioritize pattern() and swap():
    # pattern("perc", "gallop")  — shift the rhythm (USE FREQUENTLY)
    # swap("hihat", "random")    — refresh a stale sound (USE FREQUENTLY)
    # fader("hihat", "0.58")     — nudge gain ±0.03-0.08
    # breakdown("4")             — tension moment (use sparingly)
    # mute("texture")            — create space
    # Example: pattern("perc", "syncopated_b") + swap("texture", "random") + fader("kick", "0.93")

# Feedback response guide
# "build up"  → add() new channel or unmute() existing + fader() boost
# "strip back" → mute() non-essential or remove() + fader() cuts
# "stagnant"  → pattern() + swap() combo or breakdown() for reset
# "too busy"  → mute() texture/perc layers + fader() reductions
# "needs energy" → pattern() to sixteenth_drive + fader() boosts
# CRITIC      → execute exact command given

# Phase targets (active channels)
# intro: 1-2 | building: 3-4 | peak: 4-6 | stripping: 2-3 | outro: 1

# Gain ranges — tighter control for precision
# kick: 0.90-0.94 | hihat: 0.52-0.62 | bassline: 0.62-0.72
# perc: 0.37-0.47 | texture: 0.30-0.45 | clap: 0.42-0.52

# Patterns: four_on_floor, offbeat_8ths, syncopated_a, syncopated_b,
#   backbeat, sparse_accent, gallop, sixteenth_drive

# Rules
# - play("8") is one phrase. MANDATORY: 2-3 mixer moves between phrases.
# - play() blocks — it IS your only clock and timing mechanism.
# - Variables persist — do NOT redeclare them.
# - Call SUBMIT("done") only when stopping.
"""


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
) -> None:
    """Build tools, configure DSPy, invoke RLM, handle shutdown."""
    load_dotenv()
    main_lm = dspy.LM(lm)
    cheap_lm = dspy.LM(critic_lm) if critic_lm else main_lm
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

    # --- Internal helpers (closures over shared state) ---

    def _mixer_snapshot() -> dict:
        """Return a JSON-safe snapshot of mixer_state."""
        return {role: dict(ch) for role, ch in mixer_state.items()}

    def _respond(**data) -> str:
        """JSON response with mixer snapshot appended."""
        return json.dumps({**data, "mixer": _mixer_snapshot()})

    def _render_bars(n: int) -> list[str]:
        """Render N bars from current mixer state. Returns corrections."""
        all_corrections: list[str] = []
        for _ in range(n):
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
                print(f"\nReached {max_bars}-bar limit. Draining buffer...")
                if hasattr(streamer, "drain"):
                    streamer.drain()
                streamer.stop()
                print(f"Total bars played: {bars_played[0]}")
                import os

                os._exit(0)
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
            llm_feedback = _llm_critique(bar_history, bars_played[0], cheap_lm)
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
        buffer_depth."""
        n = int(bars)
        all_corrections = _render_bars(n)
        return _build_status(all_corrections, n)

    def breakdown(bars: str = "4") -> str:
        """Mute all channels except kick for N bars, then auto-restore.
        Returns JSON status."""
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
        resolved = _resolve_pattern(pattern) if type == "oneshot" else []
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
        removed = mixer_state.pop(role, None)
        if removed is None:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        print(f"  >> Removed channel: {role}")
        return _respond(removed=role)

    def fader(role: str, gain: str) -> str:
        """Adjust a channel's volume. Returns JSON with mixer snapshot."""
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        mixer_state[role]["gain"] = float(gain)
        print(f"  >> Fader: {role} -> {gain}")
        return _respond(role=role, gain=float(gain))

    def pattern(role: str, name_or_array: str) -> str:
        """Change a channel's beat pattern. Accepts named patterns or JSON
        arrays. Returns JSON with mixer snapshot."""
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
        if role not in mixer_state:
            return json.dumps({"error": f"No channel '{role}' on mixer"})
        mixer_state[role]["muted"] = True
        print(f"  >> Muted: {role}")
        return _respond(muted=role)

    def unmute(role: str) -> str:
        """Unmute a channel. Returns JSON with mixer snapshot."""
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
        streamer.stop()
        sys.exit(0)

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
    )

    try:
        result = rlm(sample_menu=role_map_json)
        print(f"\nDJ set finished: {result.final_status}")
    except KeyboardInterrupt:
        print("\nDJ set interrupted.")
    finally:
        streamer.stop()
        print(f"Total bars played: {bars_played[0]}")
