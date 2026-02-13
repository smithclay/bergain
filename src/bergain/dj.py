"""
DSPy RLM-powered streaming DJ.

The RLM writes its own DJ loop in a sandboxed REPL, driving audio output
through tool functions. Palette is pre-loaded; the RLM focuses on rendering
bars and evolving the set.
"""

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

BEAT_PATTERNS = {
    "FOUR_ON_FLOOR": [0, 1, 2, 3],
    "OFFBEAT_8THS": [0.5, 1.5, 2.5, 3.5],
    "SYNCOPATED_A": [0.5, 2.5],
    "SYNCOPATED_B": [1.5, 3.5],
    "BACKBEAT": [1, 3],
    "SPARSE_ACCENT": [1],
    "GALLOP": [0, 0.5, 2, 2.5],
    "SIXTEENTH_DRIVE": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
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


def _max_layers_at(bar: int, total: int | None) -> int:
    """Progressive density cap: intro -> peak -> outro."""
    if total is None:
        return _DEFAULT_CAP
    pct = bar / max(total, 1)
    for threshold, cap, _ in _ARC_PHASES:
        if pct < threshold:
            return cap
    return 2


def _phase_name_at(bar: int, total: int) -> str:
    """Return the structural phase name at a bar position."""
    pct = bar / max(total, 1)
    for threshold, _, name in _ARC_PHASES:
        if pct < threshold:
            return name
    return "outro phase"


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
    cap = _max_layers_at(bars_played, max_bars)
    if len(layers) > cap:
        keep = [ly for ly in layers if ly.get("role") == "kick"]
        rest = sorted(
            [ly for ly in layers if ly.get("role") != "kick"],
            key=lambda ly: ly.get("gain", 0),
            reverse=True,
        )
        layers = keep + rest[: max(0, cap - len(keep))]
        if max_bars is not None:
            phase = _phase_name_at(bars_played, max_bars)
            corrections.append(f"auto-trimmed to {len(layers)} layers ({phase})")
        else:
            corrections.append(f"auto-trimmed to {len(layers)} layers")

    bar_spec["layers"] = layers
    return bar_spec, corrections


def _compute_critique(
    history: list[dict], bar: int = 0, total: int | None = None
) -> str | None:
    """Analyze trajectory and return creative guidance (not mechanical fixes)."""
    window = min(CRITIQUE_INTERVAL, len(history))
    if window < 8:
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
    normalized_e = mean_e / _DEFAULT_CAP
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


def _llm_critique(history: list[dict], lm: dspy.LM) -> str | None:
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
You are a Berghain resident DJ reviewing a set in progress ({window} bars so far).

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
You are a Berghain techno DJ streaming audio bar-by-bar in real-time.

A palette of samples is already loaded. Start rendering bars immediately
in your FIRST code block — no setup needed.

`sample_menu` is a JSON dict of ALL available samples by role. You only
need it if you want to swap a sample later via `swap_sample()`.

# Bar Spec Format

```json
{"layers": [{"role":"kick","type":"oneshot","beats":[0,1,2,3],"gain":0.9}]}
```

- `"oneshot"`: triggers sample at beat positions (0-3 in 4/4, fractional ok)
- `"loop"`: stretches/loops sample across the full bar
- `"gain"`: 0.0 to 1.0 (texture/synth auto-capped at 0.55/0.50)

# Sound Design

Think Berghain — GROOVE, WEIGHT, ATMOSPHERE.

Gain staging:
- Kick: 0.90-0.95, 4-on-floor [0,1,2,3] — ALWAYS
- Hihat: 0.50-0.65 — offbeat [0.5,1.5,2.5,3.5] or 16ths for intensity
- Bassline: 0.60-0.75 — deep and present
- Perc: 0.35-0.50 — syncopated [1.5], [2.5], [1,3], [0.5,2.5]
- Texture: 0.30-0.50 — atmosphere, use type "loop"
- Clap: 0.40-0.55 — on [1] or [1,3], NOT every beat

Intro (first 16 bars): build from kick alone → add hat → bassline → full groove.
Body: full groove with slow evolution over 32-bar arcs.
Outro (last 16 bars): strip layers back to kick alone.

Keep hats/perc at AUDIBLE gains (0.4+), at least 3-4 layers at peak moments.

# Beat Patterns (use these names or raw arrays)
FOUR_ON_FLOOR = [0,1,2,3]        # kick standard
OFFBEAT_8THS  = [0.5,1.5,2.5,3.5] # classic hat
SYNCOPATED_A  = [0.5, 2.5]        # minimal perc
SYNCOPATED_B  = [1.5, 3.5]        # displaced feel
BACKBEAT      = [1, 3]            # clap/snare standard
SPARSE_ACCENT = [1]               # less-is-more
GALLOP        = [0, 0.5, 2, 2.5]  # driving energy
SIXTEENTH_DRIVE    = [0,0.5,1,1.5,2,2.5,3,3.5]  # maximum intensity

Variation idea: switch hihat from OFFBEAT_8THS to SYNCOPATED_A for 4 bars.

# Evolution

Berghain pacing: changes happen SLOWLY. Ride a groove for 32+ bars.

Each iteration: render 8 bars, then review any feedback before continuing.
Your loop should look like:
    for i in range(8):
        result = render_and_play_bar(...)
    feedback = check_feedback()  # ALWAYS call after each 8-bar block
    # Read feedback and adjust before next block

You do NOT have to change something every iteration — sometimes hold
the groove and let it breathe.

When you DO evolve (every 2-3 iterations), pick ONE subtle move:
- Shift a gain to reshape energy
- Move a perc hit to a different beat position
- Swap a sample: call `list_alternatives("hihat")` to see options,
  then `swap_sample("hihat", "3")` or `swap_sample("hihat", "random")`
- 2-4 bar subtractive moment (drop a layer) then restore

Bigger structural changes (new role, different bassline) every 48-64 bars.

# Feedback

Every bar, `render_and_play_bar()` returns bar count and buffer depth.
Every 8 bars: TRAJECTORY feedback (energy direction, density, repetition).
Every 16 bars: CRITIC feedback (specific, actionable creative direction).
After every 8-bar block, call `check_feedback()`. Read the feedback and adjust before rendering the next block.
Prioritize CRITIC over TRAJECTORY when they conflict.

# Constraints

- `render_and_play_bar()` blocks — this IS your clock
- Do NOT add `time.sleep()`
- Do NOT spend iterations without rendering
- Variables persist between iterations — do NOT redeclare them
- Call `SUBMIT("done")` only when stopping
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
    pending_feedback: list[str] = []

    # --- RLM Tool Functions (closures over shared state) ---

    def list_alternatives(role: str) -> str:
        """List available alternative samples for a role.
        Input: role name (e.g., "hihat")
        Returns: numbered list of alternatives."""
        alternatives = role_map.get(role, [])
        current = palette.get(role, "")
        lines = [f"Current {role}: {current.split('/')[-1]}"]
        for i, s in enumerate(alternatives):
            if s["path"] != current:
                marker = " (loop)" if s.get("loop") else ""
                lines.append(f"  {i}: {s['name']}{marker}")
        return "\n".join(lines)

    def swap_sample(role: str, index: str = "random") -> str:
        """Swap a sample by role and index number.
        Examples: swap_sample("hihat", "3") or swap_sample("perc", "random")"""
        alternatives = role_map.get(role, [])
        if not alternatives:
            return f"No alternatives for {role}"
        if index == "random":
            choice = random.choice(alternatives)
        else:
            try:
                idx = int(index)
            except ValueError:
                return f"Invalid index '{index}' — use a number or 'random'"
            if idx < 0 or idx >= len(alternatives):
                return f"Invalid index {idx} for {role} (0-{len(alternatives) - 1} available)"
            choice = alternatives[idx]
        from pydub import AudioSegment

        seg = (
            AudioSegment.from_file(choice["path"])
            .set_channels(1)
            .set_frame_rate(sample_rate)
        )
        loaded_samples[role] = seg
        palette[role] = choice["path"]
        print(f"  Swapped {role} -> {choice['name']}")
        return (
            f"Swapped {role} to {choice['name']}. Active: {list(loaded_samples.keys())}"
        )

    def render_and_play_bar(bar_spec_json: str) -> str:
        """Render one bar of audio and stream it. Blocks until buffer has space.
        Input: JSON {"layers": [{"role":"kick","type":"oneshot","beats":[0,2],"gain":0.9}, ...]}
        """
        bar_spec = json.loads(bar_spec_json)

        # Auto-apply mechanical corrections before rendering
        bar_spec, corrections = _auto_correct_bar(bar_spec, bars_played[0], max_bars)
        audio = render_bar(bar_spec, loaded_samples, bpm, sample_rate)
        streamer.enqueue(audio)
        bars_played[0] += 1
        bar_history.append(bar_spec)

        if bars_played[0] % 8 == 0:
            print(f"  Bar {bars_played[0]} (buffer: {streamer.buffer_bars})")
        for c in corrections:
            print(f"  >> {c}")

        result = f"bar={bars_played[0]} buf={streamer.buffer_bars}"
        if corrections:
            result += " [" + ", ".join(corrections) + "]"
        if bars_played[0] % CRITIQUE_INTERVAL == 0:
            critique = _compute_critique(bar_history, bars_played[0], max_bars)
            if critique:
                pending_feedback.append(f"TRAJECTORY: {critique}")
                result += " | FEEDBACK_AVAILABLE"
                print(f"  >> TRAJECTORY: {critique}")
        if bars_played[0] % LLM_CRITIC_INTERVAL == 0:
            llm_feedback = _llm_critique(bar_history, cheap_lm)
            if llm_feedback:
                pending_feedback.append(f"CRITIC: {llm_feedback}")
                result += " | FEEDBACK_AVAILABLE"
                print(f"  >> CRITIC: {llm_feedback}")
        if max_bars and bars_played[0] >= max_bars:
            print(f"\nReached {max_bars}-bar limit. Draining buffer...")
            if hasattr(streamer, "drain"):
                streamer.drain()
            streamer.stop()
            print(f"Total bars played: {bars_played[0]}")
            import os

            os._exit(0)
        return result

    def get_history(last_n: str = "32") -> str:
        """Get a summary of recent DJ history.
        Input: number of recent bars to summarize (as string, e.g. "32").
        Returns: JSON string with keys:
          - bars_summarized (int)
          - role_frequency (dict: role -> count)
          - energy_per_4bars (list of floats)
          - last_4_bars (list of bar spec dicts)
        Parse with json.loads().
        """
        n = int(last_n)
        recent = bar_history[-n:]
        # Role frequency counts
        role_counts: dict[str, int] = {}
        for bar in recent:
            for layer in bar.get("layers", []):
                role_counts[layer["role"]] = role_counts.get(layer["role"], 0) + 1
        # Energy per 4-bar group
        energy_trajectory = []
        for i in range(0, len(recent), 4):
            chunk = recent[i : i + 4]
            avg = sum(
                sum(ly.get("gain", 0.5) for ly in b.get("layers", [])) for b in chunk
            ) / max(len(chunk), 1)
            energy_trajectory.append(round(avg, 2))
        return json.dumps(
            {
                "bars_summarized": len(recent),
                "role_frequency": role_counts,
                "energy_per_4bars": energy_trajectory,
                "last_4_bars": recent[-4:],
            }
        )

    def check_feedback() -> str:
        """Check for trajectory and critic feedback. Call this between bar groups.
        Returns: feedback text, or 'none' if no feedback pending."""
        if not pending_feedback:
            return "none"
        result = "\n".join(pending_feedback)
        pending_feedback.clear()
        return result

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

    pattern_vars = "\n".join(f"{k} = {v}" for k, v in BEAT_PATTERNS.items())
    instructions = (
        DJ_INSTRUCTIONS
        + f"\nLoaded roles: {list(loaded_samples.keys())}\n\n# Available as variables:\n{pattern_vars}"
    )

    signature = dspy.Signature(DJ_SIGNATURE, instructions=instructions)
    rlm = dspy.RLM(
        signature,
        tools=[
            swap_sample,
            render_and_play_bar,
            get_history,
            check_feedback,
            list_alternatives,
        ],
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
