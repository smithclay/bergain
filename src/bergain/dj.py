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
MAX_LAYERS = 6  # auto-enforced density ceiling
GAIN_CAP = {"texture": 0.55, "synth": 0.50}  # auto-enforced gain caps


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


def _auto_correct_bar(bar_spec: dict) -> tuple[dict, list[str]]:
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

    # Enforce density ceiling: drop lowest-gain non-kick layers
    if len(layers) > MAX_LAYERS:
        keep = [ly for ly in layers if ly.get("role") == "kick"]
        rest = sorted(
            [ly for ly in layers if ly.get("role") != "kick"],
            key=lambda ly: ly.get("gain", 0),
            reverse=True,
        )
        layers = keep + rest[: MAX_LAYERS - len(keep)]
        corrections.append(f"auto-trimmed to {len(layers)} layers")

    bar_spec["layers"] = layers
    return bar_spec, corrections


def _compute_critique(history: list[dict]) -> str | None:
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
    if slope > 0.05:
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
    if window < 16:
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

# Evolution

Berghain pacing: changes happen SLOWLY. Ride a groove for 32+ bars.

Each iteration: render 8 bars, then review any feedback before continuing.
Your loop should look like:
    for i in range(8):
        result = render_and_play_bar(...)
    # After 8 bars: check result for TRAJECTORY/CRITIC feedback
    # Make ONE subtle adjustment before the next 8-bar block

You do NOT have to change something every iteration — sometimes hold
the groove and let it breathe.

When you DO evolve (every 2-3 iterations), pick ONE subtle move:
- Shift a gain to reshape energy
- Move a perc hit to a different beat position
- Swap a sample: parse `sample_menu` to find a path, then call
  `swap_sample(json.dumps({"role":"hihat","path":"sample_pack/..."}))`
- 2-4 bar subtractive moment (drop a layer) then restore

Bigger structural changes (new role, different bassline) every 48-64 bars.

# Feedback

Every bar, `render_and_play_bar()` returns bar count and buffer depth.
Every 8 bars: TRAJECTORY feedback (energy direction, density, repetition).
Every 16 bars: CRITIC feedback (specific, actionable creative direction).
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

    # --- RLM Tool Functions (closures over shared state) ---

    def swap_sample(swap_json: str) -> str:
        """Swap a single sample in the palette without affecting other roles.
        Input: JSON {"role": "hihat", "path": "sample_pack/hihat/HH_02.wav"}
        """
        spec = json.loads(swap_json)
        role = spec["role"]
        path = spec["path"]
        from pydub import AudioSegment

        seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
        loaded_samples[role] = seg
        print(f"  Swapped {role} -> {path.split('/')[-1]}")
        return f"Swapped {role}. Active roles: {list(loaded_samples.keys())}"

    def render_and_play_bar(bar_spec_json: str) -> str:
        """Render one bar of audio and stream it. Blocks until buffer has space.
        Input: JSON {"layers": [{"role":"kick","type":"oneshot","beats":[0,2],"gain":0.9}, ...]}
        """
        bar_spec = json.loads(bar_spec_json)

        # Auto-apply mechanical corrections before rendering
        bar_spec, corrections = _auto_correct_bar(bar_spec)
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
            critique = _compute_critique(bar_history)
            if critique:
                result += f"\nTRAJECTORY: {critique}"
                print(f"  >> TRAJECTORY: {critique}")
        if bars_played[0] % LLM_CRITIC_INTERVAL == 0:
            llm_feedback = _llm_critique(bar_history, cheap_lm)
            if llm_feedback:
                result += f"\nCRITIC: {llm_feedback}"
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

    instructions = DJ_INSTRUCTIONS + (f"\nLoaded roles: {list(loaded_samples.keys())}")

    signature = dspy.Signature(DJ_SIGNATURE, instructions=instructions)
    rlm = dspy.RLM(
        signature,
        tools=[swap_sample, render_and_play_bar, get_history],
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
