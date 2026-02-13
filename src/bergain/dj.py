"""
DSPy RLM-powered streaming DJ.

The RLM selects a palette from a pre-built role map, writes its own
DJ loop, and streams audio in real-time through custom tool functions.
"""

import json
import signal
import sys
from collections import defaultdict

import dspy
from dotenv import load_dotenv

from bergain.indexer import build_index
from bergain.renderer import load_palette, render_bar
from bergain.streamer import AudioStreamer

CRITIQUE_INTERVAL = 16  # bars between trajectory checks
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


DJ_SIGNATURE = "sample_menu -> final_status"

DJ_INSTRUCTIONS = """\
You are a Berghain techno DJ streaming audio bar-by-bar in real-time.

`sample_menu` is a JSON dict mapping roles to available samples:
```json
{"kick": [{"path":"...","name":"...","loop":false,"dur":0.9}, ...], ...}
```

Available roles: kick, hihat, clap, bassline, perc, synth, texture, fx,
drum_loop, hat_loop.

------------------------------------------------------------------------

# Step 1: Pick Palette and Render Immediately

In your FIRST code block:

1. Parse `sample_menu` (it's a JSON string)
2. Pick ONE sample path per role you want to use
3. Call `set_palette(json.dumps({"kick":"path","hihat":"path",...}))`
4. Immediately render at least 32 bars

Example palette pick (adapt to actual paths):
```python
import json
menu = json.loads(sample_menu)
palette = {
    "kick": menu["kick"][0]["path"],
    "hihat": menu["hihat"][2]["path"],
    "bassline": menu["bassline"][0]["path"],
    "perc": menu["perc"][3]["path"],
    "texture": menu["texture"][1]["path"],
    "fx": menu["fx"][0]["path"],
}
set_palette(json.dumps(palette))
```

Do NOT write helper functions. Do NOT print commentary. Pick and render.

------------------------------------------------------------------------

# Bar Spec Format

```json
{"layers": [{"role":"kick","type":"oneshot","beats":[0,1,2,3],"gain":0.9}]}
```

- `"oneshot"`: triggers sample at beat positions (0-3 in 4/4, fractional ok)
- `"loop"`: stretches/loops sample across the full bar
- `"gain"`: 0.0 to 1.0 (texture/synth auto-capped at 0.55/0.50)

------------------------------------------------------------------------

# Sound Design Guidelines

Think like a Berghain DJ — the goal is GROOVE, WEIGHT, and ATMOSPHERE.

## Gain staging (make it punch)
- Kick: 0.90-0.95, 4-on-floor [0,1,2,3] — ALWAYS (this is techno)
- Hihat: 0.50-0.65 (clearly audible motor, not buried)
- Bassline: 0.60-0.75 (deep and present)
- Perc: 0.35-0.50 (rhythmic accent, not ghost)
- Texture: 0.30-0.50 (wash of atmosphere)
- Synth: 0.25-0.45 (textural bed)
- Clap: 0.40-0.55 (snappy backbeat when used)
- FX: 0.30-0.45 (transition marker)

## Rhythmic interest
- Hats: use offbeat patterns [0.5,1.5,2.5,3.5] for drive,
  or 16th-note [0,0.5,1,1.5,2,2.5,3,3.5] for intensity
- Perc: place on syncopated beats like [1.5], [2.5], [1,3], [0.5,2.5]
- Clap: on [1] or [1,3] for backbeat, NOT on every beat
- Layer two rhythmic elements with complementary patterns

## Set structure (intro → body → outro)

**Intro (first 16 bars):** Build up from almost nothing.
- Bars 1-4: kick alone (or kick + sparse hat)
- Bars 5-8: add hat motor, maybe ghost perc
- Bars 9-12: bring in bassline
- Bars 13-16: add texture/perc — groove is now locked

**Body (middle bars):** Full groove with evolution.
Use 8-bar phrases within 32-bar macro blocks:
- Bars 1-8: locked groove (kick + hat + bass)
- Bars 9-16: add perc, texture starts building
- Bars 17-24: full energy — all layers present, higher gains
- Bars 25-32: brief 2-4 bar dip (drop texture/perc), then rebuild

**Outro (last 16 bars):** Wind down by stripping layers.
- Remove texture/synth first
- Then drop perc
- Then thin out hats
- Last 4 bars: kick alone or kick + minimal hat, fading gain

## What makes it NOT dull
- Hats and perc at AUDIBLE gains (0.4-0.6), not ghosted at 0.15
- At least 3-4 layers active at peak moments
- Texture/atmosphere actually present, not barely perceptible
- Occasional rhythmic variation (shift perc beats every 8 bars)
- Use hat_loop or drum_loop roles for pre-made rhythmic texture

------------------------------------------------------------------------

# Subsequent Iterations

Your REPL state persists. All variables from iteration 1 are in scope.
Do NOT reparse sample_menu or call set_palette again.

Each iteration: render at least 32 bars with ONE evolutionary change:
- Swap hat pattern (offbeat → 16ths, or thin out)
- Add/remove a layer
- Shift gains for a new energy plateau
- Change perc beat placement
- 2-4 bar subtractive moment then restore

Read TRAJECTORY feedback from render responses — it tells you about
energy direction and mix state. Use it to decide your next move.

------------------------------------------------------------------------

# Hard Constraints

- `render_and_play_bar()` blocks — this IS your clock
- Do NOT add `time.sleep()`
- Do NOT spend iterations without rendering
- Do NOT reparse sample_menu after iteration 1
- Do NOT use `globals().get()` — variables persist directly
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
) -> None:
    """Build tools, configure DSPy, invoke RLM, handle shutdown."""
    load_dotenv()
    dspy.configure(lm=dspy.LM(lm))

    print(f"Indexing samples from {sample_dir}...")
    index = build_index(sample_dir)
    role_map = _build_role_map(index)
    role_map_json = json.dumps(role_map, indent=2)
    print(f"Indexed {len(index)} samples into {len(role_map)} roles:")
    for role, samples in role_map.items():
        print(f"  {role}: {len(samples)} samples")

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
    loaded_samples: dict = {}
    bars_played = [0]
    bar_history: list[dict] = []

    # --- RLM Tool Functions (closures over shared state) ---

    def set_palette(palette_json: str) -> str:
        """Load samples for the given palette. Call once before rendering.
        Input: JSON {"kick": "path/to/kick.wav", "hihat": "path/to/hihat.wav", ...}
        """
        nonlocal loaded_samples
        palette = json.loads(palette_json)
        loaded_samples = load_palette(palette, sample_rate)
        roles = list(loaded_samples.keys())
        print(f"  Palette loaded: {roles}")
        return f"Loaded {len(loaded_samples)} samples: {roles}"

    def render_and_play_bar(bar_spec_json: str) -> str:
        """Render one bar of audio and stream it. Blocks until buffer has space.
        Input: JSON {"layers": [{"role":"kick","type":"oneshot","beats":[0,2],"gain":0.9}, ...]}

        Mechanical guardrails (gain caps, density limits) are auto-applied.
        Every 16 bars, a trajectory observation is appended — use it to guide
        your creative decisions (you do NOT need to parse or regex-match it).
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
        if max_bars and bars_played[0] >= max_bars:
            print(f"\nReached {max_bars}-bar limit. Draining buffer...")
            if hasattr(streamer, "drain"):
                streamer.drain()
            streamer.stop()
            print(f"Total bars played: {bars_played[0]}")
            import os

            os._exit(0)
        return result

    def get_status() -> str:
        """Get current playback status.
        Returns a plain string like: "bars_played=160 buffer=12/64"
        """
        return (
            f"bars_played={bars_played[0]} "
            f"buffer={streamer.buffer_bars}/{streamer.audio_queue.maxsize}"
        )

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

    signature = dspy.Signature(DJ_SIGNATURE, instructions=DJ_INSTRUCTIONS)
    rlm = dspy.RLM(
        signature,
        tools=[set_palette, render_and_play_bar, get_status, get_history],
        max_iterations=100,
        max_llm_calls=500,
        verbose=verbose,
    )

    try:
        result = rlm(sample_menu=role_map_json)
        print(f"\nDJ set finished: {result.final_status}")
    except KeyboardInterrupt:
        print("\nDJ set interrupted.")
    finally:
        streamer.stop()
        print(f"Total bars played: {bars_played[0]}")
