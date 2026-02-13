"""
DSPy RLM-powered streaming DJ.

The RLM explores the sample index, selects a palette, writes its own
DJ loop, and streams audio in real-time through custom tool functions.
"""

import json
import signal
import sys

import dspy
from dotenv import load_dotenv

from bergain.indexer import build_index
from bergain.renderer import load_palette, render_bar
from bergain.streamer import AudioStreamer

CRITIQUE_INTERVAL = 16  # bars between critique checks


def _compute_critique(history: list[dict]) -> str | None:
    """Analyze recent bars and return a critique directive, or None if on track."""
    window = min(CRITIQUE_INTERVAL, len(history))
    if window < 8:
        return None

    recent = history[-window:]
    issues: list[str] = []

    # --- Density ---
    densities = [len(b.get("layers", [])) for b in recent]
    avg_density = sum(densities) / len(densities)
    if avg_density > 5:
        issues.append(f"OVER-DENSE avg={avg_density:.1f}/bar (want <5). Drop a layer.")
    elif max(densities) > 6:
        issues.append(f"DENSITY SPIKE {max(densities)} layers. Keep max <=5.")

    # --- Energy trajectory ---
    energies = [sum(ly.get("gain", 0.5) for ly in b.get("layers", [])) for b in recent]
    mean_e = sum(energies) / len(energies)
    half = len(energies) // 2
    avg_first = sum(energies[:half]) / half
    avg_second = sum(energies[half:]) / (len(energies) - half)
    slope = (avg_second - avg_first) / len(energies)
    if slope > 0.05:
        issues.append(f"ENERGY RISING +{slope:.3f}/bar. Plateau or subtract.")
    elif slope < -0.05:
        issues.append(f"ENERGY DROPPING {slope:.3f}/bar. Stabilize.")

    # Energy variance — low = hypnotic
    variance = sum((e - mean_e) ** 2 for e in energies) / len(energies)
    if variance > 0.5:
        issues.append(f"ENERGY UNSTABLE var={variance:.2f}. Steady plateau needed.")

    # --- Repetition ---
    specs = [json.dumps(b, sort_keys=True) for b in recent]
    unique_ratio = len(set(specs)) / len(specs)
    if unique_ratio < 0.2:
        issues.append("STAGNANT. Introduce micro-variation (gain shift, ghost note).")
    elif unique_ratio > 0.9:
        issues.append("TOO VARIED. Repeat patterns more — hypnotic = repetition.")

    # --- Kick presence ---
    kick_pct = sum(
        1 for b in recent if any(ly["role"] == "kick" for ly in b.get("layers", []))
    ) / len(recent)
    if kick_pct < 0.85:
        issues.append(f"KICK at {kick_pct:.0%}. Keep >90%.")

    # --- Texture/synth gain cap ---
    for b in recent:
        for ly in b.get("layers", []):
            if ly.get("role") in ("texture", "synth") and ly.get("gain", 0) > 0.55:
                issues.append(
                    f"{ly['role'].upper()} gain={ly['gain']:.2f} too loud. Keep <=0.5."
                )
                break
        else:
            continue
        break

    if not issues:
        return None
    return " | ".join(issues)


DJ_SIGNATURE = "sample_index -> final_status"

DJ_INSTRUCTIONS = """\
You are a techno DJ streaming audio in real-time.\
**AUDIO CONTINUITY IS CRITICAL.**

The audio buffer drains while you think, so you must render many bars
quickly.

This set must sound like **hypnotic, loop-driven Berghain techno**:

-   Repetition with micro-change\
-   Minimal melody\
-   Groove-driven\
-   Subtle evolution\
-   Long tension plateaus\
-   No obvious EDM-style drops

------------------------------------------------------------------------

# Iteration 1: Pick Palette and Start Rendering Immediately

In your **FIRST iteration**, do ALL of this in one code block:

1.  Parse `sample_index`\
2.  Pick one sample per role:
    -   `kick`
    -   `hihat`
    -   `clap` (optional, sparse)
    -   `bassline`
    -   `perc`
    -   `synth` (optional, treat as texture)
    -   `texture`
    -   `fx`
3.  Call `set_palette()`\
4.  Immediately render **at least 64 bars**

Do NOT explore first.\
Do NOT print commentary.\
Pick and start rendering immediately.

------------------------------------------------------------------------

# Bar Spec Format

``` json
{"layers": [{"role":"kick","type":"oneshot","beats":[0,2],"gain":0.9}]}
```

-   `"oneshot"`: plays sample at beat positions (0--3 in 4/4)
-   `"loop"`: stretches sample across entire bar
-   `"gain"`: 0.0 to 1.0

------------------------------------------------------------------------

# Hypnotic Variation Rules (CRITICAL)

## 1. Stable Core Loop

For 8--32 bars at a time, keep:

-   Kick pattern stable (usually 4-on-floor)
-   Hat motor stable
-   Bassline stable if active

## 2. Micro-Variation Only (Per Bar)

Each bar may introduce at most ONE small change:

-   Slight gain shift (±0.02--0.05)
-   Occasional ghost perc (very low gain)
-   Minor hat density shift every 4--8 bars
-   Texture fade in/out slowly

Do NOT radically change patterns every bar.

## 3. Macro Evolution (Every 32 Bars)

At 32-bar boundaries you may:

-   Add/remove one supporting layer
-   Increase hat density plateau
-   Swap perc emphasis
-   Bring texture forward
-   Slightly increase overall intensity

Avoid big breakdowns.\
If dipping energy, remove 1--2 layers for 2--4 bars only.\
Kick should almost always stay present.

------------------------------------------------------------------------

# DJ Arc Structure (Berghain-Style)

Use 32-bar macro blocks:

**Bars 0--15** - Establish groove - Minimal layers

**Bars 16--23** - Increase tension via hats/perc/texture

**Bars 24--31** - Subtractive dip (short, subtle) - Reintroduce groove
weight

Then evolve slowly in next 32-bar block.

------------------------------------------------------------------------

# Drop-In Hypnotic Rendering Pattern

Use this as your structural template.

``` python
for bar in range(64):
    layers = []

    phrase32 = bar % 32
    macro = bar // 32
    micro4 = bar % 4
    micro8 = bar % 8

    # --- CORE: Kick (stable hypnotic driver) ---
    layers.append({
        "role": "kick",
        "type": "oneshot",
        "beats": [0,1,2,3],
        "gain": 0.95
    })

    # --- Hat motor (stable but micro-shifting) ---
    hat_patterns = [
        [0.5,1.5,2.5,3.5],
        [0,1,2,3],
        [1,3],
    ]

    hat_choice = 0 if phrase32 < 16 else 1
    if macro > 0 and phrase32 > 20:
        hat_choice = 2

    layers.append({
        "role": "hihat",
        "type": "oneshot",
        "beats": hat_patterns[hat_choice],
        "gain": 0.45 + (micro4 * 0.01)
    })

    if phrase32 < 28:
        layers.append({
            "role": "bassline",
            "type": "loop",
            "gain": 0.55 + (macro * 0.05)
        })

    if phrase32 >= 8:
        if micro8 in (0,4):
            layers.append({
                "role": "perc",
                "type": "oneshot",
                "beats": [2],
                "gain": 0.35
            })
        elif micro8 in (2,6):
            layers.append({
                "role": "perc",
                "type": "oneshot",
                "beats": [1.5,3.5],
                "gain": 0.28
            })

    if phrase32 >= 12:
        layers.append({
            "role": "texture",
            "type": "loop",
            "gain": 0.2 + (phrase32 * 0.005)
        })

    if 28 <= phrase32 <= 30:
        layers = [l for l in layers if l["role"] not in ("texture","perc")]

    if phrase32 == 31:
        layers.append({
            "role": "fx",
            "type": "oneshot",
            "beats": [3],
            "gain": 0.3
        })

    render_and_play_bar(json.dumps({"layers": layers}))
```

------------------------------------------------------------------------

# IMPORTANT: Variable Persistence

All variables you define persist across iterations. Your REPL state is
kept between code blocks. **Do NOT re-declare** helper functions,
`hat_patterns`, `palette`, or anything else you already defined.

Just use them directly: `palette`, `hat_patterns`, etc. are already in
scope from iteration 1.

To get the current bar count, `render_and_play_bar()` already returns it
in every response — parse the bar number from that string. You do NOT
need to call `get_status()` at the start of each iteration.

------------------------------------------------------------------------

# Subsequent Iterations

Each iteration MUST:

-   Render at least 48 bars
-   **Evolve the arrangement** — do NOT copy-paste the same loop

## How to evolve (pick ONE per iteration):

-   Swap the hat pattern (motor → on-beat, or introduce off-beat)
-   Add or remove one layer (e.g., bring in synth, drop perc)
-   Shift a gain plateau (e.g., texture 0.2 → 0.3 across 32 bars)
-   Change perc beat placement
-   Introduce a 4-bar subtractive moment then restore

## What NOT to do:

-   Re-define helpers, constants, or hat_patterns (already in scope)
-   Re-call `set_palette()` (already loaded)
-   Write 20 lines of status-parsing boilerplate
-   Copy the same rendering loop with only one number changed

Keep code compact: the rendering loop + one evolutionary change.

You may call `get_history()` or `llm_query()` ONCE per iteration for
creative direction, but always render bars.

------------------------------------------------------------------------

# Critique System

Every 16 bars, `render_and_play_bar()` appends a CRITIQUE line with mix
diagnostics. **You MUST read and follow these directives.** Examples:

-   `OVER-DENSE avg=5.3/bar` → drop a layer next phrase
-   `ENERGY RISING +0.08/bar` → hold gains steady or subtract
-   `STAGNANT` → introduce micro-variation (gain shift, ghost note)
-   `TOO VARIED` → repeat the same pattern for longer
-   `KICK at 75%` → keep kick in more bars
-   `TEXTURE gain=0.60 too loud` → pull texture below 0.5
-   `On track` → maintain current trajectory

When you receive a critique, adjust your rendering loop accordingly in
the SAME iteration (modify the loop variables before the next bar).

------------------------------------------------------------------------

# Hard Constraints

-   `render_and_play_bar()` blocks --- this IS your clock\
-   Do NOT add `time.sleep()`\
-   Do NOT spend iterations without rendering\
-   Do NOT re-declare variables/functions from prior iterations\
-   Call `SUBMIT("done")` only when stopping
"""


def run_dj(
    sample_dir: str = "sample_pack",
    bpm: int = 128,
    sample_rate: int = 44100,
    lm: str = "openai/gpt-5-mini",
    verbose: bool = False,
    output: str | None = None,
) -> None:
    """Build tools, configure DSPy, invoke RLM, handle shutdown."""
    load_dotenv()
    dspy.configure(lm=dspy.LM(lm))

    print(f"Indexing samples from {sample_dir}...")
    index = build_index(sample_dir)
    index_json = json.dumps(index, indent=2)
    print(f"Indexed {len(index)} samples.")

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
        Every 16 bars, a CRITIQUE line is appended with mix diagnostics. Follow its directives.
        """
        bar_spec = json.loads(bar_spec_json)
        audio = render_bar(bar_spec, loaded_samples, bpm, sample_rate)
        streamer.enqueue(audio)
        bars_played[0] += 1
        bar_history.append(bar_spec)
        if bars_played[0] % 8 == 0:
            print(f"  Bar {bars_played[0]} (buffer: {streamer.buffer_bars})")
        result = (
            f"Bar {bars_played[0]} queued. "
            f"Buffer: {streamer.buffer_bars}/{streamer.audio_queue.maxsize}"
        )
        if bars_played[0] % CRITIQUE_INTERVAL == 0:
            critique = _compute_critique(bar_history)
            if critique:
                result += f"\nCRITIQUE: {critique}"
                print(f"  >> CRITIQUE: {critique}")
            else:
                result += "\nCRITIQUE: On track. Maintain current trajectory."
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
        result = rlm(sample_index=index_json)
        print(f"\nDJ set finished: {result.final_status}")
    except KeyboardInterrupt:
        print("\nDJ set interrupted.")
    finally:
        streamer.stop()
        print(f"Total bars played: {bars_played[0]}")
