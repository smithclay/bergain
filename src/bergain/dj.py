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

DJ_SIGNATURE = "sample_index -> final_status"

DJ_INSTRUCTIONS = """\
You are a techno DJ streaming audio in real-time. AUDIO CONTINUITY IS CRITICAL.
The audio buffer drains while you think, so you must render many bars quickly.

## Iteration 1: Pick palette and start rendering immediately

In your FIRST iteration, do ALL of this in one code block:
1. Parse sample_index, pick one sample per role (kick, hihat, clap, bassline, perc, synth, texture, fx)
2. Call set_palette() with your picks
3. Immediately render at least 48 bars using render_and_play_bar()

Do NOT spend iterations just exploring — pick samples and start rendering right away.

## Bar spec format

{"layers": [{"role":"kick","type":"oneshot","beats":[0,2],"gain":0.9}, ...]}

- "oneshot": plays sample at beat positions (0-3 in 4/4 time)
- "loop": stretches sample across entire bar
- "gain": 0.0 to 1.0

## CRITICAL: Vary your patterns bar-to-bar

DO NOT render the same bar spec in every iteration of your loop. Use the bar
index to create VARIATION across bars. Example pattern:

```python
for bar in range(64):
    layers = []
    phrase = bar % 16       # position within 16-bar phrase
    section = bar // 16     # which section we're in
    breakdown = 12 <= phrase <= 15  # last 4 bars = breakdown

    # Kick: drop out during breakdowns
    if not breakdown:
        beats = [0,1,2,3] if phrase >= 8 else [0,2]
        layers.append({"role":"kick","type":"oneshot","beats":beats,"gain":0.95})

    # Hihat: vary pattern by bar position
    hh_patterns = [[1,3], [0,1,2,3], [0,2], [1,2,3]]
    layers.append({"role":"hihat","type":"oneshot","beats":hh_patterns[phrase%4],"gain":0.5})

    # Clap on backbeat
    layers.append({"role":"clap","type":"oneshot","beats":[1,3],"gain":0.7})

    # Bass and synth: loops with varying gain
    if not breakdown:
        layers.append({"role":"bassline","type":"loop","gain":0.6 + phrase*0.02})
    layers.append({"role":"synth","type":"loop","gain":0.3 if not breakdown else 0.6})

    # Perc: add layers over time
    if phrase >= 4:
        layers.append({"role":"perc","type":"oneshot","beats":[0,2] if bar%2==0 else [1,3],"gain":0.4})

    # FX: risers before drops
    if phrase in (10, 11):
        layers.append({"role":"fx","type":"oneshot","beats":[3],"gain":0.7})

    render_and_play_bar(json.dumps({"layers": layers}))
```

## Subsequent iterations: keep rendering

Each iteration MUST render at least 32 bars. You may call get_history() and
llm_query() ONCE per iteration for creative direction, but always render bars.
Keep your code compact — minimal prints, no state management boilerplate.

## DJ principles

- 16-bar phrases: build energy bars 0-7, peak bars 8-11, breakdown bars 12-15
- Breakdowns: remove kick and bass, let synth/texture/fx breathe
- Drops: re-introduce kick+bass after breakdown with full energy
- Vary hihat patterns every 2-4 bars for groove
- Change the arrangement every 32-64 bars (swap which roles are active)

## Rules

- render_and_play_bar() blocks when buffer is full — this IS your timing
- Do NOT add time.sleep() calls
- Do NOT spend iterations without rendering bars
- Call SUBMIT("done") only when you want to stop
"""


def run_dj(
    sample_dir: str = "sample_pack",
    bpm: int = 128,
    sample_rate: int = 44100,
    lm: str = "openai/gpt-5-mini",
    verbose: bool = False,
) -> None:
    """Build tools, configure DSPy, invoke RLM, handle shutdown."""
    load_dotenv()
    dspy.configure(lm=dspy.LM(lm))

    print(f"Indexing samples from {sample_dir}...")
    index = build_index(sample_dir)
    index_json = json.dumps(index, indent=2)
    print(f"Indexed {len(index)} samples.")

    streamer = AudioStreamer(sample_rate=sample_rate)
    streamer.start()
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
        """
        bar_spec = json.loads(bar_spec_json)
        audio = render_bar(bar_spec, loaded_samples, bpm, sample_rate)
        streamer.enqueue(audio)
        bars_played[0] += 1
        bar_history.append(bar_spec)
        if bars_played[0] % 8 == 0:
            print(f"  Bar {bars_played[0]} (buffer: {streamer.buffer_bars})")
        return (
            f"Bar {bars_played[0]} queued. "
            f"Buffer: {streamer.buffer_bars}/{streamer.audio_queue.maxsize}"
        )

    def get_status() -> str:
        """Get current playback status including bars played and buffer level."""
        return json.dumps(
            {
                "bars_played": bars_played[0],
                "buffer_bars": streamer.buffer_bars,
                "buffer_max": streamer.audio_queue.maxsize,
            }
        )

    def get_history(last_n: str = "32") -> str:
        """Get a summary of recent DJ history to detect repetition.
        Input: number of recent bars to summarize (as string).
        Returns: JSON with role frequency, energy trajectory, and recent bar specs.
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
