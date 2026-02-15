# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # Install dependencies
uv run bergain generate --seed 42    # Generate arrangement artifact (JSON)
uv run bergain play                  # Render latest artifact to WAV and play
uv run bergain play output/song.json # Play a specific artifact
uv run bergain play output/song.wav  # Play a WAV directly
uv run bergain dj --lm "openai/gpt-5-mini"          # Stream a live DJ set
uv run bergain dj --lm "openai/gpt-5-mini" --verbose # With RLM debug output
```

Requires Deno for DSPy's sandboxed code interpreter: `curl -fsSL https://deno.land/install.sh | sh`
Deno must be on `$PATH` (the installer puts it in `~/.deno/bin/`).

## Architecture

Bergain is a recipe-based techno song builder. The key design decision: **the artifact is a declarative arrangement, not a WAV file.** `generate` outputs a JSON arrangement describing palette + sections + bar-level layers. `play` interprets the artifact by rendering to WAV via pydub, then playing with `afplay`.

### Data flow

**Generate (static):**
```
sample_pack/ → indexer.build_index() → SamplePicker
  → recipe.generate_arrangement() → arrangement JSON
    → renderer.render() → WAV → renderer.play()
```

**DJ (streaming):**
```
sample_pack/ → indexer.build_index() → sample index JSON
  → dspy.RLM(tools=[set_palette, render_bar, ...])
    → RLM writes DJ loop in sandboxed REPL
      → render_bar() → AudioStreamer → sounddevice → speakers
```

Indexing happens at runtime inside `generate` and `dj` (fast enough to not need a separate step). The sample index is never persisted — it's built on the fly from the sample pack directory.

### DJ streaming architecture

The `dj` command replaces the static recipe with a DSPy RLM invocation. The RLM's persistent REPL environment IS the DJ's brain — it writes its own loops, selects samples, and drives the audio pipeline through tool functions injected into its sandbox.

Key mechanisms:
- **Mixer-control tools** — `play()`, `add()`, `fader()`, `pattern()`, `mute()`/`unmute()`, `breakdown()`, `swap()` are closures over shared mixer state
- **Backpressure** — `play()` blocks when the audio queue is full, naturally pacing the RLM at playback speed
- **Auto-guardrails** — `_auto_correct_bar()` caps texture/synth gains and enforces position-aware density ceilings via `_ARC_PHASES`
- **Trajectory feedback** — every 8 bars, trajectory observations (energy slope, stagnation, density) are injected into tool responses
- **LLM critic** — every 16 bars, a cheap sub-LM provides phase-aware creative direction
- **Loop auto-detection** — `add()` checks `role_map` to determine if a sample is a loop or oneshot, preventing the RLM from passing incorrect types
- **Deno sandbox** — DSPy's PythonInterpreter runs code in Deno+Pyodide; tool functions execute on the host where pydub/sounddevice live

### Arrangement model (dict-based, JSON-serializable)

The arrangement dict has: `bpm`, `sample_rate`, `palette` (role→sample path), and `sections`. Each section has `name`, `bars`, `layers`, optional `fade_in`/`fade_out` (in bars). Each layer has `role`, `type` ("oneshot" or "loop"), `gain` (linear 0-1), and for oneshots `beats` (0-indexed within a bar). Layers can be scoped to a bar range via optional `start_bar`/`end_bar`.

This model is designed so a future streaming player can read bars one-by-one.

### Module roles

- **recipe.py** — song structure logic; picks palette, builds section/layer dicts
- **renderer.py** — interprets arrangement into audio via pydub; overlay, loop, fade, normalize
- **indexer.py** — scans sample pack, extracts metadata (duration, BPM, category) via `av`
- **picker.py** — `SamplePicker` class; filters/picks samples by category, sub_type, is_loop
- **arrangement.py** — `save_arrangement` / `load_arrangement` (JSON I/O)
- **cli.py** — click CLI with three commands: `generate`, `play`, `dj`
- **dj.py** — DJ orchestration; builds RLM tool closures, configures DSPy, invokes RLM, handles shutdown
- **streamer.py** — `AudioStreamer`: pydub→numpy conversion, sounddevice `OutputStream`, threaded consumer with backpressure

## Conventions

- `av` is used only for metadata extraction (indexer) — never for audio rendering
- `pydub` handles all audio manipulation (load, overlay, fade, export)
- `sounddevice` handles real-time audio output (streamer only)
- Gain values are linear (0.0–1.0) in the arrangement, converted to dB in the renderer
- All samples are resampled to mono at the arrangement's sample_rate during render
- `sample_pack/` folder names map to categories via `CATEGORY_MAP` in indexer.py
- RLM tool functions communicate via JSON strings (serializable across the Deno sandbox boundary)
- See `RLM_LESSONS.md` for hard-won insights about using DSPy RLM for music generation

## Scoring & Evaluation

Audio quality is measured via a Modal-hosted audiobox-aesthetics endpoint:
- **Metrics**: CE (Content Enjoyment), CU (Content Usefulness), PC (Production Complexity), PQ (Production Quality)
- **Objective**: `OBJ = 0.60*CE + 0.05*CU + 0.05*PC + 0.30*PQ`
- **Palette screening**: `scripts/screen_palettes.py` renders reference arrangements across random palettes
- **Gap diagnostics**: `scripts/score_gap_test.py` isolates duration vs arc vs decision effects
- **Parallel DJ runs**: `scripts/parallel_dj.sh` runs DJ across models/prompts with auto-scoring
- Curated palettes live in `palettes/curated/` (top 10 from screening, OBJ 7.0+)
