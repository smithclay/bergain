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
- **Tools as closures** — `set_palette()`, `render_and_play_bar()`, `get_status()`, `get_history()` are closures over shared state (loaded samples, audio streamer, bar history)
- **Backpressure** — `render_and_play_bar()` blocks when the audio queue is full, naturally pacing the RLM at playback speed
- **Self-reflection** — the RLM periodically calls `get_history()` + `llm_query()` within its loop to get creative direction and evolve patterns
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
