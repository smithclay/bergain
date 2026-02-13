# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # Install dependencies
uv run bergain generate --seed 42    # Generate arrangement artifact (JSON)
uv run bergain play                  # Render latest artifact to WAV and play
uv run bergain play output/song.json # Play a specific artifact
uv run bergain play output/song.wav  # Play a WAV directly
```

## Architecture

Bergain is a recipe-based techno song builder. The key design decision: **the artifact is a declarative arrangement, not a WAV file.** `generate` outputs a JSON arrangement describing palette + sections + bar-level layers. `play` interprets the artifact by rendering to WAV via pydub, then playing with `afplay`.

### Data flow

```
sample_pack/ → indexer.build_index() → SamplePicker
  → recipe.generate_arrangement() → arrangement JSON
    → renderer.render() → WAV → renderer.play()
```

Indexing happens at runtime inside `generate` (fast enough to not need a separate step). The sample index is never persisted — it's built on the fly from the sample pack directory.

### Arrangement model (dict-based, JSON-serializable)

The arrangement dict has: `bpm`, `sample_rate`, `palette` (role→sample path), and `sections`. Each section has `name`, `bars`, `layers`, optional `fade_in`/`fade_out` (in bars). Each layer has `role`, `type` ("oneshot" or "loop"), `gain` (linear 0-1), and for oneshots `beats` (0-indexed within a bar). Layers can be scoped to a bar range via optional `start_bar`/`end_bar`.

This model is designed so a future streaming player can read bars one-by-one.

### Module roles

- **recipe.py** — song structure logic; picks palette, builds section/layer dicts
- **renderer.py** — interprets arrangement into audio via pydub; overlay, loop, fade, normalize
- **indexer.py** — scans sample pack, extracts metadata (duration, BPM, category) via `av`
- **picker.py** — `SamplePicker` class; filters/picks samples by category, sub_type, is_loop
- **arrangement.py** — `save_arrangement` / `load_arrangement` (JSON I/O)
- **cli.py** — click CLI with two commands: `generate` and `play`

## Conventions

- `av` is used only for metadata extraction (indexer) — never for audio rendering
- `pydub` handles all audio manipulation (load, overlay, fade, export)
- Gain values are linear (0.0–1.0) in the arrangement, converted to dB in the renderer
- All samples are resampled to mono at the arrangement's sample_rate during render
- `sample_pack/` folder names map to categories via `CATEGORY_MAP` in indexer.py
