# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bergain is an AI DJ system that controls Ableton Live via OSC. A DSPy Reactive Language Model (RLM) writes Python code that calls tools to compose music — browsing sounds, creating tracks, writing MIDI clips, and managing live performance in real time.

## Commands

```bash
# Run tests (no Ableton needed — uses stub session)
uv run python -m pytest tests/ -v

# Lint/format
uvx ruff check --fix . && uvx ruff format .

# Compose (palette mode — builds session grid, requires Ableton + AbletonOSC)
uv run python -m bergain "Dark Berlin techno in F minor, 130 BPM"

# Compose (live mode — real-time performance)
uv run python -m bergain --live --duration 10 "Evolving ambient in F minor"

# E2E pipeline (compose → export WAV → analyze via Modal)
uv run python scripts/e2e_compose.py "dark minimal techno" --live --duration 5

# Deploy Modal endpoints (aesthetics scoring + audio analysis)
uvx modal deploy aesthetics/app.py

# Browse Ableton sounds (requires Ableton)
uv run python -m bergain.browse search "909"
```

**Always use `uv run python` instead of `python3`** — enforced by a hookify hook.

## Architecture

### Core Pipeline

```
compose.py (CLI + DSPy RLM signatures)
    → tools.py (tool closures over Session, milestone tracking)
        → session.py (high-level: track resolution, clips, mix, recording)
            → osc.py (low-level: UDP OSC client, domain proxies, LiveAPI)
                → spec_data.py (typed AbletonOSC endpoint specs)
    → music.py (pure functions: chord math, drum/bass/pad renderers)
```

### Two Composition Modes

- **Palette mode** (`Compose` signature): RLM builds a session grid of looping clips organized by scene/energy. The human DJ fires scenes to perform. Tools: `browse`, `create_tracks`, `write_clip`, `set_mix`.
- **Live mode** (`LiveCompose` signature): RLM composes in real time while music plays. `compose_next()` is the key tool — takes a creative prompt, calls a sub-LM for musical decisions, writes clips, fires the scene, waits. Tools: `setup_session`, `compose_next`, `get_arc_summary`, `wait`, `elapsed`.

### OSC Layer

`AbletonOSC` is a single-socket UDP client. `LiveAPI` wraps it with domain-scoped proxies (`DomainProxy`). Non-indexed domains are attributes (`api.song`, `api.browser`), indexed domains are callable (`api.track(0)`, `api.clip(0, 2)`). All ~500 AbletonOSC endpoints are typed in `spec_data.py`.

### Modal Endpoints (`aesthetics/app.py`)

Two GPU/CPU classes deployed to Modal:
- **Judge** (T4 GPU): `audiobox_aesthetics` scoring — PQ, CU, CE, PC on 1-10 scale
- **Analyzer** (CPU): librosa-based analysis — key detection, structure segmentation, energy profiles, onset detection, frequency clash analysis

The analyzer URL is at `BERGAIN_ANALYZER_URL` env var (defaults to the deployed Modal endpoint).

### Audio Export

`Session.start_recording()` creates an audio capture track (Resampling input, Sends Only output) and uses Ableton's session recording. Clips from multiple scene fires are concatenated and normalized on `stop_recording()`. The capture track is auto-restored if `setup()` is called mid-recording.

## Key Constraints

- **All DSPy tools must return `str`** — the PythonInterpreter uses a Deno+Pyodide WASM sandbox; non-string returns silently break IPC.
- **`SUBMIT()` discards stdout** — the `FinalOutput` exception skips reading `buf_stdout`. Never SUBMIT in the same step as tool calls.
- **Note positions are relative to clip start** — `clip()` and `arr_clip()` expect notes starting from beat 0.
- **Volume 0.0-1.0, pan -1.0 to 1.0** — Ableton's native scale, not dB.
- **`dspy.LM()` returns dicts** — `completions[0]` is `{'text': '...', 'reasoning_content': '...'}`, extract with `.get("text")`.
- **Sub-LM energy ceiling** — without calibration anchors in prompts (0.2=ambient, 0.6=driving, 0.95=max), sub-LM caps at ~0.4 energy.
- **Live compose needs >=5 min** — RLM burns 2-4 iterations on setup; each `compose_next` takes ~60-90s.
- **DSPy caches individual LM calls** — use `cache=False` during iteration to prevent stale trajectory replay.

## Testing

Tests run against a `StubSession` that records calls without needing Ableton. The `scripts/validate_export.py` script tests export assumptions against a live Ableton instance.
