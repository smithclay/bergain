"""DSPy RLM Composer — LM-driven composition for Ableton Live.

The RLM writes code that orchestrates sub-LM calls to compose music.
The brief is a symbolic variable the LM explores through code — creative
decisions happen through programmatic llm_query() calls in loops, results
accumulate in Python variables, and tools realize them in the DAW.

CLI entry point is bergain.cli — this module exports signatures and helpers.
"""

import json
import os
import re
import time

import dspy


# ---------------------------------------------------------------------------
# DSPy Signature — the docstring IS the prompt (highest-priority position).
# Tool-specific reference goes in tool docstrings, not here.
# ---------------------------------------------------------------------------


class Compose(dspy.Signature):
    """You are building a live performance palette in a DAW by writing Python
    code. Build a grid of looping clips organized by SCENE (energy level).
    The human DJ fires scenes to perform the arrangement live.

    WORKFLOW — one phase per code step, never combine phases:
      1. BROWSE & SETUP: browse() + create_tracks() + set_tempo() — all in ONE step.
      2. PLAN: llm_query() for creative decisions — key, scenes (energy levels),
         chord progressions per scene, style choices. Plan 5-8 scenes spanning
         the full energy arc (ambient → build → peak → breakdown → peak → outro).
         Always request JSON. Parse: re.search(r'{.*}', response, re.S)
      3. BUILD PALETTE: write_clip() — ONE call per scene. Each scene fills one
         row of the session grid across all tracks. Clips loop automatically.
         Use 4-bar loops (short, tight) or 8-bar loops (more variation).
      4. MIX: set_mix() + get_status() + ready_to_submit()
      5. SUBMIT: SUBMIT(report) alone.

    RULES:
      - Use llm_query() to decide key, chords, energy levels, and style names.
      - write_clip() handles all MIDI rendering — you never build note arrays.
      - write_clip() only writes to Drums, Bass, and Pad tracks. Extra tracks
        (Lead, FX, etc.) are not filled by write_clip — use them manually.
      - Each scene needs DIFFERENT energy/style/chords — monotony is failure.
      - Scenes are indexed 0, 1, 2... — lower slots = lower energy is natural
        but not required. The DJ chooses fire order.
      - ready_to_submit() checks milestones. Only SUBMIT after it says READY.
      - Most tools return JSON (parse with json.loads). Exceptions: browse(),
        get_status(), and ready_to_submit() return plain text.
      - NEVER use f-strings with JSON content — the braces conflict with Python
        format syntax. Use plain strings or triple-quoted strings for JSON.

    EXAMPLE (each step = one code block, separate execution):
      Step 1: browse("909", "Drums"); create_tracks(json.dumps([
                {"name":"Drums","drum_kit":"909 Core Kit.adg"},
                {"name":"Bass","instrument":"Operator"},
                {"name":"Pad","sound":"Warm Pad"}])); set_tempo(130)
      Step 2: plan = llm_query('Dark techno in F minor. Build 6 scenes for '
                'live performance. Return JSON with key and scenes array, each '
                'scene having: name, slot, bars, energy, chords, drums, bass, pad.')
      Step 3: write_clip(json.dumps({"name":"Ambient","slot":0,"bars":8,
                "energy":0.2,"key":"F","chords":["Fm"],
                "drums":"minimal","bass":"none","pad":"atmospheric"}))
      Step 4: write_clip(json.dumps({"name":"Groove","slot":1,"bars":4,
                "energy":0.5,"key":"F","chords":["Fm","Cm"],
                "drums":"four_on_floor","bass":"pulsing_8th","pad":"sustained"}))
      ...
      Step N-1: set_mix(json.dumps({"Drums":0.9,"Bass":0.85,"Pad":0.7}));
                print(get_status()); print(ready_to_submit())
      Step N: SUBMIT(report)
    """

    brief: str = dspy.InputField(
        description="Creative brief describing mood, genre, tempo, and instrumentation"
    )
    report: str = dspy.OutputField(
        description="Summary of the session palette: tempo, tracks, scenes with slot numbers "
        "and energy levels and clip names, mix levels, key and chords per scene"
    )


class LiveCompose(dspy.Signature):
    """You are a live composer performing in real time. Music plays while you
    think. compose_next() is your main creative tool — give it FEELINGS and
    DIRECTION, and a sub-LM handles all musical decisions (chords, styles,
    energy values). You are the conductor, not the arranger.

    WORKFLOW:
      SETUP (one step): setup_session(config_json) — creates tracks with
        role-appropriate sounds, sets tempo, starts playback. ONE call does
        everything. Just specify name + role for each track; the search is
        optional (sensible defaults per role). Include 5 tracks: drums, bass,
        pad, stab, texture.
      EVOLVE (repeat): Call compose_next() with creative direction. It handles
        everything: sub-LM query, clip writing, scene firing, waiting.
        Check get_arc_summary() every ~5 sections to course-correct.
      FINISH: When get_arc_summary() shows final phase, set_mix() +
        ready_to_submit(), then SUBMIT(report) alone in the final step.

    ENERGY ARC — think in thirds:
      - Opening third: build gradually, stay low-to-moderate. Establish mood.
      - Middle third: create WAVES — alternate lifts and dips. Most variety.
      - Final third: wind down. Longer bars, sparser textures, gentle fade.
      Never let energy plateau for more than 3 sections — if arc summary
      shows a flat trend, force a contrast.

    RULES:
      - setup_session() handles ALL sound loading. Do NOT try to browse or
        load instruments manually — those tools are not available.
      - compose_next() takes a creative prompt, NOT parameter values.
        Good: "Bring in a deep walking bass, let the drums rest"
        Bad:  "energy 0.55, drums half_time, bass sustained"
      - compose_next() tracks history and enforces variety automatically.
      - Call get_arc_summary() to see energy curve, style stats, timing phase.
      - ready_to_submit() enforces minimum elapsed time. Don't rush.
      - Most tools return JSON (parse with json.loads). Exceptions:
        get_status() and ready_to_submit() return plain text.
      - NEVER use f-strings with JSON content — the braces conflict with Python
        format syntax. Use plain strings or triple-quoted strings for JSON.

    EXAMPLE (each step = one code block, separate execution):
      Step 1: print(setup_session(json.dumps({
                "tempo": 128,
                "tracks": [
                  {"name":"Drums","role":"drums"},
                  {"name":"Bass","role":"bass"},
                  {"name":"Pad","role":"pad"},
                  {"name":"Stab","role":"stab"},
                  {"name":"Texture","role":"texture"}
                ]})))
      Step 2: print(compose_next("Open with atmosphere — just pads and texture, "
                "no rhythm, let the space breathe. Dreamy and mysterious."))
      Step 3: print(compose_next("Introduce a gentle pulse — minimal drums, "
                "maybe a soft bass note. Keep the energy low but add momentum."))
      Step 4: print(compose_next("Build — more rhythmic drive, bring the bass "
                "forward, add stab hits and some chord movement."))
      Step 5: arc = json.loads(get_arc_summary())
              print("Phase: " + arc['phase'] + ", trend: " + arc['energy_trend'])
              # Adjust direction based on arc...
              print(compose_next("We need contrast — drop to a quiet interlude, "
                "strip away the drums, just pads and atmosphere."))
      ...
      Step N: set_mix(...); print(ready_to_submit())
      Step N+1: SUBMIT(report)
    """

    brief: str = dspy.InputField(
        description="Creative brief describing mood, genre, tempo, and target duration"
    )
    report: str = dspy.OutputField(
        description="Summary of the live performance: tempo, tracks, sections composed "
        "with timestamps and energy arc, total duration, key creative decisions"
    )


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


def _slugify(text, max_len=40):
    """Turn text into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def save_trajectory(prediction, brief, log_dir, live_history=None):
    """Save the RLM trajectory to a JSON file."""
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(brief)
    path = os.path.join(log_dir, f"{ts}_{slug}.json")

    trajectory = []
    try:
        raw = prediction.trajectory
        if isinstance(raw, str):
            trajectory = json.loads(raw)
        elif isinstance(raw, list):
            trajectory = raw
    except (AttributeError, json.JSONDecodeError, TypeError) as e:
        print(f"  [WARNING] Could not extract trajectory: {type(e).__name__}: {e}")

    data = {
        "brief": brief,
        "report": getattr(prediction, "report", ""),
        "timestamp": ts,
        "trajectory": trajectory,
    }

    if live_history:
        data["live_history"] = live_history

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  Trajectory saved to {path}")
    return path
