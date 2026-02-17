"""DSPy RLM Composer — LM-driven composition for Ableton Live.

The RLM writes code that orchestrates sub-LM calls to compose music.
The brief is a symbolic variable the LM explores through code — creative
decisions happen through programmatic llm_query() calls in loops, results
accumulate in Python variables, and tools realize them in the DAW.

Usage:
    uv run python -m bergain "Dark Berlin techno in F minor, 130 BPM, driving kick"
    uv run python -m bergain --brief-file brief.txt
    uv run python -m bergain --model openrouter/openai/gpt-5 "Dark ambient in Eb"
    uv run python -m bergain --live --duration 60 "Evolving ambient in F minor"
    uv run python -m bergain --dry-run "Test brief"
"""

import argparse
import json
import os
import re
import time

import dspy
from dotenv import load_dotenv

from .tools import make_tools

load_dotenv()


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="DSPy RLM Composer for Ableton Live")
    p.add_argument(
        "brief",
        nargs="?",
        default="",
        help="Creative brief (mood, genre, instrumentation)",
    )
    p.add_argument("--brief-file", help="Read brief from a file")
    p.add_argument(
        "--model",
        default="openrouter/openai/gpt-5",
        help="Primary LM (LiteLLM model string)",
    )
    p.add_argument(
        "--sub-model",
        default=None,
        help="Sub-LM for llm_query() (default: same as --model)",
    )
    p.add_argument(
        "--max-iterations", type=int, default=None, help="Max REPL iterations"
    )
    p.add_argument(
        "--max-llm-calls", type=int, default=None, help="Max llm_query() calls"
    )
    p.add_argument(
        "--log-dir", default="./output/compose/", help="Directory for trajectory logs"
    )
    p.add_argument(
        "--min-clips",
        type=int,
        default=None,
        help="Minimum clips before ready_to_submit() allows SUBMIT",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Live composition mode — RLM composes in real time while music plays",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target duration in minutes for live mode (default: 60)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and tools, don't connect to Ableton",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve brief
    if args.brief_file:
        with open(args.brief_file) as f:
            brief = f.read().strip()
    elif args.brief:
        brief = args.brief
    else:
        print("Error: provide a brief as argument or via --brief-file")
        return

    # Live mode needs at least 5 min for the RLM to set up and compose
    if args.live and args.duration < 5:
        print(
            f"  [NOTE] Bumping duration from {args.duration} to 5 min (minimum for live mode)"
        )
        args.duration = 5

    # Apply mode-appropriate defaults for unset arguments
    defaults = (60, 60, 3) if args.live else (30, 40, 6)
    if args.max_iterations is None:
        args.max_iterations = defaults[0]
    if args.max_llm_calls is None:
        args.max_llm_calls = defaults[1]
    if args.min_clips is None:
        args.min_clips = defaults[2]

    # Pick signature
    signature = LiveCompose if args.live else Compose
    mode_label = f"LIVE ({args.duration} min)" if args.live else "PALETTE"

    # Configure LMs (cache=False to prevent stale trajectory replay)
    lm = dspy.LM(args.model, cache=False)
    sub_lm = dspy.LM(args.sub_model, cache=False) if args.sub_model else lm
    dspy.configure(lm=lm)

    print("=== DSPy RLM Composer ===")
    print(f"  Mode:       {mode_label}")
    print(f"  Model:      {args.model}")
    print(f"  Sub-model:  {args.sub_model or args.model}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  LLM calls:  {args.max_llm_calls}")
    print(f"  Log dir:    {args.log_dir}")
    print(f"  Min clips:  {args.min_clips}")
    print(f"  Brief:      {brief[:100]}{'...' if len(brief) > 100 else ''}")
    print()

    if args.dry_run:
        print(f"  [DRY RUN] Signature: {signature.__name__}")
        print(f"  [DRY RUN] Docstring ({len(signature.__doc__)} chars):")
        print(signature.__doc__[:300] + "...")
        print("\n  [DRY RUN] Would connect to Ableton and run RLM. Exiting.")
        return

    # Connect to Ableton
    from .session import Session

    session = Session()
    tools, _, live_history = make_tools(
        session,
        min_clips=args.min_clips,
        live_mode=args.live,
        duration_minutes=args.duration,
        sub_lm=sub_lm if args.live else None,
        brief=brief,
    )

    # Build RLM — signature docstring IS the prompt (highest priority)
    composer = dspy.RLM(
        signature,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_output_chars=15000,
        tools=tools,
        sub_lm=sub_lm,
        verbose=True,
    )

    try:
        print("  Starting composition...\n")
        prediction = composer(brief=brief)
        print("\n  === Report ===")
        print(f"  {prediction.report}")
        save_trajectory(prediction, brief, args.log_dir, live_history=live_history)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping playback...")
    finally:
        session.stop()
        session.close()


if __name__ == "__main__":
    main()
