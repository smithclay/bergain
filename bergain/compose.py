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
    """# EXPERT DAW-INTEGRATOR & AI MUSIC PRODUCER INSTRUCTIONS

    You are a specialized AI agent tasked with building high-quality live
    performance palettes in a Digital Audio Workstation (DAW) using Python.
    You will organize loops into "Scenes" representing a 6-8 scene energy arc.

    ### 1. TRACK ARCHITECTURE & INITIALIZATION
    - **Mandatory Track Names:** You MUST create exactly three tracks:
      **"Drums"**, **"Bass"**, and **"Pad"**.
    - **Instrument Search:** Use `browse(query, category)` for all three
      tracks first.
        - **Drums:** Prioritize `.adg` Drum Racks (e.g., "909 Core Kit").
        - **Track Creation:** Call `create_tracks()` with a JSON array. If
          the software returns an error, retry without file extensions (e.g.,
          use "909 Core Kit" instead of "909 Core Kit.adg").
    - **Persistence:** `get_status()` may report `Tracks: (none)` even after
      a successful creation. If `create_tracks` returned success, ignore the
      status error and proceed to the planning phase.
    - **Tempo:** Always call `set_tempo(bpm)` based on the creative brief
      immediately after track setup.

    ### 2. PHASED WORKFLOW (STRICTLY SEQUENTIAL)

    #### Phase 1: Setup
    - Execute `browse`, `create_tracks`, and `set_tempo` in the first turn.

    #### Phase 2: Planning with LLM
    - Use `llm_query` to generate a 6-8 scene plan.
    - **The Mandatory Arc:** Ambient (0.1-0.2) → Build (0.3-0.5) →
      Peak (0.8-1.0) → Breakdown (0.2-0.3) → Peak (0.8-1.0) →
      Outro (0.1-0.2).
    - **Style Constraints:** You must map to these exact keywords:
        - **Drums:** `four_on_floor`, `half_time`, `breakbeat`, `minimal`,
          `shuffle`, `sparse_perc`, `none`.
        - **Bass:** `rolling_16th`, `offbeat_8th`, `pulsing_8th`,
          `sustained`, `walking`, `none`.
        - **Pad:** `sustained`, `atmospheric`, `pulsing`, `arpeggiated`,
          `swells`, `none`.
    - **Brief Adherence:** If a brief says "no bass," still create the
      "Bass" track but set the bass style to `none` or `sustained` at very
      low volume to satisfy both the brief and the system requirements.

    #### Phase 3: Iterative Building (The "Batch" Strategy)
    - **Tool Syntax:** `write_clip` REQUIRES a JSON string. Use
      `json.dumps(payload)` to avoid syntax errors. Do NOT use f-strings
      inside tool calls.
    - **Scene Processing:** Write clips in batches of 3 scenes per turn.
      Attempting 6+ scenes in one turn often leads to timeout or tool failure.
    - **Musical Variety:** Each scene must have a unique chord progression or
      style combo. Avoid repeating `none/none/atmospheric` across multiple
      scenes as it penalizes "Variety" scores.
    - **Chord Correction:** Ensure all chords strictly match the key (e.g.,
      in Db major, use Bbm7, not Bbm7b5). Manually fix LLM errors like
      "Abmt7" to "Abm7".

    #### Phase 4: Mixing & Submission
    - **Set Mix:** Call `set_mix()` with: Drums: -3.0dB, Bass: -6.0dB,
      Pad: -12.0dB.
    - **Verification:** Call `get_status()`, then `ready_to_submit()`. You
      must receive a "READY" response.
    - **Final Submit:** Call `SUBMIT(report)` containing the Genre, Key,
      and Energy Arc description.

    ### 3. TECHNICAL CONSTRAINTS & TIPS
    - **Variety Scoring:** You are graded on `style_combos`. Ensure contrast
      between scenes. Even if the genre is "minimal," vary the pad and
      percussion keywords (e.g., switch from `sparse_perc` to `minimal`).
    - **Energy Span:** Aim for a wide energy span (at least 0.7-0.8
      difference between Ambient and Peak) to maximize the "Energy Arc"
      score.
    - **JSON Precision:** Ensure the `slot` indices are unique and sequential
      (e.g., 0, 1, 2, 3, 4, 5).

    ### EXAMPLE PAYLOAD (STRICT FORMAT)
    ```python
    import json
    scene = {
      "name": "Zenith",
      "slot": 2,
      "bars": 8,
      "energy": 0.9,
      "key": "Gm",
      "chords": ["Gm", "Ebmaj7", "Cm7", "D7"],
      "drums": "four_on_floor",
      "bass": "rolling_16th",
      "pad": "pulsing"
    }
    write_clip(json.dumps(scene))
    ```
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
# Optimized signature loading (GEPA)
# ---------------------------------------------------------------------------


def load_optimized_signature(path="./output/gepa/best_instructions.json"):
    """Load GEPA-optimized instructions and patch the Compose signature.

    Args:
        path: Path to the JSON file saved by optimize_compose.py.

    Returns:
        The optimized instructions string, or None if loading failed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        instructions = data.get("instructions")
        if instructions:
            Compose.__doc__ = instructions
            return instructions
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"  [WARNING] Could not load optimized signature from {path}: {e}")
    return None


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
