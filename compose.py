"""DSPy RLM Composer — LM-driven composition for Ableton Live.

The RLM writes code that orchestrates sub-LM calls to compose music.
The brief is a symbolic variable the LM explores through code — creative
decisions happen through programmatic llm_query() calls in loops, results
accumulate in Python variables, and tools realize them in the DAW.

Usage:
    uv run python compose.py "A melancholy waltz in G minor, harpsichord and strings"
    uv run python compose.py --brief-file brief.txt
    uv run python compose.py --model openrouter/openai/gpt-5 "Dark ambient in Eb"
    uv run python compose.py --dry-run "Test brief"
"""

import argparse
import json
import os
import re
import time

import dspy
from dotenv import load_dotenv
from session import Session, Track

load_dotenv()

# ---------------------------------------------------------------------------
# Composition guide — injected as the `guide` input to the RLM.
# Keep under ~1500 tokens. Reference data only, no code templates.
# ---------------------------------------------------------------------------

GUIDE = """\
GOAL
  You are composing music in a live DAW. Your job is to CREATE CLIPS with real MIDI notes
  using the tools provided. The report you SUBMIT should describe what you ACTUALLY BUILT —
  tracks created, clips with note counts, tempo, sections. Do NOT just write a plan.
  Every run must call: browse(), create_tracks(), create_clip() (or create_arrangement_clip()).

CREATIVE DECISIONS
  Use llm_query() for musical choices — don't hardcode them in Python.
  Example:
    chords = llm_query("Suggest an 8-bar chord progression in G minor for a melancholy waltz. Return as comma-separated chord symbols like Gm, Dm, Eb, ...")
    Parse the response and use chord() tool to get MIDI pitches.
  Good uses: chord progressions, melody contours, rhythmic patterns, dynamics per phrase, section plan.
  Bad: hardcoding chord tones, melodies, or using flat velocity for every note.

NOTE FORMAT: [pitch, start_beat, duration, velocity]
  start_beat is RELATIVE TO CLIP START (always starts from 0 for each clip)
  Velocity dynamics: pp=30, p=50, mp=65, mf=80, f=100, ff=120
  Use note("G3"), chord("G","min",4), scale("G","minor",4) for pitch lookup.

TOOL RETURNS
  Most tools return JSON strings. Parse with json.loads() before indexing.
  Exceptions — plain text (do NOT json.loads):
    browse() returns names one per line — use .split('\\n').
    get_status() returns human-readable text — just print or read directly.
  Complex inputs (create_tracks, create_clip, set_mix) accept JSON strings via json.dumps().
  Check for "error" key in parsed JSON: if "error" in json.loads(result): handle it.

VALUE RANGES
  volume: 0.0 to 1.0 (0.85 = default, 0.0 = silent, 1.0 = max)
  pan: -1.0 (left) to 1.0 (right), 0.0 = center

SUBMIT RULES
  SUBMIT() ends execution IMMEDIATELY — all print output in the same step is lost.
  NEVER call SUBMIT() in the same code block as tool calls.
  Before SUBMIT, call ready_to_submit(). If it says NOT READY, complete the missing steps first.
  Only SUBMIT after ready_to_submit() returns READY.
  The report must summarize what you CREATED: tracks, clips, note counts, tempo, sections.

CLIPS
  create_clip(track, slot, length, notes) — session view. slot = section index (0, 1, 2, ...).
    Best for: loopable sections, jamming, reusable parts. Each clip is independent.
  create_arrangement_clip(track, start_beat, length, notes) — arrangement view.
    Best for: linear pieces with a fixed timeline (intro→verse→chorus→outro).
  Default to session clips. If you need a fixed timeline (e.g. A-B-A-B-Coda), use arrangement clips.

EXAMPLE — each "Step N" is a SEPARATE code execution (do NOT combine into one block):
  Step 1 — Browse & pick instruments:
    results = browse("strings", "Sounds")
    print(results)                          # print ALL results, pick best in next step
  Step 2 — Choose instruments with llm_query:
    choice = llm_query("Pick the best instruments from these browser results for a melancholy
      waltz. Available: " + results + ". I need: harpsichord, solo violin, viola, cello.
      Return one name per line, exactly matching a browser result.")
    # parse choice, then create_tracks with those sound names
  Step 3 — Setup:
    create_tracks(json.dumps([{"name":"Harpsichord","sound":"Harpsichord"}, ...]))
    set_tempo(72)
    print(get_status())                     # verify devices loaded
  Step 4 — Plan sections & harmony:
    plan = llm_query("Plan 4 sections (intro 4bars, A 8bars, B 8bars, coda 4bars) for a
      melancholy waltz in G minor. For each section give: chord symbols per bar, melody
      contour description, dynamics (pp/p/mp/mf/f). Return as structured text.")
  Step 5 — Section A clips (one section per step):
    pitches = [json.loads(chord(root, qual, 4)) for root, qual in parsed_chords]
    # build note arrays with VARIED velocity and proper durations (~30 lines), then:
    create_clip("Harpsichord", 0, 24.0, json.dumps(harpsi_notes), "A")
    create_clip("Violin", 0, 24.0, json.dumps(violin_notes), "A")
    # ... one create_clip per track
  Step 6 — Section B clips (same pattern)
  Step 7 — Mix & verify:
    set_mix(json.dumps({"Violin": {"volume": 0.85, "pan": -0.3}, ...}))
    print(get_status())
    print(ready_to_submit())                # must say READY before you SUBMIT
  Step 8 — SUBMIT (no tool calls here, only after ready_to_submit said READY):
    SUBMIT("Created 4 tracks, 12 clips across 3 sections...")

  Why separate steps? Each step can fail. Small blocks = only redo ONE step, not everything.
  Duration target: bars * beats_per_bar * 60 / tempo. Aim for 3-5 distinct sections.

REALIZATION — how to turn chords into good note arrays
  Durations (in beats) — match the instrument role:
    Bass/cello: sustain the FULL BAR (e.g. 3.0 in 3/4, 4.0 in 4/4). Short bass = silence gaps.
    Pads/strings: sustain full bar or tie across bars for legato.
    Chords (harpsichord, piano): shorter hits (0.5–1.0) for rhythm, longer (2.0+) for ballads.
    Melody: varied durations from llm_query. Mix quarters, eighths, dotted notes. Never all equal.
  Velocity — MUST vary across phrases. Flat velocity = robotic.
    Use llm_query: "Give velocity (40-110) for each of these 8 bars: gentle swell then fade"
    Or compute: crescendo = [50 + i*8 for i in range(8)], decrescendo = reverse.
    Emphasize beat 1 slightly (+10 vel) in waltz/dance styles.
    Inner voices (viola, pad) quieter than melody (-15 to -20 vel).
  Waltz accompaniment (harpsichord/piano): BOTH bass AND chord in the same track.
    Beat 1: bass root (octave 2-3), duration 1.0, strong velocity
    Beats 2-3: chord dyad/triad (octave 4-5), duration 0.9, softer velocity
    This is ONE track with 7 notes/bar, not split across tracks.

PATTERNS
  Waltz: bass on 1 + chord on 2,3 (see REALIZATION above)
  Arpeggio: cycle chord tones on subdivisions (8ths or 16ths)
  Four-on-floor: [(kick, beat*4, 0.5, 100) for beat in range(bars*4)]
  Drone/pedal: one note sustained for the entire section length
  Ghost notes: same rhythm, velocity 25-40

TIPS
  - Aim for 3-5 DISTINCT sections (intro/A/B/A'/coda). Two sections on repeat is monotonous.
  - set_mix() after clips to pan and balance: strings spread L/R, bass center.
  - Instrument selection: browse() returns many results. Print them ALL, then use llm_query()
    to pick the best match for your piece. Don't blindly take the first result.
    If browse("viola") returns NO_RESULTS, try: "viola section", "string ensemble", "strings".
  - get_status() returns plain text — just print it, don't json.loads().
  - get_params() lists adjustable device parameters (device 0 = instrument, 1+ = effects).
"""

# ---------------------------------------------------------------------------
# Tools — 22 closures over a Session instance.
# Every tool: accepts simple types, returns str, wraps errors.
# MUST return str — DSPy PythonInterpreter uses Deno+Pyodide sandbox.
# ---------------------------------------------------------------------------


def _clamp_note(note):
    """Clamp a [pitch, start, duration, velocity] to valid MIDI ranges."""
    pitch = max(0, min(127, int(note[0])))
    start = max(0.0, float(note[1]))
    duration = max(0.01, float(note[2]))
    velocity = max(1, min(127, int(note[3])))
    return (pitch, start, duration, velocity)


def _parse_notes(notes_json):
    """Parse and validate a JSON notes array."""
    notes = json.loads(notes_json)
    return [_clamp_note(n) for n in notes]


def make_tools(session, min_clips=6):
    """Create 22 tool closures over a live Session, with milestone tracking."""

    # Milestone tracker — shared mutable state across all tool closures.
    # ready_to_submit() checks these before allowing SUBMIT.
    _done = {
        "browse": False,
        "tracks": False,
        "clips": 0,
        "mix": False,
        "status_checks": 0,
    }

    def set_tempo(bpm: int) -> str:
        """Set the project tempo in BPM."""
        try:
            session.tempo(int(bpm))
            return json.dumps({"tempo": int(bpm)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_status() -> str:
        """Get current DAW state as plain text. Returns tempo, playing status, and track list."""
        try:
            _done["status_checks"] += 1
            s = session.status()
            lines = [
                f"Tempo: {s['tempo']} BPM",
                f"Playing: {'yes' if s.get('playing') else 'no'}",
            ]
            tracks = s.get("tracks", [])
            if tracks:
                lines.append("Tracks:")
                for t in tracks:
                    parts = [f"  {t['index']}: {t['name']}"]
                    if t.get("devices"):
                        parts.append(f"[{', '.join(t['devices'])}]")
                    parts.append(f"vol={t.get('volume', 0.85):.2f}")
                    parts.append(f"pan={t.get('pan', 0.0):.2f}")
                    lines.append(" ".join(parts))
            else:
                lines.append("Tracks: (none)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    def create_tracks(tracks_json: str) -> str:
        """Create MIDI tracks from a JSON array of {name, sound?, instrument?, drum_kit?, effects?, volume?, pan?}.

        Replaces ALL existing tracks. Call once with your full track layout.
        Specify sound/instrument/drum_kit to load them automatically during setup.
        volume: 0.0-1.0 (default 0.85). pan: -1.0 to 1.0 (default 0.0).
        """
        try:
            _done["tracks"] = True
            specs = json.loads(tracks_json)
            tracks = []
            for s in specs:
                tracks.append(
                    Track(
                        name=s["name"],
                        sound=s.get("sound"),
                        instrument=s.get("instrument"),
                        drum_kit=s.get("drum_kit"),
                        effects=s.get("effects", []),
                        volume=s.get("volume", 0.85),
                        pan=s.get("pan", 0.0),
                    )
                )
            count = session.setup(tracks)
            return json.dumps(
                {"tracks_created": count, "names": [t.name for t in tracks]}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def browse(query: str, category: str = "") -> str:
        """Search Ableton browser. Returns matching names, one per line.

        Optional category filter: 'Instruments', 'Sounds', 'Drums', 'Audio Effects'.
        Use result.split('\\n') to get a list of names.
        """
        try:
            _done["browse"] = True
            results = session.browse(query, category=category or None)
            names = [r["name"] for r in results[:20]]
            return "\n".join(names) if names else "NO_RESULTS"
        except Exception as e:
            return json.dumps({"error": str(e)})

    def load_instrument(track: str, name: str) -> str:
        """Load an instrument by name onto a track. Returns {"error": ...} if not found."""
        try:
            result = session.load_instrument(track, name)
            if result is None:
                return json.dumps(
                    {
                        "error": f"Instrument '{name}' not found or timed out",
                        "track": track,
                    }
                )
            return json.dumps({"loaded": "instrument", "name": result, "track": track})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def load_sound(track: str, name: str) -> str:
        """Load a sound preset by name onto a track. Returns {"error": ...} if not found."""
        try:
            result = session.load_sound(track, name)
            if result is None:
                return json.dumps(
                    {"error": f"Sound '{name}' not found or timed out", "track": track}
                )
            return json.dumps({"loaded": "sound", "name": result, "track": track})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def load_effect(track: str, name: str) -> str:
        """Load an audio effect by name onto a track. Returns {"error": ...} if not found."""
        try:
            result = session.load_effect(track, name)
            if result is None:
                return json.dumps(
                    {"error": f"Effect '{name}' not found or timed out", "track": track}
                )
            return json.dumps({"loaded": "effect", "name": result, "track": track})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def load_drum_kit(track: str, name: str) -> str:
        """Load a drum kit by name onto a track. Returns {"error": ...} if not found."""
        try:
            result = session.load_drum_kit(track, name)
            if result is None:
                return json.dumps(
                    {
                        "error": f"Drum kit '{name}' not found or timed out",
                        "track": track,
                    }
                )
            return json.dumps({"loaded": "drum_kit", "name": result, "track": track})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def create_clip(
        track: str, slot: int, length_beats: float, notes_json: str, name: str = ""
    ) -> str:
        """Create a session MIDI clip. notes_json: [[pitch, start_beat, duration, velocity], ...].

        start_beat is relative to clip start (0 = first beat of clip).
        Pitch 0-127, start >= 0, duration >= 0.01, velocity 1-127.
        """
        try:
            _done["clips"] += 1
            notes = _parse_notes(notes_json)
            result = session.clip(
                track, int(slot), float(length_beats), notes, name=name
            )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def create_arrangement_clip(
        track: str,
        start_beat: float,
        length_beats: float,
        notes_json: str,
        name: str = "",
    ) -> str:
        """Create an arrangement clip at a beat position. notes_json: [[pitch, start, dur, vel], ...].

        IMPORTANT: Note start times are RELATIVE to clip start (0 = first beat of clip).
        Do NOT use absolute beat positions. Pitch 0-127, velocity 1-127.
        """
        try:
            _done["clips"] += 1
            notes = _parse_notes(notes_json)
            result = session.arr_clip(
                track, float(start_beat), float(length_beats), notes, name=name
            )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_params(track: str, device_index: int) -> str:
        """List parameter names for a device. Device 0 = instrument, 1+ = effects."""
        try:
            params = session.params(track, int(device_index))
            return json.dumps(
                {"track": track, "device": int(device_index), "params": params}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def set_param(track: str, device_index: int, param: str, value: float) -> str:
        """Set a device parameter by name fragment (case-insensitive match)."""
        try:
            session.param(track, int(device_index), param, float(value))
            return json.dumps(
                {
                    "track": track,
                    "device": int(device_index),
                    "param": param,
                    "value": float(value),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def set_mix(levels_json: str) -> str:
        """Set volume/pan for tracks. levels_json: {"TrackName": volume} or {"TrackName": {"volume": v, "pan": p}}.

        volume: 0.0-1.0 (0.85 default). pan: -1.0 (left) to 1.0 (right).
        """
        try:
            _done["mix"] = True
            levels = json.loads(levels_json)
            session.mix(**levels)
            return json.dumps({"mixed": list(levels.keys())})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def play() -> str:
        """Start playback."""
        try:
            session.play()
            return json.dumps({"playing": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def stop() -> str:
        """Stop playback."""
        try:
            session.stop()
            return json.dumps({"playing": False})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def fire_clip(track: str, slot: int) -> str:
        """Fire a specific clip by track name and slot index."""
        try:
            session.fire_clip(track, int(slot))
            return json.dumps({"fired": track, "slot": int(slot)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_arrangement() -> str:
        """Get the current arrangement clip layout."""
        try:
            clips = session.arrangement()
            return json.dumps(clips)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def clear_arrangement() -> str:
        """Delete all arrangement clips from all tracks."""
        try:
            session.clear_arrangement()
            return json.dumps({"cleared": True})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -- Music theory helpers (eliminate RLM boilerplate) --

    _NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    def note(name: str) -> str:
        """Convert note name to MIDI number. e.g. note("G3") -> "55", note("Eb4") -> "63"."""
        try:
            s = name.strip()
            letter = s[0].upper()
            rest = s[1:]
            accidental = 0
            while rest and rest[0] in "#b":
                accidental += 1 if rest[0] == "#" else -1
                rest = rest[1:]
            octave = int(rest)
            midi = (octave + 1) * 12 + _NOTE_NAMES[letter] + accidental
            return str(max(0, min(127, midi)))
        except Exception as e:
            return json.dumps({"error": str(e)})

    _CHORD_INTERVALS = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "dom7": [0, 4, 7, 10],
        "min7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11],
        "dim7": [0, 3, 6, 9],
    }

    def chord(root: str, quality: str, octave: int) -> str:
        """Return MIDI pitches for a chord. e.g. chord("G", "min", 4) -> "[55, 58, 62]".

        Qualities: maj, min, dim, aug, dom7, min7, maj7, dim7.
        """
        try:
            base = (int(octave) + 1) * 12 + _NOTE_NAMES[root[0].upper()]
            acc = root[1:]
            for c in acc:
                base += 1 if c == "#" else -1
            intervals = _CHORD_INTERVALS.get(quality)
            if intervals is None:
                return json.dumps(
                    {
                        "error": f"Unknown quality '{quality}'. Use: {list(_CHORD_INTERVALS.keys())}"
                    }
                )
            pitches = [max(0, min(127, base + i)) for i in intervals]
            return json.dumps(pitches)
        except Exception as e:
            return json.dumps({"error": str(e)})

    _SCALE_INTERVALS = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "pentatonic": [0, 2, 4, 7, 9],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "blues": [0, 3, 5, 6, 7, 10],
    }

    def scale(root: str, mode: str, octave: int) -> str:
        """Return one octave of scale pitches. e.g. scale("G", "minor", 4) -> "[55, 57, 58, 60, 62, 63, 65]".

        Modes: major, minor, dorian, pentatonic, harmonic_minor, blues.
        """
        try:
            base = (int(octave) + 1) * 12 + _NOTE_NAMES[root[0].upper()]
            acc = root[1:]
            for c in acc:
                base += 1 if c == "#" else -1
            intervals = _SCALE_INTERVALS.get(mode)
            if intervals is None:
                return json.dumps(
                    {
                        "error": f"Unknown mode '{mode}'. Use: {list(_SCALE_INTERVALS.keys())}"
                    }
                )
            pitches = [max(0, min(127, base + i)) for i in intervals]
            return json.dumps(pitches)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def ready_to_submit() -> str:
        """Check if composition is complete enough to SUBMIT.

        Call in a SEPARATE step BEFORE the step where you SUBMIT.
        Do NOT call ready_to_submit() and SUBMIT() in the same code block.
        Returns 'READY' if all milestones are met, or lists what's still missing.
        """
        issues = []
        if not _done["browse"]:
            issues.append("browse() not called — search for instruments first")
        if not _done["tracks"]:
            issues.append("create_tracks() not called — set up your tracks")
        if _done["clips"] < min_clips:
            issues.append(
                f"only {_done['clips']} clips created (need at least {min_clips}) — build more sections"
            )
        if not _done["mix"]:
            issues.append("set_mix() not called — balance volumes and panning")
        if _done["status_checks"] < 1:
            issues.append(
                "get_status() never called — verify your work before submitting"
            )
        if issues:
            return "NOT READY to submit:\n" + "\n".join(f"  - {i}" for i in issues)
        return "READY — call SUBMIT(report) in the NEXT step (no other tool calls)."

    return [
        set_tempo,
        get_status,
        create_tracks,
        browse,
        load_instrument,
        load_sound,
        load_effect,
        load_drum_kit,
        create_clip,
        create_arrangement_clip,
        get_params,
        set_param,
        set_mix,
        play,
        stop,
        fire_clip,
        get_arrangement,
        clear_arrangement,
        note,
        chord,
        scale,
        ready_to_submit,
    ]


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------


def _slugify(text, max_len=40):
    """Turn text into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def save_trajectory(prediction, brief, log_dir):
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
    except (AttributeError, json.JSONDecodeError, TypeError):
        pass

    data = {
        "brief": brief,
        "report": getattr(prediction, "report", ""),
        "timestamp": ts,
        "trajectory": trajectory,
    }

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
        help="Creative brief (mood, genre, duration, instrumentation)",
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
    p.add_argument("--max-iterations", type=int, default=30, help="Max REPL iterations")
    p.add_argument(
        "--max-llm-calls", type=int, default=40, help="Max llm_query() calls"
    )
    p.add_argument(
        "--log-dir", default="./output/compose/", help="Directory for trajectory logs"
    )
    p.add_argument(
        "--min-clips",
        type=int,
        default=6,
        help="Minimum clips before ready_to_submit() allows SUBMIT",
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

    # Configure LMs (cache=False to prevent stale trajectory replay)
    lm = dspy.LM(args.model, cache=False)
    sub_lm = dspy.LM(args.sub_model, cache=False) if args.sub_model else lm
    dspy.configure(lm=lm)

    print("=== DSPy RLM Composer ===")
    print(f"  Model:      {args.model}")
    print(f"  Sub-model:  {args.sub_model or args.model}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  LLM calls:  {args.max_llm_calls}")
    print(f"  Log dir:    {args.log_dir}")
    print(f"  Min clips:  {args.min_clips}")
    print(f"  Brief:      {brief[:100]}{'...' if len(brief) > 100 else ''}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Tools:")
        for fn in make_tools.__code__.co_consts:
            if isinstance(fn, str) and fn.startswith("    "):
                continue
        # Show tool names from a dummy list
        tool_names = [
            "set_tempo",
            "get_status",
            "create_tracks",
            "browse",
            "load_instrument",
            "load_sound",
            "load_effect",
            "load_drum_kit",
            "create_clip",
            "create_arrangement_clip",
            "get_params",
            "set_param",
            "set_mix",
            "play",
            "stop",
            "fire_clip",
            "get_arrangement",
            "clear_arrangement",
            "note",
            "chord",
            "scale",
            "ready_to_submit",
        ]
        for name in tool_names:
            print(f"    - {name}")
        print(f"\n  [DRY RUN] Guide ({len(GUIDE)} chars):")
        print(GUIDE[:200] + "...")
        print("\n  [DRY RUN] Would connect to Ableton and run RLM. Exiting.")
        return

    # Connect to Ableton
    session = Session()
    tools = make_tools(session, min_clips=args.min_clips)

    # Build RLM
    composer = dspy.RLM(
        signature="brief, guide -> report",
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_output_chars=15000,
        tools=tools,
        sub_lm=sub_lm,
        verbose=True,
    )

    try:
        print("  Starting composition...\n")
        prediction = composer(brief=brief, guide=GUIDE)
        print("\n  === Report ===")
        print(f"  {prediction.report}")
        save_trajectory(prediction, brief, args.log_dir)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping playback...")
    finally:
        session.stop()
        session.close()


if __name__ == "__main__":
    main()
