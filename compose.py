"""DSPy RLM Composer — LM-driven composition for Ableton Live.

The RLM writes code that orchestrates sub-LM calls to compose music.
The brief is a symbolic variable the LM explores through code — creative
decisions happen through programmatic llm_query() calls in loops, results
accumulate in Python variables, and tools realize them in the DAW.

Usage:
    uv run python compose.py "Dark Berlin techno in F minor, 130 BPM, driving kick"
    uv run python compose.py --brief-file brief.txt
    uv run python compose.py --model openrouter/openai/gpt-5 "Dark ambient in Eb"
    uv run python compose.py --live --duration 60 "Evolving ambient in F minor"
    uv run python compose.py --dry-run "Test brief"
"""

import argparse
import json
import os
import random
import re
import time

import dspy
from dotenv import load_dotenv
from session import Session, Track

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
      - Each scene needs DIFFERENT energy/style/chords — monotony is failure.
      - Scenes are indexed 0, 1, 2... — lower slots = lower energy is natural
        but not required. The DJ chooses fire order.
      - ready_to_submit() checks milestones. Only SUBMIT after it says READY.
      - Most tools return JSON (parse with json.loads). Exceptions: browse() and
        get_status() return plain text.

    EXAMPLE (each step = one code block, separate execution):
      Step 1: browse("909", "Drums"); create_tracks(json.dumps([
                {"name":"Drums","drum_kit":"909 Core Kit.adg"},
                {"name":"Bass","instrument":"Operator"},
                {"name":"Pad","sound":"Warm Pad"}])); set_tempo(130)
      Step 2: plan = llm_query('Dark techno in F minor. Build 6 scenes for
                live performance. Return JSON: {"key":"F","scenes":[
                  {"name":"Ambient","slot":0,"bars":8,"energy":0.2,
                   "chords":["Fm"],"drums":"minimal","bass":"none","pad":"atmospheric"},
                  {"name":"Groove","slot":1,"bars":4,"energy":0.5,
                   "chords":["Fm","Cm"],"drums":"four_on_floor","bass":"pulsing_8th","pad":"sustained"},
                  {"name":"Drive","slot":2,"bars":4,"energy":0.7,...},
                  {"name":"Peak","slot":3,"bars":4,"energy":0.95,...},
                  {"name":"Breakdown","slot":4,"bars":8,"energy":0.3,...},
                  {"name":"Outro","slot":5,"bars":8,"energy":0.15,...}]}')
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
      SETUP (one step): browse() + create_tracks() + set_tempo() + play()
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
      - compose_next() takes a creative prompt, NOT parameter values.
        Good: "Bring in a deep walking bass, let the drums rest"
        Bad:  "energy 0.55, drums half_time, bass sustained"
      - compose_next() tracks history and enforces variety automatically.
      - Call get_arc_summary() to see energy curve, style stats, timing phase.
      - ready_to_submit() enforces minimum elapsed time. Don't rush.
      - Most tools return JSON (parse with json.loads). Exceptions: browse()
        and get_status() return plain text.

    EXAMPLE (each step = one code block, separate execution):
      Step 1: browse("909", "Drums"); create_tracks(json.dumps([
                {"name":"Drums","drum_kit":"909 Core Kit.adg"},
                {"name":"Bass","instrument":"Operator"},
                {"name":"Pad","sound":"Warm Pad"}])); set_tempo(128); play()
      Step 2: print(compose_next("Open with atmosphere — just pads, no rhythm, "
                "let the space breathe. Dreamy and mysterious."))
      Step 3: print(compose_next("Introduce a gentle pulse — minimal drums, "
                "maybe a soft bass note. Keep the energy low but add momentum."))
      Step 4: print(compose_next("Build — more rhythmic drive, bring the bass "
                "forward, add some chord movement."))
      Step 5: arc = json.loads(get_arc_summary())
              print(f"Phase: {arc['phase']}, trend: {arc['energy_trend']}")
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
# Tools — closures over a Session instance.
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


# ---------------------------------------------------------------------------
# Music theory helpers (module-level so renderers can use them)
# ---------------------------------------------------------------------------

_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "m": [0, 3, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "dom7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "min7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

_SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "blues": [0, 3, 5, 6, 7, 10],
}


def _note_to_midi(name: str) -> int:
    """Convert note name to MIDI number. 'F4' -> 65, 'Bb3' -> 58."""
    s = name.strip()
    letter = s[0].upper()
    rest = s[1:]
    accidental = 0
    while rest and rest[0] in "#b":
        accidental += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octave = int(rest) if rest else 4
    midi = (octave + 1) * 12 + _NOTE_NAMES[letter] + accidental
    return max(0, min(127, midi))


def _parse_chord_name(name: str) -> tuple:
    """Parse chord string into (root_pitch_class_0_11, interval_list).

    Examples: 'Fm' -> (5, [0,3,7]), 'Cm7' -> (0, [0,3,7,10]),
              'Ab' -> (8, [0,4,7]), 'Bbdim' -> (10, [0,3,6])
    """
    s = name.strip()
    letter = s[0].upper()
    rest = s[1:]
    accidental = 0
    while rest and rest[0] in "#b":
        accidental += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    root_pc = (_NOTE_NAMES[letter] + accidental) % 12
    # Match quality — try longest first
    quality = rest if rest else "maj"
    intervals = _CHORD_INTERVALS.get(quality)
    if intervals is None:
        # Default to major if unrecognized
        intervals = _CHORD_INTERVALS["maj"]
    return (root_pc, intervals)


def _chord_to_midi(name: str, octave: int) -> list:
    """Return MIDI pitches for a named chord at a given octave."""
    root_pc, intervals = _parse_chord_name(name)
    base = (octave + 1) * 12 + root_pc
    return [max(0, min(127, base + i)) for i in intervals]


# ---------------------------------------------------------------------------
# Renderers — return list[tuple[pitch, start, dur, vel]]
# Each uses a seeded Random for deterministic-per-section output.
# ---------------------------------------------------------------------------

# GM drum map
_KICK = 36
_SNARE = 38
_CLAP = 39
_CLOSED_HAT = 42
_OPEN_HAT = 46
_RIDE = 51
_PERC = 47


def _render_drums(bars, energy, style, section_name="x", beats_per_bar=4):
    """Render drum notes for a section.

    Returns list of (pitch, start_beat, duration, velocity) tuples.
    """
    rng = random.Random(hash(section_name) & 0xFFFFFFFF)
    notes = []
    total_beats = bars * beats_per_bar
    base_vel = int(60 + 60 * energy)

    def v():
        return max(1, min(127, base_vel + rng.randint(-15, 15)))

    if style == "four_on_floor":
        for beat in range(total_beats):
            # Kick on every beat
            if energy >= 0.0:
                notes.append((_KICK, float(beat), 0.5, v()))
            # Closed hat on 8ths
            if energy >= 0.2:
                notes.append((_CLOSED_HAT, float(beat), 0.25, v() - 15))
                notes.append((_CLOSED_HAT, beat + 0.5, 0.25, v() - 25))
            # Snare/clap on 2 and 4
            if energy >= 0.4 and beat % beats_per_bar in (1, 3):
                notes.append(
                    (_CLAP if rng.random() > 0.5 else _SNARE, float(beat), 0.5, v())
                )
            # Open hat on offbeats
            if energy >= 0.6 and beat % beats_per_bar == 2:
                notes.append((_OPEN_HAT, beat + 0.5, 0.25, v() - 10))
            # Ride layer
            if energy >= 0.6 and beat % 2 == 0:
                notes.append((_RIDE, float(beat), 0.5, v() - 20))
            # Fills every 4 bars
            if energy >= 0.8 and beat % (beats_per_bar * 4) == (beats_per_bar * 4 - 1):
                for sub in [0.0, 0.25, 0.5, 0.75]:
                    notes.append((_SNARE, beat + sub, 0.25, v()))

    elif style == "half_time":
        for beat in range(total_beats):
            # Kick on 1
            if beat % beats_per_bar == 0:
                notes.append((_KICK, float(beat), 0.5, v()))
            # Snare on 3 only (half time feel)
            if energy >= 0.3 and beat % beats_per_bar == 2:
                notes.append((_SNARE, float(beat), 0.5, v()))
            # Hats
            if energy >= 0.2:
                notes.append((_CLOSED_HAT, float(beat), 0.25, v() - 20))
            if energy >= 0.6 and beat % 2 == 1:
                notes.append((_OPEN_HAT, float(beat), 0.25, v() - 15))

    elif style == "breakbeat":
        # Syncopated kick/snare pattern repeating every 2 bars
        kick_pattern = [0, 0.75, 1.5, 2.5, 3.0]  # offbeat kicks
        snare_pattern = [1.0, 3.0, 3.5]
        pattern_len = beats_per_bar * 2
        for bar_start in range(0, total_beats, pattern_len):
            for offset in kick_pattern:
                pos = bar_start + offset
                if pos < total_beats:
                    notes.append((_KICK, pos, 0.5, v()))
            if energy >= 0.4:
                for offset in snare_pattern:
                    pos = bar_start + offset
                    if pos < total_beats:
                        notes.append((_SNARE, pos, 0.5, v()))
            if energy >= 0.2:
                for sub_beat in range(pattern_len * 2):  # 8ths
                    pos = bar_start + sub_beat * 0.5
                    if pos < total_beats:
                        notes.append((_CLOSED_HAT, pos, 0.25, v() - 20))

    elif style == "minimal":
        for beat in range(total_beats):
            # Sparse kick — always on beat 0, probabilistic elsewhere
            if beat % beats_per_bar == 0 and (beat == 0 or rng.random() < 0.7):
                notes.append((_KICK, float(beat), 0.5, v() - 10))
            # Occasional hat
            if energy >= 0.2 and rng.random() < 0.3:
                notes.append((_CLOSED_HAT, float(beat), 0.25, v() - 25))
            # Rare snare ghost
            if energy >= 0.4 and beat % (beats_per_bar * 2) == 3:
                notes.append((_SNARE, float(beat), 0.5, v() - 20))

    elif style == "shuffle":
        # Swung triplet feel — kick on 1/3, hats on triplet grid
        for beat in range(total_beats):
            # Kick on 1 and 3
            if beat % beats_per_bar in (0, 2):
                notes.append((_KICK, float(beat), 0.5, v()))
            # Swung hat triplets (straight + swing offset)
            if energy >= 0.2:
                notes.append((_CLOSED_HAT, float(beat), 0.2, v() - 20))
                notes.append((_CLOSED_HAT, beat + 0.67, 0.2, v() - 28))
            # Snare on 2 and 4 with ghost notes
            if energy >= 0.3 and beat % beats_per_bar in (1, 3):
                notes.append((_SNARE, float(beat), 0.5, v()))
            if energy >= 0.5 and rng.random() < 0.3:
                notes.append((_SNARE, beat + 0.67, 0.25, v() - 30))
            # Open hat accents
            if energy >= 0.6 and beat % beats_per_bar == 3 and rng.random() < 0.4:
                notes.append((_OPEN_HAT, beat + 0.67, 0.3, v() - 15))

    elif style == "sparse_perc":
        # Percussion texture only — no kick, no snare. Ride, hats, perc.
        for beat in range(total_beats):
            # Ride as anchor
            if beat % 2 == 0:
                notes.append((_RIDE, float(beat), 0.5, v() - 15))
            # Sporadic closed hat
            if rng.random() < 0.25 * energy:
                notes.append((_CLOSED_HAT, beat + rng.random() * 0.5, 0.2, v() - 25))
            # Perc hits on offbeats
            if energy >= 0.3 and rng.random() < 0.2:
                notes.append((_PERC, beat + 0.5, 0.3, v() - 10))
            # Open hat swells
            if energy >= 0.4 and beat % (beats_per_bar * 2) == 0:
                notes.append((_OPEN_HAT, float(beat), 0.5, v() - 20))

    return notes


def _render_bass(
    bars, energy, style, chords, key_root, section_name="x", beats_per_bar=4
):
    """Render bass notes for a section.

    chords: list of chord name strings, evenly distributed across bars.
    Returns list of (pitch, start_beat, duration, velocity) tuples.
    """
    rng = random.Random(hash(section_name) & 0xFFFFFFFF)
    notes = []
    total_beats = bars * beats_per_bar
    base_vel = int(50 + 70 * energy)

    if not chords:
        chords = [key_root]

    # Distribute chords evenly across bars
    beats_per_chord = total_beats / len(chords)

    for i, chord_name in enumerate(chords):
        root_pc, _ = _parse_chord_name(chord_name)
        chord_start = i * beats_per_chord
        chord_end = chord_start + beats_per_chord

        # Bass octave: 2 normally, occasional jump to 3 at high energy
        octave = 2
        root_midi = (octave + 1) * 12 + root_pc

        def v():
            return max(1, min(127, base_vel + rng.randint(-10, 10)))

        if style == "sustained":
            # Whole-note / half-note sustained bass
            dur = beats_per_chord if energy < 0.5 else beats_per_chord / 2
            beat = chord_start
            while beat < chord_end - 0.01:
                notes.append((root_midi, beat, dur, v()))
                beat += dur

        elif style == "pulsing_8th":
            beat = chord_start
            while beat < chord_end - 0.01:
                pitch = root_midi
                if energy >= 0.7 and rng.random() < 0.2:
                    pitch = root_midi + 12  # octave jump
                dur = 0.4 if energy < 0.5 else 0.45
                notes.append((pitch, beat, dur, v()))
                beat += 0.5

        elif style == "offbeat_8th":
            beat = chord_start + 0.5  # offbeat start
            while beat < chord_end - 0.01:
                pitch = root_midi
                if energy >= 0.7 and rng.random() < 0.15:
                    pitch = root_midi + 12
                notes.append((pitch, beat, 0.4, v()))
                beat += 1.0

        elif style == "rolling_16th":
            beat = chord_start
            fifth_midi = root_midi + 7  # perfect fifth
            while beat < chord_end - 0.01:
                pitch = root_midi
                # Add fifth on some 16ths for movement
                if rng.random() < 0.2:
                    pitch = fifth_midi
                if energy >= 0.7 and rng.random() < 0.1:
                    pitch = root_midi + 12
                vel = v()
                # Accent on beat
                if abs(beat % 1.0) < 0.01:
                    vel = min(127, vel + 15)
                notes.append((pitch, beat, 0.2, vel))
                beat += 0.25

        elif style == "walking":
            # Jazz walking bass — quarter notes stepping through chord tones
            # and chromatic passing tones
            chord_pitches = [root_midi, root_midi + 3, root_midi + 7, root_midi + 5]
            beat = chord_start
            step_idx = 0
            while beat < chord_end - 0.01:
                base = chord_pitches[step_idx % len(chord_pitches)]
                # Occasional chromatic approach from below
                if rng.random() < 0.2 and step_idx > 0:
                    base = chord_pitches[(step_idx + 1) % len(chord_pitches)] - 1
                # Occasional octave jump at high energy
                if energy >= 0.6 and rng.random() < 0.15:
                    base += 12
                notes.append((max(0, min(127, base)), beat, 0.9, v()))
                beat += 1.0
                step_idx += 1

    return notes


def _render_pads(
    bars, energy, style, chords, key_root, section_name="x", beats_per_bar=4
):
    """Render pad/chord notes for a section.

    chords: list of chord name strings, evenly distributed across bars.
    Returns list of (pitch, start_beat, duration, velocity) tuples.
    """
    rng = random.Random(hash(section_name) & 0xFFFFFFFF)
    notes = []
    total_beats = bars * beats_per_bar
    base_vel = int(40 + 60 * energy)

    if not chords:
        chords = [key_root]

    beats_per_chord = total_beats / len(chords)

    for i, chord_name in enumerate(chords):
        chord_start = i * beats_per_chord
        chord_end = chord_start + beats_per_chord

        # Voicing octave based on energy
        if energy < 0.3:
            # Open voicing: spread across octaves 3-5
            pitches = _chord_to_midi(chord_name, 3)
            # Spread: move some notes up an octave
            if len(pitches) >= 3:
                pitches = [pitches[0], pitches[1] + 12, pitches[2] + 12]
                if len(pitches) > 3:
                    pitches.append(pitches[3] + 24)
        elif energy < 0.7:
            # Close voicing in octave 4
            pitches = _chord_to_midi(chord_name, 4)
        else:
            # Stacked voicing: octave 3 + 4
            low = _chord_to_midi(chord_name, 3)
            high = _chord_to_midi(chord_name, 4)
            pitches = low + high

        def v():
            return max(1, min(127, base_vel + rng.randint(-8, 8)))

        if style == "sustained":
            dur = beats_per_chord
            for p in pitches:
                notes.append((p, chord_start, dur, v()))

        elif style == "atmospheric":
            # Longer notes with slight overlap for wash effect
            dur = beats_per_chord + 0.5
            for p in pitches:
                notes.append((p, chord_start, dur, v()))

        elif style == "pulsing":
            # Rhythmic chords — quarter or half note pulses
            pulse = 2.0 if energy < 0.5 else 1.0
            gap = 0.25
            beat = chord_start
            while beat < chord_end - 0.01:
                dur = pulse - gap
                for p in pitches:
                    notes.append((p, beat, dur, v()))
                beat += pulse

        elif style == "arpeggiated":
            # Broken chord arpeggios — 8th notes cycling through chord tones
            beat = chord_start
            idx = 0
            step = 0.5  # 8th notes
            while beat < chord_end - 0.01:
                p = pitches[idx % len(pitches)]
                vel = v()
                # Accent first note of each beat
                if abs(beat % 1.0) < 0.01:
                    vel = min(127, vel + 10)
                notes.append((p, beat, step * 0.9, vel))
                beat += step
                idx += 1

        elif style == "swells":
            # Crescendo/decrescendo pads — velocity ramps up then down
            dur = beats_per_chord
            mid = dur / 2.0
            for p in pitches:
                # Split into two notes: rising and falling velocity
                vel_rise = max(1, min(127, base_vel - 20 + rng.randint(-5, 5)))
                vel_peak = max(1, min(127, base_vel + 20 + rng.randint(-5, 5)))
                notes.append((p, chord_start, mid, vel_rise))
                notes.append((p, chord_start + mid, mid, vel_peak))

    return notes


# ---------------------------------------------------------------------------
# Tool closures
# ---------------------------------------------------------------------------


def make_tools(
    session, min_clips=6, live_mode=False, duration_minutes=60, sub_lm=None, brief=""
):
    """Create tool closures over a live Session, with milestone tracking."""

    # Milestone tracker — shared mutable state across all tool closures.
    # ready_to_submit() checks these before allowing SUBMIT.
    _done = {
        "browse": False,
        "tracks": False,
        "clips": 0,
        "mix": False,
        "status_checks": 0,
    }

    # Track role map: {"TrackName": "drums"|"bass"|"pad"} — set by create_tracks()
    _track_roles = {}

    # Live mode state — tracks elapsed time for wait/elapsed/ready_to_submit
    _live_state = {"start_time": None, "target_duration": duration_minutes}

    # Live mode history — compose_next() maintains this automatically
    _live_history = []

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
        """Create MIDI tracks. REPLACES ALL existing tracks — call ONCE with full layout.

        Args:
            tracks_json: JSON array of track specs, each with:
              name (required), sound/instrument/drum_kit (pick one to auto-load),
              effects (list of effect names), volume (0.0-1.0), pan (-1.0 to 1.0)

        Example: json.dumps([
            {"name": "Drums", "drum_kit": "909 Core Kit.adg", "volume": 0.9},
            {"name": "Bass", "instrument": "Operator", "volume": 0.85},
            {"name": "Pad", "sound": "Drifting Ambient Pad.adv", "volume": 0.8}
        ])
        """
        try:
            _done["tracks"] = True
            specs = json.loads(tracks_json)
            _track_roles.clear()
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
                # Auto-detect track role for write_clip()
                if s.get("drum_kit"):
                    _track_roles[s["name"]] = "drums"
                elif "bass" in s["name"].lower():
                    _track_roles[s["name"]] = "bass"
                else:
                    _track_roles[s["name"]] = "pad"
            count = session.setup(tracks)
            return json.dumps(
                {"tracks_created": count, "names": [t.name for t in tracks]}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def browse(query: str, category: str = "") -> str:
        """Search Ableton browser for instruments, sounds, drums, or effects.
        Returns matching names as plain text, one per line (NOT JSON).

        Args:
            query: Search term (e.g. "909", "strings", "Operator")
            category: Optional filter — 'Instruments', 'Sounds', 'Drums', 'Audio Effects'

        Returns NO_RESULTS if nothing matches — try broader terms.
        Example: browse("909", "Drums") might return "909 Core Kit.adg\\nClap 909.aif\\n..."
        Parse with: names = result.split('\\n')
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
        """Create a session MIDI clip (loopable).

        Args:
            track: Track name (must match a name from create_tracks)
            slot: Scene/row index (0, 1, 2, ...) — each slot is one clip
            length_beats: Clip length in beats (e.g. 16.0 = 4 bars in 4/4)
            notes_json: JSON array of [pitch, start_beat, duration, velocity]
            name: Clip name for the session display

        Note format: [pitch, start_beat, duration, velocity]
          - start_beat is RELATIVE to clip start (0 = first beat of clip)
          - Pitch 0-127, velocity 1-127 (VARY velocity — flat = robotic)
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
            if live_mode and _live_state["start_time"] is None:
                _live_state["start_time"] = time.time()
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

    def fire_scene(slot: int) -> str:
        """Fire a scene (all clips in a row) by slot index. Use for live transitions."""
        try:
            session.fire(int(slot))
            return json.dumps({"fired_scene": int(slot)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def write_clip(clip_json: str) -> str:
        """Write a looping clip to the session grid.

        Args:
            clip_json: JSON object with:
              name (str): Clip name, e.g. "Main Groove", "Breakdown Pad"
              slot (int): Scene/row index (0, 1, 2, ...) — each scene is a
                          different energy level or section type
              bars (int): Loop length in bars (typically 4 or 8)
              energy (float): 0.0 (silent/sparse) to 1.0 (full power)
              chords (list[str]): Chord names, e.g. ["Fm", "Cm7"]
              key (str): Key root note, e.g. "F"
              drums (str): "four_on_floor"|"half_time"|"breakbeat"|"minimal"|"shuffle"|"sparse_perc"|"none"
              bass (str): "rolling_16th"|"offbeat_8th"|"pulsing_8th"|"sustained"|"walking"|"none"
              pad (str): "sustained"|"atmospheric"|"pulsing"|"arpeggiated"|"swells"|"none"

        Creates looping session clips for each track role. Clips in the same
        slot (scene) can be fired together for instant transitions.
        Returns JSON summary of clips created.
        """
        try:
            c = json.loads(clip_json)
            name = c["name"]
            slot = int(c["slot"])
            bars = int(c["bars"])
            energy = max(0.0, min(1.0, float(c["energy"])))
            chords = c.get("chords", [])
            key_root = c.get("key", "C")
            drum_style = c.get("drums", "none")
            bass_style = c.get("bass", "none")
            pad_style = c.get("pad", "none")
            length_beats = float(bars * 4)

            clips_created = 0
            details = []

            for track_name, role in _track_roles.items():
                notes = []
                clip_name = ""

                if role == "drums" and drum_style != "none":
                    notes = _render_drums(bars, energy, drum_style, section_name=name)
                    clip_name = f"Drums {name}"
                elif role == "bass" and bass_style != "none":
                    notes = _render_bass(
                        bars, energy, bass_style, chords, key_root, section_name=name
                    )
                    clip_name = f"Bass {name}"
                elif role == "pad" and pad_style != "none":
                    notes = _render_pads(
                        bars, energy, pad_style, chords, key_root, section_name=name
                    )
                    clip_name = f"Pad {name}"

                if notes:
                    clamped = [_clamp_note(n) for n in notes]
                    session.clip(
                        track_name, slot, length_beats, clamped, name=clip_name
                    )
                    clips_created += 1
                    details.append(f"{clip_name}: {len(clamped)} notes")

            _done["clips"] += clips_created
            return json.dumps(
                {
                    "clip": name,
                    "slot": slot,
                    "clips_created": clips_created,
                    "bars": bars,
                    "energy": energy,
                    "details": details,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_elapsed():
        """Internal helper: return (elapsed_min, remaining_min) or None if not started."""
        if _live_state["start_time"] is None:
            return None
        elapsed_sec = time.time() - _live_state["start_time"]
        elapsed_min = elapsed_sec / 60.0
        remaining_min = max(0.0, _live_state["target_duration"] - elapsed_min)
        return elapsed_min, remaining_min

    def wait(bars: int) -> str:
        """Wait for a number of bars to play. Sleeps in real time based on current tempo.

        Args:
            bars: Number of bars to wait (e.g. 4 or 8)

        Use after fire_scene() to let a section play before composing the next one.
        Returns elapsed and remaining minutes.
        """
        try:
            bpm = session.status().get("tempo", 120)
            seconds = int(bars) * 4 * 60.0 / bpm
            time.sleep(seconds)
            info = _get_elapsed()
            if info:
                elapsed_min, remaining_min = info
                return json.dumps(
                    {
                        "waited_bars": int(bars),
                        "waited_seconds": round(seconds, 1),
                        "elapsed_minutes": round(elapsed_min, 1),
                        "remaining_minutes": round(remaining_min, 1),
                    }
                )
            return json.dumps(
                {"waited_bars": int(bars), "waited_seconds": round(seconds, 1)}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def elapsed() -> str:
        """Get elapsed and remaining time for the live performance.

        Returns JSON with elapsed_minutes, remaining_minutes, target_minutes.
        Call periodically to know when to wrap up.
        """
        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            return json.dumps(
                {
                    "elapsed_minutes": round(elapsed_min, 1),
                    "remaining_minutes": round(remaining_min, 1),
                    "target_minutes": _live_state["target_duration"],
                }
            )
        return json.dumps(
            {
                "elapsed_minutes": 0,
                "remaining_minutes": _live_state["target_duration"],
                "target_minutes": _live_state["target_duration"],
                "note": "Timer starts on first play() call",
            }
        )

    def compose_next(creative_prompt: str) -> str:
        """Compose and perform the next section in one step. The sub-LM makes
        all musical decisions — you provide creative DIRECTION, not parameters.

        Args:
            creative_prompt: Describe the FEELING and DIRECTION you want.
                Do NOT specify exact parameter values — let the sub-LM surprise you.
                Good: "Build tension — bring in a driving bass, keep drums sparse"
                Good: "Drop to a quiet interlude, just pads and atmosphere"
                Good: "Peak energy, all instruments full power, complex rhythms"
                Good: "Shift to a new key for freshness, walking bass, arpeggiated pad"
                Bad:  "energy 0.58, drums half_time, bass sustained, key Eb"

        Internally: calls sub-LM → writes clips → fires scene → waits.
        History is tracked and compressed automatically.
        Returns JSON summary of what was composed and performed.
        """
        try:
            if sub_lm is None:
                return json.dumps(
                    {"error": "compose_next requires sub_lm (live mode only)"}
                )

            # Compressed history: last 5 full entries + summary of earlier
            recent = _live_history[-5:]
            earlier_summary = ""
            if len(_live_history) > 5:
                earlier = _live_history[:-5]
                energies = [h["energy"] for h in earlier]
                keys_used = sorted(set(h.get("key", "?") for h in earlier))
                styles = set()
                for h in earlier:
                    styles.add(
                        (h.get("drums", "?"), h.get("bass", "?"), h.get("pad", "?"))
                    )
                earlier_summary = (
                    f"Earlier ({len(earlier)} sections): energy {min(energies):.2f}"
                    f"-{max(energies):.2f}, keys: {'/'.join(keys_used)}, "
                    f"{len(styles)} unique style combos.\n"
                )

            # Style combo frequency for variety enforcement
            combo_counts = {}
            for h in _live_history:
                combo = (
                    h.get("drums", "none"),
                    h.get("bass", "none"),
                    h.get("pad", "none"),
                )
                combo_counts[combo] = combo_counts.get(combo, 0) + 1
            overused = [
                f"{d}/{b}/{p}" for (d, b, p), c in combo_counts.items() if c >= 3
            ]
            variety_note = ""
            if overused:
                variety_note = (
                    f"OVERUSED combos (pick something DIFFERENT): "
                    f"{', '.join(overused)}.\n"
                )

            # Arc phase guidance
            arc_phase = ""
            info = _get_elapsed()
            if info:
                elapsed_min, remaining_min = info
                total = _live_state["target_duration"]
                if elapsed_min < total * 0.33:
                    arc_phase = (
                        f"OPENING third ({elapsed_min:.0f}/{total} min). "
                        "Build gradually — energy should stay low-to-moderate (0.1-0.5). "
                        "8-16 bar loops let mood settle in.\n"
                    )
                elif elapsed_min < total * 0.66:
                    arc_phase = (
                        f"MIDDLE third ({elapsed_min:.0f}/{total} min). "
                        "Create waves — alternate between lifts and dips (0.3-0.8). "
                        "This is where the most variety belongs. "
                        "4-bar loops for peaks, 8-bar for dips.\n"
                    )
                else:
                    arc_phase = (
                        f"FINAL third ({elapsed_min:.0f}/{total} min). "
                        "Begin winding down — energy should generally descend (0.5-0.1). "
                        "16-bar loops for hypnotic fade, sparser textures.\n"
                    )

            # Breakdown nudge: if energy sustained high without a valley, suggest one
            breakdown_nudge = ""
            if len(_live_history) >= 3:
                recent_3 = _live_history[-3:]
                sustained_high = all(h["energy"] >= 0.6 for h in recent_3)
                recent_has_breakdown = any(
                    h.get("bass") == "none" for h in _live_history[-4:]
                )
                if sustained_high and not recent_has_breakdown:
                    breakdown_nudge = (
                        "Consider a breakdown — sustained high energy needs "
                        "a valley to rebuild from.\n"
                    )

            next_slot = max((h["slot"] for h in _live_history), default=-1) + 1

            prompt = (
                f"You are composing the next section of a live performance.\n"
                f"Brief: {brief}\n"
                f"{arc_phase}"
                f"{earlier_summary}"
                f"Recent sections: {json.dumps(recent)}\n"
                f"{variety_note}"
                f"{breakdown_nudge}"
                f"Creative direction: {creative_prompt}\n\n"
                f"Return a single JSON object with these fields:\n"
                f"  name (str): evocative section name\n"
                f"  slot (int): use {next_slot}\n"
                f"  bars (int): 4, 8, 16, or 32\n"
                f"  energy (float): 0.0 to 1.0 — calibration: 0.2=ambient, "
                f"0.4=gentle groove, 0.6=driving, 0.8=peak power, 0.95=maximum\n"
                f"  key (str): root note like 'F', 'Eb', 'Ab'\n"
                f"  chords (list[str]): 2-4 chord names like 'Fm7', 'Abmaj9'\n"
                f"  drums: 'four_on_floor'|'half_time'|'breakbeat'|'minimal'|"
                f"'shuffle'|'sparse_perc'|'none'\n"
                f"  bass: 'rolling_16th'|'offbeat_8th'|'pulsing_8th'|'sustained'|"
                f"'walking'|'none'\n"
                f"  pad: 'sustained'|'atmospheric'|'pulsing'|'arpeggiated'|"
                f"'swells'|'none'\n\n"
                f"Make bold creative choices that serve the direction. "
                f"Respond with ONLY the JSON object."
            )

            completions = sub_lm(messages=[{"role": "user", "content": prompt}])
            raw = completions[0] if completions else ""
            # sub_lm returns dicts with 'text' key — normalise to string
            if isinstance(raw, dict):
                response_text = raw.get("text") or raw.get("content") or str(raw)
            else:
                response_text = str(raw)

            m = re.search(r"\{.*\}", response_text, re.S)
            if not m:
                return json.dumps(
                    {
                        "error": "Sub-LM returned no valid JSON",
                        "raw": response_text[:500],
                    }
                )
            section = json.loads(m.group())

            # Validate and clamp
            section.setdefault("slot", next_slot)
            section["slot"] = int(section["slot"])
            section.setdefault("bars", 8)
            section["bars"] = int(section["bars"])
            section.setdefault("energy", 0.5)
            section["energy"] = max(0.0, min(1.0, float(section["energy"])))

            # Energy guardrails: nudge sub-LM output toward arc-appropriate range
            if info:
                elapsed_min, _ = info
                total = _live_state["target_duration"]
                if elapsed_min < total * 0.33:
                    # Opening: allow 0.05–0.55
                    section["energy"] = max(0.05, min(0.55, section["energy"]))
                elif elapsed_min < total * 0.66:
                    # Middle: boost floor to 0.3 so peaks actually peak
                    section["energy"] = max(0.3, section["energy"])
                    # High-energy middle sections: cap bars at 8 for tighter loops
                    if section["energy"] >= 0.6:
                        section["bars"] = min(section["bars"], 8)
                # Final: no override, let it wind down naturally
            section.setdefault("key", "C")
            section.setdefault("chords", [section["key"] + "m7"])
            section.setdefault("drums", "none")
            section.setdefault("bass", "none")
            section.setdefault("pad", "sustained")
            section.setdefault("name", f"Section {section['slot']}")

            clip_result = write_clip(json.dumps(section))
            clip_info = json.loads(clip_result)
            if "error" in clip_info:
                return clip_result

            # Transition fades: fade out tracks that go from active to "none"
            fade_tracks = []
            if _live_history:
                prev = _live_history[-1]
                role_to_style = {"drums": "drums", "bass": "bass", "pad": "pad"}
                for track_name, role in _track_roles.items():
                    style_key = role_to_style.get(role)
                    if not style_key:
                        continue
                    was_active = prev.get(style_key, "none") != "none"
                    now_silent = section.get(style_key, "none") == "none"
                    if was_active and now_silent:
                        try:
                            session.fade(track_name, 0.0, steps=4, duration=0.5)
                            fade_tracks.append(track_name)
                        except Exception:
                            pass

            fire_scene(section["slot"])

            # Restore faded tracks to default volume after scene fires
            for track_name in fade_tracks:
                try:
                    session.fade(track_name, 0.85, steps=4, duration=0.5)
                except Exception:
                    pass

            wait_result = wait(section["bars"])
            wait_info = json.loads(wait_result)

            # Texture density: fraction of instruments active (0.0-1.0)
            styles = [
                section.get("drums", "none"),
                section.get("bass", "none"),
                section.get("pad", "none"),
            ]
            density = sum(1 for s in styles if s != "none") / len(styles)

            _live_history.append(
                {
                    "section": section["name"],
                    "slot": section["slot"],
                    "energy": section["energy"],
                    "density": round(density, 2),
                    "key": section["key"],
                    "chords": section.get("chords", [])[:4],
                    "drums": section.get("drums", "none"),
                    "bass": section.get("bass", "none"),
                    "pad": section.get("pad", "none"),
                }
            )

            return json.dumps(
                {
                    "composed": section["name"],
                    "slot": section["slot"],
                    "energy": section["energy"],
                    "key": section["key"],
                    "chords": section.get("chords", []),
                    "style": (
                        f"{section.get('drums', 'none')}/"
                        f"{section.get('bass', 'none')}/"
                        f"{section.get('pad', 'none')}"
                    ),
                    "bars": section["bars"],
                    "clips_created": clip_info.get("clips_created", 0),
                    "elapsed_minutes": wait_info.get("elapsed_minutes"),
                    "remaining_minutes": wait_info.get("remaining_minutes"),
                    "sections_so_far": len(_live_history),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_arc_summary() -> str:
        """Get a summary of the performance arc so far.

        Returns energy curve, style combo stats, time phase, and variety metrics.
        Use this to understand where you are and what to do next.
        """
        if not _live_history:
            return json.dumps({"sections": 0, "note": "No sections composed yet"})

        energies = [h["energy"] for h in _live_history]
        densities = [h.get("density", 0.33) for h in _live_history]
        keys_used = {}
        combo_counts = {}
        for h in _live_history:
            k = h.get("key", "?")
            keys_used[k] = keys_used.get(k, 0) + 1
            combo = (h.get("drums", "?"), h.get("bass", "?"), h.get("pad", "?"))
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

        # Energy bands
        bands = {"low (0-0.3)": 0, "mid (0.3-0.6)": 0, "high (0.6-1.0)": 0}
        for e in energies:
            if e < 0.3:
                bands["low (0-0.3)"] += 1
            elif e < 0.6:
                bands["mid (0.3-0.6)"] += 1
            else:
                bands["high (0.6-1.0)"] += 1

        # Recent energy trend (last 5)
        recent_energies = energies[-5:]
        trend = "flat"
        if len(recent_energies) >= 3:
            diffs = [
                recent_energies[i + 1] - recent_energies[i]
                for i in range(len(recent_energies) - 1)
            ]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0.03:
                trend = "rising"
            elif avg_diff < -0.03:
                trend = "falling"

        # Contrast between adjacent sections
        contrasts = []
        for i in range(1, len(_live_history)):
            prev, curr = _live_history[i - 1], _live_history[i]
            energy_delta = abs(curr["energy"] - prev["energy"])
            style_changes = sum(
                1
                for k in ("drums", "bass", "pad")
                if curr.get(k, "none") != prev.get(k, "none")
            )
            contrasts.append(energy_delta + style_changes * 0.1)

        avg_contrast = round(sum(contrasts) / len(contrasts), 2) if contrasts else 0.0
        # Flag monotony: low energy delta AND same instruments for 3+ sections
        low_contrast_warning = None
        if len(_live_history) >= 3:
            tail = _live_history[-3:]
            tail_energy_deltas = [
                abs(tail[i]["energy"] - tail[i - 1]["energy"])
                for i in range(1, len(tail))
            ]
            tail_style_same = all(
                tail[i].get(k) == tail[0].get(k)
                for i in range(1, len(tail))
                for k in ("drums", "bass", "pad")
            )
            if max(tail_energy_deltas) < 0.1 and tail_style_same:
                low_contrast_warning = (
                    "Low contrast for 3+ sections — risk of monotony. "
                    "Change instruments or make a bigger energy shift."
                )

        result = {
            "sections": len(_live_history),
            "energy_range": [round(min(energies), 2), round(max(energies), 2)],
            "energy_avg": round(sum(energies) / len(energies), 2),
            "energy_trend": trend,
            "energy_bands": bands,
            "density_avg": round(sum(densities) / len(densities), 2),
            "density_current": densities[-1],
            "avg_contrast": avg_contrast,
            "keys_used": keys_used,
            "unique_style_combos": len(combo_counts),
            "most_used_combos": [
                {"style": f"{d}/{b}/{p}", "count": c}
                for (d, b, p), c in sorted(combo_counts.items(), key=lambda x: -x[1])[
                    :5
                ]
            ],
        }

        if low_contrast_warning:
            result["warning"] = low_contrast_warning

        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            total = _live_state["target_duration"]
            result["elapsed_minutes"] = round(elapsed_min, 1)
            result["remaining_minutes"] = round(remaining_min, 1)
            phase = (
                "opening"
                if elapsed_min < total * 0.33
                else "middle"
                if elapsed_min < total * 0.66
                else "final"
            )
            result["phase"] = phase

        return json.dumps(result)

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
                f"only {_done['clips']} clips created (need at least {min_clips}) — build more scenes"
            )
        if live_mode:
            # In live mode: check elapsed time instead of mix requirement
            info = _get_elapsed()
            if info:
                elapsed_min, _ = info
                threshold = _live_state["target_duration"] * 0.8
                if elapsed_min < threshold:
                    issues.append(
                        f"only {elapsed_min:.1f} min elapsed (need ~{threshold:.0f} min) — keep evolving"
                    )
            else:
                issues.append("play() not called yet — start playback first")
        else:
            if not _done["mix"]:
                issues.append("set_mix() not called — balance volumes and panning")
            if _done["status_checks"] < 1:
                issues.append(
                    "get_status() never called — verify your work before submitting"
                )
        if issues:
            return "NOT READY to submit:\n" + "\n".join(f"  - {i}" for i in issues)
        return "READY — call SUBMIT(report) in the NEXT step (no other tool calls)."

    tools = [
        set_tempo,
        get_status,
        create_tracks,
        browse,
        load_instrument,
        load_sound,
        load_effect,
        load_drum_kit,
        create_clip,
        get_params,
        set_param,
        set_mix,
        play,
        stop,
        fire_clip,
        fire_scene,
        write_clip,
        ready_to_submit,
    ]
    if live_mode:
        tools.extend([wait, elapsed, compose_next, get_arc_summary])
    return tools


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

    # Live mode defaults — bump iteration/call budgets if user didn't set them
    if args.live:
        if "--max-iterations" not in " ".join(os.sys.argv):
            args.max_iterations = 60
        if "--max-llm-calls" not in " ".join(os.sys.argv):
            args.max_llm_calls = 60
        if "--min-clips" not in " ".join(os.sys.argv):
            args.min_clips = 3

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
    session = Session()
    tools = make_tools(
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
        save_trajectory(prediction, brief, args.log_dir)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping playback...")
    finally:
        session.stop()
        session.close()


if __name__ == "__main__":
    main()
