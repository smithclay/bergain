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
    """You are composing music in a live DAW by writing Python code that calls
    music tools. Describe the SHAPE of each section (energy, style, chords)
    and write_section() renders the MIDI notes for you.

    WORKFLOW — one phase per code step, never combine phases:
      1. BROWSE & SETUP: browse() + create_tracks() + set_tempo() — all in ONE step.
      2. PLAN: llm_query() for creative decisions — key, energy arc, chord progression
         per section. Must cover FULL duration. Request JSON, parse with re.search.
      3. COMPOSE: write_section() — ONE call per section, sequential on timeline.
      4. MIX: set_mix() + get_status() + ready_to_submit()
      5. SUBMIT: SUBMIT(report) alone.

    RULES:
      - Use llm_query() to decide key, chords, energy arc, and style names.
      - write_section() handles all MIDI rendering — you never build note arrays.
      - Each section needs DIFFERENT energy/style/chords — monotony is failure.
      - ready_to_submit() checks milestones. Only SUBMIT after it says READY.
      - Most tools return JSON (parse with json.loads). Exceptions: browse() and
        get_status() return plain text.

    EXAMPLE (each step = one code block, separate execution):
      Step 1: browse("909", "Drums"); create_tracks(json.dumps([
                {"name":"Drums","drum_kit":"909 Core Kit.adg"},
                {"name":"Bass","instrument":"Operator"},
                {"name":"Pad","sound":"Warm Pad"}])); set_tempo(130)
      Step 2: plan = llm_query('Dark techno in F minor, 96 bars (3 min at 130).
                Return JSON: {"key":"F","sections":[
                  {"name":"Intro","bars":16,"energy":0.3,"chords":["Fm","Cm"],
                   "drums":"minimal","bass":"sustained","pad":"atmospheric"},
                  {"name":"Build","bars":16,"energy":0.6,
                   "chords":["Fm","Cm7","Ab","Eb"],
                   "drums":"four_on_floor","bass":"pulsing_8th","pad":"sustained"},
                  ...]}')
      Step 3: write_section(json.dumps({"name":"Intro","start_beat":0,"bars":16,
                "energy":0.3,"key":"F","chords":["Fm","Cm"],
                "drums":"minimal","bass":"sustained","pad":"atmospheric"}))
      Step 4: write_section(json.dumps({"name":"Build","start_beat":64,"bars":16,
                "energy":0.6,"key":"F","chords":["Fm","Cm7","Ab","Eb"],
                "drums":"four_on_floor","bass":"pulsing_8th","pad":"sustained"}))
      ...
      Step N-1: set_mix(json.dumps({"Drums":0.9,"Bass":0.85,"Pad":0.7}));
                print(get_status()); print(ready_to_submit())
      Step N: SUBMIT(report)
    """

    brief: str = dspy.InputField(
        description="Creative brief describing mood, genre, tempo, instrumentation, and duration"
    )
    report: str = dspy.OutputField(
        description="Summary of what was BUILT in the DAW: tempo, tracks with loaded devices, "
        "sections with bar ranges and clip names and note counts, mix levels, key and progression"
    )


class ComposeSession(dspy.Signature):
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

    return notes


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

    # Track role map: {"TrackName": "drums"|"bass"|"pad"} — set by create_tracks()
    _track_roles = {}

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
                # Auto-detect track role for write_section()
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
        """Create a session MIDI clip (loopable, good for jamming).

        Args:
            track: Track name (must match a name from create_tracks)
            slot: Section index (0, 1, 2, ...) — each slot is one clip
            length_beats: Clip length in beats (e.g. 16.0 = 4 bars in 4/4)
            notes_json: JSON array of [pitch, start_beat, duration, velocity]
            name: Clip name for the arrangement display

        Note format: [pitch, start_beat, duration, velocity]
          - start_beat is RELATIVE to clip start (0 = first beat of clip)
          - Pitch 0-127, velocity 1-127 (VARY velocity — flat = robotic)
          - Duration: bass/pads = full bar (4.0), melody = mixed lengths
          - Velocity dynamics: pp=30, p=50, mp=65, mf=80, f=100, ff=120
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
        """Create an arrangement clip on the timeline (good for linear pieces).

        Args:
            track: Track name
            start_beat: Position on timeline in beats (0 = bar 1, 16 = bar 5, etc.)
            length_beats: Clip length in beats
            notes_json: JSON array of [pitch, start_beat, duration, velocity]
            name: Clip name shown in arrangement

        IMPORTANT: Note start_beat values are RELATIVE to the clip start (0 = first
        beat of THIS clip), NOT absolute timeline positions.
        See create_clip() docstring for note format and velocity guidance.
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

    def write_section(section_json: str) -> str:
        """Write a full section to the arrangement timeline.

        Args:
            section_json: JSON object with:
              name (str): Section name, e.g. "Intro", "Build A", "Peak", "Break"
              start_beat (int): Timeline position in beats (0 = bar 1)
              bars (int): Section length in bars
              energy (float): 0.0 (silent/sparse) to 1.0 (full power)
              chords (list[str]): Chord names, e.g. ["Fm", "Cm7", "Ab", "Eb"]
              key (str): Key root note, e.g. "F"
              drums (str): "four_on_floor"|"half_time"|"breakbeat"|"minimal"|"none"
              bass (str): "rolling_16th"|"offbeat_8th"|"pulsing_8th"|"sustained"|"none"
              pad (str): "sustained"|"atmospheric"|"pulsing"|"none"

        Renders drums, bass, and pad clips for this section based on energy level
        and style. Each track with a matching role gets an arrangement clip.
        Returns JSON summary of clips created.
        """
        try:
            sec = json.loads(section_json)
            name = sec["name"]
            start_beat = float(sec["start_beat"])
            bars = int(sec["bars"])
            energy = max(0.0, min(1.0, float(sec["energy"])))
            chords = sec.get("chords", [])
            key_root = sec.get("key", "C")
            drum_style = sec.get("drums", "none")
            bass_style = sec.get("bass", "none")
            pad_style = sec.get("pad", "none")
            length_beats = bars * 4

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
                    session.arr_clip(
                        track_name,
                        start_beat,
                        float(length_beats),
                        clamped,
                        name=clip_name,
                    )
                    clips_created += 1
                    details.append(f"{clip_name}: {len(clamped)} notes")

            _done["clips"] += clips_created
            return json.dumps(
                {
                    "section": name,
                    "clips_created": clips_created,
                    "bars": bars,
                    "energy": energy,
                    "details": details,
                }
            )
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
              drums (str): "four_on_floor"|"half_time"|"breakbeat"|"minimal"|"none"
              bass (str): "rolling_16th"|"offbeat_8th"|"pulsing_8th"|"sustained"|"none"
              pad (str): "sustained"|"atmospheric"|"pulsing"|"none"

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

    # Common tools shared by both modes
    _common = [
        set_tempo,
        get_status,
        create_tracks,
        browse,
        load_instrument,
        load_sound,
        load_effect,
        load_drum_kit,
        get_params,
        set_param,
        set_mix,
        play,
        stop,
        fire_clip,
        ready_to_submit,
    ]

    return {
        "arrangement": _common
        + [
            create_clip,
            create_arrangement_clip,
            get_arrangement,
            clear_arrangement,
            write_section,
        ],
        "session": _common
        + [
            create_clip,
            write_clip,
        ],
    }


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
        "--mode",
        choices=["arrangement", "session"],
        default="arrangement",
        help="Composition mode: 'arrangement' (timeline) or 'session' (clip grid)",
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

    # Select signature based on mode
    sig = ComposeSession if args.mode == "session" else Compose

    print("=== DSPy RLM Composer ===")
    print(f"  Mode:       {args.mode}")
    print(f"  Model:      {args.model}")
    print(f"  Sub-model:  {args.sub_model or args.model}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  LLM calls:  {args.max_llm_calls}")
    print(f"  Log dir:    {args.log_dir}")
    print(f"  Min clips:  {args.min_clips}")
    print(f"  Brief:      {brief[:100]}{'...' if len(brief) > 100 else ''}")
    print()

    if args.dry_run:
        print(f"  [DRY RUN] Signature: {sig.__name__}")
        print(f"  [DRY RUN] Docstring ({len(sig.__doc__)} chars):")
        print(sig.__doc__[:300] + "...")
        print("\n  [DRY RUN] Would connect to Ableton and run RLM. Exiting.")
        return

    # Connect to Ableton
    session = Session()
    tools_by_mode = make_tools(session, min_clips=args.min_clips)
    tools = tools_by_mode[args.mode]

    # Build RLM — signature docstring IS the prompt (highest priority)
    composer = dspy.RLM(
        sig,
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
