"""Music theory helpers and MIDI renderers.

Pure functions: note/chord math, drum/bass/pad rendering.
No DAW or session dependencies — everything operates on note tuples.
"""

import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

CHORD_INTERVALS = {
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

# GM drum map
KICK = 36
SNARE = 38
CLAP = 39
CLOSED_HAT = 42
OPEN_HAT = 46
RIDE = 51
PERC = 47  # GM: Low-Mid Tom, used as general percussion texture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clamp_note(note):
    """Clamp a [pitch, start, duration, velocity] to valid MIDI ranges."""
    pitch = max(0, min(127, int(note[0])))
    start = max(0.0, float(note[1]))
    duration = max(0.01, float(note[2]))
    velocity = max(1, min(127, int(note[3])))
    return (pitch, start, duration, velocity)


def note_to_midi(name: str) -> int:
    """Convert note name to MIDI number. 'F4' -> 65, 'Bb3' -> 58."""
    s = name.strip()
    letter = s[0].upper()
    rest = s[1:]
    accidental = 0
    while rest and rest[0] in "#b":
        accidental += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octave = int(rest) if rest else 4
    midi = (octave + 1) * 12 + NOTE_NAMES[letter] + accidental
    return max(0, min(127, midi))


def parse_chord_name(name: str) -> tuple:
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
    root_pc = (NOTE_NAMES[letter] + accidental) % 12
    # Match quality — try longest first
    quality = rest if rest else "maj"
    intervals = CHORD_INTERVALS.get(quality)
    if intervals is None:
        # Default to major if unrecognized
        intervals = CHORD_INTERVALS["maj"]
    return (root_pc, intervals)


def chord_to_midi(name: str, octave: int) -> list:
    """Return MIDI pitches for a named chord at a given octave."""
    root_pc, intervals = parse_chord_name(name)
    base = (octave + 1) * 12 + root_pc
    return [max(0, min(127, base + i)) for i in intervals]


# ---------------------------------------------------------------------------
# Renderers — return list[tuple[pitch, start, dur, vel]]
# Each uses a seeded Random for deterministic-per-section output.
# ---------------------------------------------------------------------------


def render_drums(bars, energy, style, section_name="x", beats_per_bar=4):
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
            notes.append((KICK, float(beat), 0.5, v()))
            # Closed hat on 8ths
            if energy >= 0.2:
                notes.append((CLOSED_HAT, float(beat), 0.25, v() - 15))
                notes.append((CLOSED_HAT, beat + 0.5, 0.25, v() - 25))
            # Snare/clap on 2 and 4
            if energy >= 0.4 and beat % beats_per_bar in (1, 3):
                notes.append(
                    (CLAP if rng.random() > 0.5 else SNARE, float(beat), 0.5, v())
                )
            # Open hat on offbeats
            if energy >= 0.6 and beat % beats_per_bar == 2:
                notes.append((OPEN_HAT, beat + 0.5, 0.25, v() - 10))
            # Ride layer
            if energy >= 0.6 and beat % 2 == 0:
                notes.append((RIDE, float(beat), 0.5, v() - 20))
            # Fills every 4 bars
            if energy >= 0.8 and beat % (beats_per_bar * 4) == (beats_per_bar * 4 - 1):
                for sub in [0.0, 0.25, 0.5, 0.75]:
                    notes.append((SNARE, beat + sub, 0.25, v()))

    elif style == "half_time":
        for beat in range(total_beats):
            # Kick on 1
            if beat % beats_per_bar == 0:
                notes.append((KICK, float(beat), 0.5, v()))
            # Snare on 3 only (half time feel)
            if energy >= 0.3 and beat % beats_per_bar == 2:
                notes.append((SNARE, float(beat), 0.5, v()))
            # Hats
            if energy >= 0.2:
                notes.append((CLOSED_HAT, float(beat), 0.25, v() - 20))
            if energy >= 0.6 and beat % 2 == 1:
                notes.append((OPEN_HAT, float(beat), 0.25, v() - 15))

    elif style == "breakbeat":
        # Syncopated kick/snare pattern repeating every 2 bars
        kick_pattern = [0, 0.75, 1.5, 2.5, 3.0]  # offbeat kicks
        snare_pattern = [1.0, 3.0, 3.5]
        pattern_len = beats_per_bar * 2
        for bar_start in range(0, total_beats, pattern_len):
            for offset in kick_pattern:
                pos = bar_start + offset
                if pos < total_beats:
                    notes.append((KICK, pos, 0.5, v()))
            if energy >= 0.4:
                for offset in snare_pattern:
                    pos = bar_start + offset
                    if pos < total_beats:
                        notes.append((SNARE, pos, 0.5, v()))
            if energy >= 0.2:
                for sub_beat in range(pattern_len * 2):  # 8ths
                    pos = bar_start + sub_beat * 0.5
                    if pos < total_beats:
                        notes.append((CLOSED_HAT, pos, 0.25, v() - 20))

    elif style == "minimal":
        for beat in range(total_beats):
            # Sparse kick — always on beat 0, probabilistic elsewhere
            if beat % beats_per_bar == 0 and (beat == 0 or rng.random() < 0.7):
                notes.append((KICK, float(beat), 0.5, v() - 10))
            # Occasional hat
            if energy >= 0.2 and rng.random() < 0.3:
                notes.append((CLOSED_HAT, float(beat), 0.25, v() - 25))
            # Rare snare ghost
            if energy >= 0.4 and beat % (beats_per_bar * 2) == 3:
                notes.append((SNARE, float(beat), 0.5, v() - 20))

    elif style == "shuffle":
        # Swung triplet feel — kick on 1/3, hats on triplet grid
        for beat in range(total_beats):
            # Kick on 1 and 3
            if beat % beats_per_bar in (0, 2):
                notes.append((KICK, float(beat), 0.5, v()))
            # Swung hat triplets (straight + swing offset)
            if energy >= 0.2:
                notes.append((CLOSED_HAT, float(beat), 0.2, v() - 20))
                notes.append((CLOSED_HAT, beat + 0.67, 0.2, v() - 28))
            # Snare on 2 and 4 with ghost notes
            if energy >= 0.3 and beat % beats_per_bar in (1, 3):
                notes.append((SNARE, float(beat), 0.5, v()))
            if energy >= 0.5 and rng.random() < 0.3:
                notes.append((SNARE, beat + 0.67, 0.25, v() - 30))
            # Open hat accents
            if energy >= 0.6 and beat % beats_per_bar == 3 and rng.random() < 0.4:
                notes.append((OPEN_HAT, beat + 0.67, 0.3, v() - 15))

    elif style == "sparse_perc":
        # Percussion texture only — no kick, no snare. Ride, hats, perc.
        for beat in range(total_beats):
            # Ride as anchor
            if beat % 2 == 0:
                notes.append((RIDE, float(beat), 0.5, v() - 15))
            # Sporadic closed hat
            if rng.random() < 0.25 * energy:
                notes.append((CLOSED_HAT, beat + rng.random() * 0.5, 0.2, v() - 25))
            # Perc hits on offbeats
            if energy >= 0.3 and rng.random() < 0.2:
                notes.append((PERC, beat + 0.5, 0.3, v() - 10))
            # Open hat swells
            if energy >= 0.4 and beat % (beats_per_bar * 2) == 0:
                notes.append((OPEN_HAT, float(beat), 0.5, v() - 20))

    return notes


def render_bass(
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
        root_pc, _ = parse_chord_name(chord_name)
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
            # Walking pattern: root, minor 3rd, 5th, 4th — intentionally dark voicing
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


def render_pads(
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
            # Open voicing: spread across octaves 3-4
            pitches = chord_to_midi(chord_name, 3)
            if len(pitches) >= 3:
                pitches = [pitches[0], pitches[1] + 12, pitches[2] + 12]
        elif energy < 0.7:
            # Close voicing in octave 4
            pitches = chord_to_midi(chord_name, 4)
        else:
            # Stacked voicing: octave 3 + 4
            low = chord_to_midi(chord_name, 3)
            high = chord_to_midi(chord_name, 4)
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
