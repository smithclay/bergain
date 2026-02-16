"""The Grand Lobby — A Wes Anderson score in 3/4.

G harmonic minor. 150 BPM. Two minutes of whimsy,
precision, and melancholy. Harpsichord, pizzicato strings,
glockenspiel, and cello. Alexandre Desplat at the front desk.
"""

from session import Set, Track, Scene

BPM = 150
M = 3  # beats per waltz measure

# G harmonic minor: G A Bb C D Eb F#
# Waltz chord voicings for pizzicato (root, [upper voices])
CHORDS = {
    "Gm": (43, [58, 62]),  # G2, Bb3+D4
    "D7": (50, [66, 69]),  # D3, F#4+A4
    "Cm": (48, [63, 67]),  # C3, Eb4+G4
    "Eb": (51, [67, 70]),  # Eb3, G4+Bb4
    "Bb": (46, [62, 65]),  # Bb2, D4+F4
}

# Chord progressions per section
PROM_PROG = [
    "Gm",
    "Gm",
    "D7",
    "D7",
    "Cm",
    "Cm",
    "D7",
    "D7",
    "Gm",
    "Gm",
    "Bb",
    "Bb",
    "Eb",
    "Eb",
    "D7",
    "D7",
    "Gm",
    "Gm",
    "D7",
    "D7",
    "Cm",
    "D7",
    "Gm",
    "Gm",
]

CHASE_PROG = [
    "Gm",
    "D7",
    "Cm",
    "D7",
    "Gm",
    "D7",
    "Eb",
    "D7",
    "Gm",
    "Bb",
    "Eb",
    "D7",
    "Cm",
    "D7",
    "Gm",
    "D7",
    "Gm",
    "Cm",
    "D7",
    "Gm",
    "Eb",
    "D7",
    "Gm",
    "Gm",
]

REPRISE_PROG = [
    "Gm",
    "Gm",
    "D7",
    "D7",
    "Cm",
    "Cm",
    "D7",
    "D7",
    "Gm",
    "Gm",
    "Bb",
    "Bb",
    "Eb",
    "Eb",
    "D7",
    "D7",
    "Gm",
    "Gm",
    "D7",
    "Gm",
]


# ---------------------------------------------------------------------------
# Waltz pattern helpers
# ---------------------------------------------------------------------------


def waltz_pizz(progression, root_vel=90, chord_vel=70):
    """Oom-pah-pah pizzicato from a chord progression."""
    notes = []
    for i, name in enumerate(progression):
        root, tones = CHORDS[name]
        b = float(i * M)
        notes.append((root, b, 0.4, root_vel))
        for p in tones:
            notes.append((p, b + 1, 0.3, chord_vel))
            notes.append((p, b + 2, 0.3, chord_vel - 5))
    return notes


def waltz_cello(progression, vel=75):
    """Sustained bass roots, one every 2 measures."""
    notes = []
    for i in range(0, len(progression), 2):
        root, _ = CHORDS[progression[i]]
        b = float(i * M)
        notes.append((root, b, float(M * 2 - 0.5), vel))
    return notes


# ---------------------------------------------------------------------------
# Section 1: The Lobby — 16 measures — Harpsichord solo
# ---------------------------------------------------------------------------


def keys_lobby(measures):
    """Harpsichord states the theme. Precise, poised, alone."""
    notes = []
    # Theme A (measures 1-8)
    theme_a = [
        (67, 0.0, 0.4, 85),  # G4 — home
        (70, 1.0, 0.4, 80),  # Bb4
        (74, 2.0, 0.7, 90),  # D5 — top of arpeggio
        (74, 3.0, 1.5, 85),  # D5 held
        (66, 5.0, 0.4, 75),  # F#4 — the Eastern European wink
        (67, 6.0, 0.4, 80),  # G4
        (69, 7.0, 0.4, 78),  # A4
        (72, 8.0, 0.7, 85),  # C5
        (70, 9.0, 2.5, 88),  # Bb4 held
        (69, 12.0, 0.4, 82),  # A4
        (70, 13.0, 0.4, 80),  # Bb4
        (72, 14.0, 0.7, 85),  # C5
        (74, 15.0, 0.5, 88),  # D5
        (72, 16.0, 0.4, 80),  # C5
        (70, 17.0, 0.4, 78),  # Bb4
        (69, 18.0, 0.5, 82),  # A4
        (66, 19.5, 0.4, 75),  # F#4
        (67, 20.5, 0.4, 78),  # G4
        (67, 21.0, 2.5, 85),  # G4 held — home
    ]
    # Theme A' (measures 9-16) — higher, more ornamental
    theme_a2 = [
        (74, 24.0, 0.4, 88),  # D5
        (75, 25.0, 0.4, 82),  # Eb5
        (74, 26.0, 0.4, 85),  # D5
        (79, 27.0, 2.0, 92),  # G5 — peak
        (78, 30.0, 0.3, 82),  # F#5
        (75, 31.0, 0.4, 80),  # Eb5
        (74, 32.0, 0.5, 78),  # D5
        (72, 33.0, 2.5, 85),  # C5 held
        (70, 36.0, 0.4, 80),  # Bb4
        (72, 37.0, 0.4, 78),  # C5
        (74, 38.0, 0.7, 82),  # D5
        (75, 39.0, 0.4, 85),  # Eb5
        (74, 40.0, 0.4, 80),  # D5
        (72, 41.0, 0.4, 78),  # C5
        (70, 42.0, 0.5, 80),  # Bb4
        (69, 43.5, 0.4, 75),  # A4
        (66, 44.5, 0.4, 72),  # F#4
        (67, 45.0, 2.5, 88),  # G4 — home
    ]
    for pitch, start, dur, vel in theme_a + theme_a2:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 2: The Promenade — 24 measures — Full ensemble
# ---------------------------------------------------------------------------


def keys_promenade(measures):
    """Harpsichord: melody restated with waltz support. More confident."""
    notes = []
    p1 = [
        (67, 0.0, 0.3, 90),
        (70, 1.0, 0.3, 85),
        (74, 2.0, 0.5, 92),
        (74, 3.0, 0.8, 88),
        (75, 4.0, 0.3, 80),
        (74, 5.0, 0.3, 82),
        (72, 6.0, 0.3, 85),
        (69, 7.0, 0.3, 80),
        (72, 8.0, 0.5, 85),
        (70, 9.0, 2.0, 88),
        (67, 12.0, 0.3, 85),
        (69, 13.0, 0.3, 82),
        (70, 14.0, 0.5, 85),
        (72, 15.0, 0.3, 88),
        (74, 16.0, 0.5, 90),
        (75, 17.0, 0.3, 82),
        (74, 18.0, 0.5, 88),
        (72, 19.0, 0.3, 82),
        (69, 20.0, 0.5, 80),
        (67, 21.0, 2.0, 85),
    ]
    p2 = [
        (74, 24.0, 0.3, 90),
        (74, 24.5, 0.2, 85),
        (75, 25.0, 0.3, 88),
        (74, 26.0, 0.5, 85),
        (79, 27.0, 1.5, 95),
        (78, 29.0, 0.3, 78),
        (79, 30.0, 0.3, 88),
        (75, 31.0, 0.3, 82),
        (72, 32.0, 0.5, 85),
        (74, 33.0, 2.0, 90),
        (70, 36.0, 0.3, 82),
        (72, 37.0, 0.3, 80),
        (74, 38.0, 0.5, 85),
        (75, 39.0, 0.5, 88),
        (74, 40.0, 0.3, 82),
        (72, 41.0, 0.5, 80),
        (70, 42.0, 0.5, 85),
        (69, 43.0, 0.3, 78),
        (66, 44.0, 0.5, 75),
        (67, 45.0, 2.0, 88),
    ]
    p3 = [
        (67, 48.0, 0.3, 85),
        (70, 49.0, 0.3, 82),
        (74, 50.0, 0.3, 88),
        (75, 51.0, 0.3, 90),
        (74, 52.0, 0.3, 85),
        (75, 53.0, 0.3, 88),
        (78, 54.0, 0.5, 92),
        (74, 55.0, 0.3, 82),
        (69, 56.0, 0.5, 80),
        (70, 57.0, 2.0, 85),
        (72, 60.0, 0.3, 85),
        (74, 61.0, 0.3, 88),
        (75, 62.0, 0.3, 90),
        (74, 63.0, 0.5, 88),
        (79, 64.0, 0.5, 95),
        (78, 65.0, 0.3, 85),
        (75, 66.0, 0.3, 82),
        (74, 67.0, 0.3, 88),
        (72, 68.0, 0.5, 85),
        (74, 69.0, 2.5, 92),
    ]
    for pitch, start, dur, vel in p1 + p2 + p3:
        notes.append((pitch, start, dur, vel))
    return notes


def strings_promenade(measures):
    return waltz_pizz(PROM_PROG, root_vel=88, chord_vel=68)


def bells_promenade(measures):
    """Glockenspiel: octave-up accents on key melody moments."""
    notes = []
    accents = [
        (79, 2.0, 0.4, 65),  # G5
        (86, 9.0, 0.8, 60),  # D6
        (82, 14.0, 0.4, 58),  # Bb5
        (86, 16.0, 0.4, 62),  # D6
        (79, 21.0, 0.8, 55),  # G5
        (91, 27.0, 1.0, 68),  # G6 — peak
        (84, 33.0, 0.8, 60),  # C6
        (79, 45.0, 0.8, 58),  # G5
        (90, 54.0, 0.4, 65),  # F#6
        (87, 57.0, 0.6, 58),  # Eb6
        (86, 64.0, 0.5, 68),  # D6
        (86, 69.0, 0.8, 62),  # D6
    ]
    for pitch, start, dur, vel in accents:
        notes.append((pitch, start, dur, vel))
    return notes


def cello_promenade(measures):
    return waltz_cello(PROM_PROG, vel=55)


# ---------------------------------------------------------------------------
# Section 3: The Chase — 24 measures — Peak energy
# ---------------------------------------------------------------------------


def keys_chase(measures):
    """Harpsichord: fast runs, Desplat at full tilt."""
    notes = []
    p1 = [
        (67, 0.0, 0.2, 88),
        (70, 0.5, 0.2, 82),
        (74, 1.0, 0.2, 90),
        (79, 1.5, 0.5, 95),
        (74, 2.0, 0.3, 82),
        (78, 3.0, 0.2, 85),
        (74, 3.5, 0.2, 80),
        (69, 4.0, 0.2, 78),
        (66, 4.5, 0.2, 75),
        (69, 5.0, 0.5, 82),
        (72, 6.0, 0.2, 85),
        (75, 6.5, 0.2, 82),
        (79, 7.0, 0.2, 88),
        (75, 7.5, 0.2, 80),
        (72, 8.0, 0.5, 85),
        (74, 9.0, 0.2, 90),
        (72, 9.5, 0.2, 82),
        (69, 10.0, 0.2, 85),
        (66, 10.5, 0.5, 88),
        (69, 11.0, 0.5, 80),
        (67, 12.0, 0.2, 85),
        (69, 12.5, 0.2, 80),
        (70, 13.0, 0.2, 82),
        (72, 13.5, 0.2, 85),
        (74, 14.0, 0.5, 90),
        (75, 15.0, 0.2, 88),
        (74, 15.5, 0.2, 82),
        (72, 16.0, 0.2, 80),
        (70, 16.5, 0.2, 78),
        (67, 17.0, 0.5, 85),
        (74, 18.0, 0.2, 90),
        (78, 18.5, 0.2, 85),
        (79, 19.0, 0.5, 95),
        (82, 19.5, 0.3, 85),
        (79, 20.0, 0.8, 92),
        (74, 21.0, 0.2, 85),
        (72, 21.5, 0.2, 80),
        (70, 22.0, 0.2, 78),
        (67, 22.5, 0.5, 82),
        (66, 23.0, 0.5, 78),
    ]
    p2 = [
        (79, 24.0, 0.2, 92),
        (74, 24.5, 0.2, 85),
        (70, 25.0, 0.2, 80),
        (67, 25.5, 0.5, 88),
        (70, 27.0, 0.2, 82),
        (74, 27.5, 0.2, 85),
        (79, 28.0, 0.5, 92),
        (82, 28.5, 0.3, 85),
        (75, 30.0, 0.2, 85),
        (79, 30.5, 0.2, 82),
        (82, 31.0, 0.5, 90),
        (79, 31.5, 0.2, 80),
        (75, 32.0, 0.5, 85),
        (74, 33.0, 0.2, 88),
        (69, 33.5, 0.2, 80),
        (66, 34.0, 0.5, 85),
        (69, 34.5, 0.2, 78),
        (74, 35.0, 0.5, 88),
        (72, 36.0, 0.2, 85),
        (67, 36.5, 0.2, 80),
        (63, 37.0, 0.5, 82),
        (67, 37.5, 0.2, 78),
        (72, 38.0, 0.5, 85),
        (74, 39.0, 0.2, 90),
        (78, 39.5, 0.2, 85),
        (74, 40.0, 0.5, 88),
        (69, 40.5, 0.2, 78),
        (66, 41.0, 0.5, 82),
        (67, 42.0, 0.2, 88),
        (70, 42.5, 0.2, 82),
        (74, 43.0, 0.2, 90),
        (79, 43.5, 0.5, 95),
        (79, 45.0, 0.3, 88),
        (74, 45.5, 0.3, 82),
        (70, 46.0, 0.3, 80),
        (67, 46.5, 1.0, 90),
    ]
    p3 = [
        (67, 48.0, 0.2, 90),
        (70, 48.33, 0.2, 85),
        (74, 48.67, 0.2, 88),
        (79, 49.0, 0.2, 92),
        (82, 49.33, 0.2, 88),
        (86, 49.67, 0.3, 95),
        (82, 50.0, 0.3, 88),
        (79, 50.5, 0.5, 92),
        (84, 51.0, 0.3, 90),
        (79, 51.5, 0.2, 85),
        (75, 52.0, 0.3, 88),
        (72, 52.5, 0.5, 85),
        (75, 53.0, 0.5, 82),
        (78, 54.0, 0.3, 92),
        (74, 54.5, 0.2, 85),
        (69, 55.0, 0.3, 82),
        (66, 55.5, 0.5, 88),
        (69, 56.0, 0.5, 80),
        (79, 57.0, 0.3, 95),
        (82, 57.5, 0.3, 90),
        (86, 58.0, 0.5, 100),
        (82, 58.5, 0.3, 88),
        (79, 59.0, 0.5, 92),
        (75, 60.0, 0.3, 85),
        (72, 60.5, 0.3, 82),
        (70, 61.0, 0.5, 80),
        (67, 61.5, 0.5, 78),
        (66, 62.0, 0.5, 75),
        (67, 63.0, 0.3, 80),
        (69, 63.5, 0.3, 78),
        (70, 64.0, 0.5, 82),
        (72, 64.5, 0.3, 80),
        (74, 65.0, 0.5, 85),
        (74, 66.0, 0.3, 88),
        (72, 66.5, 0.3, 82),
        (70, 67.0, 0.3, 80),
        (67, 67.5, 0.5, 85),
        (66, 68.0, 0.3, 78),
        (67, 69.0, 2.5, 92),
    ]
    for pitch, start, dur, vel in p1 + p2 + p3:
        notes.append((pitch, start, dur, vel))
    return notes


def strings_chase(measures):
    """Pizzicato: busier waltz with offbeat stabs."""
    notes = waltz_pizz(CHASE_PROG, root_vel=95, chord_vel=75)
    for i, name in enumerate(CHASE_PROG):
        _, tones = CHORDS[name]
        b = float(i * M)
        if i > 0 and i % 4 == 0:
            notes.append((tones[0], b - 0.5, 0.2, 72))
        if i % 2 == 0:
            notes.append((tones[1], b + 1.5, 0.2, 65))
    return notes


def bells_chase(measures):
    """Glockenspiel: gets its own counter-melody."""
    notes = []
    counter = [
        (91, 0.0, 0.3, 70),
        (87, 1.0, 0.3, 65),
        (86, 2.0, 0.5, 72),
        (84, 6.0, 0.3, 68),
        (86, 7.0, 0.3, 65),
        (87, 8.0, 0.5, 70),
        (91, 12.0, 0.3, 72),
        (90, 13.0, 0.3, 68),
        (87, 14.0, 0.5, 65),
        (86, 18.0, 0.5, 70),
        (84, 19.0, 0.3, 65),
        (82, 20.0, 0.5, 68),
        (79, 21.0, 1.0, 72),
        (91, 24.0, 0.2, 72),
        (87, 24.5, 0.2, 68),
        (86, 25.0, 0.3, 72),
        (84, 27.0, 0.3, 65),
        (86, 28.0, 0.5, 70),
        (87, 30.0, 0.2, 68),
        (91, 30.5, 0.2, 72),
        (87, 31.0, 0.5, 70),
        (86, 33.0, 0.3, 68),
        (84, 34.0, 0.3, 65),
        (86, 36.0, 0.3, 72),
        (87, 37.0, 0.3, 68),
        (91, 38.0, 0.5, 75),
        (86, 42.0, 0.3, 68),
        (84, 43.0, 0.3, 65),
        (82, 44.0, 0.3, 62),
        (91, 48.0, 0.3, 75),
        (87, 49.0, 0.3, 70),
        (86, 50.0, 0.3, 72),
        (91, 54.0, 0.3, 72),
        (90, 55.0, 0.3, 68),
        (87, 56.0, 0.3, 65),
        (86, 57.0, 0.5, 75),
        (91, 58.0, 0.5, 78),
        (86, 63.0, 0.5, 68),
        (91, 66.0, 0.3, 72),
        (86, 67.0, 0.3, 68),
        (79, 69.0, 1.5, 70),
    ]
    for pitch, start, dur, vel in counter:
        notes.append((pitch, start, dur, vel))
    return notes


def cello_chase(measures):
    """Cello: walking bass, one per measure."""
    notes = []
    for i, name in enumerate(CHASE_PROG):
        root, _ = CHORDS[name]
        b = float(i * M)
        notes.append((root, b, 2.5, 60))
    return notes


# ---------------------------------------------------------------------------
# Section 4: The Reprise — 20 measures — Instruments drop out
# ---------------------------------------------------------------------------


def keys_reprise(measures):
    """Harpsichord: theme returns, getting sparser as others leave."""
    notes = []
    theme = [
        (67, 0.0, 0.4, 78),
        (70, 1.0, 0.4, 72),
        (74, 2.0, 0.7, 82),
        (74, 3.0, 1.5, 78),
        (66, 5.0, 0.4, 68),
        (67, 6.0, 0.4, 72),
        (69, 7.0, 0.4, 70),
        (72, 8.0, 0.7, 78),
        (70, 9.0, 2.5, 80),
        (69, 12.0, 0.4, 75),
        (70, 13.0, 0.4, 72),
        (72, 14.0, 0.7, 78),
        (74, 15.0, 0.5, 80),
        (72, 16.0, 0.4, 72),
        (70, 17.0, 0.4, 70),
        (69, 18.0, 0.5, 75),
        (66, 19.5, 0.4, 68),
        (67, 20.5, 0.4, 70),
        (67, 21.0, 2.5, 78),
    ]
    variation = [
        (74, 24.0, 0.4, 75),
        (75, 25.0, 0.4, 70),
        (74, 26.0, 0.4, 72),
        (79, 27.0, 2.0, 80),
        (78, 30.0, 0.3, 72),
        (75, 31.0, 0.4, 68),
        (74, 32.0, 0.5, 70),
        (72, 33.0, 2.5, 75),
        # Strings gone — getting sparser
        (70, 36.0, 0.4, 68),
        (72, 37.0, 0.4, 65),
        (74, 38.0, 0.7, 70),
        (70, 39.0, 2.0, 72),
        (69, 42.0, 0.5, 65),
        (66, 44.0, 0.4, 60),
        (67, 45.0, 2.5, 68),
        # Cello gone — alone again
        (74, 48.0, 0.4, 62),
        (72, 50.0, 0.5, 58),
        (70, 51.0, 2.0, 60),
        (67, 54.0, 0.5, 55),
        (66, 56.0, 0.4, 50),
        (67, 57.0, 2.5, 58),
    ]
    for pitch, start, dur, vel in theme + variation:
        notes.append((pitch, start, dur, vel))
    return notes


def strings_reprise(measures):
    """Pizzicato: waltz for 12 measures, then gone."""
    return waltz_pizz(REPRISE_PROG[:12], root_vel=82, chord_vel=62)


def bells_reprise(measures):
    """Glockenspiel: 8 measures of sparse accents, then gone first."""
    notes = []
    accents = [
        (79, 2.0, 0.4, 55),  # G5
        (86, 9.0, 0.8, 50),  # D6
        (84, 14.0, 0.4, 48),  # C6
        (79, 21.0, 0.8, 45),  # G5 — last bell
    ]
    for pitch, start, dur, vel in accents:
        notes.append((pitch, start, dur, vel))
    return notes


def cello_reprise(measures):
    """Cello: sustained for 16 measures, then gone."""
    return waltz_cello(REPRISE_PROG[:16], vel=48)


# ---------------------------------------------------------------------------
# Section 5: The Curtain — 16 measures — Harpsichord alone
# ---------------------------------------------------------------------------


def keys_curtain(measures):
    """Harpsichord solo. The theme, sparser. Credits rolling."""
    notes = []
    final = [
        (67, 0.0, 0.5, 70),  # G4
        (70, 2.0, 0.5, 65),  # Bb4
        (74, 3.0, 1.0, 72),  # D5
        (74, 6.0, 2.0, 68),  # D5 held
        (66, 9.0, 0.5, 60),  # F#4
        (67, 12.0, 0.5, 65),  # G4
        (72, 14.0, 1.0, 68),  # C5
        (70, 15.0, 2.5, 70),  # Bb4 held
        (69, 21.0, 0.5, 62),  # A4
        (66, 23.0, 0.5, 55),  # F#4
        (67, 24.0, 2.0, 68),  # G4
        (74, 30.0, 1.0, 60),  # D5
        (72, 33.0, 0.5, 55),  # C5
        (70, 36.0, 1.5, 58),  # Bb4
        (69, 39.0, 0.5, 52),  # A4
        (66, 41.0, 0.5, 48),  # F#4
        # Final G minor chord
        (67, 42.0, 5.0, 62),  # G4
        (70, 42.0, 5.0, 55),  # Bb4
        (74, 42.0, 5.0, 58),  # D5
    ]
    for pitch, start, dur, vel in final:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

s = Set(
    name="The Grand Lobby",
    subtitle="A Wes Anderson Score",
    bpm=BPM,
    beats_per_bar=M,
    time_sig=(3, 4),
    tracks=[
        Track("Keys", sound="Harpsichord.adg", effects=["Reverb"], volume=0.80),
        Track("Strings", sound="Pizzicato.adv", volume=0.65, pan=-0.15),
        Track(
            "Bells",
            sound="Glockenspiel Basic.adv",
            effects=["Reverb"],
            volume=0.50,
            pan=0.20,
        ),
        Track("Cello", sound="Cello Strings.adv", volume=0.40),
    ],
)

sections = [
    Scene("The Lobby", 16, {"Keys": keys_lobby}),
    Scene(
        "The Promenade",
        24,
        {
            "Keys": keys_promenade,
            "Strings": strings_promenade,
            "Bells": bells_promenade,
            "Cello": cello_promenade,
        },
    ),
    Scene(
        "The Chase",
        24,
        {
            "Keys": keys_chase,
            "Strings": strings_chase,
            "Bells": bells_chase,
            "Cello": cello_chase,
        },
    ),
    Scene(
        "The Reprise",
        20,
        {
            "Keys": keys_reprise,
            "Strings": strings_reprise,
            "Bells": bells_reprise,
            "Cello": cello_reprise,
        },
    ),
    Scene("The Curtain", 16, {"Keys": keys_curtain}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    s.setup()

    s.session.param("Keys", 1, "decay", 0.2)
    print("    Keys/Reverb Decay -> 0.2 (small room)")

    s.session.param("Bells", 1, "decay", 0.25)
    print("    Bells/Reverb Decay -> 0.25 (sparkle)")

    s.load_scenes(sections)
    s.build_arrangement(sections)
    s.play()
    print("\n  The lobby is empty. The concierge nods.")
    s.teardown()


if __name__ == "__main__":
    main()
