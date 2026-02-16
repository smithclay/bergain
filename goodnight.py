"""Goodnight, Fog City — 45 minutes of winter SF ambient.

Eb major / C minor. 55 BPM. A slow descent into sleep.
Four tracks become three become two become one.
The fog swallows everything.
"""

from session import Set, Track, Scene, make_automation

BPM = 55

# Eb major pentatonic: Eb F Ab Bb C
# Open voicing chords: stacked 4ths/5ths, never resolving hard

# Pad chord voicings (open, stacked 4ths/5ths)
PAD_Eb = [39, 46, 51]  # Eb2 Bb2 Eb3
PAD_Cm = [36, 43, 48]  # C2  G2  C3
PAD_Ab = [32, 39, 44]  # Ab1 Eb2 Ab2
PAD_Fm = [41, 48, 53]  # F2  C3  F3


# ---------------------------------------------------------------------------
# Section 1: Fog Arrives — 80 bars — City still awake
# All 4 tracks. Most "present" section.
# ---------------------------------------------------------------------------


def dusk_fog(bars):
    """Piano: Eb pentatonic melody, dreamy but with presence.
    Notes spaced every 4-8 bars. Like hearing someone practice
    through an open window across the street."""
    notes = []
    # Phrase A: opening statement (bars 1-20)
    melody_a = [
        (63, 0.0, 6.0, 48),  # Eb4 — home, settling in
        (70, 12.0, 4.0, 42),  # Bb4 — a 5th up, open
        (68, 24.0, 5.0, 45),  # Ab4 — dropping back
        (72, 36.0, 8.0, 50),  # C5  — the highest point of the whole piece
        (63, 52.0, 4.0, 38),  # Eb4 — return
    ]
    # Phrase B: response (bars 21-40)
    melody_b = [
        (65, 80.0, 5.0, 44),  # F4  — new color
        (68, 96.0, 6.0, 48),  # Ab4 — warmth
        (70, 112.0, 4.0, 40),  # Bb4 — floating
        (75, 124.0, 8.0, 52),  # Eb5 — octave bloom
        (72, 144.0, 5.0, 42),  # C5  — settling
    ]
    # Phrase C: receding (bars 41-60)
    melody_c = [
        (63, 164.0, 8.0, 40),  # Eb4 — long, quiet
        (58, 184.0, 4.0, 35),  # Bb3 — dropping down
        (56, 200.0, 6.0, 38),  # Ab3 — lower register
        (60, 220.0, 5.0, 36),  # C4  — barely there
    ]
    # Phrase D: last gestures (bars 61-80)
    melody_d = [
        (63, 248.0, 10.0, 32),  # Eb4 — long fade
        (68, 272.0, 6.0, 28),  # Ab4 — ghostly
        (63, 296.0, 12.0, 24),  # Eb4 — dissolving into section 2
    ]
    for pitch, start, dur, vel in melody_a + melody_b + melody_c + melody_d:
        notes.append((pitch, start, dur, vel))
    # Ghost notes: octave below, very quiet, like tape bleed
    ghosts = [
        (51, 2.0, 4.0, 18),  # Eb3
        (58, 14.0, 3.0, 15),  # Bb3
        (56, 26.0, 3.0, 16),  # Ab3
        (60, 38.0, 4.0, 20),  # C4
        (51, 54.0, 3.0, 14),  # Eb3
        (53, 82.0, 3.0, 16),  # F3
        (56, 98.0, 4.0, 18),  # Ab3
    ]
    for pitch, start, dur, vel in ghosts:
        notes.append((pitch, start, dur, vel))
    return notes


def wool_fog(bars):
    """Pad: warm Eb drones, open voicings. Changes every 16 bars.
    Full presence — this is the warmest the pad will ever be."""
    notes = []
    progression = [PAD_Eb, PAD_Cm, PAD_Eb, PAD_Fm, PAD_Ab]
    vels = [38, 40, 36, 42, 35]
    for i, (chord, vel) in enumerate(zip(progression, vels)):
        start = float(i * 64)
        for p in chord:
            notes.append((p, start, 62.0, vel))
    return notes


def glass_fog(bars):
    """Saw: high shimmer, catching streetlights. Single tones
    in the upper register, very quiet, with chorus making them glow."""
    notes = []
    tones = [
        (80, 8.0, 12.0, 30),  # Ab5
        (82, 40.0, 8.0, 28),  # Bb5
        (84, 72.0, 16.0, 32),  # C6
        (80, 120.0, 10.0, 26),  # Ab5
        (75, 160.0, 14.0, 30),  # Eb5
        (82, 208.0, 8.0, 24),  # Bb5
        (80, 256.0, 12.0, 22),  # Ab5 — fading
        (75, 296.0, 16.0, 18),  # Eb5 — barely there
    ]
    for pitch, start, dur, vel in tones:
        notes.append((pitch, start, dur, vel))
    return notes


def deep_fog(bars):
    """Sub: Eb drone, fundamental anchor. Slow swells."""
    notes = []
    subs = [
        (39, 0.0, 120.0, 45),  # Eb2 — first 30 bars
        (36, 140.0, 100.0, 40),  # C2  — bars 36-61
        (39, 260.0, 56.0, 35),  # Eb2 — bars 66-80, fading
    ]
    for pitch, start, dur, vel in subs:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 2: Streetlight Halos — 80 bars
# All 4 tracks. Piano fragments spacier. Glass thins. Sub drops octave.
# ---------------------------------------------------------------------------


def dusk_halos(bars):
    """Piano: fragments further apart now. Like remembering
    a melody you heard hours ago. Notes arrive, pause, arrive."""
    notes = []
    phrases = [
        (68, 0.0, 8.0, 40),  # Ab4
        (70, 20.0, 6.0, 36),  # Bb4
        (63, 44.0, 10.0, 38),  # Eb4 — long
        (72, 76.0, 5.0, 34),  # C5
        (65, 108.0, 8.0, 30),  # F4
        (68, 144.0, 12.0, 28),  # Ab4 — very long, losing energy
        (63, 192.0, 8.0, 25),  # Eb4
        (60, 240.0, 10.0, 22),  # C4 — dropping register
        (56, 288.0, 12.0, 18),  # Ab3 — ghostly
    ]
    for pitch, start, dur, vel in phrases:
        notes.append((pitch, start, dur, vel))
    return notes


def wool_halos(bars):
    """Pad: thicker now, adding an upper voice. The warmest point
    before it starts thinning."""
    notes = []
    progression = [
        ([39, 46, 51, 58], 0.0),  # Eb2 Bb2 Eb3 Bb3 — extra voice
        ([36, 43, 48, 55], 64.0),  # C2  G2  C3  G3
        ([32, 39, 44, 51], 128.0),  # Ab1 Eb2 Ab2 Eb3
        ([41, 48, 53, 60], 192.0),  # F2  C3  F3  C4
        ([39, 46, 51], 256.0),  # Eb2 Bb2 Eb3 — back to 3 voices
    ]
    vels = [42, 40, 38, 40, 34]
    for (chord, start), vel in zip(progression, vels):
        for p in chord:
            notes.append((p, start, 60.0, vel))
    return notes


def glass_halos(bars):
    """Saw: thinning. Fewer notes, lower velocity. Stars fading."""
    notes = []
    tones = [
        (75, 16.0, 16.0, 24),  # Eb5
        (80, 80.0, 10.0, 20),  # Ab5
        (77, 160.0, 12.0, 18),  # F5
        (75, 240.0, 16.0, 14),  # Eb5 — barely audible
    ]
    for pitch, start, dur, vel in tones:
        notes.append((pitch, start, dur, vel))
    return notes


def deep_halos(bars):
    """Sub: drops an octave. Eb1 territory. Felt, not heard."""
    notes = []
    subs = [
        (27, 0.0, 140.0, 38),  # Eb1 — deep
        (24, 160.0, 120.0, 32),  # C1  — deeper
    ]
    for pitch, start, dur, vel in subs:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 3: The Richmond — 88 bars — Outer Sunset emptiness
# Dusk + Wool + Glass. No sub.
# ---------------------------------------------------------------------------


def dusk_richmond(bars):
    """Piano: barely there. One note every 16-20 bars.
    A piano left in an empty room — someone touches a key
    every few minutes."""
    notes = []
    fragments = [
        (63, 0.0, 12.0, 28),  # Eb4
        (68, 68.0, 8.0, 24),  # Ab4
        (58, 144.0, 16.0, 20),  # Bb3 — dropping
        (56, 240.0, 10.0, 16),  # Ab3 — ghostly
        (51, 312.0, 14.0, 12),  # Eb3 — last piano-ish note
    ]
    for pitch, start, dur, vel in fragments:
        notes.append((pitch, start, dur, vel))
    return notes


def wool_richmond(bars):
    """Pad: carries everything now. Longer chords, slower changes.
    22-bar holds. This is the bed the listener sinks into."""
    notes = []
    progression = [
        (PAD_Eb, 0.0),
        (PAD_Ab, 88.0),
        (PAD_Cm, 176.0),
        (PAD_Eb, 264.0),
    ]
    vels = [36, 34, 32, 30]
    for (chord, start), vel in zip(progression, vels):
        for p in chord:
            notes.append((p, start, 84.0, vel))
    return notes


def glass_richmond(bars):
    """Saw: fading out. Two notes, very quiet.
    The last glass tones of the piece."""
    notes = []
    tones = [
        (75, 32.0, 20.0, 14),  # Eb5
        (72, 180.0, 16.0, 10),  # C5 — last glass note, barely a whisper
    ]
    for pitch, start, dur, vel in tones:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 4: Last Bus — 80 bars
# Dusk + Wool + Deep. No glass.
# Something still moves, then stops. Piano dies out halfway.
# ---------------------------------------------------------------------------


def dusk_lastbus(bars):
    """Piano: walking notes in the first half, like wheels on wet
    pavement. Then silence. The bus has passed."""
    notes = []
    walking = [
        (63, 0.0, 4.0, 25),  # Eb4
        (65, 16.0, 3.0, 22),  # F4
        (63, 32.0, 5.0, 24),  # Eb4
        (60, 52.0, 4.0, 20),  # C4
        (58, 72.0, 6.0, 22),  # Bb3
        (56, 96.0, 4.0, 18),  # Ab3
        (53, 116.0, 5.0, 16),  # F3
        (51, 140.0, 8.0, 14),  # Eb3 — last piano note, fading
    ]
    for pitch, start, dur, vel in walking:
        notes.append((pitch, start, dur, vel))
    # Bars 36-80: silence. The bus has gone.
    return notes


def wool_lastbus(bars):
    """Pad: steady warmth. Picks up the weight the piano leaves behind."""
    notes = []
    progression = [
        (PAD_Eb, 0.0),
        (PAD_Cm, 80.0),
        (PAD_Fm, 160.0),
        (PAD_Eb, 240.0),
    ]
    vels = [34, 32, 30, 28]
    for (chord, start), vel in zip(progression, vels):
        for p in chord:
            notes.append((p, start, 76.0, vel))
    return notes


def deep_lastbus(bars):
    """Sub: returns as warmth, not weight. Eb2 — felt through
    the floor, like a heater kicking on."""
    notes = []
    subs = [
        (39, 0.0, 160.0, 30),  # Eb2 — gentle, continuous
        (36, 180.0, 130.0, 25),  # C2  — settling
    ]
    for pitch, start, dur, vel in subs:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 5: Radiator Hum — 96 bars — Indoors, settling
# Wool + Deep only. No piano, no glass.
# ---------------------------------------------------------------------------


def wool_radiator(bars):
    """Pad: this IS the music now. Slow-breathing Eb drone with
    gentle movement between chord tones. Like listening to your
    own heartbeat through a pillow."""
    notes = []
    progression = [
        (PAD_Eb, 0.0),
        ([39, 44, 51], 96.0),  # Eb2 Ab2 Eb3 — subtle color shift
        (PAD_Cm, 192.0),
        (PAD_Eb, 288.0),
    ]
    vels = [28, 26, 24, 22]
    for (chord, start), vel in zip(progression, vels):
        for p in chord:
            notes.append((p, start, 92.0, vel))
    return notes


def deep_radiator(bars):
    """Sub: single Eb drone for the entire section.
    The frequency of contentment."""
    return [(39, 0.0, float(bars * 4 - 4), 25)]


# ---------------------------------------------------------------------------
# Section 6: 3 AM — 96 bars — Very sparse
# Wool + Deep. Pad thins to single notes. Sub barely audible.
# ---------------------------------------------------------------------------


def wool_3am(bars):
    """Pad: no longer chords. Single notes, widely spaced.
    Eb... then nothing... then Bb... then nothing.
    The spaces between notes are the music now."""
    notes = []
    tones = [
        (51, 0.0, 32.0, 20),  # Eb3 — 8 bars of tone
        (46, 64.0, 24.0, 16),  # Bb2 — 6 bars, starts bar 16
        (48, 128.0, 28.0, 14),  # C3  — 7 bars, starts bar 32
        (51, 208.0, 20.0, 12),  # Eb3 — 5 bars, starts bar 52
        (46, 288.0, 32.0, 10),  # Bb2 — 8 bars, starts bar 72
    ]
    for pitch, start, dur, vel in tones:
        notes.append((pitch, start, dur, vel))
    return notes


def deep_3am(bars):
    """Sub: barely audible. Two long tones, like a distant ship."""
    notes = []
    subs = [
        (27, 0.0, 160.0, 16),  # Eb1
        (27, 224.0, 140.0, 10),  # Eb1 — even quieter
    ]
    for pitch, start, dur, vel in subs:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Section 7: Gone — 100 bars — Dissolving into silence
# Wool only. One note every 16-24 bars.
# ---------------------------------------------------------------------------


def wool_gone(bars):
    """Pad: one note. Then silence. Then one note. Then more silence.
    Then one last note you're not sure you heard.
    Then nothing. You're asleep."""
    notes = []
    final = [
        (51, 0.0, 24.0, 10),  # Eb3 — still here
        (46, 96.0, 16.0, 8),  # Bb2 — are you still awake?
        (51, 208.0, 20.0, 6),  # Eb3 — barely
        (39, 336.0, 32.0, 3),  # Eb2 — you imagined this
    ]
    for pitch, start, dur, vel in final:
        notes.append((pitch, start, dur, vel))
    return notes


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

s = Set(
    name="Goodnight, Fog City",
    subtitle="45 minutes of winter SF ambient",
    bpm=BPM,
    tracks=[
        Track(
            "Dusk",
            sound="E-Piano MKI Mellow.adv",
            effects=["Warm Reverb Long.adv"],
            volume=0.60,
            pan=-0.10,
        ),
        Track(
            "Wool", sound="Drifting Ambient Pad.adv", effects=["Reverb"], volume=0.55
        ),
        Track(
            "Glass",
            sound="Ethereal Brushed Bells.adg",
            effects=["Delayed Hall Reverb.adg"],
            volume=0.35,
            pan=0.15,
        ),
        Track("Deep", sound="Basic Sub Sine.adg", volume=0.45),
    ],
)

sections = [
    Scene(
        "Fog Arrives",
        80,
        {"Dusk": dusk_fog, "Wool": wool_fog, "Glass": glass_fog, "Deep": deep_fog},
    ),
    Scene(
        "Streetlight Halos",
        80,
        {
            "Dusk": dusk_halos,
            "Wool": wool_halos,
            "Glass": glass_halos,
            "Deep": deep_halos,
        },
    ),
    Scene(
        "The Richmond",
        88,
        {"Dusk": dusk_richmond, "Wool": wool_richmond, "Glass": glass_richmond},
    ),
    Scene(
        "Last Bus",
        80,
        {"Dusk": dusk_lastbus, "Wool": wool_lastbus, "Deep": deep_lastbus},
    ),
    Scene("Radiator Hum", 96, {"Wool": wool_radiator, "Deep": deep_radiator}),
    Scene("3 AM", 96, {"Wool": wool_3am, "Deep": deep_3am}),
    Scene("Gone", 100, {"Wool": wool_gone}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    s.setup()

    s.session.param("Dusk", 1, "decay", 0.5)
    print("    Dusk/Reverb Decay -> 0.5 (warm fog)")

    auto = {
        1: [("Dusk", 1, "decay", 0.6, "Dusk/Reverb -> 0.6 (deeper fog)")],
        2: [
            ("Dusk", 1, "decay", 0.75, "Dusk/Reverb -> 0.75 (heavy fog)"),
            ("Wool", 1, "decay", 0.5, "Wool/Reverb -> 0.5 (bed deepens)"),
        ],
        3: [("Dusk", 1, "decay", 0.9, "Dusk/Reverb -> 0.9 (dissolving)")],
        4: [("Wool", 1, "decay", 0.65, "Wool/Reverb -> 0.65 (radiator warmth)")],
        5: [("Wool", 1, "decay", 0.8, "Wool/Reverb -> 0.8 (3am distance)")],
        6: [("Wool", 1, "decay", 0.95, "Wool/Reverb -> 0.95 (maximum fog)")],
    }

    for sec_idx, changes in auto.items():
        sections[sec_idx].on_enter = make_automation(changes)

    s.load_scenes(sections)
    s.build_arrangement(sections)
    s.play()
    print("\n  The fog swallows everything. Goodnight.")
    s.teardown()


if __name__ == "__main__":
    main()
