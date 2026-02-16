"""One-off: MUSIC FOR RINKS

The empty curling rink as a Brian Eno ambient piece. Not a game,
not a narrative — just the building breathing. 66 BPM. The
refrigeration hum, stones sliding in memory, your breath visible
in the cold.

Technique: Music for Airports-style overlapping loops of coprime
lengths (7, 11, 13, 17 beats). Each track is a tape loop — notes
placed at intervals that never quite sync, creating slowly shifting
patterns that are always new.

Revision: Major 7ths (Db–C) for coldness, split-register Ice
(sub drone + crystalline harmonic), minor 2nd Breath for icy
beating, curling stone thuds in the piano, continuous automation.
"""

from session import Set, Track, Scene, make_automation

BPM = 66

# Db major: Db Eb F Gb Ab Bb C
# Harmonic language: open 5ths, major 7ths (Eno's "1/1" interval),
# minor 2nds (icy beating), wide registral gaps (cold hard surfaces)


# ---------------------------------------------------------------------------
# Loop engine — Music for Airports technique
# ---------------------------------------------------------------------------


def _tile(loop, loop_len, total_beats, vel_drift=0):
    """Tile a note pattern across total_beats at loop_len intervals.

    Because loop lengths are coprime (7, 11, 13, 17), the voices
    never align the same way twice. Each cycle, velocity drifts
    by vel_drift (negative = fade out).
    """
    notes = []
    cycle = 0
    while True:
        offset = cycle * loop_len
        if offset >= total_beats:
            break
        v_adj = cycle * vel_drift
        for pitch, beat, dur, vel in loop:
            t = offset + beat
            if t >= total_beats:
                continue
            d = min(dur, total_beats - t)
            v = max(1, min(127, vel + v_adj))
            notes.append((pitch, float(t), d, int(v)))
        cycle += 1
    return notes


# ---------------------------------------------------------------------------
# Stone (piano) — 7-beat loop
# Evolves across sections: pure 5th → major 7th → only the 7th
# Curling stone thuds in lower register during Stones/House sections
# ---------------------------------------------------------------------------

_STONE_A = [  # warm arrival (The Door, Sheet): pure 5th
    (68, 0.0, 0.8, 32),  # Ab4
    (73, 4.0, 0.8, 28),  # Db5
]
_STONE_B = [  # cold creeps in (Stones): major 7th + curling stone thud
    (68, 0.0, 0.8, 32),  # Ab4
    (72, 2.5, 0.8, 28),  # C5 — major 7th cracks the warmth
    (73, 4.0, 0.8, 26),  # Db5
    (49, 5.5, 0.15, 52),  # Db3 — curling stone thud
]
_STONE_C = [  # only the 7th (The House): coldest, no resolution
    (68, 0.0, 0.8, 30),  # Ab4
    (72, 3.0, 0.8, 26),  # C5 — the 7th, unresolved
    (44, 5.0, 0.15, 55),  # Ab2 — curling stone thud, lower
]


def stone(bars):
    """Ab4–Db5 pure 5th. Warm arrival."""
    return _tile(_STONE_A, 7, bars * 4)


def stone_cold(bars):
    """Ab4–C5–Db5: major 7th appears. Curling stone thud below."""
    return _tile(_STONE_B, 7, bars * 4)


def stone_house(bars):
    """Ab4–C5 only: the 7th without resolution. Coldest."""
    return _tile(_STONE_C, 7, bars * 4)


def stone_fade(bars):
    """Pure 5th returns, each cycle quieter — warmth as you leave."""
    return _tile(_STONE_A, 7, bars * 4, vel_drift=-3)


# ---------------------------------------------------------------------------
# Ice (strings) — 11-beat loop
# Split-register: sub drone + crystalline harmonic, empty middle = the rink
# Harmonic rises across sections: Bb5 → C6 → Eb6
# ---------------------------------------------------------------------------

_ICE_A = [  # Sheet: huge registral gap
    (37, 0.0, 9.0, 22),  # Db2 — low drone
    (82, 5.0, 4.0, 16),  # Bb5 — high harmonic
]
_ICE_B = [  # Stones: harmonic rises
    (37, 0.0, 9.0, 22),  # Db2 — drone holds
    (84, 5.0, 4.0, 15),  # C6 — harmonic rises
]
_ICE_C = [  # The House: widest gap, maximum cold
    (32, 0.0, 9.0, 20),  # Ab1 — sub drone
    (87, 5.0, 4.0, 14),  # Eb6 — crystalline harmonic
]


def ice(bars):
    """Db2 drone + Bb5 harmonic. The empty middle is the rink."""
    return _tile(_ICE_A, 11, bars * 4)


def ice_rise(bars):
    """Harmonic rises to C6. The ice gets more crystalline."""
    return _tile(_ICE_B, 11, bars * 4)


def ice_wide(bars):
    """Ab1 sub + Eb6 harmonic. Widest gap — maximum cold."""
    return _tile(_ICE_C, 11, bars * 4)


# ---------------------------------------------------------------------------
# Breath (synth) — 13-beat loop
# Minor 2nds create icy beating/friction at low velocity
# ---------------------------------------------------------------------------

_BREATH_A = [
    (56, 0.0, 3.0, 8),  # Ab3 — minor 2nd beating, low and gentle
    (55, 7.0, 2.5, 8),  # G3
]
_BREATH_B = [
    (49, 0.0, 3.0, 8),  # Db3 — shifted minor 2nd
    (48, 7.0, 2.5, 8),  # C3
]


def breath(bars):
    """Db6+C6 minor 2nd — icy beating, subliminal."""
    return _tile(_BREATH_A, 13, bars * 4)


def breath_drift(bars):
    """Ab5+G5 minor 2nd — the breath shifts."""
    return _tile(_BREATH_B, 13, bars * 4)


# ---------------------------------------------------------------------------
# Hum (bass) — 17-beat loop
# Db2, vel 30, fills full loop. Static and mechanical.
# At low velocity the Reese detuning becomes imperceptible — pure drone.
# ---------------------------------------------------------------------------

_HUM = [
    (37, 0.0, 16.5, 30),  # Db2 — quieter, fills full loop
]


def hum(bars):
    """Db2 drone vel 30. Static, mechanical — felt not heard."""
    return _tile(_HUM, 17, bars * 4)


def hum_fade(bars):
    """The last thing you hear as you leave."""
    return _tile(_HUM, 17, bars * 4, vel_drift=-5)


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

s = Set(
    name="Music for Rinks",
    subtitle="After Brian Eno",
    bpm=BPM,
    tracks=[
        Track(
            "Stone",
            sound="Prepared Piano Ambient.adv",
            effects=["Reverb"],
            volume=0.75,
            pan=-0.10,
        ),
        Track("Ice", sound="Vintage Strings.adv", effects=["Reverb"], volume=0.50),
        Track(
            "Breath",
            sound="Saw Elegant Thick Keys.adv",
            effects=["Chorus-Ensemble"],
            volume=0.35,
            pan=0.15,
        ),
        Track("Hum", sound="Reese Classic.adg", effects=["Reverb"], volume=0.60),
    ],
)

sections = [
    Scene("The Door", 8, {"Stone": stone}),
    Scene("Sheet", 8, {"Stone": stone, "Ice": ice}),
    Scene(
        "Stones",
        12,
        {"Stone": stone_cold, "Ice": ice_rise, "Breath": breath, "Hum": hum},
    ),
    Scene(
        "The House",
        16,
        {"Stone": stone_house, "Ice": ice_wide, "Breath": breath_drift, "Hum": hum},
    ),
    Scene("Leaving", 8, {"Stone": stone_fade, "Hum": hum_fade}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    s.setup()

    # Initial reverb: moderate, the space is still revealing itself
    s.session.param("Stone", 1, "decay", 0.5)
    print("    Stone/Reverb Decay -> 0.5 (the room appears)")

    # Per-section automation: the space slowly grows
    auto = {
        1: [  # Sheet
            ("Stone", 1, "decay", 0.55, "Stone/Reverb -> 0.55 (the room grows)"),
            ("Ice", 1, "decay", 0.6, "Ice/Reverb -> 0.6 (the surface resonates)"),
        ],
        2: [  # Stones
            ("Stone", 1, "decay", 0.65, "Stone/Reverb -> 0.65 (tails lengthen)"),
            ("Ice", 1, "decay", 0.7, "Ice/Reverb -> 0.7 (the space opens)"),
            ("Breath", 1, "Amount", 0.4, "Breath/Chorus -> 0.4 (frost appears)"),
        ],
        3: [  # The House
            ("Stone", 1, "decay", 0.85, "Stone/Reverb -> 0.85 (infinite room)"),
            ("Ice", 1, "decay", 0.85, "Ice/Reverb -> 0.85 (the space is vast)"),
            ("Breath", 1, "Amount", 0.55, "Breath/Chorus -> 0.55 (frost thickens)"),
        ],
        4: [  # Leaving
            ("Stone", 1, "decay", 0.95, "Stone/Reverb -> 0.95 (swallowed by reverb)"),
        ],
    }

    for sec_idx, changes in auto.items():
        sections[sec_idx].on_enter = make_automation(changes)

    s.load_scenes(sections)
    s.build_arrangement(sections)
    s.play()
    print("\n  The building breathes.")
    s.teardown()


if __name__ == "__main__":
    main()
