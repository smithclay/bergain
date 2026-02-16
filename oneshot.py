"""One-off: THE CURLING SCANDAL — Boards of Canada vs ABBA.

D minor as common ground. BoC owns the fog; ABBA owns the light.
The rink is where they meet. 90 BPM. The ice remembers everything.
"""

from main import (
    Set,
    Track,
    Scene,
    get_device_params,
    find_param_index,
    set_device_param,
)

BPM = 90

# D minor pentatonic (BoC): D F G A C
# D major (ABBA):           D E F# G A B C#
# Common ground: D, A, G
# The tension: F vs F#, C vs C#

# BoC: pentatonic, ascending, open 4ths/5ths
BOC_MOTIF = [62, 67, 69, 72, 74]  # D4 G4 A4 C5 D5

# ABBA: major arpeggio, bright, rising
ABBA_ARP = [74, 78, 81, 86]  # D5 F#5 A5 D6


# ---------------------------------------------------------------------------
# Section 1: The Rink — O Canada through BoC fog (8 bars)
# ---------------------------------------------------------------------------


def ice_rink(bars):
    """O Canada played through tape hiss and memory. BoC style:
    each note alone, swimming in reverb, too slow to be real.
    Like hearing it through community center walls in 1979."""
    notes = []
    anthem = [
        (66, 0.0, 5.0, 65),  # F#4 "O" — major 3rd, fading into haze
        (69, 8.0, 3.0, 58),  # A4 "Ca-" — drifting in
        (69, 12.0, 2.5, 55),  # A4 "-na-" — echo of itself
        (74, 16.0, 8.0, 72),  # D5 "-DA!" — held, wistful not triumphant
        (71, 26.0, 2.5, 42),  # B4 "Our" — barely audible
        (69, 29.0, 3.0, 38),  # A4 "home" — dissolving
    ]
    for pitch, start, dur, vel in anthem:
        notes.extend([pitch, start, dur, vel, 0])
    # BoC ghost notes: a 5th/4th below, very quiet, like tape bleed
    ghosts = [
        (59, 1.0, 3.0, 20),  # B3 — harmonic shadow of F#
        (62, 9.0, 2.0, 18),  # D4 — shadow of A
        (67, 17.0, 4.0, 22),  # G4 — 4th below D5, very BoC
    ]
    for pitch, start, dur, vel in ghosts:
        notes.extend([pitch, start, dur, vel, 0])
    return notes


# ---------------------------------------------------------------------------
# Section 2: Suspicion — BoC territory (8 bars)
# ---------------------------------------------------------------------------


def ice_suspicion(bars):
    """Piano: pentatonic fragments, BoC style. Notes drift in like
    half-remembered melodies from a childhood you're not sure was yours."""
    notes = []
    fragments = [
        (62, 0.0, 2.0, 45),  # D4
        (67, 4.5, 1.5, 38),  # G4 — the open 4th, signature BoC
        (69, 8.0, 3.0, 42),  # A4
        (72, 14.0, 1.0, 30),  # C5 — ghost
        (62, 17.0, 1.5, 40),  # D4 — return
        (69, 20.0, 2.0, 35),  # A4
        (67, 24.0, 3.0, 32),  # G4 — settling
        (62, 28.0, 4.0, 28),  # D4 — drone tail
    ]
    for pitch, start, dur, vel in fragments:
        notes.extend([pitch, start, dur, vel, 0])
    # Low octave anchor
    notes.extend([50, 0.0, 4.0, 25, 0])  # D3
    notes.extend([50, 16.0, 4.0, 22, 0])  # D3
    return notes


def shadow_suspicion(bars):
    """Pad: BoC warmth. Slow-moving open-voiced chords, stacked 4ths/5ths.
    Think 'Roygbiv' — warm but melancholy."""
    chords = [
        ([38, 45, 50], 0.0),  # D2 A2 D3 — open 5th drone
        ([36, 43, 48], 8.0),  # C2 G2 C3 — down a step
        ([34, 41, 46], 16.0),  # Bb1 F2 Bb2 — deeper
        ([33, 40, 45], 24.0),  # A1 E2 A2 — bottom
    ]
    notes = []
    for pitches, start in chords:
        for p in pitches:
            notes.extend([p, start, 7.5, 32, 0])
    return notes


def shadow_evidence(bars):
    """Pad: transitional. BoC open voicings pivoting toward ABBA's dominant.
    F major appears as the hinge — lives in both worlds. A closes the section,
    the dominant pulling toward D major like a door cracking open."""
    chords = [
        ([38, 45, 50], 0.0),  # D2 A2 D3 — still BoC
        ([36, 43, 48], 8.0),  # C2 G2 C3 — still BoC
        ([41, 48, 53], 16.0),  # F2 C3 F3 — the hinge (III — shared territory)
        ([33, 40, 45], 24.0),  # A1 E2 A2 — dominant, pulling toward light
    ]
    notes = []
    vels = [30, 32, 35, 40]
    for (pitches, start), vel in zip(chords, vels):
        for p in pitches:
            notes.extend([p, start, 7.5, vel, 0])
    return notes


# ---------------------------------------------------------------------------
# Section 3: Evidence — ABBA light breaks through (8 bars)
# ---------------------------------------------------------------------------


def ice_evidence(bars):
    """Piano holds the BoC ground. Sustained roots, resisting the brightness."""
    roots = [(62, 0.0), (58, 8.0), (60, 16.0), (57, 24.0)]
    notes = []
    for pitch, start in roots:
        notes.extend([pitch, start, 3.5, 38, 0])
        notes.extend([pitch - 12, start, 3.5, 28, 0])
    return notes


def glass_evidence(bars):
    """Saw: ABBA DNA emerging. First D major arpeggios —
    F# instead of F. Swedish sun breaking through Canadian fog.
    'Gimme Gimme Gimme' rhythm creeps in at the end."""
    notes = []
    # Pass 1: D major arpeggio, gentle, testing the air
    first = [
        (74, 0.0, 1.5, 55),  # D5
        (78, 2.5, 1.0, 50),  # F#5 — ABBA's calling card
        (81, 4.5, 1.5, 58),  # A5
        (86, 7.0, 2.0, 62),  # D6 — arrival
    ]
    for pitch, start, dur, vel in first:
        notes.extend([pitch, start, dur, vel, 0])

    # Pass 2: tighter, more confident
    second = [
        (74, 12.0, 0.75, 62),  # D5
        (78, 13.0, 0.75, 60),  # F#5
        (81, 14.0, 0.75, 65),  # A5
        (86, 15.0, 1.5, 70),  # D6
        (81, 17.0, 0.75, 58),  # A5 — descending
        (78, 18.0, 0.75, 55),  # F#5
        (74, 19.0, 2.0, 52),  # D5
    ]
    for pitch, start, dur, vel in second:
        notes.extend([pitch, start, dur, vel, 0])

    # Pass 3: "Gimme Gimme Gimme" rhythm hint
    gimme = [
        (74, 22.0, 0.25, 68),  # D5 — staccato repeats
        (74, 22.5, 0.25, 65),  # D5
        (74, 23.0, 0.25, 70),  # D5
        (77, 23.5, 0.5, 60),  # F5 — BoC fighting back with the minor 3rd
        (78, 24.5, 0.5, 72),  # F#5 — no, ABBA takes it
        (81, 25.5, 0.5, 68),  # A5
        (86, 27.0, 3.0, 75),  # D6 — held, triumphant
    ]
    for pitch, start, dur, vel in gimme:
        notes.extend([pitch, start, dur, vel, 0])
    return notes


# ---------------------------------------------------------------------------
# Section 4: The Fight Dance — 24 bars — CHARIOTS OF FIRE
#
# BoC vs ABBA, but through the Vangelis lens: that slow-motion
# dotted rhythm, majestic layered build, running on the beach
# of a frozen Canadian rink.
#
# BUILD (1-8):  Chariots rhythm emerging from BoC fog
# PEAK (9-16):  Full Chariots majesty — ABBA brightness, all layers
# BREAKDOWN+RETURN (17-24): BoC reclaims, then ABBA surges — the finish line
# ---------------------------------------------------------------------------

FIGHT_ROOTS_MINOR = [38, 34, 36, 33]  # Dm Bb C Am — BoC territory
FIGHT_ROOTS_MAJOR = [38, 35, 40, 33]  # D  B  E  A — ABBA I-vi-ii-V roots

# Chariots rhythm: short-short-short-LONG (da-da-da-DAAA)
# At 90 BPM that's roughly: 0.5 + 0.5 + 0.5 + 2.5 beats per phrase


def ice_fight(bars):
    """Piano: Chariots of Fire dotted rhythm.
    Build: BoC open chords, Chariots rhythm barely there.
    Peak: ABBA I-vi-ii-V, root leads da-da-da, full chord blooms on DAAA.
    Return: fusion — major Chariots at BoC distance."""
    notes = []
    third = bars // 3

    # BoC voicings: open, hollow (Chariots through fog)
    boc_chords = [
        [62, 67, 74],  # D4 G4 D5 — open 5ths
        [58, 65, 70],  # Bb3 F4 Bb4
        [60, 67, 72],  # C4 G4 C5
        [57, 64, 69],  # A3 E4 A4
    ]
    # ABBA voicings: I-vi-ii-V — the ii-V gives it that ABBA sophistication
    abba_chords = [
        [62, 66, 69],  # D4 F#4 A4 — D major (I)
        [59, 62, 66],  # B3 D4 F#4 — Bm (vi)
        [64, 67, 71],  # E4 G4 B4 — Em (ii)
        [57, 61, 64],  # A3 C#4 E4 — A major (V)
    ]

    # Phase 1: BoC Chariots — dotted rhythm, but sparse and foggy
    for bar in range(third):
        chord = boc_chords[bar % 4]
        base = float(bar * 4)
        vel = 32 + bar * 3
        for p in chord:
            notes.extend([p, base, 0.3, vel, 0])
            notes.extend([p, base + 0.5, 0.3, vel - 5, 0])
            notes.extend([p, base + 1.0, 0.3, vel - 3, 0])
            notes.extend([p, base + 1.5, 2.0, vel + 5, 0])

    # Phase 2: ABBA Chariots — root leads the da-da-da, full chord blooms on DAAA
    for bar in range(third):
        chord = abba_chords[bar % 4]
        base = float((bar + third) * 4)
        vel = 68 + bar * 3
        root = chord[0]
        for rep in range(2):
            off = rep * 2.0
            # da-da-da: root note alone, articulated
            notes.extend([root, base + off, 0.2, vel - 5, 0])
            notes.extend([root, base + off + 0.5, 0.2, vel - 3, 0])
            notes.extend([root, base + off + 1.0, 0.2, vel, 0])
            # DAAA: full chord blooms
            for p in chord:
                notes.extend([p, base + off + 1.5, 0.5, vel + 14, 0])

    # Phase 3: Fusion — BoC chords reclaim, then ABBA builds to the finish
    for bar in range(third):
        base = float((bar + third * 2) * 4)
        if bar < 4:
            chord = boc_chords[bar % 4]
            vel = 50
            for p in chord:
                notes.extend([p, base, 0.3, vel - 5, 0])
                notes.extend([p, base + 0.5, 0.3, vel - 8, 0])
                notes.extend([p, base + 1.0, 0.3, vel - 5, 0])
                notes.extend([p, base + 1.5, 2.5, vel, 0])
        else:
            # ABBA finish line: root drives, chord explodes
            chord = abba_chords[bar % 4]
            vel = 72 + (bar - 4) * 5
            root = chord[0]
            for rep in range(2):
                off = rep * 2.0
                notes.extend([root, base + off, 0.2, vel, 0])
                notes.extend([root, base + off + 0.5, 0.2, vel, 0])
                notes.extend([root, base + off + 1.0, 0.2, vel + 5, 0])
                for p in chord:
                    notes.extend([p, base + off + 1.5, 0.5, vel + 16, 0])
    return notes


def shadow_fight(bars):
    """Pad: Chariots sustain layer.
    Build: BoC drones swelling slowly. Peak: bright ABBA I-vi-ii-V pumping
    on the Chariots rhythm. Return: sustained glow."""
    notes = []
    third = bars // 3

    boc_pads = [
        [38, 45, 50],  # D2 A2 D3 — open drone
        [34, 41, 46],  # Bb1 F2 Bb2
        [36, 43, 48],  # C2 G2 C3
        [33, 40, 45],  # A1 E2 A2
    ]
    abba_pads = [
        [50, 54, 57],  # D3 F#3 A3 — D major (I)
        [47, 50, 54],  # B2 D3 F#3 — Bm (vi)
        [52, 55, 59],  # E3 G3 B3 — Em (ii)
        [45, 49, 52],  # A2 C#3 E3 — A major (V)
    ]

    for bar in range(bars):
        base = float(bar * 4)
        phase = 0 if bar < third else (1 if bar < third * 2 else 2)

        if phase == 0:
            chord = boc_pads[bar % 4]
            vel = 25 + bar * 2
            for p in chord:
                notes.extend([p, base, 3.8, vel, 0])
        elif phase == 1:
            chord = abba_pads[bar % 4]
            vel = 42 + bar * 3
            for p in chord:
                notes.extend([p, base, 0.4, vel, 0])
                notes.extend([p, base + 0.5, 0.4, vel, 0])
                notes.extend([p, base + 1.0, 0.4, vel, 0])
                notes.extend([p, base + 1.5, 2.3, vel + 8, 0])
        else:
            chord = abba_pads[bar % 4]
            vel = 35 + bar
            for p in chord:
                notes.extend([p, base, 3.8, vel, 0])
    return notes


def glass_fight(bars):
    """Saw: Chariots melody line — the real Vangelis.
    Phase 1: BoC pentatonic Chariots — ascending through fog.
    Phase 2: ABBA major Chariots — the beach run, full sunlight, hits 100.
    Phase 3: BoC reclaims (fading), ABBA surges for the finish."""
    notes = []
    third = bars // 3

    boc_scale = [74, 77, 79, 81, 84]  # D5 F5 G5 A5 C6 — pentatonic
    abba_scale = [74, 76, 78, 81, 86]  # D5 E5 F#5 A5 D6 — major

    # Phase 1: BoC — Chariots melody through fog
    for bar in range(third):
        base = float(bar * 4)
        vel = 40 + bar * 3
        phrase_start = (bar // 2) % 3
        if bar % 2 == 0:
            for i in range(4):
                pitch = boc_scale[(phrase_start + i) % 5]
                if i < 3:
                    notes.extend([pitch, base + i * 0.5, 0.4, vel - i * 2, 0])
                else:
                    notes.extend([pitch, base + 1.5, 2.0, vel + 5, 0])
        else:
            for i in range(3):
                pitch = boc_scale[(phrase_start + 3 - i) % 5]
                notes.extend([pitch, base + i * 0.75, 0.6, vel - 10 - i * 3, 0])

    # Phase 2: ABBA — full Chariots, major scale, hits hard
    for bar in range(third):
        base = float((bar + third) * 4)
        vel = 65 + bar * 4  # 65 → 93 across 8 bars
        for rep in range(2):
            off = rep * 2.0
            for i in range(4):
                pitch = abba_scale[i + (bar % 2)]
                if i < 3:
                    notes.extend([pitch, base + off + i * 0.33, 0.25, vel + i * 3, 0])
                else:
                    notes.extend([pitch, base + off + 1.0, 0.8, vel + 14, 0])
        if bar >= 6:
            notes.extend([86, base + 3.5, 0.5, min(vel + 22, 120), 0])

    # Phase 3: trading, then ABBA surge
    for bar in range(third):
        base = float((bar + third * 2) * 4)
        if bar < 4:
            # BoC reclaims: getting sparser, losing
            vel = 48 - bar * 3
            pitch = boc_scale[bar % 5]
            notes.extend([pitch, base, 0.4, vel, 0])
            notes.extend([boc_scale[(bar + 1) % 5], base + 0.5, 0.4, vel - 5, 0])
            notes.extend([boc_scale[(bar + 2) % 5], base + 1.0, 0.4, vel - 3, 0])
            notes.extend([boc_scale[(bar + 3) % 5], base + 1.5, 2.5, vel + 5, 0])
        else:
            # ABBA surge: finish line
            vel = 78 + (bar - 4) * 6  # 78 → 102
            for rep in range(2):
                off = rep * 2.0
                for i in range(4):
                    pitch = abba_scale[i]
                    if i < 3:
                        notes.extend(
                            [pitch, base + off + i * 0.33, 0.25, vel + i * 3, 0]
                        )
                    else:
                        notes.extend([pitch, base + off + 1.0, 0.8, vel + 14, 0])
            if bar == third - 1:
                notes.extend([86, base + 3.0, 1.0, min(vel + 20, 120), 0])
    return notes


def weight_fight(bars):
    """Bass: Chariots heartbeat underneath everything.
    Build: BoC sub-pulse, Chariots dotted rhythm on the root.
    Peak: ABBA driving 8ths, DAAA accent hits 100.
    Return: the heartbeat alone, then full gallop."""
    notes = []
    third = bars // 3
    for bar in range(bars):
        base = float(bar * 4)
        phase = 0 if bar < third else (1 if bar < third * 2 else 2)

        if phase == 0:
            # BoC: minor roots
            root = FIGHT_ROOTS_MINOR[bar % 4]
            vel = 55 + bar * 3
            notes.extend([root, base, 0.4, vel, 0])
            notes.extend([root, base + 0.5, 0.4, vel - 5, 0])
            notes.extend([root, base + 1.0, 0.4, vel - 3, 0])
            notes.extend([root + 12, base + 1.5, 2.0, vel + 8, 0])
        elif phase == 1:
            # ABBA: major roots (D-B-E-A matches I-vi-ii-V)
            root = FIGHT_ROOTS_MAJOR[bar % 4]
            for eighth in range(8):
                t = base + eighth * 0.5
                if eighth in [0, 1, 2]:
                    v = 78 + eighth * 2
                elif eighth == 3:
                    v = 100
                else:
                    v = 65
                p = root + 12 if eighth == 3 else root
                notes.extend([p, t, 0.4, v, 0])
        else:
            if bar - third * 2 < 4:
                # BoC reclaim: minor roots
                root = FIGHT_ROOTS_MINOR[bar % 4]
                notes.extend([root, base, 0.5, 68, 0])
                notes.extend([root + 12, base + 1.5, 2.0, 60, 0])
            else:
                # ABBA surge: major roots
                root = FIGHT_ROOTS_MAJOR[bar % 4]
                vel = 82 + (bar - third * 2 - 4) * 4
                notes.extend([root, base, 0.4, vel, 0])
                notes.extend([root, base + 0.5, 0.4, vel - 5, 0])
                notes.extend([root, base + 1.0, 0.4, vel, 0])
                notes.extend([root + 12, base + 1.5, 0.5, vel + 12, 0])
                notes.extend([root, base + 2.0, 0.4, vel, 0])
                notes.extend([root, base + 2.5, 0.4, vel - 5, 0])
                notes.extend([root, base + 3.0, 0.4, vel, 0])
                notes.extend([root + 12, base + 3.5, 0.5, vel + 12, 0])
    return notes


# ---------------------------------------------------------------------------
# Section 5: The Ice Remembers — détente (8 bars)
# D major melody at BoC pace. They were never enemies.
# ---------------------------------------------------------------------------


def ice_remembers(bars):
    """D major arpeggio at BoC tempo. The scandal dissolves.
    What's left is just music — warm, slow, Swedish-Canadian peace."""
    notes = []
    resolution = [
        (62, 0.0, 4.0, 42),  # D4 — home
        (66, 5.0, 3.0, 38),  # F#4 — ABBA's gift, accepted
        (69, 10.0, 3.0, 35),  # A4 — shared ground
        (74, 15.0, 5.0, 40),  # D5 — resolution
        (66, 22.0, 3.0, 30),  # F#4 — one last F#, fading
        (62, 27.0, 5.0, 25),  # D4 — home, barely there
    ]
    for pitch, start, dur, vel in resolution:
        notes.extend([pitch, start, dur, vel, 0])
    return notes


def weight_remembers(bars):
    """Low D drone. The ice is still."""
    return [38, 0.0, float(bars * 4 - 1), 55, 0]


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

s = Set(
    name="The Curling Scandal",
    subtitle="Boards of Canada vs ABBA",
    bpm=BPM,
    tracks=[
        Track(
            "Ice",
            sound="Prepared Piano Ambient.adv",
            effects=["Reverb"],
            volume=0.75,
            pan=-0.10,
        ),
        Track("Shadow", sound="Vintage Strings.adv", effects=["Reverb"], volume=0.50),
        Track(
            "Glass",
            sound="Saw Elegant Thick Keys.adv",
            effects=["Chorus-Ensemble"],
            volume=0.62,
            pan=0.18,
        ),
        Track("Weight", sound="Reese Classic.adg", effects=["Saturator"], volume=0.70),
    ],
)

sections = [
    Scene("The Rink", 8, {"Ice": ice_rink}),
    Scene("Suspicion", 8, {"Ice": ice_suspicion, "Shadow": shadow_suspicion}),
    Scene(
        "Evidence",
        8,
        {"Ice": ice_evidence, "Shadow": shadow_evidence, "Glass": glass_evidence},
    ),
    Scene(
        "The Fight Dance",
        24,
        {
            "Ice": ice_fight,
            "Shadow": shadow_fight,
            "Glass": glass_fight,
            "Weight": weight_fight,
        },
    ),
    Scene("The Ice Remembers", 8, {"Ice": ice_remembers, "Weight": weight_remembers}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    s.setup()

    # Discover device params
    ice_reverb_params = get_device_params(s.api, 0, 1)
    shadow_reverb_params = get_device_params(s.api, 1, 1)
    glass_chorus_params = get_device_params(s.api, 2, 1)
    weight_sat_params = get_device_params(s.api, 3, 1)

    ice_decay = find_param_index(ice_reverb_params, "decay")
    shadow_decay = find_param_index(shadow_reverb_params, "decay")
    glass_drywet = find_param_index(glass_chorus_params, "dry/wet") or find_param_index(
        glass_chorus_params, "amount"
    )
    weight_drive = find_param_index(weight_sat_params, "drive")

    # Initial settings
    if ice_decay is not None:
        set_device_param(s.api, 0, 1, ice_decay, 0.6)
        print("    Ice/Reverb Decay -> 0.6 (BoC fog)")

    # Per-section param automation via on_enter
    def make_auto(changes):
        def on_enter(api):
            for track, device, idx, val, desc in changes:
                set_device_param(api, track, device, idx, val)
                print(f"      [{desc}]")

        return on_enter

    auto = {}
    if shadow_decay is not None:
        auto.setdefault(1, []).append(
            (1, 1, shadow_decay, 0.5, "Shadow/Reverb Decay -> 0.5 (BoC depth)")
        )
    if glass_drywet is not None:
        auto.setdefault(2, []).append(
            (2, 1, glass_drywet, 0.6, "Glass/Chorus -> 0.6 (70s shimmer)")
        )
    if ice_decay is not None:
        auto.setdefault(3, []).append(
            (0, 1, ice_decay, 0.3, "Ice/Reverb Decay -> 0.3 (tighter for fight)")
        )
    if weight_drive is not None:
        auto.setdefault(3, []).append(
            (3, 1, weight_drive, 0.85, "Weight/Saturator Drive -> 0.85 (ABBA energy)")
        )
    if ice_decay is not None:
        auto.setdefault(4, []).append(
            (0, 1, ice_decay, 0.9, "Ice/Reverb Decay -> 0.9 (back to fog)")
        )
    if weight_drive is not None:
        auto.setdefault(4, []).append(
            (3, 1, weight_drive, 0.15, "Weight/Saturator Drive -> 0.15 (peace)")
        )

    for sec_idx, changes in auto.items():
        sections[sec_idx].on_enter = make_auto(changes)

    s.load_scenes(sections)
    s.build_arrangement(sections)
    s.play()
    print("\n  The ice remembers everything.")
    s.teardown()


if __name__ == "__main__":
    main()
