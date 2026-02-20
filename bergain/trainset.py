"""Training briefs for GEPA optimization.

Diverse set of creative briefs covering genre, tempo, mood, and key space.
Split into TRAIN (15) and VAL (5) sets.
"""

import dspy

# ---------------------------------------------------------------------------
# Full brief set — 20 diverse briefs
# ---------------------------------------------------------------------------

BRIEFS = [
    # --- TRAIN (indices 0-14) ---
    # Techno variants
    dspy.Example(brief="Dark Berlin techno in F minor, 130 BPM").with_inputs("brief"),
    dspy.Example(
        brief="Minimal techno, 128 BPM, hypnotic and repetitive with subtle evolution"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Industrial techno in A minor, 140 BPM, aggressive and mechanical"
    ).with_inputs("brief"),
    # House variants
    dspy.Example(
        brief="Deep house in D minor, 122 BPM, warm pads and rolling bass"
    ).with_inputs("brief"),
    dspy.Example(
        brief="High-energy acid house, 138 BPM, 303 bass and driving drums"
    ).with_inputs("brief"),
    # Ambient / downtempo
    dspy.Example(
        brief="Ambient downtempo, 90 BPM, lush pads and gentle bass in Eb"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Dreamy ambient in G major, 85 BPM, no drums, just evolving textures"
    ).with_inputs("brief"),
    # Dub / experimental
    dspy.Example(
        brief="Dub techno in C minor, 118 BPM, reverb-drenched chords and sparse percussion"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Experimental electronic in Bb minor, 110 BPM, glitchy and unpredictable"
    ).with_inputs("brief"),
    # Trance / euphoric
    dspy.Example(
        brief="Progressive trance in A minor, 138 BPM, euphoric builds and emotional breakdowns"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Melodic techno in E minor, 126 BPM, arpeggiated synths and driving bassline"
    ).with_inputs("brief"),
    # Mood-focused
    dspy.Example(
        brief="Melancholic electronica in Ab minor, 105 BPM, sparse and haunting"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Hypnotic late-night set in F# minor, 124 BPM, tribal percussion and deep bass"
    ).with_inputs("brief"),
    # Specific instrumentation
    dspy.Example(
        brief="Percussion-heavy groove in G minor, 132 BPM, minimal bass, complex drum patterns"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Pad-driven soundscape in Db major, 95 BPM, wide stereo, no bass"
    ).with_inputs("brief"),
    # --- VAL (indices 15-19) ---
    dspy.Example(
        brief="Dark minimal techno in E minor, 134 BPM, stripped back and intense"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Warm deep house in C major, 120 BPM, soulful chords and bouncy bass"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Atmospheric breaks in D minor, 130 BPM, cinematic pads with breakbeat drums"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Acid techno in G minor, 142 BPM, relentless 303 and pounding kicks"
    ).with_inputs("brief"),
    dspy.Example(
        brief="Chill electronic in F major, 100 BPM, gentle arpeggios and soft textures"
    ).with_inputs("brief"),
]

TRAIN = BRIEFS[:15]
VAL = BRIEFS[15:]
