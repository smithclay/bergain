"""
Builds a declarative arrangement artifact from recipe rules.
No audio loading — purely structural.
"""

import random

from bergain.picker import SamplePicker


def generate_arrangement(sample_dir: str, seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)

    picker = SamplePicker(sample_dir)

    # Pick palette — one sample per role, locked for the whole track
    palette = {
        "kick": picker.pick(category="kick")["path"],
        "hihat": picker.pick(category="hihat")["path"],
        "clap": picker.pick(category="clap")["path"],
        "bassline": picker.pick(category="bassline", is_loop=True)["path"],
        "drum_loop": picker.pick(category="drum_loop", is_loop=True)["path"],
        "synth": picker.pick(category="synth")["path"],
        "texture": picker.pick(category="texture")["path"],
        "fx": picker.pick(category="fx")["path"],
        "perc": picker.pick(category="perc")["path"],
    }

    sections = [
        {
            "name": "intro",
            "bars": 8,
            "layers": [
                {"role": "kick", "type": "oneshot", "beats": [0], "gain": 1.0},
                {"role": "texture", "type": "loop", "gain": 0.3},
            ],
            "fade_in": 4.0,
            "fade_out": None,
        },
        {
            "name": "buildup",
            "bars": 8,
            "layers": [
                {"role": "kick", "type": "oneshot", "beats": [0, 2], "gain": 1.0},
                {"role": "hihat", "type": "oneshot", "beats": [1, 3], "gain": 0.6},
                {"role": "texture", "type": "loop", "gain": 0.25},
                {"role": "perc", "type": "oneshot", "beats": [2], "gain": 0.4},
            ],
            "fade_in": None,
            "fade_out": None,
        },
        {
            "name": "drop_a",
            "bars": 16,
            "layers": [
                {"role": "kick", "type": "oneshot", "beats": [0, 1, 2, 3], "gain": 1.0},
                {"role": "hihat", "type": "oneshot", "beats": [0, 1, 2, 3], "gain": 0.5},
                {"role": "clap", "type": "oneshot", "beats": [1, 3], "gain": 0.7},
                {"role": "bassline", "type": "loop", "gain": 0.7},
                {"role": "drum_loop", "type": "loop", "gain": 0.4},
                {"role": "perc", "type": "oneshot", "beats": [1], "gain": 0.35},
            ],
            "fade_in": None,
            "fade_out": None,
        },
        {
            "name": "breakdown",
            "bars": 8,
            "layers": [
                {"role": "texture", "type": "loop", "gain": 0.4},
                {"role": "synth", "type": "loop", "gain": 0.5},
                {"role": "fx", "type": "oneshot", "beats": [0], "gain": 0.6, "start_bar": 4, "end_bar": 5},
            ],
            "fade_in": None,
            "fade_out": None,
        },
        {
            "name": "drop_b",
            "bars": 16,
            "layers": [
                {"role": "kick", "type": "oneshot", "beats": [0, 1, 2, 3], "gain": 1.0},
                {"role": "hihat", "type": "oneshot", "beats": [0, 1, 2, 3], "gain": 0.5},
                {"role": "clap", "type": "oneshot", "beats": [1, 3], "gain": 0.7},
                {"role": "bassline", "type": "loop", "gain": 0.7},
                {"role": "drum_loop", "type": "loop", "gain": 0.45},
                {"role": "perc", "type": "oneshot", "beats": [1, 3], "gain": 0.35},
                {"role": "synth", "type": "loop", "gain": 0.3},
            ],
            "fade_in": None,
            "fade_out": None,
        },
        {
            "name": "outro",
            "bars": 8,
            "layers": [
                {"role": "kick", "type": "oneshot", "beats": [0, 2], "gain": 1.0},
                {"role": "hihat", "type": "oneshot", "beats": [1, 3], "gain": 0.4},
            ],
            "fade_in": None,
            "fade_out": 6.0,
        },
    ]

    return {
        "bpm": 128,
        "sample_rate": 44100,
        "palette": palette,
        "sections": sections,
    }
