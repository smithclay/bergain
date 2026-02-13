"""
Arrangement data model — plain dicts, JSON-serializable.

An arrangement describes a song declaratively: palette of samples,
sections with bar counts, and layers per section specifying how
samples are placed (oneshot on beats or looped).
"""

import json
from pathlib import Path


def save_arrangement(arrangement: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(arrangement, f, indent=2)


def load_arrangement(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
