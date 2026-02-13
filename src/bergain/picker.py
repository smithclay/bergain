"""
SamplePicker — query the sample index and pick samples by role.
Returns metadata dicts only (no audio loading).
"""

import random

from bergain.indexer import build_index


class SamplePicker:
    def __init__(self, sample_dir: str):
        self.index = build_index(sample_dir)

    def find(
        self,
        category: str | None = None,
        sub_type: str | None = None,
        is_loop: bool | None = None,
    ) -> list[dict]:
        results = self.index
        if category:
            results = [s for s in results if s["category"] == category]
        if sub_type:
            results = [s for s in results if s.get("sub_type") == sub_type]
        if is_loop is not None:
            results = [s for s in results if s["is_loop"] == is_loop]
        return results

    def pick(
        self,
        category: str | None = None,
        sub_type: str | None = None,
        is_loop: bool | None = None,
    ) -> dict:
        candidates = self.find(category, sub_type, is_loop)
        if not candidates:
            raise ValueError(
                f"No sample found: category={category}, sub_type={sub_type}, is_loop={is_loop}"
            )
        return random.choice(candidates)
