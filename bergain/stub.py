"""Stub session for testing and offline evaluation.

Records calls without needing Ableton — used by tests and GEPA optimization.
"""


class StubSession:
    """Minimal session substitute that logs calls without Ableton."""

    def __init__(self, tempo=120):
        self._tempo = tempo
        self.calls = []

    def tempo(self, bpm):
        self._tempo = bpm
        self.calls.append(("tempo", bpm))

    def status(self):
        return {"tempo": self._tempo, "playing": True, "tracks": []}

    def setup(self, tracks):
        self.calls.append(("setup", [t.name for t in tracks]))
        return len(tracks)

    def browse(self, query, category=None):
        return [{"name": f"Fake {query}.adg"}]

    def clip(self, track, slot, length, notes, name=""):
        self.calls.append(("clip", track, slot, len(notes)))
        return {"track": track, "slot": slot, "notes": len(notes)}

    def fire(self, slot):
        self.calls.append(("fire_scene", slot))

    def fire_clip(self, track, slot):
        self.calls.append(("fire_clip", track, slot))

    def mix(self, **kwargs):
        self.calls.append(("mix", kwargs))

    def play(self):
        self.calls.append(("play",))

    def stop(self):
        self.calls.append(("stop",))

    def load_instrument(self, track, name):
        return name

    def load_sound(self, track, name):
        return name

    def load_effect(self, track, name):
        return name

    def load_drum_kit(self, track, name):
        return name

    def params(self, track, device_index):
        return ["Param A", "Param B"]

    def param(self, track, device_index, param, value):
        pass

    def close(self):
        pass
