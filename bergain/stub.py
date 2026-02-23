"""Stub session for testing and offline evaluation.

Records calls without needing Ableton — used by tests and GEPA optimization.
"""


class _StubDomainProxy:
    """Mock for api.song, api.track(n), etc."""

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key):
        return self._data.get(key)

    def call(self, method, *args):
        pass


class _StubAPI:
    """Mock for session.api — provides song and track proxies."""

    def __init__(self, tempo=120, num_scenes=8):
        self.song = _StubDomainProxy({"tempo": tempo, "num_scenes": num_scenes})

    def track(self, index):
        return _StubDomainProxy({"volume": 0.85})


class StubSession:
    """Minimal session substitute that logs calls without Ableton."""

    def __init__(self, tempo=120):
        self._tempo = tempo
        self.calls = []
        self.api = _StubAPI(tempo=tempo)

    def tempo(self, bpm):
        self._tempo = bpm
        self.api.song._data["tempo"] = bpm
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

    def fade(self, track, target, steps=4, duration=1.0):
        self.calls.append(("fade", track, target))

    def _t(self, track_name):
        """Resolve track name to index (stub always returns 0)."""
        return 0

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
