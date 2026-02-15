"""
bergain - Control Ableton Live via AbletonOSC (remix-mcp fork).

Requires the remix-mcp fork of AbletonOSC installed as a Remote Script.
This fork adds /live/browser/* endpoints for loading instruments and effects.
"""

import socket
import threading
import time
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
from spec_data import spec as _ableton_spec

REMOTE_PORT = 11000
LOCAL_PORT = 11001
TICK_DURATION = 0.5


class AbletonOSC:
    """Single-socket OSC client for AbletonOSC (remix-mcp fork).

    Sends and receives on the SAME UDP socket so the reply always comes back
    to our listening port (the fork replies to the sender's address).
    """

    def __init__(self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT):
        self._remote = (hostname, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", client_port))
        self._sock.setblocking(False)
        self._handlers = {}
        self._lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
                self._dispatch(data)
            except BlockingIOError:
                time.sleep(0.005)
            except OSError:
                break

    def _dispatch(self, data: bytes):
        try:
            msg = OscMessage(data)
            with self._lock:
                handler = self._handlers.get(msg.address)
            if handler:
                handler(msg.address, msg.params)
        except Exception:
            pass

    def _build_msg(self, address: str, params) -> bytes:
        builder = OscMessageBuilder(address)
        for p in params:
            builder.add_arg(p)
        return builder.build().dgram

    def send(self, address: str, params=()):
        """Send an OSC message (fire-and-forget)."""
        self._sock.sendto(self._build_msg(address, params), self._remote)

    def query(self, address: str, params=(), timeout: float = TICK_DURATION):
        """Send an OSC message and wait for the reply."""
        rv = None
        event = threading.Event()

        def on_reply(_addr, p):
            nonlocal rv
            rv = tuple(p)
            event.set()

        with self._lock:
            self._handlers[address] = on_reply
        self._sock.sendto(self._build_msg(address, params), self._remote)
        event.wait(timeout)
        with self._lock:
            self._handlers.pop(address, None)
        if not event.is_set():
            raise TimeoutError(f"No response from Ableton for: {address}")
        return rv

    def stop(self):
        self._running = False
        self._sock.close()
        self._thread.join()


class DomainProxy:
    """Proxy for a non-indexed AbletonOSC domain (song, view, browser, etc.)."""

    def __init__(self, osc, domain, addresses):
        self._osc = osc
        self._domain = domain
        self._addresses = addresses

    def _validate(self, address):
        if address not in self._addresses:
            raise ValueError(f"Unknown address: {address}")

    def get(self, prop):
        address = f"/live/{self._domain.name}/get/{prop}"
        self._validate(address)
        result = self._osc.query(address)
        values = result[len(self._domain.index_params) :]
        return values[0] if len(values) == 1 else values

    def set(self, prop, *args):
        address = f"/live/{self._domain.name}/set/{prop}"
        self._validate(address)
        self._osc.send(address, list(args))

    def call(self, method, *args):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        self._osc.send(address, list(args))

    def query(self, method, *args):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        result = self._osc.query(address, list(args))
        return result[len(self._domain.index_params) :]


class IndexedDomainProxy:
    """Proxy for an indexed AbletonOSC domain with indices bound up front."""

    def __init__(self, osc, domain, addresses, indices):
        self._osc = osc
        self._domain = domain
        self._addresses = addresses
        self._indices = list(indices)

    def _validate(self, address):
        if address not in self._addresses:
            raise ValueError(f"Unknown address: {address}")

    def get(self, prop):
        address = f"/live/{self._domain.name}/get/{prop}"
        self._validate(address)
        result = self._osc.query(address, self._indices)
        values = result[len(self._domain.index_params) :]
        return values[0] if len(values) == 1 else values

    def set(self, prop, *args):
        address = f"/live/{self._domain.name}/set/{prop}"
        self._validate(address)
        self._osc.send(address, self._indices + list(args))

    def call(self, method, *args):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        self._osc.send(address, self._indices + list(args))

    def query(self, method, *args):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        result = self._osc.query(address, self._indices + list(args))
        return result[len(self._domain.index_params) :]


class LiveAPI:
    """High-level wrapper around AbletonOSC with domain-scoped proxies.

    Non-indexed domains are direct attributes:
        api.song.set("tempo", 128.0)
        api.browser.call("load_instrument", "Drift")

    Indexed domains are callable, binding indices up front:
        api.clip_slot(0, 0).call("fire")
        api.clip(0, 0).set("name", "Drums")
    """

    def __init__(self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT):
        self._osc = AbletonOSC(hostname, port, client_port)
        self._spec = _ableton_spec
        self._addresses = {ep.address for d in self._spec.domains for ep in d.endpoints}
        self._domains = {d.name: d for d in self._spec.domains}
        for domain in self._spec.domains:
            if not domain.index_params:
                setattr(
                    self, domain.name, DomainProxy(self._osc, domain, self._addresses)
                )

    def _indexed(self, name, *indices):
        return IndexedDomainProxy(
            self._osc, self._domains[name], self._addresses, indices
        )

    def track(self, track_id):
        return self._indexed("track", track_id)

    def clip(self, track_id, clip_id):
        return self._indexed("clip", track_id, clip_id)

    def clip_slot(self, track_id, clip_id):
        return self._indexed("clip_slot", track_id, clip_id)

    def device(self, track_id, device_id):
        return self._indexed("device", track_id, device_id)

    def scene(self, scene_id):
        return self._indexed("scene", scene_id)

    def stop(self):
        self._osc.stop()


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def make_kick(bars=4):
    """Four-on-the-floor kick: C1 (36) every beat."""
    notes = []
    for beat in range(bars * 4):
        notes.extend([36, float(beat), 0.5, 100, 0])
    return notes


def make_hihat(bars=4):
    """Closed hi-hat on 8th notes: F#1 (42), accented on offbeats."""
    notes = []
    for eighth in range(bars * 8):
        beat = eighth * 0.5
        vel = 100 if eighth % 2 == 1 else 60
        notes.extend([42, beat, 0.25, vel, 0])
    return notes


def make_clap(bars=4):
    """Clap on beats 2 and 4: D1 (38)."""
    notes = []
    for bar in range(bars):
        for beat_in_bar in [1, 3]:
            beat = bar * 4 + beat_in_bar
            notes.extend([38, float(beat), 0.5, 90, 0])
    return notes


def make_bass(bars=4):
    """Simple 1-bar bassline looped. Notes in C minor."""
    pattern = [
        (36, 0.0, 0.75, 100),  # C2
        (36, 1.0, 0.25, 80),  # C2
        (39, 1.5, 0.5, 90),  # Eb2
        (36, 2.5, 0.5, 85),  # C2
        (34, 3.0, 0.25, 75),  # Bb1
        (36, 3.5, 0.5, 95),  # C2
    ]
    notes = []
    for bar in range(bars):
        offset = bar * 4
        for pitch, start, dur, vel in pattern:
            notes.extend([pitch, start + offset, dur, vel, 0])
    return notes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_clip(api, track, slot, length, notes, name):
    """Create a MIDI clip, populate it with notes, and name it."""
    cs = api.clip_slot(track, slot)
    if cs.get("has_clip"):
        cs.call("delete_clip")
        time.sleep(0.1)

    cs.call("create_clip", float(length))
    time.sleep(0.1)

    api.clip(track, slot).call("add/notes", *notes)
    api.clip(track, slot).set("name", name)
    print(
        f"  Track {track}, slot {slot}: '{name}' ({length} beats, {len(notes) // 5} notes)"
    )


def load_instrument(api, track, name):
    """Select a track and load an instrument by name via the browser API."""
    api.view.set("selected_track", track)
    time.sleep(0.1)
    api.browser.call("load_instrument", name)
    time.sleep(1.0)
    print(f"  Track {track}: loaded '{name}'")


def load_drum_kit(api, track, name=None):
    """Select a track and load a drum kit via the browser API."""
    api.view.set("selected_track", track)
    time.sleep(0.1)
    if name:
        api.browser.call("load_drum_kit", name)
    else:
        api.browser.call("load_drum_kit")
    time.sleep(1.0)
    print(f"  Track {track}: loaded drum kit" + (f" '{name}'" if name else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    api = LiveAPI()
    bars = 4
    beats = bars * 4
    bpm = 128

    print(f"Setting tempo to {bpm} BPM...")
    api.song.set("tempo", float(bpm))

    api.song.call("stop_playing")
    time.sleep(0.2)

    # Load instruments via browser API
    print("\nLoading instruments...")
    load_drum_kit(api, track=0)
    load_instrument(api, track=1, name="Drift")

    # Create clips
    print(f"\nCreating {bars}-bar clips...")
    drum_notes = make_kick(bars) + make_hihat(bars) + make_clap(bars)
    create_clip(api, track=0, slot=0, length=beats, notes=drum_notes, name="Drums")

    bass_notes = make_bass(bars)
    create_clip(api, track=1, slot=0, length=beats, notes=bass_notes, name="Bass")

    # Fire and play
    print("\nFiring clips...")
    api.clip_slot(0, 0).call("fire")
    api.clip_slot(1, 0).call("fire")
    api.song.call("start_playing")
    print("Playing! Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        api.song.call("stop_playing")

    api.stop()


if __name__ == "__main__":
    main()
