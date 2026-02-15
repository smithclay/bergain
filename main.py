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


def create_clip(osc, track, slot, length, notes, name):
    """Create a MIDI clip, populate it with notes, and name it."""
    has_clip = osc.query("/live/clip_slot/get/has_clip", (track, slot), timeout=1.0)
    if has_clip[2]:
        osc.send("/live/clip_slot/delete_clip", (track, slot))
        time.sleep(0.1)

    osc.send("/live/clip_slot/create_clip", (track, slot, float(length)))
    time.sleep(0.1)

    osc.send("/live/clip/add/notes", [track, slot] + notes)
    osc.send("/live/clip/set/name", [track, slot, name])
    print(
        f"  Track {track}, slot {slot}: '{name}' ({length} beats, {len(notes) // 5} notes)"
    )


def load_instrument(osc, track, name):
    """Select a track and load an instrument by name via the browser API."""
    osc.send("/live/view/set/selected_track", [track])
    time.sleep(0.1)
    osc.send("/live/browser/load_instrument", [name])
    time.sleep(1.0)
    print(f"  Track {track}: loaded '{name}'")


def load_drum_kit(osc, track, name=None):
    """Select a track and load a drum kit via the browser API."""
    osc.send("/live/view/set/selected_track", [track])
    time.sleep(0.1)
    if name:
        osc.send("/live/browser/load_drum_kit", [name])
    else:
        osc.send("/live/browser/load_drum_kit", [])
    time.sleep(1.0)
    print(f"  Track {track}: loaded drum kit" + (f" '{name}'" if name else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    osc = AbletonOSC()
    bars = 4
    beats = bars * 4
    bpm = 128

    print(f"Setting tempo to {bpm} BPM...")
    osc.send("/live/song/set/tempo", [float(bpm)])

    osc.send("/live/song/stop_playing")
    time.sleep(0.2)

    # Load instruments via browser API
    print("\nLoading instruments...")
    load_drum_kit(osc, track=0)
    load_instrument(osc, track=1, name="Drift")

    # Create clips
    print(f"\nCreating {bars}-bar clips...")
    drum_notes = make_kick(bars) + make_hihat(bars) + make_clap(bars)
    create_clip(osc, track=0, slot=0, length=beats, notes=drum_notes, name="Drums")

    bass_notes = make_bass(bars)
    create_clip(osc, track=1, slot=0, length=beats, notes=bass_notes, name="Bass")

    # Fire and play
    print("\nFiring clips...")
    osc.send("/live/clip_slot/fire", (0, 0))
    osc.send("/live/clip_slot/fire", (1, 0))
    osc.send("/live/song/start_playing")
    print("Playing! Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        osc.send("/live/song/stop_playing")

    osc.stop()


if __name__ == "__main__":
    main()
