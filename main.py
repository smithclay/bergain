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
# Musical constants
# ---------------------------------------------------------------------------

# Chord progression: Cm - Fm - Ab - Bb (i - iv - VI - VII)
BASS_ROOTS = [36, 41, 44, 46]  # C2, F2, Ab2, Bb2

CHORDS = [
    (60, 63, 67),  # Cm: C4, Eb4, G4
    (53, 56, 60),  # Fm: F3, Ab3, C4
    (56, 60, 63),  # Ab: Ab3, C4, Eb4
    (58, 62, 65),  # Bb: Bb3, D4, F4
]


# ---------------------------------------------------------------------------
# Drum pattern variants
# ---------------------------------------------------------------------------


def make_intro_drums(bars=4):
    """Light hi-hats only."""
    notes = []
    for eighth in range(bars * 8):
        beat = eighth * 0.5
        vel = 65 if eighth % 2 == 1 else 35
        notes.extend([42, beat, 0.25, vel, 0])
    return notes


def make_build_drums(bars=4):
    """Kick + hi-hats + soft claps."""
    notes = make_kick(bars) + make_hihat(bars)
    for bar in range(bars):
        for beat_in_bar in [1, 3]:
            beat = bar * 4 + beat_in_bar
            notes.extend([38, float(beat), 0.5, 55, 0])
    return notes


def make_breakdown_drums(bars=4):
    """Sparse: kick on 1, ride cymbal on quarters."""
    notes = []
    for bar in range(bars):
        notes.extend([36, float(bar * 4), 0.75, 75, 0])
        for beat in range(4):
            notes.extend([51, float(bar * 4 + beat), 0.5, 45, 0])
    return notes


# ---------------------------------------------------------------------------
# Bass pattern variants
# ---------------------------------------------------------------------------


def make_build_bass(bars=4):
    """Simple sustained roots, one per bar."""
    notes = []
    for bar in range(bars):
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        notes.extend([root, float(bar * 4), 3.0, 90, 0])
    return notes


def make_drop_bass(bars=4):
    """Syncopated bass following chord roots."""
    offsets = [
        (0.0, 0.75, 100),
        (1.0, 0.25, 80),
        (1.5, 0.5, 90),
        (2.5, 0.5, 85),
        (3.0, 0.25, 75),
        (3.5, 0.5, 95),
    ]
    notes = []
    for bar in range(bars):
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        base = bar * 4
        for start, dur, vel in offsets:
            notes.extend([root, start + base, dur, vel, 0])
    return notes


def make_breakdown_bass(bars=4):
    """Walking bass through chord tones."""
    walks = [
        [36, 39, 43, 48],  # Cm: C2, Eb2, G2, C3
        [41, 44, 48, 53],  # Fm: F2, Ab2, C3, F3
        [44, 48, 51, 56],  # Ab: Ab2, C3, Eb3, Ab3
        [46, 50, 53, 58],  # Bb: Bb2, D3, F3, Bb3
    ]
    notes = []
    for bar in range(bars):
        walk = walks[bar % len(walks)]
        for i, pitch in enumerate(walk):
            notes.extend([pitch, float(bar * 4 + i), 0.9, 80, 0])
    return notes


def make_finale_bass(bars=4):
    """Driving bass with octave jumps on offbeats."""
    notes = []
    for bar in range(bars):
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        for beat in range(4):
            t = float(bar * 4 + beat)
            notes.extend([root, t, 0.4, 100, 0])
            notes.extend([root + 12, t + 0.5, 0.35, 80, 0])
    return notes


# ---------------------------------------------------------------------------
# Chord / Pad patterns
# ---------------------------------------------------------------------------


def make_pad_sustained(bars=4):
    """Long sustained chords, one per bar."""
    notes = []
    for bar in range(bars):
        chord = CHORDS[bar % len(CHORDS)]
        for pitch in chord:
            notes.extend([pitch, float(bar * 4), 3.8, 70, 0])
    return notes


def make_pad_rhythmic(bars=4):
    """Short rhythmic chord stabs."""
    stab_beats = [0.0, 1.0, 2.5, 3.0]
    notes = []
    for bar in range(bars):
        chord = CHORDS[bar % len(CHORDS)]
        for sb in stab_beats:
            for pitch in chord:
                notes.extend([pitch, bar * 4 + sb, 0.25, 85, 0])
    return notes


# ---------------------------------------------------------------------------
# Lead melody
# ---------------------------------------------------------------------------


def make_lead(bars=4):
    """4-bar melody outlining the Cm-Fm-Ab-Bb progression."""
    phrase = [
        # Bar 1 (Cm)
        (67, 0.0, 0.75, 95),  # G4
        (70, 1.0, 0.5, 85),  # Bb4
        (72, 1.5, 1.5, 100),  # C5
        (70, 3.0, 0.5, 80),  # Bb4
        (67, 3.5, 0.5, 75),  # G4
        # Bar 2 (Fm)
        (65, 4.0, 1.0, 90),  # F4
        (68, 5.0, 0.5, 85),  # Ab4
        (72, 5.5, 1.5, 95),  # C5
        (68, 7.0, 0.5, 70),  # Ab4
        (65, 7.5, 0.5, 75),  # F4
        # Bar 3 (Ab) - climax
        (63, 8.0, 0.75, 90),  # Eb4
        (68, 9.0, 0.5, 85),  # Ab4
        (72, 9.5, 1.0, 95),  # C5
        (75, 10.5, 1.5, 100),  # Eb5
        # Bar 4 (Bb) - descend
        (74, 12.0, 1.0, 95),  # D5
        (70, 13.0, 0.5, 85),  # Bb4
        (65, 13.5, 1.0, 80),  # F4
        (62, 14.5, 1.5, 75),  # D4
    ]
    notes = []
    phrase_len = 16
    for rep in range(max(1, (bars * 4) // phrase_len)):
        offset = rep * phrase_len
        for pitch, start, dur, vel in phrase:
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


def load_effect(api, track, name):
    """Select a track and load an audio effect by name."""
    api.view.set("selected_track", track)
    time.sleep(0.1)
    api.browser.call("load_audio_effect", name)
    time.sleep(0.5)
    print(f"  Track {track}: loaded effect '{name}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    api = LiveAPI()
    bars = 4
    beats = bars * 4
    bpm = 124

    # --- Global setup ---
    print("Setting up session...")
    api.song.call("stop_playing")
    time.sleep(0.2)
    api.song.set("tempo", float(bpm))
    api.song.set("groove_amount", 0.25)
    api.song.set("metronome", False)

    # --- Load instruments (expects 4 MIDI tracks) ---
    print("\nLoading instruments...")
    load_drum_kit(api, track=0)
    load_instrument(api, track=1, name="Drift")
    load_instrument(api, track=2, name="Drift")
    load_instrument(api, track=3, name="Drift")

    # Name tracks
    api.track(0).set("name", "Drums")
    api.track(1).set("name", "Bass")
    api.track(2).set("name", "Pad")
    api.track(3).set("name", "Lead")

    # --- Effects ---
    print("\nLoading effects...")
    load_effect(api, track=2, name="Reverb")
    load_effect(api, track=3, name="Delay")

    # --- Mix ---
    api.track(0).set("volume", 0.85)
    api.track(1).set("volume", 0.80)
    api.track(2).set("volume", 0.65)
    api.track(3).set("volume", 0.70)
    api.track(2).set("panning", -0.15)
    api.track(3).set("panning", 0.15)

    # --- Create clips across 5 scenes ---
    print(f"\nCreating {bars}-bar clips across 5 scenes...")

    # Scene 0: Intro — hats + pad only
    create_clip(api, 0, 0, beats, make_intro_drums(bars), "Intro Drums")
    create_clip(api, 2, 0, beats, make_pad_sustained(bars), "Intro Pad")

    # Scene 1: Build — full drums + bass enters + pad
    create_clip(api, 0, 1, beats, make_build_drums(bars), "Build Drums")
    create_clip(api, 1, 1, beats, make_build_bass(bars), "Build Bass")
    create_clip(api, 2, 1, beats, make_pad_sustained(bars), "Build Pad")

    # Scene 2: Drop — everything, rhythmic stabs, lead enters
    drop_drums = make_kick(bars) + make_hihat(bars) + make_clap(bars)
    create_clip(api, 0, 2, beats, drop_drums, "Drop Drums")
    create_clip(api, 1, 2, beats, make_drop_bass(bars), "Drop Bass")
    create_clip(api, 2, 2, beats, make_pad_rhythmic(bars), "Drop Pad")
    create_clip(api, 3, 2, beats, make_lead(bars), "Lead")

    # Scene 3: Breakdown — sparse drums, walking bass, sustained pad
    create_clip(api, 0, 3, beats, make_breakdown_drums(bars), "Break Drums")
    create_clip(api, 1, 3, beats, make_breakdown_bass(bars), "Break Bass")
    create_clip(api, 2, 3, beats, make_pad_sustained(bars), "Break Pad")

    # Scene 4: Finale — full energy, driving bass, lead returns
    create_clip(api, 0, 4, beats, drop_drums, "Finale Drums")
    create_clip(api, 1, 4, beats, make_finale_bass(bars), "Finale Bass")
    create_clip(api, 2, 4, beats, make_pad_rhythmic(bars), "Finale Pad")
    create_clip(api, 3, 4, beats, make_lead(bars), "Finale Lead")

    # Name scenes
    scene_names = ["Intro", "Build", "Drop", "Breakdown", "Finale"]
    for i, name in enumerate(scene_names):
        api.scene(i).set("name", name)

    # --- Play through the set ---
    print("\nStarting set...")
    api.song.call("start_playing")

    scene_bars = [4, 4, 8, 4, 8]  # how long to hold each scene
    bar_duration = 4 * 60.0 / bpm

    try:
        for i, (name, nbars) in enumerate(zip(scene_names, scene_bars)):
            print(f"  >>> {name} ({nbars} bars)")
            api.scene(i).call("fire")
            time.sleep(nbars * bar_duration)

        print("\nSet complete — looping finale. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        api.song.call("stop_playing")

    api.stop()


if __name__ == "__main__":
    main()
