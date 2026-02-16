"""
bergain - Control Ableton Live via AbletonOSC (remix-mcp fork).

Requires the remix-mcp fork of AbletonOSC installed as a Remote Script.
This fork adds /live/browser/* endpoints for loading instruments and effects.
"""

import os
import socket
import threading
import time
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
from spec_data import spec as _ableton_spec

import requests

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

    def arrangement_clip(self, track_id, clip_id):
        return self._indexed("arrangement_clip", track_id, clip_id)

    def scene(self, scene_id):
        return self._indexed("scene", scene_id)

    def stop(self):
        self._osc.stop()


# ---------------------------------------------------------------------------
# Audio capture & analysis
# ---------------------------------------------------------------------------

ANALYZER_URL = os.environ.get(
    "BERGAIN_ANALYZER_URL",
    "https://bergain-aesthetics--analyzer-analyze.modal.run",
)


def capture_audio(api, start_time: float, duration: float) -> str:
    """Capture audio from Ableton via the File Recorder M4L device.

    Uses File Recorder (maxforlive.com/library/device/10536) in "Link
    Start/Stop" mode: recording starts/stops automatically with transport.

    Setup: place File Recorder on the track you want to capture (e.g. Master
    routed to a utility track). Set env vars:
      CAPTURE_TRACK   — track index where File Recorder lives (default 0)
      CAPTURE_DEVICE  — device index on that track (default 0)
      CAPTURE_DIR     — directory where File Recorder saves WAVs; defaults
                        to the Ableton project folder. File Recorder writes
                        files named "[TrackName]+[Timestamp].wav".

    Orchestrates: snapshot dir → seek → play → wait → stop → find new file.
    Returns the path to the captured WAV file.
    """
    import glob

    capture_track = int(os.environ.get("CAPTURE_TRACK", "0"))
    capture_device = int(os.environ.get("CAPTURE_DEVICE", "0"))
    capture_dir = os.environ.get("CAPTURE_DIR", "")

    if not capture_dir:
        raise ValueError(
            "CAPTURE_DIR must be set to the folder where File Recorder saves WAVs "
            "(typically your Ableton project folder)"
        )

    # Snapshot existing WAVs so we can detect the new one
    existing = set(glob.glob(os.path.join(capture_dir, "*.wav")))

    # Ensure "Link Start/Stop" is ON (File Recorder param index 1)
    # Param 0 = Rec toggle, Param 1 = Link Start/Stop
    api._osc.send(
        "/live/device/set/parameter/value", [capture_track, capture_device, 1, 1.0]
    )
    time.sleep(0.1)

    # Seek to start position
    api.song.set("current_song_time", start_time)
    time.sleep(0.1)

    # Start playback — File Recorder auto-starts recording
    api.song.call("start_playing")

    # Wait for the capture duration
    time.sleep(duration + 0.5)

    # Stop playback — File Recorder auto-stops and writes the WAV
    api.song.call("stop_playing")
    time.sleep(1.0)  # give it a moment to flush to disk

    # Find the new WAV file
    current = set(glob.glob(os.path.join(capture_dir, "*.wav")))
    new_files = current - existing

    if not new_files:
        raise FileNotFoundError(
            f"No new WAV file appeared in {capture_dir} after capture. "
            "Check that File Recorder is on the correct track/device and "
            "CAPTURE_DIR points to its output folder."
        )

    # Return the most recently modified new file
    return max(new_files, key=os.path.getmtime)


def analyze_audio(
    file_path: str,
    analysis_type: str,
    bpm: float | None = None,
    file_path_2: str | None = None,
) -> dict:
    """Send audio to the Modal Analyzer endpoint and return results."""
    files = {"file": ("audio.wav", open(file_path, "rb"), "audio/wav")}
    data = {"analysis_type": analysis_type}
    if bpm is not None:
        data["bpm"] = str(bpm)
    if file_path_2:
        files["file2"] = ("audio2.wav", open(file_path_2, "rb"), "audio/wav")

    resp = requests.post(ANALYZER_URL, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


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


def setup_tracks(api, track_specs):
    """Create fresh MIDI tracks for each spec, removing all existing tracks.

    track_specs: list of (name, instrument_loader) tuples.
    Returns the number of tracks set up.
    """
    # Create the new MIDI tracks first (can't delete all — need at least one)
    for i in range(len(track_specs)):
        api._osc.send("/live/song/create_midi_track", [0])
        time.sleep(0.3)

    # Delete the old tracks (now pushed to the end)
    num_tracks = api.song.get("num_tracks")
    while num_tracks > len(track_specs):
        api._osc.send("/live/song/delete_track", [num_tracks - 1])
        time.sleep(0.3)
        num_tracks = api.song.get("num_tracks")

    # Name and load instruments
    for t, (name, loader) in enumerate(track_specs):
        api.track(t).set("name", name)
        if loader:
            loader(api, t)

    return len(track_specs)


def create_arrangement_clip_with_notes(api, track, start, length, notes, name):
    """Create arrangement clip, populate with notes, name it. Returns clip index."""
    result = api._osc.query(
        "/live/track/create_arrangement_clip",
        [track, float(start), float(length)],
        timeout=3.0,
    )
    clip_idx = result[1]
    time.sleep(0.15)
    if notes:
        api._osc.send("/live/arrangement_clip/add/notes", [track, clip_idx] + notes)
    api.arrangement_clip(track, clip_idx).set("name", name)
    time.sleep(0.1)
    return clip_idx


def clear_arrangement(api, num_tracks):
    """Delete all arrangement clips from all tracks."""
    for t in range(num_tracks):
        for _ in range(50):
            try:
                api._osc.send("/live/track/delete_arrangement_clip", [t, 0])
                time.sleep(0.1)
            except Exception:
                break


def print_arrangement(api, num_tracks):
    """Print the full arrangement layout."""
    print("\n  Arrangement layout:")
    for t in range(num_tracks):
        try:
            names = api._osc.query("/live/track/get/arrangement_clips/name", [t])
            times = api._osc.query("/live/track/get/arrangement_clips/start_time", [t])
            clip_names = list(names[1:]) if len(names) > 1 else []
            clip_times = list(times[1:]) if len(times) > 1 else []
            track_name = api.track(t).get("name")
            for cn, ct in zip(clip_names, clip_times):
                bar = int(ct) // 4 + 1
                print(f"    {track_name:8s} | bar {bar:3d} | '{cn}'")
        except TimeoutError:
            pass


def demo_full(api):
    """Build and play a complete 5-section song demonstrating the full API.

    Demonstrates:
      - Song control (tempo, groove, metronome)
      - Track creation and deletion
      - Instrument & effect loading via browser
      - Mix (volumes, panning)
      - Arrangement clip creation with MIDI notes
      - Clip duplication, splitting, and moving
      - Rack/chain queries (Drum Rack)
      - Session clip creation (scene triggers)
      - Playback
    """
    BPM = 124
    BARS_PER_SECTION = 4
    BEATS = BARS_PER_SECTION * 4  # 16 beats per section

    # ── Song setup ──────────────────────────────────────────────────────
    print("1. Song setup")
    api.song.call("stop_playing")
    time.sleep(0.2)
    api.song.set("tempo", float(BPM))
    api.song.set("groove_amount", 0.25)
    api.song.set("metronome", False)
    print(f"   Tempo: {BPM} BPM, groove: 0.25")

    # ── Track setup ─────────────────────────────────────────────────────
    print("\n2. Track setup")
    n_tracks = setup_tracks(
        api,
        [
            ("Drums", lambda a, t: load_drum_kit(a, t)),
            ("Bass", lambda a, t: load_instrument(a, t, "Drift")),
            ("Pad", lambda a, t: load_instrument(a, t, "Drift")),
            ("Lead", lambda a, t: load_instrument(a, t, "Drift")),
        ],
    )

    # ── Effects ─────────────────────────────────────────────────────────
    print("\n3. Effects")
    load_effect(api, track=2, name="Reverb")
    load_effect(api, track=3, name="Delay")

    # ── Mix ─────────────────────────────────────────────────────────────
    print("\n4. Mix")
    mix = [(0, 0.85, 0.0), (1, 0.80, 0.0), (2, 0.65, -0.15), (3, 0.70, 0.15)]
    for t, vol, pan in mix:
        api.track(t).set("volume", vol)
        api.track(t).set("panning", pan)
        name = api.track(t).get("name")
        print(f"   {name}: vol={vol}, pan={pan}")

    # ── Chain queries ───────────────────────────────────────────────────
    print("\n5. Rack/chain inspection")
    try:
        result = api._osc.query("/live/device/get/num_chains", [0, 0], timeout=1.5)
        n_chains = result[2]
        dev_name = api.device(0, 0).get("name")
        print(f"   Drums > {dev_name}: {n_chains} chains")
        names = api._osc.query("/live/device/get/chains/name", [0, 0])
        for i, cn in enumerate(list(names[2:])[:8]):
            print(f"     [{i:2d}] {cn}")
        if n_chains > 8:
            print(f"     ... and {n_chains - 8} more")
    except (TimeoutError, Exception) as e:
        print(f"   No rack chains found: {e}")

    # ── Arrangement ─────────────────────────────────────────────────────
    print("\n6. Building arrangement")
    clear_arrangement(api, n_tracks)

    # Section layout: Intro(8) → Build(8) → Drop(16) → Breakdown(8) → Finale(16)
    # All times in beats. Each "bars" unit = BARS_PER_SECTION = 4 bars = 16 beats.
    sections = [
        # (name, start_beat, n_bars, drums_fn, bass_fn, pad_fn, lead_fn)
        ("Intro", 0, 8, make_intro_drums, None, make_pad_sustained, None),
        ("Build", 32, 8, make_build_drums, make_build_bass, make_pad_sustained, None),
        (
            "Drop",
            64,
            8,
            lambda b: make_kick(b) + make_hihat(b) + make_clap(b),
            make_drop_bass,
            make_pad_rhythmic,
            make_lead,
        ),
        (
            "Breakdown",
            96,
            8,
            make_breakdown_drums,
            make_breakdown_bass,
            make_pad_sustained,
            None,
        ),
        (
            "Finale",
            128,
            8,
            lambda b: make_kick(b) + make_hihat(b) + make_clap(b),
            make_finale_bass,
            make_pad_rhythmic,
            make_lead,
        ),
    ]

    for sec_name, start_beat, n_bars, drum_fn, bass_fn, pad_fn, lead_fn in sections:
        beats = n_bars * 4
        print(f"   {sec_name} @ bar {start_beat // 4 + 1} ({n_bars} bars)")
        fns = [drum_fn, bass_fn, pad_fn, lead_fn]
        track_names = ["Drums", "Bass", "Pad", "Lead"]
        for t, (fn, tname) in enumerate(zip(fns, track_names)):
            if fn is None:
                continue
            notes = fn(n_bars)
            clip_name = f"{sec_name} {tname}"
            create_arrangement_clip_with_notes(
                api, t, start_beat, beats, notes, clip_name
            )

    # ── Demonstrate split, duplicate, move ──────────────────────────────
    print("\n7. Arrangement operations")

    # Duplicate the Drop drums (index varies, find by name)
    drop_drums_start = 64.0
    finale_end = 128.0 + 8 * 4  # 160
    print(f"   Duplicating Drop Drums to beat {finale_end} (outro)...")
    # Find the Drop Drums clip index
    times = api._osc.query("/live/track/get/arrangement_clips/start_time", [0])
    drop_idx = None
    for i, st in enumerate(times[1:]):
        if abs(st - drop_drums_start) < 0.01:
            drop_idx = i
            break
    if drop_idx is not None:
        result = api._osc.query(
            "/live/track/duplicate_arrangement_clip",
            [0, drop_idx, finale_end],
            timeout=3.0,
        )
        print(f"     Created outro drums at clip {result[1]}")

    # Split the Finale bass at the midpoint for a breakdown effect
    finale_bass_start = 128.0
    split_point = finale_bass_start + 16.0  # split 4 bars in
    print(f"   Splitting Finale Bass at beat {split_point}...")
    times = api._osc.query("/live/track/get/arrangement_clips/start_time", [1])
    finale_bass_idx = None
    for i, st in enumerate(times[1:]):
        if abs(st - finale_bass_start) < 0.01:
            finale_bass_idx = i
            break
    if finale_bass_idx is not None:
        result = api._osc.query(
            "/live/track/split_arrangement_clip",
            [1, finale_bass_idx, split_point],
            timeout=3.0,
        )
        print(f"     Split into clips {result[1]} and {result[2]}")
        # Move the second half later to create a 2-bar gap
        gap_dest = split_point + 8.0
        print(f"   Moving second half to beat {gap_dest} (2-bar gap)...")
        result = api._osc.query(
            "/live/track/move_arrangement_clip", [1, result[2], gap_dest], timeout=3.0
        )
        print(f"     Moved to clip {result[1]}")

    print_arrangement(api, n_tracks)

    # ── Session clips (for live triggering) ─────────────────────────────
    print("\n8. Session clips")
    scene_names = ["Intro", "Build", "Drop", "Breakdown", "Finale"]
    scene_patterns = [
        # (drums, bass, pad, lead) — fn or None
        (make_intro_drums, None, make_pad_sustained, None),
        (make_build_drums, make_build_bass, make_pad_sustained, None),
        (
            lambda b: make_kick(b) + make_hihat(b) + make_clap(b),
            make_drop_bass,
            make_pad_rhythmic,
            make_lead,
        ),
        (make_breakdown_drums, make_breakdown_bass, make_pad_sustained, None),
        (
            lambda b: make_kick(b) + make_hihat(b) + make_clap(b),
            make_finale_bass,
            make_pad_rhythmic,
            make_lead,
        ),
    ]
    for scene_idx, (name, fns) in enumerate(zip(scene_names, scene_patterns)):
        for t, fn in enumerate(fns):
            if fn is not None:
                create_clip(
                    api,
                    t,
                    scene_idx,
                    BEATS,
                    fn(BARS_PER_SECTION),
                    f"{name} {['Drums', 'Bass', 'Pad', 'Lead'][t]}",
                )
        api.scene(scene_idx).set("name", name)

    # ── Song info ───────────────────────────────────────────────────────
    print("\n9. Song summary")
    print(f"   Tempo:  {api.song.get('tempo')} BPM")
    print(f"   Tracks: {api.song.get('num_tracks')}")
    print(f"   Scenes: {api.song.get('num_scenes')}")
    sig_num = api.song.get("signature_numerator")
    sig_den = api.song.get("signature_denominator")
    print(f"   Time:   {sig_num}/{sig_den}")

    # ── Playback ────────────────────────────────────────────────────────
    print("\n10. Playing arrangement...")
    api.song.set("current_song_time", 0.0)
    time.sleep(0.1)
    api.song.call("start_playing")
    bar_duration = 4 * 60.0 / BPM

    section_bars = [8, 8, 8, 8, 8, 8]  # +8 for outro
    section_names = ["Intro", "Build", "Drop", "Breakdown", "Finale", "Outro"]

    try:
        for name, nbars in zip(section_names, section_bars):
            print(f"    >>> {name} ({nbars} bars)")
            time.sleep(nbars * bar_duration)
    except KeyboardInterrupt:
        pass
    api.song.call("stop_playing")
    print("\n   Done.")

    return BPM


def main():
    import sys

    commands = {
        "info": "Show current song state",
        "demo": "Build and play a full 5-section song (instruments, arrangement, chains, mix)",
        "capture": "Capture audio and run analysis (requires File Recorder M4L + Modal)",
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("bergain — Ableton Live control via AbletonOSC\n")
        print("Usage: uv run python main.py <command>\n")
        print("Commands:")
        for cmd, desc in commands.items():
            print(f"  {cmd:10s} {desc}")
        n_eps = sum(len(d.endpoints) for d in _ableton_spec.domains)
        n_doms = len(_ableton_spec.domains)
        print(f"\n{n_doms} domains, {n_eps} endpoints")
        return

    cmd = sys.argv[1]
    api = LiveAPI()

    if cmd == "info":
        print("--- Song Info ---")
        print(f"  Tempo:    {api.song.get('tempo')} BPM")
        print(f"  Tracks:   {api.song.get('num_tracks')}")
        print(f"  Scenes:   {api.song.get('num_scenes')}")
        sig_num = api.song.get("signature_numerator")
        sig_den = api.song.get("signature_denominator")
        print(f"  Time sig: {sig_num}/{sig_den}")
        num_tracks = api.song.get("num_tracks")
        print("\n  Tracks:")
        for t in range(num_tracks):
            name = api.track(t).get("name")
            devs = api.track(t).get("num_devices")
            midi = api.track(t).get("has_midi_input")
            print(f"    [{t}] {name} ({'MIDI' if midi else 'Audio'}, {devs} devices)")
        print("\n  Arrangement:")
        has_clips = False
        for t in range(num_tracks):
            try:
                names = api._osc.query("/live/track/get/arrangement_clips/name", [t])
                times = api._osc.query(
                    "/live/track/get/arrangement_clips/start_time", [t]
                )
                clip_names = list(names[1:]) if len(names) > 1 else []
                clip_times = list(times[1:]) if len(times) > 1 else []
                track_name = api.track(t).get("name")
                for cn, ct in zip(clip_names, clip_times):
                    bar = int(ct) // 4 + 1
                    print(f"    {track_name:8s} | bar {bar:3d} | '{cn}'")
                    has_clips = True
            except TimeoutError:
                pass
        if not has_clips:
            print("    (empty)")

    elif cmd == "demo":
        demo_full(api)

    elif cmd == "capture":
        capture_dir = os.environ.get("CAPTURE_DIR", "")
        if not capture_dir:
            print("Set CAPTURE_DIR to the folder where File Recorder saves WAVs.")
            print("Requires File Recorder M4L device on a track.")
        else:
            bpm = float(api.song.get("tempo"))
            duration = 4 * 4 * 60.0 / bpm
            print(f"Capturing {duration:.1f}s of audio...")
            wav_path = capture_audio(api, start_time=0.0, duration=duration)
            print(f"Captured: {wav_path}")
            for analysis in ["key", "energy"]:
                print(f"\nAnalyzing {analysis}...")
                result = analyze_audio(wav_path, analysis, bpm=bpm)
                if analysis == "key":
                    print(
                        f"  {result['key']} {result['scale']} (confidence: {result['confidence']:.2f})"
                    )
                elif analysis == "energy":
                    print(f"  {len(result['frames'])} frames at {result['bpm']} BPM")

    api.stop()


if __name__ == "__main__":
    main()
