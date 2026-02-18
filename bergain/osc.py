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
from .spec_data import spec as _ableton_spec

REMOTE_PORT = 11000
LOCAL_PORT = 11001
TICK_DURATION = 0.5


class AbletonOSC:
    """Single-socket OSC client for AbletonOSC (remix-mcp fork).

    Sends and receives on the SAME UDP socket so the reply always comes back
    to our listening port (the fork replies to the sender's address).
    """

    def __init__(
        self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT, retries=0
    ):
        self._remote = (hostname, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", client_port))
        self._sock.setblocking(False)
        self._handlers = {}
        self._lock = threading.Lock()
        self._retries = retries

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
        except Exception as e:
            print(f"  [OSC ERROR] {type(e).__name__}: {e}")

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
        last_err = None
        for attempt in range(1 + self._retries):
            if attempt > 0:
                time.sleep(0.2 * attempt)
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
            if event.is_set():
                return rv
            last_err = TimeoutError(f"No response from Ableton for: {address}")
        raise last_err

    def stop(self):
        self._running = False
        self._sock.close()
        self._thread.join()


class DomainProxy:
    """Proxy for an AbletonOSC domain, optionally with bound indices.

    Non-indexed domains (song, view, browser) use indices=[].
    Indexed domains (track, clip, device) bind indices up front:
        api.track(0).get("name")   # indices=[0]
        api.clip(0, 2).set(...)    # indices=[0, 2]
    """

    def __init__(self, osc, domain, addresses, indices=()):
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

    def query(self, method, *args, timeout=TICK_DURATION):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        result = self._osc.query(address, self._indices + list(args), timeout=timeout)
        return result[len(self._domain.index_params) :]

    def send_batched(self, method, data, batch_size=200, values_per_item=5):
        address = f"/live/{self._domain.name}/{method}"
        self._validate(address)
        total = len(data) // values_per_item
        for i in range(0, total, batch_size):
            chunk = data[i * values_per_item : (i + batch_size) * values_per_item]
            self._osc.send(address, self._indices + chunk)
            if i + batch_size < total:
                time.sleep(0.05)


class LiveAPI:
    """High-level wrapper around AbletonOSC with domain-scoped proxies.

    Non-indexed domains are direct attributes:
        api.song.set("tempo", 128.0)
        api.browser.call("load_instrument", "Drift")

    Indexed domains are callable, binding indices up front:
        api.clip_slot(0, 0).call("fire")
        api.clip(0, 0).set("name", "Drums")
    """

    def __init__(
        self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT, retries=0
    ):
        self._osc = AbletonOSC(hostname, port, client_port, retries=retries)
        self._spec = _ableton_spec
        self._addresses = {ep.address for d in self._spec.domains for ep in d.endpoints}
        self._domains = {d.name: d for d in self._spec.domains}
        for domain in self._spec.domains:
            if not domain.index_params:
                setattr(
                    self, domain.name, DomainProxy(self._osc, domain, self._addresses)
                )

    def _indexed(self, name, *indices):
        return DomainProxy(self._osc, self._domains[name], self._addresses, indices)

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
    "https://smithclay--bergain-aesthetics-analyzer-analyze.modal.run",
)

JUDGE_URL = os.environ.get(
    "BERGAIN_JUDGE_URL",
    "https://smithclay--bergain-aesthetics-judge-score.modal.run",
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
    api.device(capture_track, capture_device).set("parameter/value", 1, 1.0)
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
    import requests

    files = {}
    handles = []
    try:
        f1 = open(file_path, "rb")
        handles.append(f1)
        files["file"] = ("audio.wav", f1, "audio/wav")
        if file_path_2:
            f2 = open(file_path_2, "rb")
            handles.append(f2)
            files["file2"] = ("audio2.wav", f2, "audio/wav")

        data = {"analysis_type": analysis_type}
        if bpm is not None:
            data["bpm"] = str(bpm)

        resp = requests.post(ANALYZER_URL, files=files, data=data, timeout=120)
        resp.raise_for_status()
        return resp.json()
    finally:
        for h in handles:
            h.close()


def score_audio(file_path: str) -> dict:
    """Send audio to the Modal Judge endpoint for audiobox_aesthetics scoring."""
    import requests

    with open(file_path, "rb") as f:
        resp = requests.post(
            JUDGE_URL,
            files={"file": ("audio.wav", f, "audio/wav")},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import sys

    commands = {
        "info": "Show current song state",
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

    if cmd == "info":
        from .session import Session

        s = Session()
        s.info()
        s.close()

    elif cmd == "capture":
        api = LiveAPI()
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
