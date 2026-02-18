"""bergain session — high-level DJ REPL and composition API.

Session wraps LiveAPI with track-name resolution, device-param caching,
and convenience methods for the live DJ workflow.
"""

import math
import os
import shutil
import struct
import time
import wave
from dataclasses import dataclass, field

from .osc import LiveAPI, REMOTE_PORT, LOCAL_PORT


def _notes_to_wire(tuples):
    """Convert [(pitch, start, dur, vel), ...] to flat AbletonOSC wire format.

    Each note becomes [pitch, start, duration, velocity, mute] where mute=0 means not muted."""
    flat = []
    for p, s, d, v in tuples:
        flat.extend([p, float(s), float(d), v, 0])
    return flat


def _normalize_wav(path, target_dbfs=-1.0):
    """Normalize a WAV file to target_dbfs peak level in-place.

    Uses stdlib wave + struct — no external deps.
    """
    target_linear = 10 ** (target_dbfs / 20.0)  # -1 dBFS ≈ 0.891

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        max_val = 32767
    elif sampwidth == 3:
        # 24-bit: unpack manually
        samples = []
        for i in range(0, len(raw), 3):
            b = raw[i : i + 3]
            val = int.from_bytes(b, "little", signed=True)
            samples.append(val)
        max_val = 2**23 - 1
        peak = max(abs(s) for s in samples) if samples else 0
        if peak == 0:
            print(f"  [normalize] {path}: silent file, skipping")
            return
        gain = (target_linear * max_val) / peak
        print(
            f"  [normalize] peak={peak}/{max_val} ({20 * math.log10(peak / max_val):.1f} dBFS), gain={gain:.2f}x"
        )
        out = bytearray()
        for s in samples:
            clamped = max(-max_val - 1, min(max_val, int(s * gain)))
            out.extend(clamped.to_bytes(3, "little", signed=True))
        with wave.open(path, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(bytes(out))
        return
    else:
        print(f"  [normalize] Unsupported sample width {sampwidth}, skipping")
        return

    # 16-bit path
    samples = list(struct.unpack(fmt, raw))
    peak = max(abs(s) for s in samples) if samples else 0
    if peak == 0:
        print(f"  [normalize] {path}: silent file, skipping")
        return

    gain = (target_linear * max_val) / peak
    print(
        f"  [normalize] peak={peak}/{max_val} ({20 * math.log10(peak / max_val):.1f} dBFS), gain={gain:.2f}x"
    )
    normalized = [max(-max_val - 1, min(max_val, int(s * gain))) for s in samples]
    out_raw = struct.pack(fmt, *normalized)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(out_raw)


def _concatenate_wavs(paths, dest):
    """Concatenate multiple WAV files into one. All must share the same format."""
    with wave.open(paths[0], "rb") as first:
        params = first.getparams()

    with wave.open(dest, "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(p, "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))


def _wav_duration(path):
    """Return duration of a WAV file in seconds."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class Session:
    """High-level DJ REPL API over LiveAPI.

    Resolves tracks by name, caches device params, and provides
    one-call methods for common operations.
    """

    def __init__(
        self, api=None, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT
    ):
        self.api = api or LiveAPI(hostname, port, client_port)
        self._tracks = {}  # name -> index
        self._scenes = {}  # name -> index
        self._param_cache = {}  # (track_idx, device_idx) -> [param_names]
        self.refresh()

    # -- Track / scene resolution ------------------------------------------------

    def _t(self, track):
        """Resolve str|int to track index."""
        if isinstance(track, int):
            return track
        if track in self._tracks:
            return self._tracks[track]
        raise KeyError(f"Unknown track: {track!r} (known: {list(self._tracks)})")

    def _s(self, scene):
        """Resolve str|int to scene index."""
        if isinstance(scene, int):
            return scene
        if scene in self._scenes:
            return self._scenes[scene]
        raise KeyError(f"Unknown scene: {scene!r} (known: {list(self._scenes)})")

    def refresh(self):
        """Re-read track and scene names from Ableton."""
        self._tracks.clear()
        self._scenes.clear()
        self._param_cache.clear()
        try:
            num_tracks = self.api.song.get("num_tracks")
            for i in range(num_tracks):
                name = self.api.track(i).get("name")
                self._tracks[name] = i
        except TimeoutError:
            print("  [WARNING] Could not read tracks from Ableton — names unavailable")
        try:
            num_scenes = self.api.song.get("num_scenes")
            for i in range(num_scenes):
                name = self.api.scene(i).get("name")
                self._scenes[name] = i
        except TimeoutError:
            print("  [WARNING] Could not read scenes from Ableton — names unavailable")

    # -- Clips -------------------------------------------------------------------

    def clip(self, track, slot, length, tuples, name=""):
        """Create a session MIDI clip from note tuples.

        Auto-creates scenes if slot index exceeds current scene count.
        Returns {"track": int, "slot": int, "name": str, "beats": float, "notes": int}.
        """
        t = self._t(track)
        # Auto-create scenes if needed
        try:
            num_scenes = self.api.song.get("num_scenes")
            while slot >= num_scenes:
                self.api.song.call("create_scene", num_scenes)
                time.sleep(0.1)
                num_scenes += 1
        except TimeoutError as e:
            raise TimeoutError(
                f"Could not check/create scenes for slot {slot}. "
                f"Current scene count may be insufficient. Original: {e}"
            ) from e
        cs = self.api.clip_slot(t, slot)
        if cs.get("has_clip"):
            cs.call("delete_clip")
            time.sleep(0.1)
        cs.call("create_clip", float(length))
        time.sleep(0.1)
        if tuples:
            wire = _notes_to_wire(tuples)
            self.api.clip(t, slot).send_batched("add/notes", wire)
        if name:
            self.api.clip(t, slot).set("name", name)
        result = {
            "track": t,
            "slot": slot,
            "name": name,
            "beats": length,
            "notes": len(tuples),
        }
        print(
            f"  Track {t}, slot {slot}: '{name}' ({length} beats, {len(tuples)} notes)"
        )
        return result

    def arr_clip(self, track, start, length, tuples, name=""):
        """Create an arrangement clip from note tuples.

        Returns {"track": int, "clip_index": int, "start": float,
                 "beats": float, "name": str, "notes": int}.
        """
        t = self._t(track)
        result = self.api.track(t).query(
            "create_arrangement_clip", float(start), float(length), timeout=3.0
        )
        clip_idx = result[0]
        time.sleep(0.15)
        if tuples:
            wire = _notes_to_wire(tuples)
            self.api.arrangement_clip(t, clip_idx).send_batched("add/notes", wire)
        if name:
            self.api.arrangement_clip(t, clip_idx).set("name", name)
        time.sleep(0.1)
        return {
            "track": t,
            "clip_index": clip_idx,
            "start": start,
            "beats": length,
            "name": name,
            "notes": len(tuples),
        }

    def fire_clip(self, track, slot=0):
        """Fire a clip by track name and slot index."""
        t = self._t(track)
        self.api.clip_slot(t, slot).call("fire")
        self._maybe_start_deferred_recording()

    def stop_clip(self, track, slot=None):
        """Stop the playing clip on a track.

        If slot is None, stops whatever's playing on the track.
        """
        t = self._t(track)
        if slot is not None:
            self.api.clip_slot(t, slot).call("stop")
        else:
            self.api.track(t).call("stop_all_clips")

    # -- Scene -------------------------------------------------------------------

    def fire(self, scene):
        """Fire a scene by name or index.

        When recording, fires clips on each music track individually to avoid
        disrupting the capture track's session recording.
        """
        s = self._s(scene)
        if getattr(self, "_recording", False):
            capture_idx = getattr(self, "_capture_track_idx", None)
            num_tracks = self.api.song.get("num_tracks")
            for t in range(num_tracks):
                if t == capture_idx:
                    continue
                try:
                    self.api.clip_slot(t, s).call("fire")
                except Exception:
                    pass
        else:
            self.api.scene(s).call("fire")
        self._maybe_start_deferred_recording()

    # -- Device params -----------------------------------------------------------

    def params(self, track, device):
        """Get parameter names for a device (cached)."""
        t = self._t(track)
        key = (t, device)
        if key not in self._param_cache:
            result = self.api.device(t, device).query(
                "get/parameters/name", timeout=2.0
            )
            self._param_cache[key] = list(result)
        return self._param_cache[key]

    def param(self, track, device, name, value=None):
        """Get or set a device parameter by name fragment (case-insensitive)."""
        t = self._t(track)
        param_names = self.params(track, device)
        fragment = name.lower()
        idx = next(
            (i for i, p in enumerate(param_names) if fragment in str(p).lower()),
            None,
        )
        if idx is None:
            raise ValueError(
                f"No param matching '{name}' on track {t} device {device}. "
                f"Available: {param_names}"
            )
        if value is not None:
            self.api.device(t, device).set("parameter/value", idx, float(value))
            return value
        result = self.api.device(t, device).query("get/parameter/value", idx)
        return result[-1]

    # -- Browser loading ---------------------------------------------------------

    def browse(self, query, category=None):
        """Search Ableton browser, return structured results.

        Returns list of {"category": ..., "name": ...} dicts.
        Optionally filter by category substring (case-insensitive).
        """
        raw = self.api.browser.query("search", query)
        if not raw:
            return []
        results = list(raw)
        pairs = [
            {"category": results[i], "name": results[i + 1]}
            for i in range(0, len(results) - 1, 2)
        ]
        if category:
            pairs = [p for p in pairs if category.lower() in p["category"].lower()]
        return pairs

    def load(self, track, name):
        """Load a browser item by name, auto-detecting type.

        Tries instrument -> sound -> drum_kit -> effect in order.
        Returns {"type": ..., "name": ...} on success, None on failure.
        """
        for method, label in [
            ("load_instrument", "instrument"),
            ("load_sound", "sound"),
            ("load_drum_kit", "drum kit"),
            ("load_audio_effect", "effect"),
        ]:
            result = self._browser_load(track, method, name, label, wait=0.5)
            if result:
                return {"type": label, "name": result}
        return None

    def _browser_load(self, track, method, name, label, wait=1.0, retries=2):
        """Load a browser item onto a track, printing success/failure.

        Retries up to `retries` times on TimeoutError with 1s backoff.
        """
        t = self._t(track)
        for attempt in range(1 + retries):
            self.api.view.set("selected_track", t)
            time.sleep(0.1)
            try:
                result = (
                    self.api.browser.query(method, name)
                    if name
                    else self.api.browser.query(method)
                )
                if not result:
                    print(f"  Track {t}: '{name}' not found in browser")
                    return None
                loaded = result[0]
                time.sleep(wait)
                print(f"  Track {t}: loaded {label} '{loaded}'")
                return loaded
            except TimeoutError:
                if attempt < retries:
                    print(
                        f"  Track {t}: load {label} '{name}' timed out, retrying ({attempt + 1}/{retries})..."
                    )
                    time.sleep(1.0)
                else:
                    print(
                        f"  Track {t}: FAILED to load {label} '{name}' after {1 + retries} attempts"
                    )
                    return None

    def load_instrument(self, track, name):
        """Load an instrument by name onto a track."""
        return self._browser_load(track, "load_instrument", name, "instrument")

    def load_drum_kit(self, track, name=None):
        """Load a drum kit onto a track."""
        return self._browser_load(track, "load_drum_kit", name, "drum kit")

    def load_sound(self, track, name):
        """Load a sound preset onto a track."""
        return self._browser_load(track, "load_sound", name, "sound")

    def load_effect(self, track, name):
        """Load an audio effect onto a track."""
        return self._browser_load(track, "load_audio_effect", name, "effect", wait=0.5)

    # -- Track setup -------------------------------------------------------------

    def setup(self, tracks):
        """Create fresh MIDI tracks from Track specs, replacing all existing.

        tracks: list of Track dataclass instances.
        """
        # Create new MIDI tracks
        for _ in tracks:
            self.api.song.call("create_midi_track", 0)
            time.sleep(0.3)

        # Delete old tracks (pushed to end)
        num_tracks = self.api.song.get("num_tracks")
        while num_tracks > len(tracks):
            self.api.song.call("delete_track", num_tracks - 1)
            time.sleep(0.3)
            num_tracks = self.api.song.get("num_tracks")

        # Name tracks and load instruments
        for i, t in enumerate(tracks):
            self.api.track(i).set("name", t.name)
            if t.instrument:
                self.load_instrument(i, t.instrument)
            elif t.drum_kit:
                self.load_drum_kit(i, t.drum_kit)
            elif t.sound:
                self.load_sound(i, t.sound)

        # Load effects
        for i, t in enumerate(tracks):
            for effect_name in t.effects:
                self.load_effect(i, effect_name)

        # Verify instruments loaded (at least 1 device per track that requested one)
        for i, t in enumerate(tracks):
            if t.instrument or t.drum_kit or t.sound:
                try:
                    num_devices = self.api.track(i).get("num_devices")
                    if num_devices == 0:
                        print(
                            f"  [setup] WARNING: Track {i} '{t.name}' has 0 devices — instrument may have failed to load"
                        )
                except TimeoutError:
                    pass

        # Set mix
        for i, t in enumerate(tracks):
            self.api.track(i).set("volume", t.volume)
            self.api.track(i).set("panning", t.pan)

        self.refresh()

        # Restore capture track if recording was active before setup.
        # The old capture track was destroyed along with all other tracks.
        # We stop any zombie session recording, recreate the capture track,
        # and defer triggering recording until the transport starts (play/fire)
        # because trigger_session_record needs a running transport to work.
        if getattr(self, "_recording", False):
            print("  [export] Restoring capture track after setup...")
            self._ensure_recording_stopped()
            time.sleep(0.5)  # let Ableton settle after track churn
            self._create_capture_track()
            self._record_pending = True
            print("  [export] Capture track ready — recording deferred until playback")

        return len(tracks)

    # -- Mix ---------------------------------------------------------------------

    def mix(self, **levels):
        """Set volume/pan for multiple tracks at once.

        Usage:
            sesh.mix(Kick=0.85, Hats=0.55)
            sesh.mix(Kick={"volume": 0.85, "pan": -0.1})
        """
        for track_name, val in levels.items():
            t = self._t(track_name)
            if isinstance(val, dict):
                if "volume" in val:
                    self.api.track(t).set("volume", float(val["volume"]))
                if "pan" in val:
                    self.api.track(t).set("panning", float(val["pan"]))
            else:
                self.api.track(t).set("volume", float(val))

    # -- Arrangement -------------------------------------------------------------

    def clear_arrangement(self, tracks=None):
        """Delete all arrangement clips from specified or all tracks."""
        if tracks is None:
            num = self.api.song.get("num_tracks")
            track_indices = range(num)
        else:
            track_indices = [self._t(t) for t in tracks]
        for t in track_indices:
            for _ in range(50):
                try:
                    self.api.track(t).call("delete_arrangement_clip", 0)
                    time.sleep(0.1)
                except Exception:
                    break

    def arrangement(self):
        """Return and print arrangement clip layout.

        Returns list of {"track": str, "bar": int, "name": str, "start_time": float}.
        """
        clips = []
        num_tracks = self.api.song.get("num_tracks")
        for t in range(num_tracks):
            try:
                name_result = self.api.track(t).query(
                    "get/arrangement_clips/name", timeout=2.0
                )
                time_result = self.api.track(t).query(
                    "get/arrangement_clips/start_time", timeout=2.0
                )
                clip_names = list(name_result) if name_result else []
                clip_times = list(time_result) if time_result else []
                track_name = self.api.track(t).get("name")
                for cn, ct in zip(clip_names, clip_times):
                    bar = int(ct) // 4 + 1
                    clips.append(
                        {"track": track_name, "bar": bar, "name": cn, "start_time": ct}
                    )
            except TimeoutError:
                pass
        print("\n  Arrangement layout:")
        for c in clips:
            print(f"    {c['track']:8s} | bar {c['bar']:3d} | '{c['name']}'")
        if not clips:
            print("    (empty)")
        return clips

    # -- Transport ---------------------------------------------------------------

    def tempo(self, bpm=None):
        """Get or set tempo."""
        if bpm is not None:
            self.api.song.set("tempo", float(bpm))
            return bpm
        return self.api.song.get("tempo")

    def play(self):
        self.api.song.call("start_playing")
        self._maybe_start_deferred_recording()

    def stop(self):
        self.api.song.call("stop_playing")

    def seek(self, beat):
        self.api.song.set("current_song_time", float(beat))

    def fade(self, track, target_volume, steps=4, duration=1.0):
        """Gradually change track volume over duration.

        Returns {"track": int, "from": float, "to": float}.
        """
        t = self._t(track)
        current = self.api.track(t).get("volume")
        step_time = duration / steps
        for i in range(1, steps + 1):
            val = current + (target_volume - current) * (i / steps)
            self.api.track(t).set("volume", float(val))
            if i < steps:
                time.sleep(step_time)
        return {"track": t, "from": current, "to": target_volume}

    # -- Status / Info -----------------------------------------------------------

    def status(self):
        """Return structured snapshot of current session state.

        Returns dict with tempo, playing, time_signature, and per-track info
        (name, volume, pan, muted, soloed, devices, playing_slot).
        """
        tempo = self.api.song.get("tempo")
        playing = self.api.song.get("is_playing")
        sig_num = self.api.song.get("signature_numerator")
        sig_den = self.api.song.get("signature_denominator")
        num_tracks = self.api.song.get("num_tracks")

        tracks = []
        for i in range(num_tracks):
            name = self.api.track(i).get("name")
            vol = self.api.track(i).get("volume")
            pan = self.api.track(i).get("panning")
            muted = self.api.track(i).get("mute")
            soloed = self.api.track(i).get("solo")
            num_devs = self.api.track(i).get("num_devices")
            devices = []
            for d in range(num_devs):
                try:
                    dev_name = self.api.device(i, d).get("name")
                    devices.append(dev_name)
                except TimeoutError:
                    break
            playing_slot = None
            try:
                playing_slot = self.api.track(i).get("playing_slot_index")
            except TimeoutError:
                pass
            tracks.append(
                {
                    "index": i,
                    "name": name,
                    "volume": vol,
                    "pan": pan,
                    "muted": muted,
                    "soloed": soloed,
                    "devices": devices,
                    "playing_slot": playing_slot,
                }
            )

        return {
            "tempo": tempo,
            "playing": playing,
            "time_signature": f"{sig_num}/{sig_den}",
            "tracks": tracks,
        }

    def info(self):
        """Pretty-print current session state. Returns the status() dict."""
        s = self.status()
        print("--- Song Info ---")
        print(f"  Tempo:    {s['tempo']} BPM")
        print(f"  Time sig: {s['time_signature']}")
        print(f"  Playing:  {s['playing']}")
        print(f"\n  Tracks ({len(s['tracks'])}):")
        for t in s["tracks"]:
            devs = ", ".join(t["devices"]) if t["devices"] else "no devices"
            slot = (
                f" [playing slot {t['playing_slot']}]"
                if t["playing_slot"] not in (None, -1)
                else ""
            )
            print(
                f"    [{t['index']}] {t['name']} vol={t['volume']:.2f} pan={t['pan']:.2f} ({devs}){slot}"
            )
        return s

    # -- Audio Export ------------------------------------------------------------

    def _create_capture_track(self):
        """Create and configure the audio capture track at the end of the track list.

        Sets up Resampling input, Sends Only output, Monitor In, and arms it.
        Stores the track index in self._capture_track_idx.
        """
        num_tracks = self.api.song.get("num_tracks")
        self.api.song.call("create_audio_track", num_tracks)
        time.sleep(0.5)

        capture_idx = num_tracks
        self._capture_track_idx = capture_idx
        self.api.track(capture_idx).set("name", "_capture")
        time.sleep(0.2)

        # Route: Resampling in, Sends Only out, Monitor In
        self.api.track(capture_idx).set("input_routing_type", "Resampling")
        time.sleep(0.3)
        output_types = list(
            self.api.track(capture_idx).query("get/available_output_routing_types")
            or []
        )
        sends_name = next((n for n in output_types if "sends" in n.lower()), None)
        if sends_name:
            self.api.track(capture_idx).set("output_routing_type", sends_name)
            time.sleep(0.3)
        self.api.track(capture_idx).set("current_monitoring_state", 0)
        time.sleep(0.2)

        self.api.track(capture_idx).set("arm", 1)
        time.sleep(0.3)

        # Verify routing was applied correctly
        try:
            routing = self.api.track(capture_idx).get("input_routing_type")
            monitoring = self.api.track(capture_idx).get("current_monitoring_state")
            armed = self.api.track(capture_idx).get("arm")
            print(
                f"  [export] Capture track verified: routing={routing}, monitor={monitoring}, arm={armed}"
            )
            if "resampling" not in str(routing).lower():
                print(
                    f"  [export] WARNING: routing is '{routing}', expected 'Resampling'"
                )
            if monitoring != 0:
                print(
                    f"  [export] WARNING: monitoring is {monitoring}, expected 0 (In)"
                )
            if armed != 1:
                print(f"  [export] WARNING: arm is {armed}, expected 1")
        except TimeoutError:
            print("  [export] WARNING: Could not verify capture track routing")

    def _ensure_recording_stopped(self):
        """Set session_record=0 and verify status reaches 0."""
        self.api.song.set("session_record", 0)
        for _ in range(8):
            time.sleep(0.5)
            try:
                status = self.api.song.get("session_record_status")
                if status == 0:
                    return
            except TimeoutError:
                continue
        print("  [export] WARNING: session_record_status not 0 after disabling")

    def _trigger_recording(self):
        """Start session recording using direct setter and verify it's active.

        Uses set/session_record=1 (direct) instead of trigger_session_record
        (toggle) to avoid ambiguous state when toggling mid-transition.
        session_record_status: 0=off, 1=transition, 2=recording.
        """
        self.api.song.set("session_record", 1)

        # Poll up to 8s for status=2
        status = 0
        for i in range(16):
            time.sleep(0.5)
            try:
                status = self.api.song.get("session_record_status")
            except TimeoutError:
                continue
            if status == 2:
                print("  [export] Recording active (status=2)")
                return
            if status == 0 and i >= 4:
                # Fell back to off — re-set
                print(
                    f"  [export] Recording status=0 after {(i + 1) * 0.5:.1f}s, re-setting..."
                )
                self.api.song.set("session_record", 1)

        print(f"  [export] WARNING: Recording status={status} after polling (wanted 2)")

    def _maybe_start_deferred_recording(self):
        """Start session recording if it was deferred (waiting for transport)."""
        if getattr(self, "_record_pending", False):
            # Verify transport is actually running before triggering
            time.sleep(2.0)
            try:
                playing = self.api.song.get("is_playing")
                if not playing:
                    print("  [export] Transport not running yet, waiting longer...")
                    time.sleep(3.0)
            except TimeoutError:
                time.sleep(2.0)
            self._trigger_recording()
            self._record_pending = False

    def start_recording(self):
        """Create a capture track and begin session recording.

        Creates an audio track with Resampling input (captures master output),
        Sends Only output (prevents feedback), arms it, and triggers session
        recording. Call stop_recording() after playback to retrieve the WAV.

        The capture track is automatically restored if setup() is called
        during recording (e.g. by setup_session in the compose pipeline).
        """
        self._recording = True
        self._record_pending = False
        self._create_capture_track()
        self._trigger_recording()

    def stop_recording(self, export_dir="exports") -> str | None:
        """Stop recording, find the captured WAV, copy to export_dir.

        Returns the path to the exported file, or None if no clip was found.
        """
        capture_idx = getattr(self, "_capture_track_idx", None)
        if capture_idx is None:
            print("  [export] No active recording to stop")
            return None

        # Stop playback (stops session recording too)
        self.api.song.call("stop_playing")
        time.sleep(1.0)

        # Collect ALL recorded clips from the capture track.
        # Session recording creates a new clip per scene fire, so audio
        # is spread across multiple slots. Concatenate them in order.
        clip_paths = []
        num_scenes = self.api.song.get("num_scenes")
        for slot in range(num_scenes):
            try:
                has = self.api.clip_slot(capture_idx, slot).query("get/has_clip")
                if has and has[0]:
                    result = self.api.clip(capture_idx, slot).query("get/file_path")
                    fp = result[0] if result else None
                    if fp and os.path.isfile(fp):
                        clip_paths.append(fp)
                        print(f"  [export] Found clip slot {slot}: {fp}")
            except Exception:
                pass

        # Concatenate clips into a single WAV, then normalize
        export_path = None
        if clip_paths:
            os.makedirs(export_dir, exist_ok=True)
            dest = os.path.join(export_dir, f"capture_{int(time.time())}.wav")
            if len(clip_paths) == 1:
                shutil.copy2(clip_paths[0], dest)
            else:
                _concatenate_wavs(clip_paths, dest)
            _normalize_wav(dest)
            export_path = dest
            size_kb = os.path.getsize(dest) / 1024
            duration = _wav_duration(dest)
            print(
                f"  [export] Saved: {dest} ({size_kb:.1f} KB, {duration:.1f}s, {len(clip_paths)} clips merged)"
            )
        else:
            print("  [export] WARNING: No recorded clips found on capture track")

        # Cleanup: disarm and delete capture track
        self._recording = False
        self.api.track(capture_idx).set("arm", 0)
        time.sleep(0.2)
        self.api.song.call("delete_track", capture_idx)
        time.sleep(0.3)
        self._capture_track_idx = None

        return export_path

    # -- Lifecycle ---------------------------------------------------------------

    def close(self):
        self.api.stop()


@dataclass
class Track:
    """Declares a track's instrument, effects, and mix."""

    name: str
    sound: str | None = None
    instrument: str | None = None
    drum_kit: str | None = None
    effects: list[str] = field(default_factory=list)
    volume: float = 0.85
    pan: float = 0.0
