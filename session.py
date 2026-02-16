"""bergain session — high-level DJ REPL and composition API.

Session wraps LiveAPI with track-name resolution, device-param caching,
and convenience methods for the live DJ workflow.

Set / Track / Scene provide the composition workflow for building
arrangements and session clips.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from main import LiveAPI, REMOTE_PORT, LOCAL_PORT


def _notes_to_wire(tuples):
    """Convert [(pitch, start, dur, vel), ...] to flat wire format for OSC."""
    flat = []
    for p, s, d, v in tuples:
        flat.extend([p, float(s), float(d), v, 0])
    return flat


def make_automation(changes):
    """Create an on_enter hook that sets device parameters.

    changes: list of (track_name, device_index, param_name, value, description)
    """

    def on_enter(session):
        for track_name, device, param_name, val, desc in changes:
            session.param(track_name, device, param_name, val)
            print(f"      [{desc}]")

    return on_enter


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
            pass
        try:
            num_scenes = self.api.song.get("num_scenes")
            for i in range(num_scenes):
                name = self.api.scene(i).get("name")
                self._scenes[name] = i
        except TimeoutError:
            pass

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
        """Fire a scene by name or index."""
        self.api.scene(self._s(scene)).call("fire")

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
        return self.api.device(t, device).get("parameter/value", idx)

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
            for i in range(0, len(results), 2)
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

    def _browser_load(self, track, method, name, label, wait=1.0):
        """Load a browser item onto a track, printing success/failure."""
        t = self._t(track)
        self.api.view.set("selected_track", t)
        time.sleep(0.1)
        try:
            result = (
                self.api.browser.query(method, name)
                if name
                else self.api.browser.query(method)
            )
            loaded = result[0] if result else name
            time.sleep(wait)
            print(f"  Track {t}: loaded {label} '{loaded}'")
            return loaded
        except TimeoutError:
            print(f"  Track {t}: FAILED to load {label} '{name}'")
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

        # Set mix
        for i, t in enumerate(tracks):
            self.api.track(i).set("volume", t.volume)
            self.api.track(i).set("panning", t.pan)

        self.refresh()
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

    # -- Lifecycle ---------------------------------------------------------------

    def close(self):
        self.api.stop()


# ---------------------------------------------------------------------------
# Composition workflow: Set / Track / Scene
# ---------------------------------------------------------------------------


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


@dataclass
class Scene:
    """A row of clips across tracks."""

    name: str
    bars: int
    clips: dict[str, Callable]
    on_enter: Callable | None = None


def _format_duration(seconds):
    """Format seconds as 'N min' or 'Ns'."""
    if seconds >= 120:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f}s"


class Set:
    """A persistent session with tracks/effects/mix — Ableton's 'Live Set'.

    Usage:
        s = Set("My Set", bpm=120, tracks=[...])
        s.setup()
        s.load_scenes(scenes)
        s.build_arrangement(scenes)
        s.play()
        s.teardown()
    """

    def __init__(
        self,
        name,
        bpm,
        tracks,
        beats_per_bar=4,
        groove=0.0,
        metronome=False,
        time_sig=None,
        subtitle=None,
        api=None,
    ):
        self.name = name
        self.subtitle = subtitle
        self.bpm = bpm
        self.tracks = tracks
        self.beats_per_bar = beats_per_bar
        self.groove = groove
        self.metronome = metronome
        self.time_sig = time_sig
        self._api = api
        self.session = None
        self._scenes = []

    def setup(self):
        """Connect to Ableton, create tracks, load instruments/effects, set mix."""
        self.session = Session(api=self._api)
        self.session.stop()
        time.sleep(0.2)

        print(f"=== {self.name.upper()} ===")
        if self.subtitle:
            print(f"=== {self.subtitle} ===")
        print()

        self.session.tempo(self.bpm)
        self.session.api.song.set("groove_amount", self.groove)
        self.session.api.song.set("metronome", self.metronome)

        if self.time_sig:
            try:
                self.session.api.song.set("signature_numerator", self.time_sig[0])
                self.session.api.song.set("signature_denominator", self.time_sig[1])
            except Exception:
                pass

        sig = f", {self.time_sig[0]}/{self.time_sig[1]}" if self.time_sig else ""
        print(f"  {self.bpm} BPM{sig}\n")

        self.session.setup(self.tracks)

    def load_scenes(self, scenes):
        """Create session clips from scenes."""
        self._scenes = scenes
        print("\n  Session clips:")
        for scene_idx, scene in enumerate(scenes):
            clip_beats = scene.bars * self.beats_per_bar
            for track_name, fn in scene.clips.items():
                clip_name = f"{scene.name} {track_name}"
                self.session.clip(
                    track_name, scene_idx, clip_beats, fn(scene.bars), clip_name
                )
            self.session.api.scene(scene_idx).set("name", scene.name)

    def build_arrangement(self, scenes):
        """Create arrangement clips from scenes, laid out linearly."""
        self._scenes = scenes
        self.session.clear_arrangement()
        total_bars = sum(s.bars for s in scenes)
        total_secs = total_bars * self.beats_per_bar * 60.0 / self.bpm
        print(f"\n  Arrangement ({total_bars} bars, {_format_duration(total_secs)}):")

        beat_offset = 0
        for scene in scenes:
            sec_beats = scene.bars * self.beats_per_bar
            bar_start = beat_offset // self.beats_per_bar + 1
            bar_end = bar_start + scene.bars - 1
            active = " ".join(
                "X" if t.name in scene.clips else "." for t in self.tracks
            )
            print(f"    {scene.name:20s} bars {bar_start:3d}-{bar_end:3d}  [{active}]")

            for track_name, fn in scene.clips.items():
                clip_name = f"{scene.name} {track_name}"
                self.session.arr_clip(
                    track_name, beat_offset, sec_beats, fn(scene.bars), clip_name
                )
            beat_offset += sec_beats

        self.session.arrangement()

    def play(self, scenes=None):
        """Play arrangement linearly, calling on_enter hooks per scene."""
        scenes = scenes or self._scenes
        bar_dur = self.beats_per_bar * 60.0 / self.bpm
        total_bars = sum(s.bars for s in scenes)
        total_secs = total_bars * bar_dur
        print(f"\n  Playing... ({_format_duration(total_secs)})")
        self.session.seek(0)
        time.sleep(0.1)
        self.session.play()

        try:
            for scene in scenes:
                if scene.on_enter:
                    scene.on_enter(self.session)
                secs = scene.bars * bar_dur
                print(
                    f"    >>> {scene.name} ({scene.bars} bars, {_format_duration(secs)})"
                )
                time.sleep(secs)
        except KeyboardInterrupt:
            pass

        self.session.stop()

    def teardown(self):
        """Stop playback + disconnect."""
        self.session.stop()
        self.session.close()
