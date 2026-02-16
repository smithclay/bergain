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

    def __init__(self, hostname="127.0.0.1", port=REMOTE_PORT, client_port=LOCAL_PORT):
        self.api = LiveAPI(hostname, port, client_port)
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
        """Create a session MIDI clip from note tuples."""
        t = self._t(track)
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
        print(
            f"  Track {t}, slot {slot}: '{name}' ({length} beats, {len(tuples)} notes)"
        )

    def arr_clip(self, track, start, length, tuples, name=""):
        """Create an arrangement clip from note tuples. Returns clip index."""
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
        return clip_idx

    # -- Scene -------------------------------------------------------------------

    def fire(self, scene):
        """Fire a scene by name or index."""
        self.api.scene(self._s(scene)).call("fire")

    # -- Device params -----------------------------------------------------------

    def _get_params(self, t, device):
        result = self.api.device(t, device).query("get/parameters/name", timeout=2.0)
        return list(result)

    def params(self, track, device):
        """Get parameter names for a device (cached)."""
        t = self._t(track)
        key = (t, device)
        if key not in self._param_cache:
            self._param_cache[key] = self._get_params(t, device)
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
        self._browser_load(track, "load_instrument", name, "instrument")

    def load_drum_kit(self, track, name=None):
        """Load a drum kit onto a track."""
        self._browser_load(track, "load_drum_kit", name, "drum kit")

    def load_sound(self, track, name):
        """Load a sound preset onto a track."""
        self._browser_load(track, "load_sound", name, "sound")

    def load_effect(self, track, name):
        """Load an audio effect onto a track."""
        self._browser_load(track, "load_audio_effect", name, "effect", wait=0.5)

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

    def print_arrangement(self):
        """Print the full arrangement layout."""
        print("\n  Arrangement layout:")
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
                    print(f"    {track_name:8s} | bar {bar:3d} | '{cn}'")
            except TimeoutError:
                pass

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

    # -- Info --------------------------------------------------------------------

    def info(self):
        """Print current song state."""
        print("--- Song Info ---")
        print(f"  Tempo:    {self.api.song.get('tempo')} BPM")
        print(f"  Tracks:   {self.api.song.get('num_tracks')}")
        print(f"  Scenes:   {self.api.song.get('num_scenes')}")
        sig_num = self.api.song.get("signature_numerator")
        sig_den = self.api.song.get("signature_denominator")
        print(f"  Time sig: {sig_num}/{sig_den}")
        num_tracks = self.api.song.get("num_tracks")
        print("\n  Tracks:")
        for t in range(num_tracks):
            name = self.api.track(t).get("name")
            devs = self.api.track(t).get("num_devices")
            midi = self.api.track(t).get("has_midi_input")
            print(f"    [{t}] {name} ({'MIDI' if midi else 'Audio'}, {devs} devices)")
        print("\n  Arrangement:")
        has_clips = False
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
                    print(f"    {track_name:8s} | bar {bar:3d} | '{cn}'")
                    has_clips = True
            except TimeoutError:
                pass
        if not has_clips:
            print("    (empty)")

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
    ):
        self.name = name
        self.subtitle = subtitle
        self.bpm = bpm
        self.tracks = tracks
        self.beats_per_bar = beats_per_bar
        self.groove = groove
        self.metronome = metronome
        self.time_sig = time_sig
        self.session = None
        self._scenes = []

    def setup(self):
        """Connect to Ableton, create tracks, load instruments/effects, set mix."""
        self.session = Session()
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

        self.session.print_arrangement()

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
