"""DSPy tool closures for the RLM Composer.

make_tools() creates tool functions closed over a Session instance.
All tools accept simple types and return str (DSPy PythonInterpreter requirement).
"""

import functools
import json

from .music import STYLE_KEYS, _clip_name_for_role, render_role


def make_tools(
    session,
    min_clips=6,
    progress=None,
):
    """Create tool closures over a live Session, with milestone tracking.

    Returns (tools_list, tools_by_name, live_history, track_roles, palette_scenes) where:
      - tools_list: list of callables for DSPy RLM
      - tools_by_name: dict mapping function name -> callable
      - live_history: list (empty, kept for API compat)
      - track_roles: dict mapping track name -> role string
      - palette_scenes: list of scene dicts captured by write_clip
    """

    # Milestone tracker — shared mutable state across all tool closures.
    # ready_to_submit() checks these before allowing SUBMIT.
    _done = {
        "browse": False,
        "tracks": False,
        "clips": 0,
        "mix": False,
        "status_checks": 0,
    }

    # Track role map: {"TrackName": "drums"|"bass"|"pad"} — set by create_tracks()
    _track_roles = {}

    # Palette scene data — captured by write_clip for evolution overlay
    _palette_scenes = []

    def _json_tool(fn):
        """Wrap a tool function to catch exceptions and return JSON error."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                if progress:
                    import time as _time

                    progress.stream.append(
                        {
                            "type": "result",
                            "content": f"{fn.__name__}: {result[:200] if isinstance(result, str) else result}",
                            "timestamp": _time.time(),
                        }
                    )
                return result
            except Exception as e:
                import traceback

                print(f"  [TOOL ERROR] {fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
                if progress:
                    import time as _time

                    progress.stream.append(
                        {
                            "type": "error",
                            "content": f"{fn.__name__}: {e}",
                            "timestamp": _time.time(),
                        }
                    )
                return json.dumps({"error": str(e)})

        return wrapper

    @_json_tool
    def set_tempo(bpm: int) -> str:
        """Set the project tempo in BPM."""
        session.tempo(int(bpm))
        return json.dumps({"tempo": int(bpm)})

    def get_status() -> str:
        """Get current DAW state as plain text. Returns tempo, playing status, and track list."""
        try:
            s = session.status()
            _done["status_checks"] += 1  # count only successful checks
            lines = [
                f"Tempo: {s['tempo']} BPM",
                f"Playing: {'yes' if s.get('playing') else 'no'}",
            ]
            tracks = s.get("tracks", [])
            if tracks:
                lines.append("Tracks:")
                for t in tracks:
                    parts = [f"  {t['index']}: {t['name']}"]
                    if t.get("devices"):
                        parts.append(f"[{', '.join(t['devices'])}]")
                    parts.append(f"vol={t.get('volume', 0.85):.2f}")
                    parts.append(f"pan={t.get('pan', 0.0):.2f}")
                    lines.append(" ".join(parts))
            else:
                lines.append("Tracks: (none)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @_json_tool
    def create_tracks(tracks_json: str) -> str:
        """Create MIDI tracks. REPLACES ALL existing tracks — call ONCE with full layout.

        Args:
            tracks_json: JSON array of track specs, each with:
              name (required), sound/instrument/drum_kit (pick one to auto-load),
              effects (list of effect names), volume (0.0-1.0), pan (-1.0 to 1.0)

        Example: json.dumps([
            {"name": "Drums", "drum_kit": "909 Core Kit.adg", "volume": 0.9},
            {"name": "Bass", "instrument": "Operator", "volume": 0.85},
            {"name": "Pad", "sound": "Drifting Ambient Pad.adv", "volume": 0.8}
        ])
        """
        from .session import Track

        _done["tracks"] = True
        if progress:
            progress.tracks_done = True
        specs = json.loads(tracks_json)
        _track_roles.clear()
        tracks = []
        for s in specs:
            tracks.append(
                Track(
                    name=s["name"],
                    sound=s.get("sound"),
                    instrument=s.get("instrument"),
                    drum_kit=s.get("drum_kit"),
                    effects=s.get("effects", []),
                    volume=s.get("volume", 0.85),
                    pan=s.get("pan", 0.0),
                )
            )
            # Auto-detect track role for write_clip()
            # Only drums/bass/pad roles get clips — extra tracks (lead, fx, etc.)
            # are left unassigned so write_clip() skips them.
            name_lower = s["name"].lower()
            if s.get("drum_kit"):
                _track_roles[s["name"]] = "drums"
            elif "bass" in name_lower:
                _track_roles[s["name"]] = "bass"
            elif any(
                kw in name_lower for kw in ("pad", "chord", "keys", "synth", "string")
            ):
                _track_roles[s["name"]] = "pad"
            elif any(kw in name_lower for kw in ("stab", "lead")):
                _track_roles[s["name"]] = "stab"
            elif any(kw in name_lower for kw in ("texture", "fx", "noise", "atmo")):
                _track_roles[s["name"]] = "texture"

        # Fallback: if no pad track was explicitly matched, assign the first
        # unmatched track so write_clip() has somewhere to put chords/pads.
        if "pad" not in _track_roles.values():
            for s in specs:
                if s["name"] not in _track_roles:
                    _track_roles[s["name"]] = "pad"
                    break
        count = session.setup(tracks)
        return json.dumps({"tracks_created": count, "names": [t.name for t in tracks]})

    @_json_tool
    def browse(query: str, category: str = "") -> str:
        """Search Ableton browser for instruments, sounds, drums, or effects.
        Returns matching names as plain text, one per line (NOT JSON).

        Args:
            query: Search term (e.g. "909", "strings", "Operator")
            category: Optional filter — 'Instruments', 'Sounds', 'Drums', 'Audio Effects'

        Returns NO_RESULTS if nothing matches — try broader terms.
        Example: browse("909", "Drums") might return "909 Core Kit.adg\\nClap 909.aif\\n..."
        Parse with: names = result.split('\\n')

        IMPORTANT:
          - For drum kits, pick .adg files (drum racks), NOT .wav/.aif (individual samples).
            A sample like "Clap 909.aif" is a single hit, not a playable kit.
          - Native Live devices (EQ Eight, Saturator, Auto Filter, etc.) do NOT appear
            in search results. Load them by exact name with load_effect() — no browse needed.
        """
        _done["browse"] = True
        if progress:
            progress.browse_done = True
        results = session.browse(query, category=category or None)
        names = [r["name"] for r in results[:20]]
        return "\n".join(names) if names else "NO_RESULTS"

    def _make_loader(method_name, label):
        """Generate a load_* tool from a session method name and display label."""

        @_json_tool
        def loader(track: str, name: str) -> str:
            result = getattr(session, method_name)(track, name)
            if result is None:
                return json.dumps(
                    {
                        "error": f"{label} '{name}' not found or timed out",
                        "track": track,
                    }
                )
            return json.dumps(
                {
                    "loaded": label.lower().replace(" ", "_"),
                    "name": result,
                    "track": track,
                }
            )

        loader.__name__ = method_name
        loader.__qualname__ = method_name
        loader.__doc__ = f'Load a {label.lower()} by name onto a track. Returns {{"error": ...}} if not found.'
        return loader

    load_instrument = _make_loader("load_instrument", "Instrument")
    load_sound = _make_loader("load_sound", "Sound")
    load_effect = _make_loader("load_effect", "Effect")
    load_drum_kit = _make_loader("load_drum_kit", "Drum kit")

    @_json_tool
    def get_params(track: str, device_index: int) -> str:
        """List parameter names for a device. Device 0 = instrument, 1+ = effects."""
        params = session.params(track, int(device_index))
        return json.dumps(
            {"track": track, "device": int(device_index), "params": params}
        )

    @_json_tool
    def set_param(track: str, device_index: int, param: str, value: float) -> str:
        """Set a device parameter by name fragment (case-insensitive match)."""
        session.param(track, int(device_index), param, float(value))
        return json.dumps(
            {
                "track": track,
                "device": int(device_index),
                "param": param,
                "value": float(value),
            }
        )

    @_json_tool
    def set_mix(levels_json: str) -> str:
        """Set volume/pan for tracks. levels_json: {"TrackName": volume} or {"TrackName": {"volume": v, "pan": p}}.

        volume: 0.0-1.0 (0.85 default). pan: -1.0 (left) to 1.0 (right).
        """
        _done["mix"] = True
        if progress:
            progress.mix_done = True
        levels = json.loads(levels_json)
        session.mix(**levels)
        return json.dumps({"mixed": list(levels.keys())})

    @_json_tool
    def play() -> str:
        """Start playback."""
        session.play()
        return json.dumps({"playing": True})

    @_json_tool
    def stop() -> str:
        """Stop playback."""
        session.stop()
        return json.dumps({"playing": False})

    @_json_tool
    def fire_clip(track: str, slot: int) -> str:
        """Fire a specific clip by track name and slot index."""
        session.fire_clip(track, int(slot))
        return json.dumps({"fired": track, "slot": int(slot)})

    @_json_tool
    def fire_scene(slot: int) -> str:
        """Fire a scene (all clips in a row) by slot index. Use for live transitions."""
        session.fire(int(slot))
        return json.dumps({"fired_scene": int(slot)})

    @_json_tool
    def write_clip(clip_json: str) -> str:
        """Write a looping clip to the session grid.

        Args:
            clip_json: JSON object with:
              name (str): Clip name, e.g. "Main Groove", "Breakdown Pad"
              slot (int): Scene/row index (0, 1, 2, ...) — each scene is a
                          different energy level or section type
              bars (int): Loop length in bars (typically 4 or 8)
              energy (float): 0.0 (silent/sparse) to 1.0 (full power)
              chords (list[str]): Chord names, e.g. ["Fm", "Cm7"]
              key (str): Key root note, e.g. "F"
              drums (str): "four_on_floor"|"half_time"|"breakbeat"|"minimal"|"shuffle"|"sparse_perc"|"none"
              bass (str): "rolling_16th"|"offbeat_8th"|"pulsing_8th"|"sustained"|"walking"|"none"
              pad (str): "sustained"|"atmospheric"|"pulsing"|"arpeggiated"|"swells"|"none"
              stab (str): "sustained"|"atmospheric"|"pulsing"|"arpeggiated"|"swells"|"none"
              texture (str): "sustained"|"atmospheric"|"pulsing"|"arpeggiated"|"swells"|"none"

        Creates looping session clips for each track role. Clips in the same
        slot (scene) can be fired together for instant transitions.
        Returns JSON summary of clips created.
        """
        c = json.loads(clip_json)
        name = c["name"]
        slot = int(c["slot"])
        bars = int(c["bars"])
        energy = max(0.0, min(1.0, float(c["energy"])))
        chords = c.get("chords", [])
        key_root = c.get("key", "C")
        length_beats = float(bars * 4)

        clips_created = 0
        details = []

        for track_name, role in _track_roles.items():
            style = c.get(role, "none")
            notes = render_role(role, style, bars, energy, chords, key_root, name)

            if notes:
                clip_name = _clip_name_for_role(role, name)
                session.clip(track_name, slot, length_beats, notes, name=clip_name)
                clips_created += 1
                details.append(f"{clip_name}: {len(notes)} notes")

        _done["clips"] += clips_created
        if progress:
            progress.clips_created = _done["clips"]

        # Capture scene data for evolution overlay
        scene = {
            "name": name,
            "slot": slot,
            "energy": energy,
            "key": key_root,
            "chords": chords,
            "bars": bars,
        }
        for key in STYLE_KEYS:
            scene[key] = c.get(key, "none")
        _palette_scenes.append(scene)

        return json.dumps(
            {
                "clip": name,
                "slot": slot,
                "clips_created": clips_created,
                "bars": bars,
                "energy": energy,
                "details": details,
            }
        )

    def ready_to_submit() -> str:
        """Check if composition is complete enough to SUBMIT.

        Call in a SEPARATE step BEFORE the step where you SUBMIT.
        Do NOT call ready_to_submit() and SUBMIT() in the same code block.
        Returns 'READY' if all milestones are met, or lists what's still missing.
        """
        issues = []
        if not _done["browse"]:
            issues.append("browse() not called — search for instruments first")
        if not _done["tracks"]:
            issues.append("create_tracks() not called — set up your tracks")
        if _done["clips"] < min_clips:
            issues.append(
                f"only {_done['clips']} clips created (need at least {min_clips}) — build more scenes"
            )
        if not _done["mix"]:
            issues.append("set_mix() not called — balance volumes and panning")
        if _done["status_checks"] < 1:
            issues.append(
                "get_status() never called — verify your work before submitting"
            )
        if issues:
            return "NOT READY to submit:\n" + "\n".join(f"  - {i}" for i in issues)
        return "READY — call SUBMIT(report) in the NEXT step (no other tool calls)."

    tools = [
        set_tempo,
        get_status,
        create_tracks,
        browse,
        load_instrument,
        load_sound,
        load_effect,
        load_drum_kit,
        get_params,
        set_param,
        set_mix,
        play,
        stop,
        fire_clip,
        fire_scene,
        write_clip,
        ready_to_submit,
    ]

    tools_by_name = {t.__name__: t for t in tools}
    return tools, tools_by_name, [], _track_roles, _palette_scenes
