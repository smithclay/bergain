"""DSPy tool closures for the RLM Composer.

make_tools() creates tool functions closed over a Session instance.
All tools accept simple types and return str (DSPy PythonInterpreter requirement).
"""

import functools
import json
import re
import time

from .music import clamp_note, render_drums, render_bass, render_pads


def make_tools(
    session, min_clips=6, live_mode=False, duration_minutes=60, sub_lm=None, brief=""
):
    """Create tool closures over a live Session, with milestone tracking.

    Returns (tools_list, tools_by_name, live_history) where:
      - tools_list: list of callables for DSPy RLM
      - tools_by_name: dict mapping function name -> callable
      - live_history: list of section dicts (mutated by compose_next)
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

    # Live mode state — tracks elapsed time for wait/elapsed/ready_to_submit
    _live_state = {"start_time": None, "target_duration": duration_minutes}

    # Live mode history — compose_next() maintains this automatically
    _live_history = []

    def _json_tool(fn):
        """Wrap a tool function to catch exceptions and return JSON error."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                import traceback

                print(f"  [TOOL ERROR] {fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
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
        levels = json.loads(levels_json)
        session.mix(**levels)
        return json.dumps({"mixed": list(levels.keys())})

    @_json_tool
    def play() -> str:
        """Start playback."""
        session.play()
        if live_mode and _live_state["start_time"] is None:
            _live_state["start_time"] = time.time()
        return json.dumps({"playing": True})

    # Known-good single-keyword fallbacks per role.
    # Multi-word descriptive queries fail Ableton's browser consistently;
    # these single keywords always return results.
    _ROLE_SEARCH_FALLBACKS = {
        "drums": [("Core Kit", "Drums"), ("909", "Drums"), ("kit", "Drums")],
        "bass": [("acid bass", "Sounds"), ("sub bass", "Sounds"), ("bass", "Sounds")],
        "pad": [("pad", "Sounds")],
        "stab": [("stab", "Sounds"), ("pluck", "Sounds"), ("chord", "Sounds")],
        "texture": [("noise", "Sounds"), ("texture", "Sounds"), ("metal", "Sounds")],
    }

    def _browse_cascade(search_term, role):
        """Cascade through search attempts until something works.

        Order: full search_term → individual tokens → role fallbacks.
        Returns (sound, drum_kit) — one will be set, or both None.
        """
        is_drums = role == "drums"
        category = "Drums" if is_drums else "Sounds"

        def _pick(names):
            """Pick best match from a list of result names."""
            if is_drums:
                adg = next((n for n in names if n.endswith(".adg")), None)
                return (None, adg or names[0]) if names else (None, None)
            else:
                adv = next((n for n in names if n.endswith(".adv")), None)
                adg = next((n for n in names if n.endswith(".adg")), None)
                return (adv or adg or (names[0] if names else None), None)

        # 1. Try the full search term
        if search_term:
            results = session.browse(search_term, category=category)
            names = [r["name"] for r in results[:20]]
            if names:
                sound, drum_kit = _pick(names)
                if sound or drum_kit:
                    return sound, drum_kit, f"'{search_term}'"

            # 2. Tokenize and try each word individually
            tokens = search_term.split()
            if len(tokens) > 1:
                for token in tokens:
                    results = session.browse(token, category=category)
                    names = [r["name"] for r in results[:20]]
                    if names:
                        sound, drum_kit = _pick(names)
                        if sound or drum_kit:
                            return sound, drum_kit, f"token '{token}'"

        # 3. Role-based fallbacks (known-good terms)
        for fallback_term, fallback_cat in _ROLE_SEARCH_FALLBACKS.get(role, []):
            results = session.browse(fallback_term, category=fallback_cat)
            names = [r["name"] for r in results[:20]]
            if names:
                sound, drum_kit = _pick(names)
                if sound or drum_kit:
                    return sound, drum_kit, f"fallback '{fallback_term}'"

        return None, None, "no results"

    # Default effects per track role
    _DEFAULT_EFFECTS = {
        "drums": ["Drum Buss", "EQ Eight"],
        "bass": ["Saturator", "EQ Eight"],
        "pad": ["Auto Filter", "Reverb"],
        "stab": ["Saturator", "Auto Filter"],
        "texture": ["Reverb", "Auto Filter"],
    }

    @_json_tool
    def setup_session(config_json: str) -> str:
        """Set up the entire session in one call: create tracks, load sounds, set tempo, start playing.

        Uses cascading search: tries your search term, then individual words,
        then known-good fallbacks per role. Always creates every track — an
        empty MIDI track is better than no track.

        Args:
            config_json: JSON object with:
              tempo (int): BPM
              tracks (list): each with:
                name (str): Track name
                role (str): "drums"|"bass"|"pad"|"stab"|"texture"
                search (str, optional): Browser query — omit to use role defaults
                effects (list[str], optional): Effect names — defaults applied per role if omitted

        Example: json.dumps({
            "tempo": 130,
            "tracks": [
                {"name": "Drums", "role": "drums"},
                {"name": "Bass", "role": "bass"},
                {"name": "Pad", "role": "pad"},
                {"name": "Stab", "role": "stab"},
                {"name": "Texture", "role": "texture"}
            ]
        })

        ONE call replaces browse + create_tracks + set_tempo + play.
        """
        from .session import Track

        config = json.loads(config_json)
        tempo = config.get("tempo", 120)
        track_specs = config.get("tracks", [])

        tracks = []
        _track_roles.clear()
        load_log = []

        for spec in track_specs:
            name = spec["name"]
            role = spec.get("role", "pad")
            search_term = spec.get("search", "")
            effects = spec.get("effects", _DEFAULT_EFFECTS.get(role, []))

            # Cascading search: full term → tokens → role fallbacks
            sound, drum_kit, matched_via = _browse_cascade(search_term, role)

            if drum_kit:
                load_log.append(f"{name}: drum kit '{drum_kit}' via {matched_via}")
            elif sound:
                load_log.append(f"{name}: sound '{sound}' via {matched_via}")
            else:
                load_log.append(f"{name}: empty MIDI track (no results)")

            tracks.append(
                Track(
                    name=name,
                    sound=sound,
                    instrument=None,
                    drum_kit=drum_kit,
                    effects=effects,
                    volume=spec.get("volume", 0.85),
                    pan=spec.get("pan", 0.0),
                )
            )
            _track_roles[name] = role

        count = session.setup(tracks)
        session.tempo(int(tempo))
        session.play()

        _done["browse"] = True
        _done["tracks"] = True
        if _live_state["start_time"] is None:
            _live_state["start_time"] = time.time()

        return json.dumps(
            {
                "tracks_created": count,
                "tempo": tempo,
                "playing": True,
                "roles": {name: role for name, role in _track_roles.items()},
                "loaded": load_log,
            }
        )

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
        drum_style = c.get("drums", "none")
        bass_style = c.get("bass", "none")
        pad_style = c.get("pad", "none")
        stab_style = c.get("stab", "none")
        texture_style = c.get("texture", "none")
        length_beats = float(bars * 4)

        clips_created = 0
        details = []

        for track_name, role in _track_roles.items():
            notes = []
            clip_name = ""

            if role == "drums" and drum_style != "none":
                notes = render_drums(bars, energy, drum_style, section_name=name)
                clip_name = f"Drums {name}"
            elif role == "bass" and bass_style != "none":
                notes = render_bass(
                    bars, energy, bass_style, chords, key_root, section_name=name
                )
                clip_name = f"Bass {name}"
            elif role == "pad" and pad_style != "none":
                notes = render_pads(
                    bars, energy, pad_style, chords, key_root, section_name=name
                )
                clip_name = f"Pad {name}"
            elif role == "stab" and stab_style != "none":
                notes = render_pads(
                    bars,
                    energy,
                    stab_style,
                    chords,
                    key_root,
                    section_name=f"Stab {name}",
                )
                # Post-process: pitch +12, duration *0.5, velocity +15
                notes = [
                    (min(127, p + 12), s, d * 0.5, min(127, v + 15))
                    for p, s, d, v in notes
                ]
                clip_name = f"Stab {name}"
            elif role == "texture" and texture_style != "none":
                notes = render_pads(
                    bars,
                    energy,
                    texture_style,
                    chords,
                    key_root,
                    section_name=f"Texture {name}",
                )
                # Post-process: pitch -12 (floor 0), velocity -20
                notes = [(max(0, p - 12), s, d, max(1, v - 20)) for p, s, d, v in notes]
                clip_name = f"Texture {name}"

            if notes:
                clamped = [clamp_note(n) for n in notes]
                session.clip(track_name, slot, length_beats, clamped, name=clip_name)
                clips_created += 1
                details.append(f"{clip_name}: {len(clamped)} notes")

        _done["clips"] += clips_created
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

    def _get_elapsed():
        """Internal helper: return (elapsed_min, remaining_min) or None if not started."""
        if _live_state["start_time"] is None:
            return None
        elapsed_sec = time.time() - _live_state["start_time"]
        elapsed_min = elapsed_sec / 60.0
        remaining_min = max(0.0, _live_state["target_duration"] - elapsed_min)
        return elapsed_min, remaining_min

    @_json_tool
    def wait(bars: int) -> str:
        """Wait for a number of bars to play. Sleeps in real time based on current tempo.

        Args:
            bars: Number of bars to wait (e.g. 4 or 8)

        Use after fire_scene() to let a section play before composing the next one.
        Returns elapsed and remaining minutes.
        """
        bpm = session.status().get("tempo", 120)
        seconds = int(bars) * 4 * 60.0 / bpm
        time.sleep(seconds)
        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            return json.dumps(
                {
                    "waited_bars": int(bars),
                    "waited_seconds": round(seconds, 1),
                    "elapsed_minutes": round(elapsed_min, 1),
                    "remaining_minutes": round(remaining_min, 1),
                }
            )
        return json.dumps(
            {"waited_bars": int(bars), "waited_seconds": round(seconds, 1)}
        )

    def elapsed() -> str:
        """Get elapsed and remaining time for the live performance.

        Returns JSON with elapsed_minutes, remaining_minutes, target_minutes.
        Call periodically to know when to wrap up.
        """
        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            return json.dumps(
                {
                    "elapsed_minutes": round(elapsed_min, 1),
                    "remaining_minutes": round(remaining_min, 1),
                    "target_minutes": _live_state["target_duration"],
                }
            )
        return json.dumps(
            {
                "elapsed_minutes": 0,
                "remaining_minutes": _live_state["target_duration"],
                "target_minutes": _live_state["target_duration"],
                "note": "Timer starts on first play() call",
            }
        )

    # -------------------------------------------------------------------
    # compose_next — split into helpers for readability
    # -------------------------------------------------------------------

    # Keys safe to embed in sub-LM prompts (excludes sub_lm_prompt/sub_lm_raw
    # which would cause recursive prompt growth).
    _HISTORY_PROMPT_KEYS = (
        "section",
        "slot",
        "energy",
        "density",
        "key",
        "chords",
        "drums",
        "bass",
        "pad",
        "stab",
        "texture",
    )

    def _strip_history(entries):
        """Strip bulky debug fields from history entries before embedding in prompts."""
        return [{k: h[k] for k in _HISTORY_PROMPT_KEYS if k in h} for h in entries]

    def _build_compose_prompt(creative_prompt, next_slot):
        """Assemble the sub-LM prompt from history, arc phase, and direction."""
        # Compressed history: last 5 full entries + summary of earlier
        recent = _live_history[-5:]
        earlier_summary = ""
        if len(_live_history) > 5:
            earlier = _live_history[:-5]
            energies = [h["energy"] for h in earlier]
            keys_used = sorted(set(h.get("key", "?") for h in earlier))
            styles = set()
            for h in earlier:
                styles.add((h.get("drums", "?"), h.get("bass", "?"), h.get("pad", "?")))
            earlier_summary = (
                f"Earlier ({len(earlier)} sections): energy {min(energies):.2f}"
                f"-{max(energies):.2f}, keys: {'/'.join(keys_used)}, "
                f"{len(styles)} unique style combos.\n"
            )

        # Style combo frequency for variety enforcement
        combo_counts = {}
        for h in _live_history:
            combo = (
                h.get("drums", "none"),
                h.get("bass", "none"),
                h.get("pad", "none"),
                h.get("stab", "none"),
                h.get("texture", "none"),
            )
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        overused = ["/".join(c) for c, n in combo_counts.items() if n >= 3]
        variety_note = ""
        if overused:
            variety_note = (
                f"OVERUSED combos (pick something DIFFERENT): {', '.join(overused)}.\n"
            )

        # Chord progression variety enforcement
        chord_history = ["/".join(h.get("chords", [])) for h in _live_history]
        chord_counts = {}
        for prog in chord_history:
            chord_counts[prog] = chord_counts.get(prog, 0) + 1
        overused_chords = [p for p, c in chord_counts.items() if c >= 2]
        chord_variety_note = ""
        if overused_chords:
            chord_variety_note = (
                f"OVERUSED chord progressions (try something new): "
                f"{', '.join(overused_chords)}.\n"
            )

        # Arc phase guidance
        arc_phase = ""
        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            total = _live_state["target_duration"]
            if elapsed_min < total * 0.33:
                arc_phase = (
                    f"OPENING third ({elapsed_min:.0f}/{total} min). "
                    "Build gradually — energy should stay low-to-moderate (0.1-0.5). "
                    "8-16 bar loops let mood settle in.\n"
                )
            elif elapsed_min < total * 0.66:
                arc_phase = (
                    f"MIDDLE third ({elapsed_min:.0f}/{total} min). "
                    "Create waves — alternate between lifts and dips (0.3-0.8). "
                    "This is where the most variety belongs. "
                    "4-bar loops for peaks, 8-bar for dips.\n"
                )
            else:
                max_energy = max((h["energy"] for h in _live_history), default=0)
                if max_energy < 0.6:
                    arc_phase = (
                        f"FINAL third ({elapsed_min:.0f}/{total} min). "
                        "You HAVEN'T PEAKED YET — push energy to 0.6+ before "
                        "winding down. This is your last chance for a climax.\n"
                    )
                else:
                    arc_phase = (
                        f"FINAL third ({elapsed_min:.0f}/{total} min). "
                        "Begin winding down — energy should generally descend "
                        "(0.5-0.1). 16-bar loops for hypnotic fade, sparser textures.\n"
                    )

        # Breakdown nudge: if energy sustained high without a valley, suggest one
        breakdown_nudge = ""
        if len(_live_history) >= 3:
            recent_3 = _live_history[-3:]
            sustained_high = all(h["energy"] >= 0.6 for h in recent_3)
            recent_has_breakdown = any(
                h.get("bass") == "none" for h in _live_history[-4:]
            )
            if sustained_high and not recent_has_breakdown:
                breakdown_nudge = (
                    "Consider a breakdown — sustained high energy needs "
                    "a valley to rebuild from.\n"
                )

        return (
            f"You are composing the next section of a live performance.\n"
            f"Brief: {brief}\n"
            f"{arc_phase}"
            f"{earlier_summary}"
            f"Recent sections: {json.dumps(_strip_history(recent))}\n"
            f"{variety_note}"
            f"{chord_variety_note}"
            f"{breakdown_nudge}"
            f"Creative direction: {creative_prompt}\n\n"
            f"Return a single JSON object with these fields:\n"
            f"  name (str): evocative section name\n"
            f"  slot (int): use {next_slot}\n"
            f"  bars (int): 4, 8, 16, or 32\n"
            f"  energy (float): 0.0 to 1.0 — calibration: 0.2=ambient, "
            f"0.4=gentle groove, 0.6=driving, 0.8=peak power, 0.95=maximum\n"
            f"  key (str): root note like 'F', 'Eb', 'Ab'\n"
            f"  chords (list[str]): 2-4 chord names like 'Fm7', 'Abmaj9'\n"
            f"  drums: 'four_on_floor'|'half_time'|'breakbeat'|'minimal'|"
            f"'shuffle'|'sparse_perc'|'none'\n"
            f"  bass: 'rolling_16th'|'offbeat_8th'|'pulsing_8th'|'sustained'|"
            f"'walking'|'none'\n"
            f"  pad: 'sustained'|'atmospheric'|'pulsing'|'arpeggiated'|"
            f"'swells'|'none'\n"
            f"  stab: 'sustained'|'atmospheric'|'pulsing'|'arpeggiated'|"
            f"'swells'|'none'\n"
            f"  texture: 'sustained'|'atmospheric'|'pulsing'|'arpeggiated'|"
            f"'swells'|'none'\n\n"
            f"Make bold creative choices that serve the direction. "
            f"Respond with ONLY the JSON object."
        )

    def _execute_section(section):
        """Write clips, handle fade transitions, fire scene, wait.

        Returns (clip_info, wait_info) dicts.
        """
        clip_result = write_clip(json.dumps(section))
        clip_info = json.loads(clip_result)
        if "error" in clip_info:
            return clip_info, None

        # Transition fades: fade out tracks that go from active to "none"
        fade_tracks = {}  # track_name -> original volume to restore
        if _live_history:
            prev = _live_history[-1]
            for track_name, role in _track_roles.items():
                if role not in ("drums", "bass", "pad", "stab", "texture"):
                    continue
                was_active = prev.get(role, "none") != "none"
                now_silent = section.get(role, "none") == "none"
                if was_active and now_silent:
                    try:
                        t = session._t(track_name)
                        original_vol = session.api.track(t).get("volume")
                        session.fade(track_name, 0.0, steps=4, duration=0.5)
                        fade_tracks[track_name] = original_vol
                    except Exception as e:
                        print(f"  [FADE] Could not fade {track_name}: {e}")

        fire_scene(section["slot"])

        # Restore faded tracks to their prior volume after scene fires
        for track_name, original_vol in fade_tracks.items():
            try:
                session.fade(track_name, original_vol, steps=4, duration=0.5)
            except Exception as e:
                print(f"  [FADE] Could not fade {track_name}: {e}")

        wait_result = wait(section["bars"])
        wait_info = json.loads(wait_result)

        return clip_info, wait_info

    @_json_tool
    def compose_next(creative_prompt: str) -> str:
        """Compose and perform the next section in one step. The sub-LM makes
        all musical decisions — you provide creative DIRECTION, not parameters.

        Args:
            creative_prompt: Describe the FEELING and DIRECTION you want.
                Do NOT specify exact parameter values — let the sub-LM surprise you.
                Good: "Build tension — bring in a driving bass, keep drums sparse"
                Good: "Drop to a quiet interlude, just pads and atmosphere"
                Good: "Peak energy, all instruments full power, complex rhythms"
                Good: "Shift to a new key for freshness, walking bass, arpeggiated pad"
                Bad:  "energy 0.58, drums half_time, bass sustained, key Eb"

        Internally: calls sub-LM → writes clips → fires scene → waits.
        History is tracked and compressed automatically.
        Returns JSON summary of what was composed and performed.
        """
        if sub_lm is None:
            return json.dumps(
                {"error": "compose_next requires sub_lm (live mode only)"}
            )

        next_slot = max((h["slot"] for h in _live_history), default=-1) + 1
        prompt = _build_compose_prompt(creative_prompt, next_slot)

        completions = sub_lm(messages=[{"role": "user", "content": prompt}])
        raw = completions[0] if completions else ""
        # sub_lm returns dicts with 'text' key — normalise to string
        if isinstance(raw, dict):
            response_text = raw.get("text") or raw.get("content") or str(raw)
        else:
            response_text = str(raw)

        m = re.search(r"\{.*\}", response_text, re.S)
        if not m:
            return json.dumps(
                {
                    "error": "Sub-LM returned no valid JSON",
                    "raw": response_text[:500],
                }
            )
        try:
            section = json.loads(m.group())
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": f"Sub-LM returned invalid JSON: {e}",
                    "raw": response_text[:500],
                }
            )

        # Validate and clamp
        section.setdefault("slot", next_slot)
        section["slot"] = int(section["slot"])
        section.setdefault("bars", 8)
        section["bars"] = int(section["bars"])
        section.setdefault("energy", 0.5)
        energy_before_clamp = float(section["energy"])
        section["energy"] = max(0.0, min(1.0, energy_before_clamp))

        # Guardrails are slightly wider than prompt suggestions to allow
        # sub-LM creative freedom while preventing extremes.
        guardrails_applied = []
        info = _get_elapsed()
        if info:
            elapsed_min, _ = info
            total = _live_state["target_duration"]
            if elapsed_min < total * 0.33:
                # Opening: allow 0.05–0.55
                clamped = max(0.05, min(0.55, section["energy"]))
                if clamped != section["energy"]:
                    guardrails_applied.append(f"opening_cap_{clamped:.2f}")
                section["energy"] = clamped
            elif elapsed_min < total * 0.66:
                # Middle: boost floor to 0.3 so peaks actually peak
                if section["energy"] < 0.3:
                    guardrails_applied.append("middle_floor_0.30")
                section["energy"] = max(0.3, section["energy"])
                # High-energy middle sections: cap bars at 8 for tighter loops
                if section["energy"] >= 0.6:
                    if section["bars"] > 8:
                        guardrails_applied.append("middle_bars_cap_8")
                    section["bars"] = min(section["bars"], 8)
            else:
                # Final phase: if no peak happened yet, enforce a floor
                max_energy_so_far = max((h["energy"] for h in _live_history), default=0)
                if max_energy_so_far < 0.6:
                    section["energy"] = max(0.5, section["energy"])
                    guardrails_applied.append("no_peak_floor_0.50")

        section.setdefault("key", "C")
        section.setdefault("chords", [section["key"] + "m7"])
        section.setdefault("drums", "none")
        section.setdefault("bass", "none")
        section.setdefault("pad", "sustained")
        section.setdefault("stab", "none")
        section.setdefault("texture", "none")
        section.setdefault("name", f"Section {section['slot']}")

        clip_info, wait_info = _execute_section(section)
        if wait_info is None:
            # _execute_section returned an error in clip_info
            return json.dumps(clip_info)

        # Texture density: fraction of instruments active (0.0-1.0)
        _STYLE_KEYS = ("drums", "bass", "pad", "stab", "texture")
        density = sum(1 for k in _STYLE_KEYS if section[k] != "none") / len(_STYLE_KEYS)

        _live_history.append(
            {
                "section": section["name"],
                "slot": section["slot"],
                "energy": section["energy"],
                "density": round(density, 2),
                "key": section["key"],
                "chords": section["chords"][:4],
                "drums": section["drums"],
                "bass": section["bass"],
                "pad": section["pad"],
                "stab": section["stab"],
                "texture": section["texture"],
                "creative_prompt": creative_prompt,
                "sub_lm_prompt": prompt,
                "sub_lm_raw": response_text[:1000],
                "energy_before_clamp": round(energy_before_clamp, 3),
                "guardrails_applied": guardrails_applied,
            }
        )

        style = "/".join(section[k] for k in _STYLE_KEYS)
        return json.dumps(
            {
                "composed": section["name"],
                "slot": section["slot"],
                "energy": section["energy"],
                "key": section["key"],
                "chords": section["chords"],
                "style": style,
                "bars": section["bars"],
                "clips_created": clip_info.get("clips_created", 0),
                "elapsed_minutes": wait_info.get("elapsed_minutes"),
                "remaining_minutes": wait_info.get("remaining_minutes"),
                "sections_so_far": len(_live_history),
            }
        )

    def get_arc_summary() -> str:
        """Get a summary of the performance arc so far.

        Returns energy curve, style combo stats, time phase, and variety metrics.
        Use this to understand where you are and what to do next.
        """
        if not _live_history:
            return json.dumps({"sections": 0, "note": "No sections composed yet"})

        energies = [h["energy"] for h in _live_history]
        densities = [h.get("density", 0.33) for h in _live_history]
        keys_used = {}
        combo_counts = {}
        for h in _live_history:
            k = h.get("key", "?")
            keys_used[k] = keys_used.get(k, 0) + 1
            combo = (
                h.get("drums", "?"),
                h.get("bass", "?"),
                h.get("pad", "?"),
                h.get("stab", "?"),
                h.get("texture", "?"),
            )
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

        # Energy bands
        bands = {"low (0-0.3)": 0, "mid (0.3-0.6)": 0, "high (0.6-1.0)": 0}
        for e in energies:
            if e < 0.3:
                bands["low (0-0.3)"] += 1
            elif e < 0.6:
                bands["mid (0.3-0.6)"] += 1
            else:
                bands["high (0.6-1.0)"] += 1

        # Recent energy trend (last 5)
        recent_energies = energies[-5:]
        trend = "flat"
        if len(recent_energies) >= 3:
            diffs = [
                recent_energies[i + 1] - recent_energies[i]
                for i in range(len(recent_energies) - 1)
            ]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0.03:
                trend = "rising"
            elif avg_diff < -0.03:
                trend = "falling"

        # Contrast between adjacent sections
        contrasts = []
        for i in range(1, len(_live_history)):
            prev, curr = _live_history[i - 1], _live_history[i]
            energy_delta = abs(curr["energy"] - prev["energy"])
            style_changes = sum(
                1
                for k in ("drums", "bass", "pad", "stab", "texture")
                if curr.get(k, "none") != prev.get(k, "none")
            )
            contrasts.append(energy_delta + style_changes * 0.1)

        avg_contrast = round(sum(contrasts) / len(contrasts), 2) if contrasts else 0.0
        # Flag monotony: low energy delta AND same instruments for 3+ sections
        low_contrast_warning = None
        if len(_live_history) >= 3:
            tail = _live_history[-3:]
            tail_energy_deltas = [
                abs(tail[i]["energy"] - tail[i - 1]["energy"])
                for i in range(1, len(tail))
            ]
            tail_style_same = all(
                tail[i].get(k) == tail[0].get(k)
                for i in range(1, len(tail))
                for k in ("drums", "bass", "pad", "stab", "texture")
            )
            if max(tail_energy_deltas) < 0.1 and tail_style_same:
                low_contrast_warning = (
                    "Low contrast for 3+ sections — risk of monotony. "
                    "Change instruments or make a bigger energy shift."
                )

        result = {
            "sections": len(_live_history),
            "energy_range": [round(min(energies), 2), round(max(energies), 2)],
            "energy_avg": round(sum(energies) / len(energies), 2),
            "energy_trend": trend,
            "energy_bands": bands,
            "density_avg": round(sum(densities) / len(densities), 2),
            "density_current": densities[-1],
            "avg_contrast": avg_contrast,
            "keys_used": keys_used,
            "unique_style_combos": len(combo_counts),
            "most_used_combos": [
                {"style": "/".join(combo), "count": c}
                for combo, c in sorted(combo_counts.items(), key=lambda x: -x[1])[:5]
            ],
        }

        if low_contrast_warning:
            result["warning"] = low_contrast_warning

        info = _get_elapsed()
        if info:
            elapsed_min, remaining_min = info
            total = _live_state["target_duration"]
            result["elapsed_minutes"] = round(elapsed_min, 1)
            result["remaining_minutes"] = round(remaining_min, 1)
            if elapsed_min < total * 0.33:
                phase = "opening"
            elif elapsed_min < total * 0.66:
                phase = "middle"
            else:
                phase = "final"
            result["phase"] = phase

        return json.dumps(result)

    def ready_to_submit() -> str:
        """Check if composition is complete enough to SUBMIT.

        Call in a SEPARATE step BEFORE the step where you SUBMIT.
        Do NOT call ready_to_submit() and SUBMIT() in the same code block.
        Returns 'READY' if all milestones are met, or lists what's still missing.
        """
        issues = []
        if live_mode:
            # In live mode, setup_session() handles browse + tracks
            if not _done["tracks"]:
                issues.append("setup_session() not called — set up your session first")
        else:
            if not _done["browse"]:
                issues.append("browse() not called — search for instruments first")
            if not _done["tracks"]:
                issues.append("create_tracks() not called — set up your tracks")
        if _done["clips"] < min_clips:
            issues.append(
                f"only {_done['clips']} clips created (need at least {min_clips}) — build more scenes"
            )
        if live_mode:
            # In live mode: check elapsed time instead of mix requirement
            info = _get_elapsed()
            if info:
                elapsed_min, _ = info
                threshold = _live_state["target_duration"] * 0.8
                if elapsed_min < threshold:
                    issues.append(
                        f"only {elapsed_min:.1f} min elapsed (need ~{threshold:.0f} min) — keep evolving"
                    )
            else:
                issues.append("play() not called yet — start playback first")
        else:
            if not _done["mix"]:
                issues.append("set_mix() not called — balance volumes and panning")
            if _done["status_checks"] < 1:
                issues.append(
                    "get_status() never called — verify your work before submitting"
                )
        if issues:
            return "NOT READY to submit:\n" + "\n".join(f"  - {i}" for i in issues)
        return "READY — call SUBMIT(report) in the NEXT step (no other tool calls)."

    if live_mode:
        # Live mode: setup_session handles all browsing/loading.
        # No browse/create_tracks/load_* — prevents manual-browse rabbit holes.
        tools = [
            setup_session,
            set_tempo,
            get_status,
            get_params,
            set_param,
            set_mix,
            stop,
            fire_clip,
            fire_scene,
            write_clip,
            ready_to_submit,
            wait,
            elapsed,
            compose_next,
            get_arc_summary,
        ]
    else:
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
    return tools, tools_by_name, _live_history
