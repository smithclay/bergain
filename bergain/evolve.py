"""Live evolution engine — evolves a palette in real time.

LiveEvolver takes a built palette and continuously transforms it, overwriting
scene slots with new musical content driven by sub-LM decisions. The user
can type creative directions at any time, or let auto_direction() fill in.
"""

import json
import re
import time

from .music import STYLE_KEYS, _clip_name_for_role, render_role


class LiveEvolver:
    """Evolve a palette in real time using a sub-LM for musical decisions.

    Args:
        session: A Session (or StubSession) instance.
        sub_lm: A dspy.LM callable for musical decisions.
        brief: The creative brief string.
        track_roles: Dict mapping track name -> role (drums/bass/pad/stab/texture).
        num_scenes: Number of scenes in the palette grid.
        duration_minutes: Target duration for evolution in minutes.
    """

    def __init__(
        self, session, sub_lm, brief, track_roles, num_scenes, duration_minutes=60
    ):
        self.session = session
        self.sub_lm = sub_lm
        self.brief = brief
        self.track_roles = track_roles
        self.num_scenes = max(1, num_scenes)
        self.duration_minutes = duration_minutes
        self.start_time = None
        self.history = []
        self._current_slot = 0
        # Track volumes faded to zero so they can be restored when roles return.
        self._muted_track_volumes = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_palette(self, palette_scenes):
        """Seed history from palette write_clip data."""
        for scene in palette_scenes:
            entry = {
                "section": scene.get("name", "Palette"),
                "slot": scene.get("slot", 0),
                "energy": scene.get("energy", 0.5),
                "key": scene.get("key", "C"),
                "chords": scene.get("chords", []),
            }
            for key in STYLE_KEYS:
                entry[key] = scene.get(key, "none")
            self.history.append(entry)
        self._current_slot = 0

    # ------------------------------------------------------------------
    # Scene slot cycling
    # ------------------------------------------------------------------

    def next_scene_slot(self):
        """Return the next scene slot (round-robin)."""
        slot = self._current_slot
        self._current_slot = (self._current_slot + 1) % self.num_scenes
        return slot

    # ------------------------------------------------------------------
    # Time tracking
    # ------------------------------------------------------------------

    def get_elapsed(self):
        """Return (elapsed_minutes, remaining_minutes)."""
        if self.start_time is None:
            return 0.0, self.duration_minutes
        elapsed_sec = time.time() - self.start_time
        elapsed_min = elapsed_sec / 60.0
        remaining_min = max(0.0, self.duration_minutes - elapsed_min)
        return elapsed_min, remaining_min

    def should_stop(self):
        """True when elapsed >= duration."""
        elapsed, _ = self.get_elapsed()
        return elapsed >= self.duration_minutes

    # ------------------------------------------------------------------
    # Auto direction
    # ------------------------------------------------------------------

    _OPENING_DIRECTIONS = [
        "atmospheric intro, build slowly",
        "establish the mood, sparse textures",
        "gentle opening, let space breathe",
        "low energy foundation, soft pads",
        "minimal pulse, hint of rhythm",
    ]

    _MIDDLE_DIRECTIONS = [
        "bring energy up, drive the rhythm",
        "drop to breakdown, strip layers",
        "switch to a new key for freshness",
        "build tension with syncopation",
        "peak energy, all instruments full",
        "quiet interlude, just atmosphere",
        "walking bass with arpeggiated pads",
        "driving four-on-floor, heavy bass",
    ]

    _FINAL_DIRECTIONS = [
        "begin winding down, fade gently",
        "strip layers, return to atmosphere",
        "gentle outro, sparse and dreamy",
        "slow fade, let the last chords ring",
        "minimal closing, just pads",
    ]

    def auto_direction(self):
        """Generate a direction based on arc phase and recent history."""
        import random

        elapsed, _ = self.get_elapsed()
        total = self.duration_minutes
        progress = elapsed / total if total > 0 else 0.5

        # Pick phase-appropriate direction pool
        if progress < 0.33:
            pool = self._OPENING_DIRECTIONS
        elif progress < 0.66:
            pool = self._MIDDLE_DIRECTIONS
        else:
            pool = self._FINAL_DIRECTIONS

        # Avoid repeating the last direction
        direction = random.choice(pool)
        if self.history:
            last_prompt = self.history[-1].get("creative_prompt", "")
            attempts = 0
            while direction == last_prompt and attempts < 5:
                direction = random.choice(pool)
                attempts += 1

        return direction

    # ------------------------------------------------------------------
    # Sub-LM prompt building (adapted from tools.py _build_compose_prompt)
    # ------------------------------------------------------------------

    _HISTORY_PROMPT_KEYS = ("section", "slot", "energy", "key", "chords") + STYLE_KEYS

    def _strip_history(self, entries):
        """Strip bulky fields from history entries before embedding in prompts."""
        return [{k: h[k] for k in self._HISTORY_PROMPT_KEYS if k in h} for h in entries]

    def _build_prompt(self, creative_prompt, slot):
        """Assemble the sub-LM prompt from history, arc phase, and direction."""
        recent = self.history[-5:]
        earlier_summary = ""
        if len(self.history) > 5:
            earlier = self.history[:-5]
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
        for h in self.history:
            combo = tuple(h.get(k, "none") for k in STYLE_KEYS)
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        overused = ["/".join(c) for c, n in combo_counts.items() if n >= 3]
        variety_note = ""
        if overused:
            variety_note = (
                f"OVERUSED combos (pick something DIFFERENT): {', '.join(overused)}.\n"
            )

        # Chord progression variety
        chord_history = ["/".join(h.get("chords", [])) for h in self.history]
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
        elapsed, remaining = self.get_elapsed()
        total = self.duration_minutes
        if total > 0:
            if elapsed < total * 0.33:
                arc_phase = (
                    f"OPENING third ({elapsed:.0f}/{total} min). "
                    "Build gradually — energy should stay low-to-moderate (0.1-0.5).\n"
                )
            elif elapsed < total * 0.66:
                arc_phase = (
                    f"MIDDLE third ({elapsed:.0f}/{total} min). "
                    "Create waves — alternate between lifts and dips (0.3-0.8).\n"
                )
            else:
                max_energy = max((h["energy"] for h in self.history), default=0)
                if max_energy < 0.6:
                    arc_phase = (
                        f"FINAL third ({elapsed:.0f}/{total} min). "
                        "You HAVEN'T PEAKED YET — push energy to 0.6+ before winding down.\n"
                    )
                else:
                    arc_phase = (
                        f"FINAL third ({elapsed:.0f}/{total} min). "
                        "Begin winding down — energy should generally descend (0.5-0.1).\n"
                    )

        # Time budget
        time_budget = ""
        if self.start_time is not None:
            try:
                bpm = float(self.session.api.song.get("tempo") or 120)
            except Exception:
                bpm = 120
            bar_duration_min = 4.0 / bpm  # (4 beats / bpm) in minutes
            remaining_bars = (
                int(remaining / bar_duration_min) if bar_duration_min > 0 else 999
            )
            time_budget = f"TIME: {elapsed:.1f}/{total} min elapsed, ~{remaining_bars} bars remaining.\n"

        return (
            f"You are composing the next section of a live performance.\n"
            f"Brief: {self.brief}\n"
            f"{time_budget}"
            f"{arc_phase}"
            f"{earlier_summary}"
            f"Recent sections: {json.dumps(self._strip_history(recent))}\n"
            f"{variety_note}"
            f"{chord_variety_note}"
            f"Creative direction: {creative_prompt}\n\n"
            f"Return a single JSON object with these fields:\n"
            f"  name (str): evocative section name\n"
            f"  slot (int): use {slot}\n"
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

    # ------------------------------------------------------------------
    # Clip writing (adapted from tools.py write_clip internals)
    # ------------------------------------------------------------------

    def _write_clips(self, section, slot):
        """Render and write clips for a section to the given slot.

        Returns number of clips created.
        """
        bars = int(section.get("bars", 8))
        energy = max(0.0, min(1.0, float(section.get("energy", 0.5))))
        chords = section.get("chords", [])
        key_root = section.get("key", "C")
        name = section.get("name", f"Evolve {slot}")
        length_beats = float(bars * 4)

        clips_created = 0
        for track_name, role in self.track_roles.items():
            style = section.get(role, "none")
            notes = render_role(role, style, bars, energy, chords, key_root, name)

            if notes:
                clip_name = _clip_name_for_role(role, name)
                self.session.clip(track_name, slot, length_beats, notes, name=clip_name)
                clips_created += 1

        return clips_created

    def _get_track_volume(self, track_name):
        """Best-effort current volume lookup for a track."""
        try:
            track_idx = self.session._t(track_name)
            volume = self.session.api.track(track_idx).get("volume")
            return float(volume)
        except Exception:
            return 0.85

    def restore_track_volumes(self, section):
        """Restore volumes for roles that are active again in this section."""
        for track_name, role in self.track_roles.items():
            if role not in STYLE_KEYS:
                continue
            now_active = section.get(role, "none") != "none"
            if not now_active:
                continue
            if track_name not in self._muted_track_volumes:
                continue

            target_volume = self._muted_track_volumes.pop(track_name)
            try:
                self.session.mix(**{track_name: float(target_volume)})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Transition handling (adapted from tools.py _execute_section)
    # ------------------------------------------------------------------

    def _handle_transitions(self, section):
        """Fade out tracks going from active to none."""
        if not self.history:
            return

        prev = self.history[-1]
        for track_name, role in self.track_roles.items():
            if role not in STYLE_KEYS:
                continue
            was_active = prev.get(role, "none") != "none"
            now_silent = section.get(role, "none") == "none"
            if was_active and now_silent:
                try:
                    prior_volume = (
                        self._get_track_volume(track_name)
                        if track_name not in self._muted_track_volumes
                        else None
                    )
                    fade_info = self.session.fade(
                        track_name, 0.0, steps=4, duration=0.5
                    )
                    if track_name not in self._muted_track_volumes:
                        from_volume = None
                        if isinstance(fade_info, dict):
                            from_volume = fade_info.get("from")
                        if from_volume is None:
                            from_volume = (
                                prior_volume if prior_volume is not None else 0.85
                            )
                        self._muted_track_volumes[track_name] = max(
                            0.0, min(1.0, float(from_volume))
                        )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Core: evolve_section
    # ------------------------------------------------------------------

    _FALLBACK_SECTION = {
        "name": "Fallback",
        "bars": 8,
        "energy": 0.5,
        "key": "C",
        "chords": ["Cm"],
        "drums": "minimal",
        "bass": "sustained",
        "pad": "atmospheric",
    }

    def _fallback_section(self):
        """Return a fallback section: copy of last history entry, or default."""
        if self.history:
            return dict(self.history[-1])
        return dict(self._FALLBACK_SECTION)

    def _parse_response(self, response_text):
        """Extract JSON section dict from sub-LM response, or None on failure."""
        m = re.search(r"\{.*\}", response_text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    def _validate_section(self, section, slot):
        """Apply defaults and clamp values on a parsed section dict."""
        section.setdefault("slot", slot)
        section["slot"] = int(section["slot"])
        section.setdefault("bars", 8)
        section["bars"] = int(section["bars"])
        section.setdefault("energy", 0.5)
        section["energy"] = max(0.0, min(1.0, float(section["energy"])))
        section.setdefault("key", "C")
        section.setdefault("chords", [section["key"] + "m7"])
        section.setdefault("drums", "none")
        section.setdefault("bass", "none")
        section.setdefault("pad", "sustained")
        section.setdefault("stab", "none")
        section.setdefault("texture", "none")
        section.setdefault("name", f"Section {slot}")
        return section

    def evolve_section(self, creative_prompt, slot):
        """Build prompt, call sub_lm, validate, write clips, return section dict."""
        prompt = self._build_prompt(creative_prompt, slot)

        completions = self.sub_lm(messages=[{"role": "user", "content": prompt}])
        raw = completions[0] if completions else ""
        if isinstance(raw, dict):
            response_text = raw.get("text") or raw.get("content") or str(raw)
        else:
            response_text = str(raw)

        section = self._parse_response(response_text)
        if section is None:
            section = self._fallback_section()
            section["slot"] = slot

        section = self._validate_section(section, slot)

        # Handle transitions, write clips
        self._handle_transitions(section)
        self._write_clips(section, slot)

        # Append to history
        density = sum(1 for k in STYLE_KEYS if section[k] != "none") / len(STYLE_KEYS)
        entry = {
            "section": section["name"],
            "slot": section["slot"],
            "energy": section["energy"],
            "density": round(density, 2),
            "key": section["key"],
            "chords": section["chords"][:4],
            "creative_prompt": creative_prompt,
        }
        for key in STYLE_KEYS:
            entry[key] = section[key]
        self.history.append(entry)

        return section
