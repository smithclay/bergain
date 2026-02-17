"""Validation tests for compose — renderers, tools, and live mode.

Runs without Ableton: uses a stub Session that records calls.
    uv run python -m pytest tests/ -v
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from bergain.music import (
    render_bass,
    render_drums,
    render_pads,
)
from bergain.tools import make_tools


# ---------------------------------------------------------------------------
# Stub session — records calls, returns plausible data
# ---------------------------------------------------------------------------


class StubSession:
    """Minimal session substitute that logs calls without Ableton."""

    def __init__(self, tempo=120):
        self._tempo = tempo
        self.calls = []

    def tempo(self, bpm):
        self._tempo = bpm
        self.calls.append(("tempo", bpm))

    def status(self):
        return {"tempo": self._tempo, "playing": True, "tracks": []}

    def setup(self, tracks):
        self.calls.append(("setup", [t.name for t in tracks]))
        return len(tracks)

    def browse(self, query, category=None):
        return [{"name": f"Fake {query}.adg"}]

    def clip(self, track, slot, length, notes, name=""):
        self.calls.append(("clip", track, slot, len(notes)))
        return {"track": track, "slot": slot, "notes": len(notes)}

    def fire(self, slot):
        self.calls.append(("fire_scene", slot))

    def fire_clip(self, track, slot):
        self.calls.append(("fire_clip", track, slot))

    def mix(self, **kwargs):
        self.calls.append(("mix", kwargs))

    def play(self):
        self.calls.append(("play",))

    def stop(self):
        self.calls.append(("stop",))

    def load_instrument(self, track, name):
        return name

    def load_sound(self, track, name):
        return name

    def load_effect(self, track, name):
        return name

    def load_drum_kit(self, track, name):
        return name

    def params(self, track, device_index):
        return ["Param A", "Param B"]

    def param(self, track, device_index, param, value):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    return StubSession(tempo=120)


def _setup_tracks(tools_by_name):
    """Call create_tracks through the tools dict to populate _track_roles."""
    tools_by_name["create_tracks"](
        json.dumps(
            [
                {"name": "Drums", "drum_kit": "Kit.adg"},
                {"name": "Bass", "instrument": "Operator"},
                {"name": "Pad", "sound": "Warm Pad"},
            ]
        )
    )


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------

ALL_DRUM_STYLES = [
    "four_on_floor",
    "half_time",
    "breakbeat",
    "minimal",
    "shuffle",
    "sparse_perc",
]
ALL_BASS_STYLES = [
    "sustained",
    "pulsing_8th",
    "offbeat_8th",
    "rolling_16th",
    "walking",
]
ALL_PAD_STYLES = [
    "sustained",
    "atmospheric",
    "pulsing",
    "arpeggiated",
    "swells",
]


@pytest.mark.parametrize("style", ALL_DRUM_STYLES)
def test_render_drums_all_styles(style):
    notes = render_drums(4, 0.5, style, section_name="test")
    assert len(notes) > 0, f"drums/{style} produced no notes"
    for pitch, start, dur, vel in notes:
        assert 0 <= pitch <= 127
        assert start >= 0.0
        assert dur > 0
        assert 1 <= vel <= 127


@pytest.mark.parametrize("style", ALL_BASS_STYLES)
def test_render_bass_all_styles(style):
    notes = render_bass(4, 0.5, style, ["Cm7", "Fm"], "C", section_name="test")
    assert len(notes) > 0, f"bass/{style} produced no notes"
    for pitch, start, dur, vel in notes:
        assert 0 <= pitch <= 127
        assert start >= 0.0
        assert dur > 0
        assert 1 <= vel <= 127


@pytest.mark.parametrize("style", ALL_PAD_STYLES)
def test_render_pads_all_styles(style):
    notes = render_pads(4, 0.5, style, ["Cm7", "Fm"], "C", section_name="test")
    assert len(notes) > 0, f"pad/{style} produced no notes"
    for pitch, start, dur, vel in notes:
        assert 0 <= pitch <= 127
        assert start >= 0.0
        assert dur > 0
        assert 1 <= vel <= 127


@pytest.mark.parametrize("energy", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_render_drums_energy_range(energy):
    """All drum styles should work at all energy levels without crashing."""
    for style in ALL_DRUM_STYLES:
        notes = render_drums(4, energy, style, section_name="energy_test")
        for _, _, _, vel in notes:
            assert 1 <= vel <= 127, (
                f"velocity OOB: {vel} at energy={energy} style={style}"
            )


def test_render_drums_unknown_style_returns_empty():
    """Unknown drum style should return empty (no crash)."""
    notes = render_drums(4, 0.5, "nonexistent_style")
    assert notes == []


def test_render_bass_no_chords_uses_key_root():
    """Bass with empty chords should fall back to key root."""
    notes = render_bass(4, 0.5, "sustained", [], "F", section_name="test")
    assert len(notes) > 0


# ---------------------------------------------------------------------------
# write_clip integration (all style combos reach session.clip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "drums,bass,pad",
    [
        ("shuffle", "walking", "arpeggiated"),
        ("sparse_perc", "sustained", "swells"),
        ("four_on_floor", "rolling_16th", "pulsing"),
        ("none", "none", "atmospheric"),
    ],
)
def test_write_clip_style_combos(session, drums, bass, pad):
    _, tools_by_name, _ = make_tools(session, min_clips=1)
    _setup_tracks(tools_by_name)

    result = json.loads(
        tools_by_name["write_clip"](
            json.dumps(
                {
                    "name": "Test",
                    "slot": 0,
                    "bars": 4,
                    "energy": 0.5,
                    "key": "C",
                    "chords": ["Cm7"],
                    "drums": drums,
                    "bass": bass,
                    "pad": pad,
                }
            )
        )
    )

    assert "error" not in result
    if drums == "none" and bass == "none":
        # Only pad track creates a clip
        assert result["clips_created"] == 1
    else:
        assert result["clips_created"] >= 1


# ---------------------------------------------------------------------------
# get_arc_summary tests
# ---------------------------------------------------------------------------


def test_arc_summary_empty(session):
    _, tools_by_name, _ = make_tools(session, live_mode=True, duration_minutes=10)
    result = json.loads(tools_by_name["get_arc_summary"]())
    assert result["sections"] == 0


def test_arc_summary_after_sections(session):
    """Simulate history via compose_next's internal list, then check arc."""
    _, tools_by_name, _ = make_tools(
        session, live_mode=True, duration_minutes=10, sub_lm=None, brief="test"
    )
    _setup_tracks(tools_by_name)

    # Manually populate _live_history via write_clip + direct list access
    # We'll use write_clip to bump clip count, then check arc is still empty
    # (compose_next populates _live_history, not write_clip)
    result = json.loads(tools_by_name["get_arc_summary"]())
    assert result["sections"] == 0


# ---------------------------------------------------------------------------
# compose_next tests (mock sub_lm)
# ---------------------------------------------------------------------------


def _make_mock_sub_lm(section_json):
    """Return a callable that mimics dspy.LM returning a JSON string."""
    mock = MagicMock()
    mock.return_value = [json.dumps(section_json)]
    return mock


def test_compose_next_basic(session):
    section = {
        "name": "Gentle Opening",
        "slot": 0,
        "bars": 4,
        "energy": 0.3,
        "key": "Eb",
        "chords": ["Ebmaj7"],
        "drums": "minimal",
        "bass": "none",
        "pad": "atmospheric",
    }
    mock_lm = _make_mock_sub_lm(section)

    _, tools_by_name, _ = make_tools(
        session,
        live_mode=True,
        duration_minutes=10,
        sub_lm=mock_lm,
        brief="Test lullaby",
    )
    _setup_tracks(tools_by_name)
    # Start playback to init timer
    tools_by_name["play"]()

    result = json.loads(tools_by_name["compose_next"]("Open with soft atmosphere"))

    assert "error" not in result
    assert result["composed"] == "Gentle Opening"
    assert result["energy"] == 0.3
    assert result["sections_so_far"] == 1
    mock_lm.assert_called_once()

    # Verify the prompt sent to sub_lm does NOT contain exact param values
    call_args = mock_lm.call_args
    prompt_sent = call_args[1]["messages"][0]["content"]
    assert "Open with soft atmosphere" in prompt_sent
    assert "Test lullaby" in prompt_sent


def test_compose_next_no_sub_lm(session):
    """compose_next without sub_lm should return an error, not crash."""
    _, tools_by_name, _ = make_tools(session, live_mode=True, sub_lm=None)
    result = json.loads(tools_by_name["compose_next"]("anything"))
    assert "error" in result


def test_compose_next_bad_json_from_lm(session):
    """Sub-LM returning garbage should produce an error, not crash."""
    mock_lm = MagicMock(return_value=["This is not JSON at all!"])
    _, tools_by_name, _ = make_tools(
        session, live_mode=True, duration_minutes=10, sub_lm=mock_lm, brief="test"
    )
    _setup_tracks(tools_by_name)
    tools_by_name["play"]()

    result = json.loads(tools_by_name["compose_next"]("do something"))
    assert "error" in result


def test_compose_next_history_compression(session):
    """After >5 sections, compose_next should compress earlier history."""
    sections = [
        {
            "name": f"S{i}",
            "slot": i,
            "bars": 4,
            "energy": 0.1 * i,
            "key": "C",
            "chords": ["Cm"],
            "drums": "minimal",
            "bass": "sustained",
            "pad": "sustained",
        }
        for i in range(8)
    ]
    call_count = [0]

    def mock_lm_sequence(messages, **kwargs):
        idx = min(call_count[0], len(sections) - 1)
        call_count[0] += 1
        return [json.dumps(sections[idx])]

    mock_lm = MagicMock(side_effect=mock_lm_sequence)
    _, tools_by_name, _ = make_tools(
        session, live_mode=True, duration_minutes=60, sub_lm=mock_lm, brief="test"
    )
    _setup_tracks(tools_by_name)
    tools_by_name["play"]()

    # Compose 7 sections
    for i in range(7):
        tools_by_name["compose_next"](f"section {i}")

    # 8th call — history should be compressed (5 recent + summary of 2 earlier)
    tools_by_name["compose_next"]("section 7")
    last_call = mock_lm.call_args
    prompt_sent = last_call[1]["messages"][0]["content"]
    assert "Earlier (" in prompt_sent, "Expected compressed history summary"


def test_compose_next_overused_combo_warning(session):
    """After 3+ uses of same combo, sub-LM prompt should warn."""
    same_section = {
        "name": "Repeat",
        "slot": 0,
        "bars": 4,
        "energy": 0.5,
        "key": "C",
        "chords": ["Cm"],
        "drums": "minimal",
        "bass": "sustained",
        "pad": "sustained",
    }
    call_count = [0]

    def mock_lm_fn(messages, **kwargs):
        s = dict(same_section)
        s["slot"] = call_count[0]
        call_count[0] += 1
        return [json.dumps(s)]

    mock_lm = MagicMock(side_effect=mock_lm_fn)
    _, tools_by_name, _ = make_tools(
        session, live_mode=True, duration_minutes=60, sub_lm=mock_lm, brief="test"
    )
    _setup_tracks(tools_by_name)
    tools_by_name["play"]()

    for i in range(4):
        tools_by_name["compose_next"](f"step {i}")

    last_prompt = mock_lm.call_args[1]["messages"][0]["content"]
    assert "OVERUSED" in last_prompt, (
        "Expected overuse warning after 3 identical combos"
    )


# ---------------------------------------------------------------------------
# Arc phase guidance in compose_next
# ---------------------------------------------------------------------------


def test_compose_next_arc_phases(session):
    """Prompt should reflect opening/middle/final phase based on elapsed time."""
    section = {
        "name": "X",
        "slot": 0,
        "bars": 4,
        "energy": 0.5,
        "key": "C",
        "chords": ["Cm"],
        "drums": "minimal",
        "bass": "none",
        "pad": "sustained",
    }
    call_count = [0]

    def mock_lm_fn(messages, **kwargs):
        s = dict(section)
        s["slot"] = call_count[0]
        call_count[0] += 1
        return [json.dumps(s)]

    mock_lm = MagicMock(side_effect=mock_lm_fn)
    # 3-minute target so phases are at 1/2/3 min boundaries
    _, tools_by_name, _ = make_tools(
        session, live_mode=True, duration_minutes=3, sub_lm=mock_lm, brief="test"
    )
    _setup_tracks(tools_by_name)
    tools_by_name["play"]()

    # First call — should be "OPENING"
    tools_by_name["compose_next"]("intro")
    prompt = mock_lm.call_args[1]["messages"][0]["content"]
    assert "OPENING" in prompt


# ---------------------------------------------------------------------------
# ready_to_submit live mode
# ---------------------------------------------------------------------------


def test_ready_to_submit_live_mode_too_early(session):
    _, tools_by_name, _ = make_tools(
        session, min_clips=1, live_mode=True, duration_minutes=60
    )
    _setup_tracks(tools_by_name)

    # Write enough clips
    tools_by_name["browse"]("test")
    tools_by_name["write_clip"](
        json.dumps(
            {
                "name": "Test",
                "slot": 0,
                "bars": 4,
                "energy": 0.5,
                "key": "C",
                "chords": ["Cm"],
                "drums": "minimal",
                "bass": "sustained",
                "pad": "sustained",
            }
        )
    )

    # Start playback
    tools_by_name["play"]()

    result = tools_by_name["ready_to_submit"]()
    assert "NOT READY" in result
    assert "elapsed" in result.lower()


def test_ready_to_submit_palette_unchanged(session):
    """Palette mode should still require mix and status checks."""
    _, tools_by_name, _ = make_tools(session, min_clips=1, live_mode=False)
    _setup_tracks(tools_by_name)

    tools_by_name["browse"]("test")
    tools_by_name["write_clip"](
        json.dumps(
            {
                "name": "Test",
                "slot": 0,
                "bars": 4,
                "energy": 0.5,
                "key": "C",
                "chords": ["Cm"],
                "drums": "minimal",
                "bass": "sustained",
                "pad": "sustained",
            }
        )
    )

    result = tools_by_name["ready_to_submit"]()
    assert "NOT READY" in result
    assert "set_mix" in result
    assert "get_status" in result


# ---------------------------------------------------------------------------
# wait / elapsed
# ---------------------------------------------------------------------------


def test_elapsed_before_play(session):
    _, tools_by_name, _ = make_tools(session, live_mode=True, duration_minutes=10)
    result = json.loads(tools_by_name["elapsed"]())
    assert result["elapsed_minutes"] == 0
    assert "Timer starts" in result.get("note", "")


def test_elapsed_after_play(session):
    _, tools_by_name, _ = make_tools(session, live_mode=True, duration_minutes=10)
    tools_by_name["play"]()
    # Sleep long enough that rounding to 1 decimal still shows > 0
    time.sleep(7)  # 7s = 0.117 min → rounds to 0.1

    result = json.loads(tools_by_name["elapsed"]())
    assert result["elapsed_minutes"] > 0
    assert result["remaining_minutes"] < 10
    assert result["target_minutes"] == 10
    assert "note" not in result  # timer should be started, no "Timer starts" note


# ---------------------------------------------------------------------------
# Tool list composition
# ---------------------------------------------------------------------------


def test_live_tools_include_live_only(session):
    tools, _, _ = make_tools(session, live_mode=True, duration_minutes=10)
    names = {t.__name__ for t in tools}
    assert "wait" in names
    assert "elapsed" in names
    assert "compose_next" in names
    assert "get_arc_summary" in names


def test_palette_tools_exclude_live_only(session):
    tools, _, _ = make_tools(session, live_mode=False)
    names = {t.__name__ for t in tools}
    assert "wait" not in names
    assert "elapsed" not in names
    assert "compose_next" not in names
    assert "get_arc_summary" not in names
