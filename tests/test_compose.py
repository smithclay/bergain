"""Validation tests for compose — renderers, tools, and palette mode.

Runs without Ableton: uses a stub Session that records calls.
    uv run python -m pytest tests/ -v
"""

import json

import pytest

from bergain.music import (
    render_bass,
    render_drums,
    render_pads,
)
from bergain.progress import ProgressState
from bergain.stub import StubSession
from bergain.tools import make_tools


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
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1)
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
# ready_to_submit palette mode
# ---------------------------------------------------------------------------


def test_ready_to_submit_palette(session):
    """Palette mode should require mix and status checks."""
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1)
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
# Tool list composition
# ---------------------------------------------------------------------------


def test_palette_tools_include_expected(session):
    tools, _, _, _, _ = make_tools(session)
    names = {t.__name__ for t in tools}
    assert "browse" in names
    assert "create_tracks" in names
    assert "write_clip" in names
    assert "set_mix" in names
    assert "ready_to_submit" in names


def test_palette_tools_exclude_live_only(session):
    tools, _, _, _, _ = make_tools(session)
    names = {t.__name__ for t in tools}
    assert "wait" not in names
    assert "elapsed" not in names
    assert "compose_next" not in names
    assert "get_arc_summary" not in names


# ---------------------------------------------------------------------------
# Palette scene capture
# ---------------------------------------------------------------------------


def test_write_clip_captures_palette_scenes(session):
    """write_clip should capture scene data in _palette_scenes."""
    _, tools_by_name, _, _, palette_scenes = make_tools(session, min_clips=1)
    _setup_tracks(tools_by_name)

    tools_by_name["write_clip"](
        json.dumps(
            {
                "name": "Intro",
                "slot": 0,
                "bars": 8,
                "energy": 0.2,
                "key": "F",
                "chords": ["Fm", "Cm7"],
                "drums": "minimal",
                "bass": "none",
                "pad": "atmospheric",
            }
        )
    )

    assert len(palette_scenes) == 1
    assert palette_scenes[0]["name"] == "Intro"
    assert palette_scenes[0]["energy"] == 0.2
    assert palette_scenes[0]["key"] == "F"


def test_track_roles_returned(session):
    """make_tools should return track_roles dict."""
    _, tools_by_name, _, track_roles, _ = make_tools(session, min_clips=1)
    _setup_tracks(tools_by_name)

    assert track_roles["Drums"] == "drums"
    assert track_roles["Bass"] == "bass"
    assert track_roles["Pad"] == "pad"


# ---------------------------------------------------------------------------
# Progress state tests — make_tools(..., progress=state)
# ---------------------------------------------------------------------------


def test_progress_browse_done(session):
    state = ProgressState()
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1, progress=state)
    assert not state.browse_done
    tools_by_name["browse"]("test")
    assert state.browse_done


def test_progress_tracks_done(session):
    state = ProgressState()
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1, progress=state)
    assert not state.tracks_done
    _setup_tracks(tools_by_name)
    assert state.tracks_done


def test_progress_clips_created(session):
    state = ProgressState()
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1, progress=state)
    _setup_tracks(tools_by_name)
    assert state.clips_created == 0
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
    assert state.clips_created >= 1


def test_progress_mix_done(session):
    state = ProgressState()
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1, progress=state)
    _setup_tracks(tools_by_name)
    assert not state.mix_done
    tools_by_name["set_mix"](json.dumps({"Drums": 0.9}))
    assert state.mix_done


# ---------------------------------------------------------------------------
# ProgressState TUI fields
# ---------------------------------------------------------------------------


def test_progress_state_stream_default():
    state = ProgressState()
    assert state.stream == []
    assert state.steer_direction == ""
    assert state.paused is False
    assert state.abort is False


def test_progress_state_stream_append():
    state = ProgressState()
    state.stream.append({"type": "step", "content": "test", "timestamp": 1.0})
    assert len(state.stream) == 1
    assert state.stream[0]["type"] == "step"


def test_progress_state_steer_direction():
    state = ProgressState()
    state.steer_direction = "more energy"
    assert state.steer_direction == "more energy"
    state.steer_direction = ""
    assert state.steer_direction == ""


def test_tool_results_stream_to_progress(session):
    """Tool calls should append result entries to progress.stream."""
    state = ProgressState()
    _, tools_by_name, _, _, _ = make_tools(session, min_clips=1, progress=state)
    _setup_tracks(tools_by_name)

    # Clear stream entries from setup
    state.stream.clear()

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

    types = [e["type"] for e in state.stream]
    assert "result" in types
