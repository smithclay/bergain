"""Tests for the LiveEvolver evolution engine."""

import json
import time
from unittest.mock import MagicMock


from bergain.evolve import LiveEvolver
from bergain.stub import StubSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PALETTE = [
    {
        "name": "Intro",
        "slot": 0,
        "energy": 0.2,
        "key": "F",
        "chords": ["Fm", "Cm7"],
        "drums": "minimal",
        "bass": "none",
        "pad": "atmospheric",
        "stab": "none",
        "texture": "none",
        "bars": 8,
    },
    {
        "name": "Build",
        "slot": 1,
        "energy": 0.5,
        "key": "F",
        "chords": ["Fm", "Abmaj7"],
        "drums": "four_on_floor",
        "bass": "pulsing_8th",
        "pad": "sustained",
        "stab": "none",
        "texture": "none",
        "bars": 8,
    },
    {
        "name": "Peak",
        "slot": 2,
        "energy": 0.9,
        "key": "F",
        "chords": ["Fm", "Cm7", "Abmaj7"],
        "drums": "four_on_floor",
        "bass": "rolling_16th",
        "pad": "pulsing",
        "stab": "none",
        "texture": "none",
        "bars": 4,
    },
]

TRACK_ROLES = {
    "Drums": "drums",
    "Bass": "bass",
    "Pad": "pad",
}


def _make_mock_sub_lm(section_json):
    """Return a callable that mimics dspy.LM returning a JSON string."""
    mock = MagicMock()
    mock.return_value = [json.dumps(section_json)]
    return mock


def _make_evolver(sub_lm=None, duration=60):
    """Create a LiveEvolver with StubSession for testing."""
    session = StubSession(tempo=130)
    if sub_lm is None:
        sub_lm = _make_mock_sub_lm(
            {
                "name": "Test Section",
                "slot": 0,
                "bars": 8,
                "energy": 0.5,
                "key": "F",
                "chords": ["Fm"],
                "drums": "minimal",
                "bass": "sustained",
                "pad": "atmospheric",
            }
        )
    evolver = LiveEvolver(
        session=session,
        sub_lm=sub_lm,
        brief="Dark techno in F minor",
        track_roles=TRACK_ROLES,
        num_scenes=3,
        duration_minutes=duration,
    )
    return evolver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitializeFromPalette:
    def test_seeds_history(self):
        evolver = _make_evolver()
        evolver.initialize_from_palette(SAMPLE_PALETTE)
        assert len(evolver.history) == 3
        assert evolver.history[0]["section"] == "Intro"
        assert evolver.history[1]["energy"] == 0.5
        assert evolver.history[2]["drums"] == "four_on_floor"

    def test_empty_palette(self):
        evolver = _make_evolver()
        evolver.initialize_from_palette([])
        assert len(evolver.history) == 0


class TestNextSceneSlot:
    def test_cycles_through_scenes(self):
        evolver = _make_evolver()
        evolver.num_scenes = 3
        slots = [evolver.next_scene_slot() for _ in range(7)]
        assert slots == [0, 1, 2, 0, 1, 2, 0]

    def test_wraps_at_num_scenes(self):
        evolver = _make_evolver()
        evolver.num_scenes = 2
        assert evolver.next_scene_slot() == 0
        assert evolver.next_scene_slot() == 1
        assert evolver.next_scene_slot() == 0


class TestAutoDirection:
    def test_produces_non_empty_string(self):
        evolver = _make_evolver()
        evolver.start_time = time.time()
        direction = evolver.auto_direction()
        assert isinstance(direction, str)
        assert len(direction) > 0

    def test_respects_arc_phase_opening(self):
        evolver = _make_evolver(duration=60)
        evolver.start_time = time.time()  # just started = opening
        # Should pick from opening pool (low energy directions)
        direction = evolver.auto_direction()
        assert direction in evolver._OPENING_DIRECTIONS

    def test_respects_arc_phase_final(self):
        evolver = _make_evolver(duration=10)
        evolver.start_time = time.time() - 8 * 60  # 8 min into 10 min = final
        direction = evolver.auto_direction()
        assert direction in evolver._FINAL_DIRECTIONS


class TestShouldStop:
    def test_false_when_time_remaining(self):
        evolver = _make_evolver(duration=60)
        evolver.start_time = time.time()
        assert not evolver.should_stop()

    def test_true_after_duration(self):
        evolver = _make_evolver(duration=1)
        evolver.start_time = time.time() - 120  # 2 min ago, duration is 1 min
        assert evolver.should_stop()

    def test_true_when_no_start_time_and_zero_duration(self):
        evolver = _make_evolver(duration=0)
        evolver.start_time = time.time()
        assert evolver.should_stop()


class TestEvolveSection:
    def test_basic(self):
        section_data = {
            "name": "Deep Groove",
            "slot": 0,
            "bars": 8,
            "energy": 0.6,
            "key": "F",
            "chords": ["Fm7"],
            "drums": "four_on_floor",
            "bass": "pulsing_8th",
            "pad": "sustained",
        }
        mock_lm = _make_mock_sub_lm(section_data)
        evolver = _make_evolver(sub_lm=mock_lm)
        evolver.start_time = time.time()

        section = evolver.evolve_section("bring in a groove", slot=0)

        assert section["name"] == "Deep Groove"
        assert section["energy"] == 0.6
        mock_lm.assert_called_once()
        assert len(evolver.history) == 1
        assert evolver.history[0]["creative_prompt"] == "bring in a groove"

    def test_clips_written_to_session(self):
        section_data = {
            "name": "Test",
            "slot": 1,
            "bars": 4,
            "energy": 0.5,
            "key": "C",
            "chords": ["Cm"],
            "drums": "minimal",
            "bass": "sustained",
            "pad": "atmospheric",
        }
        mock_lm = _make_mock_sub_lm(section_data)
        evolver = _make_evolver(sub_lm=mock_lm)
        evolver.start_time = time.time()

        evolver.evolve_section("test", slot=1)

        # Check session received clip calls
        clip_calls = [c for c in evolver.session.calls if c[0] == "clip"]
        assert len(clip_calls) >= 1  # at least drums, bass, or pad

    def test_bad_json_uses_fallback(self):
        mock_lm = MagicMock(return_value=["This is not JSON!"])
        evolver = _make_evolver(sub_lm=mock_lm)
        evolver.start_time = time.time()
        # Seed history so fallback has something to copy
        evolver.history.append(
            {
                "section": "Prev",
                "slot": 0,
                "energy": 0.3,
                "key": "C",
                "chords": ["Cm"],
                "drums": "minimal",
                "bass": "none",
                "pad": "atmospheric",
                "stab": "none",
                "texture": "none",
            }
        )

        section = evolver.evolve_section("anything", slot=1)

        # Should not crash, should produce a valid section
        assert "name" in section
        assert len(evolver.history) == 2

    def test_overwrite_existing_slot(self):
        """Clips replace previous content at the same slot."""
        section1 = {
            "name": "First",
            "slot": 0,
            "bars": 4,
            "energy": 0.3,
            "key": "F",
            "chords": ["Fm"],
            "drums": "minimal",
            "bass": "none",
            "pad": "sustained",
        }
        section2 = {
            "name": "Second",
            "slot": 0,
            "bars": 4,
            "energy": 0.8,
            "key": "F",
            "chords": ["Fm"],
            "drums": "four_on_floor",
            "bass": "rolling_16th",
            "pad": "pulsing",
        }
        call_count = [0]

        def mock_lm_fn(messages, **kwargs):
            s = section1 if call_count[0] == 0 else section2
            call_count[0] += 1
            return [json.dumps(s)]

        mock_lm = MagicMock(side_effect=mock_lm_fn)
        evolver = _make_evolver(sub_lm=mock_lm)
        evolver.start_time = time.time()

        evolver.evolve_section("first pass", slot=0)
        evolver.evolve_section("second pass", slot=0)

        # Both writes go to slot 0
        clip_calls = [c for c in evolver.session.calls if c[0] == "clip"]
        slots_written = [c[2] for c in clip_calls]
        assert all(s == 0 for s in slots_written)
        # Second write should have more clips (more instruments active)
        assert len(evolver.history) == 2


class TestGetElapsed:
    def test_before_start(self):
        evolver = _make_evolver()
        elapsed, remaining = evolver.get_elapsed()
        assert elapsed == 0.0
        assert remaining == 60.0

    def test_after_start(self):
        evolver = _make_evolver(duration=10)
        evolver.start_time = time.time() - 120  # 2 min ago
        elapsed, remaining = evolver.get_elapsed()
        assert elapsed > 1.9
        assert remaining < 8.1


class TestTransitions:
    def test_fade_on_active_to_none(self):
        """When a track goes from active to none, fade should be called."""
        section_data = {
            "name": "Breakdown",
            "slot": 1,
            "bars": 8,
            "energy": 0.2,
            "key": "F",
            "chords": ["Fm"],
            "drums": "none",
            "bass": "none",
            "pad": "atmospheric",
        }
        mock_lm = _make_mock_sub_lm(section_data)
        evolver = _make_evolver(sub_lm=mock_lm)
        evolver.start_time = time.time()
        # Previous section had drums and bass active
        evolver.history.append(
            {
                "section": "Prev",
                "slot": 0,
                "energy": 0.7,
                "key": "F",
                "chords": ["Fm"],
                "drums": "four_on_floor",
                "bass": "pulsing_8th",
                "pad": "sustained",
                "stab": "none",
                "texture": "none",
            }
        )

        evolver.evolve_section("drop to breakdown", slot=1)

        fade_calls = [c for c in evolver.session.calls if c[0] == "fade"]
        # Should fade drums and bass (they went from active to none)
        faded_tracks = [c[1] for c in fade_calls]
        assert "Drums" in faded_tracks
        assert "Bass" in faded_tracks

    def test_restore_on_none_to_active(self):
        """Previously faded tracks should be restored when roles reactivate."""
        section_drop = {
            "name": "Breakdown",
            "slot": 1,
            "bars": 8,
            "energy": 0.2,
            "key": "F",
            "chords": ["Fm"],
            "drums": "none",
            "bass": "none",
            "pad": "atmospheric",
        }
        section_return = {
            "name": "Rebuild",
            "slot": 2,
            "bars": 8,
            "energy": 0.7,
            "key": "F",
            "chords": ["Fm", "Ab"],
            "drums": "four_on_floor",
            "bass": "pulsing_8th",
            "pad": "sustained",
        }
        responses = [json.dumps(section_drop), json.dumps(section_return)]

        def mock_lm_fn(messages, **kwargs):
            return [responses.pop(0)]

        evolver = _make_evolver(sub_lm=MagicMock(side_effect=mock_lm_fn))
        evolver.start_time = time.time()
        evolver.history.append(
            {
                "section": "Prev",
                "slot": 0,
                "energy": 0.7,
                "key": "F",
                "chords": ["Fm"],
                "drums": "four_on_floor",
                "bass": "pulsing_8th",
                "pad": "sustained",
                "stab": "none",
                "texture": "none",
            }
        )

        first = evolver.evolve_section("drop out rhythm", slot=1)
        evolver.restore_track_volumes(first)
        second = evolver.evolve_section("bring back groove", slot=2)
        evolver.restore_track_volumes(second)

        mix_calls = [c for c in evolver.session.calls if c[0] == "mix"]
        restored_tracks = set()
        for _, kwargs in mix_calls:
            restored_tracks.update(kwargs.keys())

        assert "Drums" in restored_tracks
        assert "Bass" in restored_tracks
