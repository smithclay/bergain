"""Tests for bergain.eval metric functions.

Uses synthetic trajectory data to verify score ranges and feedback quality.
"""

import json

import dspy

from bergain.eval import (
    WEIGHTS,
    extract_clips_from_trajectory,
    extract_tool_calls,
    score_brief_adherence,
    score_chord_coherence,
    score_completion,
    score_contrast,
    score_efficiency,
    score_energy_arc,
    score_variety,
    structural_metric,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic trajectory builders
# ---------------------------------------------------------------------------


class FakePrediction:
    """Minimal prediction object with a trajectory attribute."""

    def __init__(self, trajectory):
        if isinstance(trajectory, list):
            self.trajectory = trajectory
        else:
            self.trajectory = json.dumps(trajectory)


def _make_trajectory(steps):
    """Build a trajectory from a list of (code, output) tuples."""
    return [{"code": code, "output": output} for code, output in steps]


def _good_trajectory():
    """A well-formed trajectory that hits all milestones."""
    return _make_trajectory(
        [
            ('browse("909", "Drums")', "909 Core Kit.adg\nClap 909.aif"),
            (
                'create_tracks(json.dumps([{"name":"Drums","drum_kit":"909 Core Kit.adg"},'
                '{"name":"Bass","instrument":"Operator"},'
                '{"name":"Pad","sound":"Warm Pad"}]))',
                '{"tracks_created": 3, "names": ["Drums", "Bass", "Pad"]}',
            ),
            (
                'write_clip(json.dumps({"name":"Ambient","slot":0,"bars":8,'
                '"energy":0.2,"key":"F","chords":["Fm"],'
                '"drums":"minimal","bass":"none","pad":"atmospheric"}))',
                '{"clip":"Ambient","slot":0,"clips_created":2,"bars":8,"energy":0.2}',
            ),
            (
                'write_clip(json.dumps({"name":"Build","slot":1,"bars":4,'
                '"energy":0.4,"key":"F","chords":["Fm","Cm"],'
                '"drums":"four_on_floor","bass":"pulsing_8th","pad":"sustained"}))',
                '{"clip":"Build","slot":1,"clips_created":3,"bars":4,"energy":0.4}',
            ),
            (
                'write_clip(json.dumps({"name":"Drive","slot":2,"bars":4,'
                '"energy":0.6,"key":"F","chords":["Fm","Abmaj7"],'
                '"drums":"four_on_floor","bass":"rolling_16th","pad":"pulsing"}))',
                '{"clip":"Drive","slot":2,"clips_created":3,"bars":4,"energy":0.6}',
            ),
            (
                'write_clip(json.dumps({"name":"Peak","slot":3,"bars":4,'
                '"energy":0.85,"key":"F","chords":["Fm","Cm","Abmaj7","Bbm"],'
                '"drums":"breakbeat","bass":"rolling_16th","pad":"arpeggiated"}))',
                '{"clip":"Peak","slot":3,"clips_created":3,"bars":4,"energy":0.85}',
            ),
            (
                'write_clip(json.dumps({"name":"Breakdown","slot":4,"bars":8,'
                '"energy":0.3,"key":"F","chords":["Fm7"],'
                '"drums":"none","bass":"sustained","pad":"swells"}))',
                '{"clip":"Breakdown","slot":4,"clips_created":2,"bars":8,"energy":0.3}',
            ),
            (
                'write_clip(json.dumps({"name":"Peak2","slot":5,"bars":4,'
                '"energy":0.8,"key":"F","chords":["Fm","Cm7"],'
                '"drums":"four_on_floor","bass":"offbeat_8th","pad":"pulsing"}))',
                '{"clip":"Peak2","slot":5,"clips_created":3,"bars":4,"energy":0.8}',
            ),
            (
                'write_clip(json.dumps({"name":"Outro","slot":6,"bars":8,'
                '"energy":0.15,"key":"F","chords":["Fm"],'
                '"drums":"sparse_perc","bass":"none","pad":"atmospheric"}))',
                '{"clip":"Outro","slot":6,"clips_created":2,"bars":8,"energy":0.15}',
            ),
            (
                'set_mix(json.dumps({"Drums":0.9,"Bass":0.85,"Pad":0.7}))',
                '{"mixed": ["Drums", "Bass", "Pad"]}',
            ),
            ("print(get_status())", "Tempo: 130 BPM\nPlaying: yes"),
            ("print(ready_to_submit())", "READY — call SUBMIT"),
        ]
    )


def _monotone_trajectory():
    """A trajectory with no variety — all scenes same energy/style."""
    clips = []
    for i in range(6):
        clips.append(
            (
                f'write_clip(json.dumps({{"name":"Scene{i}","slot":{i},"bars":4,'
                f'"energy":0.5,"key":"C","chords":["Cm"],'
                f'"drums":"four_on_floor","bass":"sustained","pad":"sustained"}}))',
                f'{{"clip":"Scene{i}","slot":{i},"clips_created":3,"bars":4,"energy":0.5}}',
            )
        )
    return _make_trajectory(
        [
            ('browse("test")', "Test.adg"),
            (
                'create_tracks(json.dumps([{"name":"Drums","drum_kit":"Kit.adg"},'
                '{"name":"Bass","instrument":"Op"},{"name":"Pad","sound":"Pad"}]))',
                '{"tracks_created":3}',
            ),
            *clips,
            ('set_mix(json.dumps({"Drums":0.9}))', '{"mixed":["Drums"]}'),
            ("print(ready_to_submit())", "READY"),
        ]
    )


# ---------------------------------------------------------------------------
# extract_tool_calls tests
# ---------------------------------------------------------------------------


def test_extract_tool_calls_good():
    pred = FakePrediction(_good_trajectory())
    calls = extract_tool_calls(pred)
    names = [name for name, _ in calls]
    assert "browse" in names
    assert "create_tracks" in names
    assert names.count("write_clip") == 7
    assert "set_mix" in names
    assert "ready_to_submit" in names


def test_extract_tool_calls_empty():
    pred = FakePrediction([])
    calls = extract_tool_calls(pred)
    assert calls == []


# ---------------------------------------------------------------------------
# extract_clips_from_trajectory tests
# ---------------------------------------------------------------------------


def test_extract_clips_good():
    pred = FakePrediction(_good_trajectory())
    clips = extract_clips_from_trajectory(pred)
    assert len(clips) >= 6
    # Check first clip has expected fields
    assert clips[0]["name"] == "Ambient"
    assert clips[0]["energy"] == 0.2


def test_extract_clips_empty():
    pred = FakePrediction([])
    clips = extract_clips_from_trajectory(pred)
    assert clips == []


# ---------------------------------------------------------------------------
# score_completion tests
# ---------------------------------------------------------------------------


def test_completion_full():
    pred = FakePrediction(_good_trajectory())
    calls = extract_tool_calls(pred)
    score, fb = score_completion(calls, min_clips=6)
    assert 0.9 <= score <= 1.0
    assert "browse OK" in fb
    assert "tracks OK" in fb
    assert "mix OK" in fb


def test_completion_empty():
    score, fb = score_completion([], min_clips=6)
    assert score == 0.0
    assert "MISSING" in fb


def test_completion_partial():
    calls = [("browse", ""), ("create_tracks", ""), ("write_clip", "")]
    score, fb = score_completion(calls, min_clips=6)
    assert 0.0 < score < 1.0
    assert "MISSING mix" in fb


# ---------------------------------------------------------------------------
# score_energy_arc tests
# ---------------------------------------------------------------------------


def test_energy_arc_good():
    clips = [
        {"energy": 0.2},
        {"energy": 0.4},
        {"energy": 0.7},
        {"energy": 0.85},
        {"energy": 0.3},
        {"energy": 0.8},
        {"energy": 0.15},
    ]
    score, fb = score_energy_arc(clips)
    assert 0.7 <= score <= 1.0
    assert "span=" in fb


def test_energy_arc_flat():
    clips = [{"energy": 0.5}] * 6
    score, fb = score_energy_arc(clips)
    assert score < 0.5  # bad arc
    assert "plateau" in fb.lower() or "span=0.00" in fb


def test_energy_arc_empty():
    score, fb = score_energy_arc([])
    assert score == 0.0


def test_energy_arc_single():
    score, fb = score_energy_arc([{"energy": 0.5}])
    assert 0.0 <= score <= 0.5


# ---------------------------------------------------------------------------
# score_contrast tests
# ---------------------------------------------------------------------------


def test_contrast_varied():
    clips = [
        {"energy": 0.2, "drums": "minimal", "bass": "none", "pad": "atmospheric"},
        {
            "energy": 0.6,
            "drums": "four_on_floor",
            "bass": "pulsing_8th",
            "pad": "sustained",
        },
        {"energy": 0.3, "drums": "none", "bass": "sustained", "pad": "swells"},
        {
            "energy": 0.8,
            "drums": "breakbeat",
            "bass": "rolling_16th",
            "pad": "arpeggiated",
        },
    ]
    score, fb = score_contrast(clips)
    assert score > 0.5
    assert "avg_contrast=" in fb


def test_contrast_monotone():
    clips = [
        {
            "energy": 0.5,
            "drums": "four_on_floor",
            "bass": "sustained",
            "pad": "sustained",
        },
    ] * 4
    score, fb = score_contrast(clips)
    assert score < 0.3
    assert "MONOTONY" in fb


def test_contrast_too_few():
    score, fb = score_contrast([{"energy": 0.5}])
    assert score == 0.5


# ---------------------------------------------------------------------------
# score_variety tests
# ---------------------------------------------------------------------------


def test_variety_unique():
    clips = [
        {"drums": "minimal", "bass": "none", "pad": "atmospheric"},
        {"drums": "four_on_floor", "bass": "pulsing_8th", "pad": "sustained"},
        {"drums": "breakbeat", "bass": "rolling_16th", "pad": "arpeggiated"},
        {"drums": "none", "bass": "sustained", "pad": "swells"},
        {"drums": "shuffle", "bass": "walking", "pad": "pulsing"},
        {"drums": "half_time", "bass": "offbeat_8th", "pad": "atmospheric"},
    ]
    # Add unique chords
    for i, c in enumerate(clips):
        c["chords"] = [f"chord{i}"]
    score, fb = score_variety(clips)
    assert score > 0.7
    assert "style_combos=" in fb


def test_variety_all_same():
    clips = [
        {
            "drums": "four_on_floor",
            "bass": "sustained",
            "pad": "sustained",
            "chords": ["Cm"],
        },
    ] * 6
    score, fb = score_variety(clips)
    assert score < 0.5


def test_variety_empty():
    score, fb = score_variety([])
    assert score == 0.0


# ---------------------------------------------------------------------------
# score_chord_coherence tests
# ---------------------------------------------------------------------------


def test_chord_coherence_in_key():
    clips = [
        {"key": "F", "chords": ["Fm", "Cm", "Ab", "Bb"]},
        {"key": "F", "chords": ["Fm7", "Cm7"]},
    ]
    score, fb = score_chord_coherence(clips)
    assert score > 0.7
    assert "chords_in_key=" in fb


def test_chord_coherence_out_of_key():
    clips = [
        {"key": "C", "chords": ["F#", "G#", "D#"]},
    ]
    score, fb = score_chord_coherence(clips)
    assert score < 0.7


def test_chord_coherence_empty():
    score, fb = score_chord_coherence([])
    assert score == 0.5


# ---------------------------------------------------------------------------
# score_efficiency tests
# ---------------------------------------------------------------------------


def test_efficiency_sweet_spot():
    # 12 steps out of 30 = 0.4 ratio (in sweet spot)
    pred = FakePrediction([{"code": "", "output": ""}] * 12)
    score, fb = score_efficiency(pred, max_iterations=30)
    assert score == 1.0
    assert "ratio=0.40" in fb


def test_efficiency_too_few():
    pred = FakePrediction([{"code": "", "output": ""}] * 3)
    score, fb = score_efficiency(pred, max_iterations=30)
    assert score < 0.5  # 3/30 = 0.1, below 0.3


def test_efficiency_too_many():
    pred = FakePrediction([{"code": "", "output": ""}] * 29)
    score, fb = score_efficiency(pred, max_iterations=30)
    assert score < 0.5  # 29/30 = 0.97, above 0.8


# ---------------------------------------------------------------------------
# score_brief_adherence tests
# ---------------------------------------------------------------------------


def test_brief_adherence_no_judge():
    example = dspy.Example(brief="Dark techno in F minor").with_inputs("brief")
    score, fb = score_brief_adherence(example, [{"name": "Test"}])
    assert score == 0.5
    assert "neutral" in fb.lower()


def test_brief_adherence_no_clips():
    example = dspy.Example(brief="Dark techno in F minor").with_inputs("brief")
    score, fb = score_brief_adherence(example, [], judge_lm=None)
    assert score == 0.5


# ---------------------------------------------------------------------------
# structural_metric composite tests
# ---------------------------------------------------------------------------


def test_structural_metric_good_trajectory():
    example = dspy.Example(brief="Dark techno in F minor, 130 BPM").with_inputs("brief")
    pred = FakePrediction(_good_trajectory())
    result = structural_metric(example, pred, max_iterations=30)
    assert hasattr(result, "score")
    assert hasattr(result, "feedback")
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.4  # good trajectory should score decently
    assert "COMPLETION" in result.feedback
    assert "ENERGY ARC" in result.feedback
    assert "COMPOSITE SCORE" in result.feedback


def test_structural_metric_empty_trajectory():
    example = dspy.Example(brief="Dark techno").with_inputs("brief")
    pred = FakePrediction([])
    result = structural_metric(example, pred, max_iterations=30)
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.3  # empty trajectory should score poorly


def test_structural_metric_monotone():
    example = dspy.Example(brief="Minimal techno").with_inputs("brief")
    pred = FakePrediction(_monotone_trajectory())
    result = structural_metric(example, pred, max_iterations=30)
    assert 0.0 <= result.score <= 1.0
    # Should score lower than the good trajectory
    good_pred = FakePrediction(_good_trajectory())
    good_result = structural_metric(example, good_pred, max_iterations=30)
    assert result.score < good_result.score


# ---------------------------------------------------------------------------
# Weight sanity check
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"


def test_all_scores_bounded():
    """All score functions should return values in [0, 1]."""
    # Completions
    for calls in [[], [("browse", "")], [("browse", ""), ("create_tracks", "")]]:
        s, _ = score_completion(calls, min_clips=6)
        assert 0.0 <= s <= 1.0, f"score_completion out of range: {s}"

    # Energy arc
    for clips in [[], [{"energy": 0.5}], [{"energy": 0.2}, {"energy": 0.8}]]:
        s, _ = score_energy_arc(clips)
        assert 0.0 <= s <= 1.0, f"score_energy_arc out of range: {s}"

    # Contrast
    for clips in [
        [{"energy": 0.5}],
        [{"energy": 0.2}, {"energy": 0.8}],
    ]:
        s, _ = score_contrast(clips)
        assert 0.0 <= s <= 1.0, f"score_contrast out of range: {s}"

    # Variety
    s, _ = score_variety([])
    assert 0.0 <= s <= 1.0
    s, _ = score_variety([{"drums": "a", "bass": "b", "pad": "c", "chords": ["Cm"]}])
    assert 0.0 <= s <= 1.0

    # Chord coherence
    s, _ = score_chord_coherence([])
    assert 0.0 <= s <= 1.0
    s, _ = score_chord_coherence([{"key": "C", "chords": ["Cm"]}])
    assert 0.0 <= s <= 1.0

    # Efficiency
    for n in [0, 5, 15, 25, 30]:
        pred = FakePrediction([{"code": "", "output": ""}] * n)
        s, _ = score_efficiency(pred, max_iterations=30)
        assert 0.0 <= s <= 1.0, f"score_efficiency out of range: {s} (n={n})"
