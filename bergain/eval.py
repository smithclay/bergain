"""Evaluation metrics for GEPA optimization of Compose signatures.

Scores RLM trajectories on structural quality without needing Ableton.
Returns dspy.Prediction(score=float, feedback=str) for GEPA's reflection.
"""

import functools
import json
import re
import threading

import dspy

from .music import parse_chord_name

# ---------------------------------------------------------------------------
# Thread-safe clip recording — captures write_clip arguments directly
# ---------------------------------------------------------------------------
# This avoids fragile trajectory text parsing by intercepting write_clip
# at the tool level. Each thread accumulates its own clip specs.

_clip_store = threading.local()


def _record_clip(spec):
    """Record a clip spec dict in thread-local storage."""
    if not hasattr(_clip_store, "clips"):
        _clip_store.clips = []
    _clip_store.clips.append(spec)


def _get_and_clear_recorded_clips():
    """Read and clear recorded clips for the current thread."""
    clips = getattr(_clip_store, "clips", [])
    _clip_store.clips = []
    return clips


def wrap_tools_for_eval(tools):
    """Wrap write_clip in the tools list to record clip specs for the metric.

    Call this on the tools list returned by make_tools() before passing to
    dspy.RLM(). The wrapped write_clip records its parsed arguments in
    thread-local storage, which structural_metric() reads automatically.

    Returns the modified tools list (mutated in-place).
    """
    for i, tool in enumerate(tools):
        if getattr(tool, "__name__", "") == "write_clip":
            original = tool

            @functools.wraps(original)
            def _recording_write_clip(clip_json, _orig=original):
                # Clear on first call per evaluation (detected by empty store)
                try:
                    spec = json.loads(clip_json)
                    _record_clip(spec)
                except (json.JSONDecodeError, TypeError):
                    pass
                return _orig(clip_json)

            tools[i] = _recording_write_clip
            break
    return tools


# ---------------------------------------------------------------------------
# Weights for composite metric (sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "completion": 0.15,
    "arc": 0.20,
    "contrast": 0.15,
    "variety": 0.15,
    "chords": 0.10,
    "efficiency": 0.10,
    "brief_adherence": 0.15,
}

# ---------------------------------------------------------------------------
# Trajectory extraction helpers
# ---------------------------------------------------------------------------


def _extract_text_fields(step):
    """Extract all text content from a trajectory step, regardless of key names.

    DSPy RLM trajectory steps may use different key names depending on version:
    code/action/program, output/observation/result, etc.
    """
    texts = []
    if isinstance(step, dict):
        for key, val in step.items():
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        texts.append(item)
    elif isinstance(step, str):
        texts.append(step)
    return texts


def extract_tool_calls(pred):
    """Extract ordered (tool_name, args_str) tuples from a prediction trajectory.

    Works with any DSPy RLM trajectory format by scanning all text fields
    in each step for tool call patterns. When a loop calls a tool N times,
    counts each result occurrence in the output.
    """
    calls = []
    trajectory = _get_trajectory(pred)

    for step in trajectory:
        texts = _extract_text_fields(step)
        # Get code-like and output-like fields separately
        code_texts = []
        output_texts = []
        if isinstance(step, dict):
            for key, val in step.items():
                if isinstance(val, str):
                    if key in ("output", "observation", "result"):
                        output_texts.append(val)
                    else:
                        code_texts.append(val)
        else:
            code_texts = texts

        # Find tool calls in code
        code_tools = set()
        for text in code_texts:
            for m in re.finditer(r"(\w+)\s*\(", text):
                name = m.group(1)
                if name in _KNOWN_TOOLS:
                    calls.append((name, text))
                    code_tools.add(name)

        # For tools called in loops: count additional occurrences in output.
        # write_clip in a loop produces N result JSONs with "clips_created".
        if "write_clip" in code_tools:
            output_all = "\n".join(output_texts)
            # Count how many write_clip results appear (minus the 1 already counted)
            extra = output_all.count('"clips_created"') - 1
            for _ in range(max(0, extra)):
                calls.append(("write_clip", output_all))

    return calls


def _find_json_objects(text):
    """Find all JSON objects in text, handling nested braces correctly."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
                i += 1
        i += 1
    return objects


def extract_clips_from_trajectory(pred):
    """Extract clip parameter dicts from write_clip() calls in the trajectory.

    Returns list of dicts with keys: name, slot, bars, energy, key, chords,
    drums, bass, pad, stab, texture.
    """
    clips = []
    trajectory = _get_trajectory(pred)

    seen = {}  # (name, slot) -> index in clips list

    def _add_clip(clip):
        key = (clip.get("name", ""), clip.get("slot", -1))
        if key in seen:
            # Replace if the new clip has more fields (richer data)
            existing = clips[seen[key]]
            if len(clip) > len(existing):
                clips[seen[key]] = clip
        else:
            seen[key] = len(clips)
            clips.append(clip)

    for step in trajectory:
        all_text = "\n".join(_extract_text_fields(step))
        all_objects = _find_json_objects(all_text)

        # Strategy 1: Find JSON objects near write_clip calls in code/action fields
        for m in re.finditer(r"write_clip\s*\(", all_text):
            rest = all_text[m.end() :]
            for obj in _find_json_objects(rest[:2000]):
                # Must look like a clip spec (has energy or slot)
                if "energy" in obj or ("slot" in obj and "name" in obj):
                    _add_clip(obj)
                    break  # one clip per write_clip call

        # Strategy 2: Find write_clip output results (has clips_created key)
        for obj in all_objects:
            if "clips_created" in obj and "clip" in obj:
                clip = {
                    "name": obj.get("clip", ""),
                    "slot": obj.get("slot", 0),
                    "energy": obj.get("energy", 0.5),
                    "bars": obj.get("bars", 4),
                }
                _add_clip(clip)

        # Strategy 3: Find clip-spec objects in plan output or variables.
        # When the RLM loops (for scene in plan: write_clip(json.dumps(scene))),
        # the clip specs appear in the plan output, not near write_clip().
        # Look for objects that have the hallmark fields of a clip spec.
        for obj in all_objects:
            if (
                isinstance(obj.get("energy"), (int, float))
                and "name" in obj
                and "slot" in obj
                and any(k in obj for k in ("drums", "bass", "pad", "chords"))
            ):
                _add_clip(obj)

    return clips


_KNOWN_TOOLS = {
    "browse",
    "create_tracks",
    "write_clip",
    "set_mix",
    "set_tempo",
    "get_status",
    "ready_to_submit",
    "fire_scene",
    "fire_clip",
    "play",
    "stop",
    "load_instrument",
    "load_sound",
    "load_effect",
    "load_drum_kit",
    "get_params",
    "set_param",
    "setup_session",
    "compose_next",
    "get_arc_summary",
    "wait",
    "elapsed",
}


def _get_trajectory(pred):
    """Normalize prediction trajectory to a list of step dicts."""
    try:
        raw = pred.trajectory
        if isinstance(raw, str):
            return json.loads(raw)
        elif isinstance(raw, list):
            return raw
    except (AttributeError, json.JSONDecodeError, TypeError):
        pass
    return []


# ---------------------------------------------------------------------------
# Sub-metric functions — each returns (score: float, feedback: str)
# ---------------------------------------------------------------------------


def score_completion(calls, min_clips=6):
    """Check if all milestones were hit: browse, tracks, clips >= min, mix, submit.

    Args:
        calls: list of (tool_name, args_str) tuples from extract_tool_calls()
        min_clips: minimum number of write_clip calls expected
    """
    tool_names = [name for name, _ in calls]

    milestones = {
        "browse": any(n in ("browse", "setup_session") for n in tool_names),
        "tracks": any(n in ("create_tracks", "setup_session") for n in tool_names),
        "clips": sum(1 for n in tool_names if n == "write_clip"),
        "mix": "set_mix" in tool_names,
        "submit_check": "ready_to_submit" in tool_names,
    }

    score = 0.0
    parts = []

    # Browse/tracks: 0.2 each
    if milestones["browse"]:
        score += 0.2
        parts.append("browse OK")
    else:
        parts.append("MISSING browse/setup")

    if milestones["tracks"]:
        score += 0.2
        parts.append("tracks OK")
    else:
        parts.append("MISSING tracks")

    # Clips: 0.3 (proportional to min_clips)
    clip_count = milestones["clips"]
    clip_ratio = min(1.0, clip_count / max(1, min_clips))
    score += 0.3 * clip_ratio
    parts.append(f"clips={clip_count}/{min_clips} ({clip_ratio:.0%})")

    # Mix: 0.15
    if milestones["mix"]:
        score += 0.15
        parts.append("mix OK")
    else:
        parts.append("MISSING mix")

    # Submit check: 0.15
    if milestones["submit_check"]:
        score += 0.15
        parts.append("submit_check OK")
    else:
        parts.append("MISSING ready_to_submit")

    return score, "; ".join(parts)


def score_energy_arc(clips):
    """Score the energy arc shape: range, build->peak->release, no plateaus.

    Args:
        clips: list of clip dicts with 'energy' field
    """
    if not clips:
        return 0.0, "No clips to evaluate"

    energies = [float(c.get("energy", 0.5)) for c in clips]
    n = len(energies)

    if n < 2:
        return 0.3, f"Only {n} clip(s) — insufficient for arc evaluation"

    # Range span: target >= 0.5
    e_min, e_max = min(energies), max(energies)
    span = e_max - e_min
    span_score = min(1.0, span / 0.5)  # 1.0 if span >= 0.5

    # Shape analysis: does it have a build and release?
    peak_idx = energies.index(e_max)
    has_build = peak_idx > 0  # peak not at very start
    has_release = peak_idx < n - 1  # peak not at very end

    shape_score = 0.0
    if has_build and has_release:
        shape_score = 1.0
    elif has_build or has_release:
        shape_score = 0.5

    # Plateau detection: penalize 3+ consecutive same-energy clips
    plateau_penalty = 0.0
    for i in range(2, n):
        if (
            abs(energies[i] - energies[i - 1]) < 0.05
            and abs(energies[i - 1] - energies[i - 2]) < 0.05
        ):
            plateau_penalty += 0.1
    plateau_penalty = min(0.3, plateau_penalty)

    score = (0.4 * span_score + 0.4 * shape_score) - plateau_penalty
    score = max(0.0, min(1.0, score + 0.2))  # base 0.2 for having clips

    parts = [
        f"range={e_min:.2f}-{e_max:.2f} (span={span:.2f})",
        f"peak_at={peak_idx}/{n}",
        f"shape={'build+release' if has_build and has_release else 'partial' if has_build or has_release else 'flat'}",
    ]
    if plateau_penalty > 0:
        parts.append(f"plateau_penalty=-{plateau_penalty:.2f}")

    return score, "; ".join(parts)


def score_contrast(clips):
    """Score local contrast between adjacent scenes.

    Reuses the avg_contrast formula from tools.py get_arc_summary():
    contrast = |delta_energy| + 0.1 * style_changes
    Target: avg_contrast >= 0.15
    """
    if len(clips) < 2:
        return 0.5, "Too few clips for contrast measurement"

    _STYLE_KEYS = ("drums", "bass", "pad", "stab", "texture")
    contrasts = []
    for i in range(1, len(clips)):
        prev, curr = clips[i - 1], clips[i]
        energy_delta = abs(
            float(curr.get("energy", 0.5)) - float(prev.get("energy", 0.5))
        )
        style_changes = sum(
            1 for k in _STYLE_KEYS if curr.get(k, "none") != prev.get(k, "none")
        )
        contrasts.append(energy_delta + style_changes * 0.1)

    avg_contrast = sum(contrasts) / len(contrasts)

    # Monotony detection: 3+ sections with same energy and styles
    monotony = False
    if len(clips) >= 3:
        for i in range(2, len(clips)):
            tail = clips[i - 2 : i + 1]
            energy_flat = all(
                abs(
                    float(tail[j].get("energy", 0.5))
                    - float(tail[0].get("energy", 0.5))
                )
                < 0.1
                for j in range(1, len(tail))
            )
            styles_same = all(
                tail[j].get(k, "none") == tail[0].get(k, "none")
                for j in range(1, len(tail))
                for k in _STYLE_KEYS
            )
            if energy_flat and styles_same:
                monotony = True
                break

    # Score: normalized around 0.15 target
    score = min(1.0, avg_contrast / 0.20)
    if monotony:
        score = max(0.0, score - 0.3)

    parts = [f"avg_contrast={avg_contrast:.3f}"]
    if monotony:
        parts.append("MONOTONY detected (3+ flat sections)")

    return score, "; ".join(parts)


def score_variety(clips):
    """Score style combo uniqueness.

    unique_style_combos / total_clips ratio >= 0.8 is ideal.
    No combo used 3+ times.
    """
    if not clips:
        return 0.0, "No clips to evaluate"

    _STYLE_KEYS = ("drums", "bass", "pad", "stab", "texture")
    combos = []
    for c in clips:
        combo = tuple(c.get(k, "none") for k in _STYLE_KEYS)
        combos.append(combo)

    unique = len(set(combos))
    total = len(combos)
    ratio = unique / total

    # Check for overused combos (3+)
    combo_counts = {}
    for combo in combos:
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    overused = {"/".join(c): n for c, n in combo_counts.items() if n >= 3}

    score = min(1.0, ratio / 0.8)  # 1.0 if ratio >= 0.8
    if overused:
        score = max(0.0, score - 0.2 * len(overused))

    # Also check chord progression variety
    chord_progs = ["/".join(c.get("chords", [])) for c in clips if c.get("chords")]
    unique_chords = len(set(chord_progs))
    total_chords = max(1, len(chord_progs))
    chord_ratio = unique_chords / total_chords

    # Blend style and chord variety
    score = 0.7 * score + 0.3 * min(1.0, chord_ratio / 0.7)

    parts = [
        f"style_combos={unique}/{total} ({ratio:.0%})",
        f"chord_progs={unique_chords}/{total_chords} ({chord_ratio:.0%})",
    ]
    if overused:
        parts.append(f"overused: {overused}")

    return score, "; ".join(parts)


def score_chord_coherence(clips):
    """Score chord-key consistency across clips.

    Checks: are the chords in the declared key?
    """
    if not clips:
        return 0.5, "No clips to evaluate"

    in_key_count = 0
    total_chords = 0
    keys_used = set()

    for c in clips:
        key = c.get("key", "C")
        chords = c.get("chords", [])
        keys_used.add(key)

        if not chords:
            continue

        # Get key root pitch class
        key_pc = parse_chord_name(key + "m")[0]  # treat key as minor root
        # Scale degrees for natural minor: 0, 2, 3, 5, 7, 8, 10
        minor_scale = {(key_pc + d) % 12 for d in [0, 2, 3, 5, 7, 8, 10]}
        # Also include major scale: 0, 2, 4, 5, 7, 9, 11
        major_scale = {(key_pc + d) % 12 for d in [0, 2, 4, 5, 7, 9, 11]}
        allowed = minor_scale | major_scale

        for chord in chords:
            total_chords += 1
            try:
                root_pc, _ = parse_chord_name(chord)
                if root_pc in allowed:
                    in_key_count += 1
            except (ValueError, IndexError, KeyError):
                pass

    if total_chords == 0:
        return 0.5, "No chords found"

    coherence = in_key_count / total_chords

    # Slight penalty for using too many different keys (unless intentional)
    key_penalty = 0.0
    if len(keys_used) > 2:
        key_penalty = 0.1 * (len(keys_used) - 2)

    score = max(0.0, min(1.0, coherence - key_penalty))

    return (
        score,
        f"chords_in_key={in_key_count}/{total_chords} ({coherence:.0%}); keys_used={keys_used}",
    )


def score_efficiency(pred, max_iterations=30):
    """Score trajectory efficiency: iterations used / max_iterations.

    Sweet spot: 0.3-0.8 ratio. Penalize >0.8 (thrashing) and <0.3 (too simplistic).
    """
    trajectory = _get_trajectory(pred)
    n_steps = len(trajectory)

    if max_iterations == 0:
        return 0.5, "max_iterations=0, cannot evaluate"

    ratio = n_steps / max_iterations

    if 0.3 <= ratio <= 0.8:
        score = 1.0
    elif ratio < 0.3:
        # Too few steps — possibly too simplistic
        score = ratio / 0.3  # linear ramp from 0 to 1
    else:
        # Too many steps — thrashing
        score = max(0.0, 1.0 - (ratio - 0.8) / 0.2)  # linear ramp from 1 to 0

    return score, f"steps={n_steps}/{max_iterations} (ratio={ratio:.2f})"


def score_brief_adherence(example, clips, judge_lm=None):
    """Score how well the composition matches the creative brief.

    Uses LLM-as-judge if judge_lm is provided, otherwise returns neutral 0.5.
    """
    brief = example.brief if hasattr(example, "brief") else str(example)

    if not judge_lm or not clips:
        return 0.5, "No judge LM or no clips — neutral score"

    # Build a summary of what was composed
    summary_parts = []
    for c in clips:
        parts = [f"'{c.get('name', '?')}'"]
        if "energy" in c:
            parts.append(f"E={c['energy']}")
        if "key" in c:
            parts.append(f"key={c['key']}")
        if "chords" in c:
            parts.append(f"chords={c['chords']}")
        for role in ("drums", "bass", "pad", "stab", "texture"):
            if c.get(role) and c[role] != "none":
                parts.append(f"{role}={c[role]}")
        summary_parts.append(", ".join(parts))

    composition_summary = "\n".join(summary_parts)

    prompt = (
        f"Rate how well this composition plan matches the creative brief.\n\n"
        f"BRIEF: {brief}\n\n"
        f"COMPOSITION SCENES:\n{composition_summary}\n\n"
        f"Score 0-10 on genre/mood/tempo alignment. "
        f"Respond with ONLY a JSON object: "
        f'{{"score": <0-10>, "rationale": "<brief explanation>"}}'
    )

    try:
        result = judge_lm(messages=[{"role": "user", "content": prompt}])
        raw = result[0]
        if isinstance(raw, dict):
            text = raw.get("text") or raw.get("content") or str(raw)
        else:
            text = str(raw)

        m = re.search(r"\{.*\}", text, re.S)
        if m:
            data = json.loads(m.group())
            raw_score = float(data.get("score", 5)) / 10.0
            rationale = data.get("rationale", "")
            return max(0.0, min(1.0, raw_score)), f"judge={raw_score:.1f}; {rationale}"
    except Exception as e:
        return 0.5, f"Judge LM error: {e}"

    return 0.5, "Could not parse judge response"


# ---------------------------------------------------------------------------
# Composite metric — the main entry point for GEPA
# ---------------------------------------------------------------------------


def structural_metric(example, pred, trace=None, **kwargs):
    """Composite structural metric for GEPA optimization.

    Combines all sub-metrics with weights, returns dspy.Prediction(score, feedback).
    Uses directly-recorded clip specs when available (via wrap_tools_for_eval),
    falling back to trajectory text parsing.
    """
    # Prefer recorded clips (reliable) over trajectory parsing (fragile)
    recorded = _get_and_clear_recorded_clips()
    clips = recorded if recorded else extract_clips_from_trajectory(pred)
    calls = extract_tool_calls(pred)

    scores = {}
    feedbacks = []

    # 1. Completion
    scores["completion"], fb = score_completion(calls, min_clips=6)
    feedbacks.append(f"COMPLETION ({scores['completion']:.2f}): {fb}")

    # 2. Energy arc
    scores["arc"], fb = score_energy_arc(clips)
    feedbacks.append(f"ENERGY ARC ({scores['arc']:.2f}): {fb}")

    # 3. Local contrast
    scores["contrast"], fb = score_contrast(clips)
    feedbacks.append(f"CONTRAST ({scores['contrast']:.2f}): {fb}")

    # 4. Style variety
    scores["variety"], fb = score_variety(clips)
    feedbacks.append(f"VARIETY ({scores['variety']:.2f}): {fb}")

    # 5. Chord coherence
    scores["chords"], fb = score_chord_coherence(clips)
    feedbacks.append(f"CHORDS ({scores['chords']:.2f}): {fb}")

    # 6. Efficiency
    max_iters = kwargs.get("max_iterations", 30)
    scores["efficiency"], fb = score_efficiency(pred, max_iterations=max_iters)
    feedbacks.append(f"EFFICIENCY ({scores['efficiency']:.2f}): {fb}")

    # 7. Brief adherence (optional LLM judge)
    judge_lm = kwargs.get("judge_lm", None)
    scores["brief_adherence"], fb = score_brief_adherence(
        example, clips, judge_lm=judge_lm
    )
    feedbacks.append(f"BRIEF ADHERENCE ({scores['brief_adherence']:.2f}): {fb}")

    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    feedback = "\n".join(feedbacks)
    feedback += f"\n\nCOMPOSITE SCORE: {total:.3f}"
    feedback += f"\nClips extracted: {len(clips)}, Tool calls: {len(calls)}"

    return dspy.Prediction(score=total, feedback=feedback)
