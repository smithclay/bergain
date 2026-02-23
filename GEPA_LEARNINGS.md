# GEPA Integration Learnings

## What We Built

An offline optimization pipeline that uses DSPy's GEPA (Genetic-Pareto) optimizer to evolve the `Compose` signature docstring — the ~2KB prompt that drives the RLM's music composition behavior. Evaluation runs against `StubSession` (no Ableton needed) using 7 structural quality metrics.

### Files

| File | Purpose |
|------|---------|
| `bergain/eval.py` | 7 weighted sub-metrics + clip recording mechanism |
| `bergain/stub.py` | StubSession extracted from tests for shared use |
| `bergain/trainset.py` | 20 diverse briefs (15 train / 5 val) |
| `scripts/optimize_compose.py` | CLI: `--dry-run`, `--budget light\|medium\|heavy`, `--compare` |
| `bergain/compose.py` | Added `load_optimized_signature()` helper |
| `bergain/cli.py` | Added `--optimized` flag |
| `tests/test_eval.py` | 30 tests covering all metric functions |

## Metric Design

Composite score (0.0–1.0) from 7 weighted dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Completion | 0.15 | All milestones hit: browse, tracks, clips ≥ 6, mix, submit |
| Energy arc | 0.20 | Range span ≥ 0.5, build→peak→release shape, no plateaus |
| Contrast | 0.15 | Adjacent scene variety (`\|Δenergy\| + 0.1 * style_changes`) |
| Variety | 0.15 | Unique style combos / total clips, chord progression diversity |
| Chords | 0.10 | Chords-in-key consistency, key coherence across clips |
| Efficiency | 0.10 | Steps used / max_iterations ratio — sweet spot 0.3–0.8 |
| Brief adherence | 0.15 | LLM-as-judge (optional, defaults to 0.5 neutral) |

Each sub-metric returns `(score, feedback_string)`. The textual feedback is critical — GEPA's reflection mechanism uses it to diagnose problems and propose prompt mutations.

## Baseline Results

5 training briefs, GPT-5 via OpenRouter, no judge LM, ~20 min total:

| Brief | Score | Clips | Notes |
|-------|-------|-------|-------|
| Dark Berlin techno (F minor, 130 BPM) | 0.881 | 8 | |
| Minimal techno (128 BPM) | 0.881 | 7 | |
| Industrial techno (A minor, 140 BPM) | 0.881 | 7 | |
| Deep house (D minor, 122 BPM) | 0.925 | 7 | Best — more RLM iterations |
| Acid house (138 BPM) | 0.876 | 8 | Slight chord repetition |
| **Average** | **0.888** | | |

The main score ceiling comes from:
- **Efficiency** (0.56): RLM finishes in 5 steps out of 30 allowed (ratio=0.17, below 0.3 sweet spot)
- **Brief adherence** (0.50): Neutral without `--judge-lm`

## Hard-Won Lesson: Don't Parse Trajectory Text

This was the single biggest debugging effort. The RLM generates Python code that calls `write_clip(json.dumps(spec))` — but the code patterns vary wildly between runs:

### Pattern 1: Inline JSON (rare, easy)
```python
write_clip('{"name": "Intro", "slot": 0, "energy": 0.2, "drums": "minimal", ...}')
```

### Pattern 2: Loop over plan variable (common, hard)
```python
for scene in plan["scenes"]:
    result = write_clip(json.dumps(scene))
```
The clip specs only exist in the `plan` variable, not as literals near `write_clip(`.

### Pattern 3: Variable reference (common, impossible to parse)
```python
spec = {"name": "Intro", "slot": 0, ...}
write_clip(json.dumps(spec))
```
The spec is a Python dict on a separate line, not valid JSON (single quotes, etc.).

### What we tried (trajectory text parsing)

1. **Regex `\{[^}]+\}`** — Failed on nested JSON (arrays inside objects)
2. **Brace-depth JSON parser** (`_find_json_objects`) — Better, found inline JSON
3. **Strategy 1** (JSON near `write_clip(` calls) — Only works for Pattern 1
4. **Strategy 2** (output results with `clips_created` key) — Finds clips but without style fields (drums/bass/pad) since `write_clip` return value doesn't include them
5. **Strategy 3** (plan-like objects with hallmark fields) — Catches Pattern 2 sometimes, but requires the plan objects to be printed to stdout
6. **Dict-based dedup** preferring richer objects — Helped merge sparse Strategy 2 clips with rich Strategy 3 clips

Results after all trajectory parsing attempts: **average 0.604** (Briefs 2&3 still at 0.368 with 0 clips).

### What actually works: Tool-level interception

`wrap_tools_for_eval(tools)` wraps the `write_clip` function to record its parsed JSON argument in `threading.local()` storage before passing through to the original. The metric reads recorded clips via `_get_and_clear_recorded_clips()`.

```python
# In _build_student:
tools, _, _ = make_tools(session, min_clips=min_clips)
wrap_tools_for_eval(tools)  # <-- this line fixed everything
```

Results: **average 0.888** — every brief now extracts 7-8 clips with full style and chord data.

The trajectory text parsing is kept as a fallback for cases where the recording mechanism isn't available (e.g., evaluating a saved trajectory offline).

## Architecture Notes

- **StubSession** records calls but the metric can't access it (it's buried inside tool closures). That's why tool-level wrapping was needed.
- **Thread safety** matters because GEPA uses `num_threads` for parallel evaluation. `threading.local()` gives each thread its own clip accumulator.
- **Write_clip return value** includes `{clip, slot, clips_created, bars, energy, details}` but NOT style fields (drums/bass/pad/stab/texture) or chords or key. These only exist in the INPUT argument.
- Each brief takes ~3-6 min (5-12 RLM calls × 15-90s each). A 5-brief dry run takes ~20 min.
- `cache=False` on the task LM is essential — without it, DSPy replays cached individual LM calls, producing identical trajectories regardless of prompt changes.

## Optimization Run: Light Budget with Gemini 3 Flash

### Setup
- **Model**: `openrouter/google/gemini-3-flash-preview` (much cheaper/faster than GPT-5)
- **Budget**: light (`auto="light"`)
- **Duration**: 3.6 hours, 153 iterations, 762 metric evaluations, 6429 RLM calls
- **GEPA metric signature**: Must accept 5 args `(gold, pred, trace, pred_name, pred_trace)` — the 3-arg version fails

### GEPA Internal Scores
- **Baseline (iteration 0)**: 0.858 on val set
- **Best program found (Program 10)**: 0.895 avg across val set
- **Pareto front**: 5 programs (IDs: 6, 7, 10, 12, 13), best individual task scores up to 0.914

### Instruction Extraction Gotcha

`result.generate.signature.instructions` and `result.generate.signature.__doc__` both returned `None`. GEPA stores its state in `gepa_state.bin` (pickle), not on the returned program object. To extract optimized instructions:

```python
import pickle
with open('output/gepa/logs/gepa_state.bin', 'rb') as f:
    state = pickle.load(f)
best_instructions = state['program_candidates'][10]['generate_action']
```

The key is `program_candidates` — a list of dicts where each has `generate_action` (the evolved instruction text) and `extract` (the output extraction prompt).

### What GEPA Discovered

The optimized instructions (Program 10, 3752 chars) evolved from the original hand-written docstring into a structured "EXPERT DAW-INTEGRATOR" prompt. Key changes:

1. **Explicit style enum constraints** — Lists every valid keyword for drums/bass/pad (the original left this implicit in the docstring)
2. **Batch strategy** — "Write clips in batches of 3 scenes per turn" (prevents timeout, more consistent execution)
3. **Mandatory energy arc template** — `Ambient(0.1-0.2) → Build(0.3-0.5) → Peak(0.8-1.0) → Breakdown → Peak → Outro` with specific ranges
4. **JSON safety** — "Use `json.dumps()`, NOT f-strings" (f-string braces conflict with JSON)
5. **Chord correction** — "Manually fix LLM errors like Abmt7 to Abm7"
6. **Variety awareness** — "Each scene must have unique chord progression or style combo" (directly addresses the variety metric)
7. **Mix defaults** — Specific dB levels: Drums -3, Bass -6, Pad -12
8. **Status persistence note** — "get_status() may report Tracks: (none) even after success — ignore and proceed"

### A/B Comparison (held-out val briefs)

Both runs used Gemini 3 Flash, 5 val briefs, no judge LM:

| Brief | Original | Optimized | Delta |
|-------|----------|-----------|-------|
| Dark minimal techno (E minor, 134 BPM) | 0.881 | **0.892** | +0.011 |
| Warm deep house (C major, 120 BPM) | **0.892** | 0.879 | -0.013 |
| Atmospheric breaks (D minor, 130 BPM) | 0.881 | **0.892** | +0.011 |
| Acid techno (G minor, 142 BPM) | 0.819 | **0.892** | **+0.073** |
| Chill electronic (F major, 100 BPM) | 0.851 | **0.862** | +0.011 |
| **Average** | **0.865** | **0.883** | **+0.019** |

### Analysis

**The win is reliability, not peak performance.** The optimized instructions:
- Raised the floor from 0.819 → 0.862 (biggest win on acid techno: +7.3%)
- Tightened the spread from 0.073 → 0.030
- Improved efficiency score from 0.56 → 0.67 (6 steps vs 5, closer to the 0.3 sweet spot)
- Won 4/5 briefs, lost 1 (deep house, -0.013)

**Gemini Flash as RLM**: Much faster (~3-6s/call vs 30-90s for GPT-5) but less reliable at following tool schemas. Common errors during optimization:
- `KeyError: 'name'` — missing required fields in write_clip JSON
- `TypeError: the JSON object must be str, bytes or bytearray, not dict` — passing Python dicts instead of JSON strings
- `'NoneType' object has no attribute 'strip'` — model returning None for code generation

These errors are actually useful signal for GEPA — they penalize instructions that don't explicitly mention the required JSON format, which is why the optimized version added the "JSON Precision" and example payload sections.

### Cost

762 metric evaluations × ~8 RLM calls each × ~$0.001/call (Gemini Flash) ≈ **~$6 total**. Very affordable for a light optimization run.

## Lessons for Future Runs

1. **Model choice matters for optimization target**: We optimized with Gemini Flash but the production model is GPT-5. The optimized instructions may be over-fitted to Gemini's failure modes (e.g., explicit JSON format warnings). Consider running optimization with the same model as production, or with a diverse set.

2. **Efficiency metric may need recalibration**: Both original and optimized consistently score 0.56-0.67 on efficiency. The RLM naturally finishes in 5-6 steps. The 0.3-0.8 sweet spot was designed for a 30-iteration budget; with efficient models, 5-6 steps IS optimal. Consider lowering the sweet spot to 0.15-0.5.

3. **GEPA went more prescriptive, not less**: Despite RLM_LESSONS.md saying "less prescriptive = better" (v7 finding), GEPA produced MORE detailed instructions (3752 chars vs ~2KB original). This might be because Gemini Flash needs more hand-holding than GPT-5. The "less prescriptive" finding may be model-dependent.

4. **Brief adherence is untested**: All runs used neutral 0.50 for brief adherence (no judge LM). Adding `--judge-lm` would enable the 0.15-weight dimension and potentially change the optimization landscape.

5. **Extract instruction from `gepa_state.bin`**: Don't rely on `result.generate.signature.instructions` — it's often None. Always load the pickle state and read `program_candidates[best_id]['generate_action']`.

## Next Steps

- Run optimization with GPT-5 as the task model (slower but matches production)
- Add `--judge-lm` for brief adherence scoring
- Recalibrate efficiency sweet spot based on observed step counts
- Try `--budget medium` for deeper exploration (current light run found improvements up to iteration 57, suggesting more budget could help)
- Compare GEPA-optimized instructions on actual Ableton compositions (Tier 2 evaluation)
