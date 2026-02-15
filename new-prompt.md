# DJ Prompt Optimization Log

## Baseline (before changes)
- Prompt: built-in DJ_INSTRUCTIONS, `temperature=0.9` (broken for gpt-5)
- Stagnation detector: compared individual bars within a phrase (always triggered)
- LLM critic: failing every call due to temperature param

| Metric | Score |
|--------|-------|
| CE (Content Enjoyment) | 5.91 |
| CU (Content Usefulness) | 7.63 |
| PC (Production Complexity) | 5.04 |
| PQ (Production Quality) | 7.51 |

**Issues observed:**
- "mix is stagnant" on every trajectory check (false positive — detector compared bars within a single play() call, which are always identical)
- RLM cargo-culted Pattern D (focused command generation via llm_query) every iteration
- LLM critic never fired (gpt-5 rejects temperature=0.9)

## v2 (previous session)

**Changes made:**
1. Fixed `temperature=0.9` → `temperature=1.0` for gpt-5 compatibility (critic now works)
2. Fixed stagnation detector: compare phrase-level snapshots (one per play() call) instead of individual bars
3. Added escalation: if stagnation persists across 3+ phrases, feedback escalates to "STOP tweaking faders, do something structural"
4. Tightened prompt: "Never use Pattern D twice in a row" rule added to force interaction pattern rotation
5. Added `ResilientInterpreter` to survive Deno sandbox crashes (tool re-registration on restart)

| Metric | Score | Delta |
|--------|-------|-------|
| CE (Content Enjoyment) | 5.52 | -0.39 |
| CU (Content Usefulness) | 7.80 | +0.17 |
| PC (Production Complexity) | 4.82 | -0.22 |
| PQ (Production Quality) | 7.48 | -0.03 |

## v7 (current — committed to DJ_INSTRUCTIONS)

### Code fixes (biggest impact)

1. **`--no-cache` flag** — `DSPY_CACHELESS` was never a valid env var. Added `cache=False, temperature=1.0` to `dspy.LM()` via CLI flag. Confirmed with MD5 comparison that runs now produce different output.

2. **Force `type=oneshot`** in `add()` — the RLM sometimes passed `type="sample"` which caused patterns to resolve to `[]` (empty beats). The renderer then looped raw samples as continuous drones instead of triggering them rhythmically. This was the root cause of the consistent CE=3.24 scoring bug affecting ~40% of runs.

3. **Force 32-bit WAV export** — `set_sample_width(4)` in streamer.py. Different sample combinations produced 16-bit vs 32-bit WAVs, creating inconsistent scorer inputs.

### Prompt changes (v7 vs baseline)

Two surgical changes to the PACING RULES and PEAK sections:

1. **16-bar holds during PEAK**: Changed "hold the groove, maybe one fader nudge per phrase" → "play('16') between actions — let the groove HYPNOTIZE. One fader nudge or pattern change per 16-bar block."

2. **Pattern evolution guidance during PEAK**: Replaced generic "add MOVEMENT" with specific pattern transition recommendations (e.g., perc: syncopated_a → gallop, hihat: offbeat_8ths → sixteenth_drive). Changed from "every 8 bars" to "every 16 bars" for more hypnotic repetition.

### Prompt variants tested and rejected

| Variant | Key Change | Result | Why Rejected |
|---------|-----------|--------|-------------|
| v3 wave_arc | Mandated 2 breakdowns, specific bar ranges | CE=5.65 (-0.30) | Too prescriptive, rushed transitions |
| v4 timbral_depth | Encouraged early sample swapping | CE=5.66 (-0.29) | RLM can't hear audio, swaps are coin flips |
| v5 swap_curate | Minimal: swap during build, curate as you go | GPT-5-mini +0.07 CE, but GLM-5/Kimi cratered | Swapping to bad samples tanks scores |
| v8 smooth_peak | Added fader automation around pattern changes | Crashed (20MB WAV) | Overwhelmed RLM with too many instructions |
| v8b tight_gains | Narrowed gain ranges | Crashed (20MB WAV) | Also overwhelmed RLM |
| v8c no_swap | "Don't swap, single breakdown only" | Tied with v7 after fixing type bug | No improvement over v7 |

### Key insights

- **Code fixes > prompt changes**: The type=oneshot bug fix had 10x more impact than any prompt change
- **Palette quality dominates variance**: Same prompt scores CE=3.24 to CE=6.57 depending on randomly selected samples
- **The RLM can't hear audio**: Encouraging sample swaps is pure randomness, not intelligent curation
- **Simpler prompts work better**: Adding structure/complexity tends to overwhelm the RLM or cause crashes
- **16-bar hypnotic holds help CE**: Letting grooves breathe longer improves enjoyment scores

### Final scores (3 replicas, fixed palette, all bugs fixed)

| Metric | v7 avg | v7 range |
|--------|--------|----------|
| CE | 6.51 | 6.45–6.56 |
| CU | 8.09 | 8.07–8.11 |
| PC | 4.87 | 4.82–4.89 |
| PQ | 8.19 | 8.17–8.20 |

Weighted score (CE×0.60 + CU×0.05 + PC×0.05 + PQ×0.30): **6.91**

### Files modified

- `src/bergain/dj.py` — DJ_INSTRUCTIONS updated to v7, `add()` forces type=oneshot, `--no-cache` support via `run_dj(no_cache=...)`, `dspy.LM(cache=False, temperature=1.0)` when enabled
- `src/bergain/cli.py` — Added `--no-cache` flag to `dj` command
- `src/bergain/streamer.py` — Force 32-bit WAV export via `set_sample_width(4)`
- `scripts/parallel_dj.sh` — Added `--no-cache` passthrough, reduced to GPT-5-mini only for fast iteration

## Palette Screening

Screened 100 random palettes using the reference arrangement renderer (not the DJ). Each palette rendered a fixed arrangement and scored via audiobox-aesthetics.

- **100 palettes screened**, top 10 curated at OBJ 7.0+
- Best palette (palette_001): OBJ=7.18
- Curated palettes saved to `palettes/curated/`
- Key finding: palette quality is the single biggest lever — OBJ range across palettes was 5.2–7.18

## Reference-vs-DJ Gap Fix (v9)

### Problem

First DJ run with palette_001 scored **OBJ=6.87** while the same palette's reference arrangement scored **OBJ=7.18**. The 0.31 gap is entirely due to the DJ's rendering path.

### Gap cause #1: Forced oneshot on loop samples (biggest impact)

`add()` did `type = "oneshot"` for ALL roles — a v7 bug fix to prevent the RLM passing `type="sample"` which caused silent bars. But curated palettes have loop samples for bassline/texture. The reference arrangement renders these as `type="loop"` (continuous fill via `_loop_to_length()`), while the DJ rendered them as oneshot hits at specific beats, creating overlapping repeated copies of multi-second loop samples.

**Fix**: Auto-detect loop samples in `add()`. Check if the role's sample in `role_map` has `loop=True`. If so, use `type="loop"`. Otherwise keep `type="oneshot"`.

### Gap cause #2: Post-limit LLM churn (wasted tokens)

After reaching 64 bars, the DJ made 13+ LLM calls doing fader/remove/pattern changes that never render. `_DJComplete` exception existed but was never raised — the status just returned `set_complete=true` which the RLM ignored.

**Fix**: Raise `_DJComplete` in `play()` when max_bars is reached. Catch it in `run_dj()` alongside `KeyboardInterrupt`.

### Gap cause #3: Late peak, early auto-trim

DJ reaches 6 layers at bar 41 but auto-trim starts at bar 48 (75% of 64). Only 7 bars of full peak. This is a prompt pacing issue — no code change needed.

### Files modified

- `src/bergain/dj.py` — `add()` auto-detects loop type from role_map, `play()` raises `_DJComplete` on max_bars, `run_dj()` catches `_DJComplete`

## Gap Analysis: Duration vs Arc vs Decisions (v10)

### Diagnostic design

Ran 4 experiments with palette_001 to isolate what causes the 0.37 OBJ gap:

| Experiment | Bars | Source | Purpose |
|-----------|------|--------|---------|
| A. Reference 32-bar | 32 | Hardcoded reference | Baseline |
| B. Reference 64-bar | 64 | Reference doubled | Isolate duration effect |
| C. DJ 32-bar | 32 | RLM-driven | Isolate DJ decisions |
| D. DJ 64-bar | 64 | RLM-driven | Current state |

### Results

| Exp | Bars | CE | CU | PC | PQ | OBJ |
|-----|------|----|----|----|----|----|
| A | 32 | 6.80 | 8.18 | 5.29 | 8.07 | 7.18 |
| B | 64 | 6.56 | 8.12 | 5.06 | 8.00 | 6.99 |
| C | 32 | 6.30 | 8.06 | 4.82 | 7.99 | 6.82 |
| D | 64 | 6.36 | 8.10 | 5.01 | 7.90 | 6.84 |

### Key findings

1. **Duration effect (B-A): -0.18 OBJ** — The scorer IS duration-sensitive. Going from 60s to 120s costs ~0.18 OBJ, primarily through CE (-0.24).

2. **DJ decisions hurt at ALL durations (C-A): -0.36 OBJ** — Even at 32 bars, the DJ loses to the reference. The gap is in gain/pattern choices, not arc structure.

3. **DJ vs reference at same duration (D-B): -0.15 OBJ** — At 64 bars, the DJ is actually close to the reference. The arc structure is acceptable.

4. **DJ duration doesn't matter (C vs D): -0.02 OBJ** — DJ scores the same at 32 and 64 bars. The duration penalty and longer build time cancel out.

### Interpretation

The gap decomposes as:
- **~0.18 OBJ from duration** — scorer penalizes 120s vs 60s tracks
- **~0.19 OBJ from DJ decisions** — gains/patterns differ from reference
- Arc structure contributes very little — D-B is only -0.15

### Arc phase fix (minor)

Extended peak phase from 37.5% to 43.8% of set, shortened stripping (12.5% → 9.4%) and outro (12.5% → 9.4%). `_ARC_PHASES` thresholds changed from `[0.125, 0.375, 0.75, 0.875]` to `[0.125, 0.375, 0.8125, 0.9063]`.

### Arc fix validation (3 replicas, palette_001, 64 bars)

| Replica | CE | CU | PC | PQ | OBJ |
|---------|------|------|------|------|------|
| r1 | 5.95 | 8.00 | 4.55 | 7.73 | 6.52 |
| r2 | 6.61 | 8.12 | 5.15 | 8.01 | 7.03 |
| r3 | 6.54 | 8.11 | 5.13 | 7.99 | 6.98 |
| **avg** | **6.37** | **8.08** | **4.94** | **7.91** | **6.84** |

**Conclusion**: Arc fix is neutral on average (6.84 vs 6.84 before). But r2 hit 7.03, matching the 64-bar reference (6.99). The gap is dominated by **stochastic DJ decision variance** — when the RLM gets lucky with gains/patterns, it matches the reference.

### Next steps

The biggest remaining lever is **DJ decision quality** — the DJ's gain and pattern choices don't match the curated reference. Potential approaches:
- Tighten gain ranges in the prompt to match reference GAINS dict
- Constrain pattern choices to the patterns that work best for each role
- Consider running DJ at 32 bars if duration penalty matters for the use case

### Files modified

- `src/bergain/dj.py` — `_ARC_PHASES` thresholds updated for longer peak
- `scripts/score_gap_test.py` — new diagnostic script
