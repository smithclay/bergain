# Plan: Fix stagnation feedback + prompt to reduce monotony

## Context

Every smoke test shows "mix is stagnant" on nearly every trajectory check, even when the RLM is actively making moves. Two root causes:

1. **Broken stagnation detector**: `_compute_critique` (line 287) compares full bar JSON specs within the last 8 bars. But within a single `play("8")` call, all 8 bars use the same mixer state → identical specs → unique_ratio ≈ 0.125 → always triggers. It's measuring within-phrase sameness, not across-phrase monotony.

2. **RLM cargo-cults Pattern D**: The prompt lists 5 interaction patterns but the rigid "2-3 moves → play()" loop means the RLM defaults to the simplest pattern every time. It never diagnoses, explores samples, or does vibe checks.

## File: `src/bergain/dj.py`

### Change 1: Fix stagnation detector (~line 285)

Compare **phrase-level snapshots** (one per `play()` call) instead of individual bars. Take the last N bars but deduplicate by comparing only one bar per phrase boundary (every 8 bars). Or simpler: compare only the *last bar of each phrase* from the recent window.

```python
# Instead of comparing all 8 individual bars (which are identical within a phrase):
# Compare the mixer state at each phrase boundary (every CRITIQUE_INTERVAL bars)
stride = CRITIQUE_INTERVAL
phrase_specs = specs[::stride] if len(specs) >= stride else specs
unique_ratio = len(set(phrase_specs)) / max(len(phrase_specs), 1)
if len(phrase_specs) >= 2 and unique_ratio < 0.5:
    observations.append("mix is stagnant — introduce a variation")
```

This way stagnation only triggers when multiple *phrases* look the same, not when bars within one phrase are identical (which is by design).

### Change 2: Add escalation counter to trajectory feedback (~line 270)

Track how many consecutive times stagnation has been flagged. When it's 2+, escalate the feedback message to push the RLM toward different patterns:

```python
if unique_ratio < 0.5:
    stagnation_streak = ...  # count consecutive stagnant critiques
    if stagnation_streak >= 2:
        observations.append(
            "mix STILL stagnant after multiple phrases — STOP tweaking faders. "
            "Do something structural: swap samples, try a breakdown(), drop a layer and rebuild, "
            "or use llm_query to DIAGNOSE why energy isn't moving"
        )
    else:
        observations.append("mix is stagnant — introduce a variation")
```

This requires tracking a `stagnation_count` in the shared closure state (alongside `bars_played`, `bar_history`, etc.).

### Change 3: Tighten prompt to discourage Pattern D repetition (~line 523)

Replace the gentle "Don't default to it every time" with a harder rule:

```
# Pattern D is the simplest but produces the stalest mixes.
# RULE: Never use Pattern D twice in a row. If your last iteration
# used llm_query to generate a JSON array of moves, this iteration
# MUST use a different approach: diagnose (A), explore samples (B),
# evaluate strategies (C), or vibe-check (E).
```

## Verification

```bash
rm -f output/smoke_test.wav; uv run bergain dj --lm "openai/gpt-5-mini" --verbose --bars 64 -o output/smoke_test.wav 2>&1 | tee /tmp/dj_smoke_test.log
# Then score:
curl -s --max-time 120 -X POST "https://smithclay--bergain-aesthetics-judge-score.modal.run" -F "file=@output/smoke_test.wav"
```

- "mix is stagnant" should NOT appear on every single trajectory check
- RLM should visibly vary its approach across iterations
- CE score (Content Enjoyment) should improve from 5.9 baseline
