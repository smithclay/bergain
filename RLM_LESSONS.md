# RLM Lessons: Using DSPy RLM for EDM Generation

Hard-won insights from many iterations of building a DJ system that uses DSPy's Reactive Language Model (RLM) to generate techno music in real time.

## The Leverage Hierarchy

After dozens of experiments, the factors that matter most (in order):

1. **Palette quality** (sample selection) — OBJ range of 5.2 to 7.18 across random palettes. The samples you feed the system matter more than anything the LLM does. Screen palettes with a deterministic reference arrangement before involving the RLM.

2. **Code correctness** — A single bug (forced `type=oneshot` on loop samples) caused a -1.0 CE drop affecting ~40% of runs. Code fixes deliver 10x the impact of prompt changes. Always validate the rendering pipeline independently.

3. **Scorer sensitivity** — The audiobox-aesthetics scorer is duration-sensitive: a 120s track scores ~0.18 OBJ lower than the same arrangement at 60s. Understand your evaluation function's biases before optimizing against it.

4. **DJ decision variance** — The RLM's stochastic gain/pattern choices account for ~0.19 OBJ of gap vs reference. When the RLM gets lucky, it matches the reference (7.03 vs 6.99). The gap is variance, not systematic error.

5. **Arc structure** — Barely matters. Changing phase allocations (more peak, less strip/outro) had zero average impact. The auto-trim guardrails handle structural pacing well enough.

6. **Prompt engineering** — Smallest lever. 6 prompt variants tested; the best was marginally better than the simplest. Simpler prompts work better than complex ones.

## RLM-Specific Gotchas

### The RLM can't hear audio
The RLM operates on text descriptions of mixer state, not on audio. It has no perceptual feedback about what sounds good. This means:
- **Sample swapping is a coin flip** — encouraging the RLM to swap samples is pure randomness, not intelligent curation. Swapping to a bad sample tanks the score.
- **Gain tuning is approximate** — the RLM picks gains from prompt-specified ranges. It can't tell if 0.55 sounds better than 0.65 for hihat.
- **Pre-curated palettes are essential** — screen samples with a deterministic renderer before the RLM touches them.

### The Deno sandbox is fragile
DSPy's PythonInterpreter runs in a Deno+Pyodide sandbox. It crashes:
- `ResilientInterpreter` re-registers tools on every `execute()` call to survive sandbox restarts.
- The RLM sometimes generates code that overwhelms the sandbox (20MB+ WAVs, infinite loops). Keep prompt complexity low.
- Tool functions must return JSON strings — everything crossing the sandbox boundary must be serializable.

### The RLM cargo-cults patterns
If the prompt shows a complex pattern (e.g., "use llm_query to generate a plan, then dispatch"), the RLM will repeat it verbatim every iteration regardless of context. This wastes tokens and adds no value.
- **Simpler interaction patterns work better** — direct tool calls beat multi-step reasoning chains.
- **Named patterns > JSON arrays** — the RLM reliably uses named patterns like `"four_on_floor"` but garbles JSON array syntax.
- **Don't teach meta-strategies** — the RLM will copy the example, not the principle.

### Post-completion token waste
After reaching `max_bars`, the RLM kept making LLM calls (fader changes, removes, pattern swaps) that never rendered. The original `set_complete=true` status was ignored by the RLM.
- **Hard stops work, soft signals don't** — raise an exception or kill the process. Status flags get lost in the RLM's code generation.

## Prompt Engineering for RLM

### What works
- **Concrete gain ranges** — `"kick 0.82-1.00 | hihat 0.47-0.67"` gives the RLM boundaries it respects.
- **Named patterns** — a dictionary of named beat patterns (`four_on_floor`, `offbeat_8ths`) that the RLM references by string. Much more reliable than expecting it to construct beat arrays.
- **16-bar holds** — telling the RLM to `play("16")` during peak produces more hypnotic, higher-scoring output than frequent small changes.
- **Phase-aware structure** — the prompt's DJ SET STRUCTURE section (intro/build/peak/strip/outro with bar ranges) maps well to the RLM's code generation style.
- **Safety rules with examples** — `"WRONG: add('kick', '0.90', '909-Bassdrum.wav') ← CRASHES sandbox"` prevents real errors.

### What doesn't work
- **Prescriptive arc shapes** — mandating "2 breakdowns at bars 32 and 64" rushed transitions and lowered CE by -0.30.
- **Timbral guidance** — "encourage early sample swapping for timbral depth" is meaningless when the RLM can't hear audio.
- **Fader automation scripts** — "gradually raise hihat from 0.50 to 0.65 over 8 bars" generates complex code that overwhelms the sandbox.
- **Multiple interaction patterns** — offering 4 different strategies (Pattern A/B/C/D) causes the RLM to cargo-cult one and ignore the rest.
- **Long prompts** — prompts over ~2000 tokens increase crash rate and reduce output quality. Every instruction competes for the RLM's limited code-generation bandwidth.

## Prompt Evolution: What We Tried and Why

The `prompts/` directory contains 11 prompt variants tested across multiple sessions. The evolution tells a clear story: every attempt to make the prompt smarter made the output worse.

### The trajectory

**baseline.txt** — The starting point. Generic DJ instructions: "hold the groove" during peak, "add MOVEMENT," three arc shape options (Hill/Ramp/Wave). Scored CE=5.91 but had confounding bugs (broken temperature, false stagnation detection).

**v3_wave_arc.txt** — First attempt at structure. Mandated a WAVE arc with 2 breakdowns at specific bar ranges. Added "TIMBRAL REFRESH" and "PATTERN EVOLUTION" sections. **Result: CE=5.65 (-0.30).** The RLM rushed through the prescriptive timeline, hitting breakdowns before building enough density. *Lesson: the RLM follows timelines literally but can't judge readiness.*

**v4_timbral_depth.txt** — Pivoted to encouraging sample exploration: "SWAP EARLY AND OFTEN during build." Encouraged auditioning 2-3 options per role. **Result: CE=5.66 (-0.29).** Swapping samples is a coin flip when you can't hear audio. Bad swaps tanked scores. *Lesson: don't ask a deaf DJ to curate.*

**v5_swap_curate.txt** — Dialed back to "curate as you build." Still encouraged swapping but more selectively. **Result: model-dependent.** GPT-5-mini improved +0.07 CE but other models cratered. *Lesson: prompt sensitivity varies across models; what helps one model hurts another.*

**v6_peak_movement.txt** — Abandoned swapping, focused on peak quality. Added specific pattern transition recommendations (perc: syncopated_a to gallop, etc.) every 8 bars. **Result: improvement, led to v7.** *Lesson: concrete pattern names work better than abstract "add movement."*

**v7_hypnotic_peak.txt** — The winner. Two surgical changes: `play("16")` during peak instead of `play("8")`, and pattern evolution every 16 bars instead of 8. **Result: CE=6.51, best scores.** Became the basis for the built-in `DJ_INSTRUCTIONS`. *Lesson: less action = better music. Hypnotic repetition is the point.*

**v8_smooth_peak.txt** — Tried to improve v7 by adding fader automation around pattern changes ("drop gain 0.05-0.10, change pattern, ride back up"). **Result: crashed (20MB WAV).** The additional code generation complexity overwhelmed the Deno sandbox. *Lesson: every prompt instruction costs code-generation bandwidth.*

**v8b_tight_gains.txt** — Tightened gain ranges from wide (kick 0.82-1.00) to narrow (kick 0.88-0.95). **Result: also crashed (20MB WAV).** Same sandbox overwhelm. *Lesson: even small prompt additions can tip the balance.*

**v8c_no_swap.txt** — v7 plus "the palette was curated for you, don't swap" and "ONE breakdown, and only ONE." **Result: tied with v7.** No improvement, no regression. *Lesson: adding constraints that the server-side guardrails already enforce doesn't help.*

**longer_phrases.txt & fader_focus.txt** — Radically different approach: code-template prompts showing setup + main loop, requiring specific actions between every `play()` call. **Result: cargo-culting and over-engineering.** The RLM copied the template structure exactly, generating 30-50 lines of code per iteration instead of 10-20. *Lesson: code templates teach syntax, not musicianship.*

### The pattern

```
More prescriptive ──────────────────────> Less prescriptive
  v3 (wave arc)    v4 (swap often)     v7 (16-bar holds)    baseline
  CE=5.65          CE=5.66             CE=6.51              CE=5.91
  WORSE            WORSE               BEST                 OK
```

Every attempt to add structure, encourage exploration, or prescribe specific behaviors made things worse. The winning move was always **subtracting**: fewer changes per phrase, longer holds, less swapping. The RLM is better as a conservative DJ than an adventurous one.

### Why code-template prompts failed

`longer_phrases.txt` and `fader_focus.txt` tried a different paradigm: instead of prose instructions, provide a Python code template with a main loop. The idea was to reduce ambiguity by showing exactly what code to generate.

Problems:
- The RLM cargo-culted the template verbatim, including placeholder comments like `# Parse direction and apply it`
- `play("4")` (fader_focus) generated 2x as many iterations as `play("8")`, doubling token cost with no quality gain
- "MANDATORY: at least one fader() call every phrase" produced constant 0.03 gain nudges that sounded like jitter, not dynamics
- The code-generation complexity made sandbox crashes more frequent

The prose-style prompt (v7) outperformed code templates because it gave the RLM freedom to vary its code structure across iterations, which paradoxically produced more consistent musical output.

## Auto-Guardrails: Server-Side > Client-Side

The most reliable quality improvements came from server-side guardrails, not prompt instructions:

| Guardrail | Implementation | Impact |
|-----------|---------------|--------|
| Loop auto-detection | `add()` checks `role_map` for loop flag | Prevented -1.0 CE bug |
| Gain caps | `GAIN_CAP = {"texture": 0.55, "synth": 0.50}` | Prevents muddy mixes |
| Density ceiling | `_ARC_PHASES` with position-aware layer caps | Enforces arc structure |
| Trajectory feedback | `_compute_critique()` every 8 bars | Catches energy drift |
| LLM critic | `_llm_critique()` every 16 bars | Phase-aware creative direction |
| Hard stop on completion | Exception raised at `max_bars` | Prevents token waste |

**Principle**: Don't ask the RLM to enforce constraints — enforce them in the tool layer and inform the RLM after the fact. The RLM is a creative driver, not a safety system.

## Evaluation Strategy

### A/B testing is noisy
Single DJ runs have high variance (OBJ range 6.52-7.03 across 3 runs with identical config). Meaningful A/B tests need 3+ replicas per condition. Even then, palette variance dominates.

### Isolate variables with controlled experiments
The most useful diagnostic was the 4-experiment gap test:
- Reference at target duration (A) — establishes ceiling
- Reference at alternate duration (B) — isolates scorer duration sensitivity
- DJ at target duration (C) — isolates DJ decision quality
- DJ at alternate duration (D) — current state

This decomposed the gap into duration (-0.18 OBJ), decisions (-0.19 OBJ), and arc (negligible).

### The reference arrangement is your oracle
A deterministic reference arrangement with curated gains and patterns is the single most useful evaluation tool. It tells you the ceiling for a given palette, lets you screen samples cheaply, and provides ground truth for debugging rendering bugs.

## The Aesthetics Scorer

### How it works

The scorer is a Modal-hosted GPU endpoint (`aesthetics/app.py`) running Meta's [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) model on a T4 GPU. It accepts a WAV file via multipart POST and returns four scores:

| Metric | Full Name | What it measures | Range |
|--------|-----------|-----------------|-------|
| **CE** | Content Enjoyment | How enjoyable/engaging the audio is | ~3-8 |
| **CU** | Content Usefulness | How useful/functional the audio is | ~7-9 |
| **PC** | Production Complexity | Perceived production sophistication | ~4-6 |
| **PQ** | Production Quality | Technical audio quality | ~7-9 |

The composite objective: `OBJ = 0.60*CE + 0.05*CU + 0.05*PC + 0.30*PQ`. CE dominates because it's the most variable and the most musically meaningful metric. CU and PC have low weight because they're noisy and less correlated with actual quality.

### Architecture

```
WAV file → curl POST → Modal serverless (T4 GPU, 120s scaledown)
  → soundfile loads to numpy → torch tensor → audiobox predictor
  → returns {"scores": {"CE": 6.80, "CU": 8.18, "PC": 5.29, "PQ": 8.07}}
```

The endpoint cold-starts in ~15s (model download + load), then processes a 120s WAV in ~2-3s. The 120s scaledown window keeps the GPU warm across a batch of scoring calls.

### Scorer quirks we discovered

- **Duration sensitivity**: A 120s track scores ~0.24 CE lower than the same arrangement at 60s. The scorer was likely trained on shorter clips and penalizes repetition over time. This means comparing scores across different durations is misleading.
- **Sample width doesn't matter much**: We forced 32-bit WAV export for consistency, but the scorer normalizes internally. The difference between 16-bit and 32-bit is negligible.
- **Normalization matters**: Both the reference renderer and FileWriter normalize to -1 dBFS before scoring. Without normalization, quiet tracks score lower on PQ.
- **CU and PQ are stable; CE and PC are volatile**: CU rarely drops below 7.5 and PQ stays in the 7.5-8.2 range regardless of arrangement quality. CE swings from 3.24 (broken drone) to 6.80 (good arrangement). This is why CE gets 60% weight.
- **The scorer can't tell "techno" from "noise"**: A track with all loops playing as continuous drones (the `type=oneshot` bug on loop samples) scored CE=3.24 — bad, but not zero. The scorer measures generic audio aesthetics, not genre-specific quality.

### How we use it

1. **Palette screening** (`scripts/screen_palettes.py`): Render a fixed 32-bar reference arrangement with each candidate palette, score all of them, rank by OBJ. This is cheap (no LLM calls, ~2s render + ~3s score per palette) and separates palette quality from DJ decision quality.

2. **DJ run scoring** (`scripts/parallel_dj.sh`): After each DJ run produces a WAV, POST it to the endpoint. This is the end-to-end metric but conflates palette + DJ decisions + duration effects.

3. **Gap diagnostics** (`scripts/score_gap_test.py`): Score controlled experiments (reference vs DJ, 32 vs 64 bars) to decompose the OBJ gap into individual factors.

## The Evaluation Toolchain

### `scripts/parallel_dj.sh` — Parallel A/B testing

Runs the DJ across multiple models and/or prompt variants in parallel, then auto-scores all outputs. This is the primary tool for prompt A/B testing.

**What it does well:**
- Generates a shared palette (from file, directory, or random) so all runs use the same samples — eliminates palette variance from the comparison
- Runs all model/prompt combinations as background processes in parallel — a 4-run comparison takes ~5 min instead of ~20 min
- Auto-scores all WAVs via the Modal endpoint after runs complete
- Saves scores, logs, prompt files, and palette into a timestamped output directory for reproducibility
- Prints a summary table with per-prompt averages

**What it doesn't do:**
- No statistical significance testing — you need to eyeball whether deltas are meaningful (and with n=1 per condition, they usually aren't)
- No automatic replicas — running 3 replicas requires manually invoking the script 3 times or extending it
- Palette variance still dominates — even with a shared palette, DJ decision variance across runs is ~0.5 OBJ. A single run per condition is insufficient to draw conclusions.

**Usage patterns that worked:**
```bash
# A/B test two prompts with same palette
./scripts/parallel_dj.sh --palette palettes/curated/palette_001.json \
    --prompt prompts/v7_hypnotic_peak.txt --prompt prompts/v8c_no_swap.txt --no-cache

# Quick single-prompt run to validate a code change
./scripts/parallel_dj.sh --palette palettes/curated/palette_001.json --no-cache

# Screen across palette quality (one run per palette)
for p in palettes/curated/palette_*.json; do
    ./scripts/parallel_dj.sh --palette "$p" --no-cache
done
```

**Key lesson**: The script is excellent for rapid iteration (run, score, compare in one command) but the results are too noisy for fine-grained prompt comparisons. Differences under 0.3 OBJ are in the noise. Use it to catch regressions and validate code fixes, not to pick between prompt variants that differ by 0.05 CE.

### `scripts/screen_palettes.py` — Palette quality screening

The most cost-effective tool in the pipeline. Renders 100 random palettes against a fixed reference arrangement and scores them all. No LLM calls, no DJ variance — pure sample quality measurement.

**Key finding from screening**: OBJ ranged from 5.2 to 7.18 across 100 palettes. The top 10 palettes (OBJ > 7.0) were saved to `palettes/curated/`. This 2-point OBJ range from palette alone dwarfs any prompt optimization gain (max 0.3 OBJ across all prompt variants).

**Smart filtering**: Uses loop samples for bassline/texture and oneshot samples for kick/hihat/perc/clap, matching the reference arrangement's expectations. Without this filtering, ~30% of palettes had mismatched sample types that produced drones or silence.

### `scripts/score_gap_test.py` — Controlled experiments

The diagnostic tool that finally explained the reference-vs-DJ gap. Runs 4 experiments varying duration (32/64 bars) and source (reference/DJ) to decompose the gap into independent factors.

**This tool proved**: The gap is ~50% scorer duration sensitivity and ~50% DJ decision variance. Arc structure barely matters. Without this decomposition, we would have spent more iterations tweaking arc phases when the real issue was elsewhere.

## Live Compose Mode: DJ Shaping Lessons

These lessons come from integrating DJ shaping research (emusicology article on how DJs shape performances) into `compose_next()` — the live-mode tool where a conductor LM sends creative prompts to a sub-LM that returns musical parameters.

### The Sub-LM Energy Ceiling Problem

Without calibration, the sub-LM (GPT-5 via OpenRouter) **caps energy at ~0.4 regardless of prompt intensity**. Prompts like "relentless peak", "full power", and "maximum intensity" all produced energy=0.3-0.4. The sub-LM plays it safe because `energy (float): 0.0 to 1.0` gives no anchor for what values mean.

**Fix: One calibration line in the sub-LM prompt.** Adding `0.2=ambient, 0.4=gentle groove, 0.6=driving, 0.8=peak power, 0.95=maximum` to the energy field description pushed peak sections to 0.76-0.95. This is the single highest-impact change — energy range went from 0.05-0.40 to 0.22-0.95.

**Fix: Phase-based energy floor.** Middle-phase sections get a floor of 0.3 energy, so the "building" part of the arc actually builds. Opening phase gets a ceiling of 0.55 to prevent premature peaks.

**Fix: Bars cap for high-energy sections.** The sub-LM defaults to 16-bar loops for everything. High-energy peaks need tighter 4-8 bar loops. Capping bars at 8 when energy ≥ 0.6 during middle phase enforces this automatically.

**Pattern**: All three fixes are tool-layer guardrails that post-process the sub-LM's output. The sub-LM doesn't know it's being clamped. This is consistent with the core lesson: server-side guardrails > prompt instructions.

### DSPy LM Return Type Gotcha

`dspy.LM()` returns **dicts with a `text` key**, not plain strings:
```python
completions = sub_lm(messages=[...])
# completions[0] = {'text': '{"name": "..."}', 'reasoning_content': '...'}
# NOT: completions[0] = '{"name": "..."}'
```
Calling `re.search()` on the raw dict crashes with `"expected string or bytes-like object, got 'dict'"`. Extract with `raw.get("text")` before parsing.

### Texture Density ≠ Energy

A loud minimal section (high energy, low density) is fundamentally different from a quiet layered ambient pad (low energy, high density). Tracking density as `count_of_active_instruments / total_instruments` alongside energy gives the conductor LM a second dimension for shaping decisions. In practice, `density_avg` stays 0.78-1.0 because the sub-LM rarely drops instruments to "none".

### Breakdown Nudge: Wired But Rarely Fires

The breakdown nudge detects 3+ consecutive sections ≥ 0.6 energy without bass="none" and injects a one-line suggestion. In testing, the sub-LM rarely sustains energy high enough for 3 sections to trigger this — even with energy guardrails, the sub-LM's creative instinct is to vary. The nudge is a safety net for longer performances where energy could plateau.

### Transition Fades: Need Extreme Contrast

Fading out tracks that go from active to "none" between sections sounds great in theory, but the sub-LM rarely makes that drastic a change. Most transitions keep all instruments active and just change styles. Fades trigger most reliably on explicit breakdowns (drums "none" → everything, or vice versa).

### RLM Iteration Budget

The conductor RLM (using `dspy.RLM`) consistently spends **2-4 iterations on setup** (browsing for instruments, creating tracks, fixing failed drum kit loads) before any `compose_next()` calls. With a 3-minute target, this leaves time for only 1-2 composed sections. With 8 minutes, you get 4-5 sections — enough for a proper arc.

**Rule of thumb**: budget at least 5 minutes of target duration for live compose. Each `compose_next()` takes ~60-90 seconds (sub-LM call + clip writing + wait time for bars to play).

### The Contrast Metric

`avg_contrast = mean(|energy_delta| + 0.1 * style_changes)` between adjacent sections. When this drops below ~0.1 AND instruments stay the same for 3+ sections, the arc summary flags a monotony warning. In practice, the sub-LM varies styles well (6 unique combos across 6 sections is typical), so monotony is more about energy flatness than style repetition.

### What We Confirmed

- **Simpler prompts still win** — we added zero text to the conductor LM's signature docstring. All improvements were tool-layer mechanics.
- **Named patterns remain the right abstraction** — the sub-LM reliably generates `"four_on_floor"`, `"rolling_16th"`, `"swells"` etc. No need for note-level control.
- **The RLM is a good conductor** — when compose_next works, the RLM makes intelligent creative choices about when to build, when to break down, and when to peak. It reads the arc summary and adjusts. The weak link was always the sub-LM's energy calibration, not the conductor's musical judgment.

## Audio Export (`--export`): Lessons from Debugging the Capture Pipeline

The `--export` flag records the entire compose session to WAV via Ableton's session recording + Resampling input routing. Getting this right required understanding several Ableton quirks and fixing three distinct layers of bugs.

### The Core Problem

The compose pipeline flow is: `start_recording()` → `setup()` (destroys all tracks) → compose → `stop_recording()`. But `setup()` deletes every track including the capture track, then creates new ones. The capture track must be restored after setup, and recording must resume.

### Use `set/session_record` Not `trigger_session_record`

**This is the single most important export fix.** AbletonOSC exposes two recording controls:

- `trigger_session_record` — a **toggle** that alternates start/stop. Not idempotent. Calling it twice stops what you just started. Gets stuck at `session_record_status=1` (transitioning) when the transport state is ambiguous.
- `set/session_record` — a **direct setter** (1=on, 0=off). Idempotent. Reliably reaches status=2.

We wasted hours debugging `trigger_session_record` stuck at status=1. The toggle is fundamentally unreliable when called programmatically because you can't know the current state with certainty. **Always use the direct setter:**

```python
# GOOD: direct setter — idempotent, reliable
self.api.song.set("session_record", 1)  # start
self.api.song.set("session_record", 0)  # stop

# BAD: toggle — unreliable, state-dependent
self.api.song.call("trigger_session_record")  # start? stop? who knows
```

Poll `session_record_status` after setting: 0=off, 1=transitioning, 2=recording. Budget up to 8s of polling (16 × 0.5s) for status=2. If status falls to 0 after settling, re-set once.

### Scene Fires Disrupt the Capture Track

**The second major bug.** `fire_scene()` fires ALL clips in a scene row — including the capture track's recording slot. This stops the recording and replays the captured silence.

Symptoms: exported WAV is 45s of silence with a tiny transient at the end. The recording clip in slot 0 captured only the pre-scene-fire period (no music yet), then scene fire stopped the recording and replayed that silent clip.

**Fix**: When recording is active, fire clips on each music track individually, skipping the capture track:

```python
def fire(self, scene):
    if self._recording:
        capture_idx = self._capture_track_idx
        for t in range(num_tracks):
            if t == capture_idx:
                continue
            self.api.clip_slot(t, scene).call("fire")
    else:
        self.api.scene(scene).call("fire")
```

This keeps the capture track's session recording uninterrupted while music clips transition normally.

### Normalization Is Required

The old reference renderer normalized to -1 dBFS before scoring. Without normalization, captured audio at -38 dBFS (1.2% peak) scores terribly on CE and PC. After normalization to -1 dBFS (89.1% peak), the same content scores much higher.

Implementation uses stdlib `wave` + `struct` — no new deps. Handles 16-bit and 24-bit WAVs. The gain factor is printed for debugging (`gain=1.58x` means the capture was already loud; `gain=667x` means it was nearly silent).

### Deferred Recording Pattern

After `setup()` restores the capture track, the transport is stopped. Recording must be deferred until the transport starts:

```python
# In setup() restore block:
self._record_pending = True

# In play()/fire()/fire_clip():
if self._record_pending:
    time.sleep(2.0)  # transport + routing need time to settle
    self.api.song.set("session_record", 1)  # direct setter
    self._record_pending = False
```

### Instrument Loading: Retry + Verify

Browser loading (`load_instrument`, `load_drum_kit`, `load_sound`) intermittently times out, especially for drum kits. A single failed load leaves a track with 0 devices, producing silence that tanks CE and PC.

**Fix**: Retry up to 2 times on TimeoutError with 1s backoff. After all loads complete, verify each track has `num_devices > 0` and print warnings for empty tracks. Default track volume boosted from 0.85 to 1.0 for MIDI instruments (inherently quieter than mixed samples; normalization handles any clipping).

### Capture Track Setup

The capture track uses:
- **Input routing**: "Resampling" (captures master output)
- **Output routing**: "Sends Only" (prevents feedback loop)
- **Monitoring**: "In" (state=0, so it actually captures)
- **Armed**: Yes (required for session recording)

After setting each property, read it back and verify. Print warnings if routing doesn't match expectations — this caught issues early.

### Export Score Progression

| Fix | CE | PC | Key Change |
|-----|----|----|-----------|
| Baseline (silent export) | 3.25 | 1.70 | No normalization, toggle recording |
| + Normalization | 3.25 | 1.70 | Audio audible but still captured silence |
| + Direct setter (`set/session_record`) | 3.70 | 4.42 | Recording actually captures audio |
| + Per-track firing (skip capture track) | 4.83 | 3.65 | Full-duration continuous audio |

CE went from 3.25 → 4.83 across these fixes. PC spiked to 4.42 when the recording first captured real audio. The remaining CE gap to 6+ likely requires fixing the Modal `structure` analysis endpoint (currently 500ing) and improving musical content density.

### Known Issues (TODO)

1. **~~Duration not respected.~~** FIXED. Three changes: (a) `compose_next()` now has a hard stop — returns error JSON when `remaining_min <= 0`, (b) bars are capped to fit within remaining time so a 16-bar section can't overshoot, (c) the sub-LM prompt includes a time budget line (`~N bars remaining, ~M sections`) with explicit wind-down instructions when nearing the end.

2. **Scene transitions are jarring.** Per-track clip firing with `clip_slot.call("fire")` triggers clips independently — they may not quantize to the same beat. Ableton's scene-level fire synchronizes all clips to the global quantize setting, but per-track firing loses this. Potential fixes: (a) set global quantize before firing, (b) fire all music tracks in a tight loop so they're within the same quantize window, (c) use `clip_slot.set("fire_button")` if it respects quantize. The current fades (`session.fade()`) help mask transitions but don't solve the root timing issue.

3. **Modal `structure` endpoint 500s.** The aesthetics scorer's structure analysis consistently fails with 500 errors, which may be dragging CE scores down since the composite score can't factor in structure. Need to debug the Modal endpoint separately.

## Palette Mode vs Live Mode: Head-to-Head Comparison

Ran both modes with the same brief ("dark minimal techno dance", 128 BPM) and scored the exports.

### Setup

- **Palette mode** (`Compose` signature): RLM browses instruments, plans 6 scenes via `llm_query()`, builds all clips, then `fire_scenes.py` fires each scene for 16 bars with session recording.
- **Live mode** (`LiveCompose` signature): RLM calls `setup_session()` then `compose_next()` repeatedly. Each section is composed, fired, and waited on in real time. Exported via the e2e pipeline.

### Results

| Metric | Palette + Fire | Live (compose_next) | Delta |
|--------|---------------|---------------------|-------|
| Duration | 4.0 min | 6.7 min | — |
| CE (enjoyment) | **5.99** | 4.47 | +1.52 |
| PQ (quality) | **7.54** | 6.88 | +0.66 |
| PC (complexity) | **4.11** | 3.74 | +0.37 |
| CU (usefulness) | **7.41** | 6.59 | +0.82 |
| **OBJ** | **6.43** | 5.26 | **+1.17** |

### Why Palette Mode Wins

1. **Transitions are cleaner.** `session.fire(scene)` fires all music-track clips in a tight loop (per-track to skip the capture track). Clips land on the same quantize boundary. Live mode's `compose_next()` writes clips and fires each scene independently, which can desync across tracks.

2. **Duration penalty.** The scorer penalizes longer audio (~0.24 CE per extra minute). The 6.7 min live capture eats ~0.6 CE vs the 4.0 min palette capture just from duration alone.

3. **Coherent planning.** Palette mode plans all scenes upfront with a single `llm_query()` call, producing a coherent energy arc (0.2 → 0.45 → 0.9 → 0.35 → 0.95 → 0.25). Live mode's sub-LM generates each section independently with only arc-summary context, producing a less structured arc.

4. **Better instruments.** Palette mode uses `browse()` to find specific instruments (909 Core Kit, Operator, Drifting Ambient Pad). Live mode's `setup_session()` picks sensible defaults but can't search the browser as thoroughly.

5. **More notes per clip.** Palette mode's peak scenes had 144 drum notes and 128 bass notes per clip. Live mode peaked at 357 drum notes (21-bar section) but with less density variation between sections.

### Implications

- **For short pieces (3-8 min):** Palette + auto-fire is strictly better. The upfront planning produces more coherent arcs, and scene-level firing gives cleaner transitions.
- **For long performances (15+ min):** Live mode's real-time adaptation may still be needed — palette mode can't react to how the music evolves.
- **Hybrid approach:** Use palette mode to build the grid, then live mode for in-performance tweaks (fader rides, filter sweeps, scene ordering). This would combine palette quality with live flexibility.

### The `fire_scenes.py` Script

Created `scripts/fire_scenes.py` to auto-fire palette scenes with recording. Key detail: must use `session.fire(scene)` (not `session.api.scene().call("fire")`) to skip the capture track during recording. The first attempt used direct scene fire, which disrupted the capture track and produced an empty export — same bug documented in the export section above.
