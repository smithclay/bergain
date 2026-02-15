#!/usr/bin/env python3
"""
Automated DJ prompt optimization via LLM-guided evolutionary search.

Generates prompt variants by tweaking identified "knobs" (phrase length,
fader emphasis, gain ranges, etc.), runs them through parallel_dj.sh with
replicas for statistical power, scores via AudioBox Aesthetics, and uses
pandas + scipy to measure whether changes are meaningful.

Usage:
    uv run python scripts/optimize_prompts.py --seed prompts/longer_phrases.txt
    uv run python scripts/optimize_prompts.py --seed prompts/longer_phrases.txt --generations 3 --variants 3
    uv run python scripts/optimize_prompts.py --seed prompts/longer_phrases.txt --replicas 5
    uv run python scripts/optimize_prompts.py --resume output/optimize_20260213_180000/
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from bergain.dj import DJ_INSTRUCTIONS

load_dotenv()

DEFAULT_WEIGHTS = {"CE": 0.60, "CU": 0.05, "PC": 0.05, "PQ": 0.30}
DEFAULT_META_LM = "openrouter/anthropic/claude-sonnet-4"
DEFAULT_DJ_LM = "openrouter/openai/gpt-5-mini"
DEFAULT_CRITIC_LM = "openrouter/openai/gpt-5-nano"
METRICS = ["CE", "CU", "PC", "PQ"]

KNOBS = [
    "phrase_length",
    "fader_mandate",
    "gain_ranges",
    "setup_structure",
    "tool_emphasis",
    "motivational_framing",
    "feedback_response_style",
    "phase_targets",
]

# ---------------------------------------------------------------------------
# Template system: numeric params for grid search, creative params for meta-LLM
# ---------------------------------------------------------------------------

NUMERIC_PARAMS = {
    "setup_phrase_bars": [2, 4, 8],
    "loop_phrase_bars": [4, 8, 16],
    "min_moves": [1, 2, 3, 4],
    "gain_range_width": [0.05, 0.10, 0.15, 0.20],
}

DEFAULT_NUMERIC_PARAMS = {
    "setup_phrase_bars": 4,
    "loop_phrase_bars": 8,
    "min_moves": 2,
    "gain_range_width": 0.05,
}

BASE_GAINS = {
    "kick": 0.92,
    "hihat": 0.57,
    "bassline": 0.67,
    "perc": 0.42,
    "texture": 0.375,
    "clap": 0.47,
}

DEFAULT_CREATIVE_PARAMS = {
    "persona": (
        "You are a Berghain resident DJ. The room breathes — layers enter and leave,\n"
        "gains shift constantly, the groove evolves every few bars. A flat mix\n"
        "clears the dance floor."
    ),
    "feedback_style": (
        "Respond with precise mixer command. Examples: fader('hihat', '0.58'), "
        "pattern('perc', 'gallop'), swap('texture', 'random'), "
        "breakdown('4'), mute('clap'). Be surgical."
    ),
    "tool_priority": (
        '# pattern("perc", "gallop")  — shift the rhythm (USE FREQUENTLY)\n'
        '# swap("hihat", "random")    — refresh a stale sound (USE FREQUENTLY)\n'
        '# fader("hihat", "0.58")     — nudge gain ±0.03-0.08\n'
        '# breakdown("4")             — tension moment (use sparingly)\n'
        '# mute("texture")            — create space\n'
        '# Example: pattern("perc", "syncopated_b") + swap("texture", "random") '
        '+ fader("kick", "0.93")'
    ),
    "feedback_guide": (
        '# "build up"  → add() new channel or unmute() existing + fader() boost\n'
        '# "strip back" → mute() non-essential or remove() + fader() cuts\n'
        '# "stagnant"  → pattern() + swap() combo or breakdown() for reset\n'
        '# "too busy"  → mute() texture/perc layers + fader() reductions\n'
        '# "needs energy" → pattern() to sixteenth_drive + fader() boosts\n'
        "# CRITIC      → execute exact command given"
    ),
}


def compute_gain_ranges(width: float) -> dict[str, tuple[float, float]]:
    """Compute per-role gain ranges from a center + half-width."""
    ranges = {}
    for role, center in BASE_GAINS.items():
        lo = max(0.0, round(center - width, 2))
        hi = min(1.0, round(center + width, 2))
        ranges[role] = (lo, hi)
    return ranges


def render_template(
    numeric_params: dict | None = None,
    creative_params: dict | None = None,
) -> str:
    """Render the DJ prompt template with the given parameters."""
    from jinja2 import Environment, FileSystemLoader

    np_ = {**DEFAULT_NUMERIC_PARAMS, **(numeric_params or {})}
    cp = {**DEFAULT_CREATIVE_PARAMS, **(creative_params or {})}
    gain_ranges = compute_gain_ranges(np_["gain_range_width"])

    template_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        keep_trailing_newline=True,
    )
    template = env.get_template("dj_template.j2")
    return template.render(**np_, **cp, gain_ranges=gain_ranges)


def trimmed_scores(
    df: pd.DataFrame, variant: str, trim_min: bool = False
) -> pd.DataFrame:
    """Get scores for a variant, optionally dropping the worst run."""
    vdf = df[df["variant"] == variant]
    if trim_min and len(vdf) >= 4:
        vdf = vdf.drop(vdf["objective"].idxmin())
    return vdf


def _build_meta_system_prompt() -> str:
    """Build the meta-system prompt from the live DJ_INSTRUCTIONS in dj.py."""
    return f"""\
You are a prompt engineer optimizing DJ instruction prompts for a techno music AI.

## The DJ prompt
Below is the current built-in DJ prompt. It is pseudocode that an LLM interprets
to control a real-time techno mixer. Your job is to produce **variants** of this
prompt that score higher on audio quality metrics. The variant must follow the
same structure and use only the tools, patterns, roles, and rules documented in
this prompt — it IS the specification.

```prompt
{DJ_INSTRUCTIONS}
```

## Tunable knobs (what you can change)
1. **phrase_length** — bars per play() call (e.g., play("4") vs play("8"))
2. **fader_mandate** — how aggressively to require fader moves between phrases
3. **gain_ranges** — per-role gain bounds (tighter or wider ranges)
4. **setup_structure** — how many phases in the intro, how fast to ramp up
5. **tool_emphasis** — which tools to highlight/encourage more
6. **motivational_framing** — the narrative/persona framing
7. **feedback_response_style** — how to respond to trajectory feedback
8. **phase_targets** — how many channels per phase (intro/building/peak/etc.)

## Hard constraints (your variant will be rejected if it violates these)
- Must contain add("kick" — kick must be the first channel added
- Must contain `while True:` — the main DJ loop
- Must contain play( — the only timing mechanism
- Must contain SUBMIT( — required for clean shutdown
- play() is the ONLY timing/waiting mechanism — no other sleep or delay calls
- All gain values must be between 0.0 and 1.0
- 15-80 non-empty lines

## Output format
Output exactly ONE complete prompt between ```prompt and ``` markers.
After the prompt block, briefly state which knob(s) you changed and why.
"""


# ---------------------------------------------------------------------------
# Objective & validation
# ---------------------------------------------------------------------------


def compute_objective(scores: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    """Weighted sum of CE/CU/PC/PQ scores."""
    return sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)


def validate_prompt(text: str) -> tuple[bool, list[str]]:
    """Check structural requirements. Returns (valid, errors)."""
    errors = []

    if 'add("kick"' not in text and "add('kick'" not in text:
        errors.append('Missing add("kick"...) — kick must be first channel')

    if "while True:" not in text:
        errors.append("Missing `while True:` main loop")

    if "play(" not in text:
        errors.append("Missing play() call")

    if "SUBMIT(" not in text:
        errors.append("Missing SUBMIT() call")

    if "time.sleep" in text:
        errors.append("Contains time.sleep — forbidden (play() is the clock)")

    for m in re.finditer(r'(?:fader|add)\([^)]*"(\d+\.?\d*)"', text):
        try:
            val = float(m.group(1))
            if not (0.0 <= val <= 1.0):
                errors.append(f"Gain value {val} out of range [0, 1]")
        except ValueError:
            pass

    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 15:
        errors.append(f"Only {len(lines)} non-empty lines (need >= 15)")
    if len(lines) > 80:
        errors.append(f"{len(lines)} non-empty lines (need <= 80)")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Meta-LLM variant generation
# ---------------------------------------------------------------------------


def generate_variant(
    parent_text: str,
    parent_scores: dict,
    parent_objective: float,
    history_df: pd.DataFrame,
    meta_lm: str,
    knob_hint: str | None = None,
) -> tuple[str, str] | None:
    """Call meta-LLM → parse prompt → validate → retry up to 3×.

    Returns (prompt_text, knob_name) or None on failure.
    """
    # Leaderboard summary (top 10 by mean objective)
    if not history_df.empty:
        top = (
            history_df.groupby("variant")["objective"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(10)
        )
        leaderboard_lines = []
        for variant, row in top.iterrows():
            # Get one representative row for knob info
            rep = history_df[history_df["variant"] == variant].iloc[0]
            scores_str = " ".join(
                f"{m}={history_df[history_df['variant'] == variant][m].mean():.2f}"
                for m in METRICS
            )
            leaderboard_lines.append(
                f"  {variant}: obj={row['mean']:.3f} (n={int(row['count'])}) "
                f"{scores_str} (knob: {rep.get('knob', 'seed')})"
            )
        leaderboard_str = "\n".join(leaderboard_lines)
    else:
        leaderboard_str = "  (no history yet)"

    tried_knobs = (
        set(history_df["knob"].dropna().unique()) if not history_df.empty else set()
    )
    tried_str = ", ".join(sorted(tried_knobs - {"seed", ""})) or "none"

    knob_instruction = ""
    if knob_hint:
        knob_instruction = f"\nFocus your mutation on the **{knob_hint}** knob."

    user_prompt = (
        f"## Parent prompt "
        f"(objective={parent_objective:.3f}, "
        f"CE={parent_scores.get('CE', 0):.2f}, "
        f"CU={parent_scores.get('CU', 0):.2f}, "
        f"PC={parent_scores.get('PC', 0):.2f}, "
        f"PQ={parent_scores.get('PQ', 0):.2f}):\n\n"
        f"```prompt\n{parent_text}\n```\n\n"
        f"## Leaderboard (top variants so far):\n{leaderboard_str}\n\n"
        f"## Knobs already tried: {tried_str}\n"
        f"{knob_instruction}\n\n"
        "Generate a new variant that improves on the parent. Change 1-2 knobs.\n"
        "Output the complete prompt between ```prompt and ``` markers.\n"
        "Then state which knob(s) you changed and why."
    )

    for attempt in range(3):
        try:
            response = litellm.completion(
                model=meta_lm,
                messages=[
                    {"role": "system", "content": _build_meta_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=2000,
            )
            content = response.choices[0].message.content

            match = re.search(r"```prompt\s*\n(.*?)```", content, re.DOTALL)
            if not match:
                match = re.search(r"```\s*\n(.*?)```", content, re.DOTALL)
            if not match:
                print(f"    Attempt {attempt + 1}: no prompt block found")
                continue

            prompt_text = match.group(1).strip()

            if "time.sleep" in prompt_text:
                print(
                    f"    Attempt {attempt + 1}: REJECTED — contains forbidden timing call"
                )
                user_prompt += (
                    "\n\nREJECTED: Your prompt contains a forbidden timing call. "
                    "play() is the ONLY timing/waiting mechanism. Remove any "
                    "other sleep or delay calls entirely. Output a corrected prompt."
                )
                continue

            valid, errors = validate_prompt(prompt_text)

            if not valid:
                print(
                    f"    Attempt {attempt + 1}: validation failed: {'; '.join(errors)}"
                )
                user_prompt += (
                    "\n\nYour previous attempt had these errors:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\nFix them and output a corrected prompt."
                )
                continue

            knob = knob_hint or "combined"
            if not knob_hint:
                after = content[match.end() :]
                for k in KNOBS:
                    if k.replace("_", " ") in after.lower() or k in after.lower():
                        knob = k
                        break

            return prompt_text, knob

        except Exception as e:
            print(f"    Attempt {attempt + 1}: meta-LLM error: {e}")

    return None


# ---------------------------------------------------------------------------
# parallel_dj.sh integration
# ---------------------------------------------------------------------------


def generate_palette(outdir: str) -> str:
    """Generate a shared palette via bergain internals. Returns palette path."""
    palette_path = os.path.join(outdir, "palette.json")
    script = (
        "import json\n"
        "from bergain.indexer import build_index\n"
        "from bergain.dj import _build_role_map, _pick_random_palette\n"
        "role_map = _build_role_map(build_index('sample_pack'))\n"
        "palette = _pick_random_palette(role_map)\n"
        f"with open({palette_path!r}, 'w') as f:\n"
        "    json.dump(palette, f, indent=2)\n"
        "print(json.dumps({r: p.split('/')[-1] for r, p in palette.items()}, indent=2))\n"
    )
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to generate palette: {result.stderr}")
        sys.exit(1)
    print(f"Palette: {result.stdout.strip()}")
    return palette_path


def run_parallel_dj(
    prompt_paths: list[str],
    outdir: str,
    palette_path: str,
    config: dict,
) -> bool:
    """Invoke parallel_dj.sh with the given prompts. Returns True on success."""
    cmd = [
        "./scripts/parallel_dj.sh",
        "--outdir",
        outdir,
        "--palette",
        palette_path,
        "--bars",
        str(config["bars"]),
        "--bpm",
        str(config["bpm"]),
        "--critic-lm",
        config["critic_lm"],
    ]
    for p in prompt_paths:
        cmd.extend(["--prompt", p])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=1200)
    return result.returncode == 0


def parse_scores(outdir: str) -> list[dict]:
    """Read scores.json from a parallel_dj.sh output dir."""
    scores_path = os.path.join(outdir, "scores.json")
    if not os.path.exists(scores_path):
        print(f"  WARNING: No scores.json found in {outdir}")
        return []
    with open(scores_path) as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def scores_to_dataframe(history: list[dict]) -> pd.DataFrame:
    """Convert lineage history entries into a flat DataFrame.

    Each row is one run (one replica of one variant). Columns include
    variant, gen, parent, knob, replica, CE, CU, PC, PQ, objective.
    """
    rows = []
    for entry in history:
        scores = entry.get("scores", {})
        rows.append(
            {
                "variant": entry["variant"],
                "gen": entry.get("gen", 0),
                "parent": entry.get("parent"),
                "knob": entry.get("knob", "seed"),
                "replica": entry.get("replica", 0),
                **{m: scores.get(m, np.nan) for m in METRICS},
                "objective": entry.get("objective", np.nan),
            }
        )
    return pd.DataFrame(rows)


def analyze_variant(
    df: pd.DataFrame,
    variant: str,
    baseline: str | None = None,
    weights: dict = DEFAULT_WEIGHTS,
    trim_min: bool = False,
) -> dict:
    """Compute descriptive stats and significance for one variant vs baseline.

    Returns a dict with mean/std/ci for each metric + objective, plus
    p-values and effect sizes if a baseline is provided.
    """
    vdf = trimmed_scores(df, variant, trim_min)
    n = len(vdf)
    result: dict = {"variant": variant, "n": n}

    for col in METRICS + ["objective"]:
        vals = vdf[col].dropna()
        result[f"{col}_mean"] = vals.mean()
        result[f"{col}_std"] = vals.std()
        if len(vals) >= 2:
            ci = stats.t.interval(
                0.95, df=len(vals) - 1, loc=vals.mean(), scale=stats.sem(vals)
            )
            result[f"{col}_ci_lo"] = ci[0]
            result[f"{col}_ci_hi"] = ci[1]
        else:
            result[f"{col}_ci_lo"] = vals.mean()
            result[f"{col}_ci_hi"] = vals.mean()

    if baseline is not None:
        bdf = trimmed_scores(df, baseline, trim_min)
        if len(bdf) >= 2 and n >= 2:
            for col in METRICS + ["objective"]:
                v_vals = vdf[col].dropna()
                b_vals = bdf[col].dropna()
                if len(v_vals) >= 2 and len(b_vals) >= 2:
                    t_stat, p_val = stats.ttest_ind(v_vals, b_vals, equal_var=False)
                    # Cohen's d
                    pooled_std = np.sqrt((v_vals.std() ** 2 + b_vals.std() ** 2) / 2)
                    d = (
                        (v_vals.mean() - b_vals.mean()) / pooled_std
                        if pooled_std > 0
                        else 0
                    )
                    result[f"{col}_p"] = p_val
                    result[f"{col}_d"] = d

    return result


def format_stats_table(
    df: pd.DataFrame,
    baseline: str | None = None,
    weights: dict = DEFAULT_WEIGHTS,
    trim_min: bool = False,
) -> str:
    """Build a rich stats table comparing all variants."""
    variants = df["variant"].unique()
    analyses = [analyze_variant(df, v, baseline, weights, trim_min) for v in variants]
    analyses.sort(key=lambda a: a.get("objective_mean", 0), reverse=True)

    lines = []
    header = (
        f"{'Rank':<5} {'Variant':<35} {'n':>3} "
        f"{'Obj':>7} {'±':>5} "
        f"{'CE':>6} {'CU':>6} {'PC':>6} {'PQ':>6}"
    )
    if baseline:
        header += f"  {'p(obj)':>7} {'d(obj)':>7} {'sig':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, a in enumerate(analyses):
        row = (
            f"{i + 1:<5} {a['variant']:<35} {a['n']:>3} "
            f"{a.get('objective_mean', 0):>7.3f} {a.get('objective_std', 0):>5.3f} "
            f"{a.get('CE_mean', 0):>6.2f} {a.get('CU_mean', 0):>6.2f} "
            f"{a.get('PC_mean', 0):>6.2f} {a.get('PQ_mean', 0):>6.2f}"
        )
        if baseline and "objective_p" in a:
            p = a["objective_p"]
            d = a["objective_d"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"  {p:>7.4f} {d:>+7.2f} {sig:>4}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log-based proxy score
# ---------------------------------------------------------------------------

# Tool-call patterns found in DJ run logs
_LOG_TOOL_RE = re.compile(
    r"\b(fader|pattern|swap|breakdown|mute|unmute|add|remove)\s*\("
)


def compute_log_proxy(log_path: str) -> dict:
    """Extract activity metrics from a DJ run log.

    Counts tool calls by type and computes a proxy score that correlates
    with content enjoyment (CE) without requiring an AudioBox call.

    Returns dict with per-action counts, actions_per_phrase, variety, proxy_score.
    """
    text = Path(log_path).read_text()
    matches = _LOG_TOOL_RE.findall(text)
    if not matches:
        return {
            "actions_per_phrase": 0.0,
            "variety": 0.0,
            "proxy_score": 0.0,
            "total_actions": 0,
            "action_counts": {},
        }

    from collections import Counter

    counts = Counter(matches)
    total = len(matches)

    # Estimate number of phrases from play() calls
    n_phrases = max(1, len(re.findall(r"\bplay\s*\(", text)))

    actions_per_phrase = total / n_phrases
    variety = len(counts) / total if total > 0 else 0.0
    proxy_score = actions_per_phrase * 0.7 + variety * 0.3

    return {
        "actions_per_phrase": round(actions_per_phrase, 3),
        "variety": round(variety, 3),
        "proxy_score": round(proxy_score, 3),
        "total_actions": total,
        "action_counts": dict(counts),
    }


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------


def append_lineage(lineage_path: str, entry: dict):
    """Append one JSON entry to the lineage JSONL file."""
    with open(lineage_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_lineage(lineage_path: str) -> list[dict]:
    """Load all entries from a lineage JSONL file."""
    entries = []
    if os.path.exists(lineage_path):
        with open(lineage_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


def run_generation(
    gen_num: int,
    parents: list[dict],
    history: list[dict],
    history_df: pd.DataFrame,
    outdir: str,
    palette_path: str,
    lineage_path: str,
    config: dict,
) -> tuple[list[dict], pd.DataFrame]:
    """Generate N variants, run with replicas via parallel_dj.sh, return results + updated df."""
    gen_dir = os.path.join(outdir, f"gen_{gen_num}")
    os.makedirs(gen_dir, exist_ok=True)

    n_variants = config["variants"]
    n_replicas = config["replicas"]

    # Assign knob hints: breadth-first in gen 1, free choice later
    tried_knobs = (
        set(history_df["knob"].dropna().unique()) if not history_df.empty else set()
    )
    untried = [k for k in KNOBS if k not in tried_knobs]
    knob_hints: list[str | None] = []
    for i in range(n_variants):
        if gen_num <= 1 and i < len(untried):
            knob_hints.append(untried[i])
        else:
            knob_hints.append(None)

    # --- Phase 1: generate variant prompts via meta-LLM ---
    # Track: variant name → {knob, parent, prompt_text, prompt_paths (per replica)}
    variant_info: list[dict] = []
    all_prompt_paths: list[str] = []

    for i in range(n_variants):
        parent = parents[i % len(parents)]
        print(
            f"\n  Generating variant {i + 1}/{n_variants} from {parent['variant']}..."
        )

        result = generate_variant(
            parent_text=parent["prompt_text"],
            parent_scores=parent.get("scores", {}),
            parent_objective=parent.get("objective", 0),
            history_df=history_df,
            meta_lm=config["meta_lm"],
            knob_hint=knob_hints[i],
        )
        if result is None:
            print(f"    FAILED to generate variant {i + 1}")
            continue

        prompt_text, knob = result
        variant_name = f"variant_{gen_num}_{i}_{knob}"

        # Write N replica prompt files (identical content, different names)
        replica_paths = []
        for r in range(n_replicas):
            replica_name = f"{variant_name}_rep{r}"
            prompt_path = os.path.join(gen_dir, f"{replica_name}.txt")
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            replica_paths.append(prompt_path)
            all_prompt_paths.append(prompt_path)

        variant_info.append(
            {
                "variant": variant_name,
                "gen": gen_num,
                "parent": parent["variant"],
                "knob": knob,
                "prompt_text": prompt_text,
                "replica_paths": replica_paths,
            }
        )
        print(f"    Generated: {variant_name} (knob: {knob}, {n_replicas} replicas)")

    if not all_prompt_paths:
        print("  No variants generated!")
        return [], history_df

    # --- Phase 2: run all replicas via parallel_dj.sh ---
    print(f"\n  Running {len(all_prompt_paths)} prompt files via parallel_dj.sh...")
    ok = run_parallel_dj(all_prompt_paths, gen_dir, palette_path, config)
    if not ok:
        print("  WARNING: parallel_dj.sh exited with non-zero status")

    # --- Phase 3: read scores and match to variants ---
    score_entries = parse_scores(gen_dir)
    label_to_scores: dict[str, dict] = {}
    for entry in score_entries:
        label_to_scores[entry["prompt"]] = entry["scores"]

    new_rows: list[dict] = []
    for vi in variant_info:
        for r, rpath in enumerate(vi["replica_paths"]):
            label = Path(rpath).stem
            scores = label_to_scores.get(label)
            if scores is None:
                print(f"    No score for {label}")
                continue

            obj = compute_objective(scores, config["weights"])
            row = {
                "variant": vi["variant"],
                "gen": vi["gen"],
                "parent": vi["parent"],
                "knob": vi["knob"],
                "replica": r,
                "scores": scores,
                "objective": obj,
            }
            append_lineage(lineage_path, row)
            history.append({**row, "prompt_text": vi["prompt_text"]})
            new_rows.append(
                {
                    **row,
                    **{m: scores.get(m, np.nan) for m in METRICS},
                }
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        history_df = pd.concat([history_df, new_df], ignore_index=True)

    # Print per-variant stats
    for vi in variant_info:
        vdf = history_df[history_df["variant"] == vi["variant"]]
        if vdf.empty:
            continue
        obj_mean = vdf["objective"].mean()
        obj_std = vdf["objective"].std()
        n = len(vdf)
        print(
            f"    {vi['variant']}: obj={obj_mean:.3f}±{obj_std:.3f} (n={n}) "
            + " ".join(f"{m}={vdf[m].mean():.2f}" for m in METRICS)
        )

    return [
        vi
        for vi in variant_info
        if not history_df[history_df["variant"] == vi["variant"]].empty
    ], history_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_grid_search(args, config: dict):
    """Run grid search over numeric template parameters."""
    import itertools
    import random

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"output/grid_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    lineage_path = os.path.join(outdir, "lineage.jsonl")

    # Generate all parameter combinations
    param_names = list(NUMERIC_PARAMS.keys())
    param_values = [NUMERIC_PARAMS[k] for k in param_names]
    all_combos = [
        dict(zip(param_names, vals)) for vals in itertools.product(*param_values)
    ]
    print(f"Total parameter combinations: {len(all_combos)}")

    if args.grid_subset and args.grid_subset < len(all_combos):
        random.shuffle(all_combos)
        all_combos = all_combos[: args.grid_subset]
        print(f"Subsampled to {len(all_combos)} combinations")

    config["trim_min"] = args.trim_min
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Generating shared palette...")
    palette_path = generate_palette(outdir)

    # Render all prompt variants
    variant_map: dict[str, dict] = {}  # variant_name → {params, prompt_text}
    for combo in all_combos:
        name = "grid_sp{}_lp{}_m{}_g{}".format(
            combo["setup_phrase_bars"],
            combo["loop_phrase_bars"],
            combo["min_moves"],
            str(int(combo["gain_range_width"] * 100)).zfill(3),
        )
        prompt_text = render_template(numeric_params=combo)
        variant_map[name] = {"params": combo, "prompt_text": prompt_text}

    # Batch execution: ~20 prompt files per parallel_dj.sh invocation
    batch_size = 10
    all_variant_names = list(variant_map.keys())
    n_replicas = config["replicas"]
    history: list[dict] = []

    for batch_start in range(0, len(all_variant_names), batch_size):
        batch_names = all_variant_names[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size
        batch_dir = os.path.join(outdir, f"batch_{batch_num}")
        os.makedirs(batch_dir, exist_ok=True)

        all_prompt_paths: list[str] = []
        replica_info: list[dict] = []  # track (variant, replica, path)

        for vname in batch_names:
            vi = variant_map[vname]
            for r in range(n_replicas):
                replica_name = f"{vname}_rep{r}"
                prompt_path = os.path.join(batch_dir, f"{replica_name}.txt")
                with open(prompt_path, "w") as f:
                    f.write(vi["prompt_text"])
                all_prompt_paths.append(prompt_path)
                replica_info.append(
                    {"variant": vname, "replica": r, "path": prompt_path}
                )

        print(
            f"\n--- Batch {batch_num} ({len(batch_names)} combos, "
            f"{len(all_prompt_paths)} runs) ---"
        )
        ok = run_parallel_dj(all_prompt_paths, batch_dir, palette_path, config)
        if not ok:
            print(f"  WARNING: batch {batch_num} exited with non-zero status")

        score_entries = parse_scores(batch_dir)
        label_to_scores: dict[str, dict] = {}
        for entry in score_entries:
            label_to_scores[entry["prompt"]] = entry["scores"]

        for ri in replica_info:
            label = Path(ri["path"]).stem
            scores = label_to_scores.get(label)
            if scores is None:
                print(f"  No score for {label}")
                continue
            obj = compute_objective(scores, config["weights"])
            row = {
                "variant": ri["variant"],
                "gen": 0,
                "parent": None,
                "knob": "grid",
                "replica": ri["replica"],
                "scores": scores,
                "objective": obj,
                "params": variant_map[ri["variant"]]["params"],
            }
            append_lineage(lineage_path, row)
            history.append(
                {**row, "prompt_text": variant_map[ri["variant"]]["prompt_text"]}
            )

    # Build results
    history_df = scores_to_dataframe(history)
    # Add params columns from history
    for key in NUMERIC_PARAMS:
        history_df[key] = [h.get("params", {}).get(key) for h in history]

    # Leaderboard sorted by (optionally trimmed) mean objective
    print(f"\n{'=' * 60}")
    print("=== GRID SEARCH COMPLETE ===")
    print(f"{'=' * 60}")

    variant_stats = []
    for vname in history_df["variant"].unique():
        vdf = trimmed_scores(history_df, vname, args.trim_min)
        if vdf.empty:
            continue
        variant_stats.append(
            {
                "variant": vname,
                "n": len(vdf),
                "obj_mean": vdf["objective"].mean(),
                "obj_std": vdf["objective"].std(),
                **{m: vdf[m].mean() for m in METRICS},
                **variant_map[vname]["params"],
            }
        )
    variant_stats.sort(key=lambda x: x["obj_mean"], reverse=True)

    header = (
        f"{'Rank':<5} {'Variant':<30} {'n':>3} "
        f"{'Obj':>7} {'±':>5} "
        f"{'CE':>6} {'CU':>6} {'PC':>6} {'PQ':>6} "
        f"{'sp':>3} {'lp':>3} {'mv':>3} {'gw':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for i, s in enumerate(variant_stats):
        print(
            f"{i + 1:<5} {s['variant']:<30} {s['n']:>3} "
            f"{s['obj_mean']:>7.3f} {s['obj_std']:>5.3f} "
            f"{s['CE']:>6.2f} {s['CU']:>6.2f} {s['PC']:>6.2f} {s['PQ']:>6.2f} "
            f"{s['setup_phrase_bars']:>3} {s['loop_phrase_bars']:>3} "
            f"{s['min_moves']:>3} {s['gain_range_width']:>5.2f}"
        )

    # Save results
    leaderboard_lines = [header, "-" * len(header)]
    for i, s in enumerate(variant_stats):
        leaderboard_lines.append(
            f"{i + 1:<5} {s['variant']:<30} {s['n']:>3} "
            f"{s['obj_mean']:>7.3f} {s['obj_std']:>5.3f} "
            f"{s['CE']:>6.2f} {s['CU']:>6.2f} {s['PC']:>6.2f} {s['PQ']:>6.2f} "
            f"{s['setup_phrase_bars']:>3} {s['loop_phrase_bars']:>3} "
            f"{s['min_moves']:>3} {s['gain_range_width']:>5.2f}"
        )
    Path(os.path.join(outdir, "leaderboard.txt")).write_text(
        "\n".join(leaderboard_lines) + "\n"
    )

    history_df.to_csv(os.path.join(outdir, "results.csv"), index=False)

    if variant_stats:
        best = variant_stats[0]
        print(f"\nBest: {best['variant']} (obj={best['obj_mean']:.3f})")
        best_entries = [
            h for h in history if h["variant"] == best["variant"] and "prompt_text" in h
        ]
        if best_entries:
            best_path = os.path.join(outdir, "best_prompt.txt")
            Path(best_path).write_text(best_entries[0]["prompt_text"])
            print(f"Best prompt saved to: {best_path}")

    print(f"\nOutput directory: {outdir}")


def run_narrow_grid(args, config: dict):
    """Two-stage narrowed grid search over the best parameter subspace.

    Stage 1: 6 combos (lp=[4,8] x g=[0.05,0.10,0.15]) x replicas
    Stage 2: top-2 combos x 3 additional replicas for validation
    """
    import itertools

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"output/narrow_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    lineage_path = os.path.join(outdir, "lineage.jsonl")

    # Fixed params from prior grid search findings
    fixed = {"setup_phrase_bars": 4, "min_moves": 2}
    narrow_grid = {
        "loop_phrase_bars": [4, 8],
        "gain_range_width": [0.05, 0.10, 0.15],
    }
    n_replicas = config["replicas"]

    param_names = list(narrow_grid.keys())
    param_values = [narrow_grid[k] for k in param_names]
    all_combos = [
        {**fixed, **dict(zip(param_names, vals))}
        for vals in itertools.product(*param_values)
    ]
    print(
        f"Narrow grid: {len(all_combos)} combos × {n_replicas} replicas = "
        f"{len(all_combos) * n_replicas} runs (stage 1)"
    )

    config["trim_min"] = args.trim_min
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(
            {**config, "fixed_params": fixed, "narrow_grid": narrow_grid}, f, indent=2
        )

    print("Generating shared palette...")
    palette_path = generate_palette(outdir)

    # Render all prompt variants
    variant_map: dict[str, dict] = {}
    for combo in all_combos:
        name = "narrow_lp{}_g{}".format(
            combo["loop_phrase_bars"],
            str(int(combo["gain_range_width"] * 100)).zfill(3),
        )
        prompt_text = render_template(numeric_params=combo)
        variant_map[name] = {"params": combo, "prompt_text": prompt_text}

    history: list[dict] = []

    def _run_batch(variant_names: list[str], stage_label: str, replicas: int):
        """Run a batch of variants and collect scores."""
        batch_dir = os.path.join(outdir, stage_label)
        os.makedirs(batch_dir, exist_ok=True)

        all_prompt_paths: list[str] = []
        replica_info: list[dict] = []

        for vname in variant_names:
            vi = variant_map[vname]
            for r in range(replicas):
                replica_name = f"{vname}_rep{r}"
                prompt_path = os.path.join(batch_dir, f"{replica_name}.txt")
                with open(prompt_path, "w") as f:
                    f.write(vi["prompt_text"])
                all_prompt_paths.append(prompt_path)
                replica_info.append(
                    {"variant": vname, "replica": r, "path": prompt_path}
                )

        print(
            f"\n--- {stage_label} ({len(variant_names)} combos, "
            f"{len(all_prompt_paths)} runs) ---"
        )
        ok = run_parallel_dj(all_prompt_paths, batch_dir, palette_path, config)
        if not ok:
            print(f"  WARNING: {stage_label} exited with non-zero status")

        score_entries = parse_scores(batch_dir)
        label_to_scores: dict[str, dict] = {}
        for entry in score_entries:
            label_to_scores[entry["prompt"]] = entry["scores"]

        # Collect log proxy scores from DJ run logs
        for ri in replica_info:
            label = Path(ri["path"]).stem
            log_path = os.path.join(batch_dir, label, "dj.log")
            proxy = compute_log_proxy(log_path) if os.path.exists(log_path) else {}

            scores = label_to_scores.get(label)
            if scores is None:
                print(f"  No score for {label}")
                continue
            obj = compute_objective(scores, config["weights"])
            row = {
                "variant": ri["variant"],
                "gen": 0,
                "parent": None,
                "knob": "narrow_grid",
                "replica": ri["replica"],
                "scores": scores,
                "objective": obj,
                "params": variant_map[ri["variant"]]["params"],
            }
            if proxy:
                row["log_proxy"] = proxy
            append_lineage(lineage_path, row)
            history.append(
                {**row, "prompt_text": variant_map[ri["variant"]]["prompt_text"]}
            )

    # ---- Stage 1: all combos ----
    all_variant_names = list(variant_map.keys())
    _run_batch(all_variant_names, "stage_1", n_replicas)

    # Build stage 1 results
    history_df = scores_to_dataframe(history)
    for key in list(fixed.keys()) + list(narrow_grid.keys()):
        history_df[key] = [h.get("params", {}).get(key) for h in history]

    # Rank by mean objective
    variant_means: dict[str, float] = {}
    for vname in history_df["variant"].unique():
        vdf = trimmed_scores(history_df, vname, args.trim_min)
        if not vdf.empty:
            variant_means[vname] = vdf["objective"].mean()

    ranking = sorted(variant_means, key=variant_means.get, reverse=True)

    print(f"\n{'=' * 60}")
    print("=== STAGE 1 LEADERBOARD ===")
    print(f"{'=' * 60}")
    _print_narrow_leaderboard(
        history_df, variant_map, variant_means, ranking, args.trim_min
    )

    # ---- Stage 2: top-2 combos, 3 additional replicas ----
    top2 = ranking[:2]
    stage2_replicas = 3
    print(
        f"\nStage 2: validating top-2 with {stage2_replicas} additional replicas each"
    )
    print(f"  Top-2: {top2}")

    _run_batch(top2, "stage_2", stage2_replicas)

    # Rebuild full results
    history_df = scores_to_dataframe(history)
    for key in list(fixed.keys()) + list(narrow_grid.keys()):
        history_df[key] = [h.get("params", {}).get(key) for h in history]

    variant_means = {}
    for vname in history_df["variant"].unique():
        vdf = trimmed_scores(history_df, vname, args.trim_min)
        if not vdf.empty:
            variant_means[vname] = vdf["objective"].mean()

    ranking = sorted(variant_means, key=variant_means.get, reverse=True)

    print(f"\n{'=' * 60}")
    print("=== FINAL LEADERBOARD (stage 1 + stage 2) ===")
    print(f"{'=' * 60}")
    _print_narrow_leaderboard(
        history_df, variant_map, variant_means, ranking, args.trim_min
    )

    # Significance test: top-1 vs top-2
    if len(ranking) >= 2:
        a = analyze_variant(
            history_df, ranking[0], baseline=ranking[1], trim_min=args.trim_min
        )
        if "objective_p" in a:
            p = a["objective_p"]
            d = a["objective_d"]
            sig = (
                "p<0.001"
                if p < 0.001
                else "p<0.01"
                if p < 0.01
                else "p<0.05"
                if p < 0.05
                else f"p={p:.3f}"
            )
            print(f"\n  #1 vs #2: {sig}, d={d:+.2f}")
        else:
            print("\n  Insufficient replicas for significance test")

    # Save artifacts
    history_df.to_csv(os.path.join(outdir, "results.csv"), index=False)

    if ranking:
        best = ranking[0]
        best_entries = [
            h for h in history if h["variant"] == best and "prompt_text" in h
        ]
        if best_entries:
            best_path = os.path.join(outdir, "best_prompt.txt")
            Path(best_path).write_text(best_entries[0]["prompt_text"])
            print(f"\nBest prompt saved to: {best_path}")

    print(f"Output directory: {outdir}")


def _print_narrow_leaderboard(
    history_df: pd.DataFrame,
    variant_map: dict,
    variant_means: dict,
    ranking: list[str],
    trim_min: bool,
):
    """Print a compact leaderboard for narrow grid results."""
    header = (
        f"{'Rank':<5} {'Variant':<20} {'n':>3} "
        f"{'Obj':>7} {'±':>5} "
        f"{'CE':>6} {'CU':>6} {'PC':>6} {'PQ':>6} "
        f"{'lp':>3} {'gw':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for i, vname in enumerate(ranking):
        vdf = trimmed_scores(history_df, vname, trim_min)
        if vdf.empty:
            continue
        params = variant_map[vname]["params"]
        print(
            f"{i + 1:<5} {vname:<20} {len(vdf):>3} "
            f"{variant_means[vname]:>7.3f} {vdf['objective'].std():>5.3f} "
            f"{vdf['CE'].mean():>6.2f} {vdf['CU'].mean():>6.2f} "
            f"{vdf['PC'].mean():>6.2f} {vdf['PQ'].mean():>6.2f} "
            f"{params['loop_phrase_bars']:>3} {params['gain_range_width']:>5.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Automated DJ prompt optimization with statistical analysis."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", help="Path to seed prompt .txt file")
    source.add_argument(
        "--template",
        action="store_true",
        help="Use Jinja2 template with default params as seed",
    )
    source.add_argument(
        "--grid",
        action="store_true",
        help="Grid search over numeric template parameters (no meta-LLM)",
    )
    source.add_argument(
        "--grid-narrow",
        action="store_true",
        help="Two-stage narrowed grid (sp=4, m=2, lp=[4,8], g=[0.05,0.10,0.15])",
    )

    parser.add_argument(
        "--generations", type=int, default=4, help="Number of generations (default: 4)"
    )
    parser.add_argument(
        "--variants", type=int, default=4, help="Variants per generation (default: 4)"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=3,
        help="Replicas per variant for statistical power (default: 3)",
    )
    parser.add_argument(
        "--bars", type=int, default=128, help="Bars per DJ run (default: 128)"
    )
    parser.add_argument("--bpm", type=int, default=128, help="BPM (default: 128)")
    parser.add_argument(
        "--meta-lm", default=DEFAULT_META_LM, help="Meta-LLM for variant generation"
    )
    parser.add_argument("--dj-lm", default=DEFAULT_DJ_LM, help="DJ LLM model ID")
    parser.add_argument(
        "--critic-lm", default=DEFAULT_CRITIC_LM, help="Critic LLM model ID"
    )
    parser.add_argument(
        "--resume", default=None, help="Resume from existing output directory"
    )
    parser.add_argument(
        "--trim-min",
        action="store_true",
        default=False,
        help="Drop lowest-scoring replica per variant (requires n>=4)",
    )
    parser.add_argument(
        "--grid-subset",
        type=int,
        default=None,
        help="Subsample N random combinations for grid search",
    )
    args = parser.parse_args()

    config = {
        "meta_lm": args.meta_lm,
        "dj_lm": args.dj_lm,
        "critic_lm": args.critic_lm,
        "bars": args.bars,
        "bpm": args.bpm,
        "generations": args.generations,
        "variants": args.variants,
        "replicas": args.replicas,
        "weights": DEFAULT_WEIGHTS,
        "trim_min": args.trim_min,
    }

    # Grid search modes — separate flow, then exit
    if args.grid:
        run_grid_search(args, config)
        return

    if args.grid_narrow:
        run_narrow_grid(args, config)
        return

    if args.resume:
        # ---- Resume from existing run ----
        outdir = args.resume.rstrip("/")
        lineage_path = os.path.join(outdir, "lineage.jsonl")
        palette_path = os.path.join(outdir, "palette.json")

        if not os.path.exists(lineage_path):
            print(f"No lineage.jsonl found in {outdir}")
            sys.exit(1)

        history = load_lineage(lineage_path)
        # Re-attach prompt texts from disk
        for entry in history:
            gen_dir = os.path.join(outdir, f"gen_{entry['gen']}")
            # Try both variant-level and replica-level filenames
            for pattern in [f"{entry['variant']}.txt", f"{entry['variant']}_rep0.txt"]:
                prompt_file = os.path.join(gen_dir, pattern)
                if os.path.exists(prompt_file):
                    entry["prompt_text"] = Path(prompt_file).read_text()
                    break

        history_df = scores_to_dataframe(history)
        last_gen = int(history_df["gen"].max())
        start_gen = last_gen + 1
        print(f"Resuming from {outdir}, generation {start_gen}")
        print(
            f"Loaded {len(history)} entries ({history_df['variant'].nunique()} variants)"
        )

        baseline = (
            history_df[history_df["gen"] == 0]["variant"].iloc[0]
            if 0 in history_df["gen"].values
            else None
        )
        print(
            f"\n{format_stats_table(history_df, baseline=baseline, trim_min=args.trim_min)}\n"
        )

    else:
        # ---- Fresh run ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"output/optimize_{timestamp}"
        os.makedirs(outdir, exist_ok=True)
        lineage_path = os.path.join(outdir, "lineage.jsonl")
        palette_path = os.path.join(outdir, "palette.json")

        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print("Generating shared palette...")
        palette_path = generate_palette(outdir)

        # --- Gen 0: score seed prompt with replicas ---
        print(f"\n{'=' * 60}")
        print(f"=== Generation 0: Score Seed ({args.replicas} replicas) ===")
        print(f"{'=' * 60}")

        if args.template:
            seed_text = render_template()
            print("Using template-rendered seed prompt")
        else:
            seed_text = Path(args.seed).read_text()
        seed_variant = "variant_0_seed"
        gen0_dir = os.path.join(outdir, "gen_0")
        os.makedirs(gen0_dir, exist_ok=True)

        # Write N replica prompt files for the seed
        seed_prompt_paths = []
        for r in range(args.replicas):
            replica_name = f"{seed_variant}_rep{r}"
            prompt_path = os.path.join(gen0_dir, f"{replica_name}.txt")
            with open(prompt_path, "w") as f:
                f.write(seed_text)
            seed_prompt_paths.append(prompt_path)

        print(f"Running seed prompt ({args.bars} bars, {args.replicas} replicas)...")
        ok = run_parallel_dj(seed_prompt_paths, gen0_dir, palette_path, config)
        if not ok:
            print(f"Seed prompt failed! Check logs in {gen0_dir}/")
            sys.exit(1)

        score_entries = parse_scores(gen0_dir)
        if not score_entries:
            print("Failed to score seed WAV!")
            sys.exit(1)

        history: list[dict] = []
        for r, sp in enumerate(seed_prompt_paths):
            label = Path(sp).stem
            matched = [e for e in score_entries if e["prompt"] == label]
            if not matched:
                print(f"  WARNING: no score for replica {label}")
                continue
            scores = matched[0]["scores"]
            obj = compute_objective(scores, config["weights"])
            entry = {
                "variant": seed_variant,
                "gen": 0,
                "parent": None,
                "knob": "seed",
                "replica": r,
                "scores": scores,
                "objective": obj,
                "prompt_text": seed_text,
            }
            append_lineage(
                lineage_path, {k: v for k, v in entry.items() if k != "prompt_text"}
            )
            history.append(entry)

        history_df = scores_to_dataframe(history)
        start_gen = 1

        seed_obj = history_df["objective"].mean()
        seed_std = history_df["objective"].std()
        print(
            f"\nSeed scored: obj={seed_obj:.3f}±{seed_std:.3f} (n={len(history_df)}) "
            + " ".join(f"{m}={history_df[m].mean():.2f}" for m in METRICS)
        )

    # ---- Generational loop ----
    baseline_variant = (
        history_df[history_df["gen"] == 0]["variant"].iloc[0]
        if 0 in history_df["gen"].values
        else None
    )

    for gen in range(start_gen, args.generations + 1):
        print(f"\n{'=' * 60}")
        print(f"=== Generation {gen}/{args.generations} ===")
        print(f"{'=' * 60}")

        # Select top-2 parents by (optionally trimmed) mean objective
        parent_means = {}
        for vname in history_df["variant"].unique():
            vdf = trimmed_scores(history_df, vname, args.trim_min)
            if not vdf.empty:
                parent_means[vname] = vdf["objective"].mean()
        parent_ranking = sorted(parent_means, key=parent_means.get, reverse=True)

        parents = []
        for variant_name in parent_ranking:
            entries_with_text = [
                h
                for h in history
                if h.get("variant") == variant_name and "prompt_text" in h
            ]
            if entries_with_text:
                rep = entries_with_text[0]
                vdf = trimmed_scores(history_df, variant_name, args.trim_min)
                parents.append(
                    {
                        "variant": variant_name,
                        "prompt_text": rep["prompt_text"],
                        "scores": {m: vdf[m].mean() for m in METRICS},
                        "objective": parent_means[variant_name],
                    }
                )
            if len(parents) >= 2:
                break

        if not parents:
            print("No parents with prompt text available!")
            break

        print(f"Parents: {[p['variant'] for p in parents]}")

        _, history_df = run_generation(
            gen_num=gen,
            parents=parents,
            history=history,
            history_df=history_df,
            outdir=outdir,
            palette_path=palette_path,
            lineage_path=lineage_path,
            config=config,
        )

        print(
            f"\n{format_stats_table(history_df, baseline=baseline_variant, trim_min=args.trim_min)}\n"
        )

    # ---- Final summary ----
    print(f"\n{'=' * 60}")
    print("=== OPTIMIZATION COMPLETE ===")
    print(f"{'=' * 60}")

    final_table = format_stats_table(
        history_df, baseline=baseline_variant, trim_min=args.trim_min
    )
    print(f"\n{final_table}\n")

    # Save leaderboard
    leaderboard_path = os.path.join(outdir, "leaderboard.txt")
    Path(leaderboard_path).write_text(final_table + "\n")

    # Save full results as CSV
    csv_path = os.path.join(outdir, "results.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"Full results: {csv_path}")

    # Save best prompt (using optionally trimmed means)
    best_means = {}
    for vname in history_df["variant"].unique():
        vdf = trimmed_scores(history_df, vname, args.trim_min)
        if not vdf.empty:
            best_means[vname] = vdf["objective"].mean()
    best_variant = max(best_means, key=best_means.get)
    best_obj = best_means[best_variant]
    print(f"Best: {best_variant} (obj={best_obj:.3f})")

    best_entries = [
        h for h in history if h.get("variant") == best_variant and "prompt_text" in h
    ]
    if best_entries:
        best_path = os.path.join(outdir, "best_prompt.txt")
        Path(best_path).write_text(best_entries[0]["prompt_text"])
        print(f"Best prompt saved to: {best_path}")

    # Significance summary
    if baseline_variant and len(history_df["variant"].unique()) > 1:
        print("\n--- Significance vs seed ---")
        for variant in history_df["variant"].unique():
            if variant == baseline_variant:
                continue
            a = analyze_variant(
                history_df, variant, baseline=baseline_variant, trim_min=args.trim_min
            )
            if "objective_p" in a:
                p = a["objective_p"]
                d = a["objective_d"]
                sig = (
                    "p<0.001"
                    if p < 0.001
                    else "p<0.01"
                    if p < 0.01
                    else "p<0.05"
                    if p < 0.05
                    else f"p={p:.3f}"
                )
                print(f"  {variant}: {sig}, d={d:+.2f}")
            else:
                print(f"  {variant}: insufficient replicas for significance test")

    print(f"\nOutput directory: {outdir}")


if __name__ == "__main__":
    main()
