#!/usr/bin/env python
"""GEPA optimization of Compose signature instructions.

Evolves the Compose docstring to produce better compositions, measured by
structural quality metrics via StubSession. No Ableton needed.

Usage:
    # Dry run — score current instructions on a few briefs
    uv run python scripts/optimize_compose.py --dry-run

    # Light optimization (~50 metric calls)
    uv run python scripts/optimize_compose.py --budget light

    # Medium optimization (~200 metric calls)
    uv run python scripts/optimize_compose.py --budget medium

    # Compare original vs optimized on held-out briefs
    uv run python scripts/optimize_compose.py --compare output/gepa/best_instructions.json
"""

import argparse
import json
import os
import sys
import time

# Unbuffered stdout so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy  # noqa: E402

from bergain.compose import Compose  # noqa: E402
from bergain.eval import structural_metric, wrap_tools_for_eval  # noqa: E402
from bergain.stub import StubSession  # noqa: E402
from bergain.tools import make_tools  # noqa: E402
from bergain.trainset import TRAIN, VAL  # noqa: E402


def _log(msg):
    """Print with timestamp prefix."""
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def _patch_lm_progress(lm, label="RLM"):
    """Patch an LM to log each call for progress visibility."""
    step_counter = [0]
    orig_call = type(lm).__call__

    class _ProgressPatched(type(lm)):
        def __call__(self, *args, **kwargs):
            step_counter[0] += 1
            _log(f"{label} call #{step_counter[0]}...")
            t0 = time.time()
            result = orig_call(self, *args, **kwargs)
            dt = time.time() - t0
            _log(f"{label} call #{step_counter[0]} done ({dt:.1f}s)")
            return result

    lm.__class__ = _ProgressPatched
    return lm, step_counter


def _build_student(stub=True, max_iterations=30, max_llm_calls=40, min_clips=6):
    """Build an RLM student program wrapping Compose with StubSession tools."""
    session = StubSession() if stub else None
    if session is None:
        raise ValueError("Live session not supported in optimization — use stub=True")

    tools, _, _ = make_tools(session, min_clips=min_clips)
    wrap_tools_for_eval(tools)

    composer = dspy.RLM(
        Compose,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_output_chars=15000,
        tools=tools,
        verbose=False,
    )
    return composer


def _dump_trajectory_debug(pred):
    """Print trajectory structure for debugging extraction issues."""
    try:
        raw = pred.trajectory
        if isinstance(raw, str):
            traj = json.loads(raw)
        elif isinstance(raw, list):
            traj = raw
        else:
            print(f"    [DEBUG] trajectory type: {type(raw)}")
            return

        print(f"    [DEBUG] trajectory: {len(traj)} steps")
        for j, step in enumerate(traj[:3]):  # first 3 steps
            if isinstance(step, dict):
                keys = list(step.keys())
                print(f"    [DEBUG]   step {j} keys: {keys}")
                for k in keys[:5]:
                    val = str(step[k])[:200]
                    print(f"    [DEBUG]     {k}: {val}")
            else:
                print(
                    f"    [DEBUG]   step {j} type: {type(step)}, val: {str(step)[:200]}"
                )
    except Exception as e:
        print(f"    [DEBUG] trajectory error: {e}")


def _run_baseline(student, briefs, judge_lm=None, max_iterations=30):
    """Run the student on briefs and print structural scores."""
    print(f"\n{'=' * 60}")
    print(f"BASELINE EVALUATION — {len(briefs)} briefs")
    print(f"{'=' * 60}\n")

    total_start = time.time()
    scores = []
    for i, example in enumerate(briefs):
        brief_text = example.brief
        _log(f"Brief {i + 1}/{len(briefs)}: {brief_text[:60]}...")

        brief_start = time.time()
        try:
            _log("Running RLM...")
            pred = student(brief=brief_text)
            rlm_elapsed = time.time() - brief_start
            _log(f"RLM finished in {rlm_elapsed:.1f}s — scoring...")

            # Dump trajectory structure for debugging (first brief only)
            if i == 0:
                _dump_trajectory_debug(pred)

            result = structural_metric(
                example,
                pred,
                judge_lm=judge_lm,
                max_iterations=max_iterations,
            )
            score = result.score
            scores.append(score)
            print(f"    Score: {score:.3f}")
            # Print sub-scores
            for line in result.feedback.split("\n"):
                if line.strip():
                    print(f"      {line.strip()}")
        except Exception as e:
            import traceback

            print(f"    ERROR: {e}")
            traceback.print_exc()
            scores.append(0.0)

        brief_elapsed = time.time() - brief_start
        _log(f"Brief {i + 1} completed in {brief_elapsed:.1f}s")
        print()

    total_elapsed = time.time() - total_start
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Average score: {avg:.3f}")
        print(f"  Min: {min(scores):.3f}  Max: {max(scores):.3f}")
        print(
            f"  Total time: {total_elapsed:.0f}s ({total_elapsed / len(scores):.0f}s/brief)"
        )

    return scores


def _run_optimization(
    student,
    budget,
    reflection_lm,
    judge_lm=None,
    output_dir="./output/gepa/",
    max_iterations=30,
    num_threads=4,
):
    """Run GEPA optimization and save results."""
    auto_map = {"light": "light", "medium": "medium", "heavy": "heavy"}
    auto = auto_map.get(budget, "light")

    print(f"\n{'=' * 60}")
    print(f"GEPA OPTIMIZATION — budget={budget} (auto={auto})")
    print(f"{'=' * 60}\n")

    # Wrap the metric to inject kwargs
    # GEPA requires 5 args: (gold, pred, trace, pred_name, pred_trace)
    def metric_fn(example, pred, trace=None, pred_name=None, pred_trace=None):
        return structural_metric(
            example,
            pred,
            trace=trace,
            judge_lm=judge_lm,
            max_iterations=max_iterations,
        )

    optimizer = dspy.GEPA(
        metric=metric_fn,
        reflection_lm=reflection_lm,
        auto=auto,
        num_threads=num_threads,
        track_stats=True,
        log_dir=os.path.join(output_dir, "logs"),
    )

    start = time.time()
    result = optimizer.compile(student, trainset=TRAIN, valset=VAL)
    elapsed = time.time() - start

    print(f"\n  Optimization completed in {elapsed / 60:.1f} minutes")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Extract optimized instructions
    optimized_instructions = None
    try:
        # Access the optimized signature's docstring
        optimized_instructions = result.generate.signature.instructions
    except AttributeError:
        try:
            optimized_instructions = result.generate.signature.__doc__
        except AttributeError:
            print("  WARNING: Could not extract optimized instructions")

    if optimized_instructions:
        output_path = os.path.join(output_dir, "best_instructions.json")
        with open(output_path, "w") as f:
            json.dump(
                {
                    "instructions": optimized_instructions,
                    "budget": budget,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "elapsed_seconds": elapsed,
                    "original_instructions": Compose.__doc__,
                },
                f,
                indent=2,
            )
        print(f"  Saved: {output_path}")

        # Print diff summary
        print(f"\n  Original instructions: {len(Compose.__doc__)} chars")
        print(f"  Optimized instructions: {len(optimized_instructions)} chars")
        print("\n  Optimized instructions preview (first 500 chars):")
        print(f"    {optimized_instructions[:500]}...")

    return result


def _run_comparison(optimized_path, max_iterations=30, judge_lm=None):
    """Compare original vs optimized instructions on validation briefs."""
    print(f"\n{'=' * 60}")
    print("A/B COMPARISON — original vs optimized")
    print(f"{'=' * 60}\n")

    with open(optimized_path) as f:
        data = json.load(f)
    optimized_instructions = data["instructions"]

    # Original
    print("--- ORIGINAL ---")
    student_orig = _build_student(max_iterations=max_iterations)
    scores_orig = _run_baseline(
        student_orig, VAL, judge_lm=judge_lm, max_iterations=max_iterations
    )

    # Optimized — patch signature docstring
    print("\n--- OPTIMIZED ---")
    Compose.__doc__ = optimized_instructions
    student_opt = _build_student(max_iterations=max_iterations)
    scores_opt = _run_baseline(
        student_opt, VAL, judge_lm=judge_lm, max_iterations=max_iterations
    )

    # Restore original
    Compose.__doc__ = data.get("original_instructions", Compose.__doc__)

    if scores_orig and scores_opt:
        avg_orig = sum(scores_orig) / len(scores_orig)
        avg_opt = sum(scores_opt) / len(scores_opt)
        delta = avg_opt - avg_orig
        print(f"\n  Original avg:  {avg_orig:.3f}")
        print(f"  Optimized avg: {avg_opt:.3f}")
        print(
            f"  Delta:         {delta:+.3f} ({'improved' if delta > 0 else 'regression' if delta < 0 else 'same'})"
        )


def _check_api_key(model):
    """Verify that the required API key is set before running."""
    from dotenv import load_dotenv

    load_dotenv()

    # Map model prefixes to required env vars
    prefix_to_env = {
        "openrouter/": "OPENROUTER_API_KEY",
        "openai/": "OPENAI_API_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
    }

    for prefix, env_var in prefix_to_env.items():
        if model.startswith(prefix):
            if not os.environ.get(env_var):
                print(f"ERROR: {env_var} not set (required for model '{model}')")
                print("  Set it in your .env file or export it:")
                print(f"    export {env_var}=your-key-here")
                sys.exit(1)
            return

    # For unknown prefixes, check common keys
    has_any = any(
        os.environ.get(v)
        for v in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_any:
        print(f"WARNING: No API key found for model '{model}'")
        print("  Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")


def main():
    parser = argparse.ArgumentParser(
        description="GEPA optimization of bergain Compose signature",
    )
    parser.add_argument(
        "--budget",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization budget (default: light)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BERGAIN_MODEL", "openrouter/openai/gpt-5"),
        help="Task LM model string",
    )
    parser.add_argument(
        "--reflection-model",
        default=None,
        help="Reflection LM for GEPA (default: same as --model)",
    )
    parser.add_argument(
        "--judge-lm",
        default=None,
        help="LM for brief adherence scoring (optional, skipped if not set)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/gepa/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max RLM iterations per evaluation",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of parallel evaluation threads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score current instructions without optimizing",
    )
    parser.add_argument(
        "--compare",
        metavar="PATH",
        help="Compare original vs optimized instructions JSON",
    )

    args = parser.parse_args()

    # Verify API key is set before burning time on initialization
    _check_api_key(args.model)

    # Print config summary
    mode = "DRY RUN" if args.dry_run else f"OPTIMIZE (budget={args.budget})"
    if args.compare:
        mode = "COMPARE"
    print(f"\n{'=' * 60}")
    print(f"  bergain GEPA — {mode}")
    print(f"  Model:          {args.model}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"{'=' * 60}")

    # Configure LMs
    _log("Initializing LMs...")
    task_lm = dspy.LM(args.model, cache=False)
    _patch_lm_progress(task_lm, label="RLM")
    dspy.configure(lm=task_lm)

    reflection_model = args.reflection_model or args.model
    reflection_lm = dspy.LM(reflection_model, temperature=1.0, max_tokens=32000)

    judge_lm = None
    if args.judge_lm:
        judge_lm = dspy.LM(args.judge_lm)

    _log("LMs ready.")

    if args.compare:
        _run_comparison(
            args.compare,
            max_iterations=args.max_iterations,
            judge_lm=judge_lm,
        )
        return

    _log("Building student RLM with StubSession...")
    student = _build_student(max_iterations=args.max_iterations)
    _log("Student ready.")

    if args.dry_run:
        # Score on a small subset of training briefs
        _run_baseline(
            student,
            TRAIN[:5],
            judge_lm=judge_lm,
            max_iterations=args.max_iterations,
        )
        return

    _run_optimization(
        student,
        budget=args.budget,
        reflection_lm=reflection_lm,
        judge_lm=judge_lm,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
