"""End-to-end compose → export → analyze pipeline.

Runs the RLM compose pipeline with audio export, then sends the exported
WAV through all aesthetics endpoints. Saves a single JSON report with
trajectory + analysis results for review.

Usage:
    uv run python scripts/e2e_compose.py "dark minimal techno" --live --duration 5
    uv run python scripts/e2e_compose.py "ambient in F minor" --live --max-iterations 10
    uv run python scripts/e2e_compose.py --help
"""

import argparse
import json
import os
import re
import time

import dspy
from dotenv import load_dotenv

load_dotenv()


def _slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def parse_args():
    p = argparse.ArgumentParser(description="E2E: compose → export → analyze → report")
    p.add_argument("brief", help="Creative brief for the composition")
    p.add_argument(
        "--model",
        default=os.environ.get("BERGAIN_MODEL", "openrouter/openai/gpt-5"),
    )
    p.add_argument("--sub-model", default=None)
    p.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Live composition mode (default: True)",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Target duration in minutes (default: 5)",
    )
    p.add_argument("--max-iterations", type=int, default=15)
    p.add_argument("--max-llm-calls", type=int, default=20)
    p.add_argument("--min-clips", type=int, default=3)
    p.add_argument("--output-dir", default="./output/e2e/")
    p.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip aesthetics analysis (just compose + export)",
    )
    return p.parse_args()


def run_analysis(wav_path, bpm):
    """Run all 4 analyses on the exported WAV."""
    from bergain.osc import analyze_audio, score_audio

    results = {}

    for analysis_type in ["key", "energy", "structure"]:
        print(f"  Analyzing: {analysis_type}...")
        try:
            results[analysis_type] = analyze_audio(wav_path, analysis_type, bpm=bpm)
            print("    OK")
        except Exception as e:
            results[analysis_type] = {"error": str(e)}
            print(f"    FAILED: {e}")

    print("  Scoring: audiobox_aesthetics...")
    try:
        results["score"] = score_audio(wav_path)
        print("    OK")
    except Exception as e:
        results["score"] = {"error": str(e)}
        print(f"    FAILED: {e}")

    return results


def print_summary(report):
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 60)
    print("E2E REPORT SUMMARY")
    print("=" * 60)

    print(f"\n  Brief: {report['brief']}")
    print(f"  WAV:   {report.get('wav_path', 'N/A')}")
    print(f"  Time:  {report['timestamp']}")

    if "report" in report:
        print(f"\n  RLM Report:\n    {report['report'][:200]}")

    analysis = report.get("analysis", {})

    if "key" in analysis and "error" not in analysis["key"]:
        k = analysis["key"]
        print(f"\n  Key: {k['key']} {k['scale']} (confidence: {k['confidence']:.2f})")

    if "energy" in analysis and "error" not in analysis["energy"]:
        frames = analysis["energy"].get("frames", [])
        if frames:
            rms_vals = [f["rms"] for f in frames]
            avg = sum(rms_vals) / len(rms_vals)
            peak = max(rms_vals)
            print(
                f"  Energy: avg_rms={avg:.4f}, peak_rms={peak:.4f}, frames={len(frames)}"
            )

    if "score" in analysis and "error" not in analysis["score"]:
        scores = analysis["score"].get("scores", {})
        for name, val in scores.items():
            if isinstance(val, (int, float)):
                print(f"  Score/{name}: {val:.3f}")

    if "structure" in analysis and "error" not in analysis["structure"]:
        print(f"  Structure: {json.dumps(analysis['structure'], indent=2)[:200]}")

    print()


def main():
    args = parse_args()
    brief = args.brief

    if args.live and args.duration < 5:
        print(f"  [NOTE] Bumping duration from {args.duration} to 5 min")
        args.duration = 5

    # Configure LMs
    lm = dspy.LM(args.model, cache=False)
    sub_lm = dspy.LM(args.sub_model, cache=False) if args.sub_model else lm
    dspy.configure(lm=lm)

    from bergain.compose import LiveCompose, Compose

    signature = LiveCompose if args.live else Compose

    print("=" * 60)
    print("E2E COMPOSE → EXPORT → ANALYZE")
    print("=" * 60)
    print(f"  Mode:       {'LIVE' if args.live else 'PALETTE'} ({args.duration} min)")
    print(f"  Model:      {args.model}")
    print(f"  Sub-model:  {args.sub_model or args.model}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  LLM calls:  {args.max_llm_calls}")
    print(f"  Brief:      {brief[:80]}{'...' if len(brief) > 80 else ''}")
    print()

    # Connect to Ableton
    from bergain.session import Session
    from bergain.tools import make_tools

    session = Session()
    tools, _, live_history = make_tools(
        session,
        min_clips=args.min_clips,
        live_mode=args.live,
        duration_minutes=args.duration,
        sub_lm=sub_lm if args.live else None,
        brief=brief,
    )

    composer = dspy.RLM(
        signature,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_output_chars=15000,
        tools=tools,
        sub_lm=sub_lm,
        verbose=True,
    )

    # --- Phase 1: Compose + Export ---
    print("\n--- Phase 1: Compose + Export ---\n")
    session.start_recording()

    prediction = None
    wav_path = None
    bpm = None
    trajectory = []
    try:
        prediction = composer(brief=brief)
        print(f"\n  RLM Report: {prediction.report}")
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping...")
    except Exception as e:
        print(f"\n  Compose error: {e}")
    finally:
        try:
            bpm = float(session.api.song.get("tempo"))
        except Exception:
            pass
        wav_path = session.stop_recording(
            export_dir=os.path.join(args.output_dir, "wav")
        )
        session.close()

    # Extract trajectory
    if prediction:
        try:
            raw = prediction.trajectory
            if isinstance(raw, str):
                trajectory = json.loads(raw)
            elif isinstance(raw, list):
                trajectory = raw
        except (AttributeError, json.JSONDecodeError, TypeError):
            pass

    # --- Phase 2: Analyze ---
    analysis = {}
    if wav_path and not args.skip_analysis:
        print("\n--- Phase 2: Aesthetics Analysis ---\n")
        analysis = run_analysis(wav_path, bpm)
    elif not wav_path:
        print("\n  Skipping analysis — no WAV exported")

    # --- Phase 3: Save Report ---
    print("\n--- Phase 3: Save Report ---\n")
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(brief)

    report = {
        "brief": brief,
        "report": getattr(prediction, "report", "") if prediction else "",
        "timestamp": ts,
        "config": {
            "model": args.model,
            "sub_model": args.sub_model or args.model,
            "live": args.live,
            "duration": args.duration,
            "max_iterations": args.max_iterations,
            "max_llm_calls": args.max_llm_calls,
        },
        "wav_path": wav_path,
        "trajectory": trajectory,
        "live_history": live_history or [],
        "analysis": analysis,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{ts}_{slug}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Report saved: {report_path}")
    print_summary(report)


if __name__ == "__main__":
    main()
