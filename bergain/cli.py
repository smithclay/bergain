"""Unified bergain CLI — compose, export, analyze in one command.

Usage:
    bergain "Dark Berlin techno, 130 BPM"                      # palette mode
    bergain --live --duration 10 "Evolving ambient in F minor"  # live mode
    bergain --analyze "score this"                              # with Modal analysis
    bergain check                                               # verify OSC connection
"""

import argparse
import json
import os
import sys
import time

import dspy
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LM patch — prints reasoning blocks and tracks call/token counts
# ---------------------------------------------------------------------------


def _patch_reasoning(lm, label="DJ", progress=None):
    """Patch a dspy.LM instance to print reasoning and track usage.

    Swaps the instance's class to a dynamic subclass that intercepts
    __call__.  isinstance(lm, dspy.LM) still passes because the patched
    class inherits from the original.
    """
    step_counter = [0]
    orig_call = type(lm).__call__

    class _Patched(type(lm)):
        def __call__(self, *args, **kwargs):
            result = orig_call(self, *args, **kwargs)
            step_counter[0] += 1

            # Extract reasoning and token usage from the response
            reasoning = ""
            tokens_this_call = 0
            if result:
                raw = result[0]
                if isinstance(raw, dict):
                    reasoning = raw.get("reasoning_content") or ""
                    usage = raw.get("usage") or {}
                    tokens_this_call = usage.get("total_tokens", 0)

            # Try to get usage from the LM's history if not in the response
            if tokens_this_call == 0:
                try:
                    hist = self.history
                    if hist:
                        latest = hist[-1]
                        usage = latest.get("usage") or latest.get("response", {}).get(
                            "usage", {}
                        )
                        tokens_this_call = usage.get("total_tokens", 0)
                except Exception:
                    pass

            # Update progress state for heartbeat display
            if progress:
                progress.llm_calls += 1
                progress.llm_tokens += tokens_this_call

                # Append to stream log for TUI
                import time as _time

                progress.stream.append(
                    {
                        "type": "step",
                        "content": f"[{label} step {step_counter[0]}]"
                        + (f" ({tokens_this_call:,} tok)" if tokens_this_call else ""),
                        "timestamp": _time.time(),
                    }
                )
                if reasoning:
                    text_for_stream = reasoning.strip()
                    if len(text_for_stream) > 800:
                        text_for_stream = text_for_stream[:800] + "..."
                    progress.stream.append(
                        {
                            "type": "reasoning",
                            "content": text_for_stream,
                            "timestamp": _time.time(),
                        }
                    )

            # Print reasoning block
            if reasoning:
                text = reasoning.strip()
                if len(text) > 800:
                    text = text[:800] + "..."
                token_note = f"  ({tokens_this_call:,} tok)" if tokens_this_call else ""
                print(f"\n  [{label} step {step_counter[0]}]{token_note}")
                for line in text.split("\n"):
                    print(f"    {line}")

            return result

    lm.__class__ = _Patched
    return lm


def _parse_args():
    p = argparse.ArgumentParser(
        prog="bergain",
        description="AI DJ — compose music in Ableton Live via OSC",
    )
    p.add_argument(
        "brief",
        nargs="?",
        default="",
        help="Creative brief (mood, genre, instrumentation), or 'check' to verify OSC",
    )
    p.add_argument("--brief-file", help="Read brief from a file")
    p.add_argument(
        "--model",
        default=os.environ.get("BERGAIN_MODEL", "openrouter/openai/gpt-5"),
        help="Primary LM (LiteLLM model string)",
    )
    p.add_argument(
        "--sub-model",
        default=None,
        help="Sub-LM for llm_query() / compose_next() (default: same as --model)",
    )
    p.add_argument(
        "--max-iterations", type=int, default=None, help="Max REPL iterations"
    )
    p.add_argument(
        "--max-llm-calls", type=int, default=None, help="Max llm_query() calls"
    )
    p.add_argument(
        "--min-clips",
        type=int,
        default=None,
        help="Minimum clips before ready_to_submit() allows SUBMIT",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Live composition mode — real-time performance",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target duration in minutes for live mode (default: 60)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without connecting to Ableton",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip audio recording/export",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Run Modal aesthetics analysis after export",
    )
    p.add_argument(
        "--bars-per-scene",
        type=int,
        default=16,
        help="Bars per scene for palette auto-fire (default: 16)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable rich progress display, use plain output",
    )
    p.add_argument(
        "--output-dir",
        default="./output/compose/",
        help="Directory for output files (trajectory, WAV, report)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# cmd_check — OSC connection verification
# ---------------------------------------------------------------------------


def cmd_check():
    """Verify OSC connection to Ableton."""
    from .session import Session

    print("Checking OSC connection to Ableton...")
    try:
        session = Session()
        s = session.status()
        print("\n  Connected!")
        print(f"  Tempo:   {s['tempo']} BPM")
        print(f"  Playing: {'yes' if s.get('playing') else 'no'}")
        tracks = s.get("tracks", [])
        print(f"  Tracks:  {len(tracks)}")
        for t in tracks:
            devs = ", ".join(t["devices"]) if t.get("devices") else "no devices"
            print(
                f"    [{t['index']}] {t['name']}  vol={t.get('volume', 0):.2f}  ({devs})"
            )
        session.close()
    except Exception as e:
        print(f"\n  Connection failed: {e}")
        print("  Is Ableton running with AbletonOSC?")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Analysis (from e2e_compose.py)
# ---------------------------------------------------------------------------


def _run_analysis(wav_path, bpm):
    """Run all analyses on the exported WAV."""
    from .osc import analyze_audio, score_audio

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


def _print_summary(report):
    """Print a human-readable summary of the report."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        parts = []
        parts.append(f"Brief: {report['brief']}")
        if report.get("wav_path"):
            parts.append(f"WAV:   {report['wav_path']}")
        if report.get("report"):
            parts.append(f"Report: {report['report'][:200]}")

        analysis = report.get("analysis", {})
        if "key" in analysis and "error" not in analysis["key"]:
            k = analysis["key"]
            parts.append(
                f"Key: {k['key']} {k['scale']} (confidence: {k['confidence']:.2f})"
            )
        if "energy" in analysis and "error" not in analysis["energy"]:
            frames = analysis["energy"].get("frames", [])
            if frames:
                rms_vals = [f["rms"] for f in frames]
                avg = sum(rms_vals) / len(rms_vals)
                parts.append(f"Energy: avg_rms={avg:.4f}, frames={len(frames)}")
        if "score" in analysis and "error" not in analysis["score"]:
            scores = analysis["score"].get("scores", {})
            score_parts = []
            for name, val in scores.items():
                if isinstance(val, (int, float)):
                    score_parts.append(f"{name}={val:.1f}")
            if score_parts:
                parts.append(f"Scores: {', '.join(score_parts)}")

        console.print(
            Panel("\n".join(parts), title="bergain report", border_style="green")
        )
    except ImportError:
        # Fallback to plain print
        print("\n" + "=" * 60)
        print("REPORT SUMMARY")
        print("=" * 60)
        print(f"  Brief: {report['brief']}")
        if report.get("wav_path"):
            print(f"  WAV:   {report['wav_path']}")
        analysis = report.get("analysis", {})
        if "score" in analysis and "error" not in analysis["score"]:
            for name, val in analysis["score"].get("scores", {}).items():
                if isinstance(val, (int, float)):
                    print(f"  {name}: {val:.1f}")
        print()


# ---------------------------------------------------------------------------
# cmd_compose — the main pipeline
# ---------------------------------------------------------------------------


def cmd_compose(args, progress_override=None):
    """Run the full compose pipeline: compose -> export -> analyze -> report."""
    from .compose import Compose, LiveCompose, _slugify
    from .progress import PlainProgress, ProgressDisplay, ProgressState
    from .session import Session
    from .tools import make_tools

    # Resolve brief
    if args.brief_file:
        with open(args.brief_file) as f:
            brief = f.read().strip()
    elif args.brief:
        brief = args.brief
    else:
        print("Error: provide a brief as argument or via --brief-file")
        sys.exit(1)

    # Live mode minimum
    if args.live and args.duration < 5:
        print(
            f"  [NOTE] Bumping duration from {args.duration} to 5 min (minimum for live mode)"
        )
        args.duration = 5

    # Apply mode-appropriate defaults
    defaults = (60, 60, 3) if args.live else (30, 40, 6)
    if args.max_iterations is None:
        args.max_iterations = defaults[0]
    if args.max_llm_calls is None:
        args.max_llm_calls = defaults[1]
    if args.min_clips is None:
        args.min_clips = defaults[2]

    signature = LiveCompose if args.live else Compose
    mode_label = f"LIVE ({args.duration} min)" if args.live else "PALETTE"

    # Progress state + display
    if progress_override:
        state = progress_override
        state.brief = brief
        state.live = args.live
        state.duration = args.duration
        state.model = args.model
        display = PlainProgress(state)  # TUI handles real display
    else:
        state = ProgressState(
            brief=brief,
            live=args.live,
            duration=args.duration,
            model=args.model,
        )

        if args.no_progress:
            display = PlainProgress(state)
        else:
            display = ProgressDisplay(state)

    # Configure LMs — patch to show reasoning + heartbeat
    lm = _patch_reasoning(dspy.LM(args.model, cache=False), label="DJ", progress=state)
    if args.sub_model:
        sub_lm = _patch_reasoning(
            dspy.LM(args.sub_model, cache=False), label="Arranger", progress=state
        )
    else:
        sub_lm = lm
    dspy.configure(lm=lm)

    if args.dry_run:
        print(f"=== bergain {mode_label} ===")
        print(f"  Model:      {args.model}")
        print(f"  Sub-model:  {args.sub_model or args.model}")
        print(f"  Iterations: {args.max_iterations}")
        print(f"  LLM calls:  {args.max_llm_calls}")
        print(f"  Min clips:  {args.min_clips}")
        print(f"  Output:     {args.output_dir}")
        print(f"  Brief:      {brief[:100]}{'...' if len(brief) > 100 else ''}")
        print(f"\n  Signature: {signature.__name__}")
        print("  [DRY RUN] Would connect to Ableton and run RLM. Exiting.")
        return

    # Connect to Ableton
    session = Session()
    tools, _, live_history = make_tools(
        session,
        min_clips=args.min_clips,
        live_mode=args.live,
        duration_minutes=args.duration,
        sub_lm=sub_lm if args.live else None,
        brief=brief,
        progress=state,
    )

    # Build RLM — verbose=False; reasoning shown via _ReasoningPrinter wrapper
    composer = dspy.RLM(
        signature,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_output_chars=15000,
        tools=tools,
        sub_lm=sub_lm,
        verbose=False,
    )

    # Start recording unless skipped
    recording = not args.skip_export
    if recording:
        session.start_recording()

    display.start()
    state.phase = "composing"

    prediction = None
    wav_path = None
    bpm = None

    try:
        prediction = composer(brief=brief)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping playback...")
    except Exception as e:
        print(f"\n  Compose error: {e}")
    finally:
        state.phase = "firing" if not args.live else "exporting"

    # Palette mode: auto-fire scenes sequentially
    if not args.live and prediction and recording:
        try:
            state.phase = "firing"
            num_scenes = session.api.song.get("num_scenes")
            bpm_val = float(session.api.song.get("tempo"))
            bar_sec = 4 * 60.0 / bpm_val

            session.api.song.call("start_playing")
            time.sleep(2)

            for scene_idx in range(num_scenes):
                session.fire(scene_idx)
                time.sleep(args.bars_per_scene * bar_sec)
        except Exception as e:
            print(f"  [FIRE] Error during scene auto-fire: {e}")

    # Stop recording, get WAV
    if recording:
        state.phase = "exporting"
        try:
            bpm = float(session.api.song.get("tempo"))
        except Exception:
            pass
        wav_dir = os.path.join(args.output_dir, "wav")
        wav_path = session.stop_recording(export_dir=wav_dir)
    else:
        session.stop()

    # Analysis
    analysis = {}
    if wav_path and args.analyze:
        state.phase = "analyzing"
        analysis = _run_analysis(wav_path, bpm)
        state.analysis = analysis

    # Save consolidated report
    state.phase = "done"
    display.stop()

    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(brief)

    trajectory = []
    if prediction:
        try:
            raw = prediction.trajectory
            if isinstance(raw, str):
                trajectory = json.loads(raw)
            elif isinstance(raw, list):
                trajectory = raw
        except (AttributeError, json.JSONDecodeError, TypeError):
            pass

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

    print(f"\n  Report saved: {report_path}")
    if wav_path:
        print(f"  WAV: {wav_path}")

    _print_summary(report)

    session.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    # Special-case: `bergain check`
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        cmd_check()
        return

    args = _parse_args()

    # No brief and no brief-file → launch TUI
    if not args.brief and not args.brief_file:
        try:
            from .tui import BergainApp

            BergainApp().run()
        except ImportError:
            print("Error: textual is required for TUI mode.")
            print("Install with: uv add textual")
            sys.exit(1)
        return

    # Has brief → headless compose (existing behavior)
    cmd_compose(args)


if __name__ == "__main__":
    main()
