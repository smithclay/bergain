#!/usr/bin/env python3
"""Diagnostic: isolate duration vs arc effects on the OBJ gap.

Runs 4 experiments with palette_001, all scored via the Modal endpoint:

  A. Reference 32-bar  — baseline (~7.18 expected)
  B. Reference 64-bar  — reference structure doubled, isolates duration effect
  C. DJ 32-bar         — RLM-driven at 32 bars, isolates DJ decision quality
  D. DJ 64-bar         — current state (~6.81 expected)

Usage:
    uv run python scripts/score_gap_test.py
    uv run python scripts/score_gap_test.py --skip-dj    # only render A/B
    uv run python scripts/score_gap_test.py --palette palettes/curated/palette_003.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bergain.renderer import load_palette, render_bar

# Reuse reference gains and patterns from screen_palettes
from screen_palettes import (
    GAINS,
    PATTERNS,
    BPM,
    SAMPLE_RATE,
    build_reference_bars,
    score_audio,
)

DEFAULT_PALETTE = "palettes/curated/palette_001.json"


def build_reference_bars_64():
    """Build 64-bar reference: proportionally doubled from 32-bar reference.

    Each 4-bar section becomes 8 bars, preserving the same arc shape.
    """
    bars = []

    # Bars 1-8: Kick only (doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                ]
            }
        )

    # Bars 9-16: + hihat (doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["offbeat_8ths"],
                        "gain": GAINS["hihat"],
                    },
                ]
            }
        )

    # Bars 17-24: + bassline (doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["offbeat_8ths"],
                        "gain": GAINS["hihat"],
                    },
                    {
                        "role": "bassline",
                        "type": "loop",
                        "gain": GAINS["bassline"],
                    },
                ]
            }
        )

    # Bars 25-32: + perc (doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["offbeat_8ths"],
                        "gain": GAINS["hihat"],
                    },
                    {
                        "role": "bassline",
                        "type": "loop",
                        "gain": GAINS["bassline"],
                    },
                    {
                        "role": "perc",
                        "type": "oneshot",
                        "beats": PATTERNS["syncopated_a"],
                        "gain": GAINS["perc"],
                    },
                ]
            }
        )

    # Bars 33-48: Full groove (doubled from 8, with pattern evolution halfway)
    # Bars 33-40: standard full groove
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["offbeat_8ths"],
                        "gain": GAINS["hihat"],
                    },
                    {"role": "bassline", "type": "loop", "gain": GAINS["bassline"]},
                    {
                        "role": "perc",
                        "type": "oneshot",
                        "beats": PATTERNS["syncopated_a"],
                        "gain": GAINS["perc"],
                    },
                    {"role": "texture", "type": "loop", "gain": GAINS["texture"]},
                    {
                        "role": "clap",
                        "type": "oneshot",
                        "beats": PATTERNS["backbeat"],
                        "gain": GAINS["clap"],
                    },
                ]
            }
        )
    # Bars 41-48: evolved patterns (gallop perc, sixteenth hihat)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["sixteenth_drive"],
                        "gain": GAINS["hihat"],
                    },
                    {"role": "bassline", "type": "loop", "gain": GAINS["bassline"]},
                    {
                        "role": "perc",
                        "type": "oneshot",
                        "beats": PATTERNS["gallop"],
                        "gain": GAINS["perc"],
                    },
                    {"role": "texture", "type": "loop", "gain": GAINS["texture"]},
                    {
                        "role": "clap",
                        "type": "oneshot",
                        "beats": PATTERNS["backbeat"],
                        "gain": GAINS["clap"],
                    },
                ]
            }
        )

    # Bars 49-56: Strip to kick + hihat + bassline + perc (doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {
                        "role": "hihat",
                        "type": "oneshot",
                        "beats": PATTERNS["offbeat_8ths"],
                        "gain": GAINS["hihat"],
                    },
                    {"role": "bassline", "type": "loop", "gain": GAINS["bassline"]},
                    {
                        "role": "perc",
                        "type": "oneshot",
                        "beats": PATTERNS["syncopated_b"],
                        "gain": GAINS["perc"],
                    },
                ]
            }
        )

    # Bars 57-64: Kick + texture only (outro, doubled from 4)
    for _ in range(8):
        bars.append(
            {
                "layers": [
                    {
                        "role": "kick",
                        "type": "oneshot",
                        "beats": PATTERNS["four_on_floor"],
                        "gain": GAINS["kick"],
                    },
                    {"role": "texture", "type": "loop", "gain": GAINS["texture"]},
                ]
            }
        )

    assert len(bars) == 64, f"Expected 64 bars, got {len(bars)}"
    return bars


def render_bars_to_audio(bars, palette):
    """Render bar specs to a normalized AudioSegment (matches FileWriter behavior)."""
    samples = load_palette(palette, SAMPLE_RATE)
    segments = []
    for bar_spec in bars:
        audio = render_bar(bar_spec, samples, BPM, SAMPLE_RATE)
        segments.append(audio)
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    # Normalize to -1 dBFS, force 32-bit (match FileWriter behavior)
    change = -1.0 - combined.max_dBFS
    combined = combined.apply_gain(change)
    combined = combined.set_sample_width(4)
    return combined


def run_dj_experiment(palette_path, bars, outdir, label):
    """Run the DJ via subprocess, return path to output WAV."""
    wav_path = outdir / f"{label}.wav"
    log_path = outdir / f"{label}.log"

    cmd = [
        "uv",
        "run",
        "bergain",
        "dj",
        "--lm",
        "openrouter/openai/gpt-5-mini",
        "--critic-lm",
        "openrouter/openai/gpt-5-nano",
        "--palette",
        str(palette_path),
        "--bars",
        str(bars),
        "--bpm",
        str(BPM),
        "-o",
        str(wav_path),
        "--no-cache",
    ]

    print(f"  Running DJ ({label}): {' '.join(cmd)}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=600
        )

    if result.returncode != 0:
        print(f"  DJ run failed for {label}! See {log_path}")
        return None

    if not wav_path.exists():
        print(f"  No WAV produced for {label}! See {log_path}")
        return None

    return wav_path


def compute_objective(scores):
    """Compute weighted OBJ score."""
    return (
        0.60 * scores.get("CE", 0)
        + 0.05 * scores.get("CU", 0)
        + 0.05 * scores.get("PC", 0)
        + 0.30 * scores.get("PQ", 0)
    )


def main():
    parser = argparse.ArgumentParser(description="Score gap diagnostic")
    parser.add_argument("--palette", default=DEFAULT_PALETTE, help="Palette JSON file")
    parser.add_argument(
        "--skip-dj", action="store_true", help="Only run reference experiments A/B"
    )
    parser.add_argument(
        "--outdir", default=None, help="Output directory (default: auto-generated)"
    )
    args = parser.parse_args()

    palette_path = Path(args.palette)
    with open(palette_path) as f:
        palette = json.load(f)
    print(f"Palette: {palette_path}")
    print(f"  {json.dumps({r: p.split('/')[-1] for r, p in palette.items()})}")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"output/gap_test_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {outdir}\n")

    results = {}

    # --- Experiment A: Reference 32-bar ---
    print("=== Experiment A: Reference 32-bar ===")
    bars_32 = build_reference_bars()
    audio_a = render_bars_to_audio(bars_32, palette)
    wav_a = outdir / "A_ref_32bar.wav"
    audio_a.export(str(wav_a), format="wav")
    duration_a = len(audio_a) / 1000
    print(f"  Rendered {len(bars_32)} bars ({duration_a:.1f}s) -> {wav_a}")
    scores_a = score_audio(audio_a)
    if scores_a:
        obj_a = compute_objective(scores_a)
        results["A"] = {"scores": scores_a, "obj": obj_a, "bars": 32}
        print(
            f"  CE={scores_a['CE']:.2f} CU={scores_a['CU']:.2f} PC={scores_a['PC']:.2f} PQ={scores_a['PQ']:.2f} OBJ={obj_a:.2f}"
        )
    else:
        print("  SCORING FAILED")

    # --- Experiment B: Reference 64-bar ---
    print("\n=== Experiment B: Reference 64-bar ===")
    bars_64 = build_reference_bars_64()
    audio_b = render_bars_to_audio(bars_64, palette)
    wav_b = outdir / "B_ref_64bar.wav"
    audio_b.export(str(wav_b), format="wav")
    duration_b = len(audio_b) / 1000
    print(f"  Rendered {len(bars_64)} bars ({duration_b:.1f}s) -> {wav_b}")
    scores_b = score_audio(audio_b)
    if scores_b:
        obj_b = compute_objective(scores_b)
        results["B"] = {"scores": scores_b, "obj": obj_b, "bars": 64}
        print(
            f"  CE={scores_b['CE']:.2f} CU={scores_b['CU']:.2f} PC={scores_b['PC']:.2f} PQ={scores_b['PQ']:.2f} OBJ={obj_b:.2f}"
        )
    else:
        print("  SCORING FAILED")

    if not args.skip_dj:
        # --- Experiment C: DJ 32-bar ---
        print("\n=== Experiment C: DJ 32-bar ===")
        wav_c = run_dj_experiment(palette_path, 32, outdir, "C_dj_32bar")
        if wav_c:
            audio_c = AudioSegment.from_file(str(wav_c))
            scores_c = score_audio(audio_c)
            if scores_c:
                obj_c = compute_objective(scores_c)
                results["C"] = {"scores": scores_c, "obj": obj_c, "bars": 32}
                print(
                    f"  CE={scores_c['CE']:.2f} CU={scores_c['CU']:.2f} PC={scores_c['PC']:.2f} PQ={scores_c['PQ']:.2f} OBJ={obj_c:.2f}"
                )
            else:
                print("  SCORING FAILED")

        # --- Experiment D: DJ 64-bar ---
        print("\n=== Experiment D: DJ 64-bar ===")
        wav_d = run_dj_experiment(palette_path, 64, outdir, "D_dj_64bar")
        if wav_d:
            audio_d = AudioSegment.from_file(str(wav_d))
            scores_d = score_audio(audio_d)
            if scores_d:
                obj_d = compute_objective(scores_d)
                results["D"] = {"scores": scores_d, "obj": obj_d, "bars": 64}
                print(
                    f"  CE={scores_d['CE']:.2f} CU={scores_d['CU']:.2f} PC={scores_d['PC']:.2f} PQ={scores_d['PQ']:.2f} OBJ={obj_d:.2f}"
                )
            else:
                print("  SCORING FAILED")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Exp':<6} {'Bars':>4} {'CE':>6} {'CU':>6} {'PC':>6} {'PQ':>6} {'OBJ':>6}")
    print("-" * 42)
    for exp in ["A", "B", "C", "D"]:
        if exp in results:
            r = results[exp]
            s = r["scores"]
            print(
                f"{exp:<6} {r['bars']:>4} "
                f"{s['CE']:>6.2f} {s['CU']:>6.2f} {s['PC']:>6.2f} {s['PQ']:>6.2f} "
                f"{r['obj']:>6.2f}"
            )

    # --- Interpretation ---
    print("\n--- Interpretation ---")
    if "A" in results and "B" in results:
        delta_ab = results["B"]["obj"] - results["A"]["obj"]
        print(f"Duration effect (B-A): {delta_ab:+.2f} OBJ")
        if abs(delta_ab) < 0.15:
            print("  -> Duration doesn't matter much. Gap is in DJ decisions/arc.")
        elif delta_ab < -0.15:
            print("  -> Duration HURTS scores. Consider running DJ at 32 bars.")
        else:
            print("  -> Longer duration HELPS scores.")

    if "A" in results and "C" in results:
        delta_ac = results["C"]["obj"] - results["A"]["obj"]
        print(f"DJ quality at 32 bars (C-A): {delta_ac:+.2f} OBJ")
        if abs(delta_ac) < 0.15:
            print("  -> At 32 bars, DJ matches reference. Problem is arc pacing at 64.")
        else:
            print(
                "  -> DJ decisions hurt even at 32 bars. Need to constrain gains/patterns."
            )

    if "A" in results and "D" in results:
        delta_ad = results["D"]["obj"] - results["A"]["obj"]
        print(f"Full DJ gap (D-A): {delta_ad:+.2f} OBJ")

    if "B" in results and "D" in results:
        delta_bd = results["D"]["obj"] - results["B"]["obj"]
        print(f"DJ vs reference at 64 bars (D-B): {delta_bd:+.2f} OBJ")
        if abs(delta_bd) < 0.15:
            print("  -> At 64 bars, DJ matches reference. Gap is purely duration.")
        else:
            print(
                "  -> DJ arc/decisions are worse than reference even at same duration."
            )

    # Save results
    results_path = outdir / "gap_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
