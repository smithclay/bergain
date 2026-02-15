#!/usr/bin/env python3
"""Screen palettes by rendering a fixed reference arrangement and scoring via aesthetics endpoint.

Generates random palettes with smart filtering (loops for bassline/texture,
oneshots for kick/hihat/perc/clap), renders a 32-bar reference arrangement,
and scores each via the Modal aesthetics endpoint.

Usage:
    uv run python scripts/screen_palettes.py --count 100 --top 10
    uv run python scripts/screen_palettes.py --count 20 --top 5 --seed 42
"""

import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bergain.dj import DEFAULT_ROLES, _build_role_map
from bergain.indexer import build_index
from bergain.renderer import load_palette, render_bar

SCORER_URL = "https://smithclay--bergain-aesthetics-judge-score.modal.run"
BPM = 128
SAMPLE_RATE = 44100

# Reference gains — midpoints of v8b tight ranges
GAINS = {
    "kick": 0.92,
    "hihat": 0.53,
    "bassline": 0.65,
    "perc": 0.40,
    "texture": 0.35,
    "clap": 0.47,
}

PATTERNS = {
    "four_on_floor": [0, 1, 2, 3],
    "offbeat_8ths": [0.5, 1.5, 2.5, 3.5],
    "syncopated_a": [0.5, 2.5],
    "syncopated_b": [1.5, 3.5],
    "backbeat": [1, 3],
    "gallop": [0, 0.5, 2, 2.5],
    "sixteenth_drive": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
}


def build_reference_bars():
    """Build 32 bar specs for the reference arrangement."""
    bars = []

    # Bars 1-4: Kick only
    for _ in range(4):
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

    # Bars 5-8: + hihat
    for _ in range(4):
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

    # Bars 9-12: + bassline (loop)
    for _ in range(4):
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

    # Bars 13-16: + perc (syncopated)
    for _ in range(4):
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

    # Bars 17-24: Full groove — all 6 roles, evolving patterns
    # Bars 17-20: standard full groove
    for _ in range(4):
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
    # Bars 21-24: evolve — gallop perc, sixteenth hihat
    for _ in range(4):
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

    # Bars 25-28: Strip to kick + hihat + bassline + perc
    for _ in range(4):
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

    # Bars 29-32: Kick + texture only (outro)
    for _ in range(4):
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

    return bars


def pick_smart_palette(role_map):
    """Pick a palette with smart filtering: loops for bassline/texture, oneshots for the rest."""
    loop_roles = {"bassline", "texture"}
    palette = {}
    for role in DEFAULT_ROLES:
        candidates = role_map.get(role, [])
        if not candidates:
            continue
        if role in loop_roles:
            filtered = [s for s in candidates if s.get("loop", False)]
        else:
            filtered = [s for s in candidates if not s.get("loop", False)]
        if not filtered:
            filtered = candidates
        palette[role] = random.choice(filtered)["path"]
    return palette


def render_reference(palette, reference_bars):
    """Render the reference arrangement for a palette. Returns AudioSegment."""
    samples = load_palette(palette, SAMPLE_RATE)
    segments = []
    for bar_spec in reference_bars:
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


def score_audio(audio):
    """POST WAV to Modal aesthetics endpoint. Returns scores dict or None."""
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    wav_bytes = buf.getvalue()

    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
            f"Content-Type: audio/wav\r\n"
            f"\r\n"
        ).encode()
        + wav_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        SCORER_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("scores", data)
    except Exception as e:
        print(f"  Scoring error: {e}")
        return None


def evaluate_palette(palette, reference_bars, idx, total):
    """Render + score a single palette. Returns (palette, scores, objective)."""
    label = palette.get("kick", "?").split("/")[-1][:20]
    print(f"  [{idx + 1}/{total}] Rendering... {label}")

    try:
        audio = render_reference(palette, reference_bars)
    except Exception as e:
        print(f"  [{idx + 1}/{total}] Render failed: {e}")
        return palette, None, -1.0

    scores = score_audio(audio)
    if scores is None:
        return palette, None, -1.0

    objective = (
        0.60 * scores.get("CE", 0)
        + 0.05 * scores.get("CU", 0)
        + 0.05 * scores.get("PC", 0)
        + 0.30 * scores.get("PQ", 0)
    )

    print(
        f"  [{idx + 1}/{total}] "
        f"CE={scores.get('CE', 0):.2f} CU={scores.get('CU', 0):.2f} "
        f"PC={scores.get('PC', 0):.2f} PQ={scores.get('PQ', 0):.2f} "
        f"OBJ={objective:.2f}"
    )
    return palette, scores, objective


def main():
    parser = argparse.ArgumentParser(description="Screen palettes for quality")
    parser.add_argument(
        "--count", type=int, default=100, help="Number of palettes to screen"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top palettes to save"
    )
    parser.add_argument(
        "--sample-dir", default="sample_pack", help="Sample pack directory"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent scoring threads"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Indexing samples from {args.sample_dir}...")
    index = build_index(args.sample_dir)
    role_map = _build_role_map(index)
    print(f"Indexed {len(index)} samples into {len(role_map)} roles:")
    for role, samples in role_map.items():
        loops = sum(1 for s in samples if s.get("loop", False))
        oneshots = len(samples) - loops
        print(f"  {role}: {len(samples)} ({loops} loops, {oneshots} oneshots)")

    print(f"\nGenerating {args.count} smart palettes...")
    palettes = []
    seen = set()
    attempts = 0
    while len(palettes) < args.count and attempts < args.count * 10:
        p = pick_smart_palette(role_map)
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            palettes.append(p)
        attempts += 1
    print(f"Generated {len(palettes)} unique palettes")

    reference_bars = build_reference_bars()
    print(f"Reference arrangement: {len(reference_bars)} bars")

    print(f"\nScreening with {args.workers} workers...")
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_palette, p, reference_bars, i, len(palettes)): i
            for i, p in enumerate(palettes)
        }
        for future in as_completed(futures):
            palette, scores, objective = future.result()
            if scores is not None:
                results.append(
                    {"palette": palette, "scores": scores, "objective": objective}
                )

    elapsed = time.time() - start_time
    print(f"\nScreened {len(results)}/{len(palettes)} palettes in {elapsed:.1f}s")

    if not results:
        print("No successful scores. Check the aesthetics endpoint.")
        return

    results.sort(key=lambda r: r["objective"], reverse=True)

    # Save results CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"palettes/screening_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "objective", "CE", "CU", "PC", "PQ"] + DEFAULT_ROLES)
        for i, r in enumerate(results):
            writer.writerow(
                [
                    i + 1,
                    f"{r['objective']:.4f}",
                    f"{r['scores']['CE']:.4f}",
                    f"{r['scores']['CU']:.4f}",
                    f"{r['scores']['PC']:.4f}",
                    f"{r['scores']['PQ']:.4f}",
                ]
                + [r["palette"].get(role, "") for role in DEFAULT_ROLES]
            )
    print(f"Results saved to {csv_path}")

    # Save top palettes
    curated_dir = Path("palettes/curated")
    curated_dir.mkdir(parents=True, exist_ok=True)

    top_n = min(args.top, len(results))
    print(f"\nTop {top_n} palettes:")
    for i, r in enumerate(results[:top_n]):
        palette_path = curated_dir / f"palette_{i + 1:03d}.json"
        with open(palette_path, "w") as f:
            json.dump(r["palette"], f, indent=2)
        print(
            f"  #{i + 1}: OBJ={r['objective']:.2f} "
            f"CE={r['scores']['CE']:.2f} PQ={r['scores']['PQ']:.2f} "
            f"-> {palette_path}"
        )

    # Summary stats
    objectives = [r["objective"] for r in results]
    median = sorted(objectives)[len(objectives) // 2]
    print(
        f"\nObjective stats: "
        f"min={min(objectives):.2f} median={median:.2f} max={max(objectives):.2f}"
    )


if __name__ == "__main__":
    main()
