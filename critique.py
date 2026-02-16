"""Critique: analyze the style of The Curling Scandal section by section.

Run oneshot.py first to build the arrangement in Ableton, then:
    CAPTURE_DIR=/path/to/wav/output uv run python critique.py

Requires File Recorder M4L on a track (CAPTURE_TRACK, CAPTURE_DEVICE env vars).
"""

import os
import sys
import time
import requests
from main import LiveAPI, capture_audio, analyze_audio

BPM = 90
BEAT_DUR = 60.0 / BPM

JUDGE_URL = os.environ.get(
    "BERGAIN_JUDGE_URL",
    "https://bergain-aesthetics--judge-score.modal.run",
)

# (name, start_beat, length_beats, expected_key)
SECTIONS = [
    ("The Rink", 0, 32, "D minor"),
    ("Suspicion", 32, 32, "D minor"),
    ("Evidence", 64, 32, "D minor/major"),
    ("The Fight Dance", 96, 96, "D minor → D major"),
    ("The Ice Remembers", 192, 32, "D major"),
]


def score_aesthetics(wav_path):
    with open(wav_path, "rb") as f:
        resp = requests.post(
            JUDGE_URL, files={"file": ("audio.wav", f, "audio/wav")}, timeout=120
        )
    resp.raise_for_status()
    return resp.json()["scores"]


def analyze(wav_path, analysis_type):
    return analyze_audio(wav_path, analysis_type, bpm=BPM)


def capture_sections(api):
    """Capture each section as a separate WAV. Returns list of (name, wav_path, section_info)."""
    captures = []
    for name, start_beat, length_beats, expected_key in SECTIONS:
        duration = length_beats * BEAT_DUR
        bars = length_beats // 4
        print(
            f"  Capturing '{name}' (beat {start_beat}, {bars} bars, {duration:.1f}s)..."
        )
        wav_path = capture_audio(api, start_time=float(start_beat), duration=duration)
        print(f"    → {os.path.basename(wav_path)}")
        captures.append((name, wav_path, length_beats, expected_key))
        time.sleep(0.5)
    return captures


def critique_section(wav_path, name, length_beats, expected_key):
    """Run all analyses on a single section."""
    r = {"name": name, "expected_key": expected_key, "bars": length_beats // 4}

    # Key
    key_data = analyze(wav_path, "key")
    r["key"] = f"{key_data['key']} {key_data['scale']}"
    r["key_conf"] = key_data["confidence"]

    # Energy
    energy_data = analyze(wav_path, "energy")
    frames = energy_data["frames"]
    rms = [f["rms"] for f in frames]
    flux = [f["flux"] for f in frames]
    centroid = [f["centroid"] for f in frames]
    r["energy_avg"] = sum(rms) / len(rms) if rms else 0
    r["energy_peak"] = max(rms) if rms else 0
    r["flux_avg"] = sum(flux) / len(flux) if flux else 0
    r["centroid_avg"] = sum(centroid) / len(centroid) if centroid else 0

    # Onsets
    onset_data = analyze(wav_path, "onsets")
    n_onsets = len(onset_data["onsets"])
    r["onset_density"] = n_onsets / length_beats if length_beats else 0

    # Aesthetics
    r["aesthetics"] = score_aesthetics(wav_path)

    return r


def print_report(results):
    baseline_energy = results[0]["energy_avg"] if results[0]["energy_avg"] > 0 else 1e-6

    print("\n" + "=" * 50)
    print("  CRITIQUE: The Curling Scandal")
    print("=" * 50)

    for r in results:
        ratio = r["energy_avg"] / baseline_energy
        bars = r["bars"]

        # Key match check
        detected = r["key"].lower()
        expected = r["expected_key"].lower()
        key_ok = any(part in detected for part in expected.replace("→", "/").split("/"))
        key_mark = "ok" if key_ok else "DRIFT"

        # Format aesthetics scores
        aes = r["aesthetics"]
        if isinstance(aes, dict):
            aes_line = "  ".join(f"{k}={v:.1f}" for k, v in aes.items())
        else:
            aes_line = str(aes)

        print(f"\n  {r['name']} ({bars} bars)")
        print(
            f"    Key:       {r['key']} (conf {r['key_conf']:.2f}) — expected: {r['expected_key']} [{key_mark}]"
        )
        print(
            f"    Energy:    avg {r['energy_avg']:.4f}  peak {r['energy_peak']:.4f}  ({ratio:.1f}x baseline)"
        )
        print(
            f"    Flux:      {r['flux_avg']:.2f}   Centroid: {r['centroid_avg']:.0f} Hz"
        )
        print(f"    Onsets:    {r['onset_density']:.1f}/beat")
        print(f"    Aesthetics: {aes_line}")

    # Summary
    print("\n" + "-" * 50)
    print("  SUMMARY")
    print("-" * 50)

    # Energy arc
    arc = "  ".join(
        f"{r['name'].split()[0]}({r['energy_avg'] / baseline_energy:.1f}x)"
        for r in results
    )
    print(f"  Energy arc: {arc}")

    # Key stability
    key_hits = 0
    for r in results:
        detected = r["key"].lower()
        expected = r["expected_key"].lower()
        if any(part in detected for part in expected.replace("→", "/").split("/")):
            key_hits += 1
    print(f"  Key stability: {key_hits}/{len(results)} sections match expected key")

    # Best/worst by aesthetics (use first numeric score found)
    def aes_score(r):
        a = r["aesthetics"]
        if isinstance(a, dict):
            return sum(a.values()) / len(a) if a else 0
        return float(a)

    best = max(results, key=aes_score)
    worst = min(results, key=aes_score)
    print(f"  Best section:  {best['name']} (aesthetics avg {aes_score(best):.1f})")
    print(f"  Weakest:       {worst['name']} (aesthetics avg {aes_score(worst):.1f})")

    # Centroid arc (brightness)
    bright_arc = "  ".join(
        f"{r['name'].split()[0]}({r['centroid_avg']:.0f}Hz)" for r in results
    )
    print(f"  Brightness:  {bright_arc}")


def main():
    if not os.environ.get("CAPTURE_DIR"):
        print("Set CAPTURE_DIR to where File Recorder saves WAVs.")
        print("  CAPTURE_DIR=/path/to/wavs uv run python critique.py")
        sys.exit(1)

    api = LiveAPI()
    api.song.call("stop_playing")
    time.sleep(0.2)

    print("=== Capturing sections ===")
    captures = capture_sections(api)

    print("\n=== Analyzing ===")
    results = []
    for name, wav_path, length_beats, expected_key in captures:
        print(f"  {name}...")
        r = critique_section(wav_path, name, length_beats, expected_key)
        results.append(r)

    print_report(results)

    api.stop()


if __name__ == "__main__":
    main()
