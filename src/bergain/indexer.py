"""
Sample pack indexer — scans audio files and extracts metadata via av.
"""

import os
import re
from pathlib import Path

import av

CATEGORY_MAP = {
    "909 + Twisted 909": "drums_909",
    "Claps": "clap",
    "DFAM-Bassline-Loops-128": "bassline",
    "Drum Loops": "drum_loop",
    "Echo-Pax-Claps-Rides": "clap_ride",
    "FX-Impacts": "fx",
    "HiHats": "hihat",
    "Kickdrums": "kick",
    "Noise-Hihat-Loops-133bpm": "noise_hat_loop",
    "Perx": "perc",
    "Synth": "synth",
    "Textures & Atmospheres & Pads": "texture",
}


def extract_bpm(filename: str) -> int | None:
    m = re.search(r"(\d{2,3})\s*bpm", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def is_loop(filename: str, category: str) -> bool:
    lower = filename.lower()
    if "loop" in lower or "bar" in lower:
        return True
    if category in ("bassline", "drum_loop", "noise_hat_loop"):
        return True
    return False


def get_audio_info(filepath: str) -> dict:
    try:
        with av.open(filepath) as container:
            stream = container.streams.audio[0]
            duration_s = (
                float(stream.duration * stream.time_base) if stream.duration else None
            )
            if duration_s is None and container.duration:
                duration_s = container.duration / av.time_base
            return {
                "duration_s": round(duration_s, 3) if duration_s else None,
                "sample_rate": stream.rate,
                "channels": stream.channels,
            }
    except Exception as e:
        return {
            "duration_s": None,
            "sample_rate": None,
            "channels": None,
            "error": str(e),
        }


def build_index(sample_dir: str) -> list[dict]:
    samples = []
    base = Path(sample_dir)

    for folder in sorted(os.listdir(base)):
        folder_path = base / folder
        if not folder_path.is_dir():
            continue

        category = CATEGORY_MAP.get(folder, folder)

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".wav", ".mp3", ".aif", ".flac")):
                continue

            filepath = str(folder_path / fname)
            rel_path = os.path.join(sample_dir, folder, fname)

            bpm = extract_bpm(fname)
            if bpm is None:
                bpm = extract_bpm(folder)

            info = get_audio_info(filepath)

            sub_type = None
            if category == "drums_909":
                lower = fname.lower()
                if "bassdrum" in lower or "kick" in lower:
                    sub_type = "kick"
                elif "snare" in lower:
                    sub_type = "snare"
                elif "clap" in lower:
                    sub_type = "clap"
                elif "hihat" in lower or "hh" in lower:
                    sub_type = "hihat"
                elif "ride" in lower:
                    sub_type = "ride"
                elif "tom" in lower:
                    sub_type = "tom"
                elif "crash" in lower:
                    sub_type = "crash"
                elif "shaker" in lower:
                    sub_type = "shaker"

            samples.append(
                {
                    "path": rel_path,
                    "filename": fname,
                    "category": category,
                    "sub_type": sub_type,
                    "is_loop": is_loop(fname, category),
                    "bpm": bpm,
                    **info,
                }
            )

    return samples
