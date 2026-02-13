"""
Renders an arrangement artifact to audio via pydub, and plays it.
"""

import math
import subprocess
from pathlib import Path

from pydub import AudioSegment


def _gain_to_db(gain: float) -> float:
    if gain <= 0:
        return -120
    return 20 * math.log10(gain)


def _loop_to_length(audio: AudioSegment, target_ms: int) -> AudioSegment:
    if len(audio) == 0:
        return AudioSegment.silent(duration=target_ms)
    result = audio
    while len(result) < target_ms:
        result += audio
    return result[:target_ms]


def _load_sample(path: str, sample_rate: int) -> AudioSegment:
    seg = AudioSegment.from_file(path)
    if seg.frame_rate != sample_rate:
        seg = seg.set_frame_rate(sample_rate)
    # convert to mono for consistent mixing
    if seg.channels > 1:
        seg = seg.set_channels(1)
    return seg


def load_palette(palette: dict[str, str], sample_rate: int) -> dict[str, AudioSegment]:
    """Load all palette samples into memory. Returns role -> AudioSegment."""
    return {role: _load_sample(path, sample_rate) for role, path in palette.items()}


def render_bar(
    bar_spec: dict,
    samples: dict[str, AudioSegment],
    bpm: float,
    sample_rate: int,
) -> AudioSegment:
    """Render a single bar from a bar spec. Returns an AudioSegment of 1 bar (4 beats).

    bar_spec: {"layers": [{"role": "kick", "type": "oneshot", "beats": [0, 2], "gain": 0.9}, ...]}
    """
    beat_ms = 60_000 / bpm
    bar_ms = int(beat_ms * 4)
    bar_audio = AudioSegment.silent(duration=bar_ms, frame_rate=sample_rate)

    for layer in bar_spec.get("layers", []):
        role = layer["role"]
        if role not in samples:
            continue
        sample = samples[role]
        gain_db = _gain_to_db(layer.get("gain", 1.0))
        adjusted = sample + gain_db

        if layer["type"] == "loop":
            looped = _loop_to_length(adjusted, bar_ms)
            bar_audio = bar_audio.overlay(looped)
        elif layer["type"] == "oneshot":
            for beat in layer.get("beats", [0]):
                pos_ms = int(beat * beat_ms)
                if pos_ms < bar_ms:
                    bar_audio = bar_audio.overlay(adjusted, position=pos_ms)

    return bar_audio


def render(arrangement: dict, output_wav_path: str) -> None:
    bpm = arrangement["bpm"]
    sample_rate = arrangement["sample_rate"]
    palette = arrangement["palette"]

    beat_ms = 60_000 / bpm
    bar_ms = beat_ms * 4

    # Load all samples once
    samples: dict[str, AudioSegment] = {}
    for role, path in palette.items():
        samples[role] = _load_sample(path, sample_rate)

    # Render each section
    rendered_sections: list[AudioSegment] = []

    for section in arrangement["sections"]:
        num_bars = section["bars"]
        section_ms = int(num_bars * bar_ms)
        section_audio = AudioSegment.silent(duration=section_ms, frame_rate=sample_rate)

        for layer in section["layers"]:
            role = layer["role"]
            sample = samples[role]
            gain_db = _gain_to_db(layer.get("gain", 1.0))
            adjusted = sample + gain_db

            start_bar = layer.get("start_bar", 0)
            end_bar = layer.get("end_bar", num_bars)

            if layer["type"] == "loop":
                active_ms = int((end_bar - start_bar) * bar_ms)
                looped = _loop_to_length(adjusted, active_ms)
                pos_ms = int(start_bar * bar_ms)
                section_audio = section_audio.overlay(looped, position=pos_ms)

            elif layer["type"] == "oneshot":
                beats = layer.get("beats", [0])
                for bar in range(start_bar, end_bar):
                    for beat in beats:
                        pos_ms = int(bar * bar_ms + beat * beat_ms)
                        if pos_ms < section_ms:
                            section_audio = section_audio.overlay(
                                adjusted, position=pos_ms
                            )

        # Apply fades (specified in bars)
        fade_in_bars = section.get("fade_in")
        if fade_in_bars:
            section_audio = section_audio.fade_in(int(fade_in_bars * bar_ms))

        fade_out_bars = section.get("fade_out")
        if fade_out_bars:
            section_audio = section_audio.fade_out(int(fade_out_bars * bar_ms))

        rendered_sections.append(section_audio)

    # Concatenate all sections
    song = rendered_sections[0]
    for sec in rendered_sections[1:]:
        song += sec

    # Normalize to -1 dBFS
    change = -1.0 - song.max_dBFS
    song = song.apply_gain(change)

    # Export
    out = Path(output_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    song.export(str(out), format="wav")

    total_s = len(song) / 1000
    print(f"Rendered {total_s:.1f}s ({total_s / 60:.1f} min) -> {out}")


def play(wav_path: str) -> None:
    print(f"Playing {wav_path} ...")
    subprocess.run(["afplay", wav_path], check=True)
