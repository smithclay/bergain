import io
from typing import Optional

import modal

try:
    from fastapi import UploadFile, File, Form
except ImportError:
    # Stubs so module loads locally without fastapi; Modal containers have it
    from typing import Any

    def File(default: Any = ...):
        return default  # type: ignore[misc]

    def Form(default: Any = ...):
        return default  # type: ignore[misc]

    class UploadFile:
        pass  # type: ignore[no-redef]


app = modal.App("bergain-aesthetics")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "soundfile",
        "audiobox_aesthetics",
        "python-multipart",
        "requests",
        "fastapi[standard]",
    )
    .run_commands(
        "python -c 'from audiobox_aesthetics.infer import initialize_predictor; initialize_predictor()'"
    )
)


@app.cls(gpu="T4", scaledown_window=120, image=image)
class Judge:
    @modal.enter()
    def load_model(self):
        from audiobox_aesthetics.infer import initialize_predictor

        self.predictor = initialize_predictor()

    @modal.fastapi_endpoint(method="POST")
    async def score(self, file: UploadFile):
        import soundfile as sf
        import torch

        data = await file.read()
        audio_np, sample_rate = sf.read(io.BytesIO(data))
        waveform = torch.from_numpy(audio_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        result = self.predictor.forward(
            [{"path": waveform, "sample_rate": sample_rate}]
        )

        return {"scores": result[0]}


# ---------------------------------------------------------------------------
# Audio analysis image (librosa-based, no torch needed)
# ---------------------------------------------------------------------------

analysis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "librosa",
        "numpy",
        "soundfile",
        "python-multipart",
        "fastapi[standard]",
    )
)

# ---------------------------------------------------------------------------
# Structure analysis image (allin1 — torch + demucs + natten)
# ---------------------------------------------------------------------------

structure_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install("torch==2.1.0", "torchaudio==2.1.0")
    .pip_install("natten==0.17.1")
    .run_commands("pip install git+https://github.com/CPJKU/madmom")
    .pip_install(
        "allin1", "demucs", "soundfile", "python-multipart", "fastapi[standard]"
    )
    # Pre-download demucs model during image build
    .run_commands(
        "python -c 'from demucs.pretrained import get_model; get_model(\"htdemucs\")'"
    )
)


@app.cls(gpu="T4", scaledown_window=120, image=structure_image)
class StructureAnalyzer:
    @modal.method()
    def analyze(self, audio_data: bytes) -> dict:
        import allin1
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            result = allin1.analyze(tmp_path, device="cuda")
            beats = result.beats
            downbeats = result.downbeats
            return {
                "bpm": result.bpm,
                "beats": beats.tolist() if hasattr(beats, "tolist") else list(beats),
                "downbeats": downbeats.tolist()
                if hasattr(downbeats, "tolist")
                else list(downbeats),
                "segments": [
                    {"start": s.start, "end": s.end, "label": s.label}
                    for s in result.segments
                ],
            }
        finally:
            os.unlink(tmp_path)


@app.cls(scaledown_window=120, image=analysis_image)
class Analyzer:
    @modal.fastapi_endpoint(method="POST")
    async def analyze(
        self,
        file: UploadFile = File(...),
        analysis_type: str = Form(...),
        bpm: Optional[float] = Form(None),
        target_bpm: Optional[float] = Form(None),
        file2: Optional[UploadFile] = File(None),
    ):
        import numpy as np
        import soundfile as sf

        data = await file.read()
        y, sr = sf.read(io.BytesIO(data))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)

        if analysis_type == "key":
            return self._detect_key(y, sr)
        elif analysis_type == "structure":
            return await self._analyze_structure(data, bpm)
        elif analysis_type == "energy":
            return self._analyze_energy(y, sr, bpm)
        elif analysis_type == "onsets":
            return self._analyze_onsets(y, sr, bpm)
        elif analysis_type == "frequency_clash":
            if file2 is None:
                return {"error": "frequency_clash requires file2"}
            data2 = await file2.read()
            y2, sr2 = sf.read(io.BytesIO(data2))
            if y2.ndim > 1:
                y2 = np.mean(y2, axis=1)
            y2 = y2.astype(np.float32)
            return self._analyze_frequency_clash(y, sr, y2, sr2)
        else:
            return {"error": f"Unknown analysis_type: {analysis_type}"}

    def _detect_key(self, y, sr):
        import librosa
        import numpy as np

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.mean(chroma, axis=1)

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        all_keys = []
        best_corr = -1.0
        best_key = "C"
        best_scale = "major"

        for i in range(12):
            rotated = np.roll(chroma_vals, -i)
            major_corr = float(np.corrcoef(rotated, major_profile)[0, 1])
            minor_corr = float(np.corrcoef(rotated, minor_profile)[0, 1])

            all_keys.append(
                {
                    "key": note_names[i],
                    "scale": "major",
                    "confidence": round(major_corr, 4),
                }
            )
            all_keys.append(
                {
                    "key": note_names[i],
                    "scale": "minor",
                    "confidence": round(minor_corr, 4),
                }
            )

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = note_names[i]
                best_scale = "major"
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = note_names[i]
                best_scale = "minor"

        all_keys.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "key": best_key,
            "scale": best_scale,
            "confidence": round(best_corr, 4),
            "all_keys": all_keys[:6],
        }

    async def _analyze_structure(self, audio_data: bytes, bpm: Optional[float]):
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: StructureAnalyzer().analyze.remote(audio_data)
        )

    def _analyze_energy(self, y, sr, bpm: Optional[float]):
        import librosa

        detected_bpm = bpm
        if detected_bpm is None:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            detected_bpm = (
                float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])
            )

        # 16th note hop
        sixteenth_dur = 60.0 / detected_bpm / 4
        hop_length = max(1, int(sixteenth_dur * sr))

        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[
            0
        ]

        n_frames = min(len(rms), len(flux), len(centroid))
        frames = []
        for i in range(n_frames):
            frames.append(
                {
                    "rms": round(float(rms[i]), 6),
                    "flux": round(float(flux[i]), 6),
                    "centroid": round(float(centroid[i]), 2),
                }
            )

        return {"bpm": round(detected_bpm, 2), "frames": frames}

    def _analyze_onsets(self, y, sr, bpm: Optional[float]):
        import librosa

        detected_bpm = bpm
        if detected_bpm is None:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            detected_bpm = (
                float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])
            )

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strengths = librosa.onset.onset_strength(y=y, sr=sr)

        onsets = []
        for frame, time_val in zip(onset_frames, onset_times):
            strength = (
                float(onset_strengths[frame]) if frame < len(onset_strengths) else 0.0
            )
            onsets.append(
                {
                    "time": round(float(time_val), 4),
                    "strength": round(strength, 4),
                }
            )

        return {"bpm": round(detected_bpm, 2), "onsets": onsets}

    def _analyze_frequency_clash(self, y1, sr1, y2, sr2):
        import librosa
        import numpy as np

        # Resample to common rate if needed
        target_sr = max(sr1, sr2)
        if sr1 != target_sr:
            y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)

        # Compute spectrograms
        S1 = np.abs(librosa.stft(y1))
        S2 = np.abs(librosa.stft(y2))

        # Trim to same length
        min_frames = min(S1.shape[1], S2.shape[1])
        S1 = S1[:, :min_frames]
        S2 = S2[:, :min_frames]

        freqs = librosa.fft_frequencies(sr=target_sr)

        # Define frequency bands
        band_defs = [
            ("sub", 20, 60),
            ("bass", 60, 250),
            ("low_mid", 250, 500),
            ("mid", 500, 2000),
            ("upper_mid", 2000, 4000),
            ("high", 4000, 20000),
        ]

        bands = []
        for name, lo, hi in band_defs:
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                bands.append(
                    {"name": name, "clash_score": 0.0, "energy_1": 0.0, "energy_2": 0.0}
                )
                continue

            e1 = float(np.mean(S1[mask, :]))
            e2 = float(np.mean(S2[mask, :]))

            # Clash = geometric mean of energies (high when both are loud in same band)
            clash = float(np.sqrt(e1 * e2))
            bands.append(
                {
                    "name": name,
                    "clash_score": round(clash, 6),
                    "energy_1": round(e1, 6),
                    "energy_2": round(e2, 6),
                }
            )

        return {"bands": bands}
