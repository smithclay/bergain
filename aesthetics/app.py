import io

import modal
from fastapi import UploadFile

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
