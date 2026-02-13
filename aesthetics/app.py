import io
import modal

app = modal.App("bergain-aesthetics")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchaudio", "audiobox_aesthetics", "python-multipart")
    .run_commands(
        "python -c 'from audiobox_aesthetics.infer import initialize_predictor; initialize_predictor()'"
    )
)


@app.cls(gpu="T4", container_idle_timeout=120, image=image)
class Judge:
    @modal.enter()
    def load_model(self):
        from audiobox_aesthetics.infer import initialize_predictor

        self.predictor = initialize_predictor()

    @modal.web_endpoint(method="POST")
    async def score(self, file: modal.asgi.UploadFile):
        import torchaudio

        data = await file.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))

        result = self.predictor.forward(
            [{"path": waveform, "sample_rate": sample_rate}]
        )

        return {"scores": result[0]}
