"""
Real-time audio streamer using sounddevice.

Consumes pydub AudioSegments from a queue and plays them
through a sounddevice OutputStream. Blocking enqueue provides
backpressure to the RLM.
"""

import queue
import threading

import numpy as np
import sounddevice as sd
from pydub import AudioSegment


def audiosegment_to_numpy(segment: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a float32 numpy array in [-1, 1]."""
    raw = np.array(segment.get_array_of_samples(), dtype=np.float32)
    max_val = float(1 << (segment.sample_width * 8 - 1))
    raw /= max_val
    if segment.channels > 1:
        raw = raw.reshape(-1, segment.channels)
    else:
        raw = raw.reshape(-1, 1)
    return raw


class AudioStreamer:
    """Threaded audio streamer with blocking enqueue for backpressure."""

    def __init__(self, sample_rate: int = 44100, queue_maxsize: int = 64):
        self.sample_rate = sample_rate
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None
        self.buffer_bars = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=0,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._consumer, daemon=True)
        self._thread.start()

    def _consumer(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._stream.write(chunk)
            self.buffer_bars = max(0, self.buffer_bars - 1)

    def enqueue(self, audio: AudioSegment) -> None:
        """Convert AudioSegment to numpy and enqueue. Blocks when queue is full."""
        data = audiosegment_to_numpy(audio)
        while not self._stop_event.is_set():
            try:
                self.audio_queue.put(data, timeout=0.5)
                self.buffer_bars += 1
                return
            except queue.Full:
                continue

    def stop(self) -> None:
        """Graceful shutdown: drain queue then stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def abort(self) -> None:
        """Immediate shutdown: clear queue and stop."""
        self._stop_event.set()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        if self._stream:
            self._stream.abort()
            self._stream.close()
            self._stream = None
        if self._thread:
            self._thread.join(timeout=2)


class FileWriter:
    """Accumulates AudioSegments and writes to a WAV file on stop().

    Duck-types with AudioStreamer so the DJ can use either backend.
    No backpressure — renders as fast as possible.
    """

    def __init__(self, output_path: str, sample_rate: int = 44100):
        self.output_path = output_path
        self.sample_rate = sample_rate
        self._segments: list[AudioSegment] = []
        self.buffer_bars = 0

    class _FakeQueue:
        maxsize = 0

    audio_queue = _FakeQueue()

    def start(self) -> None:
        pass

    def enqueue(self, audio: AudioSegment) -> None:
        self._segments.append(audio)
        self.buffer_bars += 1

    def stop(self) -> None:
        if not self._segments:
            return
        from pathlib import Path

        combined = self._segments[0]
        for seg in self._segments[1:]:
            combined += seg
        # Normalize to -1 dBFS
        change = -1.0 - combined.max_dBFS
        combined = combined.apply_gain(change)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.export(self.output_path, format="wav")
        total_s = len(combined) / 1000
        print(f"Wrote {total_s:.1f}s ({total_s / 60:.1f} min) -> {self.output_path}")

    def abort(self) -> None:
        self.stop()
