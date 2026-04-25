"""
Speech-to-Text Engine — faster-whisper (local, free, no API key)
Pre-loads model once at startup for near-zero per-call latency.
"""

import os
import io
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

WHISPER_MODEL  = os.getenv("WHISPER_MODEL",  "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
SAMPLE_RATE    = 16000   # Whisper expects 16kHz
SILENCE_THRESH = 0.01    # RMS threshold to detect end of speech
SILENCE_SECS   = 1.5     # seconds of silence before cut-off
CHUNK_SECS     = 0.1     # audio chunk size in seconds


class STTEngine:
    """
    Wraps faster-whisper.
    Model is loaded once at construction — all transcribe() calls are fast.
    """

    def __init__(self):
        self._model: WhisperModel | None = None

    def load(self):
        """Load Whisper model into memory. Call once at app startup."""
        if self._model is None:
            from src.utils.terminal_display import display
            display.info(f"Loading Whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type="int8",    # fastest on CPU
            )
            display.info("Whisper model loaded and ready.")
        return self

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file. Returns text string."""
        self._ensure_loaded()
        segments, _ = self._model.transcribe(
            audio_path,
            beam_size=1,            # fastest inference
            vad_filter=True,        # skip silence automatically
            language="en",
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_array(self, audio_np: np.ndarray) -> str:
        """Transcribe a numpy float32 audio array at 16kHz."""
        self._ensure_loaded()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        wav.write(tmp_path, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
        text = self.transcribe_file(tmp_path)
        os.unlink(tmp_path)
        return text

    def record_until_silence(self, prompt: str = "") -> str:
        """
        Record microphone input until silence detected, then transcribe.
        Blocks until patient stops speaking.
        Returns transcribed text.
        """
        from src.utils.terminal_display import display

        if prompt:
            display.patient_listening()

        audio_chunks: list[np.ndarray] = []
        silence_counter = 0
        chunk_samples   = int(SAMPLE_RATE * CHUNK_SECS)
        silence_chunks  = int(SILENCE_SECS / CHUNK_SECS)
        recording_started = False

        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time, status):
            audio_q.put(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=callback,
        ):
            while True:
                chunk = audio_q.get()
                rms   = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > SILENCE_THRESH:
                    recording_started = True
                    silence_counter = 0
                    audio_chunks.append(chunk)
                elif recording_started:
                    audio_chunks.append(chunk)
                    silence_counter += 1
                    if silence_counter >= silence_chunks:
                        break

        if not audio_chunks:
            return ""

        audio_data = np.concatenate(audio_chunks, axis=0).flatten()
        return self.transcribe_array(audio_data)

    def _ensure_loaded(self):
        if self._model is None:
            self.load()


# Module-level singleton — shared across all agents
stt = STTEngine()
