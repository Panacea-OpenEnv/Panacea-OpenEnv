"""
Speech-to-Text Engine — faster-whisper (local, free, no API key)
Pre-loads model once at startup for near-zero per-call latency.

Auto-calibration: Records 1 second of ambient noise at startup and sets
the silence threshold to 3× the ambient RMS, ensuring it works with any
microphone regardless of gain level.
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

load_dotenv()

WHISPER_MODEL   = os.getenv("WHISPER_MODEL",  "small")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")
SAMPLE_RATE     = 16000   # Whisper expects 16kHz
SILENCE_SECS    = 2.0     # seconds of silence before cut-off
CHUNK_SECS      = 0.1     # audio chunk size in seconds
MAX_RECORD_SECS = 30.0    # Max total recording time
WAIT_START_SECS = 15.0    # Max wait time before aborting if no speech

# Will be set by auto-calibration; fallback if calibration skipped
_SILENCE_THRESH = float(os.getenv("SILENCE_THRESH", "0.01"))


class STTEngine:
    """
    Wraps faster-whisper.
    Model is loaded once at construction — all transcribe() calls are fast.
    """

    def __init__(self):
        self._model = None
        self._calibrated_thresh: float | None = None

    def load(self):
        """Load Whisper model into memory. Call once at app startup."""
        if self._model is None:
            from faster_whisper import WhisperModel
            from src.utils.terminal_display import display
            display.info(f"Loading Whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type="int8",    # fastest on CPU
            )
            display.info("Whisper model loaded and ready.")

            # Auto-calibrate microphone
            self._calibrate_mic(display)
        return self

    def _calibrate_mic(self, display):
        """Record 1 second of ambient noise to auto-set the silence threshold."""
        display.info("Calibrating microphone... please stay quiet for 1 second.")
        
        chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
        rms_values = []
        calibration_chunks = int(1.0 / CHUNK_SECS)  # 1 second worth
        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time, status):
            audio_q.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                callback=callback,
            ):
                for _ in range(calibration_chunks):
                    chunk = audio_q.get(timeout=2.0)
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    rms_values.append(rms)

            if rms_values:
                ambient_rms = float(np.mean(rms_values))
                # Set threshold to 3× ambient noise floor (minimum 0.005)
                self._calibrated_thresh = max(ambient_rms * 3.0, 0.005)
                display.info(f"Mic calibrated: ambient={ambient_rms:.5f}, "
                           f"threshold={self._calibrated_thresh:.5f}")
            else:
                self._calibrated_thresh = _SILENCE_THRESH
                display.info(f"Calibration: no data, using default threshold={_SILENCE_THRESH}")

        except Exception as e:
            display.error("MIC_CALIBRATE", str(e))
            self._calibrated_thresh = _SILENCE_THRESH

    @property
    def silence_thresh(self) -> float:
        return self._calibrated_thresh if self._calibrated_thresh else _SILENCE_THRESH

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

    def flush_input_buffer(self, seconds: float = 1.2):
        """
        Drain the microphone input buffer after TTS stops.
        Prevents Whisper from transcribing leftover TTS speaker audio.
        """
        import time
        chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=chunk_samples, callback=callback,
        ):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass

    def record_until_silence(self, prompt: str = "") -> str:
        """
        Record microphone input until silence detected, then transcribe.
        Blocks until patient stops speaking.
        Returns transcribed text.
        """
        from src.utils.terminal_display import display

        if prompt:
            display.patient_listening()

        thresh = self.silence_thresh
        audio_chunks: list[np.ndarray] = []
        silence_counter = 0
        wait_counter    = 0
        total_chunks    = 0
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

                if rms > thresh:
                    if not recording_started:
                        display.info("Speech detected — recording...")
                    recording_started = True
                    silence_counter = 0
                    audio_chunks.append(chunk)
                else:
                    if recording_started:
                        audio_chunks.append(chunk)
                        silence_counter += 1
                        if silence_counter >= silence_chunks:
                            display.info("Silence detected — processing...")
                            break
                    else:
                        wait_counter += 1
                        if wait_counter >= int(WAIT_START_SECS / CHUNK_SECS):
                            display.info("No speech detected — timed out.")
                            break
                
                total_chunks += 1
                if total_chunks >= int(MAX_RECORD_SECS / CHUNK_SECS):
                    display.info("Max recording time reached — processing...")
                    break

        if not audio_chunks:
            return ""

        duration = len(audio_chunks) * CHUNK_SECS
        display.info(f"Recorded {duration:.1f}s of audio. Transcribing...")

        audio_data = np.concatenate(audio_chunks, axis=0).flatten()
        text = self.transcribe_array(audio_data)
        
        if text:
            display.info(f"Transcription: \"{text}\"")
        else:
            display.info("Whisper returned empty transcription.")
        
        return text

    def _ensure_loaded(self):
        if self._model is None:
            self.load()


# Module-level singleton — shared across all agents
stt = STTEngine()
