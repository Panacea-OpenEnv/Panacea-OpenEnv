"""
Text-to-Speech Engine — pyttsx3 (local, free, no API key)
Runs in a background thread so speaking never blocks the main async loop.
Sentence-buffered: starts speaking each sentence as GPT-4o streams it,
giving near-zero perceived latency.
"""

import os
import re
import sys
import queue
import threading
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

TTS_RATE  = int(os.getenv("TTS_RATE",  "150"))
TTS_VOICE = os.getenv("TTS_VOICE", "female").lower()

# Split on sentence-ending punctuation followed by whitespace, OR on newlines
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|\n+')


class TTSEngine:
    """
    Wraps pyttsx3 in a background thread.
    Main thread calls speak() or stream_token() — never blocks.
    """

    def __init__(self):
        self._engine:  pyttsx3.Engine | None = None
        self._queue:   queue.Queue           = queue.Queue()
        self._thread:  threading.Thread | None = None
        self._buffer:  str = ""             # partial sentence buffer for streaming
        self._running: bool = False

    def start(self):
        """Start the TTS background thread. Call once at app startup."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def speak(self, text: str):
        """Enqueue text to be spoken. Returns immediately (non-blocking)."""
        if text and text.strip():
            self._queue.put(text.strip())

    def stream_token(self, token: str):
        """
        Called per GPT-4o streaming token.
        Buffers tokens until a sentence boundary (or newline), then speaks.
        This gives near-zero perceived latency — doctor starts speaking
        before GPT-4o has finished generating the full response.
        """
        self._buffer += token
        sentences = _SENTENCE_END.split(self._buffer)
        # All complete sentences except the last (possibly incomplete) one
        for sentence in sentences[:-1]:
            if sentence.strip():
                self.speak(sentence.strip())
        self._buffer = sentences[-1]  # keep the incomplete tail

    def flush_buffer(self):
        """Speak any remaining text in the buffer (end of turn)."""
        if self._buffer.strip():
            self.speak(self._buffer.strip())
            self._buffer = ""

    def wait_until_done(self):
        """Block until the speech queue is fully drained."""
        self._queue.join()

    def stop(self):
        self._running = False
        self._queue.put(None)   # sentinel to unblock worker
        if self._thread:
            self._thread.join(timeout=3)

    # ── Background worker ─────────────────────────────────────────────────────

    def _worker(self):
        """Runs in background thread. Speaks items from queue one by one."""
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)

        # Select voice by gender preference
        voices = engine.getProperty("voices")
        selected = None
        for v in voices:
            if TTS_VOICE in v.name.lower() or TTS_VOICE in v.id.lower():
                selected = v.id
                break
        if not selected and voices:
            # fallback: female=index 1 if available, else index 0
            idx = 1 if TTS_VOICE == "female" and len(voices) > 1 else 0
            selected = voices[idx].id
        if selected:
            engine.setProperty("voice", selected)

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
                if text is None:
                    self._queue.task_done()
                    break
                engine.say(text)
                engine.runAndWait()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Bug 7 fix: log the error instead of silently swallowing
                print(f"[TTS ERROR] {e}", file=sys.stderr)
                try:
                    self._queue.task_done()
                except Exception:
                    pass

        engine.stop()


# Module-level singleton
tts = TTSEngine()
