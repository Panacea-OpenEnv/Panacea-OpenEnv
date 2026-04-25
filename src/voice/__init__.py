__all__ = ["STTEngine", "TTSEngine"]


def __getattr__(name):
    if name == "STTEngine":
        from .stt import STTEngine
        return STTEngine
    if name == "TTSEngine":
        from .tts import TTSEngine
        return TTSEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
