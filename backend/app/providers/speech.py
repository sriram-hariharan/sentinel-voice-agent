from collections.abc import AsyncIterator
from typing import Protocol


class SpeechToTextProvider(Protocol):
    async def transcribe(self, audio: bytes) -> str:
        """Return a normalized final transcript for one bounded audio turn."""
        ...


class TextToSpeechProvider(Protocol):
    def synthesize_chunks(self, text: str) -> AsyncIterator[bytes]:
        """Yield each ordered WAV chunk as its provider request completes."""
        ...
