from typing import Protocol


class SpeechToTextProvider(Protocol):
    async def transcribe(self, audio: bytes) -> str:
        """Return a normalized final transcript for one bounded audio turn."""
        ...


class TextToSpeechProvider(Protocol):
    async def synthesize(self, text: str) -> list[bytes]:
        """Return ordered WAV chunks containing the complete spoken text."""
        ...
