import logging
import re
import textwrap
from collections.abc import AsyncIterator
from typing import Any

from groq import AsyncGroq

logger = logging.getLogger(__name__)


class SpeechProviderError(RuntimeError):
    """Raised when a speech provider call fails."""


class SpeechProviderResponseError(SpeechProviderError):
    """Raised when a speech provider returns unusable content."""


def normalize_transcript(text: str) -> str:
    return " ".join(text.split())


def split_tts_text(text: str, *, max_chars: int = 190) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    normalized = normalize_transcript(text)

    if not normalized:
        return []

    sentence_like_units = re.findall(
        r".+?(?:[.!?]+(?=\s|$)|$)",
        normalized,
    )
    chunks: list[str] = []
    current = ""

    for unit in sentence_like_units:
        unit = unit.strip()
        pieces = (
            [unit]
            if len(unit) <= max_chars
            else textwrap.wrap(
                unit,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )

        for piece in pieces:
            candidate = f"{current} {piece}".strip()

            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate

    if current:
        chunks.append(current)

    return chunks


class GroqSpeechToTextProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Groq API key is required")

        self.model = model
        self._client = client or AsyncGroq(
            api_key=api_key,
            max_retries=1,
        )

    async def transcribe(self, audio: bytes) -> str:
        if not audio:
            return ""

        try:
            response = await self._client.audio.transcriptions.create(
                file=("sentinelvoice-turn.wav", audio),
                model=self.model,
                response_format="json",
                language="en",
                temperature=0,
            )
        except Exception as exc:
            raise SpeechProviderError("Groq transcription failed") from exc

        return normalize_transcript(getattr(response, "text", "") or "")


class GroqTextToSpeechProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "canopylabs/orpheus-v1-english",
        voice: str = "hannah",
        max_chars: int = 190,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Groq API key is required")

        self.model = model
        self.voice = voice
        self.max_chars = max_chars
        self._client = client or AsyncGroq(
            api_key=api_key,
            max_retries=1,
        )

    async def synthesize_chunks(self, text: str) -> AsyncIterator[bytes]:
        text_chunks = split_tts_text(
            text,
            max_chars=self.max_chars,
        )

        for chunk_index, chunk in enumerate(text_chunks, start=1):
            logger.info(
                "groq tts request started",
                extra={
                    "tts_chunk_index": chunk_index,
                    "tts_chunk_count": len(text_chunks),
                    "tts_text_length": len(chunk),
                    "tts_model": self.model,
                },
            )
            try:
                response = await self._client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=chunk,
                    response_format="wav",
                    sample_rate=24000,
                )
                audio = await response.read()
            except Exception as exc:
                raise SpeechProviderError("Groq speech synthesis failed") from exc

            if not audio:
                raise SpeechProviderResponseError(
                    "Groq returned empty speech audio"
                )

            logger.info(
                "groq tts response received",
                extra={
                    "tts_chunk_index": chunk_index,
                    "tts_chunk_count": len(text_chunks),
                    "audio_byte_count": len(audio),
                    "tts_model": self.model,
                },
            )
            yield audio

    async def synthesize(self, text: str) -> list[bytes]:
        """Collect all WAV chunks for callers that need the legacy API."""
        return [chunk async for chunk in self.synthesize_chunks(text)]
