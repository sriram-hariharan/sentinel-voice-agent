from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.providers.groq_speech import (
    GroqSpeechToTextProvider,
    GroqTextToSpeechProvider,
    SpeechProviderError,
    SpeechProviderResponseError,
    split_tts_text,
)


def _groq_client() -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock()
    client.audio.speech.create = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_stt_normalizes_mocked_groq_transcript() -> None:
    client = _groq_client()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="  What   is my\nchecking balance?  "
    )
    provider = GroqSpeechToTextProvider(
        api_key="",
        model="configured-whisper",
        client=client,
    )

    transcript = await provider.transcribe(b"wav-audio")

    assert transcript == "What is my checking balance?"
    client.audio.transcriptions.create.assert_awaited_once_with(
        file=("sentinelvoice-turn.wav", b"wav-audio"),
        model="configured-whisper",
        response_format="json",
        language="en",
        temperature=0,
    )


@pytest.mark.asyncio
async def test_empty_stt_result_is_safe() -> None:
    client = _groq_client()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text=" \n")
    provider = GroqSpeechToTextProvider(api_key="", client=client)

    assert await provider.transcribe(b"wav-audio") == ""
    assert await provider.transcribe(b"") == ""
    client.audio.transcriptions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_uses_configured_model_and_voice() -> None:
    client = _groq_client()
    response = MagicMock()
    response.read = AsyncMock(return_value=b"wav-audio")
    client.audio.speech.create.return_value = response
    provider = GroqTextToSpeechProvider(
        api_key="",
        model="configured-orpheus",
        voice="configured-voice",
        client=client,
    )

    audio = await provider.synthesize("Your balance is $125.00.")

    assert audio == [b"wav-audio"]
    client.audio.speech.create.assert_awaited_once_with(
        model="configured-orpheus",
        voice="configured-voice",
        input="Your balance is $125.00.",
        response_format="wav",
        sample_rate=24000,
    )


def test_long_tts_text_is_split_without_truncation() -> None:
    text = (
        "Your checking balance is one hundred twenty-five dollars. "
        "The account is active and available for purchases. "
        "I can also show your recent transactions if you would like."
    )

    chunks = split_tts_text(text, max_chars=70)

    assert len(chunks) > 1
    assert all(len(chunk) <= 70 for chunk in chunks)
    assert " ".join(chunks) == " ".join(text.split())


@pytest.mark.asyncio
async def test_stt_provider_failure_is_controlled() -> None:
    client = _groq_client()
    client.audio.transcriptions.create.side_effect = RuntimeError("offline")
    provider = GroqSpeechToTextProvider(api_key="", client=client)

    with pytest.raises(SpeechProviderError, match="transcription failed"):
        await provider.transcribe(b"wav-audio")


@pytest.mark.asyncio
async def test_tts_provider_failure_is_controlled() -> None:
    client = _groq_client()
    client.audio.speech.create.side_effect = RuntimeError("offline")
    provider = GroqTextToSpeechProvider(api_key="", client=client)

    with pytest.raises(SpeechProviderError, match="synthesis failed"):
        await provider.synthesize("Your balance is available.")


@pytest.mark.asyncio
async def test_empty_tts_audio_is_rejected() -> None:
    client = _groq_client()
    response = MagicMock()
    response.read = AsyncMock(return_value=b"")
    client.audio.speech.create.return_value = response
    provider = GroqTextToSpeechProvider(api_key="", client=client)

    with pytest.raises(SpeechProviderResponseError, match="empty speech audio"):
        await provider.synthesize("Your balance is available.")
