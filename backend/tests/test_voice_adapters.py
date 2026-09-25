import io
import wave

import pytest
from livekit import rtc
from livekit.agents import stt

from backend.app.observability.context import VoiceCorrelationQueue, trace_scope
from backend.app.observability.tracing import InMemoryTraceSink, use_trace_sink
from backend.app.voice.adapters import GroqSTTAdapter, GroqTTSAdapter


class StubSTTProvider:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.audio: bytes | None = None

    async def transcribe(self, audio: bytes) -> str:
        self.audio = audio
        return self.transcript


class StubTTSProvider:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.text: str | None = None

    async def synthesize(self, text: str) -> list[bytes]:
        self.text = text
        return self.chunks


def _wav_bytes(
    *,
    frames: int = 240,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * channels * sample_width)

    return output.getvalue()


def _streaming_header_wav_bytes(*, frames: int = 240) -> bytes:
    audio = bytearray(_wav_bytes(frames=frames))
    audio[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    audio[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    return bytes(audio)


@pytest.mark.asyncio
async def test_stt_adapter_sends_wav_and_returns_final_transcript() -> None:
    provider = StubSTTProvider("What is my checking balance?")
    adapter = GroqSTTAdapter(provider=provider, model="configured-whisper")
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 480,
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=480,
    )

    event = await adapter.recognize(frame)

    assert provider.audio is not None
    assert provider.audio.startswith(b"RIFF")
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "What is my checking balance?"
    assert adapter.model == "configured-whisper"
    assert adapter.provider == "groq"


@pytest.mark.asyncio
async def test_stt_adapter_handles_empty_transcript() -> None:
    provider = StubSTTProvider("")
    queue = VoiceCorrelationQueue()
    adapter = GroqSTTAdapter(
        provider=provider,
        model="configured-whisper",
        session_id="session-1",
        correlation_queue=queue,
    )
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 480,
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=480,
    )

    event = await adapter.recognize(frame)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives == []
    assert queue.consume() is None


@pytest.mark.asyncio
async def test_tts_adapter_preserves_all_ordered_wav_chunks() -> None:
    provider = StubTTSProvider([_wav_bytes(), _wav_bytes()])
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")

    frame = await adapter.synthesize("Complete response text.").collect()

    assert provider.text == "Complete response text."
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 480
    assert adapter.model == "configured-orpheus"
    assert adapter.provider == "groq"


@pytest.mark.asyncio
async def test_tts_duration_uses_actual_pcm_with_streaming_wav_header() -> None:
    wav_bytes = _streaming_header_wav_bytes(frames=240)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() == 2_147_483_647

    provider = StubTTSProvider([wav_bytes])
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        frame = await adapter.synthesize("Short response.").collect()

    completed = next(
        event for event in sink.events if event.event_name == "tts.completed"
    )
    assert frame.samples_per_channel == 240
    assert completed.metadata["generated_audio_seconds"] == pytest.approx(0.01)
    assert completed.metadata["generated_audio_seconds"] < 1


@pytest.mark.asyncio
async def test_tts_adapter_rejects_unexpected_audio_format() -> None:
    provider = StubTTSProvider([_wav_bytes(sample_rate=16000)])
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")

    with pytest.raises(ValueError, match="24 kHz, mono, 16-bit WAV"):
        await adapter.synthesize("Response text.").collect()
