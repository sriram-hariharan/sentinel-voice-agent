import asyncio
import io
import wave

import pytest
from livekit import rtc
from livekit.agents import stt

from backend.app.observability.context import VoiceCorrelationQueue, trace_scope
from backend.app.observability.tracing import InMemoryTraceSink, use_trace_sink
from backend.app.providers.groq_speech import SpeechProviderError
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

    async def synthesize_chunks(self, text: str):
        self.text = text
        for chunk in self.chunks:
            yield chunk


class ControlledTTSProvider:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.second_request_started = asyncio.Event()
        self.release_second_request = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def synthesize_chunks(self, text: str):
        del text
        try:
            yield self.chunks[0]
            self.second_request_started.set()
            await self.release_second_request.wait()
            for chunk in self.chunks[1:]:
                yield chunk
        finally:
            self.cancelled.set()


class FailingTTSProvider:
    def __init__(self, *, first_chunk: bytes | None = None) -> None:
        self.first_chunk = first_chunk

    async def synthesize_chunks(self, text: str):
        del text
        if self.first_chunk is not None:
            yield self.first_chunk
        raise SpeechProviderError("controlled synthesis failure")


class ControlledLaterFailingTTSProvider:
    def __init__(self, first_chunk: bytes) -> None:
        self.first_chunk = first_chunk
        self.second_request_started = asyncio.Event()
        self.fail_second_request = asyncio.Event()

    async def synthesize_chunks(self, text: str):
        del text
        yield self.first_chunk
        self.second_request_started.set()
        await self.fail_second_request.wait()
        raise SpeechProviderError("controlled synthesis failure")


def _wav_bytes(
    *,
    frames: int = 240,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
    sample_value: int = 0,
) -> bytes:
    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        sample = sample_value.to_bytes(sample_width, "little", signed=True)
        wav_file.writeframes(sample * frames * channels)

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
    pcm_chunks = [
        (11).to_bytes(2, "little", signed=True) * 240,
        (22).to_bytes(2, "little", signed=True) * 240,
        (33).to_bytes(2, "little", signed=True) * 240,
    ]
    provider = StubTTSProvider(
        [
            _wav_bytes(frames=240, sample_value=11),
            _wav_bytes(frames=240, sample_value=22),
            _wav_bytes(frames=240, sample_value=33),
        ]
    )
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        frame = await adapter.synthesize("Complete response text.").collect()

    assert provider.text == "Complete response text."
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 720
    assert bytes(frame.data) == b"".join(pcm_chunks)
    assert not bytes(frame.data).startswith(b"RIFF")
    assert adapter.model == "configured-orpheus"
    assert adapter.provider == "groq"
    completed = next(
        event for event in sink.events if event.event_name == "tts.completed"
    )
    assert completed.metadata["provider_request_count"] == 3
    assert completed.metadata["wav_chunk_count"] == 3


@pytest.mark.asyncio
async def test_first_pcm_is_emitted_before_second_provider_chunk_finishes() -> None:
    provider = ControlledTTSProvider(
        [
            _wav_bytes(frames=9600, sample_value=11),
            _wav_bytes(frames=240, sample_value=22),
        ]
    )
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        async with adapter.synthesize("A response requiring two chunks.") as stream:
            first_audio = asyncio.create_task(anext(stream))
            await asyncio.wait_for(provider.second_request_started.wait(), timeout=1)

            first_event = await asyncio.wait_for(first_audio, timeout=1)
            assert first_event.frame.samples_per_channel > 0
            assert provider.release_second_request.is_set() is False
            assert [event.event_name for event in sink.events].count(
                "tts.first_audio"
            ) == 1
            assert not any(
                event.event_name == "tts.completed" for event in sink.events
            )

            provider.release_second_request.set()
            remaining_frames = [event.frame async for event in stream]

    combined = rtc.combine_audio_frames([first_event.frame, *remaining_frames])
    assert combined.samples_per_channel == 9840
    completed = next(
        event for event in sink.events if event.event_name == "tts.completed"
    )
    assert completed.metadata["provider_request_count"] == 2
    assert completed.metadata["wav_chunk_count"] == 2


@pytest.mark.asyncio
async def test_cancelling_after_first_pcm_stops_future_provider_requests() -> None:
    provider = ControlledTTSProvider(
        [
            _wav_bytes(frames=9600, sample_value=11),
            _wav_bytes(frames=240, sample_value=22),
            _wav_bytes(frames=240, sample_value=33),
        ]
    )
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    stream = adapter.synthesize("A response requiring three chunks.")

    first_audio = asyncio.create_task(anext(stream))
    await asyncio.wait_for(provider.second_request_started.wait(), timeout=1)
    await asyncio.wait_for(first_audio, timeout=1)
    await stream.aclose()

    assert provider.cancelled.is_set() is True
    assert provider.release_second_request.is_set() is False


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
    assert completed.metadata["generated_audio_seconds_by_chunk"] == [
        pytest.approx(0.01)
    ]
    assert completed.metadata["generated_audio_seconds"] < 1


@pytest.mark.asyncio
async def test_tts_adapter_rejects_unexpected_audio_format() -> None:
    provider = StubTTSProvider([_wav_bytes(sample_rate=16000)])
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")

    sink = InMemoryTraceSink()

    with (
        use_trace_sink(sink),
        trace_scope(session_id="session-1"),
        pytest.raises(ValueError, match="24 kHz, mono, 16-bit WAV"),
    ):
        await adapter.synthesize("Response text.").collect()

    assert not any(event.event_name == "tts.first_audio" for event in sink.events)
    assert any(event.event_name == "tts.failed" for event in sink.events)


@pytest.mark.asyncio
async def test_tts_adapter_rejects_malformed_wav_without_emitting_pcm() -> None:
    provider = StubTTSProvider([b"not-a-wav-container"])
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with (
        use_trace_sink(sink),
        trace_scope(session_id="session-1"),
        pytest.raises(wave.Error),
    ):
        await adapter.synthesize("Response text.").collect()

    assert not any(event.event_name == "tts.first_audio" for event in sink.events)
    assert any(event.event_name == "tts.failed" for event in sink.events)


@pytest.mark.asyncio
async def test_first_provider_chunk_failure_emits_no_pcm() -> None:
    provider = FailingTTSProvider()
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with (
        use_trace_sink(sink),
        trace_scope(session_id="session-1"),
        pytest.raises(SpeechProviderError, match="controlled"),
    ):
        await adapter.synthesize("Response text.").collect()

    assert not any(event.event_name == "tts.first_audio" for event in sink.events)
    failed = next(event for event in sink.events if event.event_name == "tts.failed")
    assert "provider_request_count" not in failed.metadata
    assert failed.metadata["wav_chunk_count"] == 0


@pytest.mark.asyncio
async def test_later_provider_failure_does_not_duplicate_emitted_pcm() -> None:
    first_chunk_frames = 9600
    provider = ControlledLaterFailingTTSProvider(
        _wav_bytes(
            frames=first_chunk_frames,
            sample_value=11,
        )
    )
    adapter = GroqTTSAdapter(provider=provider, model="configured-orpheus")
    sink = InMemoryTraceSink()

    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        stream = adapter.synthesize("Response text.")
        first_audio = asyncio.create_task(anext(stream))
        await asyncio.wait_for(provider.second_request_started.wait(), timeout=1)
        first_event = await asyncio.wait_for(first_audio, timeout=1)
        emitted_frames = [first_event.frame]
        provider.fail_second_request.set()

        with pytest.raises(SpeechProviderError, match="controlled"):
            async for event in stream:
                emitted_frames.append(event.frame)

    emitted = rtc.combine_audio_frames(emitted_frames)
    assert 0 < emitted.samples_per_channel <= first_chunk_frames
    assert set(bytes(emitted.data)[::2]) == {11}
    failed = next(event for event in sink.events if event.event_name == "tts.failed")
    assert failed.metadata["wav_chunk_count"] == 1
    assert "provider_request_count" not in failed.metadata
