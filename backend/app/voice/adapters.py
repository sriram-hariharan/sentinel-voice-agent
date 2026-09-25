import io
import logging
import time
import wave
from uuid import uuid4

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    tts,
)

from backend.app.observability.context import (
    VoiceCorrelationQueue,
    trace_scope,
)
from backend.app.observability.events import TraceStatus
from backend.app.observability.tracing import emit_trace_event, trace_span
from backend.app.providers.speech import (
    SpeechToTextProvider,
    TextToSpeechProvider,
)

logger = logging.getLogger(__name__)


class GroqSTTAdapter(stt.STT):
    """Expose SentinelVoice's bounded-turn STT provider to LiveKit Agents."""

    def __init__(
        self,
        *,
        provider: SpeechToTextProvider,
        model: str,
        session_id: str | None = None,
        correlation_queue: VoiceCorrelationQueue | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._provider = provider
        self._model = model
        self._session_id = session_id
        self._correlation_queue = correlation_queue

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq"

    async def _recognize_impl(
        self,
        buffer: rtc.AudioFrame | list[rtc.AudioFrame],
        *,
        language: object,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        del language, conn_options
        combined = rtc.combine_audio_frames(buffer)
        audio = combined.to_wav_bytes()
        audio_seconds = (
            combined.samples_per_channel / combined.sample_rate
            if combined.sample_rate
            else 0.0
        )
        correlation = (
            self._correlation_queue.create(session_id=self._session_id)
            if self._correlation_queue is not None and self._session_id
            else None
        )

        if correlation is None:
            async with trace_span(
                "stt",
                component="voice",
                metadata={
                    "provider": self.provider,
                    "model": self.model,
                    "audio_seconds": audio_seconds,
                    "provider_request_count": 1,
                },
            ):
                transcript = await self._provider.transcribe(audio)
        else:
            with trace_scope(
                session_id=correlation.session_id,
                trace_id=correlation.trace_id,
                turn_id=correlation.turn_id,
            ):
                async with trace_span(
                    "stt",
                    component="voice",
                    metadata={
                        "provider": self.provider,
                        "model": self.model,
                        "audio_seconds": audio_seconds,
                        "provider_request_count": 1,
                    },
                ):
                    transcript = await self._provider.transcribe(audio)
            if transcript:
                self._correlation_queue.record(correlation)
        alternatives = (
            [stt.SpeechData(language="en", text=transcript)]
            if transcript
            else []
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=alternatives,
        )


class GroqTTSAdapter(tts.TTS):
    """Expose SentinelVoice's chunked WAV TTS provider to LiveKit Agents."""

    def __init__(self, *, provider: TextToSpeechProvider, model: str) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._provider = provider
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _GroqChunkedStream(
            tts_adapter=self,
            provider=self._provider,
            input_text=text,
            conn_options=conn_options,
        )


class _GroqChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts_adapter: GroqTTSAdapter,
        provider: TextToSpeechProvider,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts_adapter,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._provider = provider

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        logger.info(
            "groq tts adapter started",
            extra={"assistant_text_length": len(self._input_text)},
        )
        started_at = time.perf_counter()
        async with trace_span(
            "tts",
            component="voice",
            metadata={
                "provider": self._tts.provider,
                "model": self._tts.model,
                "character_count": len(self._input_text),
                "provider_request_count": 1,
            },
        ) as span:
            wav_chunks = await self._provider.synthesize(self._input_text)
            output_emitter.initialize(
                request_id=uuid4().hex,
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )

            sample_count = 0
            first_audio_emitted = False
            for wav_bytes in wav_chunks:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                    if (
                        wav_file.getframerate() != 24000
                        or wav_file.getnchannels() != 1
                        or wav_file.getsampwidth() != 2
                    ):
                        raise ValueError(
                            "Groq TTS must return 24 kHz, mono, 16-bit WAV audio"
                        )

                    declared_frame_count = wav_file.getnframes()
                    pcm_bytes = wav_file.readframes(declared_frame_count)
                    bytes_per_frame = (
                        wav_file.getnchannels() * wav_file.getsampwidth()
                    )
                    if len(pcm_bytes) % bytes_per_frame:
                        raise ValueError(
                            "Groq TTS returned incomplete PCM audio frames"
                        )
                    actual_frame_count = len(pcm_bytes) // bytes_per_frame
                    output_emitter.push(pcm_bytes)
                    sample_count += actual_frame_count
                    if not first_audio_emitted:
                        emit_trace_event(
                            "tts.first_audio",
                            component="voice",
                            status=TraceStatus.COMPLETED,
                            duration_ms=(time.perf_counter() - started_at) * 1000,
                            metadata={
                                "provider": self._tts.provider,
                                "model": self._tts.model,
                            },
                        )
                        first_audio_emitted = True
            span.set_metadata(
                wav_chunk_count=len(wav_chunks),
                generated_audio_seconds=sample_count / 24000,
            )

        logger.info(
            "groq tts audio frames yielded to livekit",
            extra={
                "wav_chunk_count": len(wav_chunks),
                "sample_count": sample_count,
            },
        )
