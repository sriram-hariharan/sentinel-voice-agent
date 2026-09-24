import io
import logging
import wave
from uuid import uuid4

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    tts,
)

from backend.app.providers.speech import (
    SpeechToTextProvider,
    TextToSpeechProvider,
)

logger = logging.getLogger(__name__)


class GroqSTTAdapter(stt.STT):
    """Expose SentinelVoice's bounded-turn STT provider to LiveKit Agents."""

    def __init__(self, *, provider: SpeechToTextProvider, model: str) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._provider = provider
        self._model = model

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
        audio = rtc.combine_audio_frames(buffer).to_wav_bytes()
        transcript = await self._provider.transcribe(audio)
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
        wav_chunks = await self._provider.synthesize(self._input_text)
        output_emitter.initialize(
            request_id=uuid4().hex,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
        )

        sample_count = 0
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

                frame_count = wav_file.getnframes()
                output_emitter.push(wav_file.readframes(frame_count))
                sample_count += frame_count

        logger.info(
            "groq tts audio frames yielded to livekit",
            extra={
                "wav_chunk_count": len(wav_chunks),
                "sample_count": sample_count,
            },
        )
