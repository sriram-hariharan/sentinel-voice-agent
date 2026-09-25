from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SentinelVoice API"
    environment: str = "development"

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sentinelvoice"
    db_user: str = "sentinelvoice_db_user"
    db_password: SecretStr | None = None

    demo_pin: SecretStr | None = None
    demo_session_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    demo_max_turns_per_session: int = Field(default=30, ge=1, le=100)
    demo_max_active_sessions: int = Field(default=20, ge=1, le=100)

    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GROQ_API_KEY",
            "SENTINELVOICE_GROQ_API_KEY",
        ),
    )
    llm_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    policy_retrieval_top_k: int = 5
    policy_retrieval_min_similarity: float = 0.70
    fastembed_cache_dir: str | None = None

    livekit_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIVEKIT_URL",
            "SENTINELVOICE_LIVEKIT_URL",
        ),
    )
    livekit_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIVEKIT_API_KEY",
            "SENTINELVOICE_LIVEKIT_API_KEY",
        ),
    )
    livekit_api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIVEKIT_API_SECRET",
            "SENTINELVOICE_LIVEKIT_API_SECRET",
        ),
    )
    livekit_agent_name: str = "sentinelvoice"
    voice_token_ttl_seconds: int = Field(default=600, ge=60, le=1800)
    api_base_url: str = "http://127.0.0.1:8000"

    stt_model: str = "whisper-large-v3-turbo"
    tts_model: str = "canopylabs/orpheus-v1-english"
    tts_voice: str = "hannah"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTINELVOICE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
