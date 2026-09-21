from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SentinelVoice API"
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTINELVOICE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
