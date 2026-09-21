from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SentinelVoice API"
    environment: str = "development"

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sentinelvoice"
    db_user: str = "sentinelvoice_db_user"
    db_password: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTINELVOICE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
