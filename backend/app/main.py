from fastapi import FastAPI

from backend.app.api.health import router as health_router
from backend.app.api.sessions import router as sessions_router
from backend.app.config.settings import get_settings
from backend.app.observability.logging import configure_runtime_logging

settings = get_settings()
configure_runtime_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(sessions_router)
