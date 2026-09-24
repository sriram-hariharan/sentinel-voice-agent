from typing import Annotated

from fastapi import Depends, HTTPException, status

from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.config.settings import Settings, get_settings
from backend.app.providers.groq_llm import GroqLLMProvider


def build_llm_provider(settings: Settings) -> GroqLLMProvider:
    api_key = (
        settings.groq_api_key.get_secret_value()
        if settings.groq_api_key is not None
        else ""
    )

    if not api_key:
        raise ValueError("Groq API key is not configured")

    return GroqLLMProvider(
        api_key=api_key,
        model=settings.llm_model,
    )


def build_agent_orchestrator(settings: Settings) -> AgentOrchestrator:
    return AgentOrchestrator(llm=build_llm_provider(settings))


def get_agent_orchestrator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentOrchestrator:
    try:
        return build_agent_orchestrator(settings)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured",
        ) from exc
