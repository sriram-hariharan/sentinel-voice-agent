from typing import Annotated

from fastapi import Depends, HTTPException, status

from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.protected_action_verifier import (
    VERIFIER_MAX_COMPLETION_TOKENS,
    LLMProtectedActionSemanticVerifier,
)
from backend.app.config.settings import Settings, get_settings
from backend.app.providers.embeddings import FastEmbedProvider
from backend.app.providers.groq_llm import GroqLLMProvider
from backend.app.rag.retrieval import PolicyRetriever


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


def build_protected_action_verifier(
    settings: Settings,
) -> LLMProtectedActionSemanticVerifier:
    """Dedicated small-budget provider; the conversational budget is unchanged."""
    api_key = (
        settings.groq_api_key.get_secret_value()
        if settings.groq_api_key is not None
        else ""
    )

    if not api_key:
        raise ValueError("Groq API key is not configured")

    return LLMProtectedActionSemanticVerifier(
        llm=GroqLLMProvider(
            api_key=api_key,
            model=settings.llm_model,
            max_completion_tokens=VERIFIER_MAX_COMPLETION_TOKENS,
        ),
    )


def build_agent_orchestrator(settings: Settings) -> AgentOrchestrator:
    embedding_provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        cache_dir=settings.fastembed_cache_dir,
    )
    return AgentOrchestrator(
        llm=build_llm_provider(settings),
        policy_retriever=PolicyRetriever(
            embedding_provider=embedding_provider,
            min_vector_similarity=(
                settings.policy_retrieval_min_similarity
            ),
        ),
        protected_action_verifier=build_protected_action_verifier(settings),
        policy_top_k=settings.policy_retrieval_top_k,
    )


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
