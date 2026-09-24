from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.dependencies import get_agent_orchestrator
from backend.app.agent.orchestrator import (
    AgentOrchestrationError,
    AgentOrchestrator,
    AgentTurnStatus,
)
from backend.app.auth.service import (
    InvalidCredentialsError,
    SessionIdentityConflictError,
    authenticate_session,
)
from backend.app.auth.sessions import (
    InMemorySessionStore,
    SessionNotFoundError,
    get_session_store,
)
from backend.app.config.settings import Settings, get_settings
from backend.app.conversation.state import ConversationPhase, ConversationState
from backend.app.db.session import get_db_session
from backend.app.providers.groq_llm import LLMProviderError
from backend.app.voice.tokens import (
    VoiceConfigurationError,
    VoiceConnectionToken,
    create_voice_connection_token,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

SessionStoreDep = Annotated[
    InMemorySessionStore,
    Depends(get_session_store),
]
DbSessionDep = Annotated[
    AsyncSession,
    Depends(get_db_session),
]
SettingsDep = Annotated[
    Settings,
    Depends(get_settings),
]
AgentOrchestratorDep = Annotated[
    AgentOrchestrator,
    Depends(get_agent_orchestrator),
]


class AuthenticateSessionRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    pin: SecretStr


class SessionResponse(BaseModel):
    session_id: str
    customer_id: UUID | None
    authenticated: bool
    conversation_phase: ConversationPhase
    turn_status: AgentTurnStatus | None
    pending_action: str | None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class MessageResponse(BaseModel):
    session_id: str
    message: str
    turn_status: AgentTurnStatus
    conversation_phase: ConversationPhase
    authenticated: bool
    customer_id: UUID | None
    executed_tools: list[str]
    pending_action: str | None


def _session_response(state: ConversationState) -> SessionResponse:
    return SessionResponse(
        session_id=state.session_id,
        customer_id=state.customer_id,
        authenticated=state.authenticated,
        conversation_phase=state.phase,
        turn_status=state.last_turn_status,
        pending_action=(
            state.pending_action.action
            if state.pending_action is not None
            else None
        ),
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    store: SessionStoreDep,
) -> SessionResponse:
    state = store.create()

    return _session_response(state)


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
)
async def get_session(
    session_id: str,
    store: SessionStoreDep,
) -> SessionResponse:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    return _session_response(state)


@router.post(
    "/{session_id}/voice/token",
    response_model=VoiceConnectionToken,
)
async def create_voice_token(
    session_id: str,
    store: SessionStoreDep,
    settings: SettingsDep,
) -> VoiceConnectionToken:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    if not state.authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required for voice",
        )

    if settings.groq_api_key is None or not (
        settings.groq_api_key.get_secret_value().strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq speech providers are not configured",
        )

    try:
        return create_voice_connection_token(
            session_id=state.session_id,
            settings=settings,
        )
    except VoiceConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit is not configured",
        ) from exc


@router.post(
    "/{session_id}/authenticate",
    response_model=SessionResponse,
)
async def authenticate(
    session_id: str,
    request: AuthenticateSessionRequest,
    store: SessionStoreDep,
    db: DbSessionDep,
    settings: SettingsDep,
) -> SessionResponse:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    if settings.demo_pin is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo authentication is not configured",
        )

    demo_pin = settings.demo_pin.get_secret_value()

    if not demo_pin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo authentication is not configured",
        )

    try:
        await authenticate_session(
            state=state,
            email=request.email,
            pin=request.pin.get_secret_value(),
            demo_pin=demo_pin,
            db=db,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        ) from exc
    except SessionIdentityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already authenticated as another customer",
        ) from exc

    return _session_response(state)


@router.post(
    "/{session_id}/messages",
    response_model=MessageResponse,
)
async def create_message(
    session_id: str,
    request: MessageRequest,
    store: SessionStoreDep,
    db: DbSessionDep,
    orchestrator: AgentOrchestratorDep,
) -> MessageResponse:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    try:
        result = await orchestrator.handle_text_turn(
            user_text=request.message,
            state=state,
            db=db,
        )
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider request failed",
        ) from exc
    except AgentOrchestrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Agent could not complete the turn",
        ) from exc

    state.last_turn_status = result.status
    return MessageResponse(
        session_id=state.session_id,
        message=result.text,
        turn_status=result.status,
        conversation_phase=state.phase,
        authenticated=state.authenticated,
        customer_id=state.customer_id,
        executed_tools=result.executed_tools,
        pending_action=(
            state.pending_action.action
            if state.pending_action is not None
            else None
        ),
    )
