from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
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
    SessionCapacityError,
    SessionNotFoundError,
    SessionTurnLimitError,
    get_session_store,
)
from backend.app.config.settings import Settings, get_settings
from backend.app.conversation.state import (
    ConversationPhase,
    ConversationState,
    VoicePlaybackState,
    VoicePlaybackStatus,
)
from backend.app.db.session import get_db_session
from backend.app.observability.context import (
    TRACE_ID_HEADER,
    TURN_ID_HEADER,
    build_trace_context,
    trace_scope,
)
from backend.app.observability.events import TraceStatus
from backend.app.observability.tracing import emit_trace_event, trace_span
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
    voice_playback: VoicePlaybackState | None
    policy_sources: list[str]


class VoicePlaybackRequest(BaseModel):
    speech_id: str = Field(min_length=1, max_length=128)
    voice_turn_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=1)
    status: VoicePlaybackStatus
    response_phase: ConversationPhase | None = None
    interruption_stop_latency_ms: float | None = Field(
        default=None,
        ge=0,
        le=60_000,
    )

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @model_validator(mode="after")
    def validate_interruption_latency(self) -> "VoicePlaybackRequest":
        if (
            self.status == VoicePlaybackStatus.INTERRUPTED
            and self.interruption_stop_latency_ms is None
        ):
            raise ValueError(
                "interruption_stop_latency_ms is required when interrupted"
            )

        if (
            self.status != VoicePlaybackStatus.INTERRUPTED
            and self.interruption_stop_latency_ms is not None
        ):
            raise ValueError(
                "interruption_stop_latency_ms is only valid when interrupted"
            )

        if (
            self.status == VoicePlaybackStatus.SCHEDULED
            and self.response_phase is None
        ):
            raise ValueError("response_phase is required when scheduled")

        if (
            self.status != VoicePlaybackStatus.SCHEDULED
            and self.response_phase is not None
        ):
            raise ValueError("response_phase is only valid when scheduled")

        return self


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class MessageResponse(BaseModel):
    session_id: str
    trace_id: str
    turn_id: str
    message: str
    turn_status: AgentTurnStatus
    conversation_phase: ConversationPhase
    authenticated: bool
    customer_id: UUID | None
    executed_tools: list[str]
    pending_action: str | None
    policy_sources: list[str]


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
        voice_playback=state.voice_playback,
        policy_sources=state.retrieved_policy_sources,
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    store: SessionStoreDep,
) -> SessionResponse:
    try:
        state = store.create()
    except SessionCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demo session capacity reached",
        ) from exc

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
    "/{session_id}/voice/playback",
    response_model=SessionResponse,
)
async def record_voice_playback(
    session_id: str,
    request: VoicePlaybackRequest,
    store: SessionStoreDep,
    trace_id: Annotated[str | None, Header(alias=TRACE_ID_HEADER)] = None,
    turn_id: Annotated[str | None, Header(alias=TURN_ID_HEADER)] = None,
) -> SessionResponse:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    try:
        correlation = build_trace_context(
            session_id=session_id,
            trace_id=trace_id,
            turn_id=turn_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace correlation header",
        ) from exc

    with trace_scope(
        session_id=session_id,
        trace_id=correlation.trace_id,
        turn_id=correlation.turn_id,
    ):
        state.record_voice_playback(
            speech_id=request.speech_id,
            voice_turn_id=request.voice_turn_id,
            sequence=request.sequence,
            status=request.status,
            interruption_stop_latency_ms=(
                request.interruption_stop_latency_ms
            ),
            response_phase=request.response_phase,
        )
        if request.status == VoicePlaybackStatus.INTERRUPTED:
            emit_trace_event(
                "voice.interruption.completed",
                component="api",
                status=TraceStatus.COMPLETED,
                duration_ms=request.interruption_stop_latency_ms,
                metadata={"speech_id": request.speech_id},
            )
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
        state, remaining_session_ttl_seconds = store.get_with_remaining_ttl(
            session_id
        )
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
            max_ttl_seconds=remaining_session_ttl_seconds,
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
    trace_id: Annotated[str | None, Header(alias=TRACE_ID_HEADER)] = None,
    turn_id: Annotated[str | None, Header(alias=TURN_ID_HEADER)] = None,
) -> MessageResponse:
    try:
        state = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    try:
        correlation = build_trace_context(
            session_id=session_id,
            trace_id=trace_id,
            turn_id=turn_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace correlation header",
        ) from exc

    try:
        # Failed provider/tool turns still consume budget because they have
        # entered real agent processing. Structurally invalid requests do not.
        store.claim_turn(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc
    except SessionTurnLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Session turn limit reached",
        ) from exc

    try:
        with trace_scope(
            session_id=session_id,
            trace_id=correlation.trace_id,
            turn_id=correlation.turn_id,
        ), trace_span(
            "agent.turn",
            component="api",
            metadata={
                "authenticated": state.authenticated,
                "input_character_count": len(request.message),
            },
        ) as span:
            result = await orchestrator.handle_text_turn(
                user_text=request.message,
                state=state,
                db=db,
            )
            span.set_metadata(
                turn_status=result.status.value,
                executed_tools=result.executed_tools,
                policy_source_count=len(result.policy_sources),
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
        trace_id=correlation.trace_id,
        turn_id=correlation.turn_id,
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
        policy_sources=result.policy_sources,
    )
