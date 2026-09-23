from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

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
from backend.app.db.session import get_db_session

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


class AuthenticateSessionRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    pin: SecretStr


class SessionResponse(BaseModel):
    session_id: str
    customer_id: UUID | None
    authenticated: bool


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    store: SessionStoreDep,
) -> SessionResponse:
    state = store.create()

    return SessionResponse(
        session_id=state.session_id,
        customer_id=None,
        authenticated=False,
    )


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

    return SessionResponse(
        session_id=state.session_id,
        customer_id=state.customer_id,
        authenticated=state.authenticated,
    )
