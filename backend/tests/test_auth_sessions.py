from datetime import date
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.service import (
    InvalidCredentialsError,
    SessionIdentityConflictError,
    authenticate_session,
)
from backend.app.auth.sessions import (
    InMemorySessionStore,
    SessionNotFoundError,
)
from backend.app.conversation.state import AuthenticationLevel
from backend.app.db.models import Customer
from backend.app.main import app

CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_CUSTOMER_ID = UUID("22222222-2222-4222-8222-222222222222")

client = TestClient(app)


def _customer(
    customer_id: UUID = CUSTOMER_ID,
    email: str = "avery.morgan@example.test",
) -> Customer:
    return Customer(
        customer_id=customer_id,
        first_name="Avery",
        last_name="Morgan",
        email=email,
        phone="+1-555-0101",
        date_of_birth=date(1992, 4, 18),
        status="ACTIVE",
        authentication_profile={
            "verification_method": "demo_pin",
            "verification_status": "ENROLLED",
        },
    )


def test_session_store_creates_server_owned_session() -> None:
    store = InMemorySessionStore()

    state = store.create()

    assert state.session_id
    assert state.authenticated is False
    assert state.customer_id is None
    assert store.get(state.session_id) is state


def test_session_store_rejects_unknown_session() -> None:
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        store.get("does-not-exist")


@pytest.mark.asyncio
async def test_valid_demo_login_authenticates_server_state() -> None:
    store = InMemorySessionStore()
    state = store.create()

    db = AsyncMock(spec=AsyncSession)
    db.scalar.return_value = _customer()

    result = await authenticate_session(
        state=state,
        email="AVERY.MORGAN@EXAMPLE.TEST",
        pin="2468",
        demo_pin="2468",
        db=db,
    )

    assert result is state
    assert state.customer_id == CUSTOMER_ID
    assert state.authentication_level == AuthenticationLevel.AUTHENTICATED
    assert state.authenticated is True


@pytest.mark.asyncio
async def test_wrong_pin_does_not_authenticate_session() -> None:
    store = InMemorySessionStore()
    state = store.create()

    db = AsyncMock(spec=AsyncSession)
    db.scalar.return_value = _customer()

    with pytest.raises(InvalidCredentialsError):
        await authenticate_session(
            state=state,
            email="avery.morgan@example.test",
            pin="wrong",
            demo_pin="2468",
            db=db,
        )

    assert state.customer_id is None
    assert state.authenticated is False


@pytest.mark.asyncio
async def test_authenticated_session_cannot_switch_customer() -> None:
    store = InMemorySessionStore()
    state = store.create()

    state.customer_id = CUSTOMER_ID
    state.authentication_level = AuthenticationLevel.AUTHENTICATED

    db = AsyncMock(spec=AsyncSession)
    db.scalar.return_value = _customer(
        customer_id=OTHER_CUSTOMER_ID,
        email="jordan.lee@example.test",
    )

    with pytest.raises(SessionIdentityConflictError):
        await authenticate_session(
            state=state,
            email="jordan.lee@example.test",
            pin="2468",
            demo_pin="2468",
            db=db,
        )

    assert state.customer_id == CUSTOMER_ID


def test_create_session_api_returns_opaque_unauthenticated_session() -> None:
    response = client.post("/sessions")

    assert response.status_code == 201

    payload = response.json()

    assert payload["session_id"]
    assert payload["customer_id"] is None
    assert payload["authenticated"] is False
