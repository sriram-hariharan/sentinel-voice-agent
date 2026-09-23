from hmac import compare_digest

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationState,
)
from backend.app.db.models import Customer


class InvalidCredentialsError(Exception):
    """Raised when synthetic login credentials are invalid."""


class SessionIdentityConflictError(Exception):
    """Raised when an authenticated session attempts to switch customers."""


async def authenticate_session(
    *,
    state: ConversationState,
    email: str,
    pin: str,
    demo_pin: str,
    db: AsyncSession,
) -> ConversationState:
    normalized_email = email.strip().lower()

    customer = await db.scalar(
        select(Customer).where(
            func.lower(Customer.email) == normalized_email,
            Customer.status == "ACTIVE",
        )
    )

    pin_matches = compare_digest(pin, demo_pin)

    profile = customer.authentication_profile if customer is not None else {}

    demo_pin_enrolled = (
        profile.get("verification_method") == "demo_pin"
        and profile.get("verification_status") == "ENROLLED"
    )

    if (
        customer is None
        or not pin_matches
        or not demo_pin_enrolled
    ):
        raise InvalidCredentialsError("Invalid credentials")

    if (
        state.authenticated
        and state.customer_id != customer.customer_id
    ):
        raise SessionIdentityConflictError(
            "Authenticated session cannot switch customer identity"
        )

    state.customer_id = customer.customer_id
    state.authentication_level = AuthenticationLevel.AUTHENTICATED

    return state
