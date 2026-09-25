from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from math import ceil
from secrets import token_urlsafe

from backend.app.config.settings import get_settings
from backend.app.conversation.state import ConversationState


class SessionNotFoundError(LookupError):
    """Raised when a server-side conversation session does not exist."""


class SessionCapacityError(RuntimeError):
    """Raised when the configured active-session capacity is exhausted."""


class SessionTurnLimitError(RuntimeError):
    """Raised when a conversation has consumed its configured turn budget."""


@dataclass
class _StoredSession:
    state: ConversationState
    created_at: datetime
    turn_count: int = 0


def _utc_now() -> datetime:
    return datetime.now(UTC)


class InMemorySessionStore:
    """Process-local V1 session store with bounded demo usage.

    The browser receives only an opaque session ID. Customer identity and
    authorization state remain server-owned. Expiry, active-session capacity,
    and turn budgets prevent a public demo from becoming an unbounded provider
    proxy.
    """

    def __init__(
        self,
        *,
        session_ttl_seconds: int = 900,
        max_active_sessions: int = 20,
        max_turns_per_session: int = 30,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be positive")
        if max_turns_per_session <= 0:
            raise ValueError("max_turns_per_session must be positive")

        self._session_ttl = timedelta(seconds=session_ttl_seconds)
        self._max_active_sessions = max_active_sessions
        self._max_turns_per_session = max_turns_per_session
        self._clock = clock
        self._sessions: dict[str, _StoredSession] = {}

    def _is_expired(
        self,
        record: _StoredSession,
        now: datetime,
    ) -> bool:
        return now >= record.created_at + self._session_ttl

    def _purge_expired(self, now: datetime) -> None:
        expired_ids = [
            session_id
            for session_id, record in self._sessions.items()
            if self._is_expired(record, now)
        ]

        for session_id in expired_ids:
            del self._sessions[session_id]

    def _get_record(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> _StoredSession:
        record = self._sessions.get(session_id)

        if record is None:
            raise SessionNotFoundError("Session not found")

        current_time = self._clock() if now is None else now

        if self._is_expired(record, current_time):
            del self._sessions[session_id]
            raise SessionNotFoundError("Session not found")

        return record

    def create(self) -> ConversationState:
        now = self._clock()
        self._purge_expired(now)

        if len(self._sessions) >= self._max_active_sessions:
            raise SessionCapacityError("Demo session capacity reached")

        while True:
            session_id = token_urlsafe(32)

            if session_id not in self._sessions:
                break

        state = ConversationState(session_id=session_id)
        self._sessions[session_id] = _StoredSession(
            state=state,
            created_at=now,
        )

        return state

    def get(self, session_id: str) -> ConversationState:
        return self._get_record(session_id).state

    def get_with_remaining_ttl(
        self,
        session_id: str,
    ) -> tuple[ConversationState, int]:
        record = self._get_record(session_id)
        now = self._clock()
        expires_at = record.created_at + self._session_ttl
        remaining_seconds = ceil((expires_at - now).total_seconds())

        if remaining_seconds <= 0:
            self._sessions.pop(session_id, None)
            raise SessionNotFoundError("Session not found")

        return record.state, remaining_seconds

    def claim_turn(self, session_id: str) -> int:
        record = self._get_record(session_id)

        if record.turn_count >= self._max_turns_per_session:
            raise SessionTurnLimitError("Session turn limit reached")

        record.turn_count += 1
        return record.turn_count


@lru_cache
def get_session_store() -> InMemorySessionStore:
    settings = get_settings()

    return InMemorySessionStore(
        session_ttl_seconds=settings.demo_session_ttl_seconds,
        max_active_sessions=settings.demo_max_active_sessions,
        max_turns_per_session=settings.demo_max_turns_per_session,
    )
