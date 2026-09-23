from functools import lru_cache
from secrets import token_urlsafe

from backend.app.conversation.state import ConversationState


class SessionNotFoundError(LookupError):
    """Raised when a server-side conversation session does not exist."""


class InMemorySessionStore:
    """Process-local V1 session store.

    The browser receives only an opaque session ID. Customer identity and
    authorization state remain server-owned.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def create(self) -> ConversationState:
        while True:
            session_id = token_urlsafe(32)

            if session_id not in self._sessions:
                break

        state = ConversationState(session_id=session_id)
        self._sessions[session_id] = state

        return state

    def get(self, session_id: str) -> ConversationState:
        state = self._sessions.get(session_id)

        if state is None:
            raise SessionNotFoundError("Session not found")

        return state


@lru_cache
def get_session_store() -> InMemorySessionStore:
    return InMemorySessionStore()
