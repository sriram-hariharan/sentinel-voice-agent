from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.tools.schemas import (
    ActionConfirmation,
    ToolExecutionContext,
)


class ConversationPhase(StrEnum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    ESCALATING = "ESCALATING"
    ENDED = "ENDED"
    FAILED = "FAILED"


class AuthenticationLevel(StrEnum):
    UNAUTHENTICATED = "UNAUTHENTICATED"
    AUTHENTICATED = "AUTHENTICATED"


class EscalationStatus(StrEnum):
    NONE = "NONE"
    ESCALATING = "ESCALATING"
    ESCALATED = "ESCALATED"


class ResourceType(StrEnum):
    ACCOUNT = "ACCOUNT"
    CARD = "CARD"
    TRANSACTION = "TRANSACTION"


class ConversationStateError(RuntimeError):
    """Raised when a conversation-state operation is invalid."""


class PendingAction(BaseModel):
    action: str = Field(min_length=1)
    resource_id: UUID
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmation_required: bool = True
    confirmation_received: bool = False

    model_config = ConfigDict(frozen=True)


class ResourceCandidate(BaseModel):
    resource_id: UUID
    label: str = Field(min_length=1, max_length=255)
    selectors: tuple[str, ...] = Field(min_length=1)
    related_account_id: UUID | None = None

    model_config = ConfigDict(frozen=True)


class PendingResourceResolution(BaseModel):
    resource_type: ResourceType
    intent: str = Field(min_length=1)
    candidates: tuple[ResourceCandidate, ...] = Field(min_length=2)
    clarification: str = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(frozen=True)


class ConversationState(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    customer_id: UUID | None = None
    authentication_level: AuthenticationLevel = (
        AuthenticationLevel.UNAUTHENTICATED
    )

    phase: ConversationPhase = ConversationPhase.LISTENING
    active_intent: str | None = None
    active_account_id: UUID | None = None
    active_card_id: UUID | None = None
    active_transaction_id: UUID | None = None

    pending_resource_resolution: PendingResourceResolution | None = None
    pending_action: PendingAction | None = None
    retrieved_policy_sources: list[str] = Field(default_factory=list)
    escalation_status: EscalationStatus = EscalationStatus.NONE
    last_tool_result: dict[str, Any] | None = None
    conversation_summary: str = ""

    model_config = ConfigDict(validate_assignment=True)

    @property
    def authenticated(self) -> bool:
        return (
            self.authentication_level
            == AuthenticationLevel.AUTHENTICATED
            and self.customer_id is not None
        )

    def request_action(
        self,
        action: str,
        resource_id: UUID,
        *,
        arguments: dict[str, Any] | None = None,
        confirmation_required: bool = True,
    ) -> None:
        self.pending_resource_resolution = None
        self.pending_action = PendingAction(
            action=action,
            resource_id=resource_id,
            arguments=dict(arguments or {}),
            confirmation_required=confirmation_required,
        )

        if confirmation_required:
            self.phase = ConversationPhase.WAITING_FOR_CONFIRMATION
        else:
            self.phase = ConversationPhase.PROCESSING

    def request_resource_resolution(
        self,
        resource_type: ResourceType,
        intent: str,
        candidates: list[ResourceCandidate],
        clarification: str,
    ) -> None:
        self.pending_action = None
        self.pending_resource_resolution = PendingResourceResolution(
            resource_type=resource_type,
            intent=intent,
            candidates=tuple(candidates),
            clarification=clarification,
        )
        self.active_intent = intent
        self.phase = ConversationPhase.PROCESSING

    def clear_resource_resolution(self) -> None:
        self.pending_resource_resolution = None

    def confirm_pending_action(self) -> None:
        pending = self.pending_action

        if pending is None:
            raise ConversationStateError(
                "No pending action exists to confirm"
            )

        if not pending.confirmation_required:
            raise ConversationStateError(
                "Pending action does not require confirmation"
            )

        self.pending_action = pending.model_copy(
            update={"confirmation_received": True}
        )
        self.phase = ConversationPhase.PROCESSING

    def cancel_pending_action(self) -> None:
        self.pending_action = None

        if self.phase not in {
            ConversationPhase.INTERRUPTED,
            ConversationPhase.ENDED,
            ConversationPhase.FAILED,
        }:
            self.phase = ConversationPhase.PROCESSING

    def complete_pending_action(
        self,
        action: str,
        resource_id: UUID,
    ) -> None:
        pending = self.pending_action

        if (
            pending is None
            or pending.action != action
            or pending.resource_id != resource_id
        ):
            raise ConversationStateError(
                "Tool result does not match the current pending action"
            )

        self.pending_action = None

        if self.phase not in {
            ConversationPhase.INTERRUPTED,
            ConversationPhase.ENDED,
            ConversationPhase.FAILED,
        }:
            self.phase = ConversationPhase.PROCESSING

    def interrupt(self) -> None:
        # If execution has already started, keep the action tracked until
        # its deterministic result is known. Otherwise abandon it.
        if self.phase != ConversationPhase.TOOL_EXECUTION:
            self.pending_action = None
            self.pending_resource_resolution = None

        self.phase = ConversationPhase.INTERRUPTED

    def resume_listening(self) -> None:
        if self.phase in {
            ConversationPhase.ENDED,
            ConversationPhase.FAILED,
        }:
            raise ConversationStateError(
                "A terminal conversation cannot resume listening"
            )

        self.phase = ConversationPhase.LISTENING

    def mark_escalating(self) -> None:
        self.escalation_status = EscalationStatus.ESCALATING
        self.phase = ConversationPhase.ESCALATING

    def mark_escalated(self) -> None:
        self.escalation_status = EscalationStatus.ESCALATED
        self.phase = ConversationPhase.ESCALATING

    def to_tool_context(self) -> ToolExecutionContext:
        confirmation = None
        pending = self.pending_action

        if (
            pending is not None
            and pending.confirmation_required
            and pending.confirmation_received
        ):
            confirmation = ActionConfirmation(
                action=pending.action,
                resource_id=pending.resource_id,
                confirmed=True,
            )

        return ToolExecutionContext(
            session_id=self.session_id,
            customer_id=self.customer_id,
            authenticated=self.authenticated,
            confirmation=confirmation,
        )
