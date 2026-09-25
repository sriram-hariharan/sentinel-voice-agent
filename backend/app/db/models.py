import uuid
from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(32))
    date_of_birth: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32))
    authentication_profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE', 'LOCKED', 'CLOSED')",
            name="ck_customers_status",
        ),
    )


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        index=True,
    )
    account_type: Mapped[str] = mapped_column(String(32))
    masked_account_number: Mapped[str] = mapped_column(String(32), unique=True)
    current_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    available_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "account_type IN ('checking', 'savings')",
            name="ck_accounts_account_type",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'FROZEN', 'CLOSED')",
            name="ck_accounts_status",
        ),
    )


class Card(Base):
    __tablename__ = "cards"

    card_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        index=True,
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.account_id"),
        index=True,
    )
    masked_card_number: Mapped[str] = mapped_column(String(32), unique=True)
    card_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    expiration_month: Mapped[int]
    expiration_year: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE', 'FROZEN', 'LOST', 'STOLEN', 'CLOSED')",
            name="ck_cards_status",
        ),
        CheckConstraint(
            "expiration_month BETWEEN 1 AND 12",
            name="ck_cards_expiration_month",
        ),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.account_id"),
        index=True,
    )
    card_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cards.card_id"),
        nullable=True,
        index=True,
    )
    merchant_name: Mapped[str] = mapped_column(String(255))
    merchant_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    transaction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    posted_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32))
    transaction_type: Mapped[str] = mapped_column(String(64))
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'POSTED', 'REVERSED', 'DECLINED')",
            name="ck_transactions_status",
        ),
    )


class Dispute(Base):
    __tablename__ = "disputes"

    dispute_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        index=True,
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("transactions.transaction_id"),
        index=True,
    )
    reason_code: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN', 'UNDER_REVIEW', 'RESOLVED', 'REJECTED', 'CANCELLED')",
            name="ck_disputes_status",
        ),
    )


class SupportCase(Base):
    __tablename__ = "support_cases"

    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    customer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str] = mapped_column(String(64))
    priority: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)
    handoff_reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "priority IN ('LOW', 'MEDIUM', 'HIGH', 'URGENT')",
            name="ck_support_cases_priority",
        ),
        CheckConstraint(
            "status IN ('OPEN', 'ESCALATED', 'RESOLVED', 'CLOSED')",
            name="ck_support_cases_status",
        ),
    )


class PolicyDocumentRecord(Base):
    __tablename__ = "policy_documents"

    document_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    title: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(32))
    effective_date: Mapped[date] = mapped_column(Date)
    source_path: Mapped[str] = mapped_column(String(500))
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PolicyChunkRecord(Base):
    __tablename__ = "policy_chunks"

    chunk_id: Mapped[str] = mapped_column(String(180), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("policy_documents.document_id", ondelete="CASCADE"),
        index=True,
    )
    section: Mapped[str] = mapped_column(String(255))
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(content, ''))",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_policy_chunks_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )
