"""add policy retrieval schema

Revision ID: a14c0f17d901
Revises: d2a42313c6f8
Create Date: 2026-09-24 20:15:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "a14c0f17d901"
down_revision: str | Sequence[str] | None = "d2a42313c6f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "policy_documents",
        sa.Column("document_id", sa.String(length=100), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("document_id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "policy_chunks",
        sa.Column("chunk_id", sa.String(length=180), nullable=False),
        sa.Column("document_id", sa.String(length=100), nullable=False),
        sa.Column("section", sa.String(length=255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', coalesce(content, ''))",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["policy_documents.document_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index(
        "ix_policy_chunks_document_id",
        "policy_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_policy_chunks_search_vector",
        "policy_chunks",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_policy_chunks_search_vector",
        table_name="policy_chunks",
        postgresql_using="gin",
    )
    op.drop_index(
        "ix_policy_chunks_document_id",
        table_name="policy_chunks",
    )
    op.drop_table("policy_chunks")
    op.drop_table("policy_documents")
