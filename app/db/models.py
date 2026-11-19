"""Additive-only PostgreSQL models. Unknown job requirements remain SQL NULL."""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    """Metadata shared by all eight persistence models."""


class CandidateProfileORM(Base):
    __tablename__ = "candidate_profiles"
    __table_args__ = (
        CheckConstraint("profile_version >= 1", name="ck_profiles_version_positive"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String)
    raw_resume_text: Mapped[str] = mapped_column(Text, default="")
    profile_version: Mapped[int] = mapped_column(Integer, default=1)
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CandidateEvidenceORM(Base):
    __tablename__ = "candidate_evidence"
    __table_args__ = (
        CheckConstraint("profile_version >= 1", name="ck_evidence_version_positive"),
        CheckConstraint(
            "source_type IN ('resume_statement', 'user_statement', "
            "'artifact_excerpt', 'not_established')",
            name="ck_evidence_source_type",
        ),
        CheckConstraint(
            "verification_status IN ('project_backed', 'resume_claimed', 'self_reported')",
            name="ck_evidence_verification_status",
        ),
    )

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_profiles.id"), index=True
    )
    skill: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String)
    verification_status: Mapped[str] = mapped_column(String)
    source_excerpt: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String)
    profile_version: Mapped[int] = mapped_column(Integer, default=1)


class PreferenceORM(Base):
    __tablename__ = "preferences"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_preferences_version_positive"),
        Index(
            "ix_preferences_one_active_per_profile",
            "profile_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_profiles.id"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    hard_constraints: Mapped[dict] = mapped_column(JSONB)
    soft_preferences: Mapped[dict] = mapped_column(JSONB)
    update_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class JobORM(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "norm_work_mode IN ('remote', 'hybrid', 'onsite', 'unknown')",
            name="ck_jobs_work_mode",
        ),
        CheckConstraint(
            "norm_remote_region IN ('india_eligible', 'global_remote', "
            "'non_india_restricted', 'unknown')",
            name="ck_jobs_remote_region",
        ),
        CheckConstraint(
            "norm_min_yoe >= 0 AND norm_min_yoe < 'Infinity'::float8",
            name="ck_jobs_min_yoe_nonnegative_finite",
        ),
    )

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String)
    raw_description: Mapped[str] = mapped_column(Text)
    excerpts: Mapped[dict] = mapped_column(JSONB, default=dict)
    norm_work_mode: Mapped[str] = mapped_column(String, default="unknown")
    norm_remote_region: Mapped[str] = mapped_column(String, default="unknown")
    norm_cities: Mapped[list[str]] = mapped_column(JSONB, default=list)
    norm_student_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    norm_degree_completed_required: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    norm_min_yoe: Mapped[float | None] = mapped_column(Float, nullable=True)
    constraint_excerpts: Mapped[dict] = mapped_column(JSONB, default=dict)
    search_tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)


class JobMetadataORM(Base):
    __tablename__ = "job_metadata"
    __table_args__ = (
        CheckConstraint(
            "source_status IN ('active', 'closed')", name="ck_job_metadata_status"
        ),
    )

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), primary_key=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String)
    source_status: Mapped[str] = mapped_column(String, default="active")
    dedup_key: Mapped[str] = mapped_column(String, index=True)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)


class JobEmbeddingORM(Base):
    __tablename__ = "job_embeddings"
    __table_args__ = (
        CheckConstraint("dimension = 768", name="ck_job_embeddings_dimension"),
    )

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), primary_key=True)
    model_id: Mapped[str] = mapped_column(String)
    dimension: Mapped[int] = mapped_column(Integer, default=768)
    content_hash: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))


class SearchRunORM(Base):
    __tablename__ = "search_runs"
    __table_args__ = (
        CheckConstraint("profile_version >= 1", name="ck_search_runs_profile_version"),
        CheckConstraint(
            "preference_version >= 1", name="ck_search_runs_preference_version"
        ),
        CheckConstraint("total_tokens >= 0", name="ck_search_runs_tokens"),
        CheckConstraint(
            "latency_ms >= 0 AND latency_ms < 'Infinity'::float8",
            name="ck_search_runs_latency",
        ),
        CheckConstraint(
            "estimated_cost_usd >= 0 AND estimated_cost_usd < 'Infinity'::float8",
            name="ck_search_runs_cost",
        ),
    )

    run_id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_profiles.id"), index=True
    )
    profile_version: Mapped[int] = mapped_column(Integer)
    preference_version: Mapped[int] = mapped_column(Integer)
    query: Mapped[str] = mapped_column(Text)
    system_variant: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    shortlist: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    needs_review_list: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    rejected_counterfactuals: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    used_mock_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TraceORM(Base):
    __tablename__ = "traces"
    __table_args__ = (
        CheckConstraint("step_index >= 0", name="ck_traces_step_index"),
        CheckConstraint("tokens_used >= 0", name="ck_traces_tokens"),
        CheckConstraint(
            "duration_ms >= 0 AND duration_ms < 'Infinity'::float8",
            name="ck_traces_duration",
        ),
    )

    trace_id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("search_runs.run_id"), index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    agent_name: Mapped[str] = mapped_column(String)
    tool_or_action: Mapped[str] = mapped_column(String)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    outcome: Mapped[str] = mapped_column(String)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    payload_summary: Mapped[dict] = mapped_column(JSONB, default=dict)
