"""Real PostgreSQL gate plus local metadata checks; no SQLite/mock fallback."""

from datetime import UTC, datetime

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from app.db.models import (
    Base,
    CandidateEvidenceORM,
    CandidateProfileORM,
    JobEmbeddingORM,
    JobMetadataORM,
    JobORM,
    PreferenceORM,
    SearchRunORM,
    TraceORM,
)
from app.db.session import SessionLocal, engine, init_db
from app.schemas.contracts import (
    EvidenceRecord,
    HardConstraints,
    PreferenceState,
    SoftPreferences,
)

EXPECTED_COLUMNS = {
    "candidate_profiles": {
        "id",
        "name",
        "raw_resume_text",
        "profile_version",
        "is_confirmed",
        "created_at",
    },
    "candidate_evidence": {
        "evidence_id",
        "profile_id",
        "skill",
        "source_type",
        "verification_status",
        "source_excerpt",
        "source_id",
        "profile_version",
    },
    "preferences": {
        "id",
        "profile_id",
        "version",
        "is_active",
        "hard_constraints",
        "soft_preferences",
        "update_note",
        "created_at",
    },
    "jobs": {
        "job_id",
        "source_url",
        "title",
        "company",
        "raw_description",
        "excerpts",
        "norm_work_mode",
        "norm_remote_region",
        "norm_cities",
        "norm_student_eligible",
        "norm_degree_completed_required",
        "norm_min_yoe",
        "constraint_excerpts",
        "search_tsv",
    },
    "job_metadata": {
        "job_id",
        "captured_at",
        "content_hash",
        "source_status",
        "dedup_key",
        "is_duplicate",
    },
    "job_embeddings": {"job_id", "model_id", "dimension", "content_hash", "embedding"},
    "search_runs": {
        "run_id",
        "profile_id",
        "profile_version",
        "preference_version",
        "query",
        "system_variant",
        "status",
        "shortlist",
        "needs_review_list",
        "rejected_counterfactuals",
        "latency_ms",
        "total_tokens",
        "estimated_cost_usd",
        "used_mock_fallback",
        "created_at",
    },
    "traces": {
        "trace_id",
        "run_id",
        "step_index",
        "agent_name",
        "tool_or_action",
        "duration_ms",
        "outcome",
        "tokens_used",
        "payload_summary",
    },
}


def test_exact_eight_models_and_columns():
    assert set(Base.metadata.tables) == set(EXPECTED_COLUMNS)
    for name, expected in EXPECTED_COLUMNS.items():
        table = Base.metadata.tables[name]
        assert set(table.columns.keys()) == expected
        assert len(table.primary_key.columns) == 1
    for model, column, target in (
        (CandidateEvidenceORM, "profile_id", "candidate_profiles.id"),
        (PreferenceORM, "profile_id", "candidate_profiles.id"),
        (JobMetadataORM, "job_id", "jobs.job_id"),
        (JobEmbeddingORM, "job_id", "jobs.job_id"),
        (SearchRunORM, "profile_id", "candidate_profiles.id"),
        (TraceORM, "run_id", "search_runs.run_id"),
    ):
        assert {
            fk.target_fullname for fk in model.__table__.c[column].foreign_keys
        } == {target}


def test_postgresql_types_and_nullability():
    for model, fields in (
        (PreferenceORM, ["hard_constraints", "soft_preferences"]),
        (JobORM, ["excerpts", "norm_cities", "constraint_excerpts"]),
        (SearchRunORM, ["shortlist", "needs_review_list", "rejected_counterfactuals"]),
        (TraceORM, ["payload_summary"]),
    ):
        for field in fields:
            assert isinstance(model.__table__.c[field].type, JSONB)
    assert isinstance(JobORM.__table__.c.search_tsv.type, TSVECTOR)
    vector_type = JobEmbeddingORM.__table__.c.embedding.type
    assert isinstance(vector_type, Vector) and vector_type.dim == 768
    assert isinstance(PreferenceORM.__table__.c.version.type, Integer)
    for field in ("norm_student_eligible", "norm_degree_completed_required"):
        column = JobORM.__table__.c[field]
        assert isinstance(column.type, Boolean) and column.nullable
    yoe = JobORM.__table__.c.norm_min_yoe
    assert isinstance(yoe.type, Float) and yoe.nullable
    for model, field in (
        (CandidateProfileORM, "created_at"),
        (PreferenceORM, "created_at"),
        (JobMetadataORM, "captured_at"),
        (SearchRunORM, "created_at"),
    ):
        column = model.__table__.c[field]
        assert isinstance(column.type, DateTime) and column.type.timezone
    assert not PreferenceORM.__table__.c.is_active.nullable


def test_postgresql_ddl_and_partial_unique_index():
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        assert "CREATE TABLE" in str(CreateTable(table).compile(dialect=dialect))
    indexes = {index.name: index for index in PreferenceORM.__table__.indexes}
    index = indexes["ix_preferences_one_active_per_profile"]
    assert index.unique
    assert str(CreateIndex(index).compile(dialect=dialect)) == (
        "CREATE UNIQUE INDEX ix_preferences_one_active_per_profile "
        "ON preferences (profile_id) WHERE is_active = true"
    )


def test_session_factory_is_synchronous_and_sessions_are_independent():
    assert engine.dialect.name == "postgresql"
    assert engine.dialect.driver == "psycopg"
    assert not engine.dialect.is_async
    with SessionLocal() as first, SessionLocal() as second:
        assert isinstance(first, Session) and isinstance(second, Session)
        assert first is not second
        assert first.get_bind() is engine


@pytest.mark.parametrize("value", [["not-a-number"], [[0.0]], {"value": 1.0}])
def test_malformed_vector_values_rejected(value):
    processor = JobEmbeddingORM.__table__.c.embedding.type.bind_processor(
        postgresql.dialect()
    )
    with pytest.raises((ValueError, TypeError)):
        processor(value)


def test_vector_serialization_preserves_768_dimensions():
    vector_type = JobEmbeddingORM.__table__.c.embedding.type
    dialect = postgresql.dialect()
    expected = [1.0] + [0.0] * 767
    serialized = vector_type.bind_processor(dialect)(expected)
    assert vector_type.result_processor(dialect, None)(serialized) == expected


def _new_preference(profile_id, version, active=True):
    return PreferenceORM(
        profile_id=profile_id,
        version=version,
        is_active=active,
        hard_constraints=HardConstraints(
            allowed_work_modes=[], allowed_cities=[]
        ).model_dump(),
        soft_preferences=SoftPreferences(
            target_roles=[], preferred_skills=[]
        ).model_dump(),
    )


def _check_preference_uniqueness(db, profile, other_profile):
    first = _new_preference(profile.id, 1)
    db.add(first)
    db.flush()
    with pytest.raises(IntegrityError) as error, db.begin_nested():
        db.add(_new_preference(profile.id, 2))
        db.flush()
    assert error.value.orig.sqlstate == "23505"
    assert (
        error.value.orig.diag.constraint_name == "ix_preferences_one_active_per_profile"
    )

    inactive = _new_preference(profile.id, 2, active=False)
    db.add_all(
        [
            inactive,
            _new_preference(profile.id, 3, active=False),
            _new_preference(other_profile.id, 1),
        ]
    )
    db.flush()
    with pytest.raises(IntegrityError) as error, db.begin_nested():
        inactive.is_active = True
        db.flush()
    assert (
        error.value.orig.diag.constraint_name == "ix_preferences_one_active_per_profile"
    )
    first.is_active = False
    db.flush()
    replacement = _new_preference(profile.id, 4)
    db.add(replacement)
    db.flush()
    active = db.scalars(
        select(PreferenceORM).where(
            PreferenceORM.profile_id == profile.id, PreferenceORM.is_active.is_(True)
        )
    ).all()
    assert active == [replacement]
    contract = PreferenceState(
        profile_id=profile.id,
        version=replacement.version,
        hard_constraints=replacement.hard_constraints,
        soft_preferences=replacement.soft_preferences,
        updated_at=replacement.created_at.isoformat(),
    )
    assert PreferenceState.model_validate_json(contract.model_dump_json()) == contract


def _check_round_trips(db, profile):
    job_id = f"test_job_{profile.id}"
    evidence = CandidateEvidenceORM(
        evidence_id=f"test_ev_{profile.id}",
        profile_id=profile.id,
        skill="Python",
        source_type="user_statement",
        verification_status="self_reported",
        source_excerpt="Synthetic statement — नमस्ते",
        source_id="test_statement",
    )
    untrusted_text = "Python internship'; DROP TABLE jobs; --"
    job = JobORM(
        job_id=job_id,
        source_url="https://example.test/job",
        title="AI Intern",
        company="Synthetic fixture",
        raw_description=untrusted_text,
        excerpts={f"{job_id}_ex_01": untrusted_text},
    )
    run = SearchRunORM(
        profile_id=profile.id,
        profile_version=1,
        preference_version=4,
        query="Python",
        system_variant="retrieval_only",
        status="completed",
    )
    db.add_all([evidence, job, run])
    db.flush()
    captured_at = datetime(2026, 11, 19, tzinfo=UTC)
    metadata = JobMetadataORM(
        job_id=job_id,
        captured_at=captured_at,
        content_hash="test_hash",
        dedup_key="test_key",
    )
    embedding = JobEmbeddingORM(
        job_id=job_id,
        model_id="gemini-embedding-2",
        content_hash="test_hash",
        embedding=[1.0] + [0.0] * 767,
    )
    trace = TraceORM(
        run_id=run.run_id,
        step_index=0,
        agent_name="test",
        tool_or_action="schema_check",
        outcome="ok",
    )
    db.add_all([metadata, embedding, trace])
    db.flush()
    db.expire_all()
    assert job.raw_description == untrusted_text
    assert job.excerpts == {f"{job_id}_ex_01": untrusted_text}
    assert job.norm_student_eligible is None
    assert job.norm_degree_completed_required is None
    assert job.norm_min_yoe is None
    assert job.norm_work_mode == "unknown" and job.norm_remote_region == "unknown"
    assert job.norm_cities == [] and job.constraint_excerpts == {}
    assert metadata.captured_at == captured_at
    assert metadata.source_status == "active" and not metadata.is_duplicate
    assert list(embedding.embedding) == [1.0] + [0.0] * 767
    assert embedding.dimension == 768
    assert (
        run.shortlist == []
        and run.needs_review_list == []
        and run.rejected_counterfactuals == []
    )
    assert (
        not run.used_mock_fallback
        and run.total_tokens == 0
        and run.estimated_cost_usd == 0
    )
    assert trace.payload_summary == {} and trace.step_index == 0
    assert profile.created_at.tzinfo is not None
    record = EvidenceRecord.model_validate(evidence, from_attributes=True)
    assert record.source_type.value == "user_statement"
    assert record.source_excerpt == evidence.source_excerpt
    job.norm_student_eligible = False
    job.norm_degree_completed_required = True
    job.norm_min_yoe = 0.0
    db.flush()
    db.expire(job)
    assert (
        job.norm_student_eligible is False
        and job.norm_degree_completed_required is True
    )
    assert job.norm_min_yoe == 0.0
    db.execute(
        text(
            "UPDATE jobs SET search_tsv = to_tsvector('english', :description) "
            "WHERE job_id = :job_id"
        ),
        {"description": "Python internship", "job_id": job_id},
    )
    assert (
        db.execute(
            text(
                "SELECT search_tsv @@ plainto_tsquery('english', :query) "
                "FROM jobs WHERE job_id = :job_id"
            ),
            {"query": "Python", "job_id": job_id},
        ).scalar_one()
        is True
    )
    return job, metadata, embedding, run, trace


def _check_invalid_writes(db, profile, job, metadata, embedding, run, trace):
    for size in (0, 767, 769):
        with pytest.raises(DataError) as error, db.begin_nested():
            embedding.embedding = [0.0] * size
            db.flush()
        assert error.value.orig.sqlstate.startswith("22")
    cases = [
        (profile, "profile_version", 0),
        (job, "norm_min_yoe", -1.0),
        (job, "norm_min_yoe", float("nan")),
        (job, "norm_min_yoe", float("inf")),
        (metadata, "source_status", "invalid"),
        (embedding, "dimension", 767),
        (run, "total_tokens", -1),
        (run, "estimated_cost_usd", -0.1),
        (trace, "step_index", -1),
        (trace, "duration_ms", -1),
        (trace, "tokens_used", -1),
    ]
    for row, field, value in cases:
        with pytest.raises(IntegrityError) as error, db.begin_nested():
            setattr(row, field, value)
            db.flush()
        assert error.value.orig.sqlstate == "23514"
    with pytest.raises(IntegrityError) as error, db.begin_nested():
        db.add(
            JobMetadataORM(
                job_id=f"missing_{profile.id}",
                captured_at=datetime.now(UTC),
                content_hash="test_hash",
                dedup_key="missing",
            )
        )
        db.flush()
    assert error.value.orig.sqlstate == "23503"


def test_live_postgresql_initialization_and_integrity():
    """The gate must fail if real PostgreSQL is unavailable."""
    init_db()
    init_db()  # Repeated startup must leave existing tables intact.
    inspector = inspect(engine)
    assert set(EXPECTED_COLUMNS).issubset(inspector.get_table_names())
    for name, fields in EXPECTED_COLUMNS.items():
        assert fields.issubset(
            {column["name"] for column in inspector.get_columns(name)}
        )
    indexes = {index["name"]: index for index in inspector.get_indexes("preferences")}
    index = indexes["ix_preferences_one_active_per_profile"]
    assert index["unique"] and index["column_names"] == ["profile_id"]
    assert "is_active = true" in index["dialect_options"]["postgresql_where"]

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            version = int(connection.scalar(text("SHOW server_version_num")))
            assert 160000 <= version < 170000
            assert connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = :name"),
                {"name": "vector"},
            )
            with SessionLocal(
                bind=connection, join_transaction_mode="create_savepoint"
            ) as db:
                profile = CandidateProfileORM(name="Schema gate fixture")
                other_profile = CandidateProfileORM(
                    name="Independent schema gate fixture"
                )
                db.add_all([profile, other_profile])
                db.flush()
                _check_preference_uniqueness(db, profile, other_profile)
                rows = _check_round_trips(db, profile)
                _check_invalid_writes(db, profile, *rows)
                fixture_ids = [profile.id, other_profile.id]
        finally:
            transaction.rollback()
        assert (
            connection.execute(
                select(CandidateProfileORM.id).where(
                    CandidateProfileORM.id.in_(fixture_ids)
                )
            ).all()
            == []
        )
