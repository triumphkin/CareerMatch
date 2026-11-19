"""Tests for Discovery Agent node, bounded searches, locked hard constraints, and trace logging.

Verification Gate (Step 2.2):
uv run pytest tests/test_discovery_agent.py confirms max 2 searches, locked hard constraints, and trace logging.
"""

import json
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.agents.discovery import (
    DISCOVERY_SYSTEM_INSTRUCTION,
    discovery_node,
    plan_discovery_queries,
    run_discovery,
)
from app.config import settings
from app.db.models import (
    Base,
    CandidateProfileORM,
    JobMetadataORM,
    JobORM,
    PreferenceORM,
    SearchRunORM,
    TraceORM,
)
from app.db.session import engine
from app.schemas.contracts import (
    DiscoveryQueryPlan,
    HardConstraints,
    LLMUsageMeta,
    SoftPreferences,
    WorkMode,
)
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts
from app.services.llm_client import GeminiService

# ---------------------------------------------------------------------------
# 1. Pydantic Contract Tests: DiscoveryQueryPlan
# ---------------------------------------------------------------------------


def test_discovery_query_plan_contract():
    plan = DiscoveryQueryPlan(
        primary_query="python backend engineer",
        fallback_broad_query="software developer",
        rationale="Targeting backend roles first, falling back to general software engineering.",
    )
    assert plan.primary_query == "python backend engineer"
    assert plan.fallback_broad_query == "software developer"
    assert "backend" in plan.rationale

    data = plan.model_dump(mode="json")
    assert json.loads(json.dumps(data)) == data

    # ConfigDict(extra="forbid") must reject extra fields
    with pytest.raises(ValidationError):
        DiscoveryQueryPlan(
            primary_query="python",
            fallback_broad_query="developer",
            rationale="testing",
            extra_field="disallowed",
        )


def test_discovery_query_plan_missing_fields():
    with pytest.raises(ValidationError):
        DiscoveryQueryPlan(
            primary_query="python", fallback_broad_query="developer"
        )  # missing rationale


# ---------------------------------------------------------------------------
# 2. Query Planning Unit Tests
# ---------------------------------------------------------------------------


def test_plan_discovery_queries_offline_mock():
    offline_config = settings.model_copy(update={"GEMINI_API_KEY": ""})
    with GeminiService(config=offline_config) as service:
        plan, usage = plan_discovery_queries(
            user_query="Senior Python Developer",
            soft_preferences=SoftPreferences(
                target_roles=["Backend Engineer"],
                preferred_skills=["FastAPI", "PostgreSQL"],
                prefer_startup=True,
            ),
            llm_service=service,
        )
        assert isinstance(plan, DiscoveryQueryPlan)
        assert plan.primary_query
        assert plan.fallback_broad_query
        assert plan.rationale
        assert isinstance(usage, LLMUsageMeta)
        assert usage.used_mock_fallback is True


def test_plan_discovery_queries_uses_fast_model_and_low_thinking():
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="fastapi engineer",
            fallback_broad_query="python developer",
            rationale="test rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    plan, usage = plan_discovery_queries(
        user_query="FastAPI",
        llm_service=mock_service,
    )

    assert plan.primary_query == "fastapi engineer"
    assert usage.model_id == settings.CHAT_MODEL_ID_FAST
    mock_service.generate_structured.assert_called_once()
    _, kwargs = mock_service.generate_structured.call_args
    assert kwargs["model_id"] == settings.CHAT_MODEL_ID_FAST
    assert kwargs["thinking_level"] == "LOW"
    assert kwargs["response_schema"] == DiscoveryQueryPlan
    assert kwargs["system_instruction"] == DISCOVERY_SYSTEM_INSTRUCTION
    assert "FastAPI" in kwargs["prompt"]


def test_plan_discovery_queries_validation():
    with pytest.raises(TypeError):
        plan_discovery_queries(123)  # type: ignore

    with pytest.raises(ValueError, match="NUL-free"):
        plan_discovery_queries("python\x00dev")

    with pytest.raises(TypeError):
        plan_discovery_queries("python", soft_preferences="invalid")  # type: ignore


# ---------------------------------------------------------------------------
# 3. Bounded Searches, Locked Constraints & Candidate Selection Unit Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def sample_hard_constraints():
    return HardConstraints(
        allowed_work_modes=[WorkMode.REMOTE, WorkMode.HYBRID],
        allowed_cities=["Bengaluru"],
        max_required_yoe=3.0,
    )


def test_run_discovery_single_search_when_eligible_at_least_5(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="primary query",
            fallback_broad_query="fallback query",
            rationale="primary first",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=True),
    )

    # 5 eligible candidates from primary search
    search_result_1 = {
        "eligible_candidates": [{"job_id": f"job_{i:03d}"} for i in range(1, 6)],
        "rejected_counterfactuals": [{"job_id": "job_099", "reason": "yoe"}],
        "candidates_evaluated": 15,
        "used_mock_fallback": True,
        "embedding_calls": 1,
    }

    with patch(
        "app.agents.discovery.search_jobs", return_value=search_result_1
    ) as mock_search:
        result = run_discovery(
            db=mock_db,
            user_query="python developer",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert mock_search.call_count == 1
        assert mock_search.call_args_list[0] == call(
            mock_db,
            query="primary query",
            hard=sample_hard_constraints,
            limit=10,
            run_id=None,
        )
        assert result["searches_used"] == 1
        assert result["candidate_job_ids"] == [f"job_{i:03d}" for i in range(1, 6)]
        assert len(result["rejected_counterfactuals"]) == 1
        assert result["discovery_rationale"] == "primary first"
        assert result["used_mock_fallback"] is True


def test_run_discovery_two_searches_when_eligible_under_5(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="narrow query",
            fallback_broad_query="broad query",
            rationale="broaden if narrow fails",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    # Primary returns only 2 jobs (< 5)
    search_result_1 = {
        "eligible_candidates": [{"job_id": "job_001"}, {"job_id": "job_002"}],
        "rejected_counterfactuals": [{"job_id": "job_091"}],
        "candidates_evaluated": 5,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    # Fallback returns 4 jobs (one overlapping job_002, three new: job_003, job_004, job_005)
    search_result_2 = {
        "eligible_candidates": [
            {"job_id": "job_002"},
            {"job_id": "job_003"},
            {"job_id": "job_004"},
            {"job_id": "job_005"},
        ],
        "rejected_counterfactuals": [{"job_id": "job_092"}],
        "candidates_evaluated": 10,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch(
        "app.agents.discovery.search_jobs",
        side_effect=[search_result_1, search_result_2],
    ) as mock_search:
        result = run_discovery(
            db=mock_db,
            user_query="niche role",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert mock_search.call_count == 2
        # Check call 1
        assert mock_search.call_args_list[0] == call(
            mock_db,
            query="narrow query",
            hard=sample_hard_constraints,
            limit=10,
            run_id=None,
        )
        # Check call 2 (LOCKED HARD CONSTRAINTS: exact same hard constraints object)
        assert mock_search.call_args_list[1] == call(
            mock_db,
            query="broad query",
            hard=sample_hard_constraints,
            limit=10,
            run_id=None,
        )

        assert result["searches_used"] == 2
        # Deduplicated merged candidate IDs: job_001, job_002, job_003, job_004, job_005
        assert result["candidate_job_ids"] == [
            "job_001",
            "job_002",
            "job_003",
            "job_004",
            "job_005",
        ]
        assert len(result["candidate_job_ids"]) == 5
        assert len(result["rejected_counterfactuals"]) == 2
        assert result["used_mock_fallback"] is False


def test_run_discovery_locked_hard_constraints(mock_db, sample_hard_constraints):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="q1",
            fallback_broad_query="q2",
            rationale="rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    empty_result = {
        "eligible_candidates": [],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 0,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch(
        "app.agents.discovery.search_jobs", return_value=empty_result
    ) as mock_search:
        run_discovery(
            db=mock_db,
            user_query="test",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert mock_search.call_count == 2
        call_hard_1 = mock_search.call_args_list[0].kwargs["hard"]
        call_hard_2 = mock_search.call_args_list[1].kwargs["hard"]

        # Hard constraints must be identical and unchanged
        assert call_hard_1.model_dump() == sample_hard_constraints.model_dump()
        assert call_hard_2.model_dump() == sample_hard_constraints.model_dump()
        assert (
            call_hard_1.allowed_work_modes
            == sample_hard_constraints.allowed_work_modes
        )
        assert call_hard_1.allowed_cities == sample_hard_constraints.allowed_cities
        assert (
            call_hard_1.max_required_yoe == sample_hard_constraints.max_required_yoe
        )


def test_run_discovery_caps_candidate_pool_at_10(mock_db, sample_hard_constraints):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="q1",
            fallback_broad_query="q2",
            rationale="rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    # Primary returns 3 (< 5)
    search_1 = {
        "eligible_candidates": [{"job_id": f"job_{i:03d}"} for i in range(1, 4)],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 3,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }
    # Fallback returns 10 new jobs
    search_2 = {
        "eligible_candidates": [{"job_id": f"job_{i:03d}"} for i in range(4, 14)],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 10,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch("app.agents.discovery.search_jobs", side_effect=[search_1, search_2]):
        result = run_discovery(
            db=mock_db,
            user_query="test",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert len(result["candidate_job_ids"]) == 10
        assert result["candidate_job_ids"] == [f"job_{i:03d}" for i in range(1, 11)]


def test_run_discovery_pure_dict_state_and_json_serialization(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="q1",
            fallback_broad_query="q2",
            rationale="rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    search_1 = {
        "eligible_candidates": [{"job_id": "job_001"}],
        "rejected_counterfactuals": [
            {
                "job_id": "job_090",
                "title": "Lead",
                "company": "Corp",
                "similarity_score": 0.85,
                "failed_constraints": [],
                "rejection_summary": "YOE exceeded",
            }
        ],
        "candidates_evaluated": 10,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch("app.agents.discovery.search_jobs", return_value=search_1):
        result = run_discovery(
            db=mock_db,
            user_query="test",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        # Must be completely JSON-serializable pure dict
        dumped = json.dumps(result)
        loaded = json.loads(dumped)

        assert isinstance(loaded["candidate_job_ids"], list)
        assert isinstance(loaded["rejected_counterfactuals"], list)
        assert isinstance(loaded["discovery_rationale"], str)
        assert isinstance(loaded["discovery_query_plan"], dict)
        assert isinstance(loaded["used_mock_fallback"], bool)
        assert isinstance(loaded["searches_used"], int)


def test_run_discovery_trace_logging(mock_db, sample_hard_constraints):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="primary",
            fallback_broad_query="fallback",
            rationale="rationale",
        ),
        LLMUsageMeta(
            model_id=settings.CHAT_MODEL_ID_FAST,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            used_mock_fallback=False,
            cached=False,
        ),
    )

    search_result = {
        "eligible_candidates": [{"job_id": f"job_{i:03d}"} for i in range(1, 6)],
        "rejected_counterfactuals": [{"job_id": "job_099"}],
        "candidates_evaluated": 15,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch("app.agents.discovery.search_jobs", return_value=search_result), patch(
        "app.agents.discovery.log_trace"
    ) as mock_log_trace:
        run_discovery(
            db=mock_db,
            user_query="python",
            hard_constraints=sample_hard_constraints,
            run_id="run_test_123",
            llm_service=mock_service,
        )

        # log_trace must be called for plan_queries and select_candidates
        assert mock_log_trace.call_count == 2

        plan_call = mock_log_trace.call_args_list[0]
        assert plan_call.args == (mock_db, "run_test_123")
        assert plan_call.kwargs["agent_name"] == "discovery"
        assert plan_call.kwargs["tool_or_action"] == "plan_queries"
        assert plan_call.kwargs["outcome"] == "success"
        assert plan_call.kwargs["tokens_used"] == 70
        assert plan_call.kwargs["payload_summary"] == {
            "model_calls": 1,
            "used_mock_fallback": False,
            "cached": False,
        }

        select_call = mock_log_trace.call_args_list[1]
        assert select_call.args == (mock_db, "run_test_123")
        assert select_call.kwargs["agent_name"] == "discovery"
        assert select_call.kwargs["tool_or_action"] == "select_candidates"
        assert select_call.kwargs["outcome"] == "success"
        assert select_call.kwargs["payload_summary"] == {
            "result_count": 5,
            "rejected_count": 1,
            "candidates_evaluated": 15,
            "used_mock_fallback": False,
            "job_ids": [f"job_{i:03d}" for i in range(1, 6)],
        }


def test_discovery_node_from_state_and_config(mock_db, sample_hard_constraints):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="q1",
            fallback_broad_query="q2",
            rationale="rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    search_result = {
        "eligible_candidates": [{"job_id": "job_001"}, {"job_id": "job_002"}],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 5,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    state = {
        "user_query": "backend engineer",
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
        "soft_preferences": {"target_roles": ["Backend Developer"]},
        "run_id": "run_node_001",
    }
    config = {"configurable": {"db": mock_db}}

    with patch("app.agents.discovery.search_jobs", return_value=search_result), patch(
        "app.agents.discovery.log_trace"
    ):
        result = discovery_node(state, config=config, llm_service=mock_service)
        assert result["candidate_job_ids"] == ["job_001", "job_002"]
        assert result["searches_used"] == 2


def test_discovery_node_missing_db():
    state = {"user_query": "backend engineer"}
    with pytest.raises(ValueError, match="database Session is required"):
        discovery_node(state)


def test_discovery_node_resolves_query_from_target_roles(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="devops",
            fallback_broad_query="sre",
            rationale="infra",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    state = {
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
        "soft_preferences": {"target_roles": ["DevOps Engineer", "SRE"]},
    }

    with patch(
        "app.agents.discovery.search_jobs", return_value={"eligible_candidates": []}
    ):
        result = discovery_node(state, db=mock_db, llm_service=mock_service)
        assert result["discovery_rationale"] == "infra"


def test_run_discovery_empty_query_without_target_roles_raises(
    mock_db, sample_hard_constraints
):
    with pytest.raises(ValueError, match="nonblank"):
        run_discovery(
            db=mock_db,
            user_query="   ",
            hard_constraints=sample_hard_constraints,
        )


def test_run_discovery_zero_candidates_both_searches(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="obscure role",
            fallback_broad_query="broad obscure role",
            rationale="both might return 0",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    empty_search = {
        "eligible_candidates": [],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 0,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch(
        "app.agents.discovery.search_jobs", return_value=empty_search
    ) as mock_search:
        result = run_discovery(
            db=mock_db,
            user_query="obscure role",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert mock_search.call_count == 2
        assert result["candidate_job_ids"] == []
        assert result["rejected_counterfactuals"] == []
        assert result["searches_used"] == 2


def test_run_discovery_respects_max_discovery_searches_setting(
    mock_db, sample_hard_constraints, monkeypatch
):
    monkeypatch.setattr(settings, "MAX_DISCOVERY_SEARCHES", 1)

    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="q1",
            fallback_broad_query="q2",
            rationale="rationale",
        ),
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, used_mock_fallback=False),
    )

    # Returns 2 jobs (< 5), but max searches is 1
    search_result = {
        "eligible_candidates": [{"job_id": "job_001"}, {"job_id": "job_002"}],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 2,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with patch(
        "app.agents.discovery.search_jobs", return_value=search_result
    ) as mock_search:
        result = run_discovery(
            db=mock_db,
            user_query="test",
            hard_constraints=sample_hard_constraints,
            llm_service=mock_service,
        )

        assert mock_search.call_count == 1
        assert result["searches_used"] == 1
        assert result["candidate_job_ids"] == ["job_001", "job_002"]


def test_run_discovery_invalid_hard_constraints_type(mock_db):
    with pytest.raises(TypeError, match="hard_constraints must be"):
        run_discovery(
            db=mock_db,
            user_query="test",
            hard_constraints="not-a-constraint",  # type: ignore
        )


def test_run_discovery_plan_error_logs_trace_and_reraises(
    mock_db, sample_hard_constraints
):
    mock_service = MagicMock(spec=GeminiService)
    mock_service.generate_structured.side_effect = RuntimeError("Provider offline")

    with patch("app.agents.discovery.log_trace") as mock_log_trace:
        with pytest.raises(RuntimeError, match="Provider offline"):
            run_discovery(
                db=mock_db,
                user_query="test",
                hard_constraints=sample_hard_constraints,
                run_id="run_err_001",
                llm_service=mock_service,
            )

        mock_log_trace.assert_called_once()
        assert mock_log_trace.call_args.kwargs["agent_name"] == "discovery"
        assert mock_log_trace.call_args.kwargs["tool_or_action"] == "plan_queries"
        assert mock_log_trace.call_args.kwargs["outcome"] == "error"


# ---------------------------------------------------------------------------
# 4. Real PostgreSQL Integration Tests (Skipped if Postgres is unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Connects to real PostgreSQL and creates an isolated test schema."""
    try:
        connection = engine.connect()
    except OperationalError:
        pytest.fail(
            "Real PostgreSQL is unavailable; no SQLite/mock DB fallback is used.",
            pytrace=False,
        )
    with connection:
        transaction = connection.begin()
        try:
            assert (
                160000
                <= int(connection.scalar(text("SHOW server_version_num")))
                < 170000
            )
            assert connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
            schema = f"discovery_test_{uuid4().hex}"
            connection.execute(CreateSchema(schema))
            scoped = connection.execution_options(schema_translate_map={None: schema})
            Base.metadata.create_all(scoped)
            with Session(
                bind=scoped,
                autoflush=False,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            ) as session:
                yield session
        finally:
            transaction.rollback()


@pytest.fixture
def seeded_db(db):
    """Seed real database with profile, preferences, search run, and jobs."""
    profile = CandidateProfileORM(
        id="profile_disc_1", name="Candidate Disc", profile_version=1
    )
    db.add(profile)
    db.flush()

    hard = HardConstraints(allowed_work_modes=[WorkMode.REMOTE], allowed_cities=[])
    soft = SoftPreferences(
        target_roles=["Python Developer"], preferred_skills=["FastAPI"]
    )
    pref = PreferenceORM(
        profile_id="profile_disc_1",
        version=1,
        is_active=True,
        hard_constraints=hard.model_dump(mode="json"),
        soft_preferences=soft.model_dump(mode="json"),
        update_note="Initial preferences",
    )
    db.add(pref)

    run = SearchRunORM(
        run_id="run_disc_001",
        profile_id="profile_disc_1",
        profile_version=1,
        preference_version=1,
        query="python remote developer",
        system_variant="multi_agent",
        status="running",
    )
    db.add(run)
    db.flush()

    for i in range(1, 4):
        job_id = f"job_{i:03d}"
        raw = f"Remote from India. Python developer role {i}. 0 years experience required. Students eligible."
        excerpts = slice_into_excerpts(job_id, raw)
        norm = normalize_job_constraints(
            {"job_id": job_id, "raw_description": raw}, excerpts
        )
        norm.pop("has_conflict", None)
        job = JobORM(
            job_id=job_id,
            source_url=f"https://example.invalid/{job_id}",
            title=f"Python Developer {i}",
            company="TechCorp",
            raw_description=raw,
            excerpts=excerpts,
            **norm,
        )
        db.add(job)
        db.add(
            JobMetadataORM(
                job_id=job_id,
                captured_at=run.created_at,
                content_hash=f"hash_{i}",
                source_status="active",
                dedup_key=job_id,
                is_duplicate=False,
            )
        )
    db.flush()
    return db


def test_postgresql_discovery_agent_end_to_end(seeded_db):
    state = {
        "profile_id": "profile_disc_1",
        "run_id": "run_disc_001",
        "user_query": "Python Developer",
    }
    result = discovery_node(state, db=seeded_db)

    assert "candidate_job_ids" in result
    assert "discovery_rationale" in result
    assert result["searches_used"] in (1, 2)

    # Verify traces were actually inserted in the real database
    traces = (
        seeded_db.query(TraceORM)
        .filter(TraceORM.run_id == "run_disc_001")
        .order_by(TraceORM.step_index.asc())
        .all()
    )
    assert len(traces) >= 2
    actions = [t.tool_or_action for t in traces]
    assert "plan_queries" in actions
    assert "select_candidates" in actions
