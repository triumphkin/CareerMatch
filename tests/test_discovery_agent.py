"""Tests for Discovery Agent node, bounded searches, locked hard constraints, and trace logging.

Verification Gate (Step 2.2):
uv run pytest tests/test_discovery_agent.py confirms max 2 searches, locked hard constraints, and trace logging.
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
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
from app.services import retrieval
from app.services.constraints import evaluate_job_constraints
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts
from app.services.llm_client import GeminiResponseError, GeminiService


def candidate(job_id):
    return {"job_id": job_id, "constraint_result": "pass"}


def rejection(job_id):
    return {
        "job_id": job_id,
        "title": "Senior role",
        "company": "Example",
        "similarity_score": 0.01,
        "failed_constraints": [
            {
                "field": "experience_limit",
                "result": "fail",
                "reason": "Too much required experience",
                "source_excerpt_ids": [],
            }
        ],
        "rejection_summary": "Experience limit exceeded",
    }


def search_output(
    ids=(), rejected=(), *, mock=False, embedding_calls=0, embedding_usage=None
):
    return {
        "eligible_candidates": [candidate(job_id) for job_id in ids],
        "rejected_counterfactuals": [rejection(job_id) for job_id in rejected],
        "candidates_evaluated": len(ids) + len(rejected),
        "used_mock_fallback": mock,
        "embedding_calls": embedding_calls,
        "embedding_usage": embedding_usage,
    }


@pytest.fixture
def planner():
    service = MagicMock(spec=GeminiService)
    service.generate_structured.return_value = (
        DiscoveryQueryPlan(
            primary_query="Python",
            fallback_broad_query="developer",
            rationale="Broaden title",
        ),
        LLMUsageMeta(
            model_id=settings.CHAT_MODEL_ID_FAST,
            total_tokens=8,
            estimated_cost_usd=0.01,
        ),
    )
    return service


def test_regression_zero_search_budget_has_no_model_or_search_calls(
    mock_db, sample_hard_constraints, planner, monkeypatch
):
    monkeypatch.setattr(settings, "MAX_DISCOVERY_SEARCHES", 0)
    with (
        patch("app.agents.discovery.search_jobs") as search,
        pytest.raises(ValueError, match="budget"),
    ):
        run_discovery(mock_db, "Python", sample_hard_constraints, llm_service=planner)
    planner.generate_structured.assert_not_called()
    search.assert_not_called()


def test_regression_primary_duplicates_do_not_suppress_fallback(
    mock_db, sample_hard_constraints, planner
):
    with patch(
        "app.agents.discovery.search_jobs",
        side_effect=[
            search_output(["job_001"] * 5, ["job_099"] * 2),
            search_output(["job_001", "job_002"], ["job_099"]),
        ],
    ) as search:
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert search.call_count == 2
    assert result["candidate_job_ids"] == ["job_001", "job_002"]
    assert len(result["rejected_counterfactuals"]) == 1


def test_regression_usage_is_preserved(mock_db, sample_hard_constraints, planner):
    usage = LLMUsageMeta(
        model_id=settings.EMBEDDING_MODEL_ID, total_tokens=3, estimated_cost_usd=0.02
    ).model_dump(mode="json")
    with patch(
        "app.agents.discovery.search_jobs",
        return_value=search_output(
            ["job_001"], embedding_calls=1, embedding_usage=usage
        ),
    ):
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert result["model_calls"] == 1
    assert result["planning_usage"]["total_tokens"] == 8
    assert result["embedding_calls"] == 2
    assert result["embedding_usage"] == [usage, usage]


def test_regression_database_session_is_not_allowed_in_graph_state(
    mock_db, sample_hard_constraints, planner
):
    state = {
        "user_query": "Python",
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
        "_db": mock_db,
    }
    with (
        patch("app.agents.discovery.search_jobs", return_value=search_output()),
        pytest.raises(ValueError, match="state"),
    ):
        discovery_node(state, llm_service=planner)
    planner.generate_structured.assert_not_called()


def test_regression_empty_query_uses_persisted_target_roles(
    mock_db, sample_hard_constraints, planner
):
    pref = {
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
        "soft_preferences": {
            "target_roles": ["Python Developer"],
            "preferred_skills": [],
        },
        "version": 1,
    }
    with (
        patch("app.agents.discovery.get_preferences", return_value=pref) as preferences,
        patch("app.agents.discovery.search_jobs", return_value=search_output()),
    ):
        discovery_node(
            {"profile_id": "profile_1", "preference_version": 1},
            db=mock_db,
            llm_service=planner,
        )
    preferences.assert_called_once_with(mock_db, "profile_1", version=1, run_id=None)
    assert "Python Developer" in planner.generate_structured.call_args.kwargs["prompt"]


def test_regression_node_preserves_prior_mock_provenance(
    mock_db, sample_hard_constraints, planner
):
    state = {
        "query": "Python",
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
        "used_mock_fallback": True,
    }
    with patch("app.agents.discovery.search_jobs", return_value=search_output()):
        result = discovery_node(state, db=mock_db, llm_service=planner)
    assert result["used_mock_fallback"] is True


@pytest.mark.parametrize("alias", ["run_id", "search_run_id"])
def test_run_without_profile_still_rejects_changed_hard_constraints(
    mock_db, planner, alias
):
    mock_db.execute.return_value.one.return_value = SimpleNamespace(
        profile_id="profile_1", preference_version=1
    )
    persisted = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    relaxed = persisted.model_copy(update={"max_required_yoe": 20.0})
    preferences = {
        "version": 1,
        "hard_constraints": persisted.model_dump(mode="json"),
        "soft_preferences": {"target_roles": ["Python"], "preferred_skills": []},
    }
    with (
        patch("app.agents.discovery.get_preferences", return_value=preferences),
        patch(
            "app.agents.discovery.search_jobs", return_value=search_output()
        ) as search,
        patch("app.agents.discovery.log_trace"),
        pytest.raises(ValueError, match="disagree"),
    ):
        discovery_node(
            {
                alias: "run_1",
                "query": "Python",
                "hard_constraints": relaxed.model_dump(mode="json"),
            },
            db=mock_db,
            llm_service=planner,
        )
    planner.generate_structured.assert_not_called()
    search.assert_not_called()


@pytest.mark.parametrize("profile", [None, "profile_1"])
def test_run_resolves_profile_and_historical_preferences_without_query(
    mock_db, planner, profile
):
    mock_db.execute.return_value.one.return_value = SimpleNamespace(
        profile_id="profile_1", preference_version=3
    )
    hard = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    preferences = {
        "version": 3,
        "hard_constraints": hard.model_dump(mode="json"),
        "soft_preferences": {
            "target_roles": ["Historical Python Role"],
            "preferred_skills": [],
        },
    }
    state = {"run_id": "run_1", "profile_id": profile}
    before = json.loads(json.dumps(state))
    with (
        patch("app.agents.discovery.get_preferences", return_value=preferences) as read,
        patch(
            "app.agents.discovery.search_jobs", return_value=search_output()
        ) as search,
        patch("app.agents.discovery.log_trace"),
    ):
        result = discovery_node(state, db=mock_db, llm_service=planner)
    read.assert_called_once_with(mock_db, "profile_1", version=3, run_id="run_1")
    assert (
        "Historical Python Role"
        in planner.generate_structured.call_args.kwargs["prompt"]
    )
    assert all(item.kwargs["hard"] == hard for item in search.call_args_list)
    assert result["profile_id"] == "profile_1" and result["preference_version"] == 3
    assert state == before


@pytest.mark.parametrize(
    "extra", [{"preference_version": 2}, {"profile_id": "another_profile"}]
)
def test_run_mismatches_fail_before_preferences_or_provider(mock_db, planner, extra):
    mock_db.execute.return_value.one.return_value = SimpleNamespace(
        profile_id="profile_1", preference_version=1
    )
    with (
        patch("app.agents.discovery.get_preferences") as preferences,
        pytest.raises(ValueError, match="disagrees"),
    ):
        discovery_node({"run_id": "run_1", **extra}, db=mock_db, llm_service=planner)
    preferences.assert_not_called()
    planner.generate_structured.assert_not_called()


@pytest.mark.parametrize(
    "field", ["primary_query", "fallback_broad_query", "rationale"]
)
@pytest.mark.parametrize("value", ["", "  ", "bad\x00text"])
def test_malformed_plan_fails_before_search(
    mock_db, sample_hard_constraints, planner, field, value
):
    plan, usage = planner.generate_structured.return_value
    planner.generate_structured.return_value = (
        plan.model_copy(update={field: value}),
        usage,
    )
    with (
        patch("app.agents.discovery.search_jobs") as search,
        pytest.raises(GeminiResponseError),
    ):
        run_discovery(mock_db, "Python", sample_hard_constraints, llm_service=planner)
    search.assert_not_called()


@pytest.mark.parametrize(
    "soft",
    [
        {"target_roles": ["   "]},
        {"target_roles": ["Python\x00"]},
        {"preferred_skills": [None]},
        {"preferred_skill": ["Python"]},
        SoftPreferences(target_roles=[], preferred_skills=[]).model_copy(
            update={"target_roles": [False]}
        ),
    ],
)
def test_invalid_soft_preferences_rejected_before_provider(planner, soft):
    with pytest.raises((TypeError, ValueError)):
        plan_discovery_queries("Python", soft, llm_service=planner)
    planner.generate_structured.assert_not_called()


def test_searches_receive_independent_copies_of_locked_constraints(
    mock_db, sample_hard_constraints, planner
):
    original = sample_hard_constraints.model_dump(mode="json")
    received = []

    def search(db, *, query, hard, limit, run_id):
        received.append(hard.model_dump(mode="json"))
        hard.allowed_cities.clear()
        hard.max_required_yoe = 999
        return search_output()

    with patch("app.agents.discovery.search_jobs", side_effect=search):
        run_discovery(mock_db, "Python", sample_hard_constraints, llm_service=planner)
    assert received == [original, original]
    assert sample_hard_constraints.model_dump(mode="json") == original


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"eligible_candidates": []},
        {**search_output(), "used_mock_fallback": "false"},
        {**search_output(), "embedding_calls": -1},
        {**search_output(), "candidates_evaluated": True},
        {**search_output(), "eligible_candidates": [candidate("not_a_corpus_id")]},
        {
            **search_output(),
            "eligible_candidates": [{"job_id": "job_001", "constraint_result": "fail"}],
        },
        {**search_output(), "eligible_candidates": None},
    ],
)
def test_malformed_retrieval_fails_closed(
    mock_db, sample_hard_constraints, planner, result
):
    with (
        patch("app.agents.discovery.search_jobs", return_value=result),
        pytest.raises((TypeError, ValueError)),
    ):
        run_discovery(mock_db, "Python", sample_hard_constraints, llm_service=planner)


def test_rejection_wins_over_eligibility_across_searches(
    mock_db, sample_hard_constraints, planner
):
    with patch(
        "app.agents.discovery.search_jobs",
        side_effect=[
            search_output(["job_001"], ["job_002"]),
            search_output(["job_002", "job_003"], ["job_001"]),
        ],
    ):
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert result["candidate_job_ids"] == ["job_003"]
    assert [item["job_id"] for item in result["rejected_counterfactuals"]] == [
        "job_002",
        "job_001",
    ]


def test_embedding_provenance_and_unknown_usage_are_not_lost(
    mock_db, sample_hard_constraints, planner
):
    mock_usage = LLMUsageMeta(
        model_id=settings.EMBEDDING_MODEL_ID, used_mock_fallback=True
    ).model_dump(mode="json")
    with patch(
        "app.agents.discovery.search_jobs",
        side_effect=[
            search_output(embedding_calls=1, embedding_usage=mock_usage),
            search_output(embedding_calls=1, embedding_usage=None),
        ],
    ):
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert result["used_mock_fallback"] is True
    assert result["embedding_usage"] == [mock_usage, None]


def test_cached_plan_usage_is_preserved(mock_db, sample_hard_constraints, planner):
    plan, _ = planner.generate_structured.return_value
    planner.generate_structured.return_value = (
        plan,
        LLMUsageMeta(model_id=settings.CHAT_MODEL_ID_FAST, cached=True),
    )
    with patch("app.agents.discovery.search_jobs", return_value=search_output()):
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert result["planning_usage"]["cached"] is True
    assert result["planning_usage"]["estimated_cost_usd"] == 0


@pytest.mark.parametrize("budget", [1, 2, 99])
def test_search_count_and_small_candidate_pool_bound(
    mock_db, sample_hard_constraints, planner, monkeypatch, budget
):
    monkeypatch.setattr(settings, "MAX_DISCOVERY_SEARCHES", budget)
    monkeypatch.setattr(settings, "MAX_CANDIDATE_POOL", 2)
    with patch(
        "app.agents.discovery.search_jobs",
        return_value=search_output(["job_001", "job_002"]),
    ) as search:
        result = run_discovery(
            mock_db, "Python", sample_hard_constraints, llm_service=planner
        )
    assert search.call_count == min(budget, 2)
    assert all(item.kwargs["limit"] == 2 for item in search.call_args_list)
    assert result["candidate_job_ids"] == ["job_001", "job_002"]


def test_zero_model_budget_prevents_planning(
    mock_db, sample_hard_constraints, planner, monkeypatch
):
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_PER_RUN", 0)
    with pytest.raises(ValueError, match="budget"):
        run_discovery(mock_db, "Python", sample_hard_constraints, llm_service=planner)
    planner.generate_structured.assert_not_called()


def test_missing_trace_run_fails_before_spending_api_calls(
    mock_db, sample_hard_constraints, planner
):
    mock_db.scalar.return_value = None
    with pytest.raises(LookupError):
        run_discovery(
            mock_db,
            "Python",
            sample_hard_constraints,
            run_id="missing",
            llm_service=planner,
        )
    planner.generate_structured.assert_not_called()


@pytest.mark.parametrize("query", [False, 0, None])
def test_invalid_explicit_query_does_not_fall_through_to_alias(
    mock_db, sample_hard_constraints, planner, query
):
    state = {
        "user_query": query,
        "query": "Python",
        "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
    }
    with pytest.raises(TypeError):
        discovery_node(state, db=mock_db, llm_service=planner)
    planner.generate_structured.assert_not_called()


@pytest.mark.parametrize(
    "config", [False, [], {"configurable": None}, {"configurable": []}]
)
def test_malformed_runtime_config_is_explicit(
    mock_db, sample_hard_constraints, planner, config
):
    with pytest.raises(TypeError, match="dictionary"):
        discovery_node(
            {
                "query": "Python",
                "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
            },
            config=config,
            db=mock_db,
            llm_service=planner,
        )
    planner.generate_structured.assert_not_called()


def test_node_rejects_pydantic_objects_in_state(
    mock_db, sample_hard_constraints, planner
):
    with pytest.raises(ValueError, match="JSON-compatible"):
        discovery_node(
            {"query": "Python", "hard_constraints": sample_hard_constraints},
            db=mock_db,
            llm_service=planner,
        )


def test_search_failure_propagates_without_reporting_selection_success(
    mock_db, sample_hard_constraints, planner
):
    with (
        patch(
            "app.agents.discovery.search_jobs",
            side_effect=RuntimeError("search failed"),
        ),
        patch("app.agents.discovery.log_trace") as trace,
        pytest.raises(RuntimeError, match="search failed"),
    ):
        run_discovery(
            mock_db,
            "Python",
            sample_hard_constraints,
            run_id="run_1",
            llm_service=planner,
        )
    assert [item.kwargs["tool_or_action"] for item in trace.call_args_list] == [
        "plan_queries"
    ]


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


def test_plan_discovery_queries_offline_mock(tmp_path):
    offline_config = settings.model_copy(
        update={
            "GEMINI_API_KEY": "",
            "LLM_CACHE_PATH": str(tmp_path / "offline.sqlite"),
        }
    )
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
        "eligible_candidates": [candidate(f"job_{i:03d}") for i in range(1, 6)],
        "rejected_counterfactuals": [rejection("job_099")],
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
        "eligible_candidates": [candidate("job_001"), candidate("job_002")],
        "rejected_counterfactuals": [rejection("job_091")],
        "candidates_evaluated": 5,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    # Fallback returns 4 jobs (one overlapping job_002, three new: job_003, job_004, job_005)
    search_result_2 = {
        "eligible_candidates": [
            candidate("job_002"),
            candidate("job_003"),
            candidate("job_004"),
            candidate("job_005"),
        ],
        "rejected_counterfactuals": [rejection("job_092")],
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
        # Check call 2: the same constraint values, in an independent copy.
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
            call_hard_1.allowed_work_modes == sample_hard_constraints.allowed_work_modes
        )
        assert call_hard_1.allowed_cities == sample_hard_constraints.allowed_cities
        assert call_hard_1.max_required_yoe == sample_hard_constraints.max_required_yoe


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
        "eligible_candidates": [candidate(f"job_{i:03d}") for i in range(1, 4)],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 3,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }
    # Fallback returns 10 new jobs
    search_2 = {
        "eligible_candidates": [candidate(f"job_{i:03d}") for i in range(4, 14)],
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
        "eligible_candidates": [candidate("job_001")],
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
        "eligible_candidates": [candidate(f"job_{i:03d}") for i in range(1, 6)],
        "rejected_counterfactuals": [rejection("job_099")],
        "candidates_evaluated": 15,
        "used_mock_fallback": False,
        "embedding_calls": 1,
    }

    with (
        patch("app.agents.discovery.search_jobs", return_value=search_result),
        patch("app.agents.discovery.log_trace") as mock_log_trace,
    ):
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
    mock_db.execute.return_value.one.return_value = SimpleNamespace(
        profile_id="profile_1", preference_version=1
    )
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
        "eligible_candidates": [candidate("job_001"), candidate("job_002")],
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

    with (
        patch(
            "app.agents.discovery.get_preferences",
            return_value={
                "version": 1,
                "hard_constraints": sample_hard_constraints.model_dump(mode="json"),
                "soft_preferences": {"target_roles": [], "preferred_skills": []},
            },
        ),
        patch("app.agents.discovery.search_jobs", return_value=search_result),
        patch("app.agents.discovery.log_trace"),
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

    with patch("app.agents.discovery.search_jobs", return_value=search_output()):
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


def test_run_discovery_zero_candidates_both_searches(mock_db, sample_hard_constraints):
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
        "eligible_candidates": [candidate("job_001"), candidate("job_002")],
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
# 4. Real PostgreSQL Integration Tests (fail explicitly if unavailable)
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


def fixture_job(i):
    """Use the established corpus phrasing; test discovery, not new NLP rules."""
    job_id = f"job_{i:03d}"
    raw = f"Remote - India. Python developer role {i}. Minimum experience is {3 if i == 4 else 0} years."
    if i != 5:
        raw += " Students eligible. A completed degree is not required."
    excerpts = slice_into_excerpts(job_id, raw)
    norm = normalize_job_constraints(
        {"job_id": job_id, "raw_description": raw}, excerpts
    )
    norm.pop("has_conflict")
    return JobORM(
        job_id=job_id,
        source_url=f"https://example.invalid/{job_id}",
        title=f"Python Developer {i}",
        company="TechCorp",
        raw_description=raw,
        excerpts=excerpts,
        search_tsv=func.to_tsvector("english", raw),
        **norm,
    )


@pytest.mark.parametrize(
    "number,expected",
    [(1, "pass"), (2, "pass"), (3, "pass"), (4, "fail"), (5, "unknown")],
)
def test_database_fixture_has_real_pass_fail_and_unknown_cases(number, expected):
    job = fixture_job(number)
    meta = JobMetadataORM(
        job_id=job.job_id,
        captured_at=datetime(2026, 11, 19, tzinfo=UTC),
        content_hash="fixture",
        source_status="active",
        dedup_key=job.job_id,
        is_duplicate=False,
    )
    result, checks = evaluate_job_constraints(
        job, meta, HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    )
    assert result.value == expected
    if number == 4:
        assert any(
            check.field == "experience_limit" and check.result == "fail"
            for check in checks
        )
    if number == 5:
        assert any(
            check.field == "student_eligibility" and check.result == "unknown"
            for check in checks
        )


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

    metadata = []
    for i in range(1, 6):
        job = fixture_job(i)
        job_id = job.job_id
        db.add(job)
        metadata.append(
            JobMetadataORM(
                job_id=job_id,
                captured_at=run.created_at,
                content_hash=f"hash_{i}",
                source_status="active",
                dedup_key=job_id,
                is_duplicate=False,
            )
        )
    # These ORM models have no relationships to order parent/child unit-of-work
    # inserts. Persist all job parents before adding their metadata children.
    db.flush()
    db.add_all(metadata)
    db.flush()
    return db


@pytest.fixture
def test_embeddings(monkeypatch):
    """Explicit deterministic provider double; PostgreSQL/search/gates stay real."""
    queries = []

    class TestEmbeddings:
        used_mock_fallback = True
        last_embedding_usage = LLMUsageMeta(
            model_id=settings.EMBEDDING_MODEL_ID,
            total_tokens=3,
            used_mock_fallback=True,
        )

        def __init__(self, *, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def embed_texts(self, texts, task_type):
            assert task_type == "RETRIEVAL_QUERY" and len(texts) == 1
            queries.extend(texts)
            return [[1.0] + [0.0] * 767]

    monkeypatch.setattr(retrieval, "GeminiService", TestEmbeddings)
    return queries


def test_postgresql_discovery_agent_end_to_end(seeded_db, planner, test_embeddings):
    state = {
        "profile_id": "profile_disc_1",
        "run_id": "run_disc_001",
        "user_query": "Python Developer",
    }
    result = discovery_node(state, db=seeded_db, llm_service=planner)

    assert result["candidate_job_ids"] == ["job_001", "job_002", "job_003", "job_005"]
    assert [item["job_id"] for item in result["rejected_counterfactuals"]] == [
        "job_004"
    ]
    assert result["searches_used"] == 2
    assert result["used_mock_fallback"] is True
    assert result["embedding_calls"] == 2
    assert result["preference_version"] == 1
    assert result["candidates_evaluated"] == 10
    planner.generate_structured.assert_called_once()
    assert test_embeddings == ["Python", "developer"]
    json.dumps(result, allow_nan=False)

    # Verify traces were actually inserted in the real database
    traces = (
        seeded_db.query(TraceORM)
        .filter(TraceORM.run_id == "run_disc_001")
        .order_by(TraceORM.step_index.asc())
        .all()
    )
    assert [trace.step_index for trace in traces] == list(range(5))
    assert [trace.tool_or_action for trace in traces] == [
        "get_preferences",
        "plan_queries",
        "search_jobs",
        "search_jobs",
        "select_candidates",
    ]
    assert sum(trace.tokens_used for trace in traces) == 14
    assert traces[-1].payload_summary["used_mock_fallback"] is True


@pytest.mark.parametrize("include_profile", [True, False])
def test_postgresql_run_uses_pinned_preferences_and_preserves_dirty_state(
    seeded_db, planner, test_embeddings, include_profile
):
    db = seeded_db
    active = db.scalar(
        select(PreferenceORM).where(PreferenceORM.profile_id == "profile_disc_1")
    )
    active.is_active = False
    db.flush()
    db.add(
        PreferenceORM(
            profile_id="profile_disc_1",
            version=2,
            is_active=True,
            hard_constraints=HardConstraints(
                allowed_work_modes=["onsite"], allowed_cities=[]
            ).model_dump(mode="json"),
            soft_preferences=SoftPreferences(
                target_roles=["Sales"], preferred_skills=[]
            ).model_dump(mode="json"),
        )
    )
    db.flush()
    job = db.get(JobORM, "job_001")
    job.title = "Pending caller edit"
    db.autoflush = True
    # Omitted query resolves from the run's historical preferences, not active v2.
    state = {"run_id": "run_disc_001"}
    if include_profile:
        state["profile_id"] = "profile_disc_1"
    before = json.loads(json.dumps(state))
    result = discovery_node(state, db=db, llm_service=planner)
    assert result["preference_version"] == 1
    assert result["candidate_job_ids"] == ["job_001", "job_002", "job_003", "job_005"]
    assert "Python Developer" in planner.generate_structured.call_args.kwargs["prompt"]
    assert state == before
    assert job.title == "Pending caller edit" and job in db.dirty
    with db.no_autoflush:
        assert (
            db.scalar(select(JobORM.title).where(JobORM.job_id == "job_001"))
            == "Python Developer 1"
        )
    planner.generate_structured.reset_mock()
    with pytest.raises(ValueError, match="disagrees"):
        discovery_node({**state, "preference_version": 2}, db=db, llm_service=planner)
    with pytest.raises(ValueError, match="disagree"):
        discovery_node(
            {
                **state,
                "hard_constraints": {"allowed_work_modes": [], "allowed_cities": []},
            },
            db=db,
            llm_service=planner,
        )
    planner.generate_structured.assert_not_called()
