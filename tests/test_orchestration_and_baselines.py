"""Step 2.5: graph unit checks and mandatory real PostgreSQL persistence.

Provider/embedding doubles are deterministic and never validate live Gemini.
Some doubles intentionally report live provenance to exercise accept paths.
The PostgreSQL test fails explicitly if the real database is unavailable.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.agents import graph
from app.agents.baselines import run_retrieval_only_baseline, run_single_agent_baseline
from app.config import Settings, settings
from app.db.models import (
    Base,
    CandidateEvidenceORM,
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
    MatchingBatchOutput,
    VerificationVerdictBatch,
)
from app.services import retrieval
from app.services.constraints import evaluate_job_constraints
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts
from app.services.llm_client import GeminiRequestError, GeminiService


def fixtures(count=7):
    jobs, reports, rejected = [], [], []
    hard = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    for number in range(1, count + 1):
        job_id = f"job_{number:03d}"
        raw = f"Remote - India. Python developer role {number}. Minimum experience is {3 if number == 4 else 0} years."
        if number != 5:
            raw += " Students eligible. A completed degree is not required."
        excerpts = slice_into_excerpts(job_id, raw)
        normalized = normalize_job_constraints(
            {"job_id": job_id, "raw_description": raw}, excerpts
        )
        normalized.pop("has_conflict")
        job = JobORM(
            job_id=job_id,
            title="Python Developer",
            company="Example",
            source_url=f"https://example.invalid/{job_id}",
            raw_description=raw,
            excerpts=excerpts,
            **normalized,
        )
        meta = JobMetadataORM(
            job_id=job_id,
            captured_at=datetime(2026, 11, 19, tzinfo=UTC),
            content_hash="fixture",
            source_status="active",
            dedup_key=job_id,
            is_duplicate=False,
        )
        overall, checks = evaluate_job_constraints(job, meta, hard)
        detail = {
            c.name: getattr(job, c.name)
            for c in JobORM.__table__.columns
            if c.name != "search_tsv"
        }
        detail["metadata"] = {
            c.name: getattr(meta, c.name) for c in JobMetadataORM.__table__.columns
        }
        detail["metadata"]["captured_at"] = meta.captured_at.isoformat()
        report = {
            "job_id": job_id,
            "result": overall.value,
            "checks": [c.model_dump(mode="json") for c in checks],
        }
        if overall == "fail":
            rejected.append(
                {
                    "job_id": job_id,
                    "title": job.title,
                    "company": job.company,
                    "similarity_score": 0.0,
                    "failed_constraints": [
                        c for c in report["checks"] if c["result"] == "fail"
                    ],
                    "rejection_summary": "Experience exceeds limit.",
                }
            )
        else:
            jobs.append(detail)
            reports.append(report)
    return hard, jobs, reports, rejected


def state(count=7, variant="multi_agent"):
    hard, jobs, reports, rejected = fixtures(count)
    return {
        "run_id": "unit_run",
        "profile_id": "candidate",
        "profile_version": 1,
        "preference_version": 1,
        "user_query": "Python",
        "system_variant": variant,
        "hard_constraints": hard.model_dump(mode="json"),
        "soft_preferences": {
            "target_roles": ["Python"],
            "preferred_skills": [],
            "prefer_startup": False,
        },
        "candidate_job_ids": [j["job_id"] for j in jobs],
        "hydrated_jobs": jobs,
        "deterministic_constraints": reports,
        "candidate_evidence": [],
        "assessments": [],
        "verification_feedback": [],
        "matching_integrity_issues": {},
        "correction_count": 0,
        "model_calls": 0,
        "embedding_calls": 0,
        "searches_used": 0,
        "used_mock_fallback": False,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "usage_records": [],
        "status": "running",
        "stop_reason": "",
        "assessments_ready": False,
        "shortlist": [],
        "needs_review_list": [],
        "rejected_counterfactuals": rejected,
    }


class Provider:
    def __init__(self, *, revise=False, mock=False, malformed=False, fail_schema=None):
        self.calls = []
        self.used_mock_fallback = mock
        self.revise, self.malformed, self.fail_schema = revise, malformed, fail_schema

    def generate_structured(self, **kwargs):
        schema = kwargs["response_schema"]
        self.calls.append(kwargs)
        if schema is self.fail_schema:
            raise GeminiRequestError(
                "Sensitive provider diagnostics must not reach traces"
            )
        prompt = json.loads(kwargs["prompt"])
        usage = LLMUsageMeta(
            model_id=kwargs["model_id"],
            total_tokens=11,
            estimated_cost_usd=0.001,
            used_mock_fallback=self.used_mock_fallback,
        )
        if schema is DiscoveryQueryPlan:
            assert kwargs["model_id"] == settings.CHAT_MODEL_ID_FAST
            assert kwargs["thinking_level"] == "LOW"
            return DiscoveryQueryPlan(
                primary_query="Python",
                fallback_broad_query="developer",
                rationale="Shared query",
            ), usage
        assert kwargs["model_id"] == settings.CHAT_MODEL_ID
        assert kwargs["thinking_level"] == "MEDIUM"
        if schema is MatchingBatchOutput:
            assessments = []
            for job in prompt["jobs"]:
                assessments.append(
                    dict(
                        **{
                            key: job[key]
                            for key in (
                                "job_id",
                                "title",
                                "company",
                                "source_url",
                                "captured_at",
                            )
                        },
                        decision="needs_review"
                        if job["deterministic_constraints"]["result"] == "unknown"
                        else "accept",
                        supported_matches=[],
                        gaps=[],
                        unknowns=[],
                        constraint_checks=job["deterministic_constraints"]["checks"],
                        internal_rank_score=0.5,
                    )
                )
            if self.malformed:
                assessments += assessments[:1]
            return MatchingBatchOutput(assessments=assessments), usage
        assert schema is VerificationVerdictBatch
        return VerificationVerdictBatch(
            verdicts=[
                {
                    "job_id": job["job_id"],
                    "verdict": "revise" if self.revise else "accept",
                    "issues": ["Check evidence"] if self.revise else [],
                    "feedback_for_matching": "Recheck evidence" if self.revise else "",
                }
                for job in prompt["jobs"]
            ]
        ), usage


@pytest.fixture
def harness(monkeypatch):
    bundle = state()
    db = MagicMock(spec=Session)
    traces, searches = [], []
    monkeypatch.setattr(graph, "log_trace", lambda *a, **kw: traces.append(kw))

    def search(db, query, hard, limit):
        searches.append((query, hard.model_dump(mode="json"), limit))
        return {
            "eligible_candidates": [
                {"job_id": r["job_id"], "constraint_result": r["result"]}
                for r in bundle["deterministic_constraints"]
            ],
            "rejected_counterfactuals": deepcopy(bundle["rejected_counterfactuals"]),
            "candidates_evaluated": len(bundle["hydrated_jobs"])
            + len(bundle["rejected_counterfactuals"]),
            "used_mock_fallback": False,
            "embedding_calls": 1,
            "embedding_usage": LLMUsageMeta(
                total_tokens=3, estimated_cost_usd=0.002
            ).model_dump(mode="json"),
        }

    def hydrate(db, incoming):
        return deepcopy(
            {
                key: bundle[key]
                for key in (
                    "candidate_job_ids",
                    "hydrated_jobs",
                    "deterministic_constraints",
                    "candidate_evidence",
                    "rejected_counterfactuals",
                )
            }
        )

    monkeypatch.setattr(graph, "hybrid_search_jobs", search)
    monkeypatch.setattr(graph, "hydrate_context", hydrate)

    def invoke(variant="multi_agent", provider=None, calls=8, clock=None, timeout=120):
        initial = state(0, variant)
        budget = graph.RunBudget(calls, timeout, **({"clock": clock} if clock else {}))
        result = graph.build_graph(
            db, budget=budget, llm_service=provider or Provider()
        ).invoke(initial)
        json.dumps(result, allow_nan=False)
        return result

    return SimpleNamespace(
        db=db, bundle=bundle, traces=traces, searches=searches, invoke=invoke
    )


def test_three_variants_share_pool_gates_evidence_and_persist_update(harness):
    results = []
    for variant, expected_calls in (
        ("retrieval_only", 2),
        ("single_agent", 3),
        ("multi_agent", 4),
    ):
        provider = Provider()
        result = harness.invoke(variant, provider)
        assert result["status"] == "completed"
        assert result["model_calls"] == expected_calls
        assert result["total_tokens"] == 11 * (expected_calls - 1) + 3
        assert len(result["shortlist"]) == 5
        assert [a["job_id"] for a in result["needs_review_list"]] == ["job_005"]
        assert [a["job_id"] for a in result["rejected_counterfactuals"]] == ["job_004"]
        assert all(a["decision"] == "accept" for a in result["shortlist"])
        assert all(not a["supported_matches"] for a in result["shortlist"])
        if variant != "multi_agent":
            assert all(
                call["response_schema"] is not VerificationVerdictBatch
                for call in provider.calls
            )
        results.append(result)
    for key in (
        "candidate_job_ids",
        "candidate_evidence",
        "hydrated_jobs",
        "deterministic_constraints",
    ):
        assert results[0][key] == results[1][key] == results[2][key]
    assert harness.searches[0] == harness.searches[1] == harness.searches[2]
    assert harness.db.execute.call_count == 3


def test_correction_exactly_once_and_unresolved_claims_removed(harness):
    provider = Provider(revise=True)
    result = harness.invoke(provider=provider)
    assert result["correction_count"] == 1
    assert result["model_calls"] == 6
    assert [c["response_schema"] for c in provider.calls].count(
        MatchingBatchOutput
    ) == 2
    assert [c["response_schema"] for c in provider.calls].count(
        VerificationVerdictBatch
    ) == 2
    assert not result["shortlist"]
    assert len(result["needs_review_list"]) == 6
    correction_prompt = json.loads(provider.calls[3]["prompt"])
    assert correction_prompt["correction_count"] == 1
    assert len(correction_prompt["verification_feedback"]) == 6


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 5])
def test_exhausted_budget_never_publishes_unaudited_accept(harness, limit):
    result = harness.invoke(provider=Provider(revise=True), calls=limit)
    assert result["model_calls"] <= limit
    assert result["status"] in ("partial", "failed")
    assert result["stop_reason"] == "model_call_budget"
    assert not result["shortlist"]
    assert all(not item["supported_matches"] for item in result["needs_review_list"])


@pytest.mark.parametrize("value", [-1, 9, True, 1.5, None])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        graph.RunBudget(value, 120)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None])
def test_invalid_timeout(value):
    with pytest.raises(ValueError):
        graph.RunBudget(8, value)


def test_zero_timeout_has_no_provider_or_retrieval_calls(harness):
    provider = Provider()
    result = harness.invoke(provider=provider, timeout=0)
    assert result["status"] == "failed"
    assert result["stop_reason"] == "timeout"
    assert not provider.calls and not harness.searches


def test_late_matching_response_is_discarded(harness):
    clock = [0.0]
    provider = Provider()
    generate = provider.generate_structured

    def late(**kw):
        output = generate(**kw)
        if kw["response_schema"] is MatchingBatchOutput:
            clock[0] = 121.0
        return output

    provider.generate_structured = late
    result = harness.invoke(provider=provider, clock=lambda: clock[0])
    assert result["status"] == "partial" and result["stop_reason"] == "timeout"
    assert not result["shortlist"]
    assert result["total_tokens"] == 25  # Late usage still costs tokens.
    assert len(provider.calls) == 2


@pytest.mark.parametrize(
    "schema", [DiscoveryQueryPlan, MatchingBatchOutput, VerificationVerdictBatch]
)
def test_provider_failure_is_persisted_without_retry_or_sensitive_text(harness, schema):
    provider = Provider(fail_schema=schema)
    result = harness.invoke(provider=provider)
    assert result["status"] in ("failed", "partial")
    assert result["stop_reason"] == "GeminiRequestError"
    assert not result["shortlist"]
    assert [call["response_schema"] for call in provider.calls].count(schema) == 1
    assert "Sensitive" not in json.dumps(harness.traces)


def test_duplicate_model_ids_fail_closed(harness):
    result = harness.invoke(provider=Provider(malformed=True))
    assert result["status"] == "partial"
    assert not result["shortlist"]
    assert result["total_tokens"] == 25
    assert result["estimated_cost_usd"] == pytest.approx(0.004)


def test_empty_pool_no_matching_or_verification_calls(harness):
    empty = state(0)
    harness.bundle.update(empty)
    provider = Provider()
    result = harness.invoke(provider=provider)
    assert result["status"] == "completed"
    assert result["shortlist"] == result["needs_review_list"] == []
    assert result["searches_used"] == 2
    assert len(provider.calls) == 1


def test_deterministic_mock_path_flagged_and_never_semantically_accepted(harness):
    first = harness.invoke(provider=Provider(mock=True))
    second = harness.invoke(provider=Provider(mock=True))
    assert first["used_mock_fallback"] and second["used_mock_fallback"]
    assert not first["shortlist"]
    for key in (
        "shortlist",
        "needs_review_list",
        "rejected_counterfactuals",
        "correction_count",
    ):
        assert first[key] == second[key]


def test_actual_offline_provider_invalid_batch_fails_honestly(harness, tmp_path):
    config = Settings(
        GEMINI_API_KEY="", LLM_CACHE_PATH=str(tmp_path / "offline.sqlite")
    )
    with GeminiService(config=config) as service:
        result = harness.invoke(provider=service)
    assert result["used_mock_fallback"] is True
    assert result["status"] == "partial"
    assert result["stop_reason"] == "GeminiResponseError"
    assert not result["shortlist"]


def test_finalize_citation_gate_cannot_be_bypassed_by_single_agent():
    item = state(2, "single_agent")
    item["assessments"] = graph.deterministic_assessments(item)
    item["assessments_ready"] = True
    item["assessments"][0]["supported_matches"] = [
        {
            "requirement": "Python",
            "candidate_evidence_ids": ["ev_fabricated"],
            "job_excerpt_ids": ["job_002_ex_00"],
            "explanation": "Invented support",
            "is_self_reported_only": False,
        }
    ]
    result = graph.finalize_results(item)
    assert [a["job_id"] for a in result["shortlist"]] == ["job_002"]
    assert result["needs_review_list"][0]["supported_matches"] == []


def test_duplicate_final_assessments_rejected():
    item = state(2)
    item["assessments"] = graph.deterministic_assessments(item)
    item["assessments"].append(deepcopy(item["assessments"][0]))
    item["assessments_ready"] = True
    with pytest.raises(ValueError, match="exactly once"):
        graph.finalize_results(item)


def test_budget_instances_are_independent():
    first, second = graph.RunBudget(8, 120), graph.RunBudget(8, 120)
    first.reserve()
    first.record(LLMUsageMeta(total_tokens=1, used_mock_fallback=True))
    assert second.calls == 0 and second.usage == [] and not second.used_mock


def test_duplicate_retrieval_ids_do_not_suppress_fallback(harness, monkeypatch):
    search = graph.hybrid_search_jobs

    def duplicates(*args, **kwargs):
        result = search(*args, **kwargs)
        result["eligible_candidates"] = result["eligible_candidates"][:1] * 8
        return result

    monkeypatch.setattr(graph, "hybrid_search_jobs", duplicates)
    result = harness.invoke(provider=Provider(revise=True))
    assert result["searches_used"] == 2
    assert result["model_calls"] == 7


def test_zero_search_budget_spends_nothing(harness, monkeypatch):
    monkeypatch.setattr(settings, "MAX_DISCOVERY_SEARCHES", 0)
    provider = Provider()
    result = harness.invoke(provider=provider)
    assert result["status"] == "failed"
    assert result["stop_reason"] == "search_budget"
    assert not provider.calls and not harness.searches


def test_zero_shortlist_and_unknown_gate(harness, monkeypatch):
    monkeypatch.setattr(settings, "MAX_SHORTLIST", 0)
    result = harness.invoke("retrieval_only")
    assert not result["shortlist"]
    assert [a["job_id"] for a in result["needs_review_list"]] == ["job_005"]


@pytest.mark.parametrize(
    "ids", [["job_001", "job_001"], [f"job_{i:03d}" for i in range(11)]]
)
def test_hydration_rejects_duplicate_and_oversized_pool_before_sql(ids):
    item = state()
    item["candidate_job_ids"] = ids
    db = MagicMock(spec=Session)
    with pytest.raises(ValueError):
        graph.hydrate_context(db, item)
    db.execute.assert_not_called()


def test_hydration_snapshot_shapes_without_database_fallback():
    """Unit row double only; the mandatory PostgreSQL tests remain separate."""
    item = state(2)
    db = MagicMock(spec=Session)
    rows = []
    for job in item["hydrated_jobs"]:
        row = {
            f"job_{column.name}": job.get(column.name)
            for column in JobORM.__table__.columns
        }
        row.update({f"meta_{key}": value for key, value in job["metadata"].items()})
        row["meta_captured_at"] = datetime.fromisoformat(row["meta_captured_at"])
        row.update(
            {
                f"ev_{column.name}": None
                for column in CandidateEvidenceORM.__table__.columns
            }
        )
        rows.append(row)
    db.execute.return_value.mappings.return_value.all.return_value = rows
    result = graph.hydrate_context(db, item)
    assert result["candidate_job_ids"] == item["candidate_job_ids"]
    assert result["deterministic_constraints"] == item["deterministic_constraints"]
    assert result["candidate_evidence"] == []
    assert db.execute.call_count == 1
    json.dumps(result, allow_nan=False)
    db.execute.return_value.mappings.return_value.all.return_value = rows[:1]
    with pytest.raises(LookupError):
        graph.hydrate_context(db, item)


@pytest.mark.parametrize("query", [None, False, "\x00"])
def test_invalid_query_never_creates_run(query, monkeypatch):
    db = MagicMock(spec=Session)
    db.scalar.return_value = 1
    bundle = state()
    monkeypatch.setattr(
        graph,
        "get_preferences",
        lambda *args: {
            "version": 1,
            "soft_preferences": bundle["soft_preferences"],
            "hard_constraints": bundle["hard_constraints"],
        },
    )
    with pytest.raises((ValueError, TypeError)):
        graph.run_multi_agent(db, "candidate", query)
    db.execute.assert_not_called()


@pytest.fixture
def postgres():
    try:
        connection = engine.connect()
    except OperationalError:
        pytest.fail(
            "Real PostgreSQL is unavailable; no SQLite/mock persistence fallback is used.",
            pytrace=False,
        )
    with connection:
        outer = connection.begin()
        try:
            assert (
                160000
                <= int(connection.scalar(text("SHOW server_version_num")))
                < 170000
            )
            assert connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
            schema = f"orchestration_{uuid4().hex}"
            connection.execute(CreateSchema(schema))
            scoped = connection.execution_options(schema_translate_map={None: schema})
            Base.metadata.create_all(scoped)
            with Session(
                bind=scoped,
                autoflush=False,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            ) as db:
                yield db, scoped
        finally:
            outer.rollback()


def seed(db):
    bundle = state()
    db.add(CandidateProfileORM(id="candidate", name="Candidate", profile_version=2))
    db.flush()
    db.add(
        PreferenceORM(
            profile_id="candidate",
            version=1,
            is_active=True,
            hard_constraints=bundle["hard_constraints"],
            soft_preferences=bundle["soft_preferences"],
        )
    )
    for number in range(1, 8):
        # Include the rejected job too; retrieval must actually eliminate it.
        job_id = f"job_{number:03d}"
        raw = f"Remote - India. Python developer role {number}. Minimum experience is {3 if number == 4 else 0} years."
        if number != 5:
            raw += " Students eligible. A completed degree is not required."
        excerpts = slice_into_excerpts(job_id, raw)
        normalized = normalize_job_constraints(
            {"job_id": job_id, "raw_description": raw}, excerpts
        )
        normalized.pop("has_conflict")
        db.add(
            JobORM(
                job_id=job_id,
                source_url=f"https://example.invalid/{job_id}",
                title="Python Developer",
                company="Example",
                raw_description=raw,
                excerpts=excerpts,
                search_tsv=func.to_tsvector("english", raw),
                **normalized,
            )
        )
    db.flush()
    for number in range(1, 8):
        job_id = f"job_{number:03d}"
        db.add(
            JobMetadataORM(
                job_id=job_id,
                captured_at=datetime(2026, 11, 19, tzinfo=UTC),
                content_hash="fixture",
                source_status="active",
                dedup_key=job_id,
                is_duplicate=False,
            )
        )
    for version in (1, 2):
        db.add(
            CandidateEvidenceORM(
                evidence_id=f"ev_candidate_{version}",
                profile_id="candidate",
                skill="Python",
                source_type="user_statement",
                verification_status="self_reported",
                source_excerpt="I use Python",
                source_id="statement",
                profile_version=version,
            )
        )
    db.commit()


@pytest.fixture
def embeddings(monkeypatch):
    class Embeddings:
        used_mock_fallback = True
        last_embedding_usage = LLMUsageMeta(total_tokens=3, used_mock_fallback=True)

        def __init__(self, **kwargs):
            self.config = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def embed_texts(self, texts, **kwargs):
            return [[1.0] + [0.0] * 767 for _ in texts]

    monkeypatch.setattr(retrieval, "GeminiService", Embeddings)


def test_postgresql_all_variants_filter_and_persist(postgres, embeddings):
    db, connection = postgres
    seed(db)
    runs = []
    for runner in (
        graph.run_multi_agent,
        run_single_agent_baseline,
        run_retrieval_only_baseline,
    ):
        run = runner(db, "candidate", "Python", llm_service=Provider())
        assert isinstance(run, SearchRunORM)
        assert run.status == "completed"
        assert run.profile_version == 2 and run.preference_version == 1
        assert run.used_mock_fallback is True  # Embeddings used a test double.
        assert [r["job_id"] for r in run.rejected_counterfactuals] == ["job_004"]
        assert len(run.shortlist) <= 5
        for item in run.shortlist:
            assert all(check["result"] == "pass" for check in item["constraint_checks"])
        assert "job_005" in [item["job_id"] for item in run.needs_review_list]
        runs.append(run)
    # Eligibility reports must be byte-for-byte identical across all variants.
    gates = [
        {
            a["job_id"]: a["constraint_checks"]
            for a in run.shortlist + run.needs_review_list
        }
        for run in runs
    ]
    assert gates[0] == gates[1] == gates[2]
    db.commit()
    with Session(bind=connection, join_transaction_mode="create_savepoint") as reader:
        for run in runs:
            persisted = reader.get(SearchRunORM, run.run_id)
            assert persisted.shortlist == run.shortlist
            assert persisted.needs_review_list == run.needs_review_list
            assert persisted.total_tokens > 0
            traces = reader.scalars(
                select(TraceORM)
                .where(TraceORM.run_id == run.run_id)
                .order_by(TraceORM.step_index)
            ).all()
            assert [trace.step_index for trace in traces] == list(range(len(traces)))
            assert all(
                "resume" not in json.dumps(trace.payload_summary).lower()
                for trace in traces
            )


def test_postgresql_hydration_single_select_pinned_evidence_and_missing_job(postgres):
    db, connection = postgres
    seed(db)
    item = state()
    item["profile_version"] = 1
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    # seed() commits the session. Rejoin the fixture's outer transaction before
    # measuring hydration, so its lazy SAVEPOINT is not counted as hydration SQL.
    db.connection()
    event.listen(connection, "before_cursor_execute", record)
    try:
        result = graph.hydrate_context(db, item)
    finally:
        event.remove(connection, "before_cursor_execute", record)
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT ")
    assert [e["evidence_id"] for e in result["candidate_evidence"]] == [
        "ev_candidate_1"
    ]
    assert result["candidate_job_ids"] == item["candidate_job_ids"]
    item["candidate_job_ids"] = ["job_999"]
    with pytest.raises(LookupError):
        graph.hydrate_context(db, item)
    item["candidate_job_ids"] = []
    result = graph.hydrate_context(db, item)
    assert result["hydrated_jobs"] == []
    assert len(result["candidate_evidence"]) == 1
