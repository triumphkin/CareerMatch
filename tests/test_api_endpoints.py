"""Step 3.1: real TestClient lifecycle on PostgreSQL plus isolated-worker tests.

Model/embedding doubles are deterministic; they are not live Gemini validation.
The mandatory PostgreSQL tests fail explicitly if the real DB is unavailable.
"""

import inspect
import json
import multiprocessing
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pymupdf as fitz
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.api import routes
from app.config import settings
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
from app.main import create_app
from app.schemas.contracts import (
    DiscoveryQueryPlan,
    LLMUsageMeta,
    MatchingBatchOutput,
    VerificationVerdictBatch,
)
from app.services import profile_service, retrieval
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts
from app.services.llm_client import GeminiRequestError

SECRET = "PRIVATE_RESUME_TEXT_DO_NOT_EXPOSE"
HARD = {"allowed_work_modes": ["remote"], "allowed_cities": []}
SOFT = {"target_roles": ["Python developer"], "preferred_skills": ["Python"]}
CONFIRM = {
    "hard_constraints": HARD,
    "soft_preferences": SOFT,
    "self_reported_skills": [
        {"skill": "Multi-agent AI", "statement": "I studied Multi-agent AI"}
    ],
}


@pytest.fixture
def client_runtime():
    # AnyIO's TestClient portal uses socketpair for cross-thread wake-ups. Some
    # restricted runners deny send(), which otherwise hangs before app startup.
    try:
        left, right = socket.socketpair()
        with left, right:
            left.send(b"x")
            assert right.recv(1) == b"x"
    except OSError:
        pytest.fail(
            "TestClient requires local socket wake-ups; this execution environment denies them.",
            pytrace=False,
        )


@pytest.fixture
def unit_client(client_runtime):
    app = create_app()
    db = MagicMock(spec=Session)
    app.dependency_overrides[routes.get_db] = lambda: db
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, db


def test_all_nine_routes_are_synchronous():
    actual = {
        (method.upper(), path)
        for path, operations in create_app().openapi()["paths"].items()
        for method in operations
        if method in {"get", "post", "patch"}
    }
    expected = {
        ("POST", "/profiles"),
        ("POST", "/profiles/{profile_id}/confirm"),
        ("PATCH", "/profiles/{profile_id}/preferences"),
        ("POST", "/searches"),
        ("GET", "/searches/{run_id}"),
        ("GET", "/searches/{run_id}/trace"),
        ("POST", "/feedback"),
        ("GET", "/health"),
        ("GET", "/ready"),
    }
    assert actual == expected
    assert all(
        not inspect.iscoroutinefunction(r.endpoint)
        for r in routes.router.routes
        if isinstance(r, APIRoute)
    )


def test_health_is_independent_of_database_and_has_request_id(client_runtime):
    app = create_app()

    def unavailable():
        raise AssertionError("Health must not connect to the database")

    app.dependency_overrides[routes.get_db] = unavailable
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "my-request-1"})
        assert response.status_code == 200 and response.json() == {"status": "ok"}
        assert response.headers["x-request-id"] == "my-request-1"
        replaced = client.get("/health", headers={"X-Request-ID": "bad id"})
        assert len(replaced.headers["x-request-id"]) == 32


@pytest.mark.parametrize(
    "path,body",
    [
        ("/profiles", {"name": "Candidate", "raw_text": None}),
        ("/profiles", {"name": " ", "raw_text": "text"}),
        ("/profiles", {"name": "Candidate", "raw_text": "\x00"}),
        ("/profiles", {"name": "Candidate", "raw_text": SECRET, "unexpected": SECRET}),
        ("/searches", {"profile_id": "candidate", "user_query": " "}),
        (
            "/searches",
            {
                "profile_id": "candidate",
                "user_query": "Python",
                "system_variant": "invalid",
            },
        ),
        (
            "/searches",
            {"profile_id": "candidate", "user_query": "Python", "timeout_seconds": 0},
        ),
        (
            "/searches",
            {"profile_id": "candidate", "user_query": "Python", "timeout_seconds": -1},
        ),
        (
            "/searches",
            {"profile_id": "candidate", "user_query": "Python", "timeout_seconds": 121},
        ),
        ("/feedback", {"run_id": "run", "job_id": "job_001", "relevance": "maybe"}),
        (
            "/profiles/candidate/confirm",
            {**CONFIRM, "confirmed_evidence_ids": ["ev_001", "ev_001"]},
        ),
        (
            "/profiles/candidate/confirm",
            {**CONFIRM, "self_reported_skills": CONFIRM["self_reported_skills"] * 2},
        ),
    ],
)
def test_invalid_requests_never_write_and_never_echo_inputs(unit_client, path, body):
    client, db = unit_client
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert SECRET not in response.text
    assert response.json()["request_id"] == response.headers["x-request-id"]
    db.commit.assert_not_called()
    db.execute.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"hard_constraints": {}},
        {"soft_preferences": None},
        {"hard_constraints": {"excluded_keywords": ["\x00"]}},
    ],
)
def test_empty_or_nul_preference_patch_is_rejected(unit_client, body):
    client, db = unit_client
    assert client.patch("/profiles/candidate/preferences", json=body).status_code == 422
    db.commit.assert_not_called()


def test_malformed_json_and_unsupported_content_type(unit_client):
    client, _ = unit_client
    assert (
        client.post(
            "/profiles",
            content=b'{"name":',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )
    assert client.post("/profiles", content=b"raw").status_code == 415


def test_pdf_empty_invalid_conflicting_and_oversized_input(unit_client, monkeypatch):
    client, db = unit_client
    for content in (b"", b"not pdf"):
        assert (
            client.post(
                "/profiles",
                data={"name": "Candidate"},
                files={"file": ("cv.pdf", content, "application/pdf")},
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/profiles",
            data={"name": "Candidate", "raw_text": "text"},
            files={"file": ("cv.pdf", b"%PDF-test", "application/pdf")},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/profiles",
            data={"name": "Candidate"},
            files={"file": ("cv.txt", b"text", "text/plain")},
        ).status_code
        == 415
    )
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    assert (
        client.post(
            "/profiles",
            data={"name": "Candidate"},
            files={"file": ("cv.pdf", b"%PDF-" + b"x" * 10, "application/pdf")},
        ).status_code
        == 413
    )
    db.commit.assert_not_called()


def test_body_limit_applies_without_content_length(unit_client):
    client, db = unit_client
    response = client.post(
        "/profiles",
        content=iter([b"x" * (routes.MAX_UPLOAD_BYTES + 65537)]),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413 and response.headers["x-request-id"]
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    "exception,status",
    [
        (GeminiRequestError(SECRET), 502),
        (profile_service.ResumeExtractionError(SECRET), 422),
        (RuntimeError(SECRET), 500),
    ],
)
def test_errors_hide_secrets_and_include_request_ids(
    unit_client, monkeypatch, exception, status
):
    client, db = unit_client

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(profile_service, "create_profile", fail)
    response = client.post("/profiles", json={"name": "Candidate", "raw_text": SECRET})
    assert response.status_code == status
    assert SECRET not in response.text
    assert response.json()["request_id"] == response.headers["x-request-id"]
    db.commit.assert_not_called()


def test_missing_resources_return_404(unit_client):
    client, db = unit_client
    db.scalar.return_value = None
    assert client.get("/searches/missing").status_code == 404
    assert client.get("/searches/missing/trace").status_code == 404
    assert client.post("/profiles/missing/confirm", json=CONFIRM).status_code == 404
    assert (
        client.post(
            "/feedback",
            json={"run_id": "missing", "job_id": "job_001", "relevance": "relevant"},
        ).status_code
        == 404
    )


def test_ready_returns_503_for_missing_vector_or_wrong_postgres(unit_client):
    client, db = unit_client
    db.get_bind.return_value.dialect.name = "postgresql"
    for version, vector in ((160000, None), (150000, "0.8.0")):
        db.scalar.side_effect = [version, vector]
        assert client.get("/ready").status_code == 503
    db.get_bind.return_value.dialect.name = "sqlite"
    assert client.get("/ready").status_code == 503


def test_trace_read_redacts_legacy_free_text_and_nested_values(unit_client):
    client, db = unit_client
    db.scalar.return_value = SimpleNamespace(run_id="run")
    db.scalars.return_value.all.return_value = [
        SimpleNamespace(
            step_index=0,
            agent_name=SECRET,
            tool_or_action=SECRET,
            outcome=SECRET,
            duration_ms=1.0,
            tokens_used=2,
            payload_summary={
                "raw_resume_text": SECRET,
                "model_calls": SECRET,
                "job_ids": ["job_001", SECRET],
                "candidate_evidence": [{"source_excerpt": SECRET}],
                "constraint_decisions": [
                    {
                        "job_id": "job_001",
                        "result": "pass",
                        "checks": [
                            {
                                "field": "experience_limit",
                                "result": "pass",
                                "reason": SECRET,
                            },
                            {"field": SECRET, "result": "pass"},
                        ],
                    }
                ],
                "correction_count": 1,
            },
        )
    ]
    response = client.get("/searches/run/trace")
    assert response.status_code == 200
    assert SECRET not in response.text
    payload = response.json()["steps"][0]["payload_summary"]
    assert payload["job_ids"] == ["job_001"] and payload["correction_count"] == 1
    assert payload["constraint_decisions"][0]["checks"] == [
        {"field": "experience_limit", "result": "pass"}
    ]


def _slow_worker(sender, url, schema, state, timeout):
    Path(state["started"]).write_text("started")
    time.sleep(30)
    Path(state["late"]).write_text("late write")
    sender.send("completed")


def _successful_worker(sender, *args):
    sender.send("completed")
    sender.close()


def _failed_worker(sender, *args):
    sender.close()


def process_db():
    db = MagicMock(spec=Session)
    db.get_bind.return_value.url = URL.create(
        "postgresql+psycopg", host="localhost", database="unit_only"
    )
    db.get_bind.return_value.get_execution_options.return_value = {}
    return db


def test_hard_timeout_kills_real_worker_without_late_writes(tmp_path):
    before = {p.pid for p in multiprocessing.active_children()}
    state = {"started": str(tmp_path / "started"), "late": str(tmp_path / "late")}
    started = time.monotonic()
    with pytest.raises(routes.SearchDeadlineExceeded):
        routes.run_isolated_search(process_db(), state, 5.0, worker=_slow_worker)
    assert time.monotonic() - started < 7.0
    assert (tmp_path / "started").exists(), (
        "Exercise in-flight work, not only spawn delay"
    )
    assert not (tmp_path / "late").exists()
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_worker_success_and_crash_are_distinguished():
    routes.run_isolated_search(process_db(), {}, 5.0, worker=_successful_worker)
    with pytest.raises(routes.SearchWorkerFailed):
        routes.run_isolated_search(process_db(), {}, 5.0, worker=_failed_worker)


@pytest.mark.parametrize(
    "timeout", [True, False, float("inf"), float("nan"), -1, 0, 121]
)
def test_timeout_input_rejects_nonfinite_boolean_and_out_of_range_values(timeout):
    with pytest.raises(ValidationError):
        routes.SearchInput(
            profile_id="candidate", user_query="Python", timeout_seconds=timeout
        )


def test_redaction_removes_nested_text_and_invalid_structured_types():
    payload = {
        "raw_resume_text": SECRET,
        "verification_feedback": [{"issues": [SECRET]}],
        "model_calls": True,
        "used_mock_fallback": SECRET,
        "job_ids": ["job_001", SECRET, "job_001"],
        "constraint_decisions": [
            {
                "job_id": "job_001",
                "result": "pass",
                "checks": [
                    {
                        "field": "student_eligibility",
                        "result": "pass",
                        "reason": SECRET,
                    },
                    {"field": [], "result": "pass"},
                ],
            }
        ],
    }
    clean = routes.redact_payload(payload)
    assert SECRET not in json.dumps(clean)
    assert "model_calls" not in clean and "used_mock_fallback" not in clean
    assert clean["job_ids"] == ["job_001"]
    assert clean["constraint_decisions"][0]["checks"] == [
        {"field": "student_eligibility", "result": "pass"}
    ]


def test_preference_patch_revalidates_merged_contract_before_any_write(monkeypatch):
    db = MagicMock(spec=Session)
    monkeypatch.setattr(
        routes, "_profile", lambda *args, **kwargs: SimpleNamespace(is_confirmed=True)
    )
    monkeypatch.setattr(
        routes,
        "get_preferences",
        lambda *args: {
            "version": 1,
            "hard_constraints": HARD,
            "soft_preferences": SOFT,
        },
    )
    update = MagicMock()
    monkeypatch.setattr(profile_service, "update_preferences", update)
    with pytest.raises(routes.HTTPException) as captured:
        routes.patch_preferences(
            "candidate",
            routes.PreferencePatch(hard_constraints={"max_required_yoe": -1}),
            db,
        )
    assert captured.value.status_code == 422
    update.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    "error,status,action",
    [
        (routes.SearchDeadlineExceeded(), 504, "hard_timeout"),
        (routes.SearchWorkerFailed(), 503, "worker_failed"),
    ],
)
def test_search_timeout_and_worker_error_persist_failure(
    unit_client, monkeypatch, error, status, action
):
    client, db = unit_client
    monkeypatch.setattr(routes, "_new_search", lambda *args: {"run_id": "test_run"})
    events = []
    monkeypatch.setattr(
        routes, "_event", lambda *args, **kwargs: events.append((args, kwargs))
    )

    def fail(*args):
        raise error

    client.app.dependency_overrides[routes.get_search_executor] = lambda: fail
    response = client.post(
        "/searches", json={"profile_id": "candidate", "user_query": "Python"}
    )
    assert response.status_code == status
    assert response.json()["detail"] == {
        "code": action,
        "run_id": "test_run",
        "usage_complete": False,
    }
    db.commit.assert_called_once()
    assert events[0][0][2] == action


@pytest.fixture
def postgres():
    try:
        connection = engine.connect()
    except OperationalError:
        pytest.fail(
            "Real PostgreSQL is unavailable; no SQLite/mock DB fallback is used.",
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
            schema = f"api_test_{uuid4().hex}"
            connection.execute(CreateSchema(schema))
            scoped = connection.execution_options(schema_translate_map={None: schema})
            Base.metadata.create_all(scoped)
            yield scoped
        finally:
            outer.rollback()


class ReasoningDouble:
    used_mock_fallback = False  # Exercise decisions; this is NOT live Gemini evidence.

    def generate_structured(self, **arguments):
        prompt = json.loads(arguments["prompt"])
        schema = arguments["response_schema"]
        usage = LLMUsageMeta(total_tokens=11, model_id=arguments["model_id"])
        if schema is DiscoveryQueryPlan:
            return DiscoveryQueryPlan(
                primary_query="Python",
                fallback_broad_query="developer",
                rationale="Test query",
            ), usage
        if schema is MatchingBatchOutput:
            return MatchingBatchOutput(
                assessments=[
                    {
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
                        "decision": "needs_review"
                        if job["deterministic_constraints"]["result"] == "unknown"
                        else "accept",
                        "supported_matches": [],
                        "gaps": [],
                        "unknowns": [],
                        "constraint_checks": job["deterministic_constraints"]["checks"],
                    }
                    for job in prompt["jobs"]
                ]
            ), usage
        assert schema is VerificationVerdictBatch
        return VerificationVerdictBatch(
            verdicts=[
                {
                    "job_id": job["job_id"],
                    "verdict": "accept",
                    "issues": [],
                    "feedback_for_matching": "",
                }
                for job in prompt["jobs"]
            ]
        ), usage


@pytest.fixture
def providers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        profile_service,
        "settings",
        settings.model_copy(update={"GEMINI_API_KEY": "test-placeholder"}),
    )

    class Extractor:
        used_mock_fallback = False

        def __init__(self, **kwargs):
            self.config = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate_structured(self, **kwargs):
            calls.append(kwargs)
            return profile_service._ResumeExtraction(
                pages=[],
                evidence=[
                    profile_service._EvidenceItem(
                        page_number=1,
                        skill="Python",
                        source_excerpt="Skills: Python",
                        origin="skill_list",
                    )
                ],
            ), LLMUsageMeta()

    class Embeddings(Extractor):
        used_mock_fallback = True
        last_embedding_usage = LLMUsageMeta(total_tokens=3, used_mock_fallback=True)

        def embed_texts(self, texts, **kwargs):
            return [[1.0] + [0.0] * 767 for _ in texts]

    monkeypatch.setattr(profile_service, "GeminiService", Extractor)
    monkeypatch.setattr(retrieval, "GeminiService", Embeddings)
    return calls


@pytest.fixture
def api(postgres, providers, client_runtime):
    app = create_app()

    def session():
        with Session(
            bind=postgres,
            autoflush=False,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        ) as db:
            yield db

    def inline(db, state, timeout):
        # Same graph and real SQL; only process isolation and Gemini are doubled.
        routes.execute_graph(db, state, timeout, llm_service=ReasoningDouble())

    app.dependency_overrides[routes.get_db] = session
    app.dependency_overrides[routes.get_search_executor] = lambda: inline
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, postgres


def seed_jobs(connection):
    with Session(bind=connection, join_transaction_mode="create_savepoint") as db:
        for number in range(1, 4):
            job_id = f"job_{number:03d}"
            raw = f"Remote - India. Python developer role {number}. Minimum experience is {3 if number == 2 else 0} years."
            if number != 3:
                raw += " Students eligible. A completed degree is not required."
            excerpts = slice_into_excerpts(job_id, raw)
            normalized = normalize_job_constraints(
                {"job_id": job_id, "raw_description": raw}, excerpts
            )
            normalized.pop("has_conflict")
            db.add(
                JobORM(
                    job_id=job_id,
                    title="Python developer",
                    company="Example",
                    source_url=f"https://example.invalid/{job_id}",
                    raw_description=raw,
                    excerpts=excerpts,
                    search_tsv=func.to_tsvector("english", raw),
                    **normalized,
                )
            )
        db.flush()
        for number in range(1, 4):
            db.add(
                JobMetadataORM(
                    job_id=f"job_{number:03d}",
                    captured_at=datetime(2026, 11, 19, tzinfo=UTC),
                    content_hash="fixture",
                    dedup_key=f"job_{number:03d}",
                    source_status="active",
                    is_duplicate=False,
                )
            )
        db.commit()


def create_draft(client):
    response = client.post(
        "/profiles", json={"name": "Candidate", "raw_text": SECRET + "\nSkills: Python"}
    )
    assert response.status_code == 201, response.text
    assert response.json()["is_confirmed"] is False
    assert response.json()["evidence"][0]["verification_status"] == "resume_claimed"
    return response.json()["profile_id"]


def test_postgresql_full_lifecycle_all_variants_and_feedback(api):
    client, connection = api
    seed_jobs(connection)
    assert client.get("/ready").status_code == 200
    profile_id = create_draft(client)
    assert (
        client.post(
            "/searches", json={"profile_id": profile_id, "user_query": "Python"}
        ).status_code
        == 409
    )
    confirmed = client.post(f"/profiles/{profile_id}/confirm", json=CONFIRM)
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["preferences"]["version"] == 1
    assert confirmed.json()["profile_version"] == 2
    assert {e["verification_status"] for e in confirmed.json()["evidence"]} == {
        "self_reported",
        "resume_claimed",
    }
    assert (
        client.post(f"/profiles/{profile_id}/confirm", json=CONFIRM).status_code == 409
    )
    patched = client.patch(
        f"/profiles/{profile_id}/preferences",
        json={"hard_constraints": {"max_required_yoe": 0.5}, "expected_version": 1},
    )
    assert patched.status_code == 200 and patched.json()["version"] == 2
    assert patched.json()["hard_constraints"]["allowed_work_modes"] == ["remote"]
    assert (
        client.patch(
            f"/profiles/{profile_id}/preferences",
            json={"soft_preferences": {"prefer_startup": True}, "expected_version": 1},
        ).status_code
        == 409
    )
    for variant in ("multi_agent", "single_agent", "retrieval_only"):
        response = client.post(
            "/searches",
            json={
                "profile_id": profile_id,
                "user_query": "Python",
                "system_variant": variant,
            },
        )
        assert response.status_code == 201, response.text
        run = response.json()
        assert run["status"] == "completed" and run["used_mock_fallback"] is True
        assert run["preference_version"] == 2 and run["profile_version"] == 2
        assert [r["job_id"] for r in run["rejected_counterfactuals"]] == ["job_002"]
        assert "job_003" in [r["job_id"] for r in run["needs_review_list"]]
        assert client.get(f"/searches/{run['run_id']}").json() == run
        trace = client.get(f"/searches/{run['run_id']}/trace")
        assert trace.status_code == 200 and SECRET not in trace.text
        steps = trace.json()["steps"]
        assert [s["step_index"] for s in steps] == list(range(len(steps)))
        assert any(s["payload_summary"].get("constraint_decisions") for s in steps)
        assert any(s["payload_summary"].get("job_ids") for s in steps)
        for relevance in ("relevant", "not_relevant"):
            feedback = client.post(
                "/feedback",
                json={
                    "run_id": run["run_id"],
                    "job_id": "job_001",
                    "relevance": relevance,
                },
            )
            assert feedback.status_code == 201
        assert (
            client.post(
                "/feedback",
                json={
                    "run_id": run["run_id"],
                    "job_id": "job_999",
                    "relevance": "relevant",
                },
            ).status_code
            == 422
        )
        with Session(
            bind=connection, join_transaction_mode="create_savepoint"
        ) as reader:
            persisted = reader.get(SearchRunORM, run["run_id"])
            assert (
                persisted.status == "completed" and persisted.used_mock_fallback is True
            )
            assert (
                len(
                    reader.scalars(
                        select(TraceORM).where(
                            TraceORM.run_id == run["run_id"],
                            TraceORM.tool_or_action == "feedback",
                        )
                    ).all()
                )
                == 2
            )


def test_postgresql_rejected_confirmation_rolls_back_and_selection_is_scoped(api):
    client, connection = api
    profile_id = create_draft(client)
    invalid = {**CONFIRM, "confirmed_evidence_ids": ["ev_foreign"]}
    assert (
        client.post(f"/profiles/{profile_id}/confirm", json=invalid).status_code == 422
    )
    with Session(bind=connection, join_transaction_mode="create_savepoint") as db:
        assert db.get(CandidateProfileORM, profile_id).is_confirmed is False
        assert db.scalar(select(func.count()).select_from(PreferenceORM)) == 0
        assert db.scalar(select(func.count()).select_from(CandidateEvidenceORM)) == 1
    response = client.post(
        f"/profiles/{profile_id}/confirm",
        json={**CONFIRM, "confirmed_evidence_ids": []},
    )
    assert response.status_code == 200
    assert len(response.json()["evidence"]) == 1
    assert response.json()["evidence"][0]["source_type"] == "user_statement"


def test_postgresql_pdf_upload_and_zero_result_search(api, providers):
    client, _ = api
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            fitz.Rect(36, 36, 550, 800),
            "Skills: Python\n"
            + "Candidate coursework and practical software exercises. " * 8,
            fontsize=12,
        )
        data = pdf.tobytes()
    draft = client.post(
        "/profiles",
        data={"name": "PDF Candidate"},
        files={"file": ("resume.pdf", data, "application/pdf")},
    )
    assert draft.status_code == 201, draft.text
    assert providers[-1]["image_bytes_list"] is None
    profile_id = draft.json()["profile_id"]
    assert (
        client.post(f"/profiles/{profile_id}/confirm", json=CONFIRM).status_code == 200
    )
    result = client.post(
        "/searches", json={"profile_id": profile_id, "user_query": "Python"}
    )
    assert result.status_code == 201, result.text
    assert result.json()["shortlist"] == result.json()["needs_review_list"] == []
    assert result.json()["status"] == "completed"


def test_postgresql_failed_search_remains_readable(api):
    client, _ = api
    profile_id = create_draft(client)
    assert (
        client.post(f"/profiles/{profile_id}/confirm", json=CONFIRM).status_code == 200
    )

    def timeout(*args):
        raise routes.SearchDeadlineExceeded()

    client.app.dependency_overrides[routes.get_search_executor] = lambda: timeout
    response = client.post(
        "/searches", json={"profile_id": profile_id, "user_query": "Python"}
    )
    assert response.status_code == 504
    run_id = response.json()["detail"]["run_id"]
    result = client.get(f"/searches/{run_id}").json()
    assert result["status"] == "failed" and result["shortlist"] == []
    assert result["usage_complete"] is False
    assert "hard_timeout" in client.get(f"/searches/{run_id}/trace").text
