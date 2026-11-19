"""Deterministic integrity tests and real PostgreSQL tool integration.

Provider doubles below test forwarding only; no live Gemini quality is claimed.
PostgreSQL tests fail explicitly if unavailable, never substitute SQLite.
"""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.agents import tools
from app.agents.validators import validate_match_assessment_integrity as validate
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
from app.schemas.contracts import EvidenceRecord, HardConstraints, JobMatchAssessment
from app.services.constraints import evaluate_job_constraints
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts
from app.services.llm_client import GeminiRequestError

GATES = [
    "listing_status",
    "work_mode_and_location",
    "student_eligibility",
    "experience_limit",
    "internal_consistency",
]


def evidence(evidence_id="ev_001", status="self_reported", profile_id="profile_1"):
    return {
        "evidence_id": evidence_id,
        "profile_id": profile_id,
        "skill": "Python",
        "source_type": "user_statement"
        if status == "self_reported"
        else "resume_statement",
        "source_excerpt": "I know Python.",
        "source_id": "statement_1",
        "verification_status": status,
    }


@pytest.fixture
def bundle():
    checks = [
        {
            "field": field,
            "result": "pass",
            "reason": "Deterministic gate passed.",
            "source_excerpt_ids": [],
        }
        for field in GATES
    ]
    assessment = {
        "job_id": "job_001",
        "title": "Developer",
        "company": "Example",
        "source_url": "https://example.invalid/jobs/1",
        "captured_at": "2026-11-19T00:00:00Z",
        "decision": "accept",
        "supported_matches": [
            {
                "requirement": "Python",
                "candidate_evidence_ids": ["ev_001"],
                "job_excerpt_ids": ["job_001_ex_01"],
                "explanation": "Python is self-reported.",
                "is_self_reported_only": True,
            }
        ],
        "gaps": [],
        "unknowns": [],
        "constraint_checks": copy.deepcopy(checks),
    }
    return (
        assessment,
        {"ev_001": evidence()},
        {"job_001_ex_01": "Python required."},
        checks,
    )


def test_valid_assessment_dict_and_models_are_accepted_without_mutation(bundle):
    before = copy.deepcopy(bundle)
    assert validate(*bundle) == []
    a, ev, ex, gates = bundle
    assert (
        validate(
            JobMatchAssessment.model_validate(a),
            {key: EvidenceRecord.model_validate(value) for key, value in ev.items()},
            ex,
            gates,
        )
        == []
    )
    assert bundle == before
    json.dumps(bundle)


@pytest.mark.parametrize("location", ["candidate", "job", "check", "trusted_check"])
def test_fabricated_citations_anywhere_fail(bundle, location):
    a, _, _, checks = bundle
    if location == "candidate":
        a["supported_matches"][0]["candidate_evidence_ids"] = ["ev_999"]
    elif location == "job":
        a["supported_matches"][0]["job_excerpt_ids"] = ["job_001_ex_999"]
    elif location == "check":
        a["constraint_checks"][0]["source_excerpt_ids"] = ["job_001_ex_999"]
    else:
        checks[0]["source_excerpt_ids"] = ["job_001_ex_999"]
    assert validate(*bundle)


def test_other_job_citation_is_rejected_even_if_present_in_allowlist(bundle):
    bundle[2]["job_002_ex_01"] = "Python required."
    bundle[0]["supported_matches"][0]["job_excerpt_ids"] = ["job_002_ex_01"]
    assert any("wrong-job" in issue for issue in validate(*bundle))


@pytest.mark.parametrize("suffix", ["00", "1", "02", "9" * 5000])
def test_exact_ids_no_guessing_renumbering_or_numeric_overflow(bundle, suffix):
    bundle[0]["supported_matches"][0]["job_excerpt_ids"] = [f"job_001_ex_{suffix}"]
    assert validate(*bundle)


@pytest.mark.parametrize(
    "text",
    [
        "GitHub proves their Python skills.",
        "Their project demonstrates Python.",
        "Project-backed Python expertise.",
        "Verified in their repository.",
        "No GitHub proof, but the project demonstrates Python.",
        "ＧｉｔＨｕｂ proves Python expertise.",
        "They built a project with Python.",
    ],
)
def test_self_reported_cannot_claim_project_proof(bundle, text):
    bundle[0]["supported_matches"][0]["explanation"] = text
    assert any("claims GitHub/project proof" in issue for issue in validate(*bundle))


@pytest.mark.parametrize(
    "text",
    [
        "Python is self-reported; no GitHub proof.",
        "Self-reported, without project evidence.",
        "Python is not project-backed.",
        "GitHub is not verified.",
        "Python is not independently verified on GitHub.",
        "GitHub proof is not available; Python is self-reported.",
        "Interested in GitHub projects; skill is self-reported.",
    ],
)
def test_honest_provenance_disclaimers_are_allowed(bundle, text):
    bundle[0]["supported_matches"][0]["explanation"] = text
    assert validate(*bundle) == []


def test_missing_self_reported_flag_is_an_issue_not_silent_rewrite(bundle):
    match = bundle[0]["supported_matches"][0]
    match["is_self_reported_only"] = False
    assert any("must be True" in issue for issue in validate(*bundle))
    assert match["is_self_reported_only"] is False


def test_mixed_evidence_and_resume_claimed_do_not_gain_project_provenance(bundle):
    bundle[1]["ev_002"] = evidence("ev_002", "resume_claimed")
    match = bundle[0]["supported_matches"][0]
    match.update(
        candidate_evidence_ids=["ev_001", "ev_002"],
        is_self_reported_only=False,
        explanation="Their GitHub proves Python expertise.",
    )
    assert any("claims GitHub/project proof" in issue for issue in validate(*bundle))
    bundle[1]["ev_002"]["verification_status"] = "project_backed"
    assert validate(*bundle) == []


@pytest.mark.parametrize("result", ["fail", "unknown"])
@pytest.mark.parametrize("decision", ["accept", "needs_review", "reject", "revise"])
def test_accept_never_overrides_deterministic_gate(bundle, result, decision):
    bundle[0]["decision"] = decision
    bundle[3][3]["result"] = result
    bundle[0]["constraint_checks"] = copy.deepcopy(bundle[3])
    issues = validate(*bundle)
    assert bool(issues) is (decision == "accept")
    if issues:
        assert any(f"forbidden by deterministic {result}" in issue for issue in issues)


def test_model_cannot_replace_unknown_with_pass(bundle):
    bundle[3][3]["result"] = "unknown"
    issues = validate(*bundle)
    assert any("forbidden" in issue for issue in issues)
    assert any("contradicts" in issue for issue in issues)


@pytest.mark.parametrize("bad", [None, [], {}, [None], ["pass"]])
def test_missing_or_malformed_constraint_reports_fail_closed(bundle, bad):
    assert validate(*bundle[:3], bad)


def test_duplicate_gate_cannot_hide_fail(bundle):
    bundle[3][0]["result"] = "fail"
    bundle[3].append({**bundle[3][0], "result": "pass"})
    issues = validate(*bundle)
    assert any("duplicate" in issue for issue in issues)
    assert any("forbidden" in issue for issue in issues)


@pytest.mark.parametrize("bad", [None, {}, [], "text"])
def test_malformed_assessment_fails_closed(bundle, bad):
    assert validate(bad, *bundle[1:]) == ["assessment: invalid schema"]


def test_unvalidated_nested_model_copies_cannot_bypass_gate(bundle):
    a = JobMatchAssessment.model_validate(bundle[0])
    a.supported_matches[0] = a.supported_matches[0].model_copy(
        update={"candidate_evidence_ids": [None]}
    )
    assert validate(a, *bundle[1:]) == ["assessment: invalid schema"]


@pytest.mark.parametrize("field", ["candidate_evidence_ids", "job_excerpt_ids"])
def test_empty_supported_claim_citations_fail(bundle, field):
    bundle[0]["supported_matches"][0][field] = []
    assert validate(*bundle)


def test_duplicate_citations_are_idempotent(bundle):
    match = bundle[0]["supported_matches"][0]
    match["candidate_evidence_ids"] *= 2
    match["job_excerpt_ids"] *= 2
    assert validate(*bundle) == []


@pytest.mark.parametrize("bad", [None, {}, {"evidence_id": "ev_001"}])
def test_invalid_evidence_record_fails_closed(bundle, bad):
    bundle[1]["ev_001"] = bad
    assert validate(*bundle)


def test_evidence_key_mismatch_and_cross_profile_citations_fail(bundle):
    bundle[1]["ev_001"]["evidence_id"] = "ev_002"
    assert validate(*bundle)
    bundle[1]["ev_001"]["evidence_id"] = "ev_001"
    bundle[1]["ev_002"] = evidence("ev_002", profile_id="someone_else")
    bundle[0]["supported_matches"][0]["candidate_evidence_ids"].append("ev_002")
    assert any("multiple candidate profiles" in issue for issue in validate(*bundle))


def test_validator_is_repeatable_and_thread_safe(bundle):
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: validate(*bundle), range(32))) == [[]] * 32


def test_empty_job_batches_do_no_database_work():
    hard = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    assert tools.get_job_details(None, []) == []
    assert tools.check_constraints(None, [], hard) == []
    assert tools.get_listing_metadata(None, []) == []


@pytest.mark.parametrize("bad", [None, "job_001", [None], [""], ["x\x00"], ["x"] * 11])
def test_job_batch_validation_precedes_database_access(bad):
    with pytest.raises((TypeError, ValueError)):
        tools.get_job_details(None, bad)


def test_search_bounds_and_usage_forwarding(monkeypatch):
    expected = {
        "eligible_candidates": [],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 0,
        "embedding_calls": 1,
        "used_mock_fallback": True,
        "embedding_usage": {"total_tokens": 7},
    }
    seen = []
    hard = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])

    def fake(db, query, passed_hard, limit):
        seen.append((query, passed_hard, limit))
        return expected

    monkeypatch.setattr(tools, "hybrid_search_jobs", fake)
    assert tools.search_jobs(None, "Python", hard, limit=100) is expected
    assert seen == [("Python", hard, min(10, tools.settings.MAX_CANDIDATE_POOL))]


@pytest.mark.parametrize("query,limit", [("", 10), ("  ", 10), ("Python", 0)])
def test_empty_search_is_real_service_noop_without_provider(query, limit):
    result = tools.search_jobs(
        None, query, HardConstraints(allowed_work_modes=[], allowed_cities=[]), limit
    )
    assert result["eligible_candidates"] == []
    assert result["embedding_calls"] == 0
    assert result["used_mock_fallback"] is False


@pytest.mark.parametrize("bad", [True, -1, 1.5, None, 2**31])
def test_invalid_search_limits_reject_before_provider(bad):
    with pytest.raises((TypeError, ValueError)):
        tools.search_jobs(None, "Python", None, bad)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("duration_ms", -1),
        ("duration_ms", float("nan")),
        ("duration_ms", float("inf")),
        ("duration_ms", 10**400),
        ("duration_ms", True),
        ("tokens_used", True),
        ("tokens_used", -1),
        ("tokens_used", 2**31),
        ("agent_name", "raw resume text"),
        ("outcome", "unexpected"),
        ("payload_summary", []),
    ],
)
def test_trace_validation_before_database_access(field, bad):
    args = {"agent_name": "Matching", "tool_or_action": "assess", field: bad}
    with pytest.raises((TypeError, ValueError)):
        tools.log_trace(None, "run_1", **args)


def test_trace_summary_redacts_raw_resume_evidence_prompts_and_credentials():
    unsafe = {
        "raw_resume_text": "PRIVATE",
        "prompt": "PRIVATE",
        "GEMINI_API_KEY": "SECRET",
        "nested": {"raw_resume_text": "PRIVATE"},
        "source_excerpt": "PRIVATE",
        "result_count": 0,
        "job_ids": ["job_001", "job_001"],
        "used_mock_fallback": False,
    }
    assert tools._safe_summary(unsafe) == {
        "result_count": 0,
        "job_ids": ["job_001"],
        "used_mock_fallback": False,
    }


@pytest.fixture
def db():
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
            schema = f"agent_tools_test_{uuid4().hex}"
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
def seeded(db):
    now = datetime(2026, 11, 19, tzinfo=UTC)
    db.add_all(
        [
            CandidateProfileORM(id="profile_1", name="Candidate", profile_version=2),
            CandidateProfileORM(id="other", name="Other"),
            CandidateProfileORM(id="empty", name="Empty"),
        ]
    )
    db.flush()
    hard = HardConstraints(allowed_work_modes=["remote"], allowed_cities=[])
    for version in (1, 2):
        db.add(
            PreferenceORM(
                profile_id="profile_1",
                version=version,
                is_active=version == 2,
                hard_constraints=hard.model_dump(mode="json"),
                soft_preferences={
                    "target_roles": [f"role_{version}"],
                    "preferred_skills": [],
                },
            )
        )
        db.add(
            CandidateEvidenceORM(
                **evidence(f"ev_{version:03d}"), profile_version=version
            )
        )
    db.add(
        CandidateEvidenceORM(
            **evidence("ev_other", profile_id="other"), profile_version=1
        )
    )
    for number in (1, 2):
        job_id = f"job_{number:03d}"
        raw = "Remote from India. Students are eligible. No completed degree required. 0 years of experience required. Python required."
        excerpts = slice_into_excerpts(job_id, raw)
        norm = normalize_job_constraints(
            {"job_id": job_id, "raw_description": raw}, excerpts
        )
        norm.pop("has_conflict")
        db.add(
            JobORM(
                job_id=job_id,
                title="Developer",
                company="Example",
                source_url=f"https://example.invalid/{job_id}",
                raw_description=raw,
                excerpts=excerpts,
                **norm,
            )
        )
    db.flush()
    for number in (1, 2):
        job_id = f"job_{number:03d}"
        db.add(
            JobMetadataORM(
                job_id=job_id,
                captured_at=now,
                content_hash="hash",
                source_status="active",
                is_duplicate=False,
                dedup_key=job_id,
            )
        )
    db.add(
        SearchRunORM(
            run_id="run_1",
            profile_id="profile_1",
            profile_version=1,
            preference_version=1,
            query="Python",
            system_variant="multi_agent",
            status="running",
        )
    )
    db.flush()
    return hard


def test_postgresql_versioned_preferences_and_cumulative_evidence(db, seeded):
    assert tools.get_preferences(db, "profile_1")["version"] == 2
    old = tools.get_preferences(db, "profile_1", 1)
    assert old["soft_preferences"]["target_roles"] == ["role_1"]
    assert [
        item["evidence_id"] for item in tools.get_candidate_evidence(db, "profile_1", 1)
    ] == ["ev_001"]
    assert [
        item["evidence_id"] for item in tools.get_candidate_evidence(db, "profile_1")
    ] == ["ev_001", "ev_002"]
    assert tools.get_candidate_evidence(db, "empty") == []
    for function, args in [
        (tools.get_preferences, ("profile_1", 3)),
        (tools.get_preferences, ("missing",)),
        (tools.get_candidate_evidence, ("missing",)),
        (tools.get_candidate_evidence, ("profile_1", 3)),
    ]:
        with pytest.raises((LookupError, ValueError)):
            function(db, *args)
    json.dumps(old)


def test_postgresql_batched_reads_order_missing_and_shared_gate(db, seeded):
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", record)
    try:
        details = tools.get_job_details(db, ["job_002", "job_001", "job_002"])
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", record)
    assert len(statements) == 1
    assert [job["job_id"] for job in details] == ["job_002", "job_001"]
    reports = tools.check_constraints(db, ["job_001", "job_002"], seeded)
    for report in reports:
        job = db.get(JobORM, report["job_id"])
        meta = db.get(JobMetadataORM, report["job_id"])
        overall, checks = evaluate_job_constraints(job, meta, seeded)
        assert report == {
            "job_id": job.job_id,
            "result": overall.value,
            "checks": [check.model_dump(mode="json") for check in checks],
        }
    assert tools.get_listing_metadata(db, ["job_001"])[0]["source_status"] == "active"
    with pytest.raises(LookupError):
        tools.get_job_details(db, ["job_001", "missing"])
    # Bound parameters treat SQL-looking input as an ID, not executable SQL.
    with pytest.raises(LookupError):
        tools.get_job_details(db, ["job_001'); DROP TABLE jobs; --"])
    assert tools.get_job_details(db, ["job_001"])
    json.dumps([details, reports])


def test_postgresql_exact_source_excerpt_and_tampering(db, seeded):
    result = tools.get_source_excerpt(db, "job_001", "job_001_ex_01")
    assert result["text"] == "Remote from India."
    with pytest.raises(LookupError):
        tools.get_source_excerpt(db, "job_001", "job_002_ex_01")
    with pytest.raises(LookupError):
        tools.get_source_excerpt(db, "job_001", "job_001_ex_00")
    job = db.get(JobORM, "job_001")
    job.excerpts = {**job.excerpts, "job_001_ex_01": "Fabricated"}
    db.flush()
    with pytest.raises(ValueError, match="does not match"):
        tools.get_source_excerpt(db, "job_001", "job_001_ex_01")


@pytest.mark.filterwarnings("error::sqlalchemy.exc.SADeprecationWarning")
def test_postgresql_tools_and_traces_preserve_unsaved_edits(db, seeded):
    job = db.get(JobORM, "job_001")
    deleted = db.get(CandidateProfileORM, "empty")
    pending = CandidateProfileORM(id="pending_profile", name="Not persisted yet")
    job.title = "Pending title"
    db.add(pending)
    db.delete(deleted)
    db.autoflush = True
    detail = tools.get_job_details(db, ["job_001"], run_id="run_1")[0]
    assert detail["title"] == "Developer"
    assert job.title == "Pending title" and job in db.dirty
    with db.no_autoflush:
        assert (
            db.scalar(select(JobORM.title).where(JobORM.job_id == "job_001"))
            == "Developer"
        )
    assert tools.check_constraints(db, ["job_001"], seeded, run_id="run_1")
    assert job in db.dirty
    assert pending in db.new
    assert deleted in db.deleted
    with db.no_autoflush:
        assert (
            db.scalar(
                select(CandidateProfileORM.id).where(
                    CandidateProfileORM.id == pending.id
                )
            )
            is None
        )
        assert (
            db.scalar(
                select(CandidateProfileORM.id).where(
                    CandidateProfileORM.id == deleted.id
                )
            )
            == "empty"
        )
        traces = db.scalars(select(TraceORM).order_by(TraceORM.step_index)).all()
    assert [trace.step_index for trace in traces] == [0, 1]


@pytest.mark.filterwarnings("error::sqlalchemy.exc.SADeprecationWarning")
def test_postgresql_trace_order_redaction_errors_and_transaction_ownership(
    db, seeded, monkeypatch
):
    first = tools.log_trace(
        db,
        "run_1",
        agent_name="Matching",
        tool_or_action="assess",
        payload_summary={
            "raw_resume_text": "PRIVATE",
            "prompt": "SECRET",
            "result_count": 0,
        },
    )
    assert first["step_index"] == 0
    assert db.get(TraceORM, first["trace_id"]).step_index == 0
    assert first["payload_summary"] == {"result_count": 0}
    tools.get_preferences(db, "profile_1", run_id="run_1")
    tools.get_candidate_evidence(db, "profile_1", run_id="run_1")
    tools.get_listing_metadata(db, ["job_001"], run_id="run_1")
    tools.get_source_excerpt(db, "job_001", "job_001_ex_01", run_id="run_1")

    def fail(*args, **kwargs):
        raise GeminiRequestError("PRIVATE PROVIDER DIAGNOSTIC")

    monkeypatch.setattr(tools, "hybrid_search_jobs", fail)
    with pytest.raises(GeminiRequestError):
        tools.search_jobs(db, "PRIVATE QUERY", seeded, run_id="run_1")
    rows = db.scalars(select(TraceORM).order_by(TraceORM.step_index)).all()
    assert [row.step_index for row in rows] == list(range(6))
    assert rows[-1].outcome == "error"
    assert "PRIVATE" not in json.dumps([row.payload_summary for row in rows])
    with pytest.raises(LookupError):
        tools.log_trace(db, "missing", agent_name="Matching", tool_or_action="assess")
    # Each test uses a savepoint: this rollback must also remove the new traces.
    db.rollback()
    assert db.scalar(select(TraceORM.trace_id)) is None
