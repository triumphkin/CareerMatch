"""HTTP contracts, transactions, redacted traces, and isolated search workers.

Feedback is an append-only TraceORM event, scoped to a job returned by a run.
Search workers receive JSON state and connection settings, never a live session.
The parent terminates overdue workers before marking the persisted run failed.
"""

import json
import math
import multiprocessing
import re
from datetime import datetime
from time import monotonic
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import create_engine, delete, func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.agents.graph import CareerMatchGraphState, RunBudget, build_graph
from app.agents.tools import get_candidate_evidence, get_preferences
from app.config import settings
from app.db.models import (
    CandidateEvidenceORM,
    CandidateProfileORM,
    JobEmbeddingORM,
    JobMetadataORM,
    JobORM,
    PreferenceORM,
    SearchRunORM,
    TraceORM,
)
from app.db.session import SessionLocal
from app.schemas.contracts import (
    EvidenceRecord,
    HardConstraints,
    JobMatchAssessment,
    PreferenceState,
    RejectedJobCounterfactual,
    SoftPreferences,
)
from app.services import profile_service
from app.services.constraints import validate_hard_constraints

router = APIRouter()
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
JobIdentifier = Annotated[str, Field(pattern=r"^job_[0-9]{3,}$", max_length=128)]
Variant = Literal["multi_agent", "single_agent", "retrieval_only"]


def get_db():
    """One session per request; mutation endpoints explicitly commit their work."""
    with SessionLocal() as db:
        yield db


Db = Annotated[Session, Depends(get_db)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @field_validator("*", mode="after")
    @classmethod
    def no_nul(cls, value):
        def visit(item):
            if isinstance(item, str) and "\x00" in item:
                raise ValueError("NUL characters are not supported")
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(key)
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value.model_dump() if isinstance(value, BaseModel) else value)
        return value


class DraftInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    raw_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("name", "raw_text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A nonblank value is required")
        return value


class UserStatement(InputModel):
    skill: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=5000)

    @field_validator("skill", "statement")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A nonblank value is required")
        return value.strip()


class ConfirmInput(InputModel):
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    confirmed_evidence_ids: list[Identifier] | None = None
    self_reported_skills: list[UserStatement] = Field(
        default_factory=list, max_length=50
    )

    @model_validator(mode="after")
    def unique(self):
        if self.confirmed_evidence_ids is not None and len(
            self.confirmed_evidence_ids
        ) != len(set(self.confirmed_evidence_ids)):
            raise ValueError("Duplicate evidence IDs")
        skills = [item.skill.casefold() for item in self.self_reported_skills]
        if len(skills) != len(set(skills)):
            raise ValueError("Duplicate self-reported skills")
        validate_hard_constraints(self.hard_constraints)
        _validate_soft(self.soft_preferences)
        return self


class PreferencePatch(InputModel):
    hard_constraints: dict | None = None
    soft_preferences: dict | None = None
    reason: str = Field(default="API preference update", min_length=1, max_length=1000)
    expected_version: int | None = Field(default=None, ge=1, le=2**31 - 1)

    @model_validator(mode="after")
    def nonempty(self):
        if (
            not (self.hard_constraints or self.soft_preferences)
            or not self.reason.strip()
        ):
            raise ValueError("A nonempty preference patch and reason are required")
        return self


class SearchInput(InputModel):
    profile_id: Identifier
    user_query: str = Field(min_length=1, max_length=4000)
    system_variant: Variant = "multi_agent"
    timeout_seconds: float = Field(default=120.0, gt=0, le=120, allow_inf_nan=False)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def numeric_timeout(cls, value):
        if isinstance(value, bool):
            # Pydantic turns ValueError into a 422; TypeError would escape as 500.
            raise ValueError("Timeout must be numeric, not boolean")  # noqa: TRY004
        return value

    @field_validator("user_query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A nonblank query is required")
        return value.strip()


class FeedbackInput(InputModel):
    run_id: Identifier
    job_id: JobIdentifier
    relevance: Literal["relevant", "not_relevant"]


class ProfileResponse(BaseModel):
    profile_id: str
    name: str
    profile_version: int
    is_confirmed: bool
    evidence: list[EvidenceRecord]


class ConfirmResponse(ProfileResponse):
    preferences: PreferenceState


class SearchResponse(BaseModel):
    run_id: str
    profile_id: str
    profile_version: int
    preference_version: int
    query: str
    system_variant: Variant
    status: Literal["running", "completed", "partial", "failed"]
    shortlist: list[JobMatchAssessment]
    needs_review_list: list[JobMatchAssessment]
    rejected_counterfactuals: list[RejectedJobCounterfactual]
    latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    used_mock_fallback: bool
    created_at: datetime
    usage_complete: bool


class FeedbackResponse(FeedbackInput):
    feedback_id: str


def _validate_soft(soft):
    if any(
        not item.strip() or "\x00" in item
        for item in soft.target_roles + soft.preferred_skills
    ):
        raise ValueError("Roles and skills must be nonblank and NUL-free")
    return soft


async def draft_input(request: Request):
    """Parse JSON text or multipart PDF/text; all DB work stays in sync routes."""
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    try:
        if content_type == "application/json":
            parsed = DraftInput.model_validate(await request.json())
            return parsed.name, None, parsed.raw_text
        if content_type not in (
            "multipart/form-data",
            "application/x-www-form-urlencoded",
        ):
            raise HTTPException(415, "Use JSON or multipart/form-data")
        async with request.form(
            max_files=1, max_fields=3, max_part_size=MAX_TEXT_CHARS
        ) as form:
            if set(form) - {"name", "raw_text", "file"} or any(
                len(form.getlist(key)) != 1 for key in form
            ):
                raise HTTPException(422, "Unexpected or duplicate form fields")
            name, raw, upload = form.get("name"), form.get("raw_text"), form.get("file")
            if upload is None:
                parsed = DraftInput.model_validate({"name": name, "raw_text": raw})
                return parsed.name, None, parsed.raw_text
            if raw is not None or not isinstance(upload, UploadFile):
                raise HTTPException(422, "Supply exactly one PDF file or raw_text")
            if upload.content_type != "application/pdf":
                raise HTTPException(415, "File must be a PDF")
            parsed = DraftInput.model_validate({"name": name, "raw_text": "PDF"})
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "PDF exceeds the upload limit")
            if not data or not data.startswith(b"%PDF-"):
                raise HTTPException(422, "Invalid PDF")
            return parsed.name, data, None
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(422, "Invalid profile input") from None


def _profile(db, profile_id, *, lock=False):
    query = select(CandidateProfileORM).where(CandidateProfileORM.id == profile_id)
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    profile = db.scalar(query)
    if profile is None:
        raise HTTPException(404, "Profile not found")
    return profile


def _profile_view(db, profile):
    return {
        "profile_id": profile.id,
        "name": profile.name,
        "profile_version": profile.profile_version,
        "is_confirmed": profile.is_confirmed,
        "evidence": get_candidate_evidence(db, profile.id, profile.profile_version),
    }


@router.post(
    "/profiles",
    status_code=201,
    response_model=ProfileResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": DraftInput.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "raw_text": {"type": "string"},
                            "file": {"type": "string", "format": "binary"},
                        },
                        "oneOf": [{"required": ["raw_text"]}, {"required": ["file"]}],
                    }
                },
            },
        },
    },
)
def create_profile(db: Db, payload: Annotated[tuple, Depends(draft_input)]):
    name, pdf, raw = payload
    profile = profile_service.create_profile(db, name, pdf_bytes=pdf, raw_text=raw)
    response = _profile_view(db, profile)
    db.commit()
    return response


@router.post("/profiles/{profile_id}/confirm", response_model=ConfirmResponse)
def confirm_profile(profile_id: Identifier, body: ConfirmInput, db: Db):
    profile = _profile(db, profile_id, lock=True)
    if profile.is_confirmed or db.scalar(
        select(PreferenceORM.id).where(PreferenceORM.profile_id == profile_id).limit(1)
    ):
        raise HTTPException(409, "Profile is already confirmed or has preferences")
    existing = set(
        db.scalars(
            select(CandidateEvidenceORM.evidence_id).where(
                CandidateEvidenceORM.profile_id == profile_id
            )
        ).all()
    )
    if body.confirmed_evidence_ids is not None:
        approved = set(body.confirmed_evidence_ids)
        if not approved <= existing:
            raise HTTPException(422, "Evidence IDs must belong to this draft")
        db.execute(
            delete(CandidateEvidenceORM).where(
                CandidateEvidenceORM.profile_id == profile_id,
                CandidateEvidenceORM.evidence_id.in_(existing - approved),
            )
        )
    for item in body.self_reported_skills:
        profile_service.add_user_statement_evidence(
            db, profile_id, item.skill, item.statement
        )
    preferences = profile_service.update_preferences(
        db,
        profile_id,
        body.hard_constraints,
        body.soft_preferences,
        "Initial confirmation",
    )
    profile.is_confirmed = True
    db.flush()
    response = {
        **_profile_view(db, profile),
        "preferences": preferences.model_dump(mode="json"),
    }
    db.commit()
    return response


@router.patch("/profiles/{profile_id}/preferences", response_model=PreferenceState)
def patch_preferences(profile_id: Identifier, body: PreferencePatch, db: Db):
    if not _profile(db, profile_id, lock=True).is_confirmed:
        raise HTTPException(409, "Confirm the profile before updating preferences")
    current = get_preferences(db, profile_id)
    if (
        body.expected_version is not None
        and current["version"] != body.expected_version
    ):
        raise HTTPException(409, "Preference version has changed")
    try:
        hard = validate_hard_constraints(
            HardConstraints.model_validate(
                {**current["hard_constraints"], **(body.hard_constraints or {})}
            )
        )
        soft = _validate_soft(
            SoftPreferences.model_validate(
                {**current["soft_preferences"], **(body.soft_preferences or {})}
            )
        )
    except (ValidationError, ValueError):
        raise HTTPException(422, "Invalid preference patch") from None
    result = profile_service.update_preferences(db, profile_id, hard, soft, body.reason)
    db.commit()
    return result.model_dump(mode="json")


def _event(db, run_id, action, payload, *, outcome="success"):
    """Serialize event sequence numbers with the same per-run lock as agent tools."""
    if (
        db.scalar(
            select(SearchRunORM.run_id)
            .where(SearchRunORM.run_id == run_id)
            .with_for_update()
        )
        is None
    ):
        raise LookupError("Search run not found")
    previous = db.scalar(
        select(func.max(TraceORM.step_index)).where(TraceORM.run_id == run_id)
    )
    if previous is not None and previous >= 2**31 - 1:
        raise OverflowError("Trace sequence exceeds the PostgreSQL integer range")
    row = TraceORM(
        run_id=run_id,
        agent_name="api",
        tool_or_action=action,
        step_index=0 if previous is None else previous + 1,
        outcome=outcome,
        payload_summary=payload,
        duration_ms=0.0,
        tokens_used=0,
    )
    db.add(row)
    db.flush()
    return row


def _new_search(db, body, request_id):
    profile = _profile(db, body.profile_id, lock=True)
    if not profile.is_confirmed:
        raise HTTPException(409, "Confirm the profile before searching")
    preferences = get_preferences(db, body.profile_id)
    run = SearchRunORM(
        run_id=str(uuid4()),
        profile_id=profile.id,
        profile_version=profile.profile_version,
        preference_version=preferences["version"],
        query=body.user_query,
        system_variant=body.system_variant,
        status="running",
    )
    db.add(run)
    db.flush()
    _event(db, run.run_id, "search_requested", {"request_id": request_id})
    state = CareerMatchGraphState(
        run_id=run.run_id,
        profile_id=run.profile_id,
        profile_version=run.profile_version,
        preference_version=run.preference_version,
        user_query=run.query,
        system_variant=run.system_variant,
        hard_constraints=preferences["hard_constraints"],
        soft_preferences=preferences["soft_preferences"],
        candidate_job_ids=[],
        hydrated_jobs=[],
        candidate_evidence=[],
        deterministic_constraints=[],
        assessments=[],
        verification_feedback=[],
        matching_integrity_issues={},
        correction_count=0,
        model_calls=0,
        embedding_calls=0,
        searches_used=0,
        used_mock_fallback=False,
        total_tokens=0,
        estimated_cost_usd=0.0,
        usage_records=[],
        status="running",
        stop_reason="",
        assessments_ready=False,
        shortlist=[],
        needs_review_list=[],
        rejected_counterfactuals=[],
    )
    db.commit()  # The isolated worker must see this pinned run and preferences.
    return state


_GATES = {
    "listing_status",
    "work_mode_and_location",
    "student_eligibility",
    "experience_limit",
    "internal_consistency",
}


def _snapshot_payload(state):
    """No arbitrary text from model output, candidate evidence, queries or exceptions."""
    return {
        "job_ids": list(state.get("candidate_job_ids", [])),
        "constraint_decisions": [
            {
                "job_id": report["job_id"],
                "result": report["result"],
                "checks": [
                    {"field": check["field"], "result": check["result"]}
                    for check in report["checks"]
                ],
            }
            for report in state.get("deterministic_constraints", [])
        ],
        "correction_count": state.get("correction_count", 0),
        "model_calls": state.get("model_calls", 0),
        "used_mock_fallback": state.get("used_mock_fallback", False),
    }


def execute_graph(db, state, timeout_seconds, *, llm_service=None):
    """Run inside a worker transaction, recording only structured graph snapshots."""
    workflow = build_graph(
        db, budget=RunBudget(8, timeout_seconds), llm_service=llm_service
    )
    for updates in workflow.stream(
        state, {"recursion_limit": 16}, stream_mode="updates"
    ):
        for node, values in updates.items():
            if node != "finalize":
                _event(db, state["run_id"], "graph_snapshot", _snapshot_payload(values))
    db.commit()


def _search_worker(sender, database_url, schema_map, state, timeout_seconds):
    """Spawn target. A killed process cannot continue SQL writes or provider retries."""
    engine = None
    try:
        milliseconds = max(1, math.ceil(timeout_seconds * 1000))
        engine = create_engine(
            database_url,
            hide_parameters=True,
            connect_args={
                "connect_timeout": max(1, min(5, math.ceil(timeout_seconds))),
                "options": f"-c statement_timeout={milliseconds} -c lock_timeout={milliseconds}",
            },
        )
        with Session(
            bind=engine.execution_options(schema_translate_map=schema_map),
            autoflush=False,
            expire_on_commit=False,
        ) as db:
            execute_graph(db, state, timeout_seconds)
        sender.send("completed")
    except (SQLAlchemyError, ValueError, TypeError, LookupError, RuntimeError, OSError):
        # Process boundary: return an explicit failure without a traceback that
        # could expose input/credentials. The parent persists the failure event.
        sender.send("worker_failed")
    finally:
        sender.close()
        if engine is not None:
            engine.dispose()


class SearchDeadlineExceeded(TimeoutError):
    """An isolated search worker was terminated at the request deadline."""


class SearchWorkerFailed(RuntimeError):
    """A worker exited without a successful committed result."""


def run_isolated_search(db, state, timeout_seconds, *, worker=_search_worker):
    """Enforce wall-clock work deadline, including spawn; no sessions cross processes."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    bind = db.get_bind()
    process = context.Process(
        target=worker,
        args=(
            sender,
            bind.url.render_as_string(hide_password=False),
            bind.get_execution_options().get("schema_translate_map"),
            state,
            timeout_seconds,
        ),
        daemon=True,
    )
    started = monotonic()
    launched = False
    try:
        process.start()
        launched = True
        sender.close()
        if not receiver.poll(max(0, timeout_seconds - (monotonic() - started))):
            raise SearchDeadlineExceeded()
        try:
            outcome = receiver.recv()
        except EOFError:
            raise SearchWorkerFailed() from None
        if outcome != "completed":
            raise SearchWorkerFailed()
    finally:
        receiver.close()
        sender.close()
        if launched:
            if process.is_alive():
                process.kill()
            process.join(timeout=1)
            process.close()


def get_search_executor():
    return run_isolated_search


def _run(db, run_id):
    run = db.scalar(
        select(SearchRunORM)
        .where(SearchRunORM.run_id == run_id)
        .execution_options(populate_existing=True)
    )
    if run is None:
        raise HTTPException(404, "Search not found")
    return run


def _run_view(db, run):
    incomplete_usage = (
        db.scalar(
            select(TraceORM.trace_id)
            .where(
                TraceORM.run_id == run.run_id,
                TraceORM.tool_or_action.in_(("hard_timeout", "worker_failed")),
            )
            .limit(1)
        )
        is not None
    )
    return {
        **{
            column.name: getattr(run, column.name)
            for column in SearchRunORM.__table__.columns
        },
        "usage_complete": not incomplete_usage and run.status == "completed",
    }


@router.post("/searches", status_code=201, response_model=SearchResponse)
def create_search(
    body: SearchInput,
    request: Request,
    db: Db,
    executor: Annotated[object, Depends(get_search_executor)],
):
    state = _new_search(db, body, request.state.request_id)
    started = monotonic()
    try:
        executor(
            db,
            state,
            min(body.timeout_seconds, request.app.state.search_timeout_seconds),
        )
    except (SearchDeadlineExceeded, SearchWorkerFailed, OSError) as error:
        action = (
            "hard_timeout"
            if isinstance(error, SearchDeadlineExceeded)
            else "worker_failed"
        )
        # The worker has stopped before this update; it cannot overwrite failure later.
        db.execute(
            update(SearchRunORM)
            .where(
                SearchRunORM.run_id == state["run_id"], SearchRunORM.status == "running"
            )
            .values(
                status="failed",
                latency_ms=(monotonic() - started) * 1000,
                used_mock_fallback=not bool(settings.GEMINI_API_KEY),
            )
        )
        _event(
            db,
            state["run_id"],
            action,
            {"request_id": request.state.request_id},
            outcome="error",
        )
        db.commit()
        raise HTTPException(
            504 if action == "hard_timeout" else 503,
            {"code": action, "run_id": state["run_id"], "usage_complete": False},
        ) from None
    return _run_view(db, _run(db, state["run_id"]))


@router.get("/searches/{run_id}", response_model=SearchResponse)
def get_search(run_id: Identifier, db: Db):
    return _run_view(db, _run(db, run_id))


def redact_payload(payload):
    """Allowlist structured values, including legacy/untrusted persisted payloads."""
    if not isinstance(payload, dict):
        return {}
    safe = {}
    for key in (
        "model_calls",
        "correction_count",
        "result_count",
        "rejected_count",
        "candidates_evaluated",
        "embedding_calls",
        "preference_version",
        "profile_version",
    ):
        value = payload.get(key)
        if type(value) is int and 0 <= value <= 2**31 - 1:
            safe[key] = value
    for key in ("used_mock_fallback", "cached"):
        if type(payload.get(key)) is bool:
            safe[key] = payload[key]
    if isinstance(payload.get("job_ids"), list):
        safe["job_ids"] = list(
            dict.fromkeys(
                item
                for item in payload["job_ids"]
                if isinstance(item, str)
                and re.fullmatch(r"job_[0-9]{3,}", item)
                and len(item) <= 128
            )
        )[:10]
    if payload.get("relevance") in ("relevant", "not_relevant"):
        safe["relevance"] = payload["relevance"]
    decisions = payload.get("constraint_decisions")
    if isinstance(decisions, list):
        safe["constraint_decisions"] = []
        for item in decisions[:10]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("job_id"), str)
                or not re.fullmatch(r"job_[0-9]{3,}", item["job_id"])
            ):
                continue
            if item.get("result") not in ("pass", "fail", "unknown") or not isinstance(
                item.get("checks"), list
            ):
                continue
            safe["constraint_decisions"].append(
                {
                    "job_id": item["job_id"],
                    "result": item["result"],
                    "checks": [
                        {"field": c["field"], "result": c["result"]}
                        for c in item["checks"]
                        if isinstance(c, dict)
                        and isinstance(c.get("field"), str)
                        and c["field"] in _GATES
                        and c.get("result") in ("pass", "fail", "unknown")
                    ],
                }
            )
    return safe


@router.get("/searches/{run_id}/trace")
def get_trace(run_id: Identifier, db: Db):
    _run(db, run_id)
    rows = db.scalars(
        select(TraceORM)
        .where(TraceORM.run_id == run_id)
        .order_by(TraceORM.step_index, TraceORM.trace_id)
    ).all()
    agents = {
        "api",
        "tools",
        "discovery",
        "hydrate_context",
        "matching",
        "verification",
        "retrieval_only",
        "finalize",
    }
    actions = {
        "run_node",
        "search_requested",
        "graph_snapshot",
        "hard_timeout",
        "worker_failed",
        "feedback",
        "get_preferences",
        "search_jobs",
        "get_job_details",
        "get_candidate_evidence",
        "get_source_excerpt",
        "check_constraints",
        "get_listing_metadata",
        "plan_queries",
        "select_candidates",
    }
    return {
        "run_id": run_id,
        "steps": [
            {
                "step_index": row.step_index,
                "agent_name": row.agent_name
                if row.agent_name in agents
                else "redacted",
                "tool_or_action": row.tool_or_action
                if row.tool_or_action in actions
                else "redacted",
                "duration_ms": row.duration_ms,
                "tokens_used": row.tokens_used,
                "outcome": row.outcome
                if row.outcome in ("success", "error", "blocked")
                else "redacted",
                "payload_summary": redact_payload(row.payload_summary),
            }
            for row in rows
        ],
    }


@router.post("/feedback", status_code=201, response_model=FeedbackResponse)
def feedback(body: FeedbackInput, db: Db):
    run = _run(db, body.run_id)
    if run.status == "running":
        raise HTTPException(409, "Search is still running")
    returned = {
        item["job_id"]
        for item in run.shortlist + run.needs_review_list + run.rejected_counterfactuals
    }
    if body.job_id not in returned:
        raise HTTPException(422, "Feedback job must belong to this search's results")
    event = _event(
        db,
        body.run_id,
        "feedback",
        {"job_ids": [body.job_id], "relevance": body.relevance},
    )
    response = {"feedback_id": event.trace_id, **body.model_dump()}
    db.commit()
    return response


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Db):
    if db.get_bind().dialect.name != "postgresql":
        raise HTTPException(503, "PostgreSQL is required")
    try:
        version = int(db.scalar(text("SHOW server_version_num")))
        vector = db.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        # Query the actual application tables too, so an uninitialized DB is not ready.
        for model in (
            CandidateProfileORM,
            CandidateEvidenceORM,
            PreferenceORM,
            JobORM,
            JobMetadataORM,
            JobEmbeddingORM,
            SearchRunORM,
            TraceORM,
        ):
            db.execute(select(model).limit(0))
    except SQLAlchemyError:
        raise HTTPException(503, "Database is not ready") from None
    if not (160000 <= version < 170000) or not vector:
        raise HTTPException(503, "PostgreSQL 16 and pgvector are required")
    return {"status": "ready", "postgresql_major": 16, "pgvector": True}
