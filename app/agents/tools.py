"""Synchronous, JSON-returning tools; callers own sessions and transactions.

Pass run_id to record a tool trace, or call log_trace explicitly for agent actions.
Reads use persisted column snapshots, never refresh/flush callers' pending edits.
Missing records raise LookupError; batches never silently omit requested jobs.
Discovery's total search budget and graph hydration belong to Steps 2.2/2.5.
"""

import math
import re
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    CandidateEvidenceORM,
    CandidateProfileORM,
    JobMetadataORM,
    JobORM,
    PreferenceORM,
    SearchRunORM,
    TraceORM,
)
from app.schemas.contracts import EvidenceRecord, HardConstraints, PreferenceState
from app.services.constraints import evaluate_job_constraints, validate_hard_constraints
from app.services.ingestion import slice_into_excerpts
from app.services.llm_client import GeminiRequestError, GeminiResponseError
from app.services.retrieval import hybrid_search_jobs


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip() or "\x00" in value or len(value) > 256:
        raise ValueError(
            f"{name} must be nonblank, NUL-free, and at most 256 characters"
        )
    return value


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= 2_147_483_647:
        raise ValueError(f"{name} is outside its supported range")
    return value


def _job_ids(job_ids: list[str]) -> list[str]:
    if not isinstance(job_ids, list):
        raise TypeError("job_ids must be a list")
    if len(job_ids) > min(10, settings.MAX_CANDIDATE_POOL):
        raise ValueError("job_ids exceeds the candidate pool limit")
    return list(dict.fromkeys(_identifier(item, "job_id") for item in job_ids))


def _job_rows(db: Session, job_ids: list[str]) -> list[tuple[JobORM, JobMetadataORM]]:
    ids = _job_ids(job_ids)
    if not ids:
        return []
    statement = (
        select(
            *(
                column.label(f"job_{column.name}")
                for column in JobORM.__table__.columns
            ),
            *(
                column.label(f"meta_{column.name}")
                for column in JobMetadataORM.__table__.columns
            ),
        )
        .select_from(JobORM)
        .outerjoin(JobMetadataORM, JobORM.job_id == JobMetadataORM.job_id)
        .where(JobORM.job_id.in_(ids))
    )
    with db.no_autoflush:
        rows = db.execute(statement).mappings().all()
    by_id = {row["job_job_id"]: row for row in rows}
    if any(
        job_id not in by_id or by_id[job_id]["meta_job_id"] is None for job_id in ids
    ):
        raise LookupError("A requested job or its listing metadata does not exist")
    return [
        (
            JobORM(
                **{
                    column.name: deepcopy(by_id[job_id][f"job_{column.name}"])
                    for column in JobORM.__table__.columns
                }
            ),
            JobMetadataORM(
                **{
                    column.name: by_id[job_id][f"meta_{column.name}"]
                    for column in JobMetadataORM.__table__.columns
                }
            ),
        )
        for job_id in ids
    ]


def _metadata(meta: JobMetadataORM) -> dict:
    return {
        "job_id": meta.job_id,
        "captured_at": meta.captured_at.isoformat(),
        "content_hash": meta.content_hash,
        "source_status": meta.source_status,
        "dedup_key": meta.dedup_key,
        "is_duplicate": meta.is_duplicate,
    }


@contextmanager
def _trace(db: Session, run_id: str | None, action: str):
    """Record safe metrics only. DB errors propagate without a second DB write.

    Error traces share the caller's transaction, so a rollback also rolls them back.
    No exception text (which could contain a prompt/key) is persisted.
    """
    summary = {}
    started = perf_counter()
    if run_id is not None:
        _identifier(run_id, "run_id")
        with db.no_autoflush:
            if (
                db.scalar(
                    select(SearchRunORM.run_id).where(SearchRunORM.run_id == run_id)
                )
                is None
            ):
                raise LookupError("Search run does not exist")
    try:
        yield summary
    except (
        ValueError,
        TypeError,
        LookupError,
        GeminiRequestError,
        GeminiResponseError,
    ):
        if run_id is not None:
            log_trace(
                db,
                run_id,
                agent_name="tools",
                tool_or_action=action,
                duration_ms=(perf_counter() - started) * 1000,
                outcome="error",
            )
        raise
    else:
        if run_id is not None:
            log_trace(
                db,
                run_id,
                agent_name="tools",
                tool_or_action=action,
                duration_ms=(perf_counter() - started) * 1000,
                outcome="success",
                tokens_used=summary.pop("tokens_used", 0),
                payload_summary=summary,
            )


def get_preferences(
    db: Session,
    profile_id: str,
    version: int | None = None,
    *,
    run_id: str | None = None,
) -> dict:
    """Read active preferences, or an exact historical version pinned by a run."""
    with _trace(db, run_id, "get_preferences") as summary:
        _identifier(profile_id, "profile_id")
        if version is not None:
            _integer(version, "version", 1)
        statement = select(*PreferenceORM.__table__.columns).where(
            PreferenceORM.profile_id == profile_id,
            PreferenceORM.is_active.is_(True)
            if version is None
            else PreferenceORM.version == version,
        )
        with db.no_autoflush:
            row = db.execute(statement).mappings().one_or_none()
        if row is None:
            raise LookupError("Requested preferences do not exist")
        result = PreferenceState(
            profile_id=row["profile_id"],
            version=row["version"],
            hard_constraints=row["hard_constraints"],
            soft_preferences=row["soft_preferences"],
            updated_at=row["created_at"].isoformat(),
        ).model_dump(mode="json")
        summary["preference_version"] = row["version"]
        return result


def search_jobs(
    db: Session,
    query: str,
    hard: HardConstraints,
    limit: int = 10,
    *,
    run_id: str | None = None,
) -> dict:
    """One bounded hybrid search; preserves embedding usage/mock provenance."""
    with _trace(db, run_id, "search_jobs") as summary:
        _integer(limit, "limit")
        result = hybrid_search_jobs(
            db, query, hard, limit=min(limit, 10, settings.MAX_CANDIDATE_POOL)
        )
        summary.update(
            result_count=len(result["eligible_candidates"]),
            rejected_count=len(result["rejected_counterfactuals"]),
            candidates_evaluated=result["candidates_evaluated"],
            used_mock_fallback=result["used_mock_fallback"],
            embedding_calls=result["embedding_calls"],
            tokens_used=(result["embedding_usage"] or {}).get("total_tokens", 0),
        )
        return result


def get_job_details(
    db: Session, job_ids: list[str], *, run_id: str | None = None
) -> list[dict]:
    """One SELECT for up to ten jobs, deduplicated in input order."""
    with _trace(db, run_id, "get_job_details") as summary:
        results = []
        for job, meta in _job_rows(db, job_ids):
            result = {
                column.name: deepcopy(getattr(job, column.name))
                for column in JobORM.__table__.columns
                if column.name != "search_tsv"
            }
            result["metadata"] = _metadata(meta)
            results.append(result)
        summary["result_count"] = len(results)
        return results


def get_candidate_evidence(
    db: Session,
    profile_id: str,
    profile_version: int | None = None,
    *,
    run_id: str | None = None,
) -> list[dict]:
    """Return cumulative evidence through the requested version, never newer facts.

    Reads are scoped to one profile; no arbitrary truncation of its evidence. Ordering
    is by revision then exact evidence_id, independent of DB insertion order.
    """
    with _trace(db, run_id, "get_candidate_evidence") as summary:
        _identifier(profile_id, "profile_id")
        if profile_version is not None:
            _integer(profile_version, "profile_version", 1)
        with db.no_autoflush:
            current = db.scalar(
                select(CandidateProfileORM.profile_version).where(
                    CandidateProfileORM.id == profile_id
                )
            )
            if current is None:
                raise LookupError("Candidate profile does not exist")
            version = current if profile_version is None else profile_version
            if version > current:
                raise ValueError("Requested profile version does not exist")
            rows = (
                db.execute(
                    select(*CandidateEvidenceORM.__table__.columns)
                    .where(
                        CandidateEvidenceORM.profile_id == profile_id,
                        CandidateEvidenceORM.profile_version <= version,
                    )
                    .order_by(
                        CandidateEvidenceORM.profile_version,
                        CandidateEvidenceORM.evidence_id,
                    )
                )
                .mappings()
                .all()
            )
        result = [
            EvidenceRecord.model_validate(
                {key: row[key] for key in EvidenceRecord.model_fields}
            ).model_dump(mode="json")
            for row in rows
        ]
        summary.update(result_count=len(result), profile_version=version)
        return result


def get_source_excerpt(
    db: Session, job_id: str, excerpt_id: str, *, run_id: str | None = None
) -> dict:
    """Resolve an exact job-scoped ID only when its text matches the source."""
    with _trace(db, run_id, "get_source_excerpt"):
        _identifier(excerpt_id, "excerpt_id")
        job, _ = _job_rows(db, [job_id])[0]
        canonical = slice_into_excerpts(job.job_id, job.raw_description)
        if not isinstance(job.excerpts, dict) or excerpt_id not in canonical:
            raise LookupError("Source excerpt does not exist")
        if job.excerpts.get(excerpt_id) != canonical[excerpt_id]:
            raise ValueError("Stored source excerpt does not match the listing")
        return {
            "job_id": job.job_id,
            "excerpt_id": excerpt_id,
            "text": canonical[excerpt_id],
        }


def check_constraints(
    db: Session, job_ids: list[str], hard: HardConstraints, *, run_id: str | None = None
) -> list[dict]:
    """Use the identical Phase 1 deterministic gate for every system variant."""
    with _trace(db, run_id, "check_constraints") as summary:
        hard = validate_hard_constraints(hard)
        result = []
        for job, meta in _job_rows(db, job_ids):
            overall, checks = evaluate_job_constraints(job, meta, hard)
            result.append(
                {
                    "job_id": job.job_id,
                    "result": overall.value,
                    "checks": [item.model_dump(mode="json") for item in checks],
                }
            )
        summary["result_count"] = len(result)
        return result


def get_listing_metadata(
    db: Session, job_ids: list[str], *, run_id: str | None = None
) -> list[dict]:
    with _trace(db, run_id, "get_listing_metadata") as summary:
        result = [_metadata(meta) for _, meta in _job_rows(db, job_ids)]
        summary["result_count"] = len(result)
        return result


def _safe_summary(payload: dict | None) -> dict:
    """Allowlist metrics/IDs; omit all arbitrary text and nested payloads."""
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("payload_summary must be a dictionary")
    safe = {}
    for key in (
        "result_count",
        "rejected_count",
        "candidates_evaluated",
        "embedding_calls",
        "model_calls",
        "correction_count",
        "preference_version",
        "profile_version",
    ):
        if key in payload:
            safe[key] = _integer(payload[key], key)
    for key in ("used_mock_fallback", "cached"):
        if key in payload:
            if not isinstance(payload[key], bool):
                raise TypeError(f"{key} must be a boolean")
            safe[key] = payload[key]
    if "job_ids" in payload:
        ids = _job_ids(payload["job_ids"])
        if any(re.fullmatch(r"job_[0-9]{3,}", item) is None for item in ids):
            raise ValueError("Trace job IDs must use the corpus ID format")
        safe["job_ids"] = ids
    return safe


def log_trace(
    db: Session,
    run_id: str,
    *,
    agent_name: str,
    tool_or_action: str,
    duration_ms: float = 0.0,
    outcome: str = "success",
    tokens_used: int = 0,
    payload_summary: dict | None = None,
) -> dict:
    """Append a zero-based ordered trace without committing caller work.

    A per-run row lock serializes concurrent writers; no process-local counter.
    Only allowlisted summary fields persist. Unknown keys (including résumé,
    prompt, evidence excerpts, credentials, and exception text) are redacted.
    """
    _identifier(run_id, "run_id")
    for name, value in (("agent_name", agent_name), ("tool_or_action", tool_or_action)):
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value) is None
        ):
            raise ValueError(f"{name} must be an action identifier, not free text")
    if outcome not in ("success", "error", "blocked"):
        raise ValueError("Unsupported trace outcome")
    _integer(tokens_used, "tokens_used")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        raise TypeError("duration_ms must be numeric")
    try:
        duration = float(duration_ms)
    except OverflowError as error:
        raise ValueError("duration_ms must be finite and nonnegative") from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_ms must be finite and nonnegative")
    payload = _safe_summary(payload_summary)
    with db.no_autoflush:
        run = db.scalar(
            select(SearchRunORM.run_id)
            .where(SearchRunORM.run_id == run_id)
            .with_for_update()
        )
        if run is None:
            raise LookupError("Search run does not exist")
        previous = db.scalar(
            select(func.max(TraceORM.step_index)).where(TraceORM.run_id == run_id)
        )
        index = _integer(0 if previous is None else previous + 1, "step_index")
        # A Core INSERT joins this session's transaction without flushing its
        # unit of work. Partial ORM flushes are deprecated; a full flush would
        # also persist unrelated pending inserts, updates, and deletes.
        trace_id = db.execute(
            TraceORM.__table__.insert()
            .values(
                run_id=run_id,
                step_index=index,
                agent_name=agent_name,
                tool_or_action=tool_or_action,
                duration_ms=duration,
                outcome=outcome,
                tokens_used=tokens_used,
                payload_summary=payload,
            )
            .returning(TraceORM.trace_id)
        ).scalar_one()
    return {
        "trace_id": trace_id,
        "run_id": run_id,
        "step_index": index,
        "agent_name": agent_name,
        "tool_or_action": tool_or_action,
        "duration_ms": duration,
        "outcome": outcome,
        "tokens_used": tokens_used,
        "payload_summary": deepcopy(payload),
    }
