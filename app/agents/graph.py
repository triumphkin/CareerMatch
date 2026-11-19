"""Bounded synchronous orchestration; sessions/clients never enter graph state.

All variants share Discovery and HydrateContext. Baseline call counts refer to
reasoning after that shared preparation. Counts include logical structured calls
and retrieval embeddings (cached/mock included), not provider-internal retries.
Timeouts are cooperative at operation boundaries: an in-flight synchronous SQL
or SDK request retains its own timeout. Late results are not accepted. The API
must provide process isolation if a strict wall-clock response limit is needed.

Public runners return a SearchRunORM persisted in the caller's transaction.
The caller owns commit/rollback, as with the existing agent tools.
"""

import json
import math
from copy import deepcopy
from time import monotonic
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from sqlalchemy import literal, select, update
from sqlalchemy.orm import Session

from app.agents.discovery import _query, _retrieval_output, plan_discovery_queries
from app.agents.matching import matching_node
from app.agents.tools import get_preferences, log_trace
from app.agents.validators import validate_match_assessment_integrity
from app.agents.verification import verification_node
from app.config import settings
from app.db.models import (
    CandidateEvidenceORM,
    CandidateProfileORM,
    JobMetadataORM,
    JobORM,
    SearchRunORM,
)
from app.schemas.contracts import (
    EvidenceRecord,
    HardConstraints,
    JobMatchAssessment,
    LLMUsageMeta,
    RejectedJobCounterfactual,
    SoftPreferences,
)
from app.services.constraints import evaluate_job_constraints
from app.services.llm_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiService,
)
from app.services.retrieval import hybrid_search_jobs

MAX_MODEL_CALLS_PER_RUN = 8
DEFAULT_TIMEOUT_SECONDS = 120.0


class CareerMatchGraphState(TypedDict, total=False):
    run_id: str
    profile_id: str
    profile_version: int
    preference_version: int
    user_query: str
    system_variant: str
    hard_constraints: dict
    soft_preferences: dict
    candidate_job_ids: list[str]
    hydrated_jobs: list[dict]
    candidate_evidence: list[dict]
    deterministic_constraints: list[dict]
    discovery_query_plan: dict
    assessments: list[dict]
    verification_feedback: list[dict]
    verification_route: str
    verification_verdicts: list[dict]
    verification_audit: list[dict]
    matching_integrity_issues: dict
    correction_count: int
    model_calls: int
    embedding_calls: int
    searches_used: int
    total_tokens: int
    estimated_cost_usd: float
    used_mock_fallback: bool
    usage_records: list[dict]
    status: str
    stop_reason: str
    assessments_ready: bool
    shortlist: list[dict]
    needs_review_list: list[dict]
    rejected_counterfactuals: list[dict]


class RunBudgetExceeded(RuntimeError):
    """A new operation cannot start within this run's bounded resources."""


class RunBudget:
    """Per-invocation runtime object; no global mutable counter or worker thread."""

    def __init__(self, max_calls, timeout_seconds, *, clock=monotonic):
        if type(max_calls) is not int or not 0 <= max_calls <= MAX_MODEL_CALLS_PER_RUN:
            raise ValueError("max_model_calls must be an integer between 0 and 8")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and nonnegative")
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + timeout_seconds
        self.max_calls = min(max_calls, settings.MAX_MODEL_CALLS_PER_RUN)
        self.calls = 0
        self.usage = []
        self.used_mock = False

    def check(self):
        if self.clock() >= self.deadline:
            raise RunBudgetExceeded("timeout")

    def reserve(self):
        self.check()
        if self.calls >= self.max_calls:
            raise RunBudgetExceeded("model_call_budget")
        self.calls += 1

    def record(self, usage):
        if usage is not None:
            item = LLMUsageMeta.model_validate(usage).model_dump(mode="json")
            self.usage.append(item)
            self.used_mock |= item["used_mock_fallback"]

    def metrics(self):
        return {
            "model_calls": self.calls,
            "usage_records": deepcopy(self.usage),
            "total_tokens": sum(item["total_tokens"] for item in self.usage),
            "estimated_cost_usd": sum(
                item["estimated_cost_usd"] for item in self.usage
            ),
            "used_mock_fallback": self.used_mock,
        }


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False))


def hydrate_context(db: Session, state: CareerMatchGraphState) -> dict:
    """One SELECT for jobs, metadata and cumulative pinned evidence; no N+1 reads."""
    ids = state["candidate_job_ids"]
    if len(ids) != len(set(ids)) or len(ids) > min(10, settings.MAX_CANDIDATE_POOL):
        raise ValueError("Candidate IDs must be unique and within the pool limit")
    anchor = select(literal(1).label("anchor")).subquery()
    statement = (
        select(
            *(c.label(f"job_{c.name}") for c in JobORM.__table__.columns),
            *(c.label(f"meta_{c.name}") for c in JobMetadataORM.__table__.columns),
            *(c.label(f"ev_{c.name}") for c in CandidateEvidenceORM.__table__.columns),
        )
        .select_from(anchor)
        .outerjoin(JobORM, JobORM.job_id.in_(ids))
        .outerjoin(JobMetadataORM, JobMetadataORM.job_id == JobORM.job_id)
        .outerjoin(
            CandidateEvidenceORM,
            (CandidateEvidenceORM.profile_id == state["profile_id"])
            & (CandidateEvidenceORM.profile_version <= state["profile_version"]),
        )
        .order_by(
            CandidateEvidenceORM.profile_version, CandidateEvidenceORM.evidence_id
        )
    )
    with db.no_autoflush:
        rows = db.execute(statement).mappings().all()
    jobs, evidence = {}, {}
    for row in rows:
        if row["ev_evidence_id"] is not None:
            evidence[row["ev_evidence_id"]] = EvidenceRecord.model_validate(
                {key: row[f"ev_{key}"] for key in EvidenceRecord.model_fields}
            ).model_dump(mode="json")
        if row["job_job_id"] is not None:
            jobs[row["job_job_id"]] = row
    if set(jobs) != set(ids) or any(jobs[i]["meta_job_id"] is None for i in ids):
        raise LookupError(
            "A requested job or its metadata disappeared before hydration"
        )
    hard = HardConstraints.model_validate(state["hard_constraints"])
    hydrated, reports = [], []
    rejected = {item["job_id"]: item for item in state["rejected_counterfactuals"]}
    for job_id in ids:
        row = jobs[job_id]
        job = JobORM(**{c.name: row[f"job_{c.name}"] for c in JobORM.__table__.columns})
        meta = JobMetadataORM(
            **{c.name: row[f"meta_{c.name}"] for c in JobMetadataORM.__table__.columns}
        )
        result, checks = evaluate_job_constraints(job, meta, hard)
        report = {
            "job_id": job_id,
            "result": result.value,
            "checks": [item.model_dump(mode="json") for item in checks],
        }
        if result == "fail":
            failed = [item for item in checks if item.result == "fail"]
            rejected[job_id] = RejectedJobCounterfactual(
                job_id=job_id,
                title=job.title,
                company=job.company,
                similarity_score=0.0,
                failed_constraints=failed,
                rejection_summary=" ".join(item.reason for item in failed),
            ).model_dump(mode="json")
            continue
        detail = {
            c.name: getattr(job, c.name)
            for c in JobORM.__table__.columns
            if c.name != "search_tsv"
        }
        detail["metadata"] = {
            c.name: getattr(meta, c.name) for c in JobMetadataORM.__table__.columns
        }
        detail["metadata"]["captured_at"] = meta.captured_at.isoformat()
        hydrated.append(detail)
        reports.append(report)
    return _snapshot(
        {
            "candidate_job_ids": [job["job_id"] for job in hydrated],
            "hydrated_jobs": hydrated,
            "candidate_evidence": list(evidence.values()),
            "deterministic_constraints": reports,
            "rejected_counterfactuals": list(rejected.values())[:5],
        }
    )


def deterministic_assessments(state, *, unfinished=False):
    """Eligibility-only results, with no fabricated match claims or rank score."""
    reports = {item["job_id"]: item for item in state["deterministic_constraints"]}
    results = []
    for job in state["hydrated_jobs"]:
        report = reports[job["job_id"]]
        decision = "accept" if report["result"] == "pass" else "needs_review"
        if unfinished:
            decision = "needs_review"
        results.append(
            JobMatchAssessment(
                **{
                    key: job[key]
                    for key in ("job_id", "title", "company", "source_url")
                },
                captured_at=job["metadata"]["captured_at"],
                decision=decision,
                supported_matches=[],
                gaps=[],
                unknowns=["Reasoning or verification did not finish."]
                if unfinished
                else [],
                constraint_checks=report["checks"],
            ).model_dump(mode="json")
        )
    return results


def finalize_results(state):
    """Reapply trusted reports/citation integrity for every variant before persistence."""
    state = _snapshot(state)
    trusted = {a["job_id"]: a for a in deterministic_assessments(state)}
    jobs = {job["job_id"]: job for job in state["hydrated_jobs"]}
    reports = {r["job_id"]: r for r in state["deterministic_constraints"]}
    evidence = {item["evidence_id"]: item for item in state["candidate_evidence"]}
    if not state["assessments_ready"]:
        assessments = deterministic_assessments(state, unfinished=True)
    else:
        assessments = state["assessments"]
    ids = [item["job_id"] for item in assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(trusted):
        raise ValueError("Final assessments must cover the hydrated pool exactly once")
    shortlist, review = [], []
    rejected = {item["job_id"]: item for item in state["rejected_counterfactuals"]}
    for raw in assessments:
        item = JobMatchAssessment.model_validate(raw).model_dump(mode="json")
        job_id = item["job_id"]
        base = trusted[job_id]
        issues = validate_match_assessment_integrity(
            item, evidence, jobs[job_id]["excerpts"], reports[job_id]["checks"]
        )
        issues += state.get("matching_integrity_issues", {}).get(job_id, [])
        # An unfinished revision and mock reasoning cannot produce a supported accept.
        if (
            issues
            or item["decision"] == "revise"
            or (
                state["used_mock_fallback"]
                and state["system_variant"] != "retrieval_only"
            )
        ):
            item = deepcopy(base)
            item.update(
                decision="needs_review", unknowns=["Assessment requires review."]
            )
        for key in (
            "title",
            "company",
            "source_url",
            "captured_at",
            "constraint_checks",
        ):
            item[key] = base[key]
        if reports[job_id]["result"] == "unknown" and item["decision"] == "accept":
            item["decision"] = "needs_review"
        if item["decision"] == "accept":
            shortlist.append(item)
        elif item["decision"] == "needs_review":
            review.append(item)
        else:
            rejected[job_id] = RejectedJobCounterfactual(
                job_id=job_id,
                title=base["title"],
                company=base["company"],
                similarity_score=item["internal_rank_score"],
                failed_constraints=[],
                rejection_summary="Reasoning rejected this deterministically eligible job.",
            ).model_dump(mode="json")
    # Stable sort keeps retrieval order for equal scores (including retrieval-only).
    shortlist.sort(key=lambda item: -item["internal_rank_score"])
    review.sort(key=lambda item: -item["internal_rank_score"])
    return {
        "shortlist": shortlist[: min(5, settings.MAX_SHORTLIST)],
        "needs_review_list": review,
        "rejected_counterfactuals": list(rejected.values())[:5],
        "status": ("partial" if shortlist or review or rejected else "failed")
        if state.get("stop_reason")
        else "completed",
    }


def build_graph(db, *, budget, llm_service=None):
    """Compile a fresh graph with run-local runtime dependencies captured in closures."""

    class MeteredService:
        def generate_structured(self, **arguments):
            budget.reserve()
            budget.used_mock |= (
                bool(llm_service.used_mock_fallback)
                if llm_service is not None
                else not bool(settings.GEMINI_API_KEY)
            )
            if llm_service is None:
                with GeminiService(config=settings) as service:
                    output = service.generate_structured(**arguments)
            else:
                output = llm_service.generate_structured(**arguments)
            # Charge usage before agent validation can reject the response.
            if isinstance(output, (tuple, list)) and len(output) == 2:
                budget.record(output[1])
            budget.check()
            return output

    metered = MeteredService()

    def guarded(name, operation):
        def node(state):
            state = _snapshot(state)
            started = budget.clock()
            try:
                budget.check()
                changes = operation(state)
                budget.check()
                state.update(changes)
            except (
                RunBudgetExceeded,
                GeminiRequestError,
                GeminiResponseError,
                TimeoutError,
                ConnectionError,
                ValueError,
                TypeError,
                LookupError,
            ) as error:
                # Never persist raw exceptions, prompts, credentials or résumé text.
                state["stop_reason"] = (
                    str(error)
                    if isinstance(error, RunBudgetExceeded)
                    else type(error).__name__
                )
            state.update(budget.metrics())
            log_trace(
                db,
                state["run_id"],
                agent_name=name,
                tool_or_action="run_node",
                outcome="blocked" if state.get("stop_reason") else "success",
                duration_ms=max(0, (budget.clock() - started) * 1000),
                payload_summary={
                    "model_calls": budget.calls,
                    "correction_count": state["correction_count"],
                    "used_mock_fallback": budget.used_mock,
                },
            )
            return _snapshot(state)

        return node

    def discover(state):
        if settings.MAX_DISCOVERY_SEARCHES == 0:
            raise RunBudgetExceeded("search_budget")
        plan, _usage = plan_discovery_queries(
            state["user_query"],
            state["soft_preferences"],
            llm_service=metered,
        )
        budget.check()
        state["discovery_query_plan"] = plan.model_dump(mode="json")
        selected, rejected = {}, {}
        for query in (plan.primary_query, plan.fallback_broad_query)[
            : min(2, settings.MAX_DISCOVERY_SEARCHES)
        ]:
            if state["searches_used"] and len(selected) >= 5:
                break
            budget.reserve()  # At most one embedding per hybrid search.
            result = _retrieval_output(
                hybrid_search_jobs(
                    db,
                    query.strip(),
                    HardConstraints.model_validate(state["hard_constraints"]),
                    limit=min(10, settings.MAX_CANDIDATE_POOL),
                )
            )
            budget.calls -= 1 - result["embedding_calls"]
            budget.record(result["embedding_usage"])
            budget.used_mock |= result["used_mock_fallback"]
            state["embedding_calls"] += result["embedding_calls"]
            state["searches_used"] += 1
            budget.check()
            for item in result["rejected"]:
                rejected.setdefault(item["job_id"], item)
                selected.pop(item["job_id"], None)
            for job_id in result["ids"]:
                if job_id not in rejected:
                    selected.setdefault(job_id, None)
            state["candidate_job_ids"] = list(selected)[
                : min(10, settings.MAX_CANDIDATE_POOL)
            ]
            state["rejected_counterfactuals"] = list(rejected.values())[:5]
        return state

    def match(state):
        state["assessments_ready"] = False
        result = matching_node(state, llm_service=metered)
        result.pop("matching_usage")
        result.pop("model_calls")
        budget.used_mock |= result.pop("used_mock_fallback")
        result["assessments_ready"] = state["system_variant"] == "single_agent"
        return result

    def verify(state):
        result = verification_node(state, llm_service=metered)
        result.pop("verification_usage")
        result.pop("model_calls")
        budget.used_mock |= result.pop("used_mock_fallback")
        result["assessments_ready"] = True
        return result

    def retrieve_only(state):
        return {
            "assessments": deterministic_assessments(state),
            "assessments_ready": True,
        }

    def finish(state):
        state.update(budget.metrics())
        result = finalize_results(state)
        values = {
            **result,
            "total_tokens": state["total_tokens"],
            "estimated_cost_usd": state["estimated_cost_usd"],
            "used_mock_fallback": state["used_mock_fallback"],
            "latency_ms": max(0, (budget.clock() - budget.started) * 1000),
        }
        with db.no_autoflush:
            db.execute(
                update(SearchRunORM)
                .where(SearchRunORM.run_id == state["run_id"])
                .values(**values)
            )
        return result

    workflow = StateGraph(CareerMatchGraphState)
    for name, operation in (
        ("discovery", discover),
        ("hydrate_context", lambda s: hydrate_context(db, s)),
        ("matching", match),
        ("verification", verify),
        ("retrieval_only", retrieve_only),
    ):
        workflow.add_node(name, guarded(name, operation))
    workflow.add_node("finalize", finish)
    workflow.add_edge(START, "discovery")
    workflow.add_conditional_edges(
        "discovery", lambda s: "finalize" if s.get("stop_reason") else "hydrate_context"
    )
    workflow.add_conditional_edges(
        "hydrate_context",
        lambda s: (
            "finalize"
            if s.get("stop_reason")
            else "retrieval_only"
            if s["system_variant"] == "retrieval_only"
            else "matching"
        ),
    )
    workflow.add_conditional_edges(
        "matching",
        lambda s: (
            "finalize"
            if s.get("stop_reason") or s["system_variant"] == "single_agent"
            else "verification"
        ),
    )
    workflow.add_conditional_edges(
        "verification",
        lambda s: "finalize" if s.get("stop_reason") else s["verification_route"],
    )
    workflow.add_edge("retrieval_only", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_search(
    db: Session,
    profile_id: str,
    user_query: str,
    *,
    system_variant="multi_agent",
    llm_service=None,
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    max_model_calls=MAX_MODEL_CALLS_PER_RUN,
) -> SearchRunORM:
    """Pin persisted profile/preferences, execute one bounded graph, return its row.

    Shared preparation uses one planning call and up to two embedding calls.
    Single-agent adds exactly one matching call for a nonempty pool. Multi-agent
    adds matching/verification and at most one pair of correction calls (max 7).
    """
    if system_variant not in ("multi_agent", "single_agent", "retrieval_only"):
        raise ValueError("Unknown system_variant")
    if (
        not isinstance(profile_id, str)
        or not profile_id.strip()
        or "\x00" in profile_id
    ):
        raise ValueError("profile_id must be nonblank and NUL-free")
    budget = RunBudget(max_model_calls, timeout_seconds)
    with db.no_autoflush:
        version = db.scalar(
            select(CandidateProfileORM.profile_version).where(
                CandidateProfileORM.id == profile_id
            )
        )
    if version is None:
        raise LookupError("Candidate profile does not exist")
    preferences = get_preferences(db, profile_id)
    query = _query(
        user_query, SoftPreferences.model_validate(preferences["soft_preferences"])
    )
    run_id = str(uuid4())
    with db.no_autoflush:
        db.execute(
            SearchRunORM.__table__.insert().values(
                run_id=run_id,
                profile_id=profile_id,
                profile_version=version,
                preference_version=preferences["version"],
                query=query,
                system_variant=system_variant,
                status="running",
            )
        )
    state = CareerMatchGraphState(
        run_id=run_id,
        profile_id=profile_id,
        profile_version=version,
        preference_version=preferences["version"],
        user_query=query,
        system_variant=system_variant,
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
    build_graph(db, budget=budget, llm_service=llm_service).invoke(
        state, {"recursion_limit": 16}
    )
    with db.no_autoflush:
        return db.scalars(
            select(SearchRunORM).where(SearchRunORM.run_id == run_id)
        ).one()


def run_multi_agent(
    db: Session, profile_id: str, user_query: str, **kwargs
) -> SearchRunORM:
    return run_search(
        db, profile_id, user_query, system_variant="multi_agent", **kwargs
    )
