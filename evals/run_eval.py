"""Sequential, isolated three-way evaluation with physical-request accounting.

Run: uv run python -m evals.run_eval
Semantic claim scores require human annotations (--claim-reviews JSON mapping
claim_review_id to total_claims, unsupported_claims, reviewer, reviewed_at).
Without annotations the rate is null with explicit coverage, never a guessed 0.
All counts include failed searches; ranking uses final phases only. Corpus setup
cost is separate from search cost but consumes the SAME session ceiling.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median
from time import monotonic
from uuid import uuid4

from google.genai import types
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from app.config import settings
from app.schemas.contracts import (
    EvidenceRecord,
    HardConstraints,
    JobMatchAssessment,
    PreferenceState,
    RejectedJobCounterfactual,
    SoftPreferences,
)
from app.services.llm_client import (
    CostAccountingError,
    CostBudgetExceeded,
    GeminiRequestError,
    GeminiResponseError,
    GeminiService,
    RequestAccounting,
    account_requests,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("retrieval_only", "single_agent", "multi_agent")
OUTPUT_KEYS = ("shortlist", "needs_review_list", "rejected_counterfactuals")
COUNT_METRICS = (
    "relevant_at_5_normal",
    "relevant_at_5_abstention",
    "returned_precision",
    "hard_constraint_violation_rate",
    "unsupported_claim_rate",
    "unknown_handling_accuracy",
    "abstention_accuracy",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def execution_digest(scenarios):
    """Review sign-off can change without changing the evaluated experiment."""
    return digest([{k: v for k, v in s.items() if k != "review"} for s in scenarios])


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def load_scenarios(path=ROOT / "evals/scenarios_30.json"):
    scenarios = json.loads(Path(path).read_text())
    if not isinstance(scenarios, list) or len(scenarios) != 30:
        raise ValueError("Expected exactly 30 scenarios")
    if [s["scenario_id"] for s in scenarios] != [f"sc_{i:03d}" for i in range(1, 31)]:
        raise ValueError("Scenario IDs must be unique and ordered sc_001..sc_030")
    if [s["split"] for s in scenarios] != ["dev"] * 22 + ["held_out"] * 8:
        raise ValueError("Expected the fixed 22 dev / 8 held-out split")
    for scenario in scenarios:
        corpus = scenario["corpus"]
        path = (ROOT / corpus["path"]).resolve()
        if not path.is_relative_to(ROOT / "data/fixtures"):
            raise ValueError("Corpus must be a local fixture")
        if hashlib.sha256(path.read_bytes()).hexdigest() != corpus["sha256"]:
            raise ValueError("Corpus hash mismatch")
        raw = json.loads(path.read_text())
        inventory = {job["job_id"] for job in raw}
        ids = corpus["fixture_job_ids"]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= inventory:
            raise ValueError("Invalid fixture job IDs")
        profile = scenario["profile"]
        prefs = PreferenceState.model_validate(scenario["preferences"])
        if prefs.profile_id != profile["profile_id"] or not profile["is_confirmed"]:
            raise ValueError("Profile/preference identity or confirmation mismatch")
        evidence_ids = []
        for entry in profile["evidence"]:
            record = EvidenceRecord.model_validate(entry["record"])
            if (
                record.profile_id != prefs.profile_id
                or not 1 <= entry["profile_version"] <= profile["profile_version"]
            ):
                raise ValueError("Evidence identity/version mismatch")
            evidence_ids.append(record.evidence_id)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Duplicate evidence IDs")
        searches = [s for s in scenario["steps"] if s["op"] == "run_search"]
        if not searches or searches[-1]["phase"] != "final":
            raise ValueError("Scenario must end with a final search")
        for step in searches:
            expected = scenario[step["expected_key"]]
            if set(expected["constraint_results"]) != set(ids):
                raise ValueError("Every fixture needs a constraint label")
    return scenarios


def expected_for(scenario, phase, variant):
    step = next(s for s in scenario["steps"] if s.get("phase") == phase)
    expected = deepcopy(scenario[step["expected_key"]])
    expected.update(expected.pop("variant_overrides", {}).get(variant, {}))
    return expected


def account_totals(records):
    known = [r["usage"] for r in records if r["usage"] is not None]
    return {
        "estimated_cost_usd": float(
            sum((Decimal(str(u["estimated_cost_usd"])) for u in known), Decimal(0))
        ),
        "reserved_or_spent_usd": float(
            sum((Decimal(str(r["reserved_usd"])) for r in records), Decimal(0))
        ),
        "total_tokens": sum(u["total_tokens"] for u in known),
        "usage_complete": len(known) == len(records),
        "used_mock_fallback": any(u["used_mock_fallback"] for u in known),
        "cached_calls": sum(u["cached"] for u in known),
        "physical_calls": sum(
            r.get(
                "physical_request",
                r["usage"] is None
                or (not r["usage"]["cached"] and not r["usage"]["used_mock_fallback"]),
            )
            for r in records
        ),
    }


def score_run(scenario, run, claim_reviews=None):
    """Explicit gold counts; citation integrity is NOT semantic entailment."""
    if claim_reviews is not None and not isinstance(claim_reviews, dict):
        raise ValueError("Claim reviews must be an object keyed by review fingerprint")
    expected = expected_for(scenario, run["phase"], run["variant"])
    outputs = {key: run.get(key, []) for key in OUTPUT_KEYS}
    all_ids = []
    for key, rows in outputs.items():
        model = (
            RejectedJobCounterfactual
            if key == "rejected_counterfactuals"
            else JobMatchAssessment
        )
        for item in rows:
            model.model_validate(item)
            if key != "rejected_counterfactuals" and item["decision"] != (
                "accept" if key == "shortlist" else "needs_review"
            ):
                raise ValueError("Decision does not match result list")
            all_ids.append(item["job_id"])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate job IDs in returned results")
    accepted = [row["job_id"] for row in outputs["shortlist"]]
    review = {row["job_id"] for row in outputs["needs_review_list"]}
    if len(accepted) > 5:
        raise ValueError("Shortlist exceeds top-five contract")
    constraints = expected["constraint_results"]
    if not set(all_ids) <= set(constraints):
        raise ValueError("Result contains a job outside the scenario corpus")
    relevant = set(expected["relevant_job_ids"])
    unknown = {key for key, value in constraints.items() if value == "unknown"}
    hydrated = set(run.get("hydrated_job_ids", []))
    faults = scenario["fault_injection"]
    applicable = bool(faults and run["variant"] in faults["applicable_variants"])
    checks = {
        "status": run["status"] in expected["run_statuses"],
        "shortlist_size": expected["shortlist_min_count"]
        <= len(accepted)
        <= expected["shortlist_max_count"],
        "allowlist": set(accepted) <= set(expected["allowed_shortlist_job_ids"]),
        "forbidden_accepts": not set(accepted)
        & set(expected["must_not_accept_job_ids"]),
        "required_review": set(expected["required_review_job_ids"]) <= review,
        "forbidden_retrieval": not hydrated
        & set(expected["must_not_retrieve_job_ids"]),
        "required_hydration": set(expected.get("must_hydrate_job_ids", [])) <= hydrated,
        "fault_coverage": not applicable or run.get("fault_injected", False),
        "correction_limit": run.get("correction_count", 0)
        <= expected.get("max_corrections", 1),
        "preference_version": run.get("preference_version")
        == expected.get("preference_version", scenario["preferences"]["version"]),
    }
    evidence_ids = {e["record"]["evidence_id"] for e in scenario["profile"]["evidence"]}
    from app.services.ingestion import slice_into_excerpts

    raw = json.loads((ROOT / scenario["corpus"]["path"]).read_text())
    excerpts = {
        j["job_id"]: slice_into_excerpts(j["job_id"], j["raw_description"]) for j in raw
    }
    citation_checks = []
    citation_ids = []
    for key, rows in outputs.items():
        for item in rows:
            for check in item[
                "failed_constraints"
                if key == "rejected_counterfactuals"
                else "constraint_checks"
            ]:
                ids = check["source_excerpt_ids"]
                if ids:
                    citation_checks.append(set(ids) <= set(excerpts[item["job_id"]]))
                    citation_ids.extend(ids)
    for key in ("shortlist", "needs_review_list"):
        for item in outputs[key]:
            for match in item["supported_matches"]:
                ev, job = match["candidate_evidence_ids"], match["job_excerpt_ids"]
                citation_checks.append(
                    bool(ev and job)
                    and set(ev) <= evidence_ids
                    and set(job) <= set(excerpts[item["job_id"]])
                )
                citation_ids.extend(ev + job)
    checks["citation_integrity"] = all(citation_checks)
    checks["removed_citations"] = not set(citation_ids) & set(
        expected.get("must_remove_citation_ids", [])
    )
    # The review fingerprint includes source evidence and gold rules so annotations
    # cannot silently carry over to changed results, a new corpus or new rules.
    review_document = {
        "scenario_id": scenario["scenario_id"],
        "variant": run["variant"],
        "phase": run["phase"],
        "outputs": outputs,
        "evidence": scenario["profile"]["evidence"],
        "corpus_sha256": scenario["corpus"]["sha256"],
        "claim_rules": expected["claim_rules"],
        "prohibited_claims": expected["prohibited_claims"],
    }
    review_id = digest(review_document)
    annotation = (claim_reviews or {}).get(review_id)
    total_claims = unsupported = 0
    if annotation is not None:
        if not isinstance(annotation, dict):
            raise ValueError("Invalid human claim annotation")
        total_claims, unsupported = (
            annotation["total_claims"],
            annotation["unsupported_claims"],
        )
        if (
            type(total_claims) is not int
            or type(unsupported) is not int
            or not 0 <= unsupported <= total_claims
            or not annotation.get("reviewer")
            or not annotation.get("reviewed_at")
        ):
            raise ValueError("Invalid human claim annotation")
    final = run["phase"] == "final"
    kind = scenario["evaluation_kind"]
    normal = kind in ("normal", "memory", "injection_probe")
    success = run["status"] == "completed" and not run.get("used_mock_fallback", False)
    return {
        "scenario_id": scenario["scenario_id"],
        "variant": run["variant"],
        "phase": run["phase"],
        "split": scenario["split"],
        "kind": kind,
        "checks": checks,
        "fault_applicable": applicable,
        "counts": {
            "relevant_at_5_normal": (len(set(accepted) & relevant), 5)
            if final and normal
            else (0, 0),
            "relevant_at_5_abstention": (len(set(accepted) & relevant), 5)
            if final and kind == "true_abstention"
            else (0, 0),
            "returned_precision": (len(set(accepted) & relevant), len(accepted))
            if final
            else (0, 0),
            "hard_constraint_violation_rate": (
                sum(constraints[j] == "fail" for j in accepted),
                len(accepted),
            ),
            "unsupported_claim_rate": (unsupported, total_claims),
            "unknown_handling_accuracy": (
                len(unknown & review - set(accepted)),
                len(unknown),
            ),
            "abstention_accuracy": (int(success and not accepted and not review), 1)
            if final and kind == "true_abstention"
            else (0, 0),
        },
        "claim_review_id": review_id,
        "claim_review_complete": annotation is not None or not any(outputs.values()),
        "claim_review_document": review_document,
        "citation_integrity": ratio(sum(citation_checks), len(citation_checks)),
    }


def summarize(scenarios, runs, records, *, stop_reason=None, claim_reviews=None):
    lookup = {s["scenario_id"]: s for s in scenarios}
    identities = [(r["scenario_id"], r["variant"], r["phase"]) for r in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate search phases")
    scores = [score_run(lookup[r["scenario_id"]], r, claim_reviews) for r in runs]
    totals = account_totals(records)
    live = (
        bool(runs)
        and not totals["used_mock_fallback"]
        and all(r.get("used_mock_fallback") is False for r in runs)
    )
    metrics = {}
    for variant in VARIANTS:
        metrics[variant] = {}
        for split in ("all", "dev", "held_out"):
            selected = [
                s
                for s in scores
                if s["variant"] == variant and (split == "all" or s["split"] == split)
            ]
            search_rows = [
                r
                for r in runs
                if r["variant"] == variant
                and (split == "all" or lookup[r["scenario_id"]]["split"] == split)
            ]
            names = COUNT_METRICS
            values = {
                name: ratio(
                    sum(s["counts"][name][0] for s in selected),
                    sum(s["counts"][name][1] for s in selected),
                )
                for name in names
            }
            memory_scenarios = [
                s
                for s in scenarios
                if s["evaluation_kind"] == "memory"
                and (split == "all" or s["split"] == split)
                and any(
                    r["scenario_id"] == s["scenario_id"] and r["phase"] == "final"
                    for r in search_rows
                )
            ]
            memory_success = 0
            for scenario in memory_scenarios:
                phases = [
                    r
                    for r in search_rows
                    if r["scenario_id"] == scenario["scenario_id"]
                ]
                phase_scores = [
                    s for s in selected if s["scenario_id"] == scenario["scenario_id"]
                ]
                memory_success += int(
                    len(phases) == 2
                    and len({r["worker_id"] for r in phases}) == 2
                    and all(all(s["checks"].values()) for s in phase_scores)
                    and all(r["status"] == "completed" for r in phases)
                )
            values["memory_correctness"] = ratio(memory_success, len(memory_scenarios))
            latencies = sorted(r["latency_ms"] for r in search_rows)
            if any(not math.isfinite(x) or x < 0 for x in latencies):
                raise ValueError("Invalid measured latency")
            values["latency_ms"] = {
                "sample_count": len(latencies),
                "samples": latencies,
                "median": median(latencies) if latencies else None,
                "p95": latencies[math.ceil(0.95 * len(latencies)) - 1]
                if latencies
                else None,
                "p95_method": "nearest_rank",
            }
            values["tokens_per_run"] = ratio(
                sum(r["total_tokens"] for r in search_rows), len(search_rows)
            )
            values["cost_usd_per_run"] = ratio(
                sum(r["estimated_cost_usd"] for r in search_rows), len(search_rows)
            )
            if any(not r.get("usage_complete", True) for r in search_rows):
                for name in ("tokens_per_run", "cost_usd_per_run"):
                    values[name] = {
                        "numerator": None,
                        "denominator": len(search_rows),
                        "value": None,
                        "known_subtotal": values[name]["numerator"],
                        "reason": "At least one request omitted usage",
                    }
            values["fault_robustness"] = ratio(
                sum(
                    all(s["checks"].values()) for s in selected if s["fault_applicable"]
                ),
                sum(s["fault_applicable"] for s in selected),
            )
            values["denominator_members"] = [
                f"{s['scenario_id']}/{s['phase']}" for s in selected
            ]
            values["claim_review_coverage"] = ratio(
                sum(s["claim_review_complete"] for s in selected), len(selected)
            )
            values["by_evaluation_kind"] = {
                kind: {
                    name: ratio(
                        sum(
                            s["counts"][name][0] for s in selected if s["kind"] == kind
                        ),
                        sum(
                            s["counts"][name][1] for s in selected if s["kind"] == kind
                        ),
                    )
                    for name in COUNT_METRICS
                }
                for kind in sorted({s["kind"] for s in selected})
            }
            metrics[variant][split] = values
    expected_runs = (
        sum(sum(step["op"] == "run_search" for step in s["steps"]) for s in scenarios)
        * 3
    )
    reviewed = all(
        s["review"]["status"] != "pending_human_review"
        and s["review"].get("reviewer")
        and s["review"].get("reviewed_at")
        for s in scenarios
    )
    complete = len(runs) == expected_runs and stop_reason is None
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "is_live_run": live,
        "complete": complete,
        "stop_reason": stop_reason,
        "gold_human_review_complete": bool(reviewed),
        "eligible_for_final_report": bool(
            live
            and complete
            and reviewed
            and totals["usage_complete"]
            and all(s["claim_review_complete"] for s in scores)
            and all(
                s["checks"]["fault_coverage"] and s["checks"]["required_hydration"]
                for s in scores
            )
        ),
        "completed_searches": len(runs),
        "expected_searches": expected_runs,
        "expected_final_cases": len(scenarios) * 3,
        "cost_accounting": totals,
        "model_ids": [
            settings.CHAT_MODEL_ID,
            settings.CHAT_MODEL_ID_FAST,
            settings.EMBEDDING_MODEL_ID,
        ],
        "dataset_sha256": digest(scenarios),
        "execution_dataset_sha256": execution_digest(scenarios),
        "metrics": metrics,
        "scores": scores,
        "runs": runs,
        "request_receipts": records,
        "metric_definitions": {
            "relevant_at_5": "Relevant accepted top-five jobs / five slots; final phases; normal, memory, injection probes pooled; true abstention separate",
            "returned_precision": "Relevant accepted jobs / accepted jobs; final phases",
            "hard_constraint_violation_rate": "Known-fail accepted jobs / all accepted jobs; all search phases",
            "unknown_handling_accuracy": "Gold unknown jobs returned for review / all gold unknown jobs; missing is incorrect",
            "unsupported_claim_rate": "Human-labeled unsupported atomic claims / reviewed atomic claims, all returned prose; unreviewed claims excluded and coverage explicit",
            "memory_correctness": "Both phases meet gold checks and execute in distinct workers / completed final memory cases",
            "abstention_accuracy": "Live completed empty accepted AND review lists / true-abstention final cases; errors do not qualify",
        },
    }


def write_summary(summary, directory=ROOT / "evals"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    live = (
        bool(summary["runs"])
        and summary["is_live_run"] is True
        and not any(r.get("used_mock_fallback") is not False for r in summary["runs"])
    )
    live = (
        live and not account_totals(summary["request_receipts"])["used_mock_fallback"]
    )
    if not live:
        summary = {**summary, "is_live_run": False, "eligible_for_final_report": False}
        print(
            "MOCK GUARD: refusing to overwrite results_summary.json or docs/evaluation_report.md."
        )
    destination = directory / (
        "results_summary.json" if live else "mock_dryrun_summary.json"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except (TypeError, ValueError, OSError):
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class FaultService(GeminiService):
    """Mutate actual responses only; preserve usage and existing validators."""

    def __init__(self, fault, **kwargs):
        super().__init__(**kwargs)
        self.fault = fault
        self.triggered = False
        self._target = False

    def _usage(self, response, model_id):
        usage = super()._usage(response, model_id)
        if self._target and self.fault["action"] == "replace_response_text":
            response.parsed = None
            response.candidates = [
                types.Candidate(
                    content=types.Content(
                        parts=[types.Part(text=self.fault["replacement_text"])]
                    ),
                    finish_reason="STOP",
                )
            ]
            self.triggered = True
        return usage

    def generate_structured(self, **kwargs):
        fault = self.fault
        self._target = bool(
            fault
            and kwargs["response_schema"].__name__ == fault.get("response_schema")
            and (not self.triggered or fault["occurrence"] == "every")
        )
        output, usage = super().generate_structured(**kwargs)
        if (
            self._target
            and fault["action"] == "replace_response_text"
            and self.used_mock_fallback
        ):
            self.triggered = True
            raise GeminiResponseError("Injected malformed offline fixture")
        if self._target and fault["action"] == "replace_supported_matches_for_job":
            body = output.model_dump(mode="json")
            for item in body["assessments"]:
                if item["job_id"] == fault["job_id"]:
                    item["supported_matches"] = deepcopy(fault["supported_matches"])
                    self.triggered = True
            output = kwargs["response_schema"].model_validate(body)
        return output, usage


def _engine(schema=None):
    args = {"connect_timeout": 5}
    if schema:
        # schema is generated internally, never derived from dataset/user input.
        if not schema.startswith("eval_") or not schema[5:].isalnum():
            raise ValueError("Invalid isolated schema")
        args["options"] = f"-csearch_path={schema},public"
    engine = create_engine(
        settings.DATABASE_URL, connect_args=args, hide_parameters=True
    )
    return (
        engine.execution_options(schema_translate_map={None: schema})
        if schema
        else engine
    )


def _seed(db, scenario):
    from app.db.models import CandidateEvidenceORM, CandidateProfileORM, PreferenceORM
    from app.services.ingestion import _ingest_prepared, _prepare

    raw = json.loads((ROOT / scenario["corpus"]["path"]).read_text())
    ids = set(scenario["corpus"]["fixture_job_ids"])
    _ingest_prepared(db, [_prepare(j) for j in raw if j["job_id"] in ids])
    profile = scenario["profile"]
    db.add(
        CandidateProfileORM(
            id=profile["profile_id"],
            name=profile["name"],
            raw_resume_text=profile["raw_resume_text"],
            profile_version=profile["profile_version"],
            is_confirmed=True,
        )
    )
    db.flush()
    for evidence in profile["evidence"]:
        db.add(
            CandidateEvidenceORM(
                **evidence["record"], profile_version=evidence["profile_version"]
            )
        )
    prefs = scenario["preferences"]
    db.add(
        PreferenceORM(
            profile_id=prefs["profile_id"],
            version=prefs["version"],
            hard_constraints=prefs["hard_constraints"],
            soft_preferences=prefs["soft_preferences"],
            is_active=True,
        )
    )
    db.commit()


def _worker(send, scenario, variant, phase, schema, remaining, initial, timeout):
    """Fresh interpreter for every search; memory updates are committed before exit."""
    ledger = RequestAccounting(
        remaining, on_change=lambda records: send.send(("receipts", records))
    )
    worker_id = uuid4().hex
    engine = None
    try:
        from app.agents import graph
        from app.api.routes import SearchInput, _new_search
        from app.db.models import Base
        from app.services.profile_service import update_preferences

        engine = _engine(schema)
        with account_requests(ledger), Session(engine, expire_on_commit=False) as db:
            if initial:
                Base.metadata.create_all(engine)
                _seed(db, scenario)
            setup_count = len(ledger.records)
            state = _new_search(
                db,
                SearchInput(
                    profile_id=scenario["profile"]["profile_id"],
                    user_query=scenario["user_query"],
                    system_variant=variant,
                    timeout_seconds=timeout,
                ),
                worker_id,
            )
            started = monotonic()
            send.send(
                (
                    "search_started",
                    {
                        "setup_count": setup_count,
                        "worker_id": worker_id,
                        "preference_version": state["preference_version"],
                    },
                )
            )
            fault = scenario["fault_injection"]
            fault = fault if fault and variant in fault["applicable_variants"] else None
            tool_triggered = False
            original = graph.hybrid_search_jobs

            def retrieval(*args, **kwargs):
                nonlocal tool_triggered
                if (
                    fault
                    and fault["hook"] == "hybrid_search_jobs_before_execution"
                    and not tool_triggered
                ):
                    tool_triggered = True
                    raise TimeoutError("Injected retrieval timeout")
                return original(*args, **kwargs)

            graph.hybrid_search_jobs = retrieval
            stop = None
            with tempfile.TemporaryDirectory(prefix="careermatch-eval-") as temporary:
                config = settings.model_copy(
                    update={"LLM_CACHE_PATH": str(Path(temporary) / "cache.sqlite")}
                )
                with FaultService(fault, config=config) as service:
                    try:
                        workflow = graph.build_graph(
                            db, budget=graph.RunBudget(8, timeout), llm_service=service
                        )
                        for snapshot in workflow.stream(
                            state, {"recursion_limit": 16}, stream_mode="values"
                        ):
                            state = snapshot
                    except (CostBudgetExceeded, CostAccountingError) as error:
                        stop = type(error).__name__
                        state["stop_reason"] = stop
                        state.update(graph.finalize_results(state))
                    finally:
                        graph.hybrid_search_jobs = original
                    fault_injected = service.triggered or tool_triggered
            db.commit()
            run = {key: state.get(key, []) for key in OUTPUT_KEYS}
            run.update(
                scenario_id=scenario["scenario_id"],
                variant=variant,
                phase=phase,
                status=state["status"],
                stop_reason=state.get("stop_reason"),
                worker_id=worker_id,
                preference_version=state["preference_version"],
                correction_count=state["correction_count"],
                hydrated_job_ids=[j["job_id"] for j in state["hydrated_jobs"]],
                fault_injected=fault_injected,
                latency_ms=(monotonic() - started) * 1000,
                **account_totals(ledger.records[setup_count:]),
            )
            run["used_mock_fallback"] |= (
                not bool(settings.GEMINI_API_KEY.strip()) or state["used_mock_fallback"]
            )
            if phase == "before_restart" and stop is None:
                patch = next(
                    s for s in scenario["steps"] if s["op"] == "update_preferences"
                )
                prefs = scenario["preferences"]
                updated = update_preferences(
                    db,
                    prefs["profile_id"],
                    HardConstraints.model_validate(
                        {**prefs["hard_constraints"], **patch["hard_constraints_patch"]}
                    ),
                    SoftPreferences.model_validate(
                        {**prefs["soft_preferences"], **patch["soft_preferences_patch"]}
                    ),
                    "Evaluation memory restart",
                )
                if updated.version != patch["expected_version"]:
                    raise ValueError("Unexpected persisted preference version")
                db.commit()
            send.send(("result", {"run": run, "stop_reason": stop}))
    except (
        SQLAlchemyError,
        OSError,
        ValueError,
        TypeError,
        LookupError,
        GeminiRequestError,
        GeminiResponseError,
        CostBudgetExceeded,
        CostAccountingError,
    ) as error:
        # Exception text may contain SQL credentials or prompts. Class only.
        send.send(("error", type(error).__name__))
    finally:
        if engine is not None:
            engine.dispose()
        send.close()


def run_worker(
    scenario,
    variant,
    phase,
    schema,
    remaining,
    initial,
    timeout=120,
    *,
    worker_target=_worker,
    grace_seconds=30,
):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(send, scenario, variant, phase, schema, remaining, initial, timeout),
    )
    records, result, stop = [], None, None
    search_info = None
    search_started = None

    def failed_search(reason):
        if search_info is None:
            return None
        return {
            "scenario_id": scenario["scenario_id"],
            "variant": variant,
            "phase": phase,
            "status": "failed",
            "stop_reason": reason,
            "worker_id": search_info["worker_id"],
            "preference_version": search_info["preference_version"],
            "shortlist": [],
            "needs_review_list": [],
            "rejected_counterfactuals": [],
            "hydrated_job_ids": [],
            "correction_count": 0,
            "fault_injected": False,
            "latency_ms": (monotonic() - search_started) * 1000,
            **account_totals(records[search_info["setup_count"] :]),
            "used_mock_fallback": not bool(settings.GEMINI_API_KEY.strip())
            or account_totals(records)["used_mock_fallback"],
        }

    started = monotonic()
    deadline = started + (600 if initial else 0) + timeout + grace_seconds
    process.start()
    send.close()
    try:
        while monotonic() < deadline:
            if receive.poll(0.2):
                try:
                    kind, payload = receive.recv()
                except EOFError:
                    break
                if kind == "receipts":
                    records = payload
                elif kind == "search_started":
                    search_info, search_started = payload, monotonic()
                    deadline = monotonic() + timeout + grace_seconds
                elif kind == "result":
                    result, stop = payload["run"], payload["stop_reason"]
                    process.join(5)
                    break
                elif kind == "error":
                    stop = payload
                    process.join(5)
                    break
            elif not process.is_alive():
                break
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
            stop = "WorkerTimeout"
        process.join(5)
        if result is None and stop is None:
            stop = "WorkerExitedWithoutResult"
        if result is None:
            result = failed_search(stop)
        return result, records, stop
    except KeyboardInterrupt:
        return (
            result or failed_search("KeyboardInterrupt"),
            records,
            "KeyboardInterrupt",
        )
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        receive.close()
        process.close()


def evaluate(
    scenarios,
    *,
    directory=ROOT / "evals",
    claim_reviews=None,
    worker=run_worker,
    engine=None,
    limit=None,
):
    limit = settings.EVAL_MAX_COST_USD if limit is None else limit
    RequestAccounting(limit)  # Validate before any side effects.
    records, runs, stop = [], [], None
    try:
        for scenario in scenarios:
            for variant in VARIANTS:
                schema = "eval_" + uuid4().hex
                if engine is not None:
                    with engine.begin() as connection:
                        connection.execute(CreateSchema(schema))
                try:
                    phases = [
                        s["phase"] for s in scenario["steps"] if s["op"] == "run_search"
                    ]
                    for index, phase in enumerate(phases):
                        remaining = float(
                            Decimal(str(limit))
                            - Decimal(
                                str(account_totals(records)["reserved_or_spent_usd"])
                            )
                        )
                        if remaining <= 0:
                            stop = "CostBudgetExceeded"
                            break
                        run, receipts, stop = worker(
                            scenario, variant, phase, schema, remaining, index == 0
                        )
                        records.extend(receipts)
                        if run is not None:
                            runs.append(run)
                        if not account_totals(records)["usage_complete"]:
                            stop = stop or "CostAccountingError"
                        if account_totals(records)["reserved_or_spent_usd"] >= limit:
                            stop = stop or "CostBudgetExceeded"
                        if stop:
                            break
                finally:
                    if engine is not None:
                        with engine.begin() as connection:
                            connection.execute(
                                DropSchema(schema, cascade=True, if_exists=True)
                            )
                if stop:
                    break
            if stop:
                break
    except (SQLAlchemyError, OSError, KeyboardInterrupt) as error:
        stop = type(error).__name__
    summary = summarize(
        scenarios, runs, records, stop_reason=stop, claim_reviews=claim_reviews
    )
    if stop == "CostBudgetExceeded":
        print(
            f"BUDGET STOP: ceiling ${limit:.2f}; next request cannot fit or ceiling reached. Saving partial results."
        )
    elif stop:
        print(
            f"EVALUATION STOP: {stop}; preserving available receipts and partial results."
        )
        for record in reversed(records):
            failure = record.get("failure")
            if failure:
                print(
                    f"Provider failure: model={record['model_id']} "
                    f"category={failure['category']} "
                    f"http_status={failure['http_status']} api_status={failure['api_status']}. "
                    "Unsettled spending reservation retained; no automatic paid retry."
                )
                break
    if runs or records:
        path = write_summary(summary, directory)
        print(
            f"Saved {path.name}: {len(runs)}/{summary['expected_searches']} searches; is_live_run={summary['is_live_run']}."
        )
    else:
        print("No measured runs; no results file written.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-reviews", type=Path)
    parser.add_argument(
        "--rescore",
        type=Path,
        help="Apply human claim reviews to saved results without DB/API calls",
    )
    args = parser.parse_args(argv)
    print(f"GEMINI_API_KEY configured: {bool(settings.GEMINI_API_KEY.strip())}")
    engine = None
    try:
        scenarios = load_scenarios()
        reviews = (
            json.loads(args.claim_reviews.read_text()) if args.claim_reviews else None
        )
        if args.rescore:
            saved = json.loads(args.rescore.read_text())
            if saved["execution_dataset_sha256"] != execution_digest(scenarios):
                raise ValueError("Saved experiment does not match the current dataset")
            summary = summarize(
                scenarios,
                saved["runs"],
                saved["request_receipts"],
                stop_reason=saved["stop_reason"],
                claim_reviews=reviews,
            )
            summary["created_at"] = saved["created_at"]
            summary["rescored_at"] = datetime.now(UTC).isoformat()
            write_summary(summary, args.rescore.parent)
            print("Rescored saved outputs; no database or Gemini calls made.")
            return 0
        engine = _engine()
        with engine.connect() as connection:
            if not connection.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
            ):
                raise ValueError("PostgreSQL pgvector extension is required")
        summary = evaluate(scenarios, claim_reviews=reviews, engine=engine)
        return 0 if summary["complete"] and summary["is_live_run"] else 1
    except SQLAlchemyError as error:
        print(
            f"PostgreSQL preflight failed: {type(error).__name__}; no Gemini calls or live results claimed."
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Evaluation preflight failed: {type(error).__name__}; no live results claimed."
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
