"""Deterministic gate: provider doubles, temporary output, no credentials/network."""

import json
import multiprocessing
import time
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types
from pydantic import BaseModel

from app.config import settings
from app.schemas.contracts import LLMUsageMeta, MatchingBatchOutput
from app.services.llm_client import (
    CostAccountingError,
    CostBudgetExceeded,
    GeminiResponseError,
    GeminiService,
    RequestAccounting,
    account_requests,
)
from evals import run_eval as runner


class Reply(BaseModel):
    message: str


def response(text='{"message":"ok"}', *, tokens=120, usage=True):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text=text)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=tokens - 100,
            total_token_count=tokens,
        )
        if usage
        else None,
    )


class Models:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.embedding_responses = []
        self.input_limit = 1000
        self.output_limit = 1000

    def get(self, **kwargs):
        return SimpleNamespace(
            input_token_limit=self.input_limit, output_token_limit=self.output_limit
        )

    def generate_content(self, **kwargs):
        self.calls.append(("chat", kwargs))
        item = self.responses.pop(0) if self.responses else response()
        if isinstance(item, Exception):
            raise item
        return item

    def embed_content(self, **kwargs):
        self.calls.append(("embed", kwargs))
        if self.embedding_responses:
            return self.embedding_responses.pop(0)
        size = len(kwargs["contents"])
        return types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(values=[1.0] + [0.0] * 767) for _ in range(size)
            ],
            usage_metadata=types.UsageMetadata(
                prompt_token_count=size, total_token_count=size
            ),
        )


@pytest.fixture
def provider(tmp_path):
    models = Models()
    config = settings.model_copy(
        update={
            "GEMINI_API_KEY": "unit-test-not-a-real-key",
            "LLM_CACHE_PATH": str(tmp_path / "cache.sqlite"),
            "LLM_RPM_LIMIT": 60_000_000,
        }
    )
    with GeminiService(config=config, client=SimpleNamespace(models=models)) as service:
        yield service, models, config


@pytest.fixture(scope="module")
def scenarios():
    return runner.load_scenarios()


def item(job_id="job_001", decision="accept", **changes):
    result = {
        "job_id": job_id,
        "title": "Synthetic title",
        "company": "Synthetic company",
        "source_url": "https://example.com/job",
        "captured_at": "2026-11-19T00:00:00Z",
        "decision": decision,
        "supported_matches": [],
        "gaps": [],
        "unknowns": [],
        "constraint_checks": [],
        "internal_rank_score": 1.0,
    }
    result.update(changes)
    return result


def run(scenario, variant="multi_agent", phase="final", **changes):
    expected = runner.expected_for(scenario, phase, variant)
    result = {
        "scenario_id": scenario["scenario_id"],
        "variant": variant,
        "phase": phase,
        "status": "completed",
        "shortlist": [item(j) for j in expected["allowed_shortlist_job_ids"][:5]],
        "needs_review_list": [
            item(j, "needs_review") for j in expected["required_review_job_ids"]
        ],
        "rejected_counterfactuals": [],
        "hydrated_job_ids": list(expected["constraint_results"]),
        "used_mock_fallback": False,
        "total_tokens": 120,
        "estimated_cost_usd": 0.01,
        "latency_ms": 10.0,
        "worker_id": f"worker-{phase}",
        "preference_version": expected.get("preference_version", 1),
        "correction_count": 0,
        "fault_injected": False,
    }
    result.update(changes)
    return result


def receipt(cost=0.01, mock=False):
    usage = LLMUsageMeta(
        model_id="unit-test",
        total_tokens=0 if mock else 120,
        estimated_cost_usd=cost,
        used_mock_fallback=mock,
    )
    return {
        "model_id": usage.model_id,
        "reserved_usd": cost,
        "usage": usage.model_dump(),
    }


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_budget_rejected(bad):
    with pytest.raises(ValueError):
        RequestAccounting(bad)


def test_exact_ceiling_and_zero_budget():
    ledger = RequestAccounting(0.1)
    ticket = ledger.reserve(0.1, "test")
    ledger.settle(ticket, LLMUsageMeta(model_id="test", estimated_cost_usd=0.1))
    with pytest.raises(CostBudgetExceeded):
        ledger.reserve(0.0001, "test")
    with pytest.raises(CostBudgetExceeded):
        RequestAccounting(0).reserve(0, "test")


def test_unsettled_reservation_blocks_further_requests():
    ledger = RequestAccounting(2)
    ledger.reserve(0.1, "test")
    with pytest.raises(CostAccountingError):
        ledger.reserve(0.1, "test")
    total = runner.account_totals(ledger.snapshot())
    assert total["reserved_or_spent_usd"] == 0.1
    assert total["estimated_cost_usd"] == 0
    assert not total["usage_complete"]


def test_budget_admission_prevents_provider_call(provider):
    service, models, _ = provider
    with (
        account_requests(RequestAccounting(0.000001)),
        pytest.raises(CostBudgetExceeded),
    ):
        service.generate_structured("hello", "system", Reply)
    assert models.calls == []


def test_malformed_paid_response_still_accounted_and_not_cached(provider):
    service, models, _ = provider
    models.responses = [response("{"), response()]
    ledger = RequestAccounting(1)
    with account_requests(ledger):
        with pytest.raises(GeminiResponseError):
            service.generate_structured("hello", "system", Reply)
        assert service.last_generation_usage.total_tokens == 120
        assert runner.account_totals(ledger.snapshot())["total_tokens"] == 120
        service.generate_structured("hello", "system", Reply)
    assert len(models.calls) == 2
    assert models.calls[0][1]["config"].max_output_tokens == 1000
    assert runner.account_totals(ledger.snapshot())["total_tokens"] == 240


def test_cache_is_zero_new_spend_and_ledger_context_restored(provider):
    service, models, _ = provider
    ledger = RequestAccounting(1)
    with account_requests(ledger):
        _, usage = service.generate_structured("hello", "system", Reply)
        _, cached = service.generate_structured("hello", "system", Reply)
    assert cached.cached and cached.total_tokens == 0 and cached.estimated_cost_usd == 0
    assert (
        runner.account_totals(ledger.snapshot())["estimated_cost_usd"]
        == usage.estimated_cost_usd
    )
    service.generate_structured("outside", "system", Reply)
    assert len(ledger.records) == 2 and len(models.calls) == 2


def test_transport_failure_stops_without_blind_billable_retry(provider):
    service, models, _ = provider
    models.responses = [httpx.ReadTimeout("synthetic")]
    ledger = RequestAccounting(1)
    with account_requests(ledger), pytest.raises(CostAccountingError):
        service.generate_structured("hello", "system", Reply)
    assert len(models.calls) == 1
    assert not runner.account_totals(ledger.snapshot())["usage_complete"]
    assert ledger.records[0]["reserved_usd"] > 0
    assert ledger.records[0]["failure"] == {
        "category": "timeout",
        "http_status": None,
        "api_status": None,
    }


@pytest.mark.parametrize(
    "code,status",
    [
        (400, "INVALID_ARGUMENT"),
        (400, "FAILED_PRECONDITION"),
        (403, "PERMISSION_DENIED"),
        (429, "RESOURCE_EXHAUSTED"),
        (503, "UNAVAILABLE"),
        (504, "DEADLINE_EXCEEDED"),
    ],
)
def test_http_failure_preserves_status_without_secrets_or_losing_reservation(
    provider, code, status, caplog
):
    service, models, config = provider
    secret = config.GEMINI_API_KEY
    private_text = "PRIVATE_RESUME_TEST_TEXT"
    models.responses = [
        errors.APIError(
            code,
            {
                "error": {
                    "message": f"{secret} {private_text}",
                    "status": status,
                }
            },
        )
    ]
    receipts = []
    ledger = RequestAccounting(1, on_change=lambda records: receipts.append(records))
    with account_requests(ledger), pytest.raises(CostAccountingError) as caught:
        service.generate_structured("hello", "system", Reply)
    assert len(models.calls) == 1
    failed = ledger.snapshot()[0]
    assert failed["usage"] is None and failed["reserved_usd"] > 0
    assert failed["failure"] == {
        "category": "http_error",
        "http_status": code,
        "api_status": status,
    }
    assert receipts[-1] == ledger.snapshot()
    public = json.dumps(receipts) + str(caught.value) + caplog.text
    assert secret not in public and private_text not in public


def test_untrusted_status_text_cannot_enter_receipts(provider):
    service, models, config = provider
    models.responses = [
        errors.APIError(
            503, {"error": {"status": config.GEMINI_API_KEY, "message": "synthetic"}}
        )
    ]
    ledger = RequestAccounting(1)
    with account_requests(ledger), pytest.raises(CostAccountingError):
        service.generate_structured("hello", "system", Reply)
    assert ledger.records[0]["failure"]["api_status"] is None
    assert config.GEMINI_API_KEY not in json.dumps(ledger.snapshot())


def test_runner_prints_and_saves_http_failure(scenarios, tmp_path, capsys):
    failure = {
        "category": "http_error",
        "http_status": 503,
        "api_status": "UNAVAILABLE",
    }

    def worker(scenario, variant, phase, schema, remaining, initial):
        return (
            run(
                scenario,
                variant,
                phase,
                status="failed",
                shortlist=[],
                usage_complete=False,
            ),
            [
                {
                    "model_id": "gemini-3.8-flash",
                    "reserved_usd": 0.1,
                    "usage": None,
                    "failure": failure,
                }
            ],
            "CostAccountingError",
        )

    summary = runner.evaluate(
        [scenarios[0]], worker=worker, directory=tmp_path, limit=1
    )
    assert summary["request_receipts"][0]["failure"] == failure
    assert summary["cost_accounting"]["reserved_or_spent_usd"] == 0.1
    assert not summary["cost_accounting"]["usage_complete"]
    assert "http_status=503 api_status=UNAVAILABLE" in capsys.readouterr().out
    saved = json.loads((tmp_path / "results_summary.json").read_text())
    assert saved["request_receipts"][0]["failure"] == failure


@pytest.mark.parametrize("failure", ["missing_usage", "truncated"])
def test_incomplete_generation_is_not_free(provider, failure):
    service, models, _ = provider
    reply = response(usage=failure != "missing_usage")
    if failure == "truncated":
        reply.candidates[0].finish_reason = "MAX_TOKENS"
    models.responses = [reply]
    ledger = RequestAccounting(1)
    with account_requests(ledger), pytest.raises(GeminiResponseError):
        service.generate_structured("hello", "system", Reply)
    totals = runner.account_totals(ledger.snapshot())
    assert totals["reserved_or_spent_usd"] > 0
    assert totals["usage_complete"] == (failure == "truncated")


@pytest.mark.parametrize("size", [0, 1, 50, 51])
def test_embedding_chunks_and_setup_cost_accounted(provider, size):
    service, models, _ = provider
    ledger = RequestAccounting(2)
    with account_requests(ledger):
        vectors = service.embed_texts(["synthetic text"] * size)
    assert len(vectors) == size
    assert len(models.calls) == (size + 49) // 50
    assert runner.account_totals(ledger.snapshot())["total_tokens"] == size
    assert service.last_embedding_usage.total_tokens == size
    assert (
        runner.account_totals(ledger.snapshot())["physical_calls"] == (size + 49) // 50
    )


def test_invalid_embedding_retains_usage_before_vector_validation(provider):
    service, models, _ = provider
    models.embedding_responses = [
        types.EmbedContentResponse(
            embeddings=[], usage_metadata=types.UsageMetadata(prompt_token_count=17)
        )
    ]
    ledger = RequestAccounting(2)
    with account_requests(ledger), pytest.raises(GeminiResponseError):
        service.embed_texts(["synthetic"])
    assert service.last_embedding_usage.total_tokens == 17
    assert runner.account_totals(ledger.snapshot())["total_tokens"] == 17


def test_missing_model_limit_blocks_before_paid_call(provider):
    service, models, _ = provider
    models.input_limit = None
    with account_requests(RequestAccounting(2)), pytest.raises(CostAccountingError):
        service.generate_structured("hello", "system", Reply)
    assert not models.calls


def test_empty_metrics_are_null_not_success(scenarios):
    summary = runner.summarize(scenarios, [], [])
    assert summary["expected_searches"] == 96
    assert summary["expected_final_cases"] == 90
    assert not summary["is_live_run"] and not summary["complete"]
    metrics = summary["metrics"]["multi_agent"]["all"]
    assert metrics["returned_precision"] == runner.ratio(0, 0)
    assert metrics["latency_ms"]["p95"] is None


def test_exact_ranking_precision_and_failed_search_denominators(scenarios):
    scenario = scenarios[0]
    success = run(scenario, shortlist=[item("job_001"), item("job_003")])
    score = runner.score_run(scenario, success)
    assert score["counts"]["relevant_at_5_normal"] == (2, 5)
    assert score["counts"]["returned_precision"] == (2, 2)
    failure = run(scenario, status="failed", shortlist=[])
    score = runner.score_run(scenario, failure)
    assert score["counts"]["relevant_at_5_normal"] == (0, 5)
    assert score["counts"]["returned_precision"] == (0, 0)


def test_duplicates_and_out_of_corpus_fail_loudly(scenarios):
    with pytest.raises(ValueError, match="Duplicate"):
        runner.score_run(scenarios[0], run(scenarios[0], shortlist=[item(), item()]))
    with pytest.raises(ValueError, match="outside"):
        runner.score_run(scenarios[0], run(scenarios[0], shortlist=[item("job_999")]))
    with pytest.raises(ValueError, match="Duplicate search"):
        runner.summarize(scenarios, [run(scenarios[0])] * 2, [])


def test_abstention_requires_success_and_no_review(scenarios):
    scenario = next(s for s in scenarios if s["evaluation_kind"] == "true_abstention")
    assert runner.score_run(scenario, run(scenario))["counts"][
        "abstention_accuracy"
    ] == (1, 1)
    for change in (
        {"status": "failed"},
        {"used_mock_fallback": True},
        {"needs_review_list": [item(decision="needs_review")]},
    ):
        assert runner.score_run(scenario, run(scenario, **change))["counts"][
            "abstention_accuracy"
        ] == (0, 1)


def test_unknown_silence_is_not_correct_and_not_abstention(scenarios):
    scenario = scenarios[24]
    score = runner.score_run(
        scenario, run(scenario, shortlist=[], needs_review_list=[])
    )
    assert score["counts"]["unknown_handling_accuracy"][0] == 0
    assert score["counts"]["unknown_handling_accuracy"][1] > 0
    assert score["counts"]["abstention_accuracy"] == (0, 0)
    assert not score["checks"]["required_review"]


def test_claim_review_is_semantic_not_just_valid_ids(scenarios):
    scenario = scenarios[0]
    row = run(
        scenario,
        shortlist=[item(gaps=["Candidate has ten years of Kubernetes experience."])],
    )
    score = runner.score_run(scenario, row)
    assert score["counts"]["unsupported_claim_rate"] == (0, 0)
    assert not score["claim_review_complete"]
    annotation = {
        score["claim_review_id"]: {
            "total_claims": 2,
            "unsupported_claims": 1,
            "reviewer": "Human",
            "reviewed_at": "2026-11-19",
        }
    }
    reviewed = runner.score_run(scenario, row, annotation)
    assert reviewed["counts"]["unsupported_claim_rate"] == (1, 2)
    row["shortlist"][0]["gaps"] = ["No evidence of ten years of Kubernetes experience."]
    assert not runner.score_run(scenario, row, annotation)["claim_review_complete"]


def test_invalid_claim_counts_rejected(scenarios):
    row = run(scenarios[0])
    score = runner.score_run(scenarios[0], row)
    with pytest.raises(ValueError, match="annotation"):
        runner.score_run(
            scenarios[0],
            row,
            {
                score["claim_review_id"]: {
                    "total_claims": 0,
                    "unsupported_claims": 1,
                    "reviewer": "Human",
                    "reviewed_at": "today",
                }
            },
        )


def test_citation_ids_checked_against_correct_job_and_profile(scenarios):
    row = run(
        scenarios[0],
        shortlist=[
            item(
                supported_matches=[
                    {
                        "requirement": "Python",
                        "explanation": "Python",
                        "candidate_evidence_ids": ["ev_999"],
                        "job_excerpt_ids": ["job_003_ex_07"],
                        "is_self_reported_only": False,
                    }
                ]
            )
        ],
    )
    score = runner.score_run(scenarios[0], row)
    assert score["citation_integrity"] == runner.ratio(0, 1)
    assert not score["checks"]["citation_integrity"]


def test_memory_requires_both_phases_and_distinct_processes(scenarios):
    scenario = scenarios[15]
    rows = [run(scenario, phase="before_restart"), run(scenario)]
    summary = runner.summarize([scenario], rows, [receipt(), receipt()])
    assert summary["metrics"]["multi_agent"]["all"][
        "memory_correctness"
    ] == runner.ratio(1, 1)
    rows[1]["worker_id"] = rows[0]["worker_id"]
    summary = runner.summarize([scenario], rows, [receipt(), receipt()])
    assert summary["metrics"]["multi_agent"]["all"][
        "memory_correctness"
    ] == runner.ratio(0, 1)


def test_latency_p95_boundaries_and_all_search_costs(scenarios):
    rows = [run(s, latency_ms=float(i)) for i, s in enumerate(scenarios[:20], 1)]
    summary = runner.summarize(scenarios, rows, [receipt()] * 20)
    metrics = summary["metrics"]["multi_agent"]["all"]
    assert metrics["latency_ms"]["median"] == 10.5
    assert metrics["latency_ms"]["p95"] == 19
    assert metrics["tokens_per_run"] == runner.ratio(2400, 20)


def test_untriggered_fault_is_failed_setup_not_robustness(scenarios):
    scenario = scenarios[17]
    score = runner.score_run(scenario, run(scenario, status="failed", shortlist=[]))
    assert not score["checks"]["fault_coverage"]
    control = runner.score_run(scenario, run(scenario, variant="retrieval_only"))
    assert not control["fault_applicable"]


def test_live_fault_mutates_response_after_usage_capture(provider, scenarios):
    _, models, config = provider
    models.responses = [response(json.dumps({"assessments": [item()]}))]
    ledger = RequestAccounting(1)
    with runner.FaultService(
        scenarios[17]["fault_injection"],
        config=config,
        client=SimpleNamespace(models=models),
    ) as service:
        with account_requests(ledger), pytest.raises(GeminiResponseError):
            service.generate_structured(
                prompt="batch",
                system_instruction="system",
                response_schema=MatchingBatchOutput,
            )
        assert service.triggered
        assert service.last_generation_usage.total_tokens == 120
    assert runner.account_totals(ledger.snapshot())["total_tokens"] == 120


@pytest.mark.parametrize("index", [18, 28])
def test_citation_fault_preserves_live_usage_and_runs_every_time(
    provider, scenarios, index
):
    _, models, config = provider
    models.responses = [
        response(json.dumps({"assessments": [item()]})) for _ in range(2)
    ]
    with runner.FaultService(
        scenarios[index]["fault_injection"],
        config=config,
        client=SimpleNamespace(models=models),
    ) as service:
        for i in range(2):
            output, usage = service.generate_structured(
                prompt=f"batch {i}",
                system_instruction="system",
                response_schema=MatchingBatchOutput,
            )
            assert (
                output.assessments[0].supported_matches[0].model_dump()
                == scenarios[index]["fault_injection"]["supported_matches"][0]
            )
            assert usage.total_tokens == 120 and not usage.used_mock_fallback
        assert service.triggered and len(models.calls) == 2


def test_budget_stop_saves_partial_before_next_case(scenarios, tmp_path, capsys):
    calls = []

    def worker(scenario, variant, phase, schema, remaining, initial):
        calls.append((scenario["scenario_id"], variant))
        return run(scenario, variant, phase), [receipt(0.01)], None

    summary = runner.evaluate(scenarios, worker=worker, directory=tmp_path, limit=0.02)
    assert len(calls) == 2
    assert summary["stop_reason"] == "CostBudgetExceeded" and not summary["complete"]
    assert (tmp_path / "results_summary.json").exists()
    assert "BUDGET STOP" in capsys.readouterr().out
    assert summary["cost_accounting"]["estimated_cost_usd"] == 0.02


def test_zero_budget_does_not_start_worker_or_write_fake_results(
    scenarios, tmp_path, capsys
):
    def forbidden(*args):
        raise AssertionError("No worker should start")

    summary = runner.evaluate(scenarios, worker=forbidden, directory=tmp_path, limit=0)
    assert not list(tmp_path.iterdir()) and not summary["is_live_run"]
    assert "BUDGET STOP" in capsys.readouterr().out


def test_late_mock_preserves_live_report_and_writes_only_mock(
    scenarios, tmp_path, capsys
):
    sentinel = b'{"previous_live":true}\n'
    (tmp_path / "results_summary.json").write_bytes(sentinel)
    report = tmp_path / "evaluation_report.md"
    report.write_text("previous live report")
    calls = []

    def worker(scenario, variant, phase, schema, remaining, initial):
        calls.append(variant)
        mock = len(calls) == 3
        return (
            run(scenario, variant, phase, used_mock_fallback=mock),
            [receipt(0.01, mock)],
            None,
        )

    summary = runner.evaluate(
        [scenarios[0]], worker=worker, directory=tmp_path, limit=1
    )
    assert not summary["is_live_run"]
    assert (tmp_path / "results_summary.json").read_bytes() == sentinel
    assert report.read_text() == "previous live report"
    assert (tmp_path / "mock_dryrun_summary.json").exists()
    assert "refusing to overwrite" in capsys.readouterr().out


def test_mock_in_setup_receipt_blocks_live_even_if_search_flag_false(
    scenarios, tmp_path
):
    summary = runner.summarize(scenarios, [run(scenarios[0])], [receipt(0, True)])
    assert not summary["is_live_run"]
    summary["is_live_run"] = True  # Writer independently rechecks provenance.
    assert runner.write_summary(summary, tmp_path).name == "mock_dryrun_summary.json"


def test_pending_gold_and_claims_never_eligible_for_final_report(scenarios):
    summary = runner.summarize(scenarios, [run(scenarios[0])], [receipt()])
    assert summary["is_live_run"]
    assert not summary["gold_human_review_complete"]
    assert not summary["eligible_for_final_report"]


def test_dataset_validation_and_hash_tampering(scenarios, tmp_path):
    assert len(scenarios) == 30
    changed = deepcopy(scenarios)
    changed[0]["corpus"]["sha256"] = "0" * 64
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_scenarios(path)


def slow_worker(send, scenario, variant, phase, schema, remaining, initial, timeout):
    send.send(("receipts", [{"model_id": "test", "reserved_usd": 0.1, "usage": None}]))
    send.send(
        (
            "search_started",
            {"setup_count": 0, "worker_id": "slow-worker", "preference_version": 1},
        )
    )
    time.sleep(30)


def test_real_spawn_timeout_retains_inflight_reservation_and_reaps_child(scenarios):
    before = {child.pid for child in multiprocessing.active_children()}
    row, receipts, stop = runner.run_worker(
        scenarios[0],
        "multi_agent",
        "final",
        "eval_test",
        1,
        True,
        timeout=0.2,
        grace_seconds=0,
        worker_target=slow_worker,
    )
    assert row["status"] == "failed" and stop == "WorkerTimeout"
    assert row["latency_ms"] > 0 and not row["usage_complete"]
    assert runner.account_totals(receipts)["reserved_or_spent_usd"] == 0.1
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_offline_provenance_deterministic_and_ledger_zero(tmp_path):
    config = settings.model_copy(
        update={
            "GEMINI_API_KEY": "",
            "LLM_CACHE_PATH": str(tmp_path / "offline.sqlite"),
        }
    )
    ledger = RequestAccounting(0)
    with GeminiService(config=config) as service, account_requests(ledger):
        first, usage = service.generate_structured("fixture", "system", Reply)
        second, cached = service.generate_structured("fixture", "system", Reply)
        assert (
            first == second and usage.used_mock_fallback and cached.used_mock_fallback
        )
    assert runner.account_totals(ledger.snapshot())["reserved_or_spent_usd"] == 0
    assert runner.account_totals(ledger.snapshot())["used_mock_fallback"]


def test_incomplete_usage_not_reported_as_exact_zero(scenarios):
    row = run(scenarios[0], usage_complete=False, total_tokens=0, estimated_cost_usd=0)
    summary = runner.summarize(
        scenarios, [row], [{"model_id": "test", "reserved_usd": 0.1, "usage": None}]
    )
    metrics = summary["metrics"]["multi_agent"]["all"]
    assert metrics["cost_usd_per_run"]["value"] is None
    assert metrics["cost_usd_per_run"]["known_subtotal"] == 0
    assert metrics["tokens_per_run"]["numerator"] is None


def test_rescore_saved_claims_needs_no_db_or_paid_replay(
    scenarios, tmp_path, monkeypatch, capsys
):
    row = run(scenarios[0])
    summary = runner.summarize(scenarios, [row], [receipt()])
    source = runner.write_summary(summary, tmp_path)
    review_id = summary["scores"][0]["claim_review_id"]
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                review_id: {
                    "total_claims": 8,
                    "unsupported_claims": 1,
                    "reviewer": "Human",
                    "reviewed_at": "2026-11-19",
                }
            }
        )
    )

    def forbidden(*args):
        raise AssertionError("Rescoring must not connect to DB")

    monkeypatch.setattr(runner, "_engine", forbidden)
    assert runner.main(["--rescore", str(source), "--claim-reviews", str(reviews)]) == 0
    rescored = json.loads(source.read_text())
    assert rescored["metrics"]["multi_agent"]["all"][
        "unsupported_claim_rate"
    ] == runner.ratio(1, 8)
    assert rescored["request_receipts"] == summary["request_receipts"]
    assert "no database or Gemini calls" in capsys.readouterr().out


def test_known_failure_violation_and_separate_fault_metrics(scenarios):
    scenario = next(
        s for s in scenarios if "fail" in s["expected"]["constraint_results"].values()
    )
    job_id = next(
        j
        for j, result in scenario["expected"]["constraint_results"].items()
        if result == "fail"
    )
    score = runner.score_run(scenario, run(scenario, shortlist=[item(job_id)]))
    assert score["counts"]["hard_constraint_violation_rate"] == (1, 1)
    summary = runner.summarize(
        scenarios,
        [run(scenarios[0]), run(scenarios[17], status="failed", shortlist=[])],
        [receipt()] * 2,
    )
    grouped = summary["metrics"]["multi_agent"]["all"]["by_evaluation_kind"]
    assert set(grouped) == {"normal", "fault"}


def test_all_30_scenarios_three_variants_memory_restart_and_reset(scenarios, tmp_path):
    calls = []

    def worker(scenario, variant, phase, schema, remaining, initial):
        calls.append((scenario["scenario_id"], variant, phase, schema, initial))
        return (
            run(
                scenario,
                variant,
                phase,
                used_mock_fallback=True,
                total_tokens=0,
                estimated_cost_usd=0,
            ),
            [receipt(0, True)],
            None,
        )

    summary = runner.evaluate(scenarios, worker=worker, directory=tmp_path, limit=1)
    assert len(calls) == 96 and summary["complete"]
    assert sum(c[4] for c in calls) == 90
    assert len({c[3] for c in calls}) == 90
    for scenario in ("sc_016", "sc_026"):
        for variant in runner.VARIANTS:
            phases = [c for c in calls if c[:2] == (scenario, variant)]
            assert [c[2] for c in phases] == ["before_restart", "final"]
            assert phases[0][3] == phases[1][3]
            assert phases[0][4] and not phases[1][4]
    assert not summary["is_live_run"]
    assert not (tmp_path / "results_summary.json").exists()


def test_setup_spending_consumes_same_ceiling_as_search(scenarios, tmp_path):
    calls = []

    def worker(scenario, variant, phase, schema, remaining, initial):
        calls.append(variant)
        return run(scenario, variant, phase), [receipt(0.03), receipt(0.01)], None

    summary = runner.evaluate(scenarios, worker=worker, directory=tmp_path, limit=0.04)
    assert len(calls) == 1 and summary["stop_reason"] == "CostBudgetExceeded"
    assert summary["cost_accounting"]["estimated_cost_usd"] == 0.04
    assert summary["metrics"]["retrieval_only"]["all"][
        "cost_usd_per_run"
    ] == runner.ratio(0.01, 1)
