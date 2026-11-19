# CareerMatch evaluation report — partial live development results

**Status: provisional; Step 3.4 is not fully verified.** This report documents the
saved live evidence. It is not a completed 30-scenario benchmark, a held-out
evaluation, or evidence that all application behavior works. The user requested
that further Gemini evaluation stop because of intermittent HTTP 503 failures.
No new model calls, prompt tuning, or held-out reevaluation were performed for
this report. No offline/mock measurements are included.

Report date: **2026-11-20, Asia/Calcutta**.

## Source and provenance

The sole measurement source is [results_summary.json](../evals/results_summary.json).
All results below refer to this single saved session, not an aggregation of
earlier retries. Its timestamp is `2026-11-19T20:31:06.007001+00:00`
(2026-11-20 02:01:06.007001 IST).

| Source property | Value |
| --- | --- |
| Source file SHA-256 | `40f9b12c30ee647fb5978ee74d453dd07103db272d0361dff09a66b2bc99fd4c` |
| Dataset SHA-256, runner canonical JSON | `8ebfc6c6861e6e41dcc14f5a9069874254f15aa603311ae8b16a0f58cc129692` |
| Execution dataset SHA-256, excluding review sign-off | `9eb3e567c514cfe44e32a1c0e759217ca4b23c5a69a8d75bb87a38d82ec64a5e` |
| `is_live_run` | `true` |
| `complete` | `false` |
| `eligible_for_final_report` | `false` |
| `gold_human_review_complete` | `false` |
| `cost_accounting.usage_complete` | `false` |
| `cost_accounting.used_mock_fallback` | `false` |
| Stop reason | `CostAccountingError` |
| Underlying recorded failure | HTTP `503`, API status `UNAVAILABLE`, model `gemini-3.5-flash-lite` |

Every recorded search explicitly has `used_mock_fallback=false`. Every available
usage receipt has the same flag. The final request has no usage receipt; its
failure and reservation are preserved rather than replaced with invented usage.
`is_live_run=true` establishes absence of mock fallback, not successful completion
or benchmark readiness. Generation of this report was permitted only after
checking that the source exists and is not mock; missing or mock evidence must
not be substituted into this document.

| Role | Model used in this session |
| --- | --- |
| Discovery | `gemini-3.5-flash-lite` |
| Matching and Verification | `gemini-3.5-flash-lite` |
| Corpus and query embeddings | `gemini-embedding-2`, 768 dimensions |

The user temporarily selected Flash-Lite for both chat settings. These numbers
do not measure the original `gemini-3.8-flash` Matching/Verification configuration.
Each variant still uses the same retrieval corpus and deterministic gates.

## Coverage and exact denominators

The runner recorded **21/96 search attempts**, consisting of the final phase of
`sc_001` through `sc_007` for all three variants. Its `completed_searches=21`
field counts recorded attempts, including the partial one: **20/21 completed**
and **1/21 partial**. The intended 96 searches comprise 90 final scenario/variant
cases plus six pre-restart memory searches. Coverage is **21/90 final cases**,
**21/69 planned dev search phases**, and **0/27 held-out search phases**. Each
split includes one memory scenario with three additional pre-restart searches.

| Variant | Recorded attempts | Completed status | Partial status |
| --- | ---: | ---: | ---: |
| `retrieval_only` | 7 | 7/7 | 0/7 |
| `single_agent` | 7 | 7/7 | 0/7 |
| `multi_agent` | 7 | 6/7 | 1/7 |

All reached cases have `evaluation_kind=normal` and `split=dev`. No memory,
unknown-only, true-abstention, fault-injection, or injection-probe scenarios were
reached. All automated expected-outcome checks pass for 20 recorded attempts;
`sc_007/multi_agent` fails the status and minimum-shortlist-size checks. Those
checks do not constitute semantic claim review.

The dataset gold labels remain pending human review. The table therefore records
observed outputs scored against provisional labels, not human-validated quality.

## Observed three-way comparison

Ratios show their exact numerator and denominator. Percentages and means are
display approximations. **N/A (0/0) means unmeasured, not zero error or success.**
All columns include their seven recorded attempts; the partial multi-agent case
is not silently dropped.

| Metric | `retrieval_only` | `single_agent` | `multi_agent` |
| --- | --- | --- | --- |
| Relevant@5, normal | 13/35 (37.142857%) | 13/35 (37.142857%) | 12/35 (34.285714%) |
| Relevant@5, true abstention | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| Returned Precision | 13/13 (100%) | 13/13 (100%) | 12/12 (100%) |
| Hard-Constraint Violation Rate | 0/13 (0%) | 0/13 (0%) | 0/12 (0%) |
| Unsupported-Claim Rate | N/A (0/0 reviewed claims) | N/A (0/0 reviewed claims) | N/A (0/0 reviewed claims) |
| Unknown Handling Accuracy | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| Memory Correctness | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| Abstention Accuracy | N/A (0/0) | N/A (0/0) | N/A (0/0) |
| Median latency, ms | 23760.691284 (n=7) | 33896.642847 (n=7) | 41325.397520 (n=7) |
| p95 latency, ms | 23864.333394 (n=7) | 37604.248890 (n=7) | 44676.381676 (n=7) |
| Tokens per search | 1585/7 = 226.428571 | 27426/7 = 3918.000000 | Unknown/7; known subtotal 47177 |
| Estimated USD per search | 0.00150490/7 = 0.000214985714 | 0.03559370/7 = 0.005084814286 | Unknown/7; known subtotal 0.04572072 |
| Claim-review coverage, search outputs | 0/7 | 0/7 | 0/7 |

Relevant@5 divides relevant accepted jobs by five slots for each normal final
case. It is not recall. These seven scenarios contain only 13 gold-relevant slots
after applying a top-five cap per scenario, so 13/35 is the maximum for this
particular subset. Returned Precision divides relevant accepted jobs by all
accepted jobs; it does not score jobs returned for review. Hard-Constraint
Violation Rate counts known-fail jobs accepted, not unknown jobs. Unknown handling
has its own denominator, which is empty in this session.

Unsupported-Claim Rate needs human-reviewed atomic claims from all returned
prose. Valid citation IDs or agreement between agents cannot establish semantic
support. No such annotations were supplied, so this rate remains unknown.
Memory requires both phases and a new worker reading persisted preferences;
abstention requires a completed live run with no accepted or review jobs in a
true-abstention scenario. Neither was exercised here.

Latency uses the recorded worker search-stage milliseconds. It includes graph
execution, provider initialization, model metadata requests, rate-limit waits,
and the search commit; it excludes initial corpus seeding and process startup.
The median of seven samples is the fourth sorted sample. p95 uses nearest rank,
`ceil(0.95 * 7) = 7`, hence the maximum observed sample. This is a small-sample
summary, not a reliable production tail-latency estimate.

## Cost, latency, and quality tradeoffs

In this subset, single-agent and multi-agent execution had higher median and p95
latency than retrieval-only. Single-agent accepted the same 13 relevant jobs as
retrieval-only while recording substantially higher search token and cost totals.
The known multi-agent search cost subtotal already exceeds the single-agent
subtotal, but its full mean cost and token count cannot be computed because one
request omitted usage. These observations do not establish superior explanation
quality for any variant because claim review is absent.

Multi-agent returned one fewer accepted job because the `sc_007` attempt stopped
on a provider failure. Counting that operational outcome is appropriate; treating
it as proof of inferior reasoning is not. The partial run returned `job_006` for
review, while the other two variants accepted it. None of these differences
justify prompt tuning on unreviewed claims or on the untouched held-out cases.

| Session accounting item | Value |
| --- | ---: |
| Metered generation/embedding attempts | 101 |
| Attempts with known usage | 100/101 |
| Unsettled attempts | 1/101 |
| Recorded cache hits | 0 |
| Known tokens, including corpus setup | 88545 |
| Known search token subtotal | 76188 |
| Known corpus setup tokens | 12357 |
| Known estimated USD, including corpus setup | 0.08529072 |
| Known search estimated USD subtotal | 0.08281932 |
| Known corpus setup estimated USD | 0.00247140 |
| Outstanding conservative reservation, USD | 0.33505280 |
| Reserved-or-spent accounting total, USD | 0.42034352 |

Dollar values are the runner's token-price estimates, not a billing statement;
corpus setup is included in the session ceiling but excluded from per-search
means. Derived subtotals above are rounded to eight decimal places to remove
binary-float representation noise. The outstanding reservation is an upper-bound
allowance, not evidence of a charge. The recorded stop was an accounting guard
following HTTP 503, not `CostBudgetExceeded`. This JSON does not store the
configured session ceiling, so the report does not infer its value from today's
settings.

## Failure case study: a provider failure hidden by the accounting wrapper

During earlier development evaluation attempts, the provider wrapper converted
HTTP errors and connection failures into the same `CostAccountingError` and
discarded the HTTP status. This made an outage indistinguishable from quota,
billing, or invalid-request failures in the saved report. That was an application
diagnostic bug; the underlying Google outage was a separate external failure.

The repair already made during Step 3.3 added `RequestAccounting.note_failure`
in [llm_client.py](../app/services/llm_client.py). It stores an allowlisted error
category, integer HTTP status, and allowlisted API status with the unsettled
receipt. It excludes provider messages, keys, and request text. The runner prints
those fields while preserving the reservation and stopping additional paid
requests. Response usage is also captured before structured-output validation,
so a billed malformed response is not silently treated as free.

The current saved session provides concrete post-repair evidence:

```json
{
  "scenario_id": "sc_007",
  "variant": "multi_agent",
  "status": "partial",
  "stop_reason": "CostAccountingError",
  "shortlist_job_ids": [],
  "needs_review_job_ids": ["job_006"],
  "last_receipt_model": "gemini-3.5-flash-lite",
  "last_receipt_usage": null,
  "last_receipt_reserved_usd": 0.3350528,
  "failure": {
    "category": "http_error",
    "http_status": 503,
    "api_status": "UNAVAILABLE"
  }
}
```

This is a projection of actual saved fields, not a synthetic replay. Existing
tests in [test_eval_runner.py](../tests/test_eval_runner.py) exercise HTTP
400/403/429/503/504 diagnostics, secret exclusion, reservation retention, output
persistence, and malformed-response accounting with explicit test doubles.
Those tests validate the repair; they are not additional benchmark observations.
The repair improves diagnosis and accounting, not provider availability. No new
application repairs were made in Step 3.4, whose only owned file is this report.

## Verification and remaining work

The blueprint's Verification Gate is descriptive, not a shell command:

> `docs/evaluation_report.md` exists with verified, live-run metrics matching
> `evals/results_summary.json`.

For this partial report, verify source existence and SHA-256; require non-mock
provenance in every available run/receipt; recompute all stored metrics from the
saved runs and current scenario labels without calling Gemini; and compare every
table cell with the source. Report consistency does not override
`eligible_for_final_report=false`.

The offline audit for this report completed with the following real output:

```text
PASS: source hashes, live provenance, and all saved metrics/scores match recomputation.
PASS: all nine metric families, latency ranks, and displayed comparison cells match saved evidence.
PASS: coverage, status counts, setup/search accounting, model IDs, and failure-case projection match.
PASS: six missing/empty/mock provenance cases rejected; zero-denominator and unknown-cost values stay unmeasured.
PARTIAL REPORT CONSISTENCY: PASS. No DB or Gemini calls made.
FULL STEP 3.4 ACCEPTANCE: DEFERRED — 21/96 attempts, 0/27 held-out phases, pending gold/claim review, incomplete usage.
```

`uv run pytest tests/test_eval_runner.py` also completed with `52 passed in 6.29s`.
Those tests use isolated doubles and are not counted as live evaluation evidence.
The source JSON checksum remained unchanged after verification.

Full Step 3.4 completion remains deferred. Required future evidence is a complete
live evaluation with recorded model identities, reviewed gold labels, human claim
reviews, complete usage accounting, and held-out evaluation after any justified
dev-only repairs. The 75 unrecorded search phases must not be treated as passes.
No held-out results, robustness success, memory correctness, or abstention accuracy
are claimed in this report. Further live calls are paused at the user's request.
