"""CareerMatch evaluation fixtures; the runner belongs to Step 3.3.

Dataset contract (scenarios_30.json, schema_version=1)
----------------------------------------------------
The top level is a list of exactly 30 scenarios: 22 dev and 8 held_out.
Scenario IDs and split assignments are fixed. Held-out combinations must not be
used for prompt tuning. Shared corpus listings across splits are intentional:
the held-out unit is the scenario/profile/preferences combination, not a listing.
All personas, evidence, and job listings are synthetic. Gold labels were authored
from the source listings, then checked against deterministic gates. They are NOT
measured outcomes. review.status remains pending_human_review until a person
reviews the source evidence and records their name and date. Do not claim a
human-reviewed benchmark, or publish final benchmark metrics, before that review.

Each scenario contains:
* scenario_id, split, title, case_tags, evaluation_kind, annotation_rationale,
  review, and schema_version: annotation and stable split metadata.
* corpus: repository-relative snapshot path, SHA-256, and exact fixture_job_ids.
  Seed only these raw listings in an isolated DB/schema for this scenario, using
  the real ingestion/deduplication rules. Keep the identical inventory and
  retrieval/gates for all variants. Do not replace retrieval with expected IDs.
  Deduplication fixtures always include job_001, the canonical listing.
* profile: synthetic profile identity, confirmed state, raw_resume_text, and
  evidence. Each evidence entry wraps an unchanged EvidenceRecord in record,
  alongside its profile_version. Seed these authored records directly; résumé
  parsing is not the variable being evaluated. Respect cumulative versions.
* preferences: an unchanged PreferenceState-shaped initial snapshot. The same
  hard constraints apply to all variants. An explicit scenario preference may
  make an injection fixture eligible; no job instruction may change preferences.
* user_query and steps: ordered setup/search/update/restart instructions. Each
  run_search phase names the expected block that applies. For memory cases,
  commit updates, terminate the old application process, and read persisted
  preferences through a new process/session; do not reuse an in-memory snapshot.
  Reset profile/preferences to the initial snapshot before EACH variant.
* fault_injection: null for ordinary scenarios; otherwise a precise hook,
  occurrence rule, variant scope, and replacement/error. must_trigger=true means
  an unexercised injection is a failed setup, never successful robustness.
  Model-response mutations in live runs happen AFTER a real provider response;
  preserve provider usage, mark fault_injected=true, and report these separately
  from unmodified model quality. Fixture-only replays must set
  used_mock_fallback=true. A tool timeout is an explicit TimeoutError, not a
  synthetic empty retrieval result. Retrieval-only is an unfaulted control for
  response faults; it never needs a matching or verification call.
* expected: primary/final gold outcomes. constraint_results covers every seeded
  job, including closed/duplicate rows the SQL search excludes. relevant_job_ids
  labels relevance AND eligibility for ranking; it is not a required ordering.
  allowed_shortlist_job_ids is an allowlist, not permission to fabricate results.
  required_review_job_ids must appear as needs_review (silence is not a pass).
  must_not_retrieve_job_ids must be absent from the hydrated candidate pool.
  Check source-backed claim_rules on ALL returned prose, not just accept rows.
  prohibited_claims are propositions: negations/disclaimers are not violations.
  variant_overrides replaces specified expected fields for the named variant;
  omitted fields retain the common expectation. Memory cases also have
  expected_before_restart; initial and final gate results must both be checked.

Scoring and honesty
-------------------
normal, memory, and injection_probe cases have positive relevance targets.
true_abstention cases seed all 300 jobs and explicitly disallow every one; they
must return no accepted or needs_review jobs. unknown_only cases have genuine
eligibility uncertainty and MUST NOT be counted as true zero-match abstention.
fault cases measure robustness separately; failure is not successful abstention.
Do not count missing fault coverage or a retrieval-only control as an agent
robustness success. Denominators must identify the scenario/variant/phase.
Score ranking on the final phase; check memory on both phases. Count tokens,
cost, and latency for every actual search, including pre-restart controls.
All final reported measurements still require used_mock_fallback=False; no
dataset label, controlled replay, or agreement between models is a measurement.
"""
