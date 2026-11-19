"""Deterministic gates plus real PostgreSQL FTS/pgvector integration.

Provider doubles and geometric vectors are explicitly test-only, not live Gemini
quality metrics. Database availability is mandatory; no SQLite/mock DB fallback.
"""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import event, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.db.models import Base, JobEmbeddingORM, JobMetadataORM, JobORM
from app.db.session import engine
from app.schemas.contracts import (
    HardConstraints,
    LLMUsageMeta,
    RejectedJobCounterfactual,
    TriState,
)
from app.services import ingestion, retrieval
from app.services.constraints import evaluate_job_constraints

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/fixtures/jobs_snapshot_300.json"
GATES = [
    "listing_status",
    "work_mode_and_location",
    "student_eligibility",
    "experience_limit",
    "internal_consistency",
]


def hard(**updates):
    return HardConstraints.model_validate(
        {
            "allowed_work_modes": ["remote", "onsite", "hybrid"],
            "allowed_cities": ["Bengaluru"],
            **updates,
        }
    )


@pytest.fixture
def corpus():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def records(raw):
    prepared = ingestion._prepare(raw)
    values = dict(prepared["normalized"])
    values.pop("has_conflict")
    job = JobORM(
        **{
            key: raw[key]
            for key in ("job_id", "source_url", "title", "company", "raw_description")
        },
        excerpts=prepared["excerpts"],
        **values,
    )
    meta = JobMetadataORM(
        job_id=raw["job_id"],
        source_status=raw["source_status"],
        is_duplicate=raw["job_id"] in ("job_031", "job_032"),
        captured_at=prepared["captured_at"],
        content_hash=prepared["content_hash"],
        dedup_key=raw["job_id"],
    )
    return job, meta


def test_all_corpus_gates_and_gold_expectations(corpus):
    results = {}
    for raw in corpus:
        job, meta = records(raw)
        overall, checks = evaluate_job_constraints(job, meta, hard())
        results[raw["job_id"]] = overall
        assert [check.field for check in checks] == GATES
        assert all(
            key in job.excerpts for check in checks for key in check.source_excerpt_ids
        )
        if any(check.result == TriState.FAIL for check in checks):
            assert overall == TriState.FAIL
        elif any(check.result == TriState.UNKNOWN for check in checks):
            assert overall == TriState.UNKNOWN
        else:
            assert overall == TriState.PASS
    for number in (*range(1, 13), 27, 28):
        assert results[f"job_{number:03d}"] == TriState.PASS
    for number in (*range(13, 24), *range(29, 36)):
        assert results[f"job_{number:03d}"] == TriState.FAIL
    for number in (24, 25, 26):
        assert results[f"job_{number:03d}"] == TriState.UNKNOWN


def test_preference_update_excludes_bengaluru_without_changing_remote(corpus):
    for number in range(9, 13):
        job, meta = records(corpus[number - 1])
        overall, checks = evaluate_job_constraints(job, meta, hard(allowed_cities=[]))
        assert overall == TriState.FAIL
        assert checks[1].source_excerpt_ids
    job, meta = records(corpus[0])
    assert (
        evaluate_job_constraints(job, meta, hard(allowed_cities=[]))[0] == TriState.PASS
    )
    job, meta = records(corpus[8])
    assert (
        evaluate_job_constraints(job, meta, hard(allowed_cities=["  BANGALORE  "]))[0]
        == TriState.PASS
    )
    assert (
        evaluate_job_constraints(job, meta, hard(allowed_work_modes=[]))[0]
        == TriState.FAIL
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "unknown"),
        (0, "pass"),
        (1, "pass"),
        (1.01, "fail"),
        (-1, "unknown"),
        (float("nan"), "unknown"),
        (float("inf"), "unknown"),
        (True, "unknown"),
        ("0", "unknown"),
    ],
)
def test_experience_boundaries(corpus, value, expected):
    job, meta = records(corpus[0])
    job.norm_min_yoe = value
    _, checks = evaluate_job_constraints(job, meta, hard())
    assert checks[3].result.value == expected


@pytest.mark.parametrize(
    ("student", "degree", "expected"),
    [
        (None, None, "unknown"),
        (True, False, "pass"),
        (False, None, "fail"),
        (None, True, "fail"),
        (False, True, "fail"),
        ("yes", False, "unknown"),
    ],
)
def test_nullable_student_degree_combination(corpus, student, degree, expected):
    job, meta = records(corpus[0])
    job.norm_student_eligible, job.norm_degree_completed_required = student, degree
    _, checks = evaluate_job_constraints(job, meta, hard())
    assert checks[2].result.value == expected
    _, disabled = evaluate_job_constraints(
        job,
        meta,
        hard(require_student_eligible=False, exclude_completed_degree_required=False),
    )
    assert disabled[2].result == TriState.PASS


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("india_eligible", "pass"),
        ("global_remote", "pass"),
        ("non_india_restricted", "fail"),
        ("unknown", "unknown"),
        (None, "unknown"),
    ],
)
def test_remote_region_policy(corpus, region, expected):
    job, meta = records(corpus[0])
    job.norm_remote_region = region
    assert evaluate_job_constraints(job, meta, hard())[1][1].result.value == expected
    assert (
        evaluate_job_constraints(job, meta, hard(require_india_remote_if_remote=False))[
            1
        ][1].result
        == TriState.PASS
    )


def test_cities_unknown_and_multi_location(corpus):
    job, meta = records(corpus[8])
    for cities, expected in (
        ([], "unknown"),
        (None, "unknown"),
        (["Pune"], "fail"),
        (["Pune", "Bengaluru"], "pass"),
    ):
        job.norm_cities = cities
        assert (
            evaluate_job_constraints(job, meta, hard())[1][1].result.value == expected
        )
    job.norm_work_mode = "hybrid"
    job.norm_cities = ["Bengaluru"]
    assert evaluate_job_constraints(job, meta, hard())[1][1].result == TriState.PASS


def test_closed_duplicate_unknown_and_fail_precedence(corpus):
    job, meta = records(corpus[23])  # missing graduation policy
    meta.source_status = "closed"
    assert evaluate_job_constraints(job, meta, hard())[0] == TriState.FAIL
    meta.source_status = "active"
    meta.is_duplicate = True
    assert evaluate_job_constraints(job, meta, hard())[0] == TriState.FAIL
    meta.is_duplicate = None
    assert evaluate_job_constraints(job, meta, hard())[1][0].result == TriState.UNKNOWN
    meta.job_id = "job_999"
    with pytest.raises(ValueError):
        evaluate_job_constraints(job, meta, hard())


def test_conflicts_have_exact_two_sided_proof(corpus):
    job, meta = records(corpus[34])
    result, checks = evaluate_job_constraints(job, meta, hard())
    assert result == TriState.FAIL
    assert checks[-1].result == TriState.FAIL
    source = [job.excerpts[key] for key in checks[-1].source_excerpt_ids]
    assert any("Remote - India" in value for value in source)
    assert any("San Francisco office" in value for value in source)


@pytest.mark.parametrize(
    "bad_id",
    ["job_999_ex_01", "job_001_ex_00", "job_001_ex_001", "job_001_ex_999", None],
)
def test_invalid_citation_ids_never_escape(corpus, bad_id):
    job, meta = records(corpus[0])
    job.constraint_excerpts["norm_min_yoe"].append(bad_id)
    overall, checks = evaluate_job_constraints(job, meta, hard())
    assert overall == TriState.UNKNOWN
    assert all(bad_id not in check.source_excerpt_ids for check in checks)


def test_missing_proof_and_conflict_flag_are_unknown(corpus):
    job, meta = records(corpus[0])
    job.constraint_excerpts.pop("has_conflict")
    assert evaluate_job_constraints(job, meta, hard())[0] == TriState.UNKNOWN
    job.constraint_excerpts = None
    assert evaluate_job_constraints(job, meta, hard())[0] == TriState.UNKNOWN


def test_excluded_keywords_literal_casefold_and_provenance(corpus):
    job, meta = records(corpus[0])
    overall, checks = evaluate_job_constraints(
        job, meta, hard(excluded_keywords=["PYTHON"])
    )
    assert overall == TriState.FAIL and checks[0].source_excerpt_ids
    assert (
        evaluate_job_constraints(job, meta, hard(excluded_keywords=[".*"]))[0]
        == TriState.PASS
    )
    for keyword in ("", "  ", "\x00"):
        with pytest.raises(ValueError):
            evaluate_job_constraints(job, meta, hard(excluded_keywords=[keyword]))


def test_determinism_no_shared_state_and_no_input_mutation(corpus):
    job, meta = records(corpus[0])
    original = copy.deepcopy(job.constraint_excerpts)
    policy = hard()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: evaluate_job_constraints(job, meta, policy), range(12))
        )
    assert all(result == results[0] for result in results)
    assert job.constraint_excerpts == original
    bad = policy.model_copy(update={"max_required_yoe": -1})
    with pytest.raises(ValidationError):
        evaluate_job_constraints(job, meta, bad)


@pytest.mark.parametrize(
    ("query", "limit", "error"),
    [
        (None, 10, TypeError),
        ([], 10, TypeError),
        ("x", -1, ValueError),
        ("x", True, TypeError),
        ("x", 1.5, TypeError),
        ("bad\x00", 10, ValueError),
    ],
)
def test_invalid_search_inputs_fail_before_database_or_model(query, limit, error):
    with pytest.raises(error):
        retrieval.hybrid_search_jobs(None, query, hard(), limit)


@pytest.mark.parametrize(("query", "limit"), [("", 10), (" \n", 10), ("Python", 0)])
def test_empty_search_is_a_noop(query, limit):
    result = retrieval.hybrid_search_jobs(None, query, hard(), limit)
    assert result["eligible_candidates"] == result["rejected_counterfactuals"] == []
    assert result["embedding_calls"] == result["candidates_evaluated"] == 0
    assert not result["used_mock_fallback"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "vectors",
    [
        None,
        [],
        [[1.0] * 767],
        [[1.0] * 769],
        [[0.0] * 768],
        [[float("nan")] * 768],
        [[float("inf")] * 768],
        [[1e40] * 768],
        [[1e-50] * 768],
        [[True] * 768],
    ],
)
def test_invalid_query_vectors_rejected(vectors):
    with pytest.raises(retrieval.GeminiResponseError):
        retrieval._query_vector(vectors)


def test_sql_uses_full_outer_join_ascending_distance_and_bound_parameters():
    compiled = retrieval._retrieval_statement().compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "FULL OUTER JOIN vector_cte" in sql
    assert "websearch_to_tsquery" in sql and "ts_rank_cd" in sql
    assert "<=>" in sql and "ASC" in sql and "row_number()" in sql
    assert (
        "coalesce(lexical_cte.lex_rank" in sql and "coalesce(vector_cte.vec_rank" in sql
    )
    assert 40 in compiled.params.values() and 25 in compiled.params.values()
    assert 100 in compiled.params.values() and 60 in compiled.params.values()
    assert "query" in compiled.params and "query_vec" in compiled.params


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
            schema = f"retrieval_test_{uuid4().hex}"
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
def test_embeddings(monkeypatch, tmp_path):
    """Deliberate geometric test vectors, never reported as live Gemini quality."""
    config = retrieval.settings.model_copy(
        update={"GEMINI_API_KEY": "", "LLM_CACHE_PATH": str(tmp_path / "test.sqlite")}
    )
    monkeypatch.setattr(retrieval, "settings", config)
    monkeypatch.setattr(ingestion, "settings", config)
    calls = []

    class TestService:
        used_mock_fallback = True
        last_embedding_usage = LLMUsageMeta(
            model_id=config.EMBEDDING_MODEL_ID, used_mock_fallback=True
        )

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def embed_texts(self, texts, task_type):
            calls.append((list(texts), task_type))
            return [[1.0] + [0.0] * 767 for _ in texts]

    monkeypatch.setattr(retrieval, "GeminiService", TestService)
    monkeypatch.setattr(ingestion, "GeminiService", TestService)
    return calls


def test_postgresql_gold_rejections_unknown_and_preference_change(
    db, test_embeddings, corpus, tmp_path
):
    ids = {1, 9, 13, 17, 24, 29, 31, 35}
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps([raw for raw in corpus if int(raw["job_id"][4:]) in ids])
    )
    ingestion.ingest_jobs_snapshot(db, str(path))
    test_embeddings.clear()
    result = retrieval.hybrid_search_jobs(db, "Python", hard())
    assert result["used_mock_fallback"] is True
    assert len(test_embeddings) == 1 and test_embeddings[0] == (
        ["Python"],
        "RETRIEVAL_QUERY",
    )
    eligible = {row["job_id"]: row for row in result["eligible_candidates"]}
    rejected = {row["job_id"]: row for row in result["rejected_counterfactuals"]}
    assert eligible["job_001"]["constraint_result"] == "pass"
    assert eligible["job_024"]["constraint_result"] == "unknown"
    assert {"job_013", "job_017", "job_035"} <= rejected.keys()
    assert not {"job_029", "job_031"} & (eligible.keys() | rejected.keys())
    assert result["candidates_evaluated"] == 6
    for row in rejected.values():
        RejectedJobCounterfactual.model_validate(row)
        assert row["failed_constraints"]
        for check in row["failed_constraints"]:
            assert check["result"] == "fail" and check["source_excerpt_ids"]
            assert all(
                key in db.get(JobORM, row["job_id"]).excerpts
                for key in check["source_excerpt_ids"]
            )
    changed = retrieval.hybrid_search_jobs(db, "Python", hard(allowed_cities=[]))
    assert "job_009" not in {row["job_id"] for row in changed["eligible_candidates"]}
    assert "job_009" in {row["job_id"] for row in changed["rejected_counterfactuals"]}
    json.dumps(changed, allow_nan=False)


def seed_rank_job(
    db,
    corpus,
    number,
    lexical_text,
    vector,
    *,
    status="active",
    duplicate=False,
    model=None,
    stale=False,
    mock=True,
):
    raw = dict(
        corpus[0],
        job_id=f"job_{number:03d}",
        company=f"Rank Company {number}",
        title=f"Rank Role {number}",
    )
    job, meta = records(raw)
    job.search_tsv = func.to_tsvector("english", lexical_text)
    job.constraint_excerpts = {
        **job.constraint_excerpts,
        "_embedding": {
            "model_id": retrieval.settings.EMBEDDING_MODEL_ID,
            "used_mock_fallback": mock,
        },
    }
    meta.source_status, meta.is_duplicate = status, duplicate
    db.add(job)
    db.flush()
    db.add(meta)
    if vector is not None:
        db.add(
            JobEmbeddingORM(
                job_id=job.job_id,
                dimension=768,
                model_id=model or retrieval.settings.EMBEDDING_MODEL_ID,
                content_hash="stale" if stale else meta.content_hash,
                embedding=vector,
            )
        )
    db.flush()


def test_postgresql_full_outer_join_ranks_and_null_safe_rrf(
    db, test_embeddings, corpus
):
    near = [1.0] + [0.0] * 767
    far = [0.0, 1.0] + [0.0] * 766
    seed_rank_job(db, corpus, 101, "unrelated", near)  # Vector-only rank 1.
    seed_rank_job(db, corpus, 102, "needle needle needle", None)  # Lexical-only rank 1.
    seed_rank_job(db, corpus, 103, "needle", far)  # Rank 2 in both lists.
    seed_rank_job(db, corpus, 104, "needle needle needle needle", near, status="closed")
    seed_rank_job(db, corpus, 105, "needle needle needle needle", near, duplicate=True)
    statements = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db.connection(), "before_cursor_execute", observe)
    try:
        result = retrieval.hybrid_search_jobs(db, "needle", hard())
    finally:
        event.remove(db.connection(), "before_cursor_execute", observe)
    rows = {row["job_id"]: row for row in result["eligible_candidates"]}
    assert set(rows) == {"job_101", "job_102", "job_103"}
    assert rows["job_101"]["vec_rank"] == 1 and rows["job_101"]["lex_rank"] is None
    assert rows["job_102"]["lex_rank"] == 1 and rows["job_102"]["vec_rank"] is None
    assert rows["job_103"]["vec_rank"] == rows["job_103"]["lex_rank"] == 2
    assert rows["job_101"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 160)
    assert rows["job_102"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 160)
    assert rows["job_103"]["rrf_score"] == pytest.approx(2 / 62)
    assert [row["job_id"] for row in result["eligible_candidates"]] == [
        "job_103",
        "job_101",
        "job_102",
    ]
    assert (
        len(statements) == 2
    )  # One cheap existence probe and one batched retrieval/hydration query.
    assert "FULL OUTER JOIN" in statements[1]


def test_postgresql_stale_wrong_model_and_mixed_provenance_vectors_excluded(
    db, test_embeddings, corpus
):
    near = [1.0] + [0.0] * 767
    seed_rank_job(db, corpus, 101, "unrelated", near, stale=True)
    seed_rank_job(db, corpus, 102, "unrelated", near, model="other-model")
    seed_rank_job(db, corpus, 103, "unrelated", near, mock=False)
    seed_rank_job(db, corpus, 104, "unrelated", [0.0] * 768)
    seed_rank_job(db, corpus, 105, "needle", near, stale=True)
    result = retrieval.hybrid_search_jobs(db, "needle", hard())
    assert [row["job_id"] for row in result["eligible_candidates"]] == ["job_105"]
    assert result["eligible_candidates"][0]["vec_rank"] is None


def test_postgresql_limits_and_true_zero_results(db, test_embeddings, corpus):
    assert (
        retrieval.hybrid_search_jobs(db, "needle", hard())["candidates_evaluated"] == 0
    )
    assert test_embeddings == []
    for number in range(101, 146):
        seed_rank_job(db, corpus, number, "needle", [1.0] + [0.0] * 767)
    result = retrieval.hybrid_search_jobs(db, "needle", hard(), limit=100)
    assert result["candidates_evaluated"] == len(result["eligible_candidates"]) == 25
    assert result["eligible_candidates"][0]["vec_rank"] == 1
    assert result["eligible_candidates"][-1]["vec_rank"] == 25
    denied = retrieval.hybrid_search_jobs(db, "needle", hard(allowed_work_modes=[]))
    assert (
        denied["eligible_candidates"] == []
        and len(denied["rejected_counterfactuals"]) == 5
    )
    assert (
        retrieval.hybrid_search_jobs(db, "needle", hard(), limit=1)[
            "eligible_candidates"
        ][0]["job_id"]
        == "job_101"
    )


def test_postgresql_query_injection_and_provider_failure(
    db, test_embeddings, corpus, monkeypatch
):
    seed_rank_job(db, corpus, 101, "needle", None)
    result = retrieval.hybrid_search_jobs(db, "needle'; DROP TABLE jobs; --", hard())
    assert db.scalar(select(func.count()).select_from(JobORM)) == 1
    assert isinstance(result["eligible_candidates"], list)

    def timeout(self, texts, task_type):
        raise retrieval.GeminiResponseError("Synthetic provider timeout")

    monkeypatch.setattr(retrieval.GeminiService, "embed_texts", timeout)
    with pytest.raises(retrieval.GeminiResponseError, match="timeout"):
        retrieval.hybrid_search_jobs(db, "needle", hard())
    assert db.scalar(select(func.count()).select_from(JobORM)) == 1


def test_malformed_excerpt_container_and_large_numeric_value(corpus):
    job, meta = records(corpus[0])
    job.excerpts = ["malformed"]
    state, checks = evaluate_job_constraints(
        job, meta, hard(excluded_keywords=["Python"])
    )
    assert state == TriState.FAIL
    assert checks[0].source_excerpt_ids == []
    assert checks[-1].result == TriState.UNKNOWN
    job, meta = records(corpus[0])
    job.norm_min_yoe = 10**1000
    assert evaluate_job_constraints(job, meta, hard())[0] == TriState.UNKNOWN


def test_postgresql_search_preserves_unflushed_caller_changes(
    db, test_embeddings, corpus
):
    seed_rank_job(db, corpus, 101, "needle", [1.0] + [0.0] * 767)
    job = db.get(JobORM, "job_101")
    saved_title = job.title
    job.title = "Unflushed caller edit"
    result = retrieval.hybrid_search_jobs(db, "needle", hard())
    assert job.title == "Unflushed caller edit" and job in db.dirty
    assert result["eligible_candidates"][0]["title"] == saved_title
    with db.no_autoflush:
        assert (
            db.scalar(select(JobORM.title).where(JobORM.job_id == "job_101"))
            == saved_title
        )


@pytest.mark.parametrize(
    ("title", "keyword", "expected"),
    [
        ("Developer C++", "C", "pass"),
        ("Developer C#", "C", "pass"),
        ("Developer C", "C", "fail"),
        ("Developer C++", "C++", "fail"),
        ("Developer C#", "C#", "fail"),
        ("Developer C++", "C#", "pass"),
        ("Developer C#", "C++", "pass"),
        ("Developer C++.", "c++", "fail"),
        ("Developer (C)", "C", "fail"),
        ("Developer MySQL", "SQL", "pass"),
        ("Developer .*", ".*", "fail"),
    ],
)
def test_keyword_technology_boundaries_and_literal_matching(
    corpus, title, keyword, expected
):
    job, meta = records(corpus[0])
    job.title = title
    assert (
        evaluate_job_constraints(job, meta, hard(excluded_keywords=[keyword]))[1][
            0
        ].result.value
        == expected
    )


def test_keyword_phrases_do_not_cross_independent_fields(corpus):
    job, meta = records(corpus[0])
    job.title, job.company = "AI Intern", "Remote Labs"
    policy = hard(excluded_keywords=["Intern Remote"])
    assert evaluate_job_constraints(job, meta, policy)[0] == TriState.PASS
    job.title = "AI Intern Remote"
    assert evaluate_job_constraints(job, meta, policy)[0] == TriState.FAIL
    job.title, job.company = "AI Intern", "Intern Remote Labs"
    assert evaluate_job_constraints(job, meta, policy)[0] == TriState.FAIL


@pytest.mark.parametrize("keywords", [[], ["Python"]])
def test_oversized_excerpt_suffix_is_unknown_without_integer_conversion(
    corpus, keywords
):
    job, meta = records(corpus[0])
    bad_id = job.job_id + "_ex_" + "9" * 5000
    job.excerpts[bad_id] = job.raw_description
    job.constraint_excerpts["norm_min_yoe"].append(bad_id)
    overall, checks = evaluate_job_constraints(
        job, meta, hard(excluded_keywords=keywords)
    )
    assert checks[-1].result == TriState.UNKNOWN
    assert overall == (TriState.FAIL if keywords else TriState.UNKNOWN)
    assert all(bad_id not in check.source_excerpt_ids for check in checks)


def test_existing_but_unrelated_citation_is_not_support(corpus):
    job, meta = records(corpus[12])
    title_id = next(iter(job.excerpts))
    job.constraint_excerpts["norm_min_yoe"] = [title_id]
    overall, checks = evaluate_job_constraints(job, meta, hard())
    assert overall == TriState.FAIL  # The known 3-year requirement still fails.
    assert checks[3].source_excerpt_ids == []
    assert checks[-1].result == TriState.UNKNOWN


def test_altered_normalized_value_cannot_create_a_false_pass(corpus):
    job, meta = records(corpus[12])
    job.norm_min_yoe = 0.0
    original = copy.deepcopy(job.constraint_excerpts)
    overall, checks = evaluate_job_constraints(job, meta, hard())
    assert overall == TriState.UNKNOWN
    assert checks[-1].result == TriState.UNKNOWN
    assert checks[3].source_excerpt_ids == []
    assert job.norm_min_yoe == 0.0 and job.constraint_excerpts == original


def test_excerpt_must_have_its_original_number_and_complete_content(corpus):
    job, meta = records(corpus[0])
    first, second = list(job.excerpts)[:2]
    job.excerpts[first], job.excerpts[second] = (
        job.excerpts[second],
        job.excerpts[first],
    )
    overall, checks = evaluate_job_constraints(job, meta, hard())
    assert overall == TriState.UNKNOWN
    assert all(
        first not in check.source_excerpt_ids and second not in check.source_excerpt_ids
        for check in checks
    )


def test_missing_provenance_does_not_override_fail_precedence(corpus):
    job, meta = records(corpus[12])
    job.constraint_excerpts["norm_min_yoe"] = []
    overall, checks = evaluate_job_constraints(job, meta, hard())
    assert checks[3].result == TriState.FAIL
    assert checks[-1].result == TriState.UNKNOWN
    assert overall == TriState.FAIL


def test_proof_order_and_duplicates_preserve_deterministic_result(corpus):
    job, meta = records(corpus[0])
    original = evaluate_job_constraints(job, meta, hard())
    for field, ids in job.constraint_excerpts.items():
        if field.startswith("norm_"):
            job.constraint_excerpts[field] = list(reversed(ids)) + ids
    assert evaluate_job_constraints(job, meta, hard()) == original
