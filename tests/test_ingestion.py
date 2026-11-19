"""Real PostgreSQL/pgvector gate with explicit deterministic offline embeddings.

No database mocks or availability skips. Each integration test uses a fresh
transactional schema that is rolled back, preserving the user's existing data.
These tests do not claim to validate live Gemini responses (Step 1.4 does that).
"""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema

from app.db.models import Base, JobEmbeddingORM, JobMetadataORM, JobORM
from app.db.session import engine
from app.services import ingestion
from app.services.ingestion import (
    apply_update_fixture,
    ingest_jobs_snapshot,
    normalize_job_constraints,
    slice_into_excerpts,
)
from app.services.llm_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiService,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/fixtures/jobs_snapshot_300.json"
FIELDS = (
    "norm_work_mode",
    "norm_remote_region",
    "norm_cities",
    "norm_student_eligible",
    "norm_degree_completed_required",
    "norm_min_yoe",
)


@pytest.fixture
def corpus():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def normalized(description, job_id="job_901"):
    raw = {"job_id": job_id, "raw_description": description}
    return normalize_job_constraints(raw, slice_into_excerpts(job_id, description))


def test_slicer_preserves_substrings_decimals_unicode_and_numbering():
    raw = "AI role\nMinimum 3.5 years of experience. Students are eligible!\n\nWork in India — नमस्ते."
    excerpts = slice_into_excerpts("job_001", raw)
    assert list(excerpts) == [f"job_001_ex_{i:02d}" for i in range(1, 5)]
    assert list(excerpts.values()) == [
        "AI role",
        "Minimum 3.5 years of experience.",
        "Students are eligible!",
        "Work in India — नमस्ते.",
    ]
    assert all(value in raw for value in excerpts.values())
    many = slice_into_excerpts(
        "job_1000", " ".join(f"Sentence {i}." for i in range(101))
    )
    assert len(many) == 101
    assert many["job_1000_ex_99"] == "Sentence 98."
    assert many["job_1000_ex_100"] == "Sentence 99."
    assert many["job_1000_ex_101"] == "Sentence 100."
    assert slice_into_excerpts(
        "job_001", "Dr. Rao works in the U.S. office. Apply today."
    ) == {
        "job_001_ex_01": "Dr. Rao works in the U.S. office.",
        "job_001_ex_02": "Apply today.",
    }


@pytest.mark.parametrize(
    "job_id", [None, "", "job_000", "job_-01", "job_01", "job_0001", "wrong", 1]
)
def test_slicer_rejects_invalid_ids(job_id):
    with pytest.raises(ValueError):
        slice_into_excerpts(job_id, "Text.")


@pytest.mark.parametrize("raw", [None, [], 0])
def test_slicer_rejects_nontext(raw):
    with pytest.raises(TypeError):
        slice_into_excerpts("job_001", raw)


@pytest.mark.parametrize("raw", ["", " \n\t\r\n"])
def test_empty_text_has_no_evidence_or_invented_requirements(raw):
    assert slice_into_excerpts("job_001", raw) == {}
    result = normalized(raw)
    assert result["norm_work_mode"] == result["norm_remote_region"] == "unknown"
    assert result["norm_cities"] == []
    assert all(result[field] is None for field in FIELDS[3:])
    assert not result["has_conflict"]
    assert all(result["constraint_excerpts"][field] == [] for field in FIELDS)


def test_entire_corpus_normalization_and_exact_citations(corpus):
    results = {}
    for raw in corpus:
        excerpts = slice_into_excerpts(raw["job_id"], raw["raw_description"])
        result = normalize_job_constraints(raw, excerpts)
        results[raw["job_id"]] = result
        assert all(value in raw["raw_description"] for value in excerpts.values())
        for field in FIELDS:
            assert all(key in excerpts for key in result["constraint_excerpts"][field])
        assert result["has_conflict"] == (raw["job_id"] == "job_035")
    for number in range(1, 13):
        result = results[f"job_{number:03d}"]
        assert result["norm_student_eligible"] is True
        assert result["norm_degree_completed_required"] is False
        assert result["norm_min_yoe"] == 0.0
        if number in (2, 5, 8, 9, 10, 11, 12):
            assert result["norm_work_mode"] == "onsite"
            assert result["norm_cities"] == ["Bengaluru"]
        else:
            assert result["norm_work_mode"] == "remote"
            assert result["norm_remote_region"] == "india_eligible"
    for number, years in zip(range(13, 17), (3, 4, 5, 3), strict=True):
        assert results[f"job_{number:03d}"]["norm_min_yoe"] == years
    for number in range(17, 21):
        assert (
            results[f"job_{number:03d}"]["norm_remote_region"] == "non_india_restricted"
        )
    for number in range(21, 24):
        assert results[f"job_{number:03d}"]["norm_student_eligible"] is False
        assert results[f"job_{number:03d}"]["norm_degree_completed_required"] is True
    for number in (24, 26):
        assert results[f"job_{number:03d}"]["norm_student_eligible"] is None
        assert results[f"job_{number:03d}"]["norm_degree_completed_required"] is None
    for number in (25, 26):
        assert results[f"job_{number:03d}"]["norm_remote_region"] == "unknown"
    for number in (27, 28):
        assert results[f"job_{number:03d}"]["norm_min_yoe"] == 0.0
    assert results["job_033"]["norm_remote_region"] == "non_india_restricted"
    assert results["job_034"]["norm_min_yoe"] == 3.0
    conflict = results["job_035"]
    assert conflict["norm_work_mode"] == "unknown"
    assert conflict["norm_cities"] == ["San Francisco"]
    assert conflict["constraint_excerpts"]["conflict_fields"] == ["norm_work_mode"]
    proof = conflict["constraint_excerpts"]["norm_work_mode"]
    excerpts = slice_into_excerpts("job_035", corpus[34]["raw_description"])
    assert any("Remote - India" in excerpts[key] for key in proof)
    assert any("San Francisco office" in excerpts[key] for key in proof)
    for raw in corpus[35:]:
        result = results[raw["job_id"]]
        assert result["norm_work_mode"] in ("remote", "hybrid", "onsite")
        assert result["norm_student_eligible"] is not None
        assert result["norm_degree_completed_required"] is not None
        assert result["norm_min_yoe"] is not None


@pytest.mark.parametrize(
    ("description", "field", "expected"),
    [
        ("Remote. Region to be agreed.", "norm_remote_region", "unknown"),
        ("Internship, send your CV.", "norm_student_eligible", None),
        ("Minimum 3.5 years of professional experience.", "norm_min_yoe", 3.5),
        ("Minimum -1 years of experience.", "norm_min_yoe", None),
        ("Preferred: 10+ years of experience.", "norm_min_yoe", None),
        ("No prior professional experience is required.", "norm_min_yoe", 0.0),
        (
            "Remote - Global. Applicants worldwide, including residents of India, are eligible.",
            "norm_remote_region",
            "global_remote",
        ),
        (
            "Hybrid - Pune. Work from the Pune office three days per week and remotely within India on other days.",
            "norm_work_mode",
            "hybrid",
        ),
        ("Onsite - Bangalore.", "norm_cities", ["Bengaluru"]),
        ("Remote work is not available.", "norm_work_mode", "unknown"),
        ("Onsite work is not required.", "norm_work_mode", "unknown"),
        ("This is not an onsite position.", "norm_work_mode", "unknown"),
        ("Our team has 10+ years of experience.", "norm_min_yoe", None),
        ("3+ years of professional experience required.", "norm_min_yoe", 3.0),
    ],
)
def test_conservative_normalization(description, field, expected):
    assert normalized(description)[field] == expected


@pytest.mark.parametrize(
    ("description", "field"),
    [
        (
            "Students are eligible. Current students are not eligible.",
            "norm_student_eligible",
        ),
        (
            "A completed degree is required. A completed degree is not required.",
            "norm_degree_completed_required",
        ),
        ("Remote - India. US residents only.", "norm_remote_region"),
        (
            "No prior experience is required. Minimum 3 years of experience.",
            "norm_min_yoe",
        ),
        (
            "Students are eligible. A completed degree is required.",
            "norm_student_eligible",
        ),
    ],
)
def test_conflicts_never_become_unconditional_passes(description, field):
    result = normalized(description)
    assert result["has_conflict"]
    assert field in result["constraint_excerpts"]["conflict_fields"]
    assert result[field] in (None, "unknown")
    assert result["constraint_excerpts"][field]


def test_normalization_is_independent_of_ids_and_thread_safe(corpus):
    raw = corpus[34]["raw_description"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: normalized(raw, "job_901"), range(12)))
    assert all(value == outputs[0] for value in outputs)
    assert outputs[0]["has_conflict"]
    with pytest.raises(ValueError):
        normalize_job_constraints(corpus[0], {"job_001_ex_01": "Invented evidence"})


def write_snapshot(tmp_path, data):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "missing_title",
        "none_description",
        "zero_id",
        "invalid_date",
        "naive_date",
        "invalid_status",
        "bad_url",
        "nul",
    ],
)
def test_malformed_snapshot_rejected_before_any_db_or_embedding_call(
    tmp_path, corpus, change
):
    raw = copy.deepcopy(corpus[:1])
    if change == "duplicate_id":
        raw.append(copy.deepcopy(raw[0]))
    else:
        field, value = {
            "missing_title": ("title", ""),
            "none_description": ("raw_description", None),
            "zero_id": ("job_id", "job_000"),
            "invalid_date": ("captured_at", "bad"),
            "naive_date": ("captured_at", "2026-11-19"),
            "invalid_status": ("source_status", "maybe"),
            "bad_url": ("source_url", "javascript:bad"),
            "nul": ("company", "bad\x00value"),
        }[change]
        raw[0][field] = value
    with pytest.raises(ValueError):
        ingest_jobs_snapshot(None, write_snapshot(tmp_path, raw))


def test_empty_invalid_json_and_duplicate_keys(tmp_path):
    report = ingest_jobs_snapshot(None, write_snapshot(tmp_path, []))
    assert report["ingested"] == report["embedded"] == report["embedding_calls"] == 0
    assert not report["used_mock_fallback"]
    path = tmp_path / "broken.json"
    for content in ("null", "{}", "[", '[{"job_id":"job_001","job_id":"job_002"}]'):
        path.write_text(content)
        with pytest.raises((ValueError, TypeError)):
            ingest_jobs_snapshot(None, str(path))
    with pytest.raises(FileNotFoundError):
        ingest_jobs_snapshot(None, str(tmp_path / "absent.json"))


@pytest.fixture
def db():
    try:
        connection = engine.connect()
    except OperationalError:
        pytest.fail(
            "Real PostgreSQL is unavailable. Start the configured Docker db; no SQLite/mock database fallback is used.",
            pytrace=False,
        )
    with connection:
        transaction = connection.begin()
        try:
            version = int(connection.scalar(text("SHOW server_version_num")))
            assert 160000 <= version < 170000, "Gate requires PostgreSQL 16"
            assert connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ), "Initialize the pgvector extension before running this gate"
            schema = f"ingestion_test_{uuid4().hex}"
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
def offline_embeddings(monkeypatch, tmp_path):
    config = ingestion.settings.model_copy(
        update={
            "GEMINI_API_KEY": "",
            "LLM_CACHE_PATH": str(tmp_path / "test_embedding_cache.sqlite"),
        }
    )
    monkeypatch.setattr(ingestion, "settings", config)
    calls = []

    class CountingOfflineService(GeminiService):
        def embed_texts(self, texts, task_type="RETRIEVAL_DOCUMENT"):
            assert self.used_mock_fallback
            calls.append(list(texts))
            return super().embed_texts(texts, task_type)

    monkeypatch.setattr(ingestion, "GeminiService", CountingOfflineService)
    return calls


def count(db, model):
    return db.scalar(select(func.count()).select_from(model))


def test_postgresql_300_rows_incremental_reuse_and_update(
    db, offline_embeddings, tmp_path, corpus
):
    first = ingest_jobs_snapshot(db, str(SNAPSHOT))
    assert first["inserted"] == first["embedded"] == first["ingested"] == 300
    assert first["embedding_calls"] == 1 and first["duplicates"] == 2
    assert first["used_mock_fallback"] is True
    assert len(offline_embeddings) == 1 and len(offline_embeddings[0]) == 300
    for model in (JobORM, JobMetadataORM, JobEmbeddingORM):
        assert count(db, model) == 300
    db.expire_all()
    for raw in corpus:
        job = db.get(JobORM, raw["job_id"])
        assert job.excerpts == slice_into_excerpts(
            raw["job_id"], raw["raw_description"]
        )
        embedding = db.get(JobEmbeddingORM, raw["job_id"])
        assert len(embedding.embedding) == 768
        assert (
            embedding.content_hash == db.get(JobMetadataORM, raw["job_id"]).content_hash
        )
        assert embedding.model_id == ingestion.settings.EMBEDDING_MODEL_ID
    assert db.get(JobORM, "job_035").constraint_excerpts["has_conflict"] is True
    assert db.get(JobORM, "job_024").norm_student_eligible is None
    assert db.get(JobMetadataORM, "job_001").captured_at.tzinfo is not None
    assert set(
        db.scalars(select(JobMetadataORM.job_id).where(JobMetadataORM.is_duplicate))
    ) == {"job_031", "job_032"}
    assert (
        db.get(JobMetadataORM, "job_031").dedup_key
        == db.get(JobMetadataORM, "job_001").dedup_key
    )
    assert db.scalar(
        select(
            JobORM.search_tsv.op("@@")(func.plainto_tsquery("english", "Python"))
        ).where(JobORM.job_id == "job_001")
    )
    second = ingest_jobs_snapshot(db, str(SNAPSHOT))
    assert second["embedded"] == second["embedding_calls"] == second["inserted"] == 0
    assert second["used_mock_fallback"] is True and len(offline_embeddings) == 1
    closed_original = dict(corpus[0], source_status="closed")
    assert (
        ingest_jobs_snapshot(db, write_snapshot(tmp_path, [closed_original]))[
            "embedded"
        ]
        == 0
    )
    assert db.get(JobMetadataORM, "job_001").is_duplicate is True
    assert db.get(JobMetadataORM, "job_031").is_duplicate is False
    assert ingest_jobs_snapshot(db, str(SNAPSHOT))["embedded"] == 0
    # Actual embedding rows, not metadata alone, decide whether reuse is safe.
    db.get(JobEmbeddingORM, "job_005").model_id = "obsolete-model"
    db.get(JobEmbeddingORM, "job_006").content_hash = "stale"
    db.delete(db.get(JobEmbeddingORM, "job_007"))
    db.get(JobORM, "job_008").constraint_excerpts = {"_embedding": None}
    db.get(JobORM, "job_009").constraint_excerpts = []
    third = ingest_jobs_snapshot(db, str(SNAPSHOT))
    assert third["embedded"] == 5 and len(offline_embeddings[-1]) == 5
    before = {
        row.job_id: row.content_hash for row in db.scalars(select(JobEmbeddingORM))
    }
    updated = apply_update_fixture(db)
    assert updated["embedded"] == 1 and updated["embedding_calls"] == 1
    assert db.get(JobORM, "job_003").norm_min_yoe == 2.0
    assert db.get(JobMetadataORM, "job_004").source_status == "closed"
    after = {
        row.job_id: row.content_hash for row in db.scalars(select(JobEmbeddingORM))
    }
    assert [key for key in before if before[key] != after[key]] == ["job_003"]
    assert apply_update_fixture(db)["embedded"] == 0
    changed = copy.deepcopy(corpus[9:10])
    changed[0]["title"] += " Security"
    changed[0]["raw_description"] += "\n\nTest string: '; DROP TABLE jobs; --"
    assert ingest_jobs_snapshot(db, write_snapshot(tmp_path, changed))["embedded"] == 1
    assert count(db, JobORM) == 300
    assert db.get(JobORM, "job_010").raw_description == changed[0]["raw_description"]
    # Metadata-only changes do not need new embeddings.
    changed[0]["source_url"] += "?utm_source=other"
    changed[0]["source_status"] = "closed"
    changed[0]["captured_at"] = "2026-11-20T10:00:00+05:30"
    assert ingest_jobs_snapshot(db, write_snapshot(tmp_path, changed))["embedded"] == 0
    assert db.get(JobMetadataORM, "job_010").captured_at.hour == 4


def test_postgresql_dedup_transitive_incremental_and_split(
    db, offline_embeddings, tmp_path, corpus
):
    a, b, c = copy.deepcopy(corpus[:3])
    # A -> B by canonical URL, B -> C by normalized company/title.
    b["source_url"] = (
        a["source_url"].upper().replace("/ROLES/JOB_001", "/roles/job_001")
        + "/?utm_source=x#section"
    )
    b["company"], b["title"] = "Bridge Company", "Bridge Role"
    c["company"], c["title"] = "  BRIDGE   COMPANY ", "bridge role"
    ingest_jobs_snapshot(db, write_snapshot(tmp_path, [c]))
    ingest_jobs_snapshot(db, write_snapshot(tmp_path, [b, a]))
    assert db.get(JobMetadataORM, "job_001").is_duplicate is False
    assert all(
        db.get(JobMetadataORM, key).is_duplicate for key in ("job_002", "job_003")
    )
    assert (
        len(
            {
                db.get(JobMetadataORM, key).dedup_key
                for key in ("job_001", "job_002", "job_003")
            }
        )
        == 1
    )
    b["source_url"] = "https://separate.example/role?id=2"
    b["title"] = "Separate Role"
    ingest_jobs_snapshot(db, write_snapshot(tmp_path, [b]))
    assert all(not row.is_duplicate for row in db.scalars(select(JobMetadataORM)))
    assert len({row.dedup_key for row in db.scalars(select(JobMetadataORM))}) == 3


def test_postgresql_embedding_failures_rollback_and_preserve_caller(
    db, offline_embeddings, monkeypatch, tmp_path, corpus
):
    ingest_jobs_snapshot(db, write_snapshot(tmp_path, corpus[:1]))
    original = db.get(JobORM, "job_001").raw_description
    changed = copy.deepcopy(corpus[:2])
    changed[0]["raw_description"] += "\nChanged content."
    path = write_snapshot(tmp_path, changed)

    def failure(self, texts, task_type):
        raise GeminiRequestError("Simulated provider timeout; no fallback")

    monkeypatch.setattr(ingestion.GeminiService, "embed_texts", failure)
    with pytest.raises(GeminiRequestError, match="timeout"):
        ingest_jobs_snapshot(db, path)
    assert (
        count(db, JobORM)
        == count(db, JobMetadataORM)
        == count(db, JobEmbeddingORM)
        == 1
    )
    assert db.get(JobORM, "job_001").raw_description == original
    for bad in (
        None,
        [],
        [None, None],
        [[0.0] * 768] * 2,
        [[float("nan")] * 768] * 2,
        [[1.0] * 767] * 2,
        [[1e40] * 768] * 2,
        [[1e-50] * 768] * 2,
    ):
        monkeypatch.setattr(
            ingestion.GeminiService,
            "embed_texts",
            lambda self, texts, task_type, bad=bad: bad,
        )
        with pytest.raises(GeminiResponseError):
            ingest_jobs_snapshot(db, path)
        assert count(db, JobORM) == 1
        assert db.get(JobORM, "job_001").raw_description == original
    monkeypatch.setattr(
        ingestion.GeminiService, "embed_texts", GeminiService.embed_texts
    )
    normalizer = ingestion.normalize_job_constraints

    def invalid_db_value(raw, excerpts):
        return {**normalizer(raw, excerpts), "norm_min_yoe": -1.0}

    monkeypatch.setattr(ingestion, "normalize_job_constraints", invalid_db_value)
    with pytest.raises(IntegrityError):
        ingest_jobs_snapshot(db, path)
    assert count(db, JobORM) == count(db, JobEmbeddingORM) == 1
    assert db.get(JobORM, "job_001").raw_description == original


def test_postgresql_provenance_transition_never_reuses_mock_as_live(
    db, offline_embeddings, monkeypatch, tmp_path, corpus
):
    path = write_snapshot(tmp_path, corpus[:1])
    assert ingest_jobs_snapshot(db, path)["used_mock_fallback"]
    # Simulate live mode only to assert re-embedding is attempted, never claim live vectors.
    monkeypatch.setattr(
        ingestion,
        "settings",
        ingestion.settings.model_copy(
            update={"GEMINI_API_KEY": "test-placeholder-not-a-real-key"}
        ),
    )

    class UnavailableLiveService:
        def __init__(self, **kwargs):
            raise GeminiRequestError("Live embedding requested; simulated outage")

    monkeypatch.setattr(ingestion, "GeminiService", UnavailableLiveService)
    with pytest.raises(GeminiRequestError, match="Live embedding requested"):
        ingest_jobs_snapshot(db, path)
    assert (
        db.get(JobORM, "job_001").constraint_excerpts["_embedding"][
            "used_mock_fallback"
        ]
        is True
    )


def test_postgresql_missing_update_fixture_is_atomic(db, offline_embeddings):
    with pytest.raises(ValueError, match="requires ingested"):
        apply_update_fixture(db)
    assert count(db, JobORM) == 0 and not offline_embeddings


def test_hash_tracks_exact_embedding_text(corpus):
    prepared = ingestion._prepare(corpus[0])
    expected = "\n".join(
        corpus[0][field] for field in ("title", "company", "raw_description")
    )
    assert (
        prepared["content_hash"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    )
    changed = dict(corpus[0], source_status="closed")
    assert ingestion._prepare(changed)["content_hash"] == prepared["content_hash"]
    changed["title"] += " New Title"
    assert ingestion._prepare(changed)["content_hash"] != prepared["content_hash"]


def test_url_normalization_preserves_identity_query_parameters():
    normalize = ingestion._normalized_url
    assert (
        normalize("https://JOBS.EXAMPLE:443/roles/1/?utm_source=a&b=2&a=1#top")
        == "https://jobs.example/roles/1?a=1&b=2"
    )
    assert normalize("https://jobs.example/?id=1") != normalize(
        "https://jobs.example/?id=2"
    )
    assert normalize("https://jobs.example:8443/role") != normalize(
        "https://jobs.example/role"
    )
    for url in (
        "https://name:secret@jobs.example/role",
        "file:///tmp/role",
        "https://jobs.example:bad/role",
    ):
        with pytest.raises(ValueError):
            normalize(url)


def test_dedup_component_order_and_numeric_id_boundary():
    jobs = {
        "job_1001": JobORM(
            job_id="job_1001",
            company="B",
            title="Two",
            source_url="https://jobs.example/other",
        ),
        "job_1000": JobORM(
            job_id="job_1000",
            company=" b ",
            title="TWO",
            source_url="https://jobs.example/role?utm_source=test",
        ),
        "job_999": JobORM(
            job_id="job_999",
            company="A",
            title="One",
            source_url="https://jobs.example/role",
        ),
    }
    groups = ingestion._dedup_groups(jobs)
    assert not groups["job_999"][1]
    assert groups["job_1000"][1] and groups["job_1001"][1]
    assert len({value[0] for value in groups.values()}) == 1
    assert groups == ingestion._dedup_groups(dict(reversed(list(jobs.items()))))
    statuses = {"job_999": "closed", "job_1000": "closed", "job_1001": "active"}
    promoted = ingestion._dedup_groups(jobs, statuses)
    assert not promoted["job_1001"][1]
    assert promoted["job_999"][1] and promoted["job_1000"][1]


@pytest.mark.parametrize(
    ("description", "field", "value"),
    [
        ("This role is not fully remote.", "norm_work_mode", "unknown"),
        ("This is not a hybrid role.", "norm_work_mode", "unknown"),
        (
            "Hybrid work is not available; this role is fully remote.",
            "norm_work_mode",
            "remote",
        ),
        ("No remote work is allowed.", "norm_work_mode", "unknown"),
        ("Remote - India. Remote work is not available.", "has_conflict", True),
        (
            "Onsite - Bengaluru. Onsite attendance is not required.",
            "has_conflict",
            True,
        ),
        ("Hybrid - Pune. Hybrid work is prohibited.", "has_conflict", True),
        ("Minimum 3 years of experience is not required.", "norm_min_yoe", None),
        ("Applicants do not require 3 years of experience.", "norm_min_yoe", None),
        ("No minimum 3 years of experience.", "norm_min_yoe", None),
        ("Minimum 3 years of experience; SQL preferred.", "norm_min_yoe", 3.0),
        ("Minimum 3 years of experience, preferred: SQL.", "norm_min_yoe", 3.0),
        (
            "Preferred: 5 years of experience; required: minimum 2 years of experience.",
            "norm_min_yoe",
            2.0,
        ),
        (
            "Students are eligible, but graduates are not eligible.",
            "norm_student_eligible",
            True,
        ),
        (
            "Students are eligible and graduates are not eligible.",
            "norm_student_eligible",
            True,
        ),
        (
            "Students are eligible, but current students are not eligible.",
            "has_conflict",
            True,
        ),
    ],
)
def test_review_regressions_for_negation_and_clause_scope(description, field, value):
    result = normalized(description)
    assert result[field] == value
    if result["has_conflict"]:
        assert result["constraint_excerpts"]["conflict_fields"]
    assert all(
        key.startswith("job_901_ex_")
        for name in FIELDS
        for key in result["constraint_excerpts"][name]
    )


@pytest.mark.parametrize(
    "vector",
    [
        None,
        [],
        [None] * 768,
        [True] * 768,
        ["1"] * 768,
        [1.0] * 767,
        [1.0] * 769,
        [0.0] * 768,
        [float("inf")] * 768,
        [float("nan")] * 768,
        [1e40] * 768,
        [1e-50] * 768,
        [10**1000] * 768,
    ],
)
def test_invalid_float32_vectors_rejected_before_database_writes(vector):
    with pytest.raises(GeminiResponseError):
        ingestion._validated_embedding(vector)


def test_float32_validation_preserves_valid_vectors_and_input():
    vector = [1.0, -0.5] + [0.0] * 766
    result = ingestion._validated_embedding(vector)
    assert result == vector and result is not vector
    assert ingestion._validated_embedding([1e-40] + [0.0] * 767)[0] > 0


@pytest.mark.parametrize(
    "proof", [None, [], {"_embedding": None}, {"_embedding": []}, {}]
)
def test_missing_or_malformed_embedding_provenance_invalidates_reuse(proof):
    assert ingestion._embedding_provenance(proof) == {}


@pytest.mark.parametrize(
    "url",
    [
        "https://jobs.example:0/role",
        "https://jobs.example:65536/role",
        "https://jobs.example:bad/role",
        "https://jobs.example/a b",
        "https://jobs.example/role\n",
        "https://jobs.example/role?id=%ff",
    ],
)
def test_malformed_urls_do_not_silently_alias_valid_urls(url):
    with pytest.raises(ValueError):
        ingestion._normalized_url(url)


def test_url_normalization_keeps_repeated_query_value_order():
    assert ingestion._normalized_url(
        "https://jobs.example/?id=1&id=2"
    ) != ingestion._normalized_url("https://jobs.example/?id=2&id=1")
