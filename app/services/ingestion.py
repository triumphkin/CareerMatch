"""Deterministic ingestion; callers commit their synchronous Session explicitly.

No listings are fetched from the network. GeminiService is used only for batched
embeddings. Its offline provenance is persisted under constraint_excerpts['_embedding']
so synthetic vectors cannot silently survive a subsequent live ingestion.
The existing JSONB also stores has_conflict and conflict_fields (no schema change).
"""

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import JobEmbeddingORM, JobMetadataORM, JobORM
from app.services.llm_client import GeminiResponseError, GeminiService

_FIELDS = (
    "norm_work_mode",
    "norm_remote_region",
    "norm_cities",
    "norm_student_eligible",
    "norm_degree_completed_required",
    "norm_min_yoe",
)
_CITIES = (
    "Bengaluru",
    "Delhi",
    "Hyderabad",
    "Pune",
    "Mumbai",
    "Chennai",
    "San Francisco",
    "New York",
    "Toronto",
    "Vancouver",
)
_UPDATE_TEXT = "Required: at least 2 years of professional experience."


def _check_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not re.fullmatch(r"job_[0-9]{3,}", job_id):
        raise ValueError("Expected a positive canonical job_NNN identifier")
    number = int(job_id[4:])
    if number < 1 or job_id != f"job_{number:03d}":
        raise ValueError("Expected a positive canonical job_NNN identifier")


def slice_into_excerpts(job_id: str, raw_text: str) -> dict[str, str]:
    """One sentence per excerpt, with headings/list lines kept as atomic blocks.

    Whitespace at the edges is removed; each excerpt is otherwise an exact raw
    substring. Decimal numbers and common dotted abbreviations are not split.
    """
    _check_job_id(job_id)
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    blocks = []
    for line in raw_text.splitlines():
        start = 0
        for match in re.finditer(r'[.!?]["\u201d\u2019]?\s+', line):
            end = match.start() + 1
            prefix = line[start:end]
            if re.search(
                r"\b(?:Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|U\.S|U\.K)\.$", prefix, re.IGNORECASE
            ):
                continue
            block = line[start : match.end()].strip()
            if block:
                blocks.append(block)
            start = match.end()
        if line[start:].strip():
            blocks.append(line[start:].strip())
    return {f"{job_id}_ex_{index:02d}": block for index, block in enumerate(blocks, 1)}


def normalize_job_constraints(raw_job: dict, excerpts: dict[str, str]) -> dict:
    """Conservative text rules, independent of IDs/categories and any model output.

    Unknown booleans/experience are None; unknown enum fields use 'unknown'.
    Conflicting assertions retain both citations but resolve the field to unknown.
    Multiple nonzero skill-specific experience minima resolve to their maximum.
    """
    if not isinstance(raw_job, dict):
        raise TypeError("raw_job must be a dict")
    job_id = raw_job.get("job_id")
    expected = slice_into_excerpts(job_id, raw_job.get("raw_description"))
    if not isinstance(excerpts, dict) or excerpts != expected:
        raise ValueError(
            "Excerpts must exactly match the deterministic raw-text slicing"
        )
    evidence = {field: defaultdict(list) for field in _FIELDS}
    denied_modes = defaultdict(list)

    def record(field, value, excerpt_id):
        if excerpt_id not in evidence[field][value]:
            evidence[field][value].append(excerpt_id)

    for excerpt_id, sentence in _constraint_clauses(expected):
        # Optional skills/experience must never become hard requirements.
        if re.search(r"\bpreferred\b|nice to have|\boptional\b", sentence):
            continue
        denied_here = set()
        for mode, pattern in (
            ("remote", "remote"),
            ("hybrid", "hybrid"),
            ("onsite", "on[ -]?site"),
        ):
            if re.search(
                rf"\bnot (?:an? )?(?:fully )?{pattern}\b|"
                rf"\b{pattern}(?: work| working| attendance| position| role)? (?:is |are )?"
                r"(?:not (?:required|available|allowed|permitted)|unavailable|prohibited)\b|"
                rf"\bno {pattern}(?: work| working)?\b",
                sentence,
            ):
                denied_here.add(mode)
                denied_modes[mode].append(excerpt_id)
        hybrid = bool(re.search(r"\bhybrid\b|office three days per week", sentence))
        remote = bool(
            re.search(
                r"^(?:100%\s*)?remote(?:\s*[-\u2013:]|[.!]|$)|\bfully remote\b|"
                r"\bwork (?:is performed )?remotely\b|\bcan work remotely\b|"
                r"remote work is (?:available|allowed|permitted)\b",
                sentence,
            )
        )
        onsite = bool(
            re.search(
                r"\bon[ -]?site\b|\boffice.*(?:five|5) days per week|"
                r"(?:five|5) days/week.*office",
                sentence,
            )
        )
        hybrid = hybrid and "hybrid" not in denied_here
        remote = remote and "remote" not in denied_here
        onsite = onsite and "onsite" not in denied_here
        if hybrid:
            record("norm_work_mode", "hybrid", excerpt_id)
        else:
            if remote:
                record("norm_work_mode", "remote", excerpt_id)
            if onsite:
                record("norm_work_mode", "onsite", excerpt_id)
        if re.search(
            r"remote\s*[-\u2013:]\s*india\b|residents anywhere in india|"
            r"applicants in india can work remotely",
            sentence,
        ):
            record("norm_remote_region", "india_eligible", excerpt_id)
        if re.search(r"remote\s*[-\u2013:]\s*global\b|applicants worldwide", sentence):
            record("norm_remote_region", "global_remote", excerpt_id)
        if re.search(
            r"(?:us|u\.s\.|united states|canada) residents only|"
            r"must reside in (?:the united states|canada)|"
            r"india-based applicants are not eligible",
            sentence,
        ):
            record("norm_remote_region", "non_india_restricted", excerpt_id)
        if onsite or hybrid or "office" in sentence:
            for city in _CITIES:
                if re.search(r"\b" + re.escape(city.casefold()) + r"\b", sentence):
                    record("norm_cities", city, excerpt_id)
            if re.search(r"\bbangalore\b", sentence):
                record("norm_cities", "Bengaluru", excerpt_id)
        if re.search(
            r"\bstudents(?:\s+who\b[^,;.!?]*)?\s+(?:are\s+)?(?:not eligible|ineligible)\b",
            sentence,
        ):
            record("norm_student_eligible", False, excerpt_id)
        if re.search(r"\bstudents (?:are )?(?:eligible|welcome)\b", sentence):
            record("norm_student_eligible", True, excerpt_id)
        if re.search(
            r"\bdegree (?:is )?not required\b|\bno (?:completed )?degree (?:is )?required",
            sentence,
        ):
            record("norm_degree_completed_required", False, excerpt_id)
        if re.search(
            r"\bcompleted (?:bachelor[\u2019\x27]s |master[\u2019\x27]s )?degree (?:is )?required\b|"
            r"\bcompleted graduation (?:is )?required\b",
            sentence,
        ):
            record("norm_degree_completed_required", True, excerpt_id)
        if re.search(
            r"no (?:prior )?(?:professional )?experience is required", sentence
        ):
            record("norm_min_yoe", 0.0, excerpt_id)
        if re.search(
            r"\bexperience (?:is )?not required\b|\bnot (?:requires?|requiring)\b|\bno minimum\b",
            sentence,
        ):
            continue
        for pattern in (
            r"(?:minimum|at least|requires?|requiring)\s+(\d+(?:\.\d+)?)\+?\s+years? (?:of )?(?:professional |relevant )?experience",
            r"minimum experience is (\d+(?:\.\d+)?) years?",
            r"(?:^|required:\s*)(\d+(?:\.\d+)?)\+\s+years? (?:of )?(?:professional |relevant )?experience",
        ):
            for match in re.finditer(pattern, sentence):
                years = float(match.group(1))
                if math.isfinite(years):
                    record("norm_min_yoe", years, excerpt_id)

    result = {}
    conflicts = []
    proof = {}
    for field in _FIELDS:
        values = set(evidence[field])
        proof[field] = [
            key
            for key in expected
            if any(key in ids for ids in evidence[field].values())
        ]
        if field == "norm_cities":
            result[field] = sorted(values)
            continue
        if field == "norm_remote_region" and values == {
            "india_eligible",
            "global_remote",
        }:
            values = {"global_remote"}
        if (
            field == "norm_min_yoe"
            and values
            and (0.0 not in values or len(values) == 1)
        ):
            values = {max(values)}
        unknown = (
            "unknown" if field in ("norm_work_mode", "norm_remote_region") else None
        )
        result[field] = next(iter(values)) if len(values) == 1 else unknown
        if len(values) > 1:
            conflicts.append(field)
    if set(evidence["norm_work_mode"]) & set(denied_modes):
        conflicts.append("norm_work_mode")
        result["norm_work_mode"] = "unknown"
    work_proof = set(proof["norm_work_mode"])
    work_proof.update(key for keys in denied_modes.values() for key in keys)
    proof["norm_work_mode"] = [key for key in expected if key in work_proof]
    if (
        result["norm_student_eligible"] is True
        and result["norm_degree_completed_required"] is True
    ):
        conflicts.extend(("norm_student_eligible", "norm_degree_completed_required"))
        result["norm_student_eligible"] = result["norm_degree_completed_required"] = (
            None
        )
    proof["has_conflict"] = bool(conflicts)
    proof["conflict_fields"] = sorted(set(conflicts))
    result["has_conflict"] = bool(conflicts)
    result["constraint_excerpts"] = proof
    return result


def _constraint_clauses(excerpts: dict[str, str]):
    """Keep proving IDs while separating required, optional and contrasting clauses."""
    for key, original in excerpts.items():
        sentence = original.casefold()
        if re.search(
            r"system override|ignore all (?:gaps|instructions)|output decision|\bclaim \d+ years",
            sentence,
        ):
            continue
        for clause in re.split(
            r";|\b(?:but|however|whereas)\b|,\s*(?=(?:preferred|required|optional|nice to have)\b)",
            sentence,
        ):
            if clause.strip(" ,"):
                yield key, clause.strip(" ,")


def _normalized_url(value: str) -> str:
    if not isinstance(value, str) or re.search(r"[\x00-\x20\x7f]", value):
        raise ValueError(
            "source_url must be a URL without whitespace or control characters"
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "source_url must be an absolute HTTP(S) URL without credentials"
        )
    port = parsed.port
    if port == 0:
        raise ValueError("source_url port must be between 1 and 65535")
    host = parsed.hostname.casefold()
    if ":" in host:
        host = f"[{host}]"
    if port and (parsed.scheme, port) not in (("https", 443), ("http", 80)):
        host += f":{port}"
    query = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True, errors="strict")
        if not key.casefold().startswith("utm_")
        and key.casefold() not in ("gclid", "fbclid")
    ]
    return urlunsplit(
        (
            parsed.scheme,
            host,
            parsed.path.rstrip("/") or "/",
            urlencode(sorted(query, key=lambda pair: pair[0])),
            "",
        )
    )


def _text_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _embedding_text(raw: dict) -> str:
    return "\n".join(raw[field] for field in ("title", "company", "raw_description"))


def _prepare(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise TypeError("Each listing must be an object")
    for field in (
        "job_id",
        "source_url",
        "title",
        "company",
        "raw_description",
        "captured_at",
        "source_status",
    ):
        if (
            not isinstance(raw.get(field), str)
            or not raw[field].strip()
            or "\x00" in raw[field]
        ):
            raise ValueError(f"Listing requires a nonempty, NUL-free string: {field}")
    _check_job_id(raw["job_id"])
    _normalized_url(raw["source_url"])
    if raw["source_status"] not in ("active", "closed"):
        raise ValueError("source_status must be active or closed")
    captured = datetime.fromisoformat(raw["captured_at"])
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("captured_at must include a timezone")
    excerpts = slice_into_excerpts(raw["job_id"], raw["raw_description"])
    normalized = normalize_job_constraints(raw, excerpts)
    return {
        "raw": dict(raw),
        "captured_at": captured.astimezone(UTC),
        "excerpts": excerpts,
        "normalized": normalized,
        "content_hash": hashlib.sha256(
            _embedding_text(raw).encode("utf-8")
        ).hexdigest(),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _dedup_groups(
    jobs: dict[str, JobORM], statuses: dict[str, str] | None = None
) -> dict[str, tuple[str, bool]]:
    """Connected components handle URL OR company/title, including transitive matches."""
    ids = sorted(
        jobs,
        key=lambda key: (
            (statuses or {}).get(key, "active") == "closed",
            int(key[4:]) if re.fullmatch(r"job_\d+", key) else math.inf,
            key,
        ),
    )
    parent = {key: key for key in ids}
    order = {key: index for index, key in enumerate(ids)}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for key in ids:
        job = jobs[key]
        for token in (
            ("url", _normalized_url(job.source_url)),
            ("title", _text_key(job.company), _text_key(job.title)),
        ):
            if token in seen:
                first, second = root(key), root(seen[token])
                if first != second:
                    winner, loser = sorted((first, second), key=order.__getitem__)
                    parent[loser] = winner
            else:
                seen[token] = key
    return {
        key: (hashlib.sha256(root(key).encode()).hexdigest(), root(key) != key)
        for key in ids
    }


def _validated_embedding(vector: object) -> list[float]:
    """Validate the float32 values PostgreSQL will actually store, including underflow."""
    message = "Invalid ingestion embedding; expected a finite, nonzero 768-dimensional float32 vector"
    if not isinstance(vector, (list, tuple)) or len(vector) != 768:
        raise GeminiResponseError(message)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in vector
    ):
        raise GeminiResponseError(message)
    try:
        stored = list(struct.unpack("!768f", struct.pack("!768f", *vector)))
    except (OverflowError, struct.error) as error:
        raise GeminiResponseError(message) from error
    if not all(math.isfinite(value) for value in stored) or math.hypot(*stored) == 0:
        raise GeminiResponseError(message)
    return stored


def _embedding_provenance(proof: object) -> dict:
    if not isinstance(proof, dict) or not isinstance(proof.get("_embedding"), dict):
        return {}
    return proof["_embedding"]


def _ingest_prepared(db: Session, prepared: list[dict]) -> dict:
    report = {
        "ingested": len(prepared),
        "inserted": 0,
        "embedded": 0,
        "embedding_calls": 0,
        "duplicates": 0,
        "used_mock_fallback": False,
        "embedding_model_id": settings.EMBEDDING_MODEL_ID,
    }
    if not prepared:
        return report
    if db.get_bind().dialect.name != "postgresql":
        raise ValueError("Ingestion requires real PostgreSQL with pgvector")
    # Savepoint failures leave the caller's transaction usable; caller owns commit.
    with db.begin_nested():
        # Serialize corpus writers across processes, not just threads in this worker.
        db.execute(select(func.pg_advisory_xact_lock(1668110606)))
        jobs = {
            row.job_id: row
            for row in db.scalars(
                select(JobORM).execution_options(populate_existing=True)
            )
        }
        metadata = {
            row.job_id: row
            for row in db.scalars(
                select(JobMetadataORM).execution_options(populate_existing=True)
            )
        }
        embeddings = {
            row.job_id: row
            for row in db.scalars(
                select(JobEmbeddingORM).execution_options(populate_existing=True)
            )
        }
        pending = []
        for item in prepared:
            raw = item["raw"]
            key = raw["job_id"]
            job = jobs.get(key)
            previous = _embedding_provenance(job.constraint_excerpts) if job else {}
            embedding = embeddings.get(key)
            expected_mock = not bool(settings.GEMINI_API_KEY.strip())
            reuse = (
                embedding is not None
                and embedding.content_hash == item["content_hash"]
                and embedding.model_id == settings.EMBEDDING_MODEL_ID
                and embedding.dimension == 768
                and previous.get("used_mock_fallback") is expected_mock
                and previous.get("model_id") == settings.EMBEDDING_MODEL_ID
            )
            if job is None:
                job = JobORM(job_id=key)
                jobs[key] = job
                db.add(job)
                report["inserted"] += 1
            for field in ("source_url", "title", "company", "raw_description"):
                setattr(job, field, raw[field])
            job.excerpts = item["excerpts"]
            for field in _FIELDS:
                setattr(job, field, item["normalized"][field])
            job.constraint_excerpts = dict(item["normalized"]["constraint_excerpts"])
            if reuse:
                job.constraint_excerpts = {
                    **job.constraint_excerpts,
                    "_embedding": previous,
                }
                report["used_mock_fallback"] |= previous["used_mock_fallback"]
            else:
                pending.append(item)
            job.search_tsv = func.to_tsvector("english", _embedding_text(raw))
            meta = metadata.get(key)
            if meta is None:
                meta = JobMetadataORM(job_id=key)
                metadata[key] = meta
            meta.captured_at = item["captured_at"]
            meta.content_hash = item["content_hash"]
            meta.source_status = raw["source_status"]
        if pending:
            with GeminiService(config=settings) as service:
                vectors = service.embed_texts(
                    [_embedding_text(item["raw"]) for item in pending],
                    task_type="RETRIEVAL_DOCUMENT",
                )
                mock = service.used_mock_fallback
            if not isinstance(vectors, list) or len(vectors) != len(pending):
                raise GeminiResponseError("Embedding count does not match changed jobs")
            for item, vector in zip(pending, vectors, strict=True):
                vector = _validated_embedding(vector)
                key = item["raw"]["job_id"]
                embedding = embeddings.get(key)
                if embedding is None:
                    embedding = JobEmbeddingORM(job_id=key)
                    embeddings[key] = embedding
                embedding.model_id = settings.EMBEDDING_MODEL_ID
                embedding.dimension = 768
                embedding.content_hash = item["content_hash"]
                embedding.embedding = vector
                jobs[key].constraint_excerpts = {
                    **jobs[key].constraint_excerpts,
                    "_embedding": {
                        "used_mock_fallback": mock,
                        "model_id": settings.EMBEDDING_MODEL_ID,
                    },
                }
            report.update(embedded=len(pending), embedding_calls=1)
            report["used_mock_fallback"] |= mock
        groups = _dedup_groups(
            jobs, {key: meta.source_status for key, meta in metadata.items()}
        )
        for key, meta in metadata.items():
            meta.dedup_key, meta.is_duplicate = groups[key]
        report["duplicates"] = sum(
            metadata[item["raw"]["job_id"]].is_duplicate for item in prepared
        )
        # Explicit parent-first flush avoids relying on ORM relationships (none exist).
        db.flush()
        db.add_all(list(metadata.values()) + list(embeddings.values()))
        db.flush()
    return report


def ingest_jobs_snapshot(db: Session, snapshot_path: str) -> dict:
    """Validate the whole file before writing. Empty snapshots are no-ops, not deletes.

    embedding_calls counts batched service invocations; the provider handles its
    own request chunking. Repeated hashes must also match model AND provenance.
    """
    raw_jobs = json.loads(
        Path(snapshot_path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(raw_jobs, list):
        raise TypeError("Snapshot must be a JSON list")
    prepared = [_prepare(raw) for raw in raw_jobs]
    ids = [item["raw"]["job_id"] for item in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("Snapshot contains duplicate job IDs")
    return _ingest_prepared(db, prepared)


def apply_update_fixture(db: Session) -> dict:
    """Idempotently change job_003's experience requirement and close job_004.

    Reuses ingestion (including hashing, citations, FTS and embedding refresh).
    The source snapshot is never edited; caller commits or rolls back explicitly.
    """
    with db.begin_nested():
        db.execute(select(func.pg_advisory_xact_lock(1668110606)))
        ids = ("job_003", "job_004")
        jobs = {
            job.job_id: job
            for job in db.scalars(
                select(JobORM)
                .where(JobORM.job_id.in_(ids))
                .execution_options(populate_existing=True)
            )
        }
        metas = {
            meta.job_id: meta
            for meta in db.scalars(
                select(JobMetadataORM)
                .where(JobMetadataORM.job_id.in_(ids))
                .execution_options(populate_existing=True)
            )
        }
        if set(jobs) != set(ids) or set(metas) != set(ids):
            raise ValueError(
                "Update fixture requires ingested job_003 and job_004 with metadata"
            )
        prepared = []
        for key in ids:
            job, meta = jobs[key], metas[key]
            raw = {
                field: getattr(job, field)
                for field in (
                    "job_id",
                    "source_url",
                    "title",
                    "company",
                    "raw_description",
                )
            }
            raw.update(
                captured_at=meta.captured_at.isoformat(),
                source_status=meta.source_status,
            )
            if key == "job_003" and _UPDATE_TEXT not in raw["raw_description"]:
                zero = "No prior professional experience is required; minimum experience is 0 years."
                if zero not in raw["raw_description"]:
                    raise ValueError(
                        "job_003 no longer matches the expected update fixture"
                    )
                raw["raw_description"] = raw["raw_description"].replace(
                    zero, _UPDATE_TEXT
                )
            if key == "job_004":
                raw["source_status"] = "closed"
            prepared.append(_prepare(raw))
        report = _ingest_prepared(db, prepared)
        return {**report, "modified_job_id": "job_003", "closed_job_id": "job_004"}
