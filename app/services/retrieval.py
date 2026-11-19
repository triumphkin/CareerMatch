"""Synchronous PostgreSQL FTS + cosine retrieval, followed by the shared gates.

Results contain only JSON-compatible values. A query vector is generated once.
FTS-only rows remain searchable when their vector is missing/stale/incompatible;
mock and live vectors never share the vector ranking. RRF scores are rankings,
not calibrated probabilities. Equal scores are ordered by job_id for repeatability.
"""

import math
import struct

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, bindparam, exists, func, literal, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import JobEmbeddingORM, JobMetadataORM, JobORM
from app.schemas.contracts import HardConstraints, RejectedJobCounterfactual, TriState
from app.services.constraints import evaluate_job_constraints, validate_hard_constraints
from app.services.llm_client import GeminiResponseError, GeminiService


def _query_vector(vectors: object) -> list[float]:
    message = "Query embedding must contain one finite, nonzero 768-dimensional float32 vector"
    if (
        not isinstance(vectors, list)
        or len(vectors) != 1
        or not isinstance(vectors[0], (list, tuple))
        or len(vectors[0]) != 768
    ):
        raise GeminiResponseError(message)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in vectors[0]
    ):
        raise GeminiResponseError(message)
    try:
        vector = list(struct.unpack("!768f", struct.pack("!768f", *vectors[0])))
    except (OverflowError, struct.error) as error:
        raise GeminiResponseError(message) from error
    if not all(math.isfinite(value) for value in vector) or math.hypot(*vector) == 0:
        raise GeminiResponseError(message)
    return vector


def _retrieval_statement():
    active = (
        JobMetadataORM.source_status == "active",
        JobMetadataORM.is_duplicate.is_(False),
    )
    query = func.websearch_to_tsquery("english", bindparam("query"))
    lexical_score = func.ts_rank_cd(JobORM.search_tsv, query)
    lexical_order = (lexical_score.desc(), JobORM.job_id.asc())
    lexical_cte = (
        select(
            JobORM.job_id,
            func.row_number().over(order_by=lexical_order).label("lex_rank"),
        )
        .join(JobMetadataORM, JobMetadataORM.job_id == JobORM.job_id)
        .where(*active, JobORM.search_tsv.op("@@")(query))
        .order_by(*lexical_order)
        .limit(40)
        .cte("lexical_cte")
    )
    distance = JobEmbeddingORM.embedding.cosine_distance(
        bindparam("query_vec", type_=Vector(768))
    )
    vector_order = (distance.asc(), JobORM.job_id.asc())
    vector_cte = (
        select(
            JobORM.job_id,
            func.row_number().over(order_by=vector_order).label("vec_rank"),
        )
        .join(JobMetadataORM, JobMetadataORM.job_id == JobORM.job_id)
        .join(JobEmbeddingORM, JobEmbeddingORM.job_id == JobORM.job_id)
        .where(
            *active,
            JobEmbeddingORM.model_id == bindparam("model_id"),
            JobEmbeddingORM.dimension == 768,
            JobEmbeddingORM.content_hash == JobMetadataORM.content_hash,
            JobORM.constraint_excerpts["_embedding"]["model_id"].as_string()
            == bindparam("model_id"),
            JobORM.constraint_excerpts["_embedding"]["used_mock_fallback"]
            == bindparam("embedding_is_mock", type_=JSONB),
            distance >= 0.0,
            distance <= 2.0,
        )
        .order_by(*vector_order)
        .limit(40)
        .cte("vector_cte")
    )
    fused_id = func.coalesce(lexical_cte.c.job_id, vector_cte.c.job_id)
    score = (
        literal(1.0, Float) / (60 + func.coalesce(lexical_cte.c.lex_rank, 100))
        + literal(1.0, Float) / (60 + func.coalesce(vector_cte.c.vec_rank, 100))
    ).label("rrf_score")
    fused = (
        select(
            fused_id.label("job_id"),
            lexical_cte.c.lex_rank,
            vector_cte.c.vec_rank,
            score,
        )
        .select_from(
            lexical_cte.join(
                vector_cte, lexical_cte.c.job_id == vector_cte.c.job_id, full=True
            )
        )
        .order_by(score.desc(), fused_id.asc())
        .limit(25)
        .cte("fused_candidates")
    )
    return (
        select(
            *(
                column.label(f"job_{column.name}")
                for column in JobORM.__table__.columns
            ),
            *(
                column.label(f"meta_{column.name}")
                for column in JobMetadataORM.__table__.columns
            ),
            fused.c.lex_rank,
            fused.c.vec_rank,
            fused.c.rrf_score,
        )
        .select_from(JobORM)
        .join(fused, fused.c.job_id == JobORM.job_id)
        .join(JobMetadataORM, JobMetadataORM.job_id == JobORM.job_id)
        .order_by(fused.c.rrf_score.desc(), JobORM.job_id.asc())
    )


def hybrid_search_jobs(
    db: Session, query: str, hard: HardConstraints, limit: int = 10
) -> dict:
    """Evaluate at most 25 fused rows; return up to limit candidates and 5 rejections.

    Empty/whitespace queries or limit=0 are no-ops. A limit above 25 is naturally
    bounded by the evaluated pool. No profile or query cache can retain stale
    preferences. Retrieval never commits or modifies persisted data.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if "\x00" in query:
        raise ValueError("query must be NUL-free")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    hard = validate_hard_constraints(hard)
    result = {
        "eligible_candidates": [],
        "rejected_counterfactuals": [],
        "candidates_evaluated": 0,
        "used_mock_fallback": False,
        "embedding_calls": 0,
        "embedding_usage": None,
    }
    if not query.strip() or limit == 0:
        return result
    if db.get_bind().dialect.name != "postgresql":
        raise ValueError("Hybrid retrieval requires PostgreSQL and pgvector")
    # Do not spend an embedding request on a corpus with no active, unique rows.
    with db.no_autoflush:
        if not db.scalar(
            select(
                exists().where(
                    JobMetadataORM.source_status == "active",
                    JobMetadataORM.is_duplicate.is_(False),
                )
            )
        ):
            return result
        with GeminiService(config=settings) as service:
            vector = _query_vector(
                service.embed_texts([query], task_type="RETRIEVAL_QUERY")
            )
            mock = service.used_mock_fallback
            usage = service.last_embedding_usage
        result.update(
            used_mock_fallback=mock,
            embedding_calls=1,
            embedding_usage=usage.model_dump(mode="json")
            if usage is not None
            else None,
        )
        rows = (
            db.execute(
                _retrieval_statement(),
                {
                    "query": query,
                    "query_vec": vector,
                    "model_id": settings.EMBEDDING_MODEL_ID,
                    "embedding_is_mock": mock,
                },
            )
            .mappings()
            .all()
        )
        for row in rows:
            # Detached snapshots avoid stale identity-map values and never discard
            # the caller's dirty ORM attributes through populate_existing refreshes.
            job = JobORM(
                **{
                    column.name: row[f"job_{column.name}"]
                    for column in JobORM.__table__.columns
                }
            )
            meta = JobMetadataORM(
                **{
                    column.name: row[f"meta_{column.name}"]
                    for column in JobMetadataORM.__table__.columns
                }
            )
            lex_rank, vec_rank, score = (
                row["lex_rank"],
                row["vec_rank"],
                row["rrf_score"],
            )
            overall, checks = evaluate_job_constraints(job, meta, hard)
            result["candidates_evaluated"] += 1
            if overall == TriState.FAIL:
                if len(result["rejected_counterfactuals"]) < 5:
                    failures = [item for item in checks if item.result == TriState.FAIL]
                    rejection = RejectedJobCounterfactual(
                        job_id=job.job_id,
                        title=job.title,
                        company=job.company,
                        similarity_score=float(score),
                        failed_constraints=failures,
                        rejection_summary=" ".join(
                            f"{item.field}: {item.reason}" for item in failures
                        ),
                    )
                    result["rejected_counterfactuals"].append(
                        rejection.model_dump(mode="json")
                    )
            elif len(result["eligible_candidates"]) < limit:
                result["eligible_candidates"].append(
                    {
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "source_url": job.source_url,
                        "captured_at": meta.captured_at.isoformat(),
                        "rrf_score": float(score),
                        "lex_rank": lex_rank,
                        "vec_rank": vec_rank,
                        "constraint_result": overall.value,
                        "constraint_checks": [
                            item.model_dump(mode="json") for item in checks
                        ],
                    }
                )
    return result
