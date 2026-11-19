"""Fair baselines sharing Discovery, hydration, and deterministic eligibility.

Retrieval-only has no model-written explanations. Single-agent performs one
batched matching call after shared preparation, without an audit/correction loop.
Runners persist in the caller's transaction; the caller commits or rolls back.
"""

from sqlalchemy.orm import Session

from app.agents.graph import run_search
from app.db.models import SearchRunORM


def run_retrieval_only_baseline(
    db: Session,
    profile_id: str,
    user_query: str,
    **kwargs,
) -> SearchRunORM:
    return run_search(
        db, profile_id, user_query, system_variant="retrieval_only", **kwargs
    )


def run_single_agent_baseline(
    db: Session,
    profile_id: str,
    user_query: str,
    **kwargs,
) -> SearchRunORM:
    return run_search(
        db, profile_id, user_query, system_variant="single_agent", **kwargs
    )
