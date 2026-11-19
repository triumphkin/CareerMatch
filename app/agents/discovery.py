"""Discovery Agent node translating user query + preferences into bounded candidate jobs.

Spec (Step 2.2):
1. Append DiscoveryQueryPlan to app/schemas/contracts.py.
2. Call GeminiService.generate_structured(..., model_id=settings.CHAT_MODEL_ID_FAST, thinking_level="LOW", response_schema=DiscoveryQueryPlan).
3. Execute search_jobs(primary_query, hard_constraints). If < 5 eligible jobs return and searches_used < 2,
   run fallback_broad_query without relaxing hard constraints and merge deduplicated results.
4. Write pure dicts (candidate_job_ids, rejected_counterfactuals, discovery_rationale) to graph state and log traces.
"""

import re
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.agents.tools import get_preferences, log_trace, search_jobs
from app.config import settings
from app.schemas.contracts import (
    DiscoveryQueryPlan,
    HardConstraints,
    LLMUsageMeta,
    SoftPreferences,
)
from app.services.constraints import validate_hard_constraints
from app.services.llm_client import GeminiService

DISCOVERY_SYSTEM_INSTRUCTION = (
    "You are CareerMatch's Discovery Agent. Your role is to translate a candidate's "
    "search query and profile preferences into two structured search queries:\n"
    "1. primary_query: A targeted hybrid search query focusing on the user's specific target roles, domains, and core skills.\n"
    "2. fallback_broad_query: A broader query capturing adjacent titles and general skills to ensure sufficient candidates if the primary search is too narrow.\n"
    "3. rationale: A concise explanation of the query generation strategy.\n\n"
    "Do not include location, work mode, or experience requirements in the queries; "
    "those are enforced deterministically by separate hard constraints."
)


def plan_discovery_queries(
    user_query: str,
    soft_preferences: SoftPreferences | dict | None = None,
    *,
    llm_service: GeminiService | None = None,
) -> tuple[DiscoveryQueryPlan, LLMUsageMeta]:
    """Generate structured primary and fallback broad retrieval queries."""
    if not isinstance(user_query, str):
        raise TypeError("user_query must be a string")
    if "\x00" in user_query:
        raise ValueError("user_query must be NUL-free")

    soft = None
    if soft_preferences is not None:
        if isinstance(soft_preferences, SoftPreferences):
            soft = soft_preferences
        elif isinstance(soft_preferences, dict):
            soft_dict = {
                "target_roles": soft_preferences.get("target_roles", []),
                "preferred_skills": soft_preferences.get("preferred_skills", []),
                "prefer_startup": soft_preferences.get("prefer_startup", False),
            }
            soft = SoftPreferences.model_validate(soft_dict)
        else:
            raise TypeError("soft_preferences must be a SoftPreferences or dict")

    query_text = user_query.strip()
    if not query_text:
        if soft is not None and soft.target_roles:
            query_text = ", ".join(soft.target_roles)
        else:
            raise ValueError(
                "user_query must be nonblank when no target roles are provided"
            )

    prompt_lines = [f"User Search Query: {query_text}"]
    if soft is not None:
        if soft.target_roles:
            prompt_lines.append(f"Target Roles: {', '.join(soft.target_roles)}")
        if soft.preferred_skills:
            prompt_lines.append(f"Preferred Skills: {', '.join(soft.preferred_skills)}")
        if soft.prefer_startup:
            prompt_lines.append("Company Preference: Prefers startup environments")

    prompt = "\n".join(prompt_lines)

    if llm_service is not None:
        return llm_service.generate_structured(
            prompt=prompt,
            system_instruction=DISCOVERY_SYSTEM_INSTRUCTION,
            response_schema=DiscoveryQueryPlan,
            model_id=settings.CHAT_MODEL_ID_FAST,
            thinking_level="LOW",
        )

    with GeminiService(config=settings) as service:
        return service.generate_structured(
            prompt=prompt,
            system_instruction=DISCOVERY_SYSTEM_INSTRUCTION,
            response_schema=DiscoveryQueryPlan,
            model_id=settings.CHAT_MODEL_ID_FAST,
            thinking_level="LOW",
        )


def run_discovery(
    db: Session,
    user_query: str,
    hard_constraints: HardConstraints | dict,
    soft_preferences: SoftPreferences | dict | None = None,
    *,
    run_id: str | None = None,
    llm_service: GeminiService | None = None,
) -> dict[str, Any]:
    """Execute query planning, bounded hybrid searches (max 2), and candidate selection."""
    if not isinstance(user_query, str):
        raise TypeError("user_query must be a string")
    if "\x00" in user_query:
        raise ValueError("user_query must be NUL-free")

    if isinstance(hard_constraints, dict):
        hard = HardConstraints.model_validate(hard_constraints)
    elif isinstance(hard_constraints, HardConstraints):
        hard = hard_constraints
    else:
        raise TypeError("hard_constraints must be a HardConstraints instance or dict")
    hard = validate_hard_constraints(hard)

    plan_start = perf_counter()
    try:
        plan, usage = plan_discovery_queries(
            user_query=user_query,
            soft_preferences=soft_preferences,
            llm_service=llm_service,
        )
        plan_duration_ms = (perf_counter() - plan_start) * 1000
    except Exception:
        if run_id is not None:
            log_trace(
                db,
                run_id,
                agent_name="discovery",
                tool_or_action="plan_queries",
                duration_ms=(perf_counter() - plan_start) * 1000,
                outcome="error",
            )
        raise

    if run_id is not None:
        log_trace(
            db,
            run_id,
            agent_name="discovery",
            tool_or_action="plan_queries",
            duration_ms=plan_duration_ms,
            outcome="success",
            tokens_used=usage.total_tokens,
            payload_summary={
                "model_calls": 1,
                "used_mock_fallback": usage.used_mock_fallback,
                "cached": usage.cached,
            },
        )

    candidate_pool_limit = min(settings.MAX_CANDIDATE_POOL, 10)
    searches_used = 0
    overall_used_mock_fallback = usage.used_mock_fallback
    total_candidates_evaluated = 0

    # Primary search
    primary_q = plan.primary_query.strip() or user_query.strip()
    search1 = search_jobs(
        db,
        query=primary_q,
        hard=hard,
        limit=candidate_pool_limit,
        run_id=run_id,
    )
    searches_used += 1
    total_candidates_evaluated += search1.get("candidates_evaluated", 0)
    overall_used_mock_fallback = (
        overall_used_mock_fallback or search1.get("used_mock_fallback", False)
    )

    eligible_candidates = list(search1.get("eligible_candidates", []))
    rejected_counterfactuals = list(search1.get("rejected_counterfactuals", []))

    # Fallback search if < 5 eligible candidates and searches_used < 2
    fallback_q = plan.fallback_broad_query.strip()
    max_searches = min(2, settings.MAX_DISCOVERY_SEARCHES)
    if len(eligible_candidates) < 5 and searches_used < max_searches and fallback_q:
        search2 = search_jobs(
            db,
            query=fallback_q,
            hard=hard,  # Locked hard constraints: never relaxed
            limit=candidate_pool_limit,
            run_id=run_id,
        )
        searches_used += 1
        total_candidates_evaluated += search2.get("candidates_evaluated", 0)
        overall_used_mock_fallback = (
            overall_used_mock_fallback or search2.get("used_mock_fallback", False)
        )

        seen_job_ids = {c["job_id"] for c in eligible_candidates}
        for candidate in search2.get("eligible_candidates", []):
            if candidate["job_id"] not in seen_job_ids:
                seen_job_ids.add(candidate["job_id"])
                eligible_candidates.append(candidate)

        seen_rejected_ids = {r["job_id"] for r in rejected_counterfactuals}
        for counterfactual in search2.get("rejected_counterfactuals", []):
            if counterfactual["job_id"] not in seen_rejected_ids:
                seen_rejected_ids.add(counterfactual["job_id"])
                rejected_counterfactuals.append(counterfactual)

    eligible_candidates = eligible_candidates[:candidate_pool_limit]
    candidate_job_ids = [c["job_id"] for c in eligible_candidates]
    pure_counterfactuals = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in rejected_counterfactuals[:5]
    ]

    if run_id is not None:
        summary_payload = {
            "result_count": len(candidate_job_ids),
            "rejected_count": len(pure_counterfactuals),
            "candidates_evaluated": total_candidates_evaluated,
            "used_mock_fallback": overall_used_mock_fallback,
        }
        if all(re.fullmatch(r"job_[0-9]{3,}", jid) for jid in candidate_job_ids):
            summary_payload["job_ids"] = candidate_job_ids

        log_trace(
            db,
            run_id,
            agent_name="discovery",
            tool_or_action="select_candidates",
            duration_ms=(perf_counter() - plan_start) * 1000,
            outcome="success",
            payload_summary=summary_payload,
        )

    return {
        "candidate_job_ids": candidate_job_ids,
        "rejected_counterfactuals": pure_counterfactuals,
        "discovery_rationale": plan.rationale,
        "discovery_query_plan": plan.model_dump(mode="json"),
        "used_mock_fallback": overall_used_mock_fallback,
        "searches_used": searches_used,
    }


def discovery_node(
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    db: Session | None = None,
    llm_service: GeminiService | None = None,
) -> dict[str, Any]:
    """LangGraph node interface for Discovery Agent."""
    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary")

    resolved_db = db
    if resolved_db is None and config is not None:
        resolved_db = config.get("configurable", {}).get("db")
    if resolved_db is None:
        resolved_db = state.get("_db") or state.get("db")
    if resolved_db is None:
        raise ValueError("A database Session is required for discovery")

    user_query = state.get("user_query") or state.get("query")
    if user_query is None:
        soft_raw = state.get("soft_preferences") or state.get("soft")
        if isinstance(soft_raw, dict) and soft_raw.get("target_roles"):
            user_query = ", ".join(soft_raw["target_roles"])
        elif isinstance(soft_raw, SoftPreferences) and soft_raw.target_roles:
            user_query = ", ".join(soft_raw.target_roles)
        else:
            raise ValueError("user_query or query must be provided in state")

    run_id = state.get("run_id") or state.get("search_run_id")

    hard_raw = state.get("hard_constraints") or state.get("hard")
    soft_raw = state.get("soft_preferences") or state.get("soft")

    if hard_raw is None:
        profile_id = state.get("profile_id")
        if not profile_id:
            raise ValueError(
                "Either hard_constraints or profile_id must be provided in state"
            )
        pref_dict = get_preferences(
            resolved_db,
            profile_id,
            version=state.get("preference_version"),
            run_id=run_id,
        )
        hard_raw = pref_dict.get("hard_constraints")
        if soft_raw is None:
            soft_raw = pref_dict.get("soft_preferences")

    return run_discovery(
        db=resolved_db,
        user_query=user_query,
        hard_constraints=hard_raw,
        soft_preferences=soft_raw,
        run_id=run_id,
        llm_service=llm_service,
    )
