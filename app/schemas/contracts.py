"""Additive-only contracts shared by services, agents, and API endpoints."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSourceType(str, Enum):
    RESUME_STATEMENT = "resume_statement"
    USER_STATEMENT = "user_statement"
    ARTIFACT_EXCERPT = "artifact_excerpt"
    NOT_ESTABLISHED = "not_established"


class VerificationStatus(str, Enum):
    PROJECT_BACKED = "project_backed"
    RESUME_CLAIMED = "resume_claimed"
    SELF_REPORTED = "self_reported"


class TriState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class DecisionType(str, Enum):
    ACCEPT = "accept"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"
    REVISE = "revise"


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class RemoteRegion(str, Enum):
    INDIA_ELIGIBLE = "india_eligible"
    GLOBAL_REMOTE = "global_remote"
    NON_INDIA_RESTRICTED = "non_india_restricted"
    UNKNOWN = "unknown"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    profile_id: str
    skill: str
    source_type: EvidenceSourceType
    source_excerpt: str
    source_id: str
    verification_status: VerificationStatus


class HardConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_work_modes: list[WorkMode]
    allowed_cities: list[str]
    require_india_remote_if_remote: bool = True
    require_student_eligible: bool = True
    exclude_completed_degree_required: bool = True
    max_required_yoe: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    excluded_keywords: list[str] = Field(default_factory=list)


class SoftPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_roles: list[str]
    preferred_skills: list[str]
    prefer_startup: bool = False


class PreferenceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    version: int = Field(ge=1)
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    updated_at: str


class ConstraintCheckItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    result: TriState
    reason: str
    source_excerpt_ids: list[str]


class SupportedMatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    candidate_evidence_ids: list[str]
    job_excerpt_ids: list[str]
    explanation: str
    is_self_reported_only: bool = False


class JobMatchAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str
    company: str
    source_url: str
    captured_at: str
    decision: DecisionType
    supported_matches: list[SupportedMatchItem]
    gaps: list[str]
    unknowns: list[str]
    constraint_checks: list[ConstraintCheckItem]
    internal_rank_score: float = Field(default=0.0, allow_inf_nan=False)


class MatchingBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[JobMatchAssessment]


class RejectedJobCounterfactual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str
    company: str
    similarity_score: float = Field(allow_inf_nan=False)
    failed_constraints: list[ConstraintCheckItem]
    rejection_summary: str


class LLMUsageMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cached: bool = False
    used_mock_fallback: bool = False


class DiscoveryQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_query: str
    fallback_broad_query: str
    rationale: str


class VerificationFeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    issues: list[str]
    required_action: str


class VerificationVerdictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    verdict: DecisionType
    issues: list[str]
    feedback_for_matching: str


class VerificationVerdictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[VerificationVerdictItem]
