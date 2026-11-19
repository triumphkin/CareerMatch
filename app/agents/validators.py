"""Pure deterministic integrity checks, shared by every evaluation variant.

The return value is an ordered list of issues: [] means integrity checks passed.
Inputs are never mutated. The caller must block/downgrade an assessment with issues.
This checks references and explicit provenance claims, not general textual
entailment; the batched semantic audit belongs to Step 2.4.
"""

import re
import unicodedata
from collections.abc import Mapping

from pydantic import ValidationError

from app.schemas.contracts import (
    ConstraintCheckItem,
    DecisionType,
    EvidenceRecord,
    JobMatchAssessment,
    TriState,
    VerificationStatus,
)

_GATE_FIELDS = frozenset(
    {
        "listing_status",
        "work_mode_and_location",
        "student_eligibility",
        "experience_limit",
        "internal_consistency",
    }
)
_SOURCE = r"(?:github|projects?|repos?|repositories|repository|artifacts?|portfolio|codebase|demo)"
_PROOF = r"(?:proof|evidence|backed|proven|proves?|demonstrat\w*|verif\w*|confirm\w*|validat\w*|showcas\w*|built|implemented)"


def _claims_project_proof(explanation: str) -> bool:
    text = unicodedata.normalize("NFKC", explanation).casefold().replace("’", "'")
    # Remove explicit disclaimers, not entire sentences: a later positive claim
    # ("no GitHub proof, but project-verified") must still be caught.
    text = re.sub(
        rf"\b(?:no|without|lacks?)\s+(?:(?:independent|verified|supporting|any|actual|external)\s+)*"
        rf"{_SOURCE}(?:[ -]+(?:project|repository|artifact))?[ -]+(?:proof|evidence|verification|support)\b",
        "",
        text,
    )
    text = re.sub(
        rf"\b(?:not|never|isn't|wasn't)\s+(?:independently\s+)?(?:{_SOURCE}[ -]+)?{_PROOF}\b(?:\s+(?:by|through|with|in|on)\s+{_SOURCE})?",
        "",
        text,
    )
    text = re.sub(
        rf"\b{_SOURCE}[ -]+(?:proof|evidence|verification)\s+(?:(?:is|was)\s+)?"
        r"(?:not\s+(?:available|provided|established)|unavailable|missing|absent)\b",
        "",
        text,
    )
    for clause in re.split(r"[.!?;\n]|\b(?:but|however|yet)\b", text):
        if re.search(rf"\b{_SOURCE}\b", clause) and re.search(rf"\b{_PROOF}\b", clause):
            return True
    return False


def _validated(model, value):
    # Revalidate even already-instantiated models: model_copy(update=...) bypasses
    # Pydantic validation and must not be a route around this gate.
    if isinstance(value, model):
        value = {field: getattr(value, field) for field in model.model_fields}
    return model.model_validate(value)


def validate_match_assessment_integrity(
    assessment: JobMatchAssessment | dict,
    valid_evidence_map: Mapping[str, EvidenceRecord | dict],
    job_excerpts_map: Mapping[str, str],
    deterministic_constraints: list[ConstraintCheckItem | dict],
) -> list[str]:
    """Validate against one profile/version's evidence and ONE job's excerpt map.

    deterministic_constraints is the five-item list from the Phase 1 gate (or
    check_constraints(...)[i]['checks']), never the model's constraint report.
    Missing/malformed/duplicate gates fail closed. Known fail/unknown gates block
    accept independently of any optimistic checks supplied by the model.
    """
    try:
        item = _validated(JobMatchAssessment, assessment)
        # Nested model instances also need revalidation after unvalidated copies.
        item = JobMatchAssessment.model_validate(
            item.model_dump(mode="python", warnings=False)
        )
    except (ValidationError, TypeError, ValueError):
        return ["assessment: invalid schema"]
    issues = []
    if not isinstance(valid_evidence_map, Mapping):
        return ["valid_evidence_map: expected an evidence-ID mapping"]
    if not isinstance(job_excerpts_map, Mapping):
        return ["job_excerpts_map: expected this job's excerpt-ID mapping"]

    def check_excerpts(ids: list[str], path: str):
        for excerpt_id in dict.fromkeys(ids):
            if (
                not excerpt_id.startswith(f"{item.job_id}_ex_")
                or excerpt_id not in job_excerpts_map
                or not isinstance(job_excerpts_map[excerpt_id], str)
                or not job_excerpts_map[excerpt_id].strip()
            ):
                # Do not echo an untrusted ID; it could actually contain a prompt
                # or résumé excerpt. The path identifies the item to correct.
                issues.append(f"{path}: unknown, wrong-job, or invalid excerpt ID")

    cited_profiles = set()
    for index, match in enumerate(item.supported_matches):
        path = f"supported_matches[{index}]"
        if not match.candidate_evidence_ids:
            issues.append(f"{path}: supported claim requires candidate evidence")
        if not match.job_excerpt_ids:
            issues.append(f"{path}: supported claim requires a job excerpt")
        check_excerpts(match.job_excerpt_ids, path)
        records = []
        for evidence_id in dict.fromkeys(match.candidate_evidence_ids):
            if evidence_id not in valid_evidence_map:
                issues.append(f"{path}: unknown candidate evidence ID")
                continue
            try:
                evidence = _validated(EvidenceRecord, valid_evidence_map[evidence_id])
            except (ValidationError, TypeError, ValueError):
                issues.append(f"{path}: invalid evidence record")
                continue
            if evidence.evidence_id != evidence_id:
                issues.append(f"{path}: evidence map key does not match record ID")
                continue
            cited_profiles.add(evidence.profile_id)
            records.append(evidence)
        self_only = bool(records) and all(
            record.verification_status == VerificationStatus.SELF_REPORTED
            for record in records
        )
        if self_only and not match.is_self_reported_only:
            issues.append(f"{path}: is_self_reported_only must be True")
        if records and not self_only and match.is_self_reported_only:
            issues.append(f"{path}: is_self_reported_only contradicts cited evidence")
        if (
            records
            and not any(
                record.verification_status == VerificationStatus.PROJECT_BACKED
                for record in records
            )
            and _claims_project_proof(match.explanation)
        ):
            issues.append(
                f"{path}: explanation claims GitHub/project proof without project-backed evidence"
            )
    if len(cited_profiles) > 1:
        issues.append("assessment: cited evidence spans multiple candidate profiles")

    trusted = {}
    if not isinstance(deterministic_constraints, list):
        issues.append("deterministic_constraints: expected a complete gate list")
    else:
        for index, check in enumerate(deterministic_constraints):
            try:
                check = _validated(ConstraintCheckItem, check)
            except (ValidationError, TypeError, ValueError):
                issues.append(f"deterministic_constraints[{index}]: invalid gate")
                continue
            if check.field in trusted:
                issues.append("deterministic_constraints: duplicate gate field")
            # Check every row before storing so duplicate-pass cannot hide a fail.
            if item.decision == DecisionType.ACCEPT and check.result != TriState.PASS:
                issues.append(
                    f"decision: accept is forbidden by deterministic {check.result.value}"
                )
            trusted[check.field] = check.result
            check_excerpts(
                check.source_excerpt_ids, f"deterministic_constraints[{index}]"
            )
    if set(trusted) != _GATE_FIELDS:
        issues.append("deterministic_constraints: incomplete or unexpected gate fields")

    reported = {}
    for index, check in enumerate(item.constraint_checks):
        check_excerpts(check.source_excerpt_ids, f"constraint_checks[{index}]")
        if check.field in reported:
            issues.append("constraint_checks: duplicate gate field")
        reported[check.field] = check.result
        if check.field in trusted and check.result != trusted[check.field]:
            issues.append(
                f"constraint_checks[{index}]: contradicts deterministic result"
            )
    if set(reported) != set(trusted):
        issues.append("constraint_checks: fields do not match deterministic report")
    return list(dict.fromkeys(issues))
