"""Shared deterministic gates for every evaluation variant; no model calls.

An empty allowlist allows no choices. Hybrid jobs require an allowed office city;
remote-region restrictions apply to remote jobs. Explicit keyword exclusions live
in listing_status, keeping the blueprint's five-gate contract intact.
"""

import math
import re
import sys
import unicodedata

from app.db.models import JobMetadataORM, JobORM
from app.schemas.contracts import ConstraintCheckItem, HardConstraints, TriState
from app.services.ingestion import normalize_job_constraints, slice_into_excerpts

_FIELDS = (
    "norm_work_mode",
    "norm_remote_region",
    "norm_cities",
    "norm_student_eligible",
    "norm_degree_completed_required",
    "norm_min_yoe",
)


def _canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _city(value: str) -> str:
    result = _canonical(value)
    return "bengaluru" if result == "bangalore" else result


def validate_hard_constraints(hard: HardConstraints) -> HardConstraints:
    if not isinstance(hard, HardConstraints):
        raise TypeError("hard must be a HardConstraints instance")
    validated = HardConstraints.model_validate(hard.model_dump())
    for value in validated.allowed_cities + validated.excluded_keywords:
        if not value.strip() or "\x00" in value:
            raise ValueError(
                "Cities and excluded keywords must be nonempty and NUL-free"
            )
    return validated


def _overall(states) -> TriState:
    states = list(states)
    if TriState.FAIL in states:
        return TriState.FAIL
    if TriState.UNKNOWN in states:
        return TriState.UNKNOWN
    return TriState.PASS


def _source_excerpts(job: JobORM) -> tuple[dict[str, str], bool]:
    """Validate against Step 1.6's slicer without parsing untrusted ID suffixes."""
    if not isinstance(job.excerpts, dict) or not isinstance(job.raw_description, str):
        return {}, False
    try:
        expected = slice_into_excerpts(job.job_id, job.raw_description)
    except (ValueError, TypeError):
        return {}, False
    valid = {
        key: value for key, value in expected.items() if job.excerpts.get(key) == value
    }
    return valid, bool(expected) and job.excerpts == expected


def _citation_map(job: JobORM) -> tuple[dict[str, list[str]], bool]:
    """Require citations and normalized values to agree with deterministic ingestion.

    This checks support recognized by the existing normalizer, not arbitrary
    natural-language entailment. Mismatches make integrity unknown; the blueprint's
    fail-over-unknown precedence remains unchanged.
    """
    empty = {field: [] for field in _FIELDS}
    if not isinstance(job.constraint_excerpts, dict):
        return empty, False
    valid, complete = _source_excerpts(job)
    try:
        excerpts = slice_into_excerpts(job.job_id, job.raw_description)
        derived = normalize_job_constraints(
            {"job_id": job.job_id, "raw_description": job.raw_description}, excerpts
        )
    except (ValueError, TypeError):
        return empty, False
    proof = job.constraint_excerpts
    expected_proof = derived["constraint_excerpts"]
    result = {}
    for field in _FIELDS:
        ids = proof.get(field, [])
        expected_ids = expected_proof[field]
        supported = set(expected_ids) & valid.keys()
        if not isinstance(ids, list):
            ids = []
            complete = False
        if any(not isinstance(key, str) or key not in supported for key in ids):
            complete = False
        supplied = {key for key in ids if isinstance(key, str) and key in supported}
        if supplied != set(expected_ids):
            complete = False
        value = getattr(job, field)
        agrees = value == derived[field]
        if field in ("norm_student_eligible", "norm_degree_completed_required"):
            agrees = value is derived[field]
        if field == "norm_min_yoe" and isinstance(value, bool):
            agrees = False
        if not agrees:
            complete = False
            supplied = set()
        result[field] = [key for key in expected_ids if key in supplied]
    if (
        proof.get("has_conflict") is not derived["has_conflict"]
        or proof.get("conflict_fields") != expected_proof["conflict_fields"]
    ):
        complete = False
    return result, complete


def evaluate_job_constraints(
    job: JobORM, meta: JobMetadataORM, hard: HardConstraints
) -> tuple[TriState, list[ConstraintCheckItem]]:
    if not isinstance(job, JobORM) or not isinstance(meta, JobMetadataORM):
        raise TypeError("job and meta must be ORM records")
    if not isinstance(job.job_id, str) or not job.job_id or meta.job_id != job.job_id:
        raise ValueError("Job and metadata identifiers must match")
    hard = validate_hard_constraints(hard)
    proof, citations_complete = _citation_map(job)

    def citations(*fields):
        return sorted(
            {key for field in fields for key in proof.get(field, [])},
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )

    def check(field, state, reason, ids=()):
        return ConstraintCheckItem(
            field=field, result=state, reason=reason, source_excerpt_ids=list(ids)
        )

    checks = []
    status = TriState.PASS
    reasons = []
    if meta.source_status == "closed":
        status = TriState.FAIL
        reasons.append("Listing is closed.")
    elif meta.source_status != "active":
        status = TriState.UNKNOWN
        reasons.append("Listing status is unknown.")
    if meta.is_duplicate is True:
        status = TriState.FAIL
        reasons.append("Listing is a duplicate.")
    elif meta.is_duplicate is not False:
        status = _overall((status, TriState.UNKNOWN))
        reasons.append("Duplicate status is unknown.")
    keyword_ids = set()
    searchable = [
        _canonical(value)
        for value in (job.title, job.company, job.raw_description)
        if isinstance(value, str)
    ]
    valid_excerpts, _ = _source_excerpts(job)
    for keyword in hard.excluded_keywords:
        # '+' and '#' belong to technology names: C, C++, and C# are distinct.
        pattern = r"(?<![\w+#])" + re.escape(_canonical(keyword)) + r"(?![\w+#])"
        if any(re.search(pattern, value) for value in searchable):
            status = TriState.FAIL
            reasons.append(f"Listing contains excluded keyword: {keyword}.")
            for key, value in valid_excerpts.items():
                if re.search(pattern, _canonical(value)):
                    keyword_ids.add(key)
    checks.append(
        check(
            "listing_status",
            status,
            " ".join(reasons) or "Listing is active and non-duplicate.",
            [key for key in valid_excerpts if key in keyword_ids],
        )
    )

    mode = job.norm_work_mode
    allowed_modes = {value.value for value in hard.allowed_work_modes}
    location_ids = citations("norm_work_mode")
    if not allowed_modes:
        location_state, reason = (
            TriState.FAIL,
            "No work modes are allowed by the current preferences.",
        )
    elif mode not in ("remote", "hybrid", "onsite"):
        location_state, reason = TriState.UNKNOWN, "Work mode is not established."
    elif mode not in allowed_modes:
        location_state, reason = TriState.FAIL, f"Work mode {mode} is not allowed."
    elif mode == "remote":
        location_ids = citations("norm_work_mode", "norm_remote_region")
        if not hard.require_india_remote_if_remote:
            location_state, reason = (
                TriState.PASS,
                "Remote work is allowed; India eligibility is not required.",
            )
        elif job.norm_remote_region in ("india_eligible", "global_remote"):
            location_state, reason = (
                TriState.PASS,
                "Remote role explicitly permits India-based applicants.",
            )
        elif job.norm_remote_region == "non_india_restricted":
            location_state, reason = (
                TriState.FAIL,
                "Remote role excludes India-based applicants.",
            )
        else:
            location_state, reason = (
                TriState.UNKNOWN,
                "Remote-region eligibility is not established.",
            )
    else:
        location_ids = citations("norm_work_mode", "norm_cities")
        allowed_cities = {_city(value) for value in hard.allowed_cities}
        cities = job.norm_cities
        if not allowed_cities:
            location_state, reason = (
                TriState.FAIL,
                f"No office cities are allowed for this {mode} role.",
            )
        elif (
            not isinstance(cities, list)
            or not cities
            or any(not isinstance(value, str) or not value.strip() for value in cities)
        ):
            location_state, reason = (
                TriState.UNKNOWN,
                "Required office location is not established.",
            )
        elif allowed_cities & {_city(value) for value in cities}:
            location_state, reason = (
                TriState.PASS,
                "An explicitly listed office city is allowed.",
            )
        else:
            location_state, reason = (
                TriState.FAIL,
                "Required office cities are outside the allowed city list.",
            )
    checks.append(check("work_mode_and_location", location_state, reason, location_ids))

    student_states, student_reasons, student_fields = [], [], []
    if hard.require_student_eligible:
        student_fields.append("norm_student_eligible")
        if job.norm_student_eligible is True:
            student_states.append(TriState.PASS)
            student_reasons.append("Current students are eligible.")
        elif job.norm_student_eligible is False:
            student_states.append(TriState.FAIL)
            student_reasons.append("Current students are ineligible.")
        else:
            student_states.append(TriState.UNKNOWN)
            student_reasons.append("Student eligibility is not established.")
    if hard.exclude_completed_degree_required:
        student_fields.append("norm_degree_completed_required")
        if job.norm_degree_completed_required is True:
            student_states.append(TriState.FAIL)
            student_reasons.append("A completed degree is required.")
        elif job.norm_degree_completed_required is False:
            student_states.append(TriState.PASS)
            student_reasons.append("A completed degree is explicitly not required.")
        else:
            student_states.append(TriState.UNKNOWN)
            student_reasons.append("Completed-degree policy is not established.")
    checks.append(
        check(
            "student_eligibility",
            _overall(student_states),
            " ".join(student_reasons)
            or "Student and completed-degree restrictions are disabled.",
            citations(*student_fields),
        )
    )

    years = job.norm_min_yoe
    if (
        isinstance(years, bool)
        or not isinstance(years, (int, float))
        or not 0 <= years <= sys.float_info.max
        or not math.isfinite(years)
    ):
        experience, reason = (
            TriState.UNKNOWN,
            "Minimum required experience is not established.",
        )
    elif years > hard.max_required_yoe:
        experience, reason = (
            TriState.FAIL,
            f"Required experience {years:g} years exceeds the {hard.max_required_yoe:g}-year limit.",
        )
    else:
        experience, reason = (
            TriState.PASS,
            f"Required experience {years:g} years is within the {hard.max_required_yoe:g}-year limit.",
        )
    checks.append(
        check("experience_limit", experience, reason, citations("norm_min_yoe"))
    )

    source = (
        job.constraint_excerpts if isinstance(job.constraint_excerpts, dict) else {}
    )
    conflicts = source.get("conflict_fields")
    flag = source.get("has_conflict")
    if flag is True:
        fields = (
            [
                field
                for field in conflicts
                if isinstance(field, str) and field in _FIELDS
            ]
            if isinstance(conflicts, list)
            else list(_FIELDS)
        )
        consistent, reason = (
            TriState.FAIL,
            "Listing contains conflicting internal requirements.",
        )
        conflict_ids = citations(*fields)
    elif flag is not False or conflicts != [] or not citations_complete:
        consistent, reason = (
            TriState.UNKNOWN,
            "Consistency or exact source-citation integrity is not established.",
        )
        conflict_ids = []
    else:
        consistent, reason = (
            TriState.PASS,
            "Normalization found no internal conflicts; source citations are valid.",
        )
        conflict_ids = []
    checks.append(check("internal_consistency", consistent, reason, conflict_ids))
    return _overall(item.result for item in checks), checks
