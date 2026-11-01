"""Anomaly detectors for Phase 4.

Each detector receives the injected fixture list (or a raw data slice) and
returns DetectionResult records.  Running detectors on clean baseline data
must return zero results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from learnpath_ai.generate.clock import format_timestamp
from learnpath_ai.quality.anomalies import (
    ISSUE_TYPES,
    AnomalyFixture,
    AnomalyFixtures,
    _PHANTOM_SKILL_ID,
)

UTC = timezone.utc


class QualityError(ValueError):
    """Raised when a fixture or input is structurally invalid."""


@dataclass
class DetectionResult:
    """Structured output of a single anomaly detection."""

    issue_id: str
    anomaly_type: str
    severity: str
    # DB-schema fields
    source_table: str
    source_pk: str
    # Report-level fields (not DB columns)
    entity_type: str
    entity_id: str
    detected_at: str
    reason_code: str
    description: str
    detection_status: str = "detected"
    affected_record_reference: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def detect_prereq_violation(
    fixtures: list[AnomalyFixture],
) -> list[DetectionResult]:
    """Detect REC_PREREQ_VIOLATION fixtures.

    Only fixtures of type REC_PREREQ_VIOLATION with a non-empty
    'missing_prerequisite_skill_id' in their details are flagged.
    """
    results: list[DetectionResult] = []
    for fix in fixtures:
        if fix.issue_type != "REC_PREREQ_VIOLATION":
            continue
        d = fix.details
        missing = d.get("missing_prerequisite_skill_id", "")
        learner_id = d.get("learner_id", "")
        content_id = d.get("content_id", "")
        if not missing:
            continue
        results.append(DetectionResult(
            issue_id=fix.issue_id,
            anomaly_type="REC_PREREQ_VIOLATION",
            severity=fix.severity,
            source_table=fix.source_table,
            source_pk=fix.source_pk,
            entity_type="recommendation",
            entity_id=fix.source_pk,
            detected_at=fix.detected_at,
            reason_code="PREREQ_NOT_SATISFIED",
            description=(
                f"Recommendation {fix.source_pk} for learner {learner_id} on "
                f"content {content_id} violates prerequisite {missing}."
            ),
            details=d,
        ))
    return results


def detect_timestamp_order(
    fixtures: list[AnomalyFixture],
) -> list[DetectionResult]:
    """Detect ATTEMPT_TIMESTAMP_ORDER fixtures.

    Flags any fixture whose injected_record has submitted_at < started_at,
    or scored_at < submitted_at.
    """
    results: list[DetectionResult] = []
    for fix in fixtures:
        if fix.issue_type != "ATTEMPT_TIMESTAMP_ORDER":
            continue

        rec = fix.injected_record
        started_raw = rec.get("started_at", "")
        submitted_raw = rec.get("submitted_at", "")

        detected = False
        violated_rule = ""
        if started_raw and submitted_raw:
            try:
                started_dt = _parse_ts(started_raw)
                submitted_dt = _parse_ts(submitted_raw)
                if submitted_dt < started_dt:
                    detected = True
                    violated_rule = "submitted_at < started_at"
            except Exception:
                detected = True
                violated_rule = "unparseable timestamps"

        scored_raw = rec.get("scored_at", "")
        if submitted_raw and scored_raw:
            try:
                submitted_dt = _parse_ts(submitted_raw)
                scored_dt = _parse_ts(scored_raw)
                if scored_dt < submitted_dt:
                    detected = True
                    violated_rule = "scored_at < submitted_at"
            except Exception:
                detected = True
                violated_rule = "unparseable scored_at"

        # Also check details if injected_record is empty (fixture-only mode)
        if not detected:
            d = fix.details
            s_raw = d.get("started_at", "")
            sub_raw = d.get("submitted_at", "")
            if s_raw and sub_raw:
                try:
                    if _parse_ts(sub_raw) < _parse_ts(s_raw):
                        detected = True
                        violated_rule = d.get("violated_rule", "submitted_at < started_at")
                except Exception:
                    detected = True
                    violated_rule = "unparseable"

        if detected:
            attempt_id = rec.get("attempt_id") or fix.source_pk
            results.append(DetectionResult(
                issue_id=fix.issue_id,
                anomaly_type="ATTEMPT_TIMESTAMP_ORDER",
                severity=fix.severity,
                source_table=fix.source_table,
                source_pk=fix.source_pk,
                entity_type="attempt",
                entity_id=attempt_id,
                detected_at=fix.detected_at,
                reason_code="TIMESTAMP_ORDER_VIOLATION",
                description=(
                    f"Attempt {attempt_id}: {violated_rule} "
                    f"(started={started_raw}, submitted={submitted_raw})."
                ),
                details=fix.details,
            ))
    return results


def detect_content_missing_skill(
    fixtures: list[AnomalyFixture],
    known_skill_ids: set[str] | None = None,
) -> list[DetectionResult]:
    """Detect CONTENT_MISSING_SKILL fixtures.

    Flags any fixture where details.missing_skill_id is not in known_skill_ids,
    or where the injected_record.skill_id equals the phantom SKL-999999.
    """
    results: list[DetectionResult] = []
    for fix in fixtures:
        if fix.issue_type != "CONTENT_MISSING_SKILL":
            continue

        d = fix.details
        missing_id = d.get("missing_skill_id", "")
        content_id = d.get("content_id", fix.source_pk)

        is_missing = (missing_id == _PHANTOM_SKILL_ID) or (
            known_skill_ids is not None and missing_id not in known_skill_ids
        )
        if is_missing:
            results.append(DetectionResult(
                issue_id=fix.issue_id,
                anomaly_type="CONTENT_MISSING_SKILL",
                severity=fix.severity,
                source_table=fix.source_table,
                source_pk=fix.source_pk,
                entity_type="content_skill",
                entity_id=fix.source_pk,
                detected_at=fix.detected_at,
                reason_code="BROKEN_FOREIGN_KEY",
                description=(
                    f"content_skill mapping for {content_id} references "
                    f"skill {missing_id} which does not exist."
                ),
                details=d,
            ))
    return results


def detect_mastery_future_attempt(
    fixtures: list[AnomalyFixture],
) -> list[DetectionResult]:
    """Detect MASTERY_FUTURE_ATTEMPT fixtures.

    Flags any fixture where attempt_started_at > as_of_timestamp.
    """
    results: list[DetectionResult] = []
    for fix in fixtures:
        if fix.issue_type != "MASTERY_FUTURE_ATTEMPT":
            continue

        d = fix.details
        as_of_raw = d.get("as_of_timestamp", "")
        started_raw = d.get("attempt_started_at", "")

        detected = False
        if as_of_raw and started_raw:
            try:
                as_of_dt = _parse_ts(as_of_raw)
                started_dt = _parse_ts(started_raw)
                detected = started_dt > as_of_dt
            except Exception:
                detected = True

        if detected:
            attempt_id = d.get("attempt_id", fix.source_pk)
            results.append(DetectionResult(
                issue_id=fix.issue_id,
                anomaly_type="MASTERY_FUTURE_ATTEMPT",
                severity=fix.severity,
                source_table=fix.source_table,
                source_pk=fix.source_pk,
                entity_type="attempt",
                entity_id=attempt_id,
                detected_at=fix.detected_at,
                reason_code="TEMPORAL_LEAKAGE",
                description=(
                    f"Mastery as_of {as_of_raw}: attempt {attempt_id} "
                    f"started at {started_raw} is AFTER the cutoff — must be excluded."
                ),
                details=d,
            ))
    return results


def detect_path_loop(
    fixtures: list[AnomalyFixture],
) -> list[DetectionResult]:
    """Detect PATH_LOOP_NO_PROGRESS fixtures.

    Flags any fixture where path_items contains a repeated content_id
    without score improvement between occurrences.
    """
    results: list[DetectionResult] = []
    for fix in fixtures:
        if fix.issue_type != "PATH_LOOP_NO_PROGRESS":
            continue

        d = fix.details
        path_items = d.get("path_items", [])
        learner_id = d.get("learner_id", "")
        path_id = d.get("path_id", fix.source_pk)

        # Find repeated content IDs
        seen: dict[str, list[float]] = {}
        loop_detected = False
        for item in path_items:
            cid = item.get("content_id", "")
            score = float(item.get("score", 0.0))
            if cid in seen:
                prev_scores = seen[cid]
                if score <= max(prev_scores):
                    loop_detected = True
                prev_scores.append(score)
            else:
                seen[cid] = [score]

        # Also check the details fields directly (positions / scores list)
        positions = d.get("positions", [])
        scores = d.get("scores", [])
        if len(positions) >= 2 and len(scores) >= 2:
            if scores[-1] <= scores[0]:
                loop_detected = True

        if loop_detected:
            results.append(DetectionResult(
                issue_id=fix.issue_id,
                anomaly_type="PATH_LOOP_NO_PROGRESS",
                severity=fix.severity,
                source_table=fix.source_table,
                source_pk=fix.source_pk,
                entity_type="learning_path",
                entity_id=path_id,
                detected_at=fix.detected_at,
                reason_code="LOOP_WITHOUT_PROGRESS",
                description=(
                    f"Path {path_id} for learner {learner_id} has repeated "
                    "content without score improvement."
                ),
                details=d,
            ))
    return results


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------


def detect_all_anomalies(
    fixtures: AnomalyFixtures,
    master_tables: Any | None = None,
) -> list[DetectionResult]:
    """Run all five detectors and return merged results.

    Passing master_tables enables foreign-key checks (e.g. known skill IDs).
    """
    known_skill_ids: set[str] | None = None
    if master_tables is not None:
        skills_list: list[dict] = []
        if hasattr(master_tables, "skills"):
            skills_list = list(master_tables.skills)
        elif isinstance(master_tables, dict):
            skills_list = master_tables.get("skills", [])
        known_skill_ids = {s["skill_id"] for s in skills_list if "skill_id" in s}

    results: list[DetectionResult] = []
    results.extend(detect_prereq_violation(fixtures.rec_prereq_violations))
    results.extend(detect_timestamp_order(fixtures.attempt_timestamp_orders))
    results.extend(detect_content_missing_skill(
        fixtures.content_missing_skills,
        known_skill_ids=known_skill_ids,
    ))
    results.extend(detect_mastery_future_attempt(fixtures.mastery_future_attempts))
    results.extend(detect_path_loop(fixtures.path_loop_no_progress))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime:
    """Parse a timestamp string to an aware UTC datetime."""
    text = value.strip()
    # Handle the "2025-01-01 08:00:00.000000" format from format_timestamp()
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    if not text.endswith("Z") and "+" not in text:
        text += "+00:00"
    return datetime.fromisoformat(text)

