"""Anomaly fixture generation for Phase 4.

Generates deterministic synthetic bad records for all five anomaly types.
Each fixture carries enough metadata to be written to a quarantine CSV and
matched against the data_quality_issue DB schema.

Anomaly types (ENUM from schema.sql):
  A  REC_PREREQ_VIOLATION
  B  ATTEMPT_TIMESTAMP_ORDER
  C  CONTENT_MISSING_SKILL
  D  MASTERY_FUTURE_ATTEMPT
  E  PATH_LOOP_NO_PROGRESS
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from learnpath_ai.generate.clock import format_timestamp, parse_iso_utc
from learnpath_ai.ids import format_id

UTC = timezone.utc

# Guaranteed non-existent skill ID for Anomaly C
_PHANTOM_SKILL_ID = "SKL-999999"

# Issue type constants (must match schema ENUM exactly)
ISSUE_TYPES = (
    "REC_PREREQ_VIOLATION",
    "ATTEMPT_TIMESTAMP_ORDER",
    "CONTENT_MISSING_SKILL",
    "MASTERY_FUTURE_ATTEMPT",
    "PATH_LOOP_NO_PROGRESS",
)


@dataclass
class AnomalyFixture:
    """Single injected anomaly record, schema-compatible with data_quality_issue."""

    issue_id: str
    issue_type: str
    severity: str
    source_table: str
    source_pk: str
    status: str
    expected_decision: str
    details: dict[str, Any]
    detected_at: str
    created_at: str
    # Additional fields used by detectors but not stored as DB columns
    injected_record: dict[str, Any] = field(default_factory=dict)
    affected_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.issue_type not in ISSUE_TYPES:
            raise ValueError(f"Unknown issue_type: {self.issue_type!r}")
        if self.severity not in ("low", "medium", "high", "critical"):
            raise ValueError(f"Unknown severity: {self.severity!r}")
        if self.status not in ("open", "quarantined", "accepted_exception", "resolved"):
            raise ValueError(f"Unknown status: {self.status!r}")
        if not self.issue_id.startswith("DQI-"):
            raise ValueError(f"issue_id must start with 'DQI-': {self.issue_id!r}")

    def as_db_row(self) -> dict[str, Any]:
        """Return a dict matching data_quality_issue columns."""
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "source_table": self.source_table,
            "source_pk": self.source_pk,
            "severity": self.severity,
            "status": self.status,
            "expected_decision": self.expected_decision,
            "details": json.dumps(self.details, separators=(",", ":")),
            "detected_at": self.detected_at,
            "created_at": self.created_at,
        }


@dataclass
class AnomalyFixtures:
    """Container for all five anomaly type fixture lists."""

    rec_prereq_violations: list[AnomalyFixture]
    attempt_timestamp_orders: list[AnomalyFixture]
    content_missing_skills: list[AnomalyFixture]
    mastery_future_attempts: list[AnomalyFixture]
    path_loop_no_progress: list[AnomalyFixture]

    def all(self) -> list[AnomalyFixture]:
        return (
            self.rec_prereq_violations
            + self.attempt_timestamp_orders
            + self.content_missing_skills
            + self.mastery_future_attempts
            + self.path_loop_no_progress
        )

    def by_type(self) -> dict[str, list[AnomalyFixture]]:
        return {
            "REC_PREREQ_VIOLATION": self.rec_prereq_violations,
            "ATTEMPT_TIMESTAMP_ORDER": self.attempt_timestamp_orders,
            "CONTENT_MISSING_SKILL": self.content_missing_skills,
            "MASTERY_FUTURE_ATTEMPT": self.mastery_future_attempts,
            "PATH_LOOP_NO_PROGRESS": self.path_loop_no_progress,
        }


def _now_utc_str() -> str:
    return format_timestamp(datetime.now(UTC))


def _make_id(seq: int) -> str:
    return format_id("DQI", seq)


def _coerce_master(master_tables: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept either a MasterTables dataclass or a plain dict."""
    if hasattr(master_tables, "__dict__") and hasattr(master_tables, "courses"):
        return {
            "courses": list(master_tables.courses),
            "learners": list(master_tables.learners),
            "skills": list(master_tables.skills),
            "skill_prerequisites": list(master_tables.skill_prerequisites),
            "content_items": list(master_tables.content_items),
            "content_skills": list(master_tables.content_skills),
        }
    return dict(master_tables)  # type: ignore[arg-type]


def _coerce_activity(activity_tables: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept either an ActivityTables dataclass or a plain dict."""
    if hasattr(activity_tables, "__dict__") and hasattr(activity_tables, "attempts"):
        return {
            "attempts": list(activity_tables.attempts),
            "assessment_results": list(activity_tables.assessment_results),
        }
    return dict(activity_tables)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-type generators
# ---------------------------------------------------------------------------


def _gen_rec_prereq_violations(
    master: dict[str, list[dict]],
    activity: dict[str, list[dict]],
    rng: np.random.Generator,
    n: int,
    seq_start: int,
    now_str: str,
) -> list[AnomalyFixture]:
    """Anomaly A: recommendation violates a prerequisite gate."""
    fixtures: list[AnomalyFixture] = []

    prereqs = master.get("skill_prerequisites", [])
    content_skills = master.get("content_skills", [])
    content_items = master.get("content_items", [])
    learners = master.get("learners", [])

    # Build lookup: skill_id -> list of prerequisite_skill_ids
    prereq_map: dict[str, list[str]] = {}
    for row in prereqs:
        skill = row.get("skill_id", "")
        pre = row.get("prerequisite_skill_id", "")
        if skill and pre:
            prereq_map.setdefault(skill, []).append(pre)

    # Build lookup: content_id -> primary skill (first "primary" role or any)
    content_primary_skill: dict[str, str] = {}
    for row in content_skills:
        cid = row.get("content_id", "")
        sid = row.get("skill_id", "")
        role = row.get("skill_role", "")
        if cid and sid and role == "primary" and cid not in content_primary_skill:
            content_primary_skill[cid] = sid

    # Find content items whose primary skill has at least one prerequisite
    gated_content = [
        c for c in content_items
        if content_primary_skill.get(c["content_id"]) in prereq_map
    ]

    if not gated_content or not learners:
        # Fallback: synthetic fixture with placeholder IDs
        for i in range(n):
            seq = seq_start + i
            learner_id = learners[i % len(learners)]["learner_id"] if learners else "LRN-000001"
            fixtures.append(_synthetic_prereq_fixture(
                seq, learner_id, "CNT-000001", "SKL-000001", "SKL-000002", now_str
            ))
        return fixtures

    rng.shuffle(gated_content)
    for i in range(n):
        seq = seq_start + i
        content = gated_content[i % len(gated_content)]
        content_id = content["content_id"]
        primary_skill = content_primary_skill[content_id]
        prereq_list = prereq_map[primary_skill]
        missing_prereq = prereq_list[0]
        learner = learners[int(rng.integers(0, len(learners)))]
        learner_id = learner["learner_id"]

        rec_id = f"REC-{seq:06d}"
        as_of = "2025-06-15 10:00:00.000000"
        recommended_at = "2025-06-15 10:05:00.000000"

        injected = {
            "recommendation_id": rec_id,
            "learner_id": learner_id,
            "content_id": content_id,
            "recommended_at": recommended_at,
            "as_of_timestamp": as_of,
        }

        fixtures.append(AnomalyFixture(
            issue_id=_make_id(seq),
            issue_type="REC_PREREQ_VIOLATION",
            severity="critical",
            source_table="recommendation",
            source_pk=rec_id,
            status="quarantined",
            expected_decision="reject",
            details={
                "learner_id": learner_id,
                "content_id": content_id,
                "primary_skill_id": primary_skill,
                "missing_prerequisite_skill_id": missing_prereq,
                "reason": (
                    f"Learner {learner_id} was recommended {content_id} "
                    f"but has not satisfied prerequisite {missing_prereq} "
                    f"required for primary skill {primary_skill}."
                ),
            },
            detected_at=now_str,
            created_at=now_str,
            injected_record=injected,
            affected_ids=[learner_id, content_id, primary_skill, missing_prereq],
        ))
    return fixtures


def _synthetic_prereq_fixture(seq, learner_id, content_id, skill_id, prereq_id, now_str):
    rec_id = f"REC-{seq:06d}"
    return AnomalyFixture(
        issue_id=_make_id(seq),
        issue_type="REC_PREREQ_VIOLATION",
        severity="critical",
        source_table="recommendation",
        source_pk=rec_id,
        status="quarantined",
        expected_decision="reject",
        details={
            "learner_id": learner_id,
            "content_id": content_id,
            "primary_skill_id": skill_id,
            "missing_prerequisite_skill_id": prereq_id,
            "reason": (
                f"Learner {learner_id} was recommended {content_id} "
                f"but has not satisfied prerequisite {prereq_id}."
            ),
        },
        detected_at=now_str,
        created_at=now_str,
        injected_record={
            "recommendation_id": rec_id,
            "learner_id": learner_id,
            "content_id": content_id,
        },
        affected_ids=[learner_id, content_id, skill_id, prereq_id],
    )


def _gen_attempt_timestamp_orders(
    activity: dict[str, list[dict]],
    rng: np.random.Generator,
    n: int,
    seq_start: int,
    now_str: str,
) -> list[AnomalyFixture]:
    """Anomaly B: submitted_at < started_at (impossible ordering)."""
    fixtures: list[AnomalyFixture] = []
    attempts = activity.get("attempts", [])

    scored_attempts = [a for a in attempts if a.get("status") == "scored"]
    if not scored_attempts:
        scored_attempts = attempts

    for i in range(n):
        seq = seq_start + i
        if scored_attempts:
            base = scored_attempts[int(rng.integers(0, len(scored_attempts)))]
            attempt_id = base["attempt_id"]
            learner_id = base.get("learner_id", "LRN-000001")
            content_id = base.get("content_id", "CNT-000001")
            original_started = base.get("started_at", "2025-01-01 08:00:00.000000")
            original_submitted = base.get("submitted_at") or "2025-01-01 09:00:00.000000"
        else:
            attempt_id = f"ATT-LRN-{seq:06d}-000001"
            learner_id = f"LRN-{seq:06d}"
            content_id = "CNT-000001"
            original_started = "2025-03-10 08:00:00.000000"
            original_submitted = "2025-03-10 09:00:00.000000"

        # Corrupt: set submitted_at 3600 seconds BEFORE started_at
        try:
            started_dt = datetime.fromisoformat(original_started.replace(" ", "T") + "+00:00")
        except Exception:
            started_dt = datetime(2025, 3, 10, 8, 0, 0, tzinfo=UTC)

        bad_submitted_dt = started_dt - timedelta(seconds=3600)
        bad_submitted = format_timestamp(bad_submitted_dt)

        injected = {
            "attempt_id": attempt_id,
            "learner_id": learner_id,
            "content_id": content_id,
            "started_at": original_started,
            "submitted_at": bad_submitted,  # intentionally BEFORE started_at
            "status": "submitted",
        }

        fixtures.append(AnomalyFixture(
            issue_id=_make_id(seq),
            issue_type="ATTEMPT_TIMESTAMP_ORDER",
            severity="high",
            source_table="attempt",
            source_pk=attempt_id,
            status="quarantined",
            expected_decision="reject",
            details={
                "attempt_id": attempt_id,
                "started_at": original_started,
                "submitted_at": bad_submitted,
                "violated_rule": "submitted_at must be >= started_at",
                "reason": (
                    f"Attempt {attempt_id}: submitted_at ({bad_submitted}) "
                    f"is earlier than started_at ({original_started})."
                ),
            },
            detected_at=now_str,
            created_at=now_str,
            injected_record=injected,
            affected_ids=[attempt_id, learner_id],
        ))
    return fixtures


def _gen_content_missing_skills(
    master: dict[str, list[dict]],
    rng: np.random.Generator,
    n: int,
    seq_start: int,
    now_str: str,
) -> list[AnomalyFixture]:
    """Anomaly C: content-skill mapping references non-existent skill."""
    fixtures: list[AnomalyFixture] = []
    content_items = master.get("content_items", [])

    for i in range(n):
        seq = seq_start + i
        if content_items:
            content = content_items[int(rng.integers(0, len(content_items)))]
            content_id = content["content_id"]
        else:
            content_id = f"CNT-{(i + 1):06d}"

        # SKL-999999 is guaranteed non-existent
        phantom_skill = _PHANTOM_SKILL_ID

        injected = {
            "content_id": content_id,
            "skill_id": phantom_skill,
            "skill_role": "supporting",
            "weight": 0.5,
        }

        fixtures.append(AnomalyFixture(
            issue_id=_make_id(seq),
            issue_type="CONTENT_MISSING_SKILL",
            severity="high",
            source_table="content_skill",
            source_pk=f"{content_id}:{phantom_skill}",
            status="quarantined",
            expected_decision="reject",
            details={
                "content_id": content_id,
                "missing_skill_id": phantom_skill,
                "reason": (
                    f"content_skill mapping references skill {phantom_skill} "
                    f"which does not exist in the skills table."
                ),
            },
            detected_at=now_str,
            created_at=now_str,
            injected_record=injected,
            affected_ids=[content_id, phantom_skill],
        ))
    return fixtures


def _gen_mastery_future_attempts(
    activity: dict[str, list[dict]],
    rng: np.random.Generator,
    n: int,
    seq_start: int,
    now_str: str,
) -> list[AnomalyFixture]:
    """Anomaly D: mastery fixture that would include a future attempt."""
    fixtures: list[AnomalyFixture] = []
    attempts = activity.get("attempts", [])

    scored = [a for a in attempts if a.get("status") == "scored"]
    if not scored:
        scored = attempts

    for i in range(n):
        seq = seq_start + i
        if scored:
            attempt = scored[int(rng.integers(0, len(scored)))]
            attempt_id = attempt["attempt_id"]
            learner_id = attempt.get("learner_id", "LRN-000001")
            content_id = attempt.get("content_id", "CNT-000001")
            attempt_started = attempt.get("started_at", "2025-06-01 08:00:00.000000")
        else:
            attempt_id = f"ATT-LRN-{seq:06d}-000001"
            learner_id = f"LRN-{seq:06d}"
            content_id = "CNT-000001"
            attempt_started = "2025-12-01 10:00:00.000000"

        # as_of_timestamp is set BEFORE the attempt started_at
        try:
            attempt_dt = datetime.fromisoformat(attempt_started.replace(" ", "T") + "+00:00")
        except Exception:
            attempt_dt = datetime(2025, 12, 1, 10, 0, 0, tzinfo=UTC)

        as_of_dt = attempt_dt - timedelta(days=30)
        as_of_str = format_timestamp(as_of_dt)

        fixtures.append(AnomalyFixture(
            issue_id=_make_id(seq),
            issue_type="MASTERY_FUTURE_ATTEMPT",
            severity="critical",
            source_table="attempt",
            source_pk=attempt_id,
            status="quarantined",
            expected_decision="exclude",
            details={
                "as_of_timestamp": as_of_str,
                "attempt_id": attempt_id,
                "attempt_started_at": attempt_started,
                "learner_id": learner_id,
                "content_id": content_id,
                "reason": (
                    f"Mastery calculation as_of {as_of_str} would include "
                    f"attempt {attempt_id} with started_at {attempt_started}, "
                    "which is AFTER the as_of cutoff (temporal leakage)."
                ),
            },
            detected_at=now_str,
            created_at=now_str,
            injected_record={
                "as_of_timestamp": as_of_str,
                "attempt_id": attempt_id,
                "attempt_started_at": attempt_started,
                "learner_id": learner_id,
            },
            affected_ids=[attempt_id, learner_id],
        ))
    return fixtures


def _gen_path_loop_no_progress(
    master: dict[str, list[dict]],
    rng: np.random.Generator,
    n: int,
    seq_start: int,
    now_str: str,
) -> list[AnomalyFixture]:
    """Anomaly E: learning path with repeated content and no score improvement."""
    fixtures: list[AnomalyFixture] = []
    learners = master.get("learners", [])
    content_items = master.get("content_items", [])

    for i in range(n):
        seq = seq_start + i
        learner = learners[int(rng.integers(0, len(learners)))] if learners else {"learner_id": f"LRN-{(i+1):06d}"}
        content = content_items[int(rng.integers(0, len(content_items)))] if content_items else {"content_id": "CNT-000001", "course_id": "CRS-000001"}
        learner_id = learner["learner_id"]
        content_id = content["content_id"]
        course_id = content.get("course_id", "CRS-000001")

        path_id = f"PATH-2025-{seq:06d}"
        # Synthetic path items: same content appears twice, score does not improve
        path_items = [
            {"lpi_id": f"LPI-{seq:06d}", "position": 1, "content_id": content_id, "score": 45.0, "status": "completed"},
            {"lpi_id": f"LPI-{seq+1000:06d}", "position": 2, "content_id": "CNT-000002", "score": 50.0, "status": "completed"},
            {"lpi_id": f"LPI-{seq+2000:06d}", "position": 3, "content_id": content_id, "score": 44.0, "status": "completed"},
        ]

        fixtures.append(AnomalyFixture(
            issue_id=_make_id(seq),
            issue_type="PATH_LOOP_NO_PROGRESS",
            severity="medium",
            source_table="learning_path",
            source_pk=path_id,
            status="quarantined",
            expected_decision="review",
            details={
                "learner_id": learner_id,
                "path_id": path_id,
                "course_id": course_id,
                "repeated_content_id": content_id,
                "positions": [1, 3],
                "scores": [45.0, 44.0],
                "path_items": path_items,
                "reason": (
                    f"Learning path {path_id} for learner {learner_id} contains "
                    f"content {content_id} at positions 1 and 3 with no score "
                    "improvement (45.0 → 44.0). Possible loop without progress."
                ),
            },
            detected_at=now_str,
            created_at=now_str,
            injected_record={
                "path_id": path_id,
                "learner_id": learner_id,
                "course_id": course_id,
                "path_items": path_items,
            },
            affected_ids=[learner_id, path_id, content_id],
        ))
    return fixtures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_anomaly_fixtures(
    master_tables: Any,
    activity_tables: Any,
    *,
    profile: str = "dev",
    seed: int | None = None,
    n_per_type: int | None = None,
    config: dict | None = None,
) -> AnomalyFixtures:
    """Generate deterministic anomaly fixtures for all five types.

    Parameters
    ----------
    master_tables:
        MasterTables dataclass or dict-of-lists (courses, learners, etc.)
    activity_tables:
        ActivityTables dataclass or dict-of-lists (attempts, assessment_results)
    profile:
        'dev' or 'full'
    seed:
        RNG seed; falls back to config default or 20261022
    n_per_type:
        Override number of fixtures per type (default from config)
    config:
        Loaded app config dict; if None uses defaults
    """
    cfg = config or {}
    if seed is None:
        seed = cfg.get("reproducibility", {}).get("default_seed", 20261022)

    profiles_cfg = cfg.get("dataset_profiles", {})
    profile_cfg = profiles_cfg.get(profile, {})
    if n_per_type is None:
        n_per_type = int(profile_cfg.get("anomaly_injections_per_type", 5))

    rng = np.random.default_rng(seed)
    now_str = _now_utc_str()

    master = _coerce_master(master_tables)
    activity = _coerce_activity(activity_tables)

    # Sequence starts: each type gets its own non-overlapping ID range
    # A: 1..n, B: n+1..2n, C: 2n+1..3n, D: 3n+1..4n, E: 4n+1..5n
    starts = [1 + i * n_per_type for i in range(5)]

    a_fixtures = _gen_rec_prereq_violations(master, activity, rng, n_per_type, starts[0], now_str)
    b_fixtures = _gen_attempt_timestamp_orders(activity, rng, n_per_type, starts[1], now_str)
    c_fixtures = _gen_content_missing_skills(master, rng, n_per_type, starts[2], now_str)
    d_fixtures = _gen_mastery_future_attempts(activity, rng, n_per_type, starts[3], now_str)
    e_fixtures = _gen_path_loop_no_progress(master, rng, n_per_type, starts[4], now_str)

    # Validate all
    for fix in a_fixtures + b_fixtures + c_fixtures + d_fixtures + e_fixtures:
        fix.validate()

    return AnomalyFixtures(
        rec_prereq_violations=a_fixtures,
        attempt_timestamp_orders=b_fixtures,
        content_missing_skills=c_fixtures,
        mastery_future_attempts=d_fixtures,
        path_loop_no_progress=e_fixtures,
    )

