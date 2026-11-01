"""Learning-path generation engine (Phase 7 & 7.1).

Workflow
--------
For each non-withdrawn learner with an enrolled course:

1. Validate skill prerequisite DAG via `require_dag`.
2. Load that learner's recommendations from Phase 6 at a given as_of_timestamp.
3. Apply the same hard prerequisite eligibility gate used in Phase 6.
   Content that fails the gate is excluded with no exception.
4. Keep only recommendations whose as_of_timestamp is <= the supplied cutoff.
   This preserves temporal correctness: no future mastery or attempt data leaks in.
5. Order items by (rank_position ASC, recommendation score DESC, content_id ASC)
   for a deterministic sequence.
6. Detect duplicate content IDs in a single path (loop / cycle guard).
7. Enforce one active path per (learner_id, course_id): if a learner-course pair
   already has an active path in the generated set, skip creating a second one.
8. Produce LearningPath + LearningPathItem records.
9. Support progress tracking:
   - planned -> in_progress -> completed
   - planned -> skipped
   - auto-completion of LearningPath when all items reach terminal status (completed/skipped)
     and at least one completed item exists.
10. Support replanning:
   - Archive previous active path to 'abandoned'
   - Emit new LearningPath with new path_id (PATH-YYYY-XXXXXX)
   - Preserve historical records without modification
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.clock import format_timestamp, parse_iso_utc
from learnpath_ai.prerequisites import PrerequisiteEdge, require_dag

UTC = timezone.utc
ENGINE_VERSION = "path-v1"


class LearningPathError(ValueError):
    """Raised when path generation or tracking encounters an invalid operation."""


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningPathRecord:
    path_id: str
    learner_id: str
    course_id: str
    status: str          # 'draft', 'active', 'completed', 'abandoned'
    generated_at: str    # UTC timestamp string
    activated_at: str    # set when active/completed/abandoned
    completed_at: str    # set when completed
    generator_version: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "learner_id": self.learner_id,
            "course_id": self.course_id,
            "status": self.status,
            "generated_at": self.generated_at,
            "activated_at": self.activated_at,
            "completed_at": self.completed_at,
            "generator_version": self.generator_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class LearningPathItemRecord:
    path_item_id: str
    path_id: str
    sequence_no: int
    content_id: str
    recommendation_id: str   # empty string if not linked
    status: str              # 'planned', 'in_progress', 'completed', 'skipped'
    planned_at: str
    completed_at: str        # set when completed or skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_item_id": self.path_item_id,
            "path_id": self.path_id,
            "sequence_no": self.sequence_no,
            "content_id": self.content_id,
            "recommendation_id": self.recommendation_id,
            "status": self.status,
            "planned_at": self.planned_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _path_id(year: int, seq: int) -> str:
    """Format a PATH-YYYY-000001 identifier."""
    return f"PATH-{year:04d}-{seq:06d}"


def _lpi_id(seq: int) -> str:
    """Format an LPI-000001 identifier."""
    return f"LPI-{seq:06d}"


# ---------------------------------------------------------------------------
# Progress Tracking Functions
# ---------------------------------------------------------------------------

VALID_ITEM_TRANSITIONS = {
    "planned": {"in_progress", "skipped"},
    "in_progress": {"completed"},
    "completed": set(),
    "skipped": set(),
}


def update_item_status(
    item: LearningPathItemRecord,
    new_status: str,
    *,
    timestamp: str | datetime | None = None,
) -> LearningPathItemRecord:
    """Transition an item to new_status with validation of allowed transitions."""
    current_status = item.status
    allowed = VALID_ITEM_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise LearningPathError(
            f"Invalid item status transition from '{current_status}' to '{new_status}'. "
            f"Allowed transitions: {allowed if allowed else 'none (terminal state)'}."
        )

    ts_str = format_timestamp(_parse(timestamp)) if timestamp else item.planned_at

    completed_at = ts_str if new_status in {"completed", "skipped"} else ""
    return replace(
        item,
        status=new_status,
        completed_at=completed_at,
    )


def start_path_item(
    item: LearningPathItemRecord,
    *,
    timestamp: str | datetime | None = None,
) -> LearningPathItemRecord:
    """Transition planned -> in_progress."""
    return update_item_status(item, "in_progress", timestamp=timestamp)


def complete_path_item(
    item: LearningPathItemRecord,
    *,
    timestamp: str | datetime | None = None,
) -> LearningPathItemRecord:
    """Transition in_progress -> completed."""
    return update_item_status(item, "completed", timestamp=timestamp)


def skip_path_item(
    item: LearningPathItemRecord,
    *,
    timestamp: str | datetime | None = None,
) -> LearningPathItemRecord:
    """Transition planned -> skipped."""
    return update_item_status(item, "skipped", timestamp=timestamp)


def check_and_update_path_completion(
    path: LearningPathRecord,
    items: list[LearningPathItemRecord],
    *,
    timestamp: str | datetime | None = None,
) -> LearningPathRecord:
    """Check if all items in the path are finished and update path to completed if so."""
    path_items = [it for it in items if it.path_id == path.path_id]
    if not path_items:
        return path

    terminal_statuses = {"completed", "skipped"}
    all_terminal = all(it.status in terminal_statuses for it in path_items)
    any_completed = any(it.status == "completed" for it in path_items)

    if all_terminal and any_completed and path.status == "active":
        ts_str = format_timestamp(_parse(timestamp)) if timestamp else path_items[-1].completed_at or path.generated_at
        return replace(
            path,
            status="completed",
            completed_at=ts_str,
            updated_at=ts_str,
        )

    return path


# ---------------------------------------------------------------------------
# Replanning Functions
# ---------------------------------------------------------------------------

def replan_learning_path(
    existing_paths: list[LearningPathRecord],
    existing_items: list[LearningPathItemRecord],
    learner_id: str,
    course_id: str,
    master_tables: Any,
    recommendations: list[dict[str, Any]],
    mastery_snapshots: list[dict[str, Any]],
    *,
    as_of_timestamp: str | datetime | None = None,
    generated_at: str | datetime | None = None,
    engine_version: str = ENGINE_VERSION,
) -> tuple[list[LearningPathRecord], list[LearningPathItemRecord]]:
    """Replan an existing learner's path while preserving history.

    - The currently active path for (learner_id, course_id) is marked 'abandoned'.
    - A new active path is created with a new path_id.
    - All previous paths and items remain in the returned lists.
    - Exactly one active path exists for (learner_id, course_id).
    """
    as_of_dt = _parse(as_of_timestamp) if as_of_timestamp else datetime.now(UTC)
    gen_dt = _parse(generated_at) if generated_at else as_of_dt
    gen_str = format_timestamp(gen_dt)

    updated_paths: list[LearningPathRecord] = []
    found_active = False

    for p in existing_paths:
        if p.learner_id == learner_id and p.course_id == course_id and p.status == "active":
            # Archive previous active path to abandoned
            abandoned_p = replace(p, status="abandoned", updated_at=gen_str)
            updated_paths.append(abandoned_p)
            found_active = True
        else:
            updated_paths.append(p)

    # Next sequence numbers based on existing
    import re
    max_path_seq = 0
    for p in existing_paths:
        m = re.match(r"^PATH-\d{4}-(\d{6})$", p.path_id)
        if m:
            max_path_seq = max(max_path_seq, int(m.group(1)))

    max_item_seq = 0
    for it in existing_items:
        m = re.match(r"^LPI-(\d{6})$", it.path_item_id)
        if m:
            max_item_seq = max(max_item_seq, int(m.group(1)))

    # Filter master data for only this learner
    learners = master_tables.get("learners", []) if isinstance(master_tables, dict) else getattr(master_tables, "learners", [])
    single_learner = [l for l in learners if str(l.get("learner_id")) == learner_id]
    if isinstance(master_tables, dict):
        learner_master = dict(master_tables)
        learner_master["learners"] = single_learner
    else:
        learner_master = {"learners": single_learner}
        for k in ["courses", "content_items", "content_skills", "skills", "skill_prerequisites"]:
            learner_master[k] = getattr(master_tables, k, [])

    new_paths, new_items = generate_learning_paths(
        learner_master,
        recommendations,
        mastery_snapshots,
        as_of_timestamp=as_of_timestamp,
        generated_at=generated_at,
        engine_version=engine_version,
        start_path_seq=max_path_seq + 1,
        start_item_seq=max_item_seq + 1,
    )

    all_paths = updated_paths + new_paths
    all_items = list(existing_items) + new_items
    return all_paths, all_items


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def generate_learning_paths(
    master_tables: Any,
    recommendations: list[dict[str, Any]],
    mastery_snapshots: list[dict[str, Any]],
    *,
    as_of_timestamp: str | datetime | None = None,
    generated_at: str | datetime | None = None,
    engine_version: str = ENGINE_VERSION,
    start_path_seq: int = 1,
    start_item_seq: int = 1,
) -> tuple[list[LearningPathRecord], list[LearningPathItemRecord]]:
    """Generate LearningPath and LearningPathItem records for all eligible learners.

    Parameters
    ----------
    master_tables:
        dict with keys: learners, courses, content_items, content_skills,
        skills, skill_prerequisites.
    recommendations:
        List of recommendation row dicts from Phase 6 CSV.
    mastery_snapshots:
        List of mastery snapshot row dicts from Phase 5 CSV.
    as_of_timestamp:
        Feature cutoff. Only recommendations with as_of_timestamp <= this value
        are used. Defaults to the max as_of_timestamp in recommendations.
    generated_at:
        Wall-clock timestamp for path record generation. Must be >= as_of_timestamp.
        Defaults to as_of_timestamp to keep outputs deterministic.
    engine_version:
        String tag written to each LearningPath record.
    start_path_seq:
        Initial sequence number for path ID generation (for replanning support).
    start_item_seq:
        Initial sequence number for path item ID generation.

    Returns
    -------
    (paths, items) — two parallel lists.
    """

    def _get(src: Any, key: str) -> list[dict[str, Any]]:
        if hasattr(src, key):
            return list(getattr(src, key))
        if isinstance(src, dict):
            return list(src.get(key, []))
        return []

    learners = _get(master_tables, "learners")
    content_items = _get(master_tables, "content_items")
    content_skills = _get(master_tables, "content_skills")
    skills = _get(master_tables, "skills")
    prereqs = _get(master_tables, "skill_prerequisites")

    # --- Parse as_of_dt ---
    if as_of_timestamp is None:
        if recommendations:
            as_of_dt = max(
                _parse(r["as_of_timestamp"]) for r in recommendations
            )
        else:
            as_of_dt = datetime.now(UTC)
    else:
        as_of_dt = _parse(as_of_timestamp)

    if generated_at is None:
        gen_dt = as_of_dt
    else:
        gen_dt = _parse(generated_at)

    if gen_dt < as_of_dt:
        raise LearningPathError(
            f"generated_at ({gen_dt}) must be >= as_of_timestamp ({as_of_dt})"
        )

    as_of_str = format_timestamp(as_of_dt)
    gen_str = format_timestamp(gen_dt)
    year = gen_dt.year

    # --- Build eligibility structures and validate DAG ---
    known_skill_ids = {str(s["skill_id"]) for s in skills}

    prereq_edges: list[PrerequisiteEdge] = []
    for p in prereqs:
        sid = str(p["skill_id"])
        psid = str(p["prerequisite_skill_id"])
        rtype = str(p.get("relationship_type", "required"))
        min_m = float(p.get("min_mastery_required", 0.60))
        if sid in known_skill_ids and psid in known_skill_ids:
            prereq_edges.append(
                PrerequisiteEdge(
                    skill_id=sid,
                    prerequisite_skill_id=psid,
                    relationship_type=rtype,
                    min_mastery_required=min_m,
                )
            )

    # 3. Prerequisite graph cycle validation via require_dag
    require_dag(prereq_edges)

    content_links: list[ContentSkillLink] = []
    for cs in content_skills:
        cid = str(cs["content_id"])
        sid = str(cs["skill_id"])
        role = str(cs.get("skill_role", "primary"))
        if sid in known_skill_ids:
            content_links.append(
                ContentSkillLink(content_id=cid, skill_id=sid, skill_role=role)
            )

    content_by_id: dict[str, dict[str, Any]] = {
        str(c["content_id"]): c for c in content_items
    }

    # --- Build mastery index: learner_id -> {skill_id: float} at as_of_dt ---
    mastery_by_learner: dict[str, dict[str, float]] = {}
    for snap in mastery_snapshots:
        snap_asof = _parse(snap.get("as_of_timestamp", ""))
        if snap_asof <= as_of_dt:
            lid = str(snap["learner_id"])
            sid = str(snap["skill_id"])
            score = float(snap.get("mastery_score", 0.0))
            mastery_by_learner.setdefault(lid, {})[sid] = score

    # --- Filter recommendations to those <= as_of_dt ---
    valid_recs: list[dict[str, Any]] = []
    for r in recommendations:
        rec_asof = _parse(r.get("as_of_timestamp", ""))
        if rec_asof <= as_of_dt:
            valid_recs.append(r)

    recs_by_learner: dict[str, list[dict[str, Any]]] = {}
    for r in valid_recs:
        lid = str(r["learner_id"])
        recs_by_learner.setdefault(lid, []).append(r)

    learner_by_id: dict[str, dict[str, Any]] = {
        str(l["learner_id"]): l for l in learners
    }

    paths: list[LearningPathRecord] = []
    items: list[LearningPathItemRecord] = []

    path_seq = start_path_seq
    item_seq = start_item_seq

    active_pairs: set[tuple[str, str]] = set()

    sorted_learner_ids = sorted(recs_by_learner.keys())

    for lid in sorted_learner_ids:
        learner = learner_by_id.get(lid)
        if learner is None:
            continue
        if str(learner.get("status", "")).lower() == "withdrawn":
            continue

        course_id = str(learner.get("current_course_id", "") or "")
        if not course_id:
            continue

        pair = (lid, course_id)
        if pair in active_pairs:
            continue
        active_pairs.add(pair)

        learner_mastery = mastery_by_learner.get(lid, {})
        learner_recs = recs_by_learner[lid]

        eligible_recs: list[dict[str, Any]] = []
        for r in learner_recs:
            cid = str(r["content_id"])
            content = content_by_id.get(cid)
            if content is None:
                continue
            if str(content.get("status", "")).lower() != "published":
                continue
            if content_passes_required_prerequisites(
                cid,
                content_links,
                prereq_edges,
                learner_mastery,
                missing_mastery=0.0,
            ):
                eligible_recs.append(r)

        if not eligible_recs:
            continue

        # Safeguard: deduplicate repeated content IDs
        seen_content: set[str] = set()
        deduped_recs: list[dict[str, Any]] = []
        for r in eligible_recs:
            cid = str(r["content_id"])
            if cid not in seen_content:
                seen_content.add(cid)
                deduped_recs.append(r)

        def _sort_key(r: dict[str, Any]) -> tuple[int, float, str]:
            try:
                rank = int(r.get("rank_position", 99))
            except (TypeError, ValueError):
                rank = 99
            try:
                score = -float(r.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            return (rank, score, str(r.get("content_id", "")))

        ordered_recs = sorted(deduped_recs, key=_sort_key)

        path_id = _path_id(year, path_seq)
        path_seq += 1

        path = LearningPathRecord(
            path_id=path_id,
            learner_id=lid,
            course_id=course_id,
            status="active",
            generated_at=gen_str,
            activated_at=gen_str,
            completed_at="",
            generator_version=engine_version,
            created_at=gen_str,
            updated_at=gen_str,
        )
        paths.append(path)

        for seq_no, r in enumerate(ordered_recs, start=1):
            lpi_id = _lpi_id(item_seq)
            item_seq += 1

            item = LearningPathItemRecord(
                path_item_id=lpi_id,
                path_id=path_id,
                sequence_no=seq_no,
                content_id=str(r["content_id"]),
                recommendation_id=str(r.get("recommendation_id", "")),
                status="planned",
                planned_at=gen_str,
                completed_at="",
            )
            items.append(item)

    return paths, items


def _parse(val: Any) -> datetime:
    """Parse a string or datetime to UTC-aware datetime."""
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    s = str(val).strip()
    return parse_iso_utc(s)
