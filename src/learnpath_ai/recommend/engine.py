"""Deterministic, explainable rule-based recommendation engine (Phase 6).

Follows standard pipeline:
  For each eligible learner (status != 'withdrawn') and their enrolled course:
  1. Load latest mastery snapshot per skill at as_of_timestamp.
  2. Identify candidate content items in the learner's course that are published.
  3. Hard prerequisite eligibility filter:
     - Check content primary skills against skill_prerequisites.
     - Any content failing required prerequisites is strictly excluded.
     - Anomaly C check: content mapped to nonexistent skills is excluded.
  4. Heuristic ranking score formula (rule-v1):
     - Skill Gap: (1.0 - mastery_score) * weight_cs
     - Difficulty Suitability: 1.0 - abs(content_diff - target_diff)/4.0
     - Prior Performance Factor: boost retry after fail, discount completed pass
     - Composite Score:
       Score = 0.40 * SkillGap + 0.25 * DiffSuitability + 0.20 * MappingWeight + 0.15 * PerformanceFactor
     - Clamped to [0.000000, 1.000000] matching DECIMAL(8,6).
  5. Deterministic tie-breaking:
     Sort by: score DESC, skill_gap DESC, difficulty_match DESC, content_id ASC.
  6. Return Top-K recommendations (default K=5) with rank_position 1..K.
  7. Generate explainable reason codes array.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.clock import format_timestamp, parse_iso_utc
from learnpath_ai.ids import format_id
from learnpath_ai.prerequisites import PrerequisiteEdge

UTC = timezone.utc
ENGINE_VERSION = "rule-v1"


class RecommendationError(ValueError):
    """Raised when recommendation input or calculation is invalid."""


@dataclass(frozen=True)
class RecommendationRecord:
    recommendation_id: str
    learner_id: str
    content_id: str
    path_id: str | None
    rank_position: int
    engine_version: str
    reason_codes: list[str]
    score: float
    status: str
    recommended_at: str
    as_of_timestamp: str
    expires_at: str | None
    created_at: str
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "learner_id": self.learner_id,
            "content_id": self.content_id,
            "path_id": self.path_id or "",
            "rank_position": self.rank_position,
            "engine_version": self.engine_version,
            "reason_codes": json.dumps(self.reason_codes),
            "score": f"{self.score:.6f}",
            "status": self.status,
            "recommended_at": self.recommended_at,
            "as_of_timestamp": self.as_of_timestamp,
            "expires_at": self.expires_at or "",
            "created_at": self.created_at,
        }


def _parse_ts(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    s = str(val).strip()
    return parse_iso_utc(s)


def generate_recommendations(
    master_tables: Any,
    activity_tables: Any,
    mastery_snapshots: Any,
    *,
    as_of_timestamp: str | datetime | None = None,
    recommended_at: str | datetime | None = None,
    k: int = 5,
    config: dict[str, Any] | None = None,
    engine_version: str = ENGINE_VERSION,
) -> list[RecommendationRecord]:
    """Generate top-K rule-based recommendations for all eligible learners.

    Parameters:
      master_tables: MasterTables or dict containing courses, learners, skills, skill_prerequisites, content_items, content_skills
      activity_tables: ActivityTables or dict containing attempts, assessment_results
      mastery_snapshots: list of MasterySnapshotRecord or list of dicts from Phase 5
      as_of_timestamp: cutoff timestamp for features. Events after this are excluded.
      recommended_at: recommendation issuance timestamp (>= as_of_timestamp).
      k: top-K activities to recommend per learner (default 5, max 20).
      config: application config dictionary.
      engine_version: recommendation engine version tag.
    """
    cfg = config or {}
    rec_cfg = cfg.get("recommendation", {})
    if k is None:
        k = int(rec_cfg.get("k", 5))
    k = max(1, min(20, k))

    # Helper to extract list from dataclass or dict
    def _get_list(source: Any, key: str) -> list[dict[str, Any]]:
        if hasattr(source, key):
            return getattr(source, key)
        if isinstance(source, dict) and key in source:
            return source[key]
        return []

    learners = _get_list(master_tables, "learners")
    courses = _get_list(master_tables, "courses")
    skills = _get_list(master_tables, "skills")
    prereqs = _get_list(master_tables, "skill_prerequisites")
    content_items = _get_list(master_tables, "content_items")
    content_skills = _get_list(master_tables, "content_skills")
    attempts = _get_list(activity_tables, "attempts")
    results = _get_list(activity_tables, "assessment_results")

    # Mastery snapshots normalization
    snapshots_list: list[dict[str, Any]] = []
    if isinstance(mastery_snapshots, list):
        for snap in mastery_snapshots:
            if hasattr(snap, "to_dict"):
                snapshots_list.append(snap.to_dict())
            elif isinstance(snap, dict):
                snapshots_list.append(snap)

    # 1. Determine as_of_dt and recommended_at_dt
    if as_of_timestamp is None:
        if snapshots_list:
            as_of_dt = max(_parse_ts(s["as_of_timestamp"]) for s in snapshots_list)
        elif attempts:
            as_of_dt = max(_parse_ts(a.get("submitted_at") or a.get("started_at")) for a in attempts)
        else:
            as_of_dt = datetime.now(UTC)
    else:
        as_of_dt = _parse_ts(as_of_timestamp)

    if recommended_at is None:
        recommended_at_dt = as_of_dt
    else:
        recommended_at_dt = _parse_ts(recommended_at)

    if as_of_dt > recommended_at_dt:
        raise RecommendationError(
            f"as_of_timestamp ({as_of_dt}) must be <= recommended_at ({recommended_at_dt})"
        )

    as_of_str = format_timestamp(as_of_dt)
    rec_at_str = format_timestamp(recommended_at_dt)
    expires_at_dt = recommended_at_dt + timedelta(days=14)
    expires_at_str = format_timestamp(expires_at_dt)
    created_at_str = rec_at_str

    # 2. Index Skills and check validity (Anomaly C protection)
    known_skill_ids = {str(s["skill_id"]) for s in skills}

    # 3. Build Prerequisite Edges
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

    # 4. Index Content-Skill Links & primary skill map
    content_links: list[ContentSkillLink] = []
    content_primary_skill: dict[str, str] = {}
    content_skills_by_cid: dict[str, list[dict[str, Any]]] = {}

    for cs in content_skills:
        cid = str(cs["content_id"])
        sid = str(cs["skill_id"])
        role = str(cs.get("skill_role", "primary"))
        weight = float(cs.get("weight", 1.0))

        # Check for nonexistent skill (Anomaly C guard)
        if sid not in known_skill_ids:
            continue

        content_links.append(ContentSkillLink(content_id=cid, skill_id=sid, skill_role=role))
        content_skills_by_cid.setdefault(cid, []).append({
            "skill_id": sid,
            "skill_role": role,
            "weight": weight,
        })
        if role == "primary" and cid not in content_primary_skill:
            content_primary_skill[cid] = sid

    # Fallback primary skill if only supporting skills exist
    for cid, s_list in content_skills_by_cid.items():
        if cid not in content_primary_skill and s_list:
            content_primary_skill[cid] = s_list[0]["skill_id"]

    # 5. Index Content Items by course (published only)
    published_content_by_course: dict[str, list[dict[str, Any]]] = {}
    content_by_id: dict[str, dict[str, Any]] = {}
    for c in content_items:
        cid = str(c["content_id"])
        c_status = str(c.get("status", "")).lower()
        content_by_id[cid] = c
        # Only published content is suitable for active recommendations
        if c_status == "published":
            # Must also map to at least one valid known skill
            if cid in content_skills_by_cid:
                crs_id = str(c.get("course_id", ""))
                published_content_by_course.setdefault(crs_id, []).append(c)

    # 6. Index Mastery Snapshots: (learner_id, skill_id) -> float
    # Only snapshots valid at as_of_dt
    mastery_by_learner: dict[str, dict[str, float]] = {}
    for s in snapshots_list:
        lid = str(s["learner_id"])
        sid = str(s["skill_id"])
        snap_as_of = _parse_ts(s.get("as_of_timestamp"))
        if snap_as_of <= as_of_dt:
            score = float(s.get("mastery_score", 0.0))
            # Store latest score
            mastery_by_learner.setdefault(lid, {})[sid] = score

    # 7. Index Historical Attempts & Results <= as_of_dt
    results_by_attempt: dict[str, dict[str, Any]] = {}
    for r in results:
        res_dt = _parse_ts(r.get("scored_at") or r.get("created_at"))
        if res_dt <= as_of_dt:
            aid = str(r["attempt_id"])
            results_by_attempt[aid] = r

    learner_attempt_history: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for a in attempts:
        att_dt = _parse_ts(a.get("started_at"))
        if att_dt <= as_of_dt:
            lid = str(a["learner_id"])
            cid = str(a["content_id"])
            aid = str(a["attempt_id"])
            res = results_by_attempt.get(aid)
            learner_attempt_history.setdefault(lid, {}).setdefault(cid, []).append({
                "attempt_id": aid,
                "status": a.get("status"),
                "started_at": att_dt,
                "result": res,
            })

    # Starting level target difficulty mapping
    level_target_diff = {
        "beginner": 1.5,
        "intermediate": 3.0,
        "advanced": 4.5,
    }

    # 8. Generate recommendations per learner
    recommendations: list[RecommendationRecord] = []
    rec_seq = 1

    # Filter out withdrawn learners (never recommend to withdrawn learners)
    active_learners = [
        l for l in learners
        if str(l.get("status", "")).lower() != "withdrawn"
    ]
    # Sort deterministically
    sorted_learners = sorted(active_learners, key=lambda l: str(l["learner_id"]))

    for learner in sorted_learners:
        lid = str(learner["learner_id"])
        start_level = str(learner.get("starting_level", "beginner")).lower()
        target_diff = level_target_diff.get(start_level, 2.5)

        # Determine target course
        course_id = str(learner.get("current_course_id", ""))
        candidates = published_content_by_course.get(course_id, [])

        if not candidates and published_content_by_course:
            # Fallback if no course assigned or course has no published content:
            # use all published content in first available course
            first_course = sorted(published_content_by_course.keys())[0]
            candidates = published_content_by_course[first_course]

        if not candidates:
            continue

        learner_mastery = mastery_by_learner.get(lid, {})
        learner_history = learner_attempt_history.get(lid, {})

        # Step 4: Hard prerequisite eligibility gate
        eligible_candidates: list[dict[str, Any]] = []
        for cand in candidates:
            cid = str(cand["content_id"])
            if content_passes_required_prerequisites(
                cid,
                content_links,
                prereq_edges,
                learner_mastery,
                missing_mastery=0.0,
            ):
                eligible_candidates.append(cand)

        if not eligible_candidates:
            continue

        # Step 5: Scoring each eligible candidate
        scored_candidates: list[tuple[float, float, float, str, dict[str, Any], list[str]]] = []

        for cand in eligible_candidates:
            cid = str(cand["content_id"])
            diff = float(cand.get("difficulty", 2.5))
            prim_skill = content_primary_skill.get(cid, "")
            current_m = learner_mastery.get(prim_skill, 0.0)

            # Mapping weight of primary skill
            w_cs = 1.0
            for cs_info in content_skills_by_cid.get(cid, []):
                if cs_info["skill_id"] == prim_skill:
                    w_cs = cs_info["weight"]
                    break

            # 1. Skill Gap: lower mastery means higher priority
            skill_gap = max(0.0, 1.0 - current_m)
            weighted_skill_gap = skill_gap * w_cs

            # 2. Difficulty Suitability: match learner starting level / target difficulty
            # Clamped in [0, 1]
            diff_match = max(0.0, 1.0 - abs(diff - target_diff) / 4.0)

            # 3. Prior Performance / History factor
            # Check previous attempts on this content
            prior_attempts = learner_history.get(cid, [])
            perf_factor = 0.50
            reasons: list[str] = []

            # Add primary reason code
            if current_m < 0.60:
                reasons.append("GAP_PRIMARY_SKILL")
                reasons.append("LOW_MASTERY_PRIORITY")
            else:
                reasons.append("PREREQUISITE_SATISFIED")

            if diff_match >= 0.70:
                reasons.append("DIFFICULTY_MATCH")

            if not prior_attempts:
                # Fresh content
                perf_factor = 0.70
            else:
                # Has prior attempts
                last_att = prior_attempts[-1]
                res = last_att.get("result")
                if res is not None:
                    outcome = str(res.get("outcome", "")).lower()
                    if outcome == "fail":
                        # Retry after fail
                        perf_factor = 0.90
                        reasons.append("RETRY_AFTER_FAIL")
                    elif outcome == "pass":
                        # Already passed: lower priority to avoid pointless repetition
                        perf_factor = 0.20
                    else:
                        perf_factor = 0.50
                else:
                    # Incomplete/unscored
                    perf_factor = 0.60

            # Composite ranking score:
            # Score = 0.40 * weighted_skill_gap + 0.25 * diff_match + 0.20 * w_cs + 0.15 * perf_factor
            score = (
                0.40 * weighted_skill_gap
                + 0.25 * diff_match
                + 0.20 * w_cs
                + 0.15 * perf_factor
            )
            score = round(max(0.0, min(1.0, score)), 6)

            # Deduplicate reasons while preserving order
            seen_r: set[str] = set()
            clean_reasons = []
            for r_code in reasons:
                if r_code not in seen_r:
                    clean_reasons.append(r_code)
                    seen_r.add(r_code)

            # Tuple for sorting: score DESC, skill_gap DESC, diff_match DESC, cid ASC
            scored_candidates.append(
                (score, skill_gap, diff_match, cid, cand, clean_reasons)
            )

        # Deterministic sorting
        scored_candidates.sort(
            key=lambda x: (-x[0], -x[1], -x[2], x[3])
        )

        # Top-K
        top_k = scored_candidates[:k]

        for rank, (score, s_gap, d_match, cid, cand, reasons) in enumerate(top_k, start=1):
            rec_id = format_id("REC", rec_seq)
            rec_seq += 1

            prim_skill = content_primary_skill.get(cid, "")
            explanation = {
                "targeted_skill_id": prim_skill,
                "learner_mastery": learner_mastery.get(prim_skill, 0.0),
                "skill_gap": s_gap,
                "content_difficulty": float(cand.get("difficulty", 2.5)),
                "difficulty_match": d_match,
                "prerequisites_verified": True,
            }

            recommendations.append(
                RecommendationRecord(
                    recommendation_id=rec_id,
                    learner_id=lid,
                    content_id=cid,
                    path_id=None,
                    rank_position=rank,
                    engine_version=engine_version,
                    reason_codes=reasons,
                    score=score,
                    status="pending",
                    recommended_at=rec_at_str,
                    as_of_timestamp=as_of_str,
                    expires_at=expires_at_str,
                    created_at=created_at_str,
                    explanation=explanation,
                )
            )

    return recommendations

