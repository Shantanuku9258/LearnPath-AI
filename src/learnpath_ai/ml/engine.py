"""ML-based recommendation engine (Phase 8).

Reuses:
- Hard prerequisite gate (`content_passes_required_prerequisites`)
- Schema column conventions (`RecommendationRecord`)
- Deterministic ID formatting (`REC-XXXXXX`)
- Reason codes explainability
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.clock import format_timestamp, parse_iso_utc
from learnpath_ai.ids import format_id
from learnpath_ai.ml.features import (
    calculate_difficulty_suitability,
    extract_eight_features,
)
from learnpath_ai.ml.model import ModelArtifact
from learnpath_ai.prerequisites import PrerequisiteEdge
from learnpath_ai.recommend.engine import RecommendationRecord

UTC = timezone.utc
ENGINE_VERSION = "ml-v1"


def _parse_ts(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    return parse_iso_utc(str(val).strip())


def generate_ml_recommendations(
    master_tables: Any,
    activity_tables: Any,
    mastery_snapshots: Any,
    model_artifact: ModelArtifact,
    *,
    as_of_timestamp: str | datetime | None = None,
    recommended_at: str | datetime | None = None,
    k: int = 5,
    engine_version: str = ENGINE_VERSION,
) -> list[RecommendationRecord]:
    """Rank eligible content items using ML predictions while strictly respecting prerequisite gates."""
    def _get_list(src: Any, key: str) -> list[dict[str, Any]]:
        if hasattr(src, key):
            return list(getattr(src, key))
        if isinstance(src, dict) and key in src:
            return list(src[key])
        return []

    learners = _get_list(master_tables, "learners")
    courses = _get_list(master_tables, "courses")
    skills = _get_list(master_tables, "skills")
    prereqs = _get_list(master_tables, "skill_prerequisites")
    content_items = _get_list(master_tables, "content_items")
    content_skills = _get_list(master_tables, "content_skills")
    attempts = _get_list(activity_tables, "attempts")
    results = _get_list(activity_tables, "assessment_results")

    # Mastery snapshots
    snapshots_list: list[dict[str, Any]] = []
    if isinstance(mastery_snapshots, list):
        for snap in mastery_snapshots:
            if hasattr(snap, "to_dict"):
                snapshots_list.append(snap.to_dict())
            elif isinstance(snap, dict):
                snapshots_list.append(snap)

    if as_of_timestamp is None:
        if snapshots_list:
            as_of_dt = max(_parse_ts(s["as_of_timestamp"]) for s in snapshots_list)
        else:
            as_of_dt = datetime.now(UTC)
    else:
        as_of_dt = _parse_ts(as_of_timestamp)

    rec_at_dt = _parse_ts(recommended_at) if recommended_at else as_of_dt
    as_of_str = format_timestamp(as_of_dt)
    rec_at_str = format_timestamp(rec_at_dt)
    expires_at_str = format_timestamp(rec_at_dt + timedelta(days=14))

    known_skill_ids = {str(s["skill_id"]) for s in skills}

    prereq_edges: list[PrerequisiteEdge] = []
    for p in prereqs:
        sid = str(p["skill_id"])
        psid = str(p["prerequisite_skill_id"])
        if sid in known_skill_ids and psid in known_skill_ids:
            prereq_edges.append(
                PrerequisiteEdge(
                    skill_id=sid,
                    prerequisite_skill_id=psid,
                    relationship_type=str(p.get("relationship_type", "required")),
                    min_mastery_required=float(p.get("min_mastery_required", 0.60)),
                )
            )

    content_links: list[ContentSkillLink] = []
    content_primary_skill: dict[str, str] = {}
    content_skills_by_cid: dict[str, list[dict[str, Any]]] = {}

    for cs in content_skills:
        cid = str(cs["content_id"])
        sid = str(cs["skill_id"])
        role = str(cs.get("skill_role", "primary"))
        weight = float(cs.get("weight", 1.0))
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

    for cid, s_list in content_skills_by_cid.items():
        if cid not in content_primary_skill and s_list:
            content_primary_skill[cid] = s_list[0]["skill_id"]

    published_content_by_course: dict[str, list[dict[str, Any]]] = {}
    for c in content_items:
        cid = str(c["content_id"])
        if str(c.get("status", "")).lower() == "published" and cid in content_skills_by_cid:
            crs_id = str(c.get("course_id", ""))
            published_content_by_course.setdefault(crs_id, []).append(c)

    # Learner mastery index <= as_of_dt
    mastery_by_learner: dict[str, dict[str, float]] = {}
    for s in snapshots_list:
        snap_as_of = _parse_ts(s.get("as_of_timestamp"))
        if snap_as_of <= as_of_dt:
            lid = str(s["learner_id"])
            sid = str(s["skill_id"])
            score = float(s.get("mastery_score", 0.0))
            mastery_by_learner.setdefault(lid, {})[sid] = score

    # Activity history strictly before as_of_dt (prior = started before T, not at T)
    # Results scored at or before as_of_dt are valid evidence (scoring can happen at T)
    results_by_attempt = {
        str(r["attempt_id"]): r for r in results
        if _parse_ts(r.get("scored_at") or r.get("created_at")) <= as_of_dt
    }

    learner_attempt_history: dict[str, dict[str, list[dict[str, Any]]]] = {}
    learner_overall_attempts: dict[str, int] = {}
    learner_overall_scored: dict[str, int] = {}
    learner_overall_passes: dict[str, int] = {}

    for a in attempts:
        att_dt = _parse_ts(a.get("started_at"))
        if att_dt < as_of_dt:  # strict: prior = started before T (matches training semantics)
            lid = str(a["learner_id"])
            cid = str(a["content_id"])
            aid = str(a["attempt_id"])
            status = str(a.get("status", "")).strip().lower()

            if status != "invalid":
                learner_overall_attempts[lid] = learner_overall_attempts.get(lid, 0) + 1

            res = results_by_attempt.get(aid)
            is_scored = res is not None and res.get("scored_at") and _parse_ts(res["scored_at"]) <= as_of_dt
            passed = 1 if is_scored and (res.get("outcome") == "pass" or int(res.get("passed", 0)) == 1) else 0

            if is_scored:
                learner_overall_scored[lid] = learner_overall_scored.get(lid, 0) + 1
                if passed:
                    learner_overall_passes[lid] = learner_overall_passes.get(lid, 0) + 1

            learner_attempt_history.setdefault(lid, {}).setdefault(cid, []).append({
                "status": status,
                "passed": passed,
            })

    recommendations: list[RecommendationRecord] = []
    rec_seq = 1

    active_learners = [
        l for l in learners
        if str(l.get("status", "")).lower() != "withdrawn"
    ]
    sorted_learners = sorted(active_learners, key=lambda l: str(l["learner_id"]))

    for learner in sorted_learners:
        lid = str(learner["learner_id"])
        start_level = str(learner.get("starting_level", "beginner")).lower()

        course_id = str(learner.get("current_course_id", ""))
        candidates = published_content_by_course.get(course_id, [])
        if not candidates and published_content_by_course:
            first_course = sorted(published_content_by_course.keys())[0]
            candidates = published_content_by_course[first_course]

        if not candidates:
            continue

        learner_mastery = mastery_by_learner.get(lid, {})
        learner_history = learner_attempt_history.get(lid, {})

        total_prior_attempts = learner_overall_attempts.get(lid, 0)
        total_scored = learner_overall_scored.get(lid, 0)
        total_passes = learner_overall_passes.get(lid, 0)
        overall_pass_rate = (total_passes / total_scored) if total_scored > 0 else 0.5

        # Hard prerequisite eligibility filter
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

        feature_vectors: list[list[float]] = []
        metadata_list: list[dict[str, Any]] = []

        for cand in eligible_candidates:
            cid = str(cand["content_id"])
            diff = float(cand.get("difficulty", 2.5))
            diff_match = calculate_difficulty_suitability(diff, start_level)
            prim_skill = content_primary_skill.get(cid, "")
            current_m = learner_mastery.get(prim_skill, 0.0)
            skill_gap = max(0.0, 1.0 - current_m)

            w_cs = 1.0
            for cs_info in content_skills_by_cid.get(cid, []):
                if cs_info["skill_id"] == prim_skill:
                    w_cs = cs_info["weight"]
                    break

            prior_atts = learner_history.get(cid, [])
            prior_count = sum(1 for a in prior_atts if a.get("status") != "invalid")
            prior_passes = sum(a["passed"] for a in prior_atts)

            vec = extract_eight_features(
                primary_skill_mastery=current_m,
                content_difficulty=diff,
                diff_suitability=diff_match,
                primary_skill_weight=w_cs,
                prior_attempts_on_content=prior_count,
                prior_passes_on_content=prior_passes,
                overall_prior_attempts=total_prior_attempts,
                overall_prior_pass_rate=overall_pass_rate,
            )
            feature_vectors.append(vec)

            # Determine explainability reasons
            reasons = []
            if current_m < 0.60:
                reasons.append("GAP_PRIMARY_SKILL")
                reasons.append("LOW_MASTERY_PRIORITY")
            else:
                reasons.append("PREREQUISITE_SATISFIED")

            if diff_match >= 0.70:
                reasons.append("DIFFICULTY_MATCH")

            if prior_count > 0 and prior_passes == 0:
                reasons.append("RETRY_AFTER_FAIL")

            metadata_list.append({
                "content_id": cid,
                "cand": cand,
                "skill_gap": skill_gap,
                "diff_match": diff_match,
                "reasons": reasons,
                "targeted_skill_id": prim_skill,
                "learner_mastery": current_m,
            })

        # Predict ML probabilities
        probs = model_artifact.predict_proba(feature_vectors)

        scored_candidates = []
        for prob, meta in zip(probs, metadata_list):
            scored_candidates.append(
                (prob, meta["skill_gap"], meta["diff_match"], meta["content_id"], meta["cand"], meta["reasons"], meta)
            )

        # Sort: prob DESC, skill_gap DESC, diff_match DESC, content_id ASC
        scored_candidates.sort(
            key=lambda x: (-x[0], -x[1], -x[2], x[3])
        )

        top_k = scored_candidates[:k]

        for rank, (score, s_gap, d_match, cid, cand, reasons, meta) in enumerate(top_k, start=1):
            rec_id = format_id("REC", rec_seq)
            rec_seq += 1

            explanation = {
                "targeted_skill_id": meta["targeted_skill_id"],
                "learner_mastery": meta["learner_mastery"],
                "skill_gap": s_gap,
                "content_difficulty": float(cand.get("difficulty", 2.5)),
                "difficulty_match": d_match,
                "predicted_probability": round(score, 6),
                "model_version": model_artifact.model_version,
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
                    score=round(score, 6),
                    status="pending",
                    recommended_at=rec_at_str,
                    as_of_timestamp=as_of_str,
                    expires_at=expires_at_str,
                    created_at=rec_at_str,
                    explanation=explanation,
                )
            )

    return recommendations

