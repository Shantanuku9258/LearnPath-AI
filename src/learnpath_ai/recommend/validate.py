"""Validation logic for recommendations dataset (Phase 6)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.clock import parse_iso_utc
from learnpath_ai.generate.columns import (
    RECOMMENDATION_COLUMNS,
    RECOMMENDATION_MANIFEST_NAME,
)
from learnpath_ai.ids import is_valid_id
from learnpath_ai.prerequisites import PrerequisiteEdge
from learnpath_ai.recommend.writer import _sha256

VALID_STATUSES = {"pending", "accepted", "dismissed", "expired", "completed", "quarantined"}


class RecommendationValidationError(ValueError):
    """Raised when recommendation dataset violates schema constraints, eligibility or manifest."""


def validate_recommendations(
    recommendations: list[dict[str, Any]],
    master_tables: Any | None = None,
    mastery_snapshots: list[dict[str, Any]] | None = None,
    *,
    expected_profile: str | None = None,
) -> None:
    """Validate a list of recommendation dictionaries against schema, constraints and eligibility."""
    if not recommendations:
        raise RecommendationValidationError("Recommendation dataset is empty.")

    known_learners: set[str] | None = None
    known_content: set[str] | None = None
    known_skills: set[str] | None = None
    prereq_edges: list[PrerequisiteEdge] = []
    content_links: list[ContentSkillLink] = []

    if master_tables is not None:
        def _get_list(key: str) -> list[dict[str, Any]]:
            if hasattr(master_tables, key):
                return getattr(master_tables, key)
            if isinstance(master_tables, dict) and key in master_tables:
                return master_tables[key]
            return []

        known_learners = {str(r["learner_id"]) for r in _get_list("learners")}
        known_content = {str(r["content_id"]) for r in _get_list("content_items")}
        known_skills = {str(r["skill_id"]) for r in _get_list("skills")}

        for p in _get_list("skill_prerequisites"):
            sid = str(p["skill_id"])
            psid = str(p["prerequisite_skill_id"])
            if sid in known_skills and psid in known_skills:
                prereq_edges.append(
                    PrerequisiteEdge(
                        skill_id=sid,
                        prerequisite_skill_id=psid,
                        relationship_type=str(p.get("relationship_type", "required")),
                        min_mastery_required=float(p.get("min_mastery_required", 0.60)),
                    )
                )

        for cs in _get_list("content_skills"):
            cid = str(cs["content_id"])
            sid = str(cs["skill_id"])
            if sid in known_skills:
                content_links.append(
                    ContentSkillLink(
                        content_id=cid,
                        skill_id=sid,
                        skill_role=str(cs.get("skill_role", "primary")),
                    )
                )

    # Index mastery snapshots if provided: (learner_id, as_of_ts) -> dict[skill_id, float]
    mastery_map: dict[tuple[str, str], dict[str, float]] = {}
    if mastery_snapshots:
        for ms in mastery_snapshots:
            lid = str(ms["learner_id"])
            sid = str(ms["skill_id"])
            as_of_ts = str(ms["as_of_timestamp"])
            score = float(ms.get("mastery_score", 0.0))
            mastery_map.setdefault((lid, as_of_ts), {})[sid] = score

    seen_rec_ids: set[str] = set()
    seen_learner_content_asof: set[tuple[str, str, str, str]] = set()
    seen_learner_rank_asof: set[tuple[str, int, str, str]] = set()

    for idx, row in enumerate(recommendations):
        # 1. Recommendation ID format and uniqueness
        rid = row.get("recommendation_id", "")
        if not is_valid_id(rid, "REC"):
            raise RecommendationValidationError(
                f"Row {idx}: Invalid recommendation_id format '{rid}'. Must match '^REC-[0-9]{{6}}$'."
            )
        if rid in seen_rec_ids:
            raise RecommendationValidationError(f"Row {idx}: Duplicate recommendation_id '{rid}'.")
        seen_rec_ids.add(rid)

        # 2. Foreign keys: learner_id, content_id
        lid = row.get("learner_id", "")
        if not is_valid_id(lid, "LRN"):
            raise RecommendationValidationError(f"Row {idx}: Invalid learner_id format '{lid}'.")
        if known_learners and lid not in known_learners:
            raise RecommendationValidationError(f"Row {idx}: Unknown learner_id '{lid}'.")

        cid = row.get("content_id", "")
        if not is_valid_id(cid, "CNT"):
            raise RecommendationValidationError(f"Row {idx}: Invalid content_id format '{cid}'.")
        if known_content and cid not in known_content:
            raise RecommendationValidationError(f"Row {idx}: Unknown content_id '{cid}'.")

        # 3. Rank position: between 1 and 20
        try:
            rank = int(row.get("rank_position", 0))
        except (ValueError, TypeError):
            raise RecommendationValidationError(f"Row {idx}: Unparseable rank_position.")
        if rank < 1 or rank > 20:
            raise RecommendationValidationError(
                f"Row {idx}: rank_position {rank} out of bounds [1, 20]."
            )

        # 4. Engine version & status
        engine_ver = row.get("engine_version", "")
        if not engine_ver:
            raise RecommendationValidationError(f"Row {idx}: Missing engine_version.")

        status = row.get("status", "")
        if status not in VALID_STATUSES:
            raise RecommendationValidationError(
                f"Row {idx}: Invalid status '{status}'. Must be one of {VALID_STATUSES}."
            )

        # 5. Score in [0.0, 1.0] if present
        score_val = row.get("score")
        if score_val is not None and score_val != "":
            try:
                score = float(score_val)
                if score < 0.0 or score > 1.0:
                    raise RecommendationValidationError(
                        f"Row {idx}: Score {score} out of range [0.0, 1.0]."
                    )
            except ValueError:
                raise RecommendationValidationError(f"Row {idx}: Unparseable score '{score_val}'.")

        # 6. Timestamps: as_of_timestamp <= recommended_at
        as_of_raw = row.get("as_of_timestamp", "")
        rec_at_raw = row.get("recommended_at", "")
        if not as_of_raw or not rec_at_raw:
            raise RecommendationValidationError(f"Row {idx}: Missing timestamps.")

        as_of_dt = parse_iso_utc(as_of_raw)
        rec_at_dt = parse_iso_utc(rec_at_raw)
        if as_of_dt > rec_at_dt:
            raise RecommendationValidationError(
                f"Row {idx}: Temporal violation: as_of_timestamp ({as_of_raw}) > recommended_at ({rec_at_raw})."
            )

        # 7. Uniqueness constraints
        # uq_rec_learner_content_asof_engine (learner_id, content_id, as_of_timestamp, engine_version)
        key1 = (lid, cid, as_of_raw, engine_ver)
        if key1 in seen_learner_content_asof:
            raise RecommendationValidationError(
                f"Row {idx}: Duplicate recommendation for (learner, content, as_of, engine): {key1}"
            )
        seen_learner_content_asof.add(key1)

        # uq_rec_learner_asof_engine_rank (learner_id, as_of_timestamp, engine_version, rank_position)
        key2 = (lid, rank, as_of_raw, engine_ver)
        if key2 in seen_learner_rank_asof:
            raise RecommendationValidationError(
                f"Row {idx}: Duplicate rank for learner at cutoff: {key2}"
            )
        seen_learner_rank_asof.add(key2)

        # 8. Reason codes must be a valid non-empty JSON list
        raw_reasons = row.get("reason_codes", "[]")
        try:
            parsed_reasons = json.loads(raw_reasons) if isinstance(raw_reasons, str) else raw_reasons
            if not isinstance(parsed_reasons, list) or not parsed_reasons:
                raise RecommendationValidationError(
                    f"Row {idx}: reason_codes must be a non-empty list, got {raw_reasons}"
                )
        except json.JSONDecodeError:
            raise RecommendationValidationError(f"Row {idx}: Invalid JSON in reason_codes: {raw_reasons}")

        # 9. Hard prerequisite eligibility verification
        if prereq_edges and content_links:
            learner_m = mastery_map.get((lid, as_of_raw), {})
            if not content_passes_required_prerequisites(
                cid, content_links, prereq_edges, learner_m, missing_mastery=0.0
            ):
                raise RecommendationValidationError(
                    f"Row {idx}: Prerequisite violation! Content {cid} does not satisfy required prerequisites for learner {lid}."
                )


def validate_recommendation_dataset_dir(
    dir_path: Path,
    *,
    expected_profile: str | None = None,
    verify_manifest: bool = True,
    master_tables: Any | None = None,
    mastery_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate recommendations.csv in directory and optionally verify recommendation_manifest.json."""
    if not dir_path.is_dir():
        raise RecommendationValidationError(f"Directory not found: {dir_path}")

    csv_path = dir_path / "recommendations.csv"
    if not csv_path.is_file():
        raise RecommendationValidationError(f"recommendations.csv not found in {dir_path}")

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != RECOMMENDATION_COLUMNS:
            raise RecommendationValidationError(
                f"Columns in {csv_path.name} do not match schema. Expected {RECOMMENDATION_COLUMNS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    validate_recommendations(
        rows,
        master_tables=master_tables,
        mastery_snapshots=mastery_snapshots,
        expected_profile=expected_profile,
    )

    manifest_info = {}
    if verify_manifest:
        manifest_path = dir_path / RECOMMENDATION_MANIFEST_NAME
        if not manifest_path.is_file():
            raise RecommendationValidationError(f"Manifest {RECOMMENDATION_MANIFEST_NAME} not found in {dir_path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_info = json.load(fh)

        stored_checksums = manifest_info.get("checksums_sha256") or manifest_info.get("checksums", {})
        actual_checksum = _sha256(csv_path)
        expected_checksum = stored_checksums.get("recommendations.csv")
        if actual_checksum != expected_checksum:
            raise RecommendationValidationError(
                f"Checksum mismatch for recommendations.csv: expected {expected_checksum}, got {actual_checksum}"
            )

        if expected_profile:
            actual_profile = manifest_info.get("dataset_profile")
            if actual_profile != expected_profile:
                raise RecommendationValidationError(
                    f"Profile mismatch: expected {expected_profile}, manifest specifies {actual_profile}"
                )

    return {
        "directory": str(dir_path),
        "profile": manifest_info.get("dataset_profile") or expected_profile,
        "row_counts": {"recommendations.csv": len(rows)},
        "manifest": manifest_info,
    }

