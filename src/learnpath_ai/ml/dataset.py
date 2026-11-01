r"""ML dataset construction and chronological splitting (Phase 8).

Target Definition:
------------------
We predict `success` on a learner-content interaction:
  `label = 1` if attempt outcome == 'pass' (or passed == 1)
  `label = 0` if attempt outcome in ('fail', 'incomplete') or abandoned.

Feature Construction (Point-in-Time, Zero Leakage):
---------------------------------------------------
For each attempt $i$ at time $T = \text{started\_at}$:
- All learner mastery features are calculated strictly using attempts with $\text{started\_at} < T$.
- Content difficulty from `content_items.csv`.
- Target difficulty alignment based on learner `starting_level`.
- Prior attempt counts and prior pass count on this content prior to $T$.
- Content-skill mapping weight for the primary skill.
- Learner overall historical pass rate prior to $T$.

Chronological Splitting:
------------------------
Dataset is sorted chronologically by $T$ and partitioned into:
- 70% Train
- 15% Validation
- 15% Test
Guaranteeing $T_{\text{train}} \le T_{\text{val}} \le T_{\text{test}}$, with 0 leakage tolerance.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learnpath_ai.ml.features import (
    FEATURE_NAMES,
    calculate_difficulty_suitability,
    calculate_features_from_history,
    parse_feature_timestamp,
)

_parse_ts = parse_feature_timestamp


@dataclass(frozen=True)
class MLTrainingExample:
    attempt_id: str
    learner_id: str
    content_id: str
    timestamp: str  # started_at
    primary_skill_mastery: float
    content_difficulty: float
    diff_suitability: float
    primary_skill_weight: float
    learner_prior_attempts_on_content: int
    learner_prior_passes_on_content: int
    learner_overall_prior_attempts: int
    learner_overall_prior_pass_rate: float
    label: int  # 1 = pass, 0 = fail / non-pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def feature_vector(self) -> list[float]:
        return [
            self.primary_skill_mastery,
            self.content_difficulty,
            self.diff_suitability,
            self.primary_skill_weight,
            float(self.learner_prior_attempts_on_content),
            float(self.learner_prior_passes_on_content),
            float(self.learner_overall_prior_attempts),
            self.learner_overall_prior_pass_rate,
        ]


@dataclass
class DatasetSplit:
    train: list[MLTrainingExample]
    val: list[MLTrainingExample]
    test: list[MLTrainingExample]


def build_ml_dataset(
    master_tables: Any,
    activity_tables: Any,
) -> list[MLTrainingExample]:
    """Construct point-in-time training examples from master and activity tables."""
    def _get(src: Any, key: str) -> list[dict[str, Any]]:
        if hasattr(src, key):
            return list(getattr(src, key))
        if isinstance(src, dict):
            return list(src.get(key, []))
        return []

    learners = {str(l["learner_id"]): l for l in _get(master_tables, "learners")}
    content_items = {str(c["content_id"]): c for c in _get(master_tables, "content_items")}
    content_skills = _get(master_tables, "content_skills")
    attempts = _get(activity_tables, "attempts")
    results = _get(activity_tables, "assessment_results")

    # Primary skill and weight map
    content_primary_skill: dict[str, str] = {}
    content_skill_weight: dict[str, float] = {}
    for cs in content_skills:
        cid = str(cs["content_id"])
        sid = str(cs["skill_id"])
        role = str(cs.get("skill_role", "primary"))
        weight = float(cs.get("weight", 1.0))
        if role == "primary" or cid not in content_primary_skill:
            content_primary_skill[cid] = sid
            content_skill_weight[cid] = weight

    # Map assessment results by attempt_id, preserving scored_at for temporal gating
    results_by_attempt: dict[str, dict[str, Any]] = {
        str(r["attempt_id"]): r for r in results
    }
    # Build scored_at map for strict temporal leakage prevention:
    # An outcome is only usable at time T if its scored_at <= T.
    result_scored_at: dict[str, datetime] = {}
    for r in results:
        aid = str(r["attempt_id"])
        raw_sat = r.get("scored_at")
        if raw_sat is not None and str(raw_sat).strip():
            result_scored_at[aid] = _parse_ts(raw_sat)

    # Difficulty targets by starting level
    level_target_diff = {
        "beginner": 1.5,
        "intermediate": 3.0,
        "advanced": 4.5,
    }

    # Chronologically sort all attempts
    sorted_attempts = sorted(
        attempts,
        key=lambda a: _parse_ts(a.get("started_at")),
    )

    examples: list[MLTrainingExample] = []

    # Per-learner running history list (tuples of attempt info) for strict gating.
    # Each entry: (attempt_id, started_at_dt, content_id, status, prim_skill, label, score_val)
    # We do NOT use incremental trackers; instead we scan prior_records at each T
    # and only include outcomes whose scored_at <= T (strict temporal gate).
    learner_records: dict[str, list[tuple[str, datetime, str, str, str, int, float]]] = {}

    for att in sorted_attempts:
        aid = str(att["attempt_id"])
        lid = str(att["learner_id"])
        cid = str(att["content_id"])
        status = str(att.get("status", "")).lower()

        outcome = ""
        passed_val = 0
        score_val = 0.0

        res = results_by_attempt.get(aid)
        is_training_target = False
        if res or status == "scored":
            is_training_target = True
            outcome = str(res.get("outcome", "")).lower() if res else ""
            passed_val = int(res.get("passed", 0)) if res else 0
            score_val = float(res.get("score", 0.0)) if (res and res.get("score") is not None and res.get("score") != "") else 0.0

        label = 1 if (outcome == "pass" or passed_val == 1) else 0

        # Features calculated strictly PRIOR to this attempt start time T:
        T = _parse_ts(att["started_at"])

        if is_training_target:
            learner_info = learners.get(lid, {})
            start_level = str(learner_info.get("starting_level", "beginner")).lower()

            content_info = content_items.get(cid, {})
            c_diff = float(content_info.get("difficulty", 2.5))
            diff_suit = calculate_difficulty_suitability(c_diff, start_level)

            prim_skill = content_primary_skill.get(cid, "")
            prim_weight = content_skill_weight.get(cid, 1.0)

            # --- Strict scored_at and started_at gating ---
            prior_records = learner_records.get(lid, [])

            primary_skill_evidence: list[dict[str, Any]] = []
            prior_c_attempts = 0
            prior_c_passes = 0
            prior_all_attempts = 0
            prior_all_scored = 0
            prior_all_passes = 0

            for p_aid, p_start, p_cid, p_status, p_skill, p_label, p_score in prior_records:
                # Behavioral attempt counts: valid attempts started strictly prior to T
                if p_status != "invalid":
                    if p_cid == cid:
                        prior_c_attempts += 1
                    prior_all_attempts += 1

                # Scored outcomes and mastery evidence: require scored_at <= T
                p_scored_at = result_scored_at.get(p_aid)
                outcome_available = p_scored_at is not None and p_scored_at <= T

                if outcome_available and p_status == "scored":
                    prior_all_scored += 1
                    prior_all_passes += p_label
                    if p_cid == cid:
                        prior_c_passes += p_label

                    # Phase 5 mastery evidence for primary skill
                    if p_skill == prim_skill:
                        norm_score = max(0.0, min(1.0, p_score / 100.0))
                        primary_skill_evidence.append({
                            "attempt_id": p_aid,
                            "event_time": p_scored_at,
                            "normalized_score": norm_score,
                            "weight": prim_weight,
                        })

            features = calculate_features_from_history(
                primary_skill_evidence=primary_skill_evidence,
                content_difficulty=c_diff,
                diff_suitability=diff_suit,
                primary_skill_weight=prim_weight,
                prior_attempts_on_content=prior_c_attempts,
                prior_passes_on_content=prior_c_passes,
                overall_prior_attempts=prior_all_attempts,
                overall_scored_attempts=prior_all_scored,
                overall_prior_passes=prior_all_passes,
                as_of_dt=T,
            )

            ex = MLTrainingExample(
                attempt_id=aid,
                learner_id=lid,
                content_id=cid,
                timestamp=str(att["started_at"]),
                primary_skill_mastery=features[0],
                content_difficulty=features[1],
                diff_suitability=features[2],
                primary_skill_weight=features[3],
                learner_prior_attempts_on_content=int(features[4]),
                learner_prior_passes_on_content=int(features[5]),
                learner_overall_prior_attempts=int(features[6]),
                learner_overall_prior_pass_rate=features[7],
                label=label,
            )
            examples.append(ex)

        # Append current attempt record for subsequent attempts to use
        learner_records.setdefault(lid, []).append(
            (aid, T, cid, status, content_primary_skill.get(cid, ""), label, score_val)
        )

    return examples


def chronological_split(
    examples: list[MLTrainingExample],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetSplit:
    """Chronologically split examples into Train, Validation, and Test sets."""
    if not examples:
        return DatasetSplit(train=[], val=[], test=[])

    sorted_ex = sorted(examples, key=lambda e: _parse_ts(e.timestamp))
    n = len(sorted_ex)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = sorted_ex[:train_end]
    val = sorted_ex[train_end:val_end]
    test = sorted_ex[val_end:]

    return DatasetSplit(train=train, val=val, test=test)


def write_dataset_to_csv(examples: list[MLTrainingExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        return
    fieldnames = list(examples[0].to_dict().keys())
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for ex in examples:
            writer.writerow(ex.to_dict())
