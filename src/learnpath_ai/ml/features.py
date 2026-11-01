"""Shared ML feature extraction definitions and functions (Phase 8 - F-10).

Ensures 100% strict parity between ML dataset generation (training) and ML recommendation
generation (inference).

Features (in exact order):
1. primary_skill_mastery: Phase 5 mastery logic (decay, content-skill weight, credibility shrinkage)
2. content_difficulty: content item difficulty rating
3. diff_suitability: suitability based on learner starting level
4. primary_skill_weight: weight of primary skill mapping
5. learner_prior_attempts_on_content: prior non-invalid attempts on this content item
6. learner_prior_passes_on_content: prior scored passes on this content item
7. learner_overall_prior_attempts: prior non-invalid attempts across all content
8. learner_overall_prior_pass_rate: overall passes / scored attempts (or 0.5 default)
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from learnpath_ai.generate.clock import parse_iso_utc
from learnpath_ai.mastery.calculator import calculate_single_mastery

UTC = timezone.utc

FEATURE_NAMES: list[str] = [
    "primary_skill_mastery",
    "content_difficulty",
    "diff_suitability",
    "primary_skill_weight",
    "learner_prior_attempts_on_content",
    "learner_prior_passes_on_content",
    "learner_overall_prior_attempts",
    "learner_overall_prior_pass_rate",
]

LEVEL_TARGET_DIFF: dict[str, float] = {
    "beginner": 1.5,
    "intermediate": 3.0,
    "advanced": 4.5,
}


def parse_feature_timestamp(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    return parse_iso_utc(str(val).strip())


def calculate_difficulty_suitability(
    content_difficulty: float,
    learner_starting_level: str | None,
) -> float:
    """Calculate difficulty suitability score between 0.0 and 1.0."""
    level = str(learner_starting_level or "beginner").strip().lower()
    target_diff = LEVEL_TARGET_DIFF.get(level, 2.5)
    return max(0.0, 1.0 - abs(content_difficulty - target_diff) / 4.0)


def extract_eight_features(
    *,
    primary_skill_mastery: float,
    content_difficulty: float,
    diff_suitability: float,
    primary_skill_weight: float,
    prior_attempts_on_content: int,
    prior_passes_on_content: int,
    overall_prior_attempts: int,
    overall_prior_pass_rate: float,
) -> list[float]:
    """Assemble the eight-feature vector with exact ordering, typing, and standard rounding."""
    return [
        round(float(primary_skill_mastery), 4),
        round(float(content_difficulty), 2),
        round(float(diff_suitability), 4),
        round(float(primary_skill_weight), 4),
        float(prior_attempts_on_content),
        float(prior_passes_on_content),
        float(overall_prior_attempts),
        round(float(overall_prior_pass_rate), 4),
    ]


def calculate_features_from_history(
    *,
    primary_skill_evidence: list[dict[str, Any]],
    content_difficulty: float,
    diff_suitability: float,
    primary_skill_weight: float,
    prior_attempts_on_content: int,
    prior_passes_on_content: int,
    overall_prior_attempts: int,
    overall_scored_attempts: int,
    overall_prior_passes: int,
    as_of_dt: datetime,
) -> list[float]:
    """Calculate the complete 8-feature vector using history known at as_of_dt.
    
    Reuses Phase 5 calculate_single_mastery for primary_skill_mastery.
    """
    mastery_score, _, _ = calculate_single_mastery(
        primary_skill_evidence,
        as_of_dt,
        half_life_days=90.0,
        credibility_k=1.0,
        scale_min=0.0,
        scale_max=1.0,
    )
    
    pass_rate = (
        (overall_prior_passes / overall_scored_attempts)
        if overall_scored_attempts > 0
        else 0.5
    )

    return extract_eight_features(
        primary_skill_mastery=mastery_score,
        content_difficulty=content_difficulty,
        diff_suitability=diff_suitability,
        primary_skill_weight=primary_skill_weight,
        prior_attempts_on_content=prior_attempts_on_content,
        prior_passes_on_content=prior_passes_on_content,
        overall_prior_attempts=overall_prior_attempts,
        overall_prior_pass_rate=pass_rate,
    )

