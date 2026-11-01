"""Recommendation package (Phase 6).

Provides deterministic, explainable rule-based learning activity recommendation
subject to strict prerequisite gating and temporal cutoff constraints.
"""

from __future__ import annotations

from learnpath_ai.recommend.engine import (
    RecommendationError,
    RecommendationRecord,
    generate_recommendations,
)
from learnpath_ai.recommend.validate import (
    RecommendationValidationError,
    validate_recommendation_dataset_dir,
    validate_recommendations,
)
from learnpath_ai.recommend.writer import (
    write_recommendation_dataset,
    write_recommendation_manifest,
)

__all__ = [
    "RecommendationError",
    "RecommendationRecord",
    "RecommendationValidationError",
    "generate_recommendations",
    "validate_recommendation_dataset_dir",
    "validate_recommendations",
    "write_recommendation_dataset",
    "write_recommendation_manifest",
]

