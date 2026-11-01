"""ML Recommendation Extension package (Phase 8)."""

from learnpath_ai.ml.dataset import (
    DatasetSplit,
    MLTrainingExample,
    build_ml_dataset,
    chronological_split,
)
from learnpath_ai.ml.engine import generate_ml_recommendations
from learnpath_ai.ml.evaluate import evaluate_recommendation_strategies
from learnpath_ai.ml.model import ModelArtifact, train_model

__all__ = [
    "DatasetSplit",
    "MLTrainingExample",
    "ModelArtifact",
    "build_ml_dataset",
    "chronological_split",
    "evaluate_recommendation_strategies",
    "generate_ml_recommendations",
    "train_model",
]

