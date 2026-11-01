"""Model training and serialization pipeline (Phase 8)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from learnpath_ai.ml.dataset import FEATURE_NAMES, DatasetSplit, MLTrainingExample

UTC = timezone.utc
MODEL_VERSION = "ml-v1"


@dataclass
class ModelArtifact:
    model: LogisticRegression
    feature_names: list[str]
    model_version: str
    trained_at: str
    metrics: dict[str, float]
    coefficients: dict[str, float]
    intercept: float

    def predict_proba(self, feature_vectors: list[list[float]]) -> list[float]:
        if not feature_vectors:
            return []
        X = np.array(feature_vectors)
        probs = self.model.predict_proba(X)[:, 1]
        return [float(p) for p in probs]

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f"{self.model_version}.pkl"
        meta_file = model_dir / f"{self.model_version}_meta.json"

        with model_file.open("wb") as fh:
            pickle.dump(self.model, fh)

        meta = {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
        }
        with meta_file.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        return model_file

    @classmethod
    def load(cls, model_dir: Path, version: str = MODEL_VERSION) -> ModelArtifact:
        model_file = model_dir / f"{version}.pkl"
        meta_file = model_dir / f"{version}_meta.json"

        if not model_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(f"Model artifacts not found in {model_dir} for {version}")

        with model_file.open("rb") as fh:
            model = pickle.load(fh)
        with meta_file.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        return cls(
            model=model,
            feature_names=meta["feature_names"],
            model_version=meta["model_version"],
            trained_at=meta["trained_at"],
            metrics=meta["metrics"],
            coefficients=meta["coefficients"],
            intercept=meta["intercept"],
        )


def train_model(
    split: DatasetSplit,
    *,
    seed: int = 20261022,
    model_version: str = MODEL_VERSION,
) -> ModelArtifact:
    """Train explainable Logistic Regression model on chronological split."""
    if not split.train:
        raise ValueError("Cannot train model on empty training set.")

    X_train = np.array([ex.feature_vector() for ex in split.train])
    y_train = np.array([ex.label for ex in split.train])

    # Check class diversity
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training dataset has only one class.")

    model = LogisticRegression(
        C=1.0,
        random_state=seed,
        max_iter=500,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    # Validation evaluation
    val_metrics: dict[str, float] = {}
    if split.val:
        X_val = np.array([ex.feature_vector() for ex in split.val])
        y_val = np.array([ex.label for ex in split.val])
        y_probs = model.predict_proba(X_val)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        val_metrics["accuracy"] = float(accuracy_score(y_val, y_preds))
        val_metrics["precision"] = float(precision_score(y_val, y_preds, zero_division=0))
        val_metrics["recall"] = float(recall_score(y_val, y_preds, zero_division=0))
        val_metrics["f1"] = float(f1_score(y_val, y_preds, zero_division=0))
        try:
            val_metrics["roc_auc"] = float(roc_auc_score(y_val, y_probs))
        except Exception:
            val_metrics["roc_auc"] = 0.5
        try:
            val_metrics["log_loss"] = float(log_loss(y_val, y_probs, labels=[0, 1]))
        except Exception:
            val_metrics["log_loss"] = 0.6931

    # Feature coefficients mapping
    coeffs = {
        name: float(coef)
        for name, coef in zip(FEATURE_NAMES, model.coef_[0])
    }
    intercept = float(model.intercept_[0])

    artifact = ModelArtifact(
        model=model,
        feature_names=list(FEATURE_NAMES),
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
        metrics=val_metrics,
        coefficients=coeffs,
        intercept=intercept,
    )
    return artifact
