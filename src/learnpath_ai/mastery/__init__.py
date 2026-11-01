"""Mastery estimation package (Phase 5).

Provides deterministic, explainable learner-skill mastery estimation
subject to temporal cutoff constraints.
"""

from __future__ import annotations

from learnpath_ai.mastery.calculator import (
    MasteryCalculationError,
    MasterySnapshotRecord,
    calculate_mastery_snapshots,
)
from learnpath_ai.mastery.validate import (
    MasteryValidationError,
    validate_mastery_dataset_dir,
    validate_mastery_snapshots,
)
from learnpath_ai.mastery.writer import (
    write_mastery_dataset,
    write_mastery_manifest,
)

__all__ = [
    "MasteryCalculationError",
    "MasterySnapshotRecord",
    "MasteryValidationError",
    "calculate_mastery_snapshots",
    "validate_mastery_dataset_dir",
    "validate_mastery_snapshots",
    "write_mastery_dataset",
    "write_mastery_manifest",
]

