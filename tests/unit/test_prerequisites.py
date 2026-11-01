from __future__ import annotations

import pytest

from learnpath_ai.prerequisites import (
    PrerequisiteEdge,
    PrerequisiteGraphError,
    find_cycle,
    require_dag,
)


def test_dag_is_accepted() -> None:
    edges = [
        PrerequisiteEdge("SKL-000002", "SKL-000001"),
        PrerequisiteEdge("SKL-000003", "SKL-000002"),
    ]
    require_dag(edges)
    assert find_cycle(edges) is None


def test_cycle_is_rejected() -> None:
    edges = [
        PrerequisiteEdge("SKL-000002", "SKL-000001"),
        PrerequisiteEdge("SKL-000001", "SKL-000002"),
    ]
    with pytest.raises(PrerequisiteGraphError, match="cycle"):
        require_dag(edges)


def test_self_edge_is_rejected() -> None:
    with pytest.raises(PrerequisiteGraphError, match="Self-edge"):
        require_dag([PrerequisiteEdge("SKL-000001", "SKL-000001")])
