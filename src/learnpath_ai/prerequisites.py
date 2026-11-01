"""Skill prerequisite graph helpers.

Cycles cannot be expressed as a MySQL CHECK. Application code must reject them.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from collections.abc import Iterable, Sequence


class PrerequisiteGraphError(ValueError):
    """The prerequisite graph is not a DAG or contains an illegal self-edge."""


@dataclass(frozen=True)
class PrerequisiteEdge:
    skill_id: str
    prerequisite_skill_id: str
    relationship_type: str = "required"
    min_mastery_required: float = 0.6

    def is_required(self) -> bool:
        return self.relationship_type == "required"


def require_dag(edges: Iterable[PrerequisiteEdge]) -> None:
    cycle = find_cycle(list(edges))
    if cycle:
        raise PrerequisiteGraphError(
            "Skill prerequisite graph contains a cycle: " + " -> ".join(cycle)
        )


def find_cycle(edges: Sequence[PrerequisiteEdge]) -> list[str] | None:
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()

    for edge in edges:
        if edge.skill_id == edge.prerequisite_skill_id:
            raise PrerequisiteGraphError(
                f"Self-edge is not allowed: {edge.skill_id} -> {edge.skill_id}"
            )
        # Edge direction: prerequisite must be learned before skill.
        adjacency[edge.prerequisite_skill_id].append(edge.skill_id)
        indegree[edge.skill_id] += 1
        indegree.setdefault(edge.prerequisite_skill_id, 0)
        nodes.add(edge.skill_id)
        nodes.add(edge.prerequisite_skill_id)

    queue = deque(node for node in nodes if indegree[node] == 0)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in adjacency[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if seen == len(nodes):
        return None

    remaining = [node for node in nodes if indegree[node] > 0]
    if not remaining:
        return None
    start = remaining[0]
    cycle = [start]
    current = start
    remaining_set = set(remaining)
    for _ in range(len(remaining_set) + 1):
        nxts = [n for n in adjacency[current] if n in remaining_set]
        if not nxts:
            break
        current = nxts[0]
        cycle.append(current)
        if current == start:
            break
    return cycle
