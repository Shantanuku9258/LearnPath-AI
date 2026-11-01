"""Evaluation and fair comparison suite: rule-v1 vs ml-v1 (Phase 8).

Implements:
- Precision@K
- Recall@K
- NDCG@K
- Coverage (learners covered with recommendations)
- Cold-start comparison
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from learnpath_ai.recommend.engine import RecommendationRecord


@dataclass
class StrategyMetrics:
    strategy_name: str
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    coverage: float
    total_learners: int
    covered_learners: int
    cold_start_coverage: float


def compute_dcg(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(ranked_items: list[str], relevant_set: set[str], k: int) -> float:
    relevances = [1.0 if item in relevant_set else 0.0 for item in ranked_items[:k]]
    actual_dcg = compute_dcg(relevances, k)
    # Ideal DCG has all 1s first
    ideal_relevances = [1.0] * min(k, len(relevant_set))
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_recommendations_for_test_slice(
    recommendations: list[RecommendationRecord],
    ground_truth_test_attempts: list[dict[str, Any]],
    eligible_learners: list[str],
    cold_start_learners: list[str],
    *,
    k_list: list[int] = (1, 3, 5),
    strategy_name: str = "strategy",
) -> StrategyMetrics:
    """Evaluate a set of recommendations against observed future successes in the test window."""
    # Build ground truth: learner_id -> set of content_ids passed in test window
    relevant_by_learner: dict[str, set[str]] = {}
    for att in ground_truth_test_attempts:
        lid = str(att["learner_id"])
        cid = str(att["content_id"])
        passed = int(att.get("passed", 0)) == 1 or str(att.get("outcome", "")).lower() == "pass"
        if passed:
            relevant_by_learner.setdefault(lid, set()).add(cid)

    # Group recommendations by learner
    recs_by_learner: dict[str, list[str]] = {}
    for r in sorted(recommendations, key=lambda x: x.rank_position):
        recs_by_learner.setdefault(r.learner_id, []).append(r.content_id)

    # Coverage
    total_learners = len(eligible_learners)
    covered_learners = sum(1 for lid in eligible_learners if lid in recs_by_learner and recs_by_learner[lid])
    coverage = covered_learners / total_learners if total_learners > 0 else 0.0

    # Cold start coverage
    cold_start_set = set(cold_start_learners)
    cs_covered = sum(1 for lid in cold_start_learners if lid in recs_by_learner and recs_by_learner[lid])
    cs_coverage = cs_covered / len(cold_start_learners) if cold_start_learners else 1.0

    precisions: dict[int, list[float]] = {k: [] for k in k_list}
    recalls: dict[int, list[float]] = {k: [] for k in k_list}
    ndcgs: dict[int, list[float]] = {k: [] for k in k_list}

    # Evaluate across learners with observed relevant items in test slice
    for lid, rel_set in relevant_by_learner.items():
        if not rel_set:
            continue
        ranked = recs_by_learner.get(lid, [])

        for k in k_list:
            top_k = ranked[:k]
            hits = sum(1 for c in top_k if c in rel_set)

            p_k = hits / k if k > 0 else 0.0
            r_k = hits / len(rel_set) if len(rel_set) > 0 else 0.0
            ndcg_k = compute_ndcg(ranked, rel_set, k)

            precisions[k].append(p_k)
            recalls[k].append(r_k)
            ndcgs[k].append(ndcg_k)

    avg_precision = {
        k: (sum(precisions[k]) / len(precisions[k])) if precisions[k] else 0.0
        for k in k_list
    }
    avg_recall = {
        k: (sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0
        for k in k_list
    }
    avg_ndcg = {
        k: (sum(ndcgs[k]) / len(ndcgs[k])) if ndcgs[k] else 0.0
        for k in k_list
    }

    return StrategyMetrics(
        strategy_name=strategy_name,
        precision_at_k=avg_precision,
        recall_at_k=avg_recall,
        ndcg_at_k=avg_ndcg,
        coverage=round(coverage, 4),
        total_learners=total_learners,
        covered_learners=covered_learners,
        cold_start_coverage=round(cs_coverage, 4),
    )


def evaluate_recommendation_strategies(
    rule_recs: list[RecommendationRecord],
    ml_recs: list[RecommendationRecord],
    ground_truth_test_attempts: list[dict[str, Any]],
    eligible_learners: list[str],
    cold_start_learners: list[str],
    *,
    k_list: list[int] = (1, 3, 5),
) -> dict[str, StrategyMetrics]:
    rule_metrics = evaluate_recommendations_for_test_slice(
        rule_recs,
        ground_truth_test_attempts,
        eligible_learners,
        cold_start_learners,
        k_list=k_list,
        strategy_name="rule-v1",
    )
    ml_metrics = evaluate_recommendations_for_test_slice(
        ml_recs,
        ground_truth_test_attempts,
        eligible_learners,
        cold_start_learners,
        k_list=k_list,
        strategy_name="ml-v1",
    )
    return {
        "rule-v1": rule_metrics,
        "ml-v1": ml_metrics,
    }

