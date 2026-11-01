# Evaluation methodology — LearnPath AI

No metrics have been computed yet. This document defines **how** they will be measured. Do not treat planned formulas as completed results.

## 1. Principles

1. **Chronological splits.** Order by `as_of_timestamp` / `recommended_at`. Default 70% train / 15% validation / 15% test by time, not by shuffled rows.
2. **No leakage.** Features for a recommendation at time `T` may use only records with event time `≤ T`. Ground-truth anomaly labels are never features.
3. **Simulated ≠ real.** Scores, completions, and “improvement” flags are generated. They support engineering evaluation, not claims about live learners.
4. **Reproducibility.** Same seed + profile ⇒ same IDs, counts, and checksums.

## 2. Integrity and reconciliation

| Check | Method | Success |
|---|---|---|
| Primary-key uniqueness | Count vs `COUNT(DISTINCT pk)` per table | Equal |
| Foreign-key integrity | Anti-join orphans; MySQL FK load without error | Zero orphans |
| ID format | Regex per entity | 100% match |
| Referential load order | Masters then events | Load transaction commits |
| Reconciliation | Generated CSV row counts vs MySQL `COUNT(*)` | Exact match |
| Checksums | SHA-256 of canonical sorted CSV | Match across reruns with same seed |

Reports will write actual counts; Phase 0 invents none.

## 3. Anomaly detection

Five required types:

| Code | Detection (planned) | Expected decision |
|---|---|---|
| `REC_PREREQ_VIOLATION` | Recommended content’s required skills have unmet `min_mastery_required` at `as_of_timestamp` | Quarantine recommendation; do not serve |
| `ATTEMPT_TIMESTAMP_ORDER` | `submitted_at < started_at`, or later `attempt_number` has earlier start | Quarantine attempt; exclude from mastery |
| `CONTENT_MISSING_SKILL` | Content–skill row points at unknown `skill_id`, or published content has no skills | Quarantine mapping / content |
| `MASTERY_FUTURE_ATTEMPT` | Snapshot evidence includes attempt with timestamp `> as_of_timestamp` | Reject snapshot; recompute |
| `PATH_LOOP_NO_PROGRESS` | Path repeats a content item (or cycle of items) with no score/mastery increase over N visits | Flag path; break cycle in repair policy |

**Detection rate** (after labels exist):

```text
recall    = TP / (TP + FN)   # of injected issues found
precision = TP / (TP + FP)
```

`ground_truth` is the only label source. Detectors read operational tables, not the label file.

**Prerequisite violation rate** (served recommendations):

```text
violations / recommendations_with_status_not_quarantined
```

Baseline generator should target **zero** served violations; injected anomalies should be quarantined.

## 4. Recommendation quality

Universe at time `T`: published content in the learner’s course that is eligible under prerequisites and not retired.

| Metric | Definition |
|---|---|
| Coverage | Learners with ≥1 valid recommendation / eligible learners |
| Reason-code coverage | Recommendations with ≥1 known reason code / all recommendations |
| Precision@K | Share of top-K items that are relevant in the outcome window |
| Recall@K | Share of relevant items that appear in the top-K |

**Relevance (simulated), in priority order:**

1. **Primary — skill improvement:** `recommendation_outcome.improved = 1` (or a positive `score_delta` on a mapped primary skill) inside `window_hours`.
2. **Secondary — successful completion:** `recommendation_outcome.completed = 1` in the same window.

An item that improved skill is relevant even if it was not marked completed. Completion without improvement is a weaker relevant outcome. These flags are synthetic proxies, not classroom evidence.

`K` defaults to 1, 3, 5 (`config/default.json`).

Compare `rule-v1` vs `ml-v1` on the **same** chronological test slice. Report both; do not declare a winner until numbers exist.

## 5. Temporal leakage checks

Automated tests will fail the run if any of these hold:

- Feature row joins an attempt with `submitted_at > as_of_timestamp`.
- Train split contains a recommendation with `as_of_timestamp` after the earliest test `as_of_timestamp` (overlapping time if using a strict cutoff — the cutoff is a single timestamp `T_split`).
- Model features include columns from `data/ground_truth` or `data_quality_issue.expected_decision`.
- Mastery `evidence_cutoff_ts > as_of_timestamp`.

`leakage_tolerance_seconds` is 0.

## 6. Runtime statistics

Log wall time and peak memory (where available) for:

- generation
- validation
- MySQL load
- mastery
- ranking (`rule-v1`, `ml-v1`)
- evaluation

Full profile target scale: 10,000 learners, 500 skills, 2,000 content items, 500,000 attempts. No runtime numbers are recorded yet.

## 7. Report artifacts (planned)

Written under `reports/` after a successful pipeline:

- `integrity_report.json`
- `anomaly_report.json`
- `recommendation_metrics.json`
- `leakage_audit.json`
- `runtime_stats.json`
- `reproducibility.json` (counts + checksums)

## 8. What this evaluation is not

It is not a randomized controlled trial, not a claim of learning gain in production, and not a substitute for human review of educational content.
