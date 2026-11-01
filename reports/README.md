# Pipeline Reports & Evaluation Artifacts

This directory stores evaluation summaries, benchmark metrics, data quality logs, and integrity audits generated across project phases.

## Existing Historical Reports

The following verified reports are preserved in this directory as official project benchmarks:

### 1. Strategy Evaluation Reports
- **`evaluation_f03_dev.txt`**: Canonical out-of-sample chronological evaluation report for the **DEV profile** (467 test attempts observed after cutoff `2026-09-03`). Compares `rule-v1` against `ml-v1` on Coverage, Precision@K, Recall@K, and NDCG@K ($K \in \{1, 3, 5\}$).
- **`evaluation_f03_full.txt`**: Canonical zero-shot transfer evaluation report on the **FULL profile** (47,613 test attempts observed after cutoff `2026-09-16`). Evaluates the DEV-trained model on 9,502 eligible learners and 1,452 cold-start learners.
- **`recommendation_metrics.json`**: Machine-readable JSON summary of recommendation engine performance metrics and coverage statistics.

### 2. Data Quality & Anomaly Reports
- **`quality_summary_dev.txt`**: Human-readable text summary of anomaly detection and quarantine results for the DEV profile.
- **`quality_report_dev.json`**: Structured JSON report documenting detected anomaly counts, issue types, and detection rates.
- **`quality_report_dev.csv`**: Tabular export of data quality issues cataloged by entity and severity.

### 3. Pipeline Integrity Reports
- **`integrity_report.json`**: Verification report auditing cryptographic SHA-256 checksums, manifest integrity, and foreign-key consistency across generated profiles.

## Runtime Report Generation

When CLI evaluation commands are executed:
- `python -m learnpath_ai evaluate compare --profile dev`
- `python -m learnpath_ai evaluate compare --profile full`

The commands print live evaluation tables directly to stdout to enable non-destructive comparison against the canonical historical reports (`evaluation_f03_dev.txt` and `evaluation_f03_full.txt`) without automatically overwriting stored benchmarks.
