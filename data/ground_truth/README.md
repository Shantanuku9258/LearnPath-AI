# Ground Truth Data

This directory contains benchmark ground-truth datasets used to validate anomaly detection and quarantine pipelines.

## Contents

- `ground_truth_anomalies.csv`: Canonical record of 25 deterministically injected anomalies (5 per scenario) created during development quality audits.

## Schema

`ground_truth_anomalies.csv` adheres to the following structure:

| Column | Type | Description |
|---|---|---|
| `issue_id` | String | Unique anomaly identifier (e.g., `DQI-000001`). |
| `issue_type` | String | The specific anomaly category classification. |
| `source_table` | String | Originating entity table (e.g., `recommendation`, `attempt`, `content_skill`). |
| `source_pk` | String | Primary key of the corrupted record. |
| `expected_decision` | String | Ground-truth decision label (`reject` for all injected anomalies). |

## Supported Anomaly Categories

1. **`REC_PREREQ_VIOLATION`**: Synthetic recommendations for content whose required prerequisite skills are not satisfied by learner mastery.
2. **`ATTEMPT_TIMESTAMP_ORDER`**: Chronologically invalid attempts where `submitted_at` precedes `started_at`.
3. **`CONTENT_MISSING_SKILL`**: Content mappings referencing nonexistent skill identifiers (`SKL-999999`).
4. **`MASTERY_FUTURE_ATTEMPT`**: Temporal leakage where attempt timestamps occur after the snapshot `as_of_timestamp`.
5. **`PATH_LOOP_NO_PROGRESS`**: Repetitive learning path loops where identical content items are assigned without score improvement.

## Usage Rule

> **Important:** Ground-truth labels stored in this directory are exclusively used by the test suite and evaluation scripts to verify anomaly detection precision and recall. They are **strictly prohibited** from being exposed to or ingested by recommendation algorithms or machine learning feature pipelines as training inputs.
