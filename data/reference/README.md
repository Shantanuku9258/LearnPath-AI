# Reference Dataset (FULL Profile)

This directory contains the canonical, full-scale synthetic reference dataset for LearnPath AI. It represents a complete curriculum graph and learner population used for large-scale pipeline verification and out-of-sample evaluation.

## Profile Specifications

The FULL profile encompasses 40 courses and 10,000 learners, generating:
- **500,000 attempts** and **317,419 scored assessment results**.
- **638,203 temporal mastery snapshots**.
- **20,698 recommendations** and **9,502 active learning paths**.

## Included Entity Files

| CSV File | Record Count | Description |
|---|---|---|
| `courses.csv` | 40 | Catalog of courses across multiple domains and difficulty levels. |
| `learners.csv` | 10,000 | Student profiles, target difficulties, and enrollment statuses. |
| `skills.csv` | 500 | Distinct competencies defined across subject domains. |
| `skill_prerequisites.csv` | 1,441 | Directed Acyclic Graph (DAG) edges specifying required/recommended prerequisite mastery. |
| `content_items.csv` | 2,000 | Learning activities (videos, readings, exercises, projects). |
| `content_skills.csv` | 4,861 | Multi-skill mappings with primary/secondary roles and weights. |
| `attempts.csv` | 500,000 | Chronological interaction and attempt logs. |
| `assessment_results.csv` | 317,419 | Scored assessment outcomes linked 1:1 to scored attempts. |
| `mastery_snapshots.csv` | 638,203 | Point-in-time learner-skill mastery snapshots computed with recency decay. |
| `recommendations.csv` | 20,698 | Generated top-K activity recommendations with explainable reason codes. |
| `learning_paths.csv` | 9,502 | Structured sequential paths (at most 1 active path per learner/course). |
| `learning_path_items.csv` | 20,698 | Ordered activity steps comprising the generated learning paths. |

## Manifests & Cryptographic Checksums

All CSV files in this directory are governed by cryptographic SHA-256 manifests that record exact row counts and file hashes:
- `manifest.json` / `master_manifest.json`: Master entities (courses, learners, skills, prereqs, content).
- `activity_manifest.json`: Attempts and assessment results.
- `mastery_manifest.json`: Mastery snapshots.
- `recommendation_manifest.json`: Recommendations.
- `learning_path_manifest.json`: Learning paths and path items.

These manifests are verified programmatically via `python -m learnpath_ai validate <group> --profile full`.
