# Sample Dataset (DEV Profile)

This directory contains the compact development sample dataset for LearnPath AI. It is designed for fast local iteration, integration testing, demonstration, and continuous integration environments where processing the full 500,000-attempt reference dataset is impractical.

## Profile Specifications

The DEV profile provides a complete, mathematically coherent curriculum graph:
- **8 courses** and **100 learners**.
- **50 skills** and **94 prerequisite edges**.
- **100 content items** and **213 skill mappings**.
- **5,000 attempts** and **3,112 scored assessment results**.
- **920 temporal mastery snapshots**.
- **348 recommendations** and **97 active learning paths**.

## Included Entity Files

| CSV File | Record Count | Description |
|---|---|---|
| `courses.csv` | 8 | Development course catalog across introductory and intermediate tiers. |
| `learners.csv` | 100 | Sample cohort representing diverse initial proficiencies and cold-start states. |
| `skills.csv` | 50 | Foundational and advanced competencies. |
| `skill_prerequisites.csv` | 94 | Prerequisite relationships verified for acyclicity via `require_dag`. |
| `content_items.csv` | 100 | Diverse activity formats across difficulty levels 1 to 5. |
| `content_skills.csv` | 213 | Multi-skill mappings with primary/secondary weightings. |
| `attempts.csv` | 5,000 | Realistic activity events including cold-start and repeated attempts. |
| `assessment_results.csv` | 3,112 | Scored outcomes verified for score reconciliation against attempts. |
| `mastery_snapshots.csv` | 920 | Time-decayed mastery snapshots subject to point-in-time cutoffs. |
| `recommendations.csv` | 348 | Rule-based recommendations (`rule-v1`) with explainability tags. |
| `learning_paths.csv` | 97 | Personalized sequential paths (single active path constraint enforced). |
| `learning_path_items.csv` | 348 | Sequenced learning steps mapped to recommended content items. |

## Manifests & Verification

Every file in this directory is cataloged in cryptographic SHA-256 manifests:
- `manifest.json` / `master_manifest.json`
- `activity_manifest.json`
- `mastery_manifest.json`
- `recommendation_manifest.json`
- `learning_path_manifest.json`

Verify dataset integrity using `python -m learnpath_ai validate <group> --profile dev`.
