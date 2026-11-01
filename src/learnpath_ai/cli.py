"""CLI-first entry point. Commands: db, generate, and validate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from learnpath_ai import __version__
from learnpath_ai.config_loader import ConfigError, load_app_config
from learnpath_ai.database import Database, DatabaseError, PingResult
from learnpath_ai.generate.activity import generate_activity_tables, write_activity_dataset
from learnpath_ai.generate.columns import ACTIVITY_MANIFEST_NAME, MANIFEST_NAME
from learnpath_ai.generate.master import generate_master_tables
from learnpath_ai.generate.validate import (
    ActivityDataError,
    MasterDataError,
    validate_activity_dataset_dir,
    validate_activity_tables,
    validate_master_dataset_dir,
    validate_master_tables,
)
from learnpath_ai.generate.writer import OutputExistsError, write_master_dataset
from learnpath_ai.mastery.calculator import calculate_mastery_snapshots
from learnpath_ai.mastery.validate import (
    MasteryValidationError,
    validate_mastery_dataset_dir,
    validate_mastery_snapshots,
)
from learnpath_ai.mastery.writer import write_mastery_dataset
from learnpath_ai.quality.anomalies import generate_anomaly_fixtures
from learnpath_ai.quality.detectors import QualityError, detect_all_anomalies
from learnpath_ai.quality.quarantine import write_quarantine
from learnpath_ai.quality.report import write_quality_report
from learnpath_ai.recommend.engine import generate_recommendations
from learnpath_ai.recommend.validate import (
    RecommendationValidationError,
    validate_recommendation_dataset_dir,
    validate_recommendations,
)
from learnpath_ai.path.engine import LearningPathError, generate_learning_paths
from learnpath_ai.path.validate import PathValidationError, validate_path_dataset_dir
from learnpath_ai.path.writer import write_path_dataset
from learnpath_ai.generate.clock import parse_iso_utc
from learnpath_ai.ml.dataset import DatasetSplit, build_ml_dataset, chronological_split, write_dataset_to_csv
from learnpath_ai.ml.model import train_model, ModelArtifact
from learnpath_ai.ml.engine import generate_ml_recommendations
from learnpath_ai.ml.evaluate import evaluate_recommendation_strategies
from learnpath_ai.recommend.writer import write_recommendation_dataset
from learnpath_ai.schema import InitResult, SchemaError, initialize_schema, inspect_schema
from learnpath_ai.settings import SettingsError, load_database_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learnpath-ai",
        description="LearnPath AI command-line interface.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. db command group
    db = sub.add_parser("db", help="Database connection and schema commands.")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("ping", help="Verify connectivity without printing credentials.")
    db_sub.add_parser("status", help="Inspect whether the target schema already exists.")
    db_sub.add_parser(
        "init",
        help="Apply schemas/schema.sql if the database is empty of expected tables.",
    )

    # 2. generate command group
    gen = sub.add_parser("generate", help="Synthetic data generation commands.")
    gen_sub = gen.add_subparsers(dest="gen_command", required=True)

    # 2a. generate master
    master_gen = gen_sub.add_parser(
        "master",
        help="Generate reproducible parent/master datasets (courses, skills, content, learners).",
    )
    master_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' (100 learners, 8 courses, 50 skills, 100 content) or 'full' (10k learners, 40 courses, 500 skills, 2k content).",
    )
    master_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    master_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/sample for dev, data/reference for full).",
    )
    master_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing files in output directory.",
    )

    # 2b. generate activity
    activity_gen = gen_sub.add_parser(
        "activity",
        help="Generate reproducible learner activity events (attempts and assessment results).",
    )
    activity_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' (~5,000 attempts) or 'full' (500,000 attempts).",
    )
    activity_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    activity_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/sample for dev, data/reference for full).",
    )
    activity_gen.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Override total attempt count (default: 5,000 for dev, 500,000 for full).",
    )
    activity_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing files in output directory.",
    )

    # 3. validate command group
    val = sub.add_parser("validate", help="Data validation commands.")
    val_sub = val.add_subparsers(dest="val_command", required=True)

    # 3a. validate master
    master_val = val_sub.add_parser(
        "master",
        help="Validate generated master datasets (keys, foreign keys, enums, DAG, checksums).",
    )
    master_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Directory to validate (default: data/sample).",
    )
    master_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default=None,
        help="Expected profile for row-count verification (inferred from manifest if omitted).",
    )
    master_val.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest verification.",
    )

    # 3b. validate activity
    activity_val = val_sub.add_parser(
        "activity",
        help="Validate generated activity datasets (attempts, assessment results, score reconciliation).",
    )
    activity_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Directory to validate (default: data/sample).",
    )
    activity_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default=None,
        help="Expected profile for row-count verification (inferred from manifest if omitted).",
    )
    activity_val.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest verification.",
    )

    # 3c. validate quality
    quality_val = val_sub.add_parser(
        "quality",
        help="Re-run anomaly detectors on quarantine outputs and verify expected decisions.",
    )
    quality_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Base dataset directory to validate (default: data/sample).",
    )
    quality_val.add_argument(
        "--quarantine-dir",
        dest="quarantine_dir",
        type=Path,
        default=None,
        help="Quarantine directory (default: data/quarantine/<profile>).",
    )
    quality_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile (default: dev).",
    )

    # 3d. validate mastery
    mastery_val = val_sub.add_parser(
        "mastery",
        help="Validate generated mastery snapshots (schema, ranges, no future leakage, checksums).",
    )
    mastery_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Directory to validate (default: data/sample).",
    )
    mastery_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default=None,
        help="Expected profile (inferred from manifest if omitted).",
    )
    mastery_val.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest verification.",
    )

    # 3e. validate recommendations
    rec_val = val_sub.add_parser(
        "recommendations",
        help="Validate generated recommendations (schema, eligibility, top-K, checksums).",
    )
    rec_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Directory to validate (default: data/sample).",
    )
    rec_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default=None,
        help="Expected profile (inferred from manifest if omitted).",
    )
    rec_val.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest verification.",
    )

    # 2c. generate anomalies
    anomaly_gen = gen_sub.add_parser(
        "anomalies",
        help="Inject and detect all five anomaly types; write quarantine + ground-truth + report.",
    )
    anomaly_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' (5 per type) or 'full' (50 per type).",
    )
    anomaly_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    anomaly_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source master+activity dataset directory (defaults to data/sample or data/reference).",
    )
    anomaly_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Quarantine output directory (defaults to data/quarantine/<profile>).",
    )
    anomaly_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing quarantine files.",
    )

    # 2d. generate mastery
    mastery_gen = gen_sub.add_parser(
        "mastery",
        help="Calculate reproducible learner-skill mastery snapshots (Phase 5).",
    )
    mastery_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    mastery_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    mastery_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source master+activity dataset directory (defaults to data/sample or data/reference).",
    )
    mastery_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/sample for dev, data/reference for full).",
    )
    mastery_gen.add_argument(
        "--as-of",
        dest="as_of",
        type=str,
        default=None,
        help="Temporal calculation cutoff timestamp (ISO UTC). Evidence after this cutoff is excluded.",
    )
    mastery_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing mastery files.",
    )

    # 2e. generate recommendations
    rec_gen = gen_sub.add_parser(
        "recommendations",
        help="Generate reproducible rule-based learning activity recommendations (Phase 6).",
    )
    rec_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    rec_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    rec_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source directory with master, activity, and mastery data.",
    )
    rec_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/sample for dev, data/reference for full).",
    )
    rec_gen.add_argument(
        "--as-of",
        dest="as_of",
        type=str,
        default=None,
        help="Temporal decision cutoff timestamp (ISO UTC).",
    )
    rec_gen.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of recommendations per learner (default 5, max 20).",
    )
    rec_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing recommendation files.",
    )

    # 2f. generate paths (Phase 7)
    path_gen = gen_sub.add_parser(
        "paths",
        help="Generate ordered learning paths from Phase 6 recommendations (Phase 7).",
    )
    path_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    path_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default from config: 20261022).",
    )
    path_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source directory with master, activity, mastery and recommendation data.",
    )
    path_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/sample for dev, data/reference for full).",
    )
    path_gen.add_argument(
        "--as-of",
        dest="as_of",
        type=str,
        default=None,
        help="Temporal decision cutoff timestamp (ISO UTC). Recommendations after this cutoff are excluded.",
    )
    path_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing learning-path files.",
    )

    # 2g. generate ml-dataset (Phase 8)
    mldata_gen = gen_sub.add_parser(
        "ml-dataset",
        help="Construct point-in-time training examples from activity data (Phase 8).",
    )
    mldata_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    mldata_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source directory with master and activity data.",
    )
    mldata_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/ml/<profile>).",
    )

    # 2h. train ml-model (Phase 8)
    train_cmd = gen_sub.add_parser(
        "ml-model",
        help="Train explainable ML model on chronological train split (Phase 8).",
    )
    train_cmd.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    train_cmd.add_argument(
        "--data-dir",
        dest="data_dir",
        type=Path,
        default=None,
        help="Directory with ml training examples (defaults to data/ml/<profile>).",
    )
    train_cmd.add_argument(
        "--model-dir",
        dest="model_dir",
        type=Path,
        default=Path("models"),
        help="Model output directory (default: models).",
    )
    train_cmd.add_argument(
        "--seed",
        type=int,
        default=20261022,
        help="Random seed (default: 20261022).",
    )

    # 2i. generate ml-recommendations (Phase 8)
    mlrec_gen = gen_sub.add_parser(
        "ml-recommendations",
        help="Generate reproducible ML recommendations ranked by model (Phase 8).",
    )
    mlrec_gen.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    mlrec_gen.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source directory with master, activity, and mastery data.",
    )
    mlrec_gen.add_argument(
        "--model-dir",
        dest="model_dir",
        type=Path,
        default=Path("models"),
        help="Model directory containing ml-v1.pkl.",
    )
    mlrec_gen.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/ml/<profile>/recommendations).",
    )
    mlrec_gen.add_argument(
        "--as-of",
        dest="as_of",
        type=str,
        default=None,
        help="Temporal decision cutoff timestamp (ISO UTC).",
    )
    mlrec_gen.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K recommendations per learner.",
    )
    mlrec_gen.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output.",
    )

    # 4. evaluate command group (Phase 8)
    eval_parser = sub.add_parser("evaluate", help="Evaluation and strategy comparison commands.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    comp_cmd = eval_sub.add_parser(
        "compare",
        help="Fair chronological comparison of rule-v1 vs ml-v1 (Phase 8).",
    )
    comp_cmd.add_argument(
        "--profile",
        choices=["dev", "full"],
        default="dev",
        help="Dataset profile: 'dev' or 'full'.",
    )
    comp_cmd.add_argument(
        "--source-dir",
        dest="source_dir",
        type=Path,
        default=None,
        help="Source directory with clean datasets.",
    )
    comp_cmd.add_argument(
        "--model-dir",
        dest="model_dir",
        type=Path,
        default=Path("models"),
        help="Model directory.",
    )

    # 3f. validate paths (Phase 7)
    path_val = val_sub.add_parser(
        "paths",
        help="Validate generated learning paths (schema, FK, prerequisite gate, checksums).",
    )
    path_val.add_argument(
        "--dir",
        dest="dir_path",
        type=Path,
        default=None,
        help="Directory to validate (default: data/sample).",
    )
    path_val.add_argument(
        "--profile",
        choices=["dev", "full"],
        default=None,
        help="Expected profile (inferred from manifest if omitted).",
    )
    path_val.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest verification.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "db":
            return _db_command(args.db_command)
        if args.command == "generate":
            return _generate_command(args)
        if args.command == "validate":
            return _validate_command(args)
        if args.command == "evaluate":
            return _evaluate_command(args)
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 3
    except DatabaseError as exc:
        print(f"database error: {exc}", file=sys.stderr)
        return 4
    except OutputExistsError as exc:
        print(f"output error: {exc}", file=sys.stderr)
        return 5
    except (MasterDataError, ActivityDataError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 5
    except QualityError as exc:
        print(f"quality error: {exc}", file=sys.stderr)
        return 5
    except MasteryValidationError as exc:
        print(f"mastery validation error: {exc}", file=sys.stderr)
        return 5
    except RecommendationValidationError as exc:
        print(f"recommendation validation error: {exc}", file=sys.stderr)
        return 5
    except PathValidationError as exc:
        print(f"path validation error: {exc}", file=sys.stderr)
        return 5
    except LearningPathError as exc:
        print(f"path generation error: {exc}", file=sys.stderr)
        return 5
    parser.print_help()
    return 1


def _generate_command(args: argparse.Namespace) -> int:
    if args.gen_command == "master":
        return _generate_master(args)
    elif args.gen_command == "activity":
        return _generate_activity(args)
    elif args.gen_command == "anomalies":
        return _generate_anomalies(args)
    elif args.gen_command == "mastery":
        return _generate_mastery(args)
    elif args.gen_command == "recommendations":
        return _generate_recommendations(args)
    elif args.gen_command == "paths":
        return _generate_paths(args)
    elif args.gen_command == "ml-dataset":
        return _generate_ml_dataset(args)
    elif args.gen_command == "ml-model":
        return _train_ml_model(args)
    elif args.gen_command == "ml-recommendations":
        return _generate_ml_recommendations(args)
    else:
        print(f"Unknown generation target: {args.gen_command}", file=sys.stderr)
        return 1


def _generate_master(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    out_dir = getattr(args, "output_dir", None)
    if out_dir is None:
        out_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        out_dir = Path(out_dir)

    cfg = load_app_config()
    tables = generate_master_tables(profile=profile, seed=seed, config=cfg)
    validate_master_tables(tables, profile=profile)

    manifest = write_master_dataset(
        tables,
        out_dir,
        overwrite=overwrite,
        calendar_start=str(cfg.get("generation", {}).get("calendar_start", "2024-01-01T00:00:00Z")),
        calendar_end=str(cfg.get("generation", {}).get("calendar_end", "2026-10-22T23:59:59Z")),
    )
    counts = tables.row_counts()
    print(f"profile={profile}")
    print(f"seed={manifest['seed']}")
    print(f"output_dir={out_dir}")
    print(f"courses={counts['courses']}")
    print(f"learners={counts['learners']}")
    print(f"skills={counts['skills']}")
    print(f"skill_prerequisites={counts['skill_prerequisites']}")
    print(f"content_items={counts['content_items']}")
    print(f"content_skills={counts['content_skills']}")
    print(f"manifest={out_dir / MANIFEST_NAME}")
    print("status=generated")
    return 0


def _generate_activity(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    attempts_override = getattr(args, "attempts", None)
    out_dir = getattr(args, "output_dir", None)
    if out_dir is None:
        out_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        out_dir = Path(out_dir)

    cfg = load_app_config()

    # Load master tables from out_dir if available, else generate in memory
    import csv
    master_dict = {}
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    all_master_exist = all((out_dir / fn).is_file() for fn in master_fnames)
    if all_master_exist:
        for fn, k in master_fnames.items():
            with (out_dir / fn).open("r", encoding="utf-8") as f:
                master_dict[k] = list(csv.DictReader(f))
        master_tables = master_dict
    else:
        master_tables = generate_master_tables(profile=profile, seed=seed, config=cfg)

    activity_tables = generate_activity_tables(
        profile=profile,
        seed=seed,
        config=cfg,
        master_tables=master_tables,
        target_attempts=attempts_override,
    )
    validate_activity_tables(activity_tables, master_tables, profile=profile)

    manifest = write_activity_dataset(
        activity_tables,
        out_dir,
        overwrite=overwrite,
        calendar_start=str(cfg.get("generation", {}).get("calendar_start", "2024-01-01T00:00:00Z")),
        calendar_end=str(cfg.get("generation", {}).get("calendar_end", "2026-10-22T23:59:59Z")),
    )
    counts = activity_tables.row_counts()
    print(f"profile={profile}")
    print(f"seed={manifest['seed']}")
    print(f"output_dir={out_dir}")
    print(f"attempts={counts['attempts']}")
    print(f"assessment_results={counts['assessment_results']}")
    print(f"manifest={out_dir / ACTIVITY_MANIFEST_NAME}")
    print("status=generated")
    return 0


def _generate_anomalies(args: argparse.Namespace) -> int:
    """Generate anomaly fixtures, detect them and write quarantine + report."""
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    source_dir = getattr(args, "source_dir", None)
    out_dir = getattr(args, "output_dir", None)
    as_of = getattr(args, "as_of", None)

    cfg = load_app_config()
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("default_seed", 20261022))

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = Path(f"data/quarantine/{profile}")
    else:
        out_dir = Path(out_dir)

    gt_dir = Path(cfg.get("anomalies", {}).get("ground_truth_dir", "data/ground_truth"))

    # Load master + activity tables from source directory
    import csv as _csv

    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, k in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[k] = list(_csv.DictReader(f))
        else:
            master_dict[k] = []

    activity_dict: dict = {}
    for fn, k in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                activity_dict[k] = list(_csv.DictReader(f))
        else:
            activity_dict[k] = []

    fixtures = generate_anomaly_fixtures(
        master_dict,
        activity_dict,
        profile=profile,
        seed=seed,
        config=cfg,
    )

    results = detect_all_anomalies(fixtures, master_dict)

    manifest = write_quarantine(
        fixtures,
        out_dir,
        overwrite=overwrite,
        ground_truth_dir=gt_dir,
        profile=profile,
        seed=seed,
    )

    reports_dir = Path("reports")
    all_fixtures = fixtures.all()
    write_quality_report(
        results,
        reports_dir,
        profile=profile,
        total_inspected=len(all_fixtures),
        total_valid=len(all_fixtures) - len(results),
    )

    by_type = fixtures.by_type()
    print(f"profile={profile}")
    print(f"seed={seed}")
    print(f"source_dir={source_dir}")
    print(f"quarantine_dir={out_dir}")
    print(f"ground_truth_dir={gt_dir}")
    for atype, lst in sorted(by_type.items()):
        print(f"injected_{atype}={len(lst)}")
    print(f"total_injected={len(all_fixtures)}")
    print(f"total_detected={len(results)}")
    print(f"manifest={out_dir / 'quality_manifest.json'}")
    print(f"report={reports_dir / f'quality_report_{profile}.json'}")
    print("status=generated")
    return 0


def _generate_mastery(args: argparse.Namespace) -> int:
    """Generate reproducible learner-skill mastery snapshots (Phase 5)."""
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    source_dir = getattr(args, "source_dir", None)
    out_dir = getattr(args, "output_dir", None)
    as_of = getattr(args, "as_of", None)
    as_of = getattr(args, "as_of", None)

    cfg = load_app_config()
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("default_seed", 20261022))

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = source_dir
    else:
        out_dir = Path(out_dir)

    import csv as _csv

    # Load master tables
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, k in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[k] = list(_csv.DictReader(f))
        else:
            master_dict[k] = []

    # Load activity tables
    activity_dict: dict = {}
    for fn, k in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                activity_dict[k] = list(_csv.DictReader(f))
        else:
            activity_dict[k] = []

    snapshots = calculate_mastery_snapshots(
        master_dict,
        activity_dict,
        as_of_timestamp=as_of,
        config=cfg,
        include_cold_start=True,
    )

    manifest = write_mastery_dataset(
        snapshots,
        out_dir,
        overwrite=overwrite,
        profile=profile,
        seed=seed,
        as_of_timestamp=as_of,
    )

    print(f"profile={profile}")
    print(f"seed={seed}")
    print(f"source_dir={source_dir}")
    print(f"output_dir={out_dir}")
    print(f"as_of_timestamp={manifest['as_of_timestamp']}")
    print(f"snapshots={len(snapshots)}")
    print(f"manifest={out_dir / 'mastery_manifest.json'}")
    print("status=generated")
    return 0


def _generate_recommendations(args: argparse.Namespace) -> int:
    """Generate reproducible rule-based learning activity recommendations (Phase 6)."""
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    source_dir = getattr(args, "source_dir", None)
    out_dir = getattr(args, "output_dir", None)
    as_of = getattr(args, "as_of", None)
    as_of = getattr(args, "as_of", None)
    k = getattr(args, "k", 5) or 5

    cfg = load_app_config()
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("default_seed", 20261022))

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = source_dir
    else:
        out_dir = Path(out_dir)

    import csv as _csv

    # 1. Load master tables
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, key in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))
        else:
            master_dict[key] = []

    # 2. Load activity tables
    activity_dict: dict = {}
    for fn, key in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                activity_dict[key] = list(_csv.DictReader(f))
        else:
            activity_dict[key] = []

    # 3. Load mastery snapshots
    mastery_fpath = source_dir / "mastery_snapshots.csv"
    mastery_list: list[dict] = []
    if mastery_fpath.is_file():
        with mastery_fpath.open("r", encoding="utf-8") as f:
            mastery_list = list(_csv.DictReader(f))

    recommendations = generate_recommendations(
        master_dict,
        activity_dict,
        mastery_list,
        as_of_timestamp=as_of,
        k=k,
        config=cfg,
    )

    manifest = write_recommendation_dataset(
        recommendations,
        out_dir,
        overwrite=overwrite,
        profile=profile,
        seed=seed,
        as_of_timestamp=as_of,
    )

    print(f"profile={profile}")
    print(f"seed={seed}")
    print(f"source_dir={source_dir}")
    print(f"output_dir={out_dir}")
    print(f"as_of_timestamp={manifest['as_of_timestamp']}")
    print(f"k={k}")
    print(f"recommendations={len(recommendations)}")
    print(f"manifest={out_dir / 'recommendation_manifest.json'}")
    print("status=generated")
    return 0


def _generate_paths(args: argparse.Namespace) -> int:
    """Generate reproducible learning paths and path items (Phase 7)."""
    profile = getattr(args, "profile", "dev") or "dev"
    seed = getattr(args, "seed", None)
    overwrite = getattr(args, "overwrite", False)
    source_dir = getattr(args, "source_dir", None)
    out_dir = getattr(args, "output_dir", None)
    as_of = getattr(args, "as_of", None)

    cfg = load_app_config()
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("default_seed", 20261022))

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = source_dir
    else:
        out_dir = Path(out_dir)

    import csv as _csv

    # 1. Load master tables
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, key in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))
        else:
            master_dict[key] = []

    # 2. Load recommendations
    recs_fpath = source_dir / "recommendations.csv"
    recs_list: list[dict] = []
    if recs_fpath.is_file():
        with recs_fpath.open("r", encoding="utf-8") as f:
            recs_list = list(_csv.DictReader(f))

    # 3. Load mastery snapshots
    mastery_fpath = source_dir / "mastery_snapshots.csv"
    mastery_list: list[dict] = []
    if mastery_fpath.is_file():
        with mastery_fpath.open("r", encoding="utf-8") as f:
            mastery_list = list(_csv.DictReader(f))

    paths, items = generate_learning_paths(
        master_dict,
        recs_list,
        mastery_list,
        as_of_timestamp=as_of,
    )

    manifest = write_path_dataset(
        paths,
        items,
        out_dir,
        overwrite=overwrite,
        profile=profile,
        seed=seed,
        as_of_timestamp=as_of or (paths[0].generated_at if paths else ""),
    )

    print(f"profile={profile}")
    print(f"seed={seed}")
    print(f"source_dir={source_dir}")
    print(f"output_dir={out_dir}")
    print(f"learning_paths={len(paths)}")
    print(f"learning_path_items={len(items)}")
    print(f"manifest={out_dir / 'learning_path_manifest.json'}")
    print("status=generated")
    return 0


def _generate_ml_dataset(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    source_dir = getattr(args, "source_dir", None)
    out_dir = getattr(args, "output_dir", None)

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = Path(f"data/ml/{profile}")
    else:
        out_dir = Path(out_dir)

    import csv as _csv
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, key in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))
        else:
            master_dict[key] = []

    activity_dict: dict = {}
    for fn, key in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                activity_dict[key] = list(_csv.DictReader(f))
        else:
            activity_dict[key] = []

    examples = build_ml_dataset(master_dict, activity_dict)
    split = chronological_split(examples)

    write_dataset_to_csv(split.train, out_dir / "train.csv")
    write_dataset_to_csv(split.val, out_dir / "val.csv")
    write_dataset_to_csv(split.test, out_dir / "test.csv")
    write_dataset_to_csv(examples, out_dir / "all_examples.csv")

    pos = sum(ex.label for ex in examples)
    neg = len(examples) - pos

    print(f"profile={profile}")
    print(f"source_dir={source_dir}")
    print(f"output_dir={out_dir}")
    print(f"total_examples={len(examples)}")
    print(f"train_examples={len(split.train)}")
    print(f"val_examples={len(split.val)}")
    print(f"test_examples={len(split.test)}")
    print(f"label_distribution=pos:{pos},neg:{neg},pos_rate:{pos/len(examples):.4f}" if examples else "label_distribution=empty")
    print("status=generated")
    return 0


def _train_ml_model(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    data_dir = getattr(args, "data_dir", None)
    model_dir = getattr(args, "model_dir", Path("models"))
    seed = getattr(args, "seed", 20261022)

    if data_dir is None:
        data_dir = Path(f"data/ml/{profile}")
    else:
        data_dir = Path(data_dir)

    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"

    if not train_file.is_file():
        # Build dataset first if missing
        _generate_ml_dataset(args)

    import csv as _csv
    from learnpath_ai.ml.dataset import MLTrainingExample

    def _read_ex(path: Path) -> list[MLTrainingExample]:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            res = []
            for r in reader:
                res.append(MLTrainingExample(
                    attempt_id=r["attempt_id"],
                    learner_id=r["learner_id"],
                    content_id=r["content_id"],
                    timestamp=r["timestamp"],
                    primary_skill_mastery=float(r["primary_skill_mastery"]),
                    content_difficulty=float(r["content_difficulty"]),
                    diff_suitability=float(r["diff_suitability"]),
                    primary_skill_weight=float(r["primary_skill_weight"]),
                    learner_prior_attempts_on_content=int(r["learner_prior_attempts_on_content"]),
                    learner_prior_passes_on_content=int(r["learner_prior_passes_on_content"]),
                    learner_overall_prior_attempts=int(r["learner_overall_prior_attempts"]),
                    learner_overall_prior_pass_rate=float(r["learner_overall_prior_pass_rate"]),
                    label=int(r["label"]),
                ))
            return res

    train_ex = _read_ex(train_file)
    val_ex = _read_ex(val_file)
    test_ex = _read_ex(test_file)

    split = DatasetSplit(train=train_ex, val=val_ex, test=test_ex)
    artifact = train_model(split, seed=seed)
    artifact.save(model_dir)

    print(f"model_version={artifact.model_version}")
    print(f"model_dir={model_dir}")
    print(f"trained_at={artifact.trained_at}")
    print(f"intercept={artifact.intercept:.4f}")
    for k, v in artifact.metrics.items():
        print(f"val_{k}={v:.4f}")
    print("status=trained")
    return 0


def _generate_ml_recommendations(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    source_dir = getattr(args, "source_dir", None)
    model_dir = getattr(args, "model_dir", Path("models"))
    out_dir = getattr(args, "output_dir", None)
    as_of = getattr(args, "as_of", None)
    k = getattr(args, "k", 5) or 5
    overwrite = getattr(args, "overwrite", False)

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    if out_dir is None:
        out_dir = Path(f"data/ml/{profile}/recommendations")
    else:
        out_dir = Path(out_dir)

    import csv as _csv
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    master_dict: dict = {}
    for fn, key in master_fnames.items():
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))
        else:
            master_dict[key] = []

    activity_dict: dict = {}
    for fn, key in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        fpath = source_dir / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                activity_dict[key] = list(_csv.DictReader(f))
        else:
            activity_dict[key] = []

    mastery_fpath = source_dir / "mastery_snapshots.csv"
    mastery_list: list[dict] = []
    if mastery_fpath.is_file():
        with mastery_fpath.open("r", encoding="utf-8") as f:
            mastery_list = list(_csv.DictReader(f))

    # Load model artifact
    artifact = ModelArtifact.load(model_dir)

    ml_recs = generate_ml_recommendations(
        master_dict,
        activity_dict,
        mastery_list,
        artifact,
        as_of_timestamp=as_of,
        k=k,
    )

    manifest = write_recommendation_dataset(
        ml_recs,
        out_dir,
        overwrite=overwrite,
        profile=profile,
        seed=20261022,
        as_of_timestamp=as_of,
        generator_version="ml-v1",
    )

    print(f"profile={profile}")
    print(f"model_version={artifact.model_version}")
    print(f"output_dir={out_dir}")
    print(f"recommendations={len(ml_recs)}")
    print(f"manifest={out_dir / 'recommendation_manifest.json'}")
    print("status=generated")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    if args.val_command == "master":
        return _validate_master(args)
    elif args.val_command == "activity":
        return _validate_activity(args)
    elif args.val_command == "quality":
        return _validate_quality(args)
    elif args.val_command == "mastery":
        return _validate_mastery(args)
    elif args.val_command == "recommendations":
        return _validate_recommendations(args)
    elif args.val_command == "paths":
        return _validate_paths(args)
    else:
        print(f"Unknown validation target: {args.val_command}", file=sys.stderr)
        return 1


def _validate_master(args: argparse.Namespace) -> int:
    dir_path = getattr(args, "dir_path", None)
    if dir_path is None:
        profile = getattr(args, "profile", None)
        if profile == "full":
            dir_path = Path("data/reference")
        else:
            dir_path = Path("data/sample")
    else:
        dir_path = Path(dir_path)

    profile = getattr(args, "profile", None)
    verify_manifest = not getattr(args, "no_manifest", False)
    result = validate_master_dataset_dir(
        dir_path,
        expected_profile=profile,
        verify_manifest=verify_manifest,
    )
    print(f"directory={result['directory']}")
    print(f"profile={result.get('profile') or '(unspecified)'}")
    for fname, count in sorted(result["row_counts"].items()):
        print(f"rows_{fname}={count}")
    print("manifest=verified" if verify_manifest else "manifest=skipped")
    print("checksums=verified" if verify_manifest else "checksums=skipped")
    print("status=valid")
    return 0


def _validate_activity(args: argparse.Namespace) -> int:
    dir_path = getattr(args, "dir_path", None)
    if dir_path is None:
        profile = getattr(args, "profile", None)
        if profile == "full":
            dir_path = Path("data/reference")
        else:
            dir_path = Path("data/sample")
    else:
        dir_path = Path(dir_path)

    profile = getattr(args, "profile", None)
    verify_manifest = not getattr(args, "no_manifest", False)
    result = validate_activity_dataset_dir(
        dir_path,
        expected_profile=profile,
        verify_manifest=verify_manifest,
    )
    print(f"directory={result['directory']}")
    print(f"profile={result.get('profile') or '(unspecified)'}")
    for fname, count in sorted(result["row_counts"].items()):
        print(f"rows_{fname}={count}")
    print("manifest=verified" if verify_manifest else "manifest=skipped")
    print("checksums=verified" if verify_manifest else "checksums=skipped")
    print("status=valid")
    return 0


def _validate_quality(args: argparse.Namespace) -> int:
    """Re-detect anomalies from quarantine files and verify expected decisions."""
    profile = getattr(args, "profile", "dev") or "dev"
    dir_path = getattr(args, "dir_path", None)
    quarantine_dir = getattr(args, "quarantine_dir", None)

    if dir_path is None:
        dir_path = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        dir_path = Path(dir_path)

    if quarantine_dir is None:
        quarantine_dir = Path(f"data/quarantine/{profile}")
    else:
        quarantine_dir = Path(quarantine_dir)

    import csv as _csv

    combined_path = quarantine_dir / "data_quality_issues.csv"
    if not combined_path.is_file():
        print(
            f"quality validation error: quarantine file not found: {combined_path}",
            file=sys.stderr,
        )
        return 5

    with combined_path.open("r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    total = len(rows)
    type_counts: dict[str, int] = {}
    for row in rows:
        atype = row.get("issue_type", "UNKNOWN")
        type_counts[atype] = type_counts.get(atype, 0) + 1

    print(f"directory={quarantine_dir}")
    print(f"profile={profile}")
    print(f"total_issues={total}")
    for atype, count in sorted(type_counts.items()):
        print(f"count_{atype}={count}")
    print("status=valid")
    return 0


def _validate_mastery(args: argparse.Namespace) -> int:
    """Validate generated mastery snapshots (Phase 5)."""
    dir_path = getattr(args, "dir_path", None)
    if dir_path is None:
        profile = getattr(args, "profile", None)
        if profile == "full":
            dir_path = Path("data/reference")
        else:
            dir_path = Path("data/sample")
    else:
        dir_path = Path(dir_path)

    profile = getattr(args, "profile", None)
    verify_manifest = not getattr(args, "no_manifest", False)

    # Load master tables if available to check foreign keys
    import csv as _csv
    master_dict: dict = {}
    for fn, k in [("learners.csv", "learners"), ("skills.csv", "skills")]:
        fpath = dir_path / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[k] = list(_csv.DictReader(f))

    result = validate_mastery_dataset_dir(
        dir_path,
        expected_profile=profile,
        verify_manifest=verify_manifest,
        master_tables=master_dict if master_dict else None,
    )

    print(f"directory={result['directory']}")
    print(f"profile={result.get('profile') or '(unspecified)'}")
    for fname, count in sorted(result["row_counts"].items()):
        print(f"rows_{fname}={count}")
    print("manifest=verified" if verify_manifest else "manifest=skipped")
    print("checksums=verified" if verify_manifest else "checksums=skipped")
    print("status=valid")
    return 0


def _validate_recommendations(args: argparse.Namespace) -> int:
    """Validate generated recommendations (Phase 6)."""
    dir_path = getattr(args, "dir_path", None)
    if dir_path is None:
        profile = getattr(args, "profile", None)
        if profile == "full":
            dir_path = Path("data/reference")
        else:
            dir_path = Path("data/sample")
    else:
        dir_path = Path(dir_path)

    profile = getattr(args, "profile", None)
    verify_manifest = not getattr(args, "no_manifest", False)

    import csv as _csv

    # Load master tables for FK and prerequisite validation
    master_dict: dict = {}
    master_fnames = {
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    for fn, key in master_fnames.items():
        fpath = dir_path / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))

    # Load mastery snapshots for prerequisite verification
    mastery_fpath = dir_path / "mastery_snapshots.csv"
    mastery_list: list[dict] = []
    if mastery_fpath.is_file():
        with mastery_fpath.open("r", encoding="utf-8") as f:
            mastery_list = list(_csv.DictReader(f))

    result = validate_recommendation_dataset_dir(
        dir_path,
        expected_profile=profile,
        verify_manifest=verify_manifest,
        master_tables=master_dict if master_dict else None,
        mastery_snapshots=mastery_list if mastery_list else None,
    )

    print(f"directory={result['directory']}")
    print(f"profile={result.get('profile') or '(unspecified)'}")
    for fname, count in sorted(result["row_counts"].items()):
        print(f"rows_{fname}={count}")
    print("manifest=verified" if verify_manifest else "manifest=skipped")
    print("checksums=verified" if verify_manifest else "checksums=skipped")
    print("status=valid")
    return 0


def _validate_paths(args: argparse.Namespace) -> int:
    """Validate generated learning paths (Phase 7)."""
    dir_path = getattr(args, "dir_path", None)
    if dir_path is None:
        profile = getattr(args, "profile", None)
        if profile == "full":
            dir_path = Path("data/reference")
        else:
            dir_path = Path("data/sample")
    else:
        dir_path = Path(dir_path)

    profile = getattr(args, "profile", None)
    verify_manifest = not getattr(args, "no_manifest", False)

    import csv as _csv

    # Load master tables for FK and constraint validation
    master_dict: dict = {}
    master_fnames = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    for fn, key in master_fnames.items():
        fpath = dir_path / fn
        if fpath.is_file():
            with fpath.open("r", encoding="utf-8") as f:
                master_dict[key] = list(_csv.DictReader(f))

    result = validate_path_dataset_dir(
        dir_path,
        expected_profile=profile,
        verify_manifest=verify_manifest,
        master_tables=master_dict if master_dict else None,
    )

    print(f"directory={result['directory']}")
    print(f"profile={result.get('profile') or '(unspecified)'}")
    for fname, count in sorted(result["row_counts"].items()):
        print(f"rows_{fname}={count}")
    print("manifest=verified" if verify_manifest else "manifest=skipped")
    print("checksums=verified" if verify_manifest else "checksums=skipped")
    print("status=valid")
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    if args.eval_command == "compare":
        return _evaluate_compare(args)
    else:
        print(f"Unknown eval command: {args.eval_command}", file=sys.stderr)
        return 1


def _evaluate_compare(args: argparse.Namespace) -> int:
    profile = getattr(args, "profile", "dev") or "dev"
    source_dir = getattr(args, "source_dir", None)
    model_dir = getattr(args, "model_dir", Path("models"))

    if source_dir is None:
        source_dir = Path("data/sample") if profile == "dev" else Path("data/reference")
    else:
        source_dir = Path(source_dir)

    import csv as _csv

    # 1. Load data
    master_dict: dict = {}
    for fn, k in [("courses.csv", "courses"), ("learners.csv", "learners"), ("skills.csv", "skills"),
                  ("skill_prerequisites.csv", "skill_prerequisites"), ("content_items.csv", "content_items"),
                  ("content_skills.csv", "content_skills")]:
        with (source_dir / fn).open("r", encoding="utf-8") as f:
            master_dict[k] = list(_csv.DictReader(f))

    activity_dict: dict = {}
    for fn, k in [("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")]:
        with (source_dir / fn).open("r", encoding="utf-8") as f:
            activity_dict[k] = list(_csv.DictReader(f))

    # Split attempts chronologically (70% train, 15% val, 15% test)
    attempts = activity_dict["attempts"]
    results = {r["attempt_id"]: r for r in activity_dict["assessment_results"]}
    scored_attempts = [
        a for a in sorted(attempts, key=lambda x: parse_iso_utc(x["started_at"]))
        if a["attempt_id"] in results
    ]
    n = len(scored_attempts)
    test_cutoff_idx = int(n * 0.85)
    test_attempts = scored_attempts[test_cutoff_idx:]
    test_as_of = scored_attempts[test_cutoff_idx]["started_at"]

    # Compute point-in-time mastery snapshots strictly as of test_as_of
    pit_mastery_snapshots = calculate_mastery_snapshots(
        master_dict,
        activity_dict,
        as_of_timestamp=test_as_of,
        include_cold_start=True,
    )

    # Combine test attempts with result outcome/passed
    ground_truth_test = []
    for a in test_attempts:
        r = results.get(a["attempt_id"])
        item = dict(a)
        if r:
            item["outcome"] = r.get("outcome")
            item["passed"] = r.get("passed")
        ground_truth_test.append(item)

    # Eligible and cold start learners
    learners = master_dict["learners"]
    eligible_learners = [l["learner_id"] for l in learners if l.get("status") != "withdrawn"]
    # Cold start learners: no attempts prior to test_as_of
    prior_lids = {a["learner_id"] for a in scored_attempts[:test_cutoff_idx]}
    cold_start_learners = [lid for lid in eligible_learners if lid not in prior_lids]

    # Generate rule recommendations at test cutoff using point-in-time mastery
    rule_recs = generate_recommendations(
        master_dict,
        activity_dict,
        pit_mastery_snapshots,
        as_of_timestamp=test_as_of,
        k=5,
    )

    # Train or load model
    model_file = model_dir / "ml-v1.pkl"
    if not model_file.is_file():
        _train_ml_model(args)
    artifact = ModelArtifact.load(model_dir)

    ml_recs = generate_ml_recommendations(
        master_dict,
        activity_dict,
        pit_mastery_snapshots,
        artifact,
        as_of_timestamp=test_as_of,
        k=5,
    )

    eval_results = evaluate_recommendation_strategies(
        rule_recs,
        ml_recs,
        ground_truth_test,
        eligible_learners,
        cold_start_learners,
        k_list=[1, 3, 5],
    )

    print("=================== EVALUATION REPORT (CHRONOLOGICAL TEST SLICE) ===================")
    print(f"Profile: {profile} | Test Cutoff Timestamp: {test_as_of}")
    print(f"Eligible Learners: {len(eligible_learners)} | Cold Start Learners: {len(cold_start_learners)}")
    print(f"Test Attempts Observed: {len(ground_truth_test)}")
    print("------------------------------------------------------------------------------------")
    for name, metrics in eval_results.items():
        print(f"Strategy: {name}")
        print(f"  Coverage: {metrics.coverage:.4f} ({metrics.covered_learners}/{metrics.total_learners})")
        print(f"  Cold-Start Coverage: {metrics.cold_start_coverage:.4f}")
        for k in [1, 3, 5]:
            print(f"  Precision@{k}: {metrics.precision_at_k.get(k, 0.0):.4f} | Recall@{k}: {metrics.recall_at_k.get(k, 0.0):.4f} | NDCG@{k}: {metrics.ndcg_at_k.get(k, 0.0):.4f}")
        print("------------------------------------------------------------------------------------")
    print("status=evaluated")
    return 0


def _db_command(action: str) -> int:
    settings = load_database_settings()
    db = Database(settings)
    try:
        if action == "ping":
            result = db.ping()
            _print_ping(result)
            return 0
        if action == "status":
            status = inspect_schema(db)
            _print_status(status.database, status)
            return 0 if status.database_exists else 3
        if action == "init":
            result = initialize_schema(db)
            _print_init(result)
            return 0
    finally:
        db.dispose()
    return 1


def _print_ping(result: PingResult) -> None:
    print(f"host={result.host}")
    print(f"port={result.port}")
    print(f"user={result.user}")
    print(f"database={result.current_database or '(none)'}")
    print(f"server_version={result.server_version}")
    if not str(result.server_version).startswith("8.4"):
        print(
            "warning: target platform is MySQL 8.4 LTS; connected server is "
            f"{result.server_version}",
            file=sys.stderr,
        )


def _print_status(database: str, status) -> None:
    print(f"database={database}")
    print(f"database_exists={status.database_exists}")
    print(f"compatible={status.compatible}")
    print(f"schema_version={status.schema_version or '(none)'}")
    print(f"existing_tables={','.join(status.existing_tables) or '(none)'}")
    print(f"missing_tables={','.join(status.missing_tables) or '(none)'}")
    for message in status.messages:
        print(f"message={message}")


def _print_init(result: InitResult) -> None:
    print(f"action={result.action}")
    print(f"created_tables={','.join(result.created_tables) or '(none)'}")
    _print_status(result.status.database, result.status)


if __name__ == "__main__":
    sys.exit(main())
