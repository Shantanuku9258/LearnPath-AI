"""Generate realistic, reproducible learner attempts and assessment results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

from learnpath_ai.config_loader import ConfigError, load_app_config
from learnpath_ai.generate.clock import CalendarClock, parse_iso_utc
from learnpath_ai.generate.columns import (
    ACTIVITY_FILES,
    ACTIVITY_MANIFEST_NAME,
    ASSESSMENT_RESULT_COLUMNS,
    ATTEMPT_COLUMNS,
)
from learnpath_ai.generate.master import MasterTables, generate_master_tables
from learnpath_ai.generate.writer import OutputExistsError, _sha256
from learnpath_ai.ids import format_attempt_id, format_id

GENERATOR_VERSION = "activity-v1"
TIMESTAMP_POLICY = (
    "Attempt timestamps are UTC instants strictly >= learner registration date (effective_from/created_at) "
    "and within [calendar_start, calendar_end] sampled using numpy.random.Generator with configured seed. "
    "Submitted timestamps are strictly >= started_at and scored_at >= submitted_at."
)


@dataclass(frozen=True)
class ActivityTables:
    seed: int
    profile: str
    generator_version: str
    attempts: list[dict[str, Any]]
    assessment_results: list[dict[str, Any]]

    def row_counts(self) -> dict[str, int]:
        return {
            "attempts": len(self.attempts),
            "assessment_results": len(self.assessment_results),
        }


def generate_activity_tables(
    *,
    profile: str = "dev",
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    master_tables: MasterTables | dict[str, list[dict[str, Any]]] | None = None,
    target_attempts: int | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
) -> ActivityTables:
    """Generate reproducible attempts and assessment results matching master tables."""
    app = config or load_app_config()
    profiles = app.get("dataset_profiles") or {}
    if profile not in profiles and target_attempts is None:
        raise ConfigError(f"Unknown dataset profile {profile!r}.")

    prof_cfg = profiles.get(profile, {})
    resolved_seed = int(seed if seed is not None else app.get("reproducibility", {}).get("default_seed", 20261022))

    if target_attempts is not None:
        total_attempts = int(target_attempts)
    else:
        if profile == "dev":
            total_attempts = int(prof_cfg.get("attempts", 5000))
            if total_attempts < 5000:
                total_attempts = 5000
        else:
            total_attempts = int(prof_cfg.get("attempts", 500000))

    if total_attempts < 1:
        raise ConfigError(f"Target attempts must be >= 1, got {total_attempts}")

    # Load or generate master tables
    if master_tables is None:
        master = generate_master_tables(profile=profile, seed=resolved_seed, config=app)
    elif isinstance(master_tables, MasterTables):
        master = master_tables
    elif isinstance(master_tables, dict):
        master = MasterTables(
            seed=resolved_seed,
            profile=profile,
            generator_version="master-v1",
            courses=master_tables.get("courses", []),
            learners=master_tables.get("learners", []),
            skills=master_tables.get("skills", []),
            skill_prerequisites=master_tables.get("skill_prerequisites", []),
            content_items=master_tables.get("content_items", []),
            content_skills=master_tables.get("content_skills", []),
        )
    else:
        raise ConfigError(f"Unsupported master_tables type: {type(master_tables)}")

    if not master.learners:
        raise ConfigError("Cannot generate activity without learners.")
    if not master.content_items:
        raise ConfigError("Cannot generate activity without content items.")

    gen_cfg = app.get("generation") or {}
    c_start_str = calendar_start or gen_cfg.get("calendar_start", "2024-01-01T00:00:00Z")
    c_end_str = calendar_end or gen_cfg.get("calendar_end", "2026-10-22T23:59:59Z")
    start_dt = parse_iso_utc(c_start_str)
    end_dt = parse_iso_utc(c_end_str)
    end_micros = int(end_dt.timestamp() * 1_000_000)

    rng = np.random.default_rng(resolved_seed)
    pass_threshold = float(gen_cfg.get("pass_score_threshold", 70.0))

    # Organize content items by course
    content_by_course: dict[str, list[dict[str, Any]]] = {}
    for item in master.content_items:
        content_by_course.setdefault(item["course_id"], []).append(item)

    all_courses = [c["course_id"] for c in master.courses]

    # Cold start learners (10% of learners have 0 attempts)
    num_learners = len(master.learners)
    cold_start_count = max(1, int(num_learners * 0.10)) if num_learners >= 10 else 0
    active_learners = master.learners[cold_start_count:] if cold_start_count > 0 else master.learners

    if not active_learners:
        active_learners = master.learners

    n_active = len(active_learners)

    # Assign activity weights using Pareto distribution
    base_weights = rng.pareto(a=1.5, size=n_active) + 0.1
    learner_attempt_probs = base_weights / np.sum(base_weights)

    # Sample attempt counts per learner deterministically via multinomial
    learner_counts = rng.multinomial(total_attempts, learner_attempt_probs)

    # Ability map by starting_level
    level_mean_map = {
        "beginner": 65.0,
        "intermediate": 75.0,
        "advanced": 85.0,
    }

    # Pre-parse learner effective_from in microseconds
    learner_start_micros: list[int] = []
    for l in active_learners:
        ef = l.get("effective_from") or l.get("created_at")
        learner_start_micros.append(int(parse_iso_utc(ef).timestamp() * 1_000_000))

    attempts: list[dict[str, Any]] = []
    assessment_results: list[dict[str, Any]] = []
    result_seq = 1

    for idx, count in enumerate(learner_counts):
        if count == 0:
            continue

        learner = active_learners[idx]
        lid = learner["learner_id"]
        l_start_us = learner_start_micros[idx]
        l_level = learner.get("starting_level", "beginner")
        base_ability = level_mean_map.get(l_level, 70.0)

        cid = learner.get("current_course_id")
        if not cid or cid not in content_by_course:
            cid = all_courses[idx % len(all_courses)]

        course_contents = content_by_course.get(cid, master.content_items)
        n_c = len(course_contents)

        selected_content_indices = rng.integers(0, n_c, size=count)
        # Create repeated attempts on some content items (~20%)
        if count > 4:
            repeat_target = selected_content_indices[0]
            for r in range(min(3, count // 4)):
                selected_content_indices[r + 1] = repeat_target

        content_attempt_counts: dict[str, int] = {}

        # Draw sorted offsets in microseconds
        max_span = max(60_000_000, end_micros - l_start_us)
        raw_offsets = np.sort(rng.integers(0, max_span, size=count, dtype=np.int64))
        started_micros = l_start_us + raw_offsets
        # Format timestamps vectorized using numpy datetime64
        started_str_array = np.char.replace(started_micros.astype("datetime64[us]").astype(str), "T", " ")

        # Draw durations in minutes for all attempts of this learner
        est_minutes_list = [int(course_contents[si].get("estimated_minutes", 30)) for si in selected_content_indices]
        durations_mins = rng.normal(loc=est_minutes_list, scale=np.maximum(2, np.array(est_minutes_list) * 0.25))
        durations_mins = np.maximum(1, durations_mins.astype(int))
        durations_us = durations_mins.astype(np.int64) * 60_000_000

        submitted_micros = np.minimum(end_micros, started_micros + durations_us)
        submitted_str_array = np.char.replace(submitted_micros.astype("datetime64[us]").astype(str), "T", " ")

        status_rolls = rng.random(size=count)
        perf_noises = rng.normal(loc=0.0, scale=12.0, size=count)
        scored_delays = rng.integers(1, 300, size=count, dtype=np.int64) * 1_000_000

        scored_micros = np.minimum(end_micros, submitted_micros + scored_delays)
        scored_str_array = np.char.replace(scored_micros.astype("datetime64[us]").astype(str), "T", " ")

        m_delta_rolls = rng.uniform(0.02, 0.15, size=count)
        m_delta_fail_rolls = rng.uniform(-0.05, 0.02, size=count)

        for seq_i in range(count):
            seq_num = seq_i + 1
            att_id = format_attempt_id(lid, seq_num)
            c_item = course_contents[selected_content_indices[seq_i]]
            content_id = c_item["content_id"]
            content_type = c_item.get("content_type", "article")
            difficulty = float(c_item.get("difficulty", 2.50))

            attempt_number = content_attempt_counts.get(content_id, 0) + 1
            content_attempt_counts[content_id] = attempt_number

            s_roll = status_rolls[seq_i]
            if content_type in ("quiz", "exercise", "project"):
                if s_roll < 0.85:
                    status = "scored"
                elif s_roll < 0.93:
                    status = "submitted"
                elif s_roll < 0.98:
                    status = "in_progress"
                else:
                    status = "abandoned"
            else:
                if s_roll < 0.45:
                    status = "scored"
                elif s_roll < 0.85:
                    status = "submitted"
                elif s_roll < 0.95:
                    status = "in_progress"
                else:
                    status = "abandoned"

            started_at_str = str(started_str_array[seq_i])
            created_at_str = started_at_str
            submitted_at_str = str(submitted_str_array[seq_i]) if status in ("scored", "submitted") else ""

            score_val: float | None = None
            max_score = 100.00
            score_str = ""

            if status == "scored":
                rep_bonus = min(12.0, (attempt_number - 1) * 5.0)
                mean_perf = base_ability - (difficulty - 3.0) * 8.0 + rep_bonus
                raw_score = float(mean_perf + perf_noises[seq_i])
                score_val = round(max(0.00, min(100.00, raw_score)), 2)
                score_str = f"{score_val:.2f}"

            attempts.append(
                {
                    "attempt_id": att_id,
                    "learner_id": lid,
                    "content_id": content_id,
                    "attempt_number": attempt_number,
                    "status": status,
                    "started_at": started_at_str,
                    "submitted_at": submitted_at_str,
                    "score": score_str,
                    "max_score": f"{max_score:.2f}",
                    "source_recommendation_id": "",
                    "created_at": created_at_str,
                }
            )

            if status == "scored" and score_val is not None:
                res_id = format_id("RES", result_seq)
                result_seq += 1
                passed_int = 1 if score_val >= pass_threshold else 0
                outcome = "pass" if passed_int == 1 else "fail"
                scored_at_str = str(scored_str_array[seq_i])

                if passed_int == 1:
                    m_delta = round(float(m_delta_rolls[seq_i]), 4)
                else:
                    m_delta = round(float(m_delta_fail_rolls[seq_i]), 4)

                assessment_results.append(
                    {
                        "result_id": res_id,
                        "attempt_id": att_id,
                        "outcome": outcome,
                        "score": f"{score_val:.2f}",
                        "passed": str(passed_int),
                        "mastery_delta": f"{m_delta:.4f}",
                        "scored_at": scored_at_str,
                        "created_at": scored_at_str,
                    }
                )

    return ActivityTables(
        seed=resolved_seed,
        profile=profile,
        generator_version=GENERATOR_VERSION,
        attempts=attempts,
        assessment_results=assessment_results,
    )


def write_activity_dataset(
    tables: ActivityTables,
    output_dir: Path,
    *,
    overwrite: bool = False,
    calendar_start: str = "2024-01-01T00:00:00Z",
    calendar_end: str = "2026-10-22T23:59:59Z",
) -> dict[str, Any]:
    """Write attempts.csv, assessment_results.csv and activity_manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [output_dir / name for name in ACTIVITY_FILES]
    planned.append(output_dir / ACTIVITY_MANIFEST_NAME)
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    attempts_path = output_dir / "attempts.csv"
    _write_csv_fast(attempts_path, ATTEMPT_COLUMNS, tables.attempts)
    results_path = output_dir / "assessment_results.csv"
    _write_csv_fast(results_path, ASSESSMENT_RESULT_COLUMNS, tables.assessment_results)

    checksums = {
        "attempts.csv": _sha256(attempts_path),
        "assessment_results.csv": _sha256(results_path),
    }
    row_counts = {
        "attempts.csv": len(tables.attempts),
        "assessment_results.csv": len(tables.assessment_results),
    }

    manifest = {
        "generator_version": tables.generator_version,
        "seed": tables.seed,
        "dataset_profile": tables.profile,
        "timestamp_policy": TIMESTAMP_POLICY,
        "calendar_start": calendar_start,
        "calendar_end": calendar_end,
        "row_counts": row_counts,
        "checksums": checksums,
        "checksums_sha256": checksums,
        "files": list(ACTIVITY_FILES),
    }

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / ACTIVITY_MANIFEST_NAME).write_bytes(manifest_bytes)
    return manifest


def _write_csv_fast(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        col_list = list(columns)
        batch_size = 50000
        batch = []
        for row in rows:
            batch.append([row.get(col, "") for col in col_list])
            if len(batch) >= batch_size:
                writer.writerows(batch)
                batch.clear()
        if batch:
            writer.writerows(batch)
