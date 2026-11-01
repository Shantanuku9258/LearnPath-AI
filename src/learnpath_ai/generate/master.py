"""Generate parent/master tables in dependency order."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from learnpath_ai.config_loader import ConfigError, load_app_config, profile_counts
from learnpath_ai.generate.catalog import (
    BLOOM_LEVELS,
    CONTENT_TYPES,
    CONTENT_VERBS,
    DIFFICULTY_BANDS,
    DOMAINS,
    GIVEN_NAMES,
    LEARNER_LEVELS,
    LEARNER_STATUSES,
    course_title_for,
    skill_name_for,
)
from learnpath_ai.generate.clock import CalendarClock, format_timestamp, parse_iso_utc
from learnpath_ai.ids import format_id
from learnpath_ai.prerequisites import PrerequisiteEdge, require_dag

GENERATOR_VERSION = "master-v1"


@dataclass(frozen=True)
class MasterTables:
    seed: int
    profile: str
    generator_version: str
    courses: list[dict[str, Any]]
    learners: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    skill_prerequisites: list[dict[str, Any]]
    content_items: list[dict[str, Any]]
    content_skills: list[dict[str, Any]]

    def row_counts(self) -> dict[str, int]:
        return {
            "courses": len(self.courses),
            "learners": len(self.learners),
            "skills": len(self.skills),
            "skill_prerequisites": len(self.skill_prerequisites),
            "content_items": len(self.content_items),
            "content_skills": len(self.content_skills),
        }


def generate_master_tables(
    *,
    profile: str = "dev",
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    counts: dict[str, int] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> MasterTables:
    app = config or load_app_config()
    if counts is not None:
        required = ("learners", "courses", "skills", "content_items")
        missing = [key for key in required if key not in counts]
        if missing:
            raise ConfigError(f"Custom counts missing keys: {', '.join(missing)}")
        resolved_counts = {key: int(counts[key]) for key in required}
        for key, value in resolved_counts.items():
            if value < 1:
                raise ConfigError(f"Count {key!r} must be >= 1, got {value}.")
        if resolved_counts["skills"] < 2:
            raise ConfigError("At least 2 skills are required to form a prerequisite graph.")
        if resolved_counts["content_items"] < resolved_counts["courses"]:
            raise ConfigError("content_items must be >= courses so every course can own content.")
    else:
        resolved_counts = profile_counts(app, profile)
    generation = dict(app.get("generation") or {})
    if generation_overrides:
        generation.update(generation_overrides)

    resolved_seed = int(seed if seed is not None else app["reproducibility"]["default_seed"])
    rng = np.random.default_rng(resolved_seed)
    clock = CalendarClock(
        parse_iso_utc(str(generation.get("calendar_start", "2024-01-01T00:00:00Z"))),
        parse_iso_utc(str(generation.get("calendar_end", "2026-10-22T23:59:59Z"))),
        rng,
    )

    courses = _generate_courses(rng, clock, resolved_counts["courses"])
    skills = _generate_skills(rng, clock, resolved_counts["skills"])
    skill_prerequisites = _generate_prerequisites(
        rng, clock, skills, generation, resolved_seed, app
    )
    content_items = _generate_content(
        rng, clock, courses, skills, resolved_counts["content_items"], generation, skill_prerequisites=skill_prerequisites
    )
    content_skills = _generate_content_skills(
        rng, clock, content_items, skills, generation, courses=courses, skill_prerequisites=skill_prerequisites
    )
    learners = _generate_learners(rng, clock, courses, resolved_counts["learners"])

    return MasterTables(
        seed=resolved_seed,
        profile=profile,
        generator_version=str(generation.get("generator_version", GENERATOR_VERSION)),
        courses=courses,
        learners=learners,
        skills=skills,
        skill_prerequisites=skill_prerequisites,
        content_items=content_items,
        content_skills=content_skills,
    )


def _choice(rng: Generator, values: tuple[str, ...] | list[str], p: list[float] | None = None) -> str:
    index = int(rng.choice(len(values), p=p))
    return values[index]


def _lifecycle(clock: CalendarClock, status: str, *, published_statuses: set[str]) -> dict[str, str]:
    created, updated = clock.ordered_pair()
    created_s = format_timestamp(created)
    updated_s = format_timestamp(updated)
    published_at = created_s if status in published_statuses else ""
    archived_at = updated_s if status in {"archived", "retired", "withdrawn"} else ""
    return {
        "created_at": created_s,
        "updated_at": updated_s,
        "effective_from": created_s,
        "effective_to": archived_at,
        "published_at": published_at,
        "archived_at": archived_at,
    }


def _generate_courses(rng: Generator, clock: CalendarClock, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domain_counts = {domain: 0 for domain in DOMAINS}
    weights = [0.05, 0.85, 0.10]
    for seq in range(1, n + 1):
        domain = DOMAINS[(seq - 1) % len(DOMAINS)]
        local = domain_counts[domain]
        domain_counts[domain] += 1
        status = _choice(rng, ("draft", "published", "archived"), p=weights)
        band = DIFFICULTY_BANDS[min(local, len(DIFFICULTY_BANDS) - 1)]
        life = _lifecycle(clock, status, published_statuses={"published", "archived"})
        rows.append(
            {
                "course_id": format_id("CRS", seq),
                "title": course_title_for(domain, local),
                "domain": domain,
                "difficulty_band": band,
                "status": status,
                "published_at": life["published_at"],
                "archived_at": life["archived_at"] if status == "archived" else "",
                "created_at": life["created_at"],
                "updated_at": life["updated_at"],
                "effective_from": life["effective_from"],
                "effective_to": life["effective_to"] if status == "archived" else "",
            }
        )
    return rows


def _generate_skills(rng: Generator, clock: CalendarClock, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domain_counts = {domain: 0 for domain in DOMAINS}
    for seq in range(1, n + 1):
        domain = DOMAINS[(seq - 1) % len(DOMAINS)]
        local = domain_counts[domain]
        domain_counts[domain] += 1
        bloom = BLOOM_LEVELS[min(local * len(BLOOM_LEVELS) // max(n // len(DOMAINS), 1), len(BLOOM_LEVELS) - 1)]
        status = _choice(rng, ("active", "deprecated"), p=[0.94, 0.06])
        created, updated = clock.ordered_pair()
        rows.append(
            {
                "skill_id": format_id("SKL", seq),
                "skill_name": skill_name_for(domain, local),
                "domain": domain,
                "bloom_level": bloom,
                "status": status,
                "created_at": format_timestamp(created),
                "updated_at": format_timestamp(updated),
                "effective_from": format_timestamp(created),
                "effective_to": format_timestamp(updated) if status == "deprecated" else "",
            }
        )
    names = [(row["skill_name"], row["domain"]) for row in rows]
    if len(names) != len(set(names)):
        raise ConfigError("Generated skill names are not unique within domain.")
    return rows


def _generate_prerequisites(
    rng: Generator,
    clock: CalendarClock,
    skills: list[dict[str, Any]],
    generation: dict[str, Any],
    seed: int,
    app_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    max_prereq = int(generation.get("max_prerequisites_per_skill", 3))
    extra_p = float(generation.get("prerequisite_edge_probability", 0.35))
    cross_p = float(generation.get("cross_domain_prereq_probability", 0.12))
    required_ratio = float(generation.get("required_edge_ratio", 0.80))
    mastery_cfg = (app_config or {}).get("mastery") or {}
    min_mastery = float(
        generation.get("min_mastery_required")
        or mastery_cfg.get("min_prerequisite_mastery", 0.6)
    )

    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for skill in skills:
        by_domain[skill["domain"]].append(skill)

    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    prereq_counts: dict[str, int] = defaultdict(int)

    def add_edge(skill_id: str, prereq_id: str, rel: str) -> bool:
        if skill_id == prereq_id:
            return False
        key = (skill_id, prereq_id)
        if key in seen:
            return False
        if prereq_counts[skill_id] >= max_prereq:
            return False
        seen.add(key)
        prereq_counts[skill_id] += 1
        edges.append((skill_id, prereq_id, rel))
        return True

    for domain in DOMAINS:
        ordered = by_domain[domain]
        for index, skill in enumerate(ordered):
            if index == 0:
                continue
            add_edge(skill["skill_id"], ordered[index - 1]["skill_id"], "required")
            lookback = 2
            while prereq_counts[skill["skill_id"]] < max_prereq and index - lookback >= 0:
                if float(rng.random()) < extra_p:
                    rel = "required" if float(rng.random()) < required_ratio else "recommended"
                    add_edge(skill["skill_id"], ordered[index - lookback]["skill_id"], rel)
                lookback += 1

    domain_index = {domain: i for i, domain in enumerate(DOMAINS)}
    for skill in skills:
        d_i = domain_index[skill["domain"]]
        if d_i == 0:
            continue
        if prereq_counts[skill["skill_id"]] >= max_prereq:
            continue
        if float(rng.random()) >= cross_p:
            continue
        prior_domain = DOMAINS[d_i - 1]
        prior_skills = by_domain[prior_domain]
        if not prior_skills:
            continue
        # Earlier skills in the prior domain only — keeps the graph a DAG.
        pick = prior_skills[int(rng.integers(0, max(1, (len(prior_skills) + 1) // 2)))]
        add_edge(skill["skill_id"], pick["skill_id"], "required")

    dag_edges = [
        PrerequisiteEdge(skill_id=s, prerequisite_skill_id=p, relationship_type=rel)
        for s, p, rel in edges
    ]
    require_dag(dag_edges)

    rows: list[dict[str, Any]] = []
    for skill_id, prereq_id, rel in edges:
        created, _updated = clock.ordered_pair()
        created_s = format_timestamp(created)
        rows.append(
            {
                "skill_id": skill_id,
                "prerequisite_skill_id": prereq_id,
                "relationship_type": rel,
                "min_mastery_required": f"{min_mastery:.4f}",
                "created_at": created_s,
                "effective_from": created_s,
                "effective_to": "",
            }
        )
    rows.sort(key=lambda row: (row["skill_id"], row["prerequisite_skill_id"]))
    return rows


def _generate_content(
    rng: Generator,
    clock: CalendarClock,
    courses: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    n: int,
    generation: dict[str, Any],
    skill_prerequisites: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    types = tuple(generation.get("content_types") or CONTENT_TYPES)
    weights = generation.get("content_type_weights")
    if weights is not None:
        if len(weights) != len(types):
            raise ConfigError("content_type_weights length must match content_types.")
        total = float(sum(weights))
        if total <= 0:
            raise ConfigError("content_type_weights must sum to a positive number.")
        probs = [float(w) / total for w in weights]
    else:
        probs = None

    duration_by_type = {
        "video": (8, 45),
        "article": (10, 40),
        "quiz": (15, 50),
        "exercise": (20, 90),
        "project": (60, 240),
    }
    skills_by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for skill in skills:
        skills_by_domain[skill["domain"]].append(skill)

    # Foundational items designation (1-2 items per course where safely possible)
    num_courses = len(courses)
    num_foundational_per_course = 2 if n >= num_courses * 2 else 1

    rows: list[dict[str, Any]] = []
    for seq in range(1, n + 1):
        course = courses[(seq - 1) % len(courses)]
        item_index_in_course = (seq - 1) // len(courses)
        is_foundational = item_index_in_course < num_foundational_per_course

        content_type = _choice(rng, list(types), p=probs)
        domain_skills = skills_by_domain[course["domain"]] or skills
        topic = domain_skills[(seq - 1) % len(domain_skills)]["skill_name"]
        verb = CONTENT_VERBS[(seq - 1) % len(CONTENT_VERBS)]
        low, high = duration_by_type.get(content_type, (10, 60))
        minutes = int(rng.integers(low, high + 1))

        if is_foundational:
            # Beginner difficulty between 1.00 and 1.50, always published
            difficulty = round(float(rng.uniform(1.00, 1.50)), 2)
            status = "published"
            # Consume random status choice from rng to preserve deterministic stream
            _ = _choice(rng, ("draft", "published", "retired"), p=[0.08, 0.84, 0.08])
        else:
            difficulty = round(float(rng.uniform(1.00, 5.00)), 2)
            status = _choice(rng, ("draft", "published", "retired"), p=[0.08, 0.84, 0.08])

        created, updated = clock.ordered_pair()
        created_s = format_timestamp(created)
        updated_s = format_timestamp(updated)
        rows.append(
            {
                "content_id": format_id("CNT", seq),
                "course_id": course["course_id"],
                "title": f"{verb} {topic}",
                "content_type": content_type,
                "difficulty": f"{difficulty:.2f}",
                "estimated_minutes": str(minutes),
                "status": status,
                "created_at": created_s,
                "updated_at": updated_s,
                "effective_from": created_s,
                "effective_to": updated_s if status == "retired" else "",
            }
        )
    return rows


def _generate_content_skills(
    rng: Generator,
    clock: CalendarClock,
    content_items: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    generation: dict[str, Any],
    courses: list[dict[str, Any]] | None = None,
    skill_prerequisites: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    min_n = int(generation.get("skills_per_content_min", 1))
    max_n = int(generation.get("skills_per_content_max", 4))
    if min_n < 1 or max_n < min_n:
        raise ConfigError("skills_per_content_min/max are invalid.")

    course_domain_map: dict[str, str] = {}
    if courses:
        course_domain_map = {c["course_id"]: c["domain"] for c in courses}

    def domain_for_course_id(course_id: str) -> str:
        if course_id in course_domain_map:
            return course_domain_map[course_id]
        seq = int(course_id.split("-")[1])
        return DOMAINS[(seq - 1) % len(DOMAINS)]

    skills_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for skill in skills:
        skills_by_domain[skill["domain"]].append(skill["skill_id"])
    all_skill_ids = [skill["skill_id"] for skill in skills]

    # Discover root skills with 0 required prerequisites dynamically
    skills_with_req_prereqs: set[str] = set()
    if skill_prerequisites:
        for p in skill_prerequisites:
            if str(p.get("relationship_type", "required")).strip().lower() == "required":
                skills_with_req_prereqs.add(str(p["skill_id"]))

    root_skills_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for s in skills:
        sid = str(s["skill_id"])
        if sid not in skills_with_req_prereqs:
            root_skills_by_domain[s["domain"]].append(sid)

    num_courses = len(courses) if courses else 1
    num_foundational_per_course = 2 if len(content_items) >= num_courses * 2 else 1

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for seq_idx, content in enumerate(content_items, start=1):
        item_index_in_course = (seq_idx - 1) // num_courses
        is_foundational = item_index_in_course < num_foundational_per_course
        domain = domain_for_course_id(content["course_id"])
        domain_roots = root_skills_by_domain.get(domain) or [all_skill_ids[0]]

        if is_foundational:
            # Deterministically assign exactly 1 root skill from the course's domain
            # Round-robin if multiple root skills exist in the domain
            root_skill_id = domain_roots[item_index_in_course % len(domain_roots)]
            chosen = [root_skill_id]
        else:
            pool = list(skills_by_domain[domain]) or list(all_skill_ids)
            k = int(rng.integers(min_n, min(max_n, len(pool)) + 1))
            chosen = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]

        for index, skill_id in enumerate(chosen):
            key = (content["content_id"], skill_id)
            if key in seen:
                continue
            seen.add(key)
            created, _ = clock.ordered_pair()
            if index == 0:
                role = "primary"
                weight = 1.0000
            else:
                role = "supporting"
                weight = round(float(rng.uniform(0.2500, 0.8000)), 4)
            rows.append(
                {
                    "content_id": content["content_id"],
                    "skill_id": skill_id,
                    "skill_role": role,
                    "weight": f"{weight:.4f}",
                    "created_at": format_timestamp(created),
                }
            )
    rows.sort(key=lambda row: (row["content_id"], row["skill_id"]))
    return rows


def _generate_learners(
    rng: Generator,
    clock: CalendarClock,
    courses: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    published = [c for c in courses if c["status"] == "published"] or courses
    rows: list[dict[str, Any]] = []
    status_weights = [0.78, 0.10, 0.07, 0.05]
    level_weights = [0.55, 0.32, 0.13]
    for seq in range(1, n + 1):
        status = _choice(rng, LEARNER_STATUSES, p=status_weights)
        level = _choice(rng, LEARNER_LEVELS, p=level_weights)
        given = GIVEN_NAMES[(seq - 1) % len(GIVEN_NAMES)]
        course = published[(seq - 1) % len(published)]
        current_course = "" if status == "withdrawn" else course["course_id"]
        created, updated = clock.ordered_pair()
        created_s = format_timestamp(created)
        updated_s = format_timestamp(updated)
        cohort_year = 2024 + ((seq - 1) % 3)
        rows.append(
            {
                "learner_id": format_id("LRN", seq),
                "display_name": f"{given} Learner {seq:06d}",
                "cohort_code": f"C{cohort_year}-Q{((seq - 1) % 4) + 1}",
                "starting_level": level,
                "status": status,
                "current_course_id": current_course,
                "timezone": "UTC",
                "created_at": created_s,
                "updated_at": updated_s,
                "effective_from": created_s,
                "effective_to": updated_s if status == "withdrawn" else "",
            }
        )
    return rows
