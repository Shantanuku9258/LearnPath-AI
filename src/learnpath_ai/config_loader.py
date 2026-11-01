"""Load `config/default.json` without touching the database."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from learnpath_ai.settings import project_root


class ConfigError(ValueError):
    """Invalid application or generation configuration."""


def default_config_path() -> Path:
    return project_root() / "config" / "default.json"


def load_app_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_config_path()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("Application config must be a JSON object.")
    return payload


def profile_counts(config: dict[str, Any], profile: str) -> dict[str, int]:
    profiles = config.get("dataset_profiles") or {}
    if profile not in profiles:
        known = ", ".join(sorted(profiles)) or "(none)"
        raise ConfigError(f"Unknown dataset profile {profile!r}. Known: {known}.")
    raw = profiles[profile]
    required = ("learners", "courses", "skills", "content_items")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"Profile {profile!r} missing keys: {', '.join(missing)}")
    counts = {key: int(raw[key]) for key in required}
    for key, value in counts.items():
        if value < 1:
            raise ConfigError(f"Profile {profile!r} {key} must be >= 1, got {value}.")
    if counts["skills"] < 2:
        raise ConfigError("At least 2 skills are required to form a prerequisite graph.")
    if counts["content_items"] < counts["courses"]:
        raise ConfigError("content_items must be >= courses so every course can own content.")
    return counts


def merged_generation(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    generation = deepcopy(config.get("generation") or {})
    if overrides:
        generation.update(overrides)
    return generation
