"""Deterministic public identifiers."""

from __future__ import annotations

import re

PREFIX_PATTERNS = {
    "LRN": re.compile(r"^LRN-[0-9]{6}$"),
    "CRS": re.compile(r"^CRS-[0-9]{6}$"),
    "SKL": re.compile(r"^SKL-[0-9]{6}$"),
    "CNT": re.compile(r"^CNT-[0-9]{6}$"),
    "ATT": re.compile(r"^ATT-LRN-[0-9]{6}-[0-9]{6}$"),
    "RES": re.compile(r"^RES-[0-9]{6}$"),
    "MST": re.compile(r"^MST-[0-9]{6}$"),
    "DQI": re.compile(r"^DQI-[0-9]{6}$"),
    "REC": re.compile(r"^REC-[0-9]{6}$"),
    "PATH": re.compile(r"^PATH-[0-9]{4}-[0-9]{6}$"),
    "LPI": re.compile(r"^LPI-[0-9]{6}$"),
}


class IdError(ValueError):
    """Invalid identifier request or format."""


def format_id(prefix: str, sequence: int, *, width: int = 6) -> str:
    if prefix not in PREFIX_PATTERNS:
        raise IdError(f"Unsupported ID prefix: {prefix}")
    if sequence < 1:
        raise IdError(f"ID sequence must be >= 1, got {sequence}")
    return f"{prefix}-{sequence:0{width}d}"


def format_attempt_id(learner_id: str, sequence: int, *, width: int = 6) -> str:
    if not is_valid_id(learner_id, "LRN"):
        raise IdError(f"Invalid learner_id for attempt: {learner_id}")
    if sequence < 1:
        raise IdError(f"Attempt sequence must be >= 1, got {sequence}")
    return f"ATT-{learner_id}-{sequence:0{width}d}"


def is_valid_id(value: str, prefix: str) -> bool:
    pattern = PREFIX_PATTERNS.get(prefix)
    if pattern is None:
        return False
    return bool(pattern.fullmatch(value))
