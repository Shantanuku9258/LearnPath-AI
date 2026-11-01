"""Deterministic UTC timestamps derived from a seeded RNG."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from numpy.random import Generator

UTC = timezone.utc


def parse_iso_utc(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    naive = value.astimezone(UTC).replace(tzinfo=None)
    return naive.strftime("%Y-%m-%d %H:%M:%S.%f")


class CalendarClock:
    def __init__(self, start: datetime, end: datetime, rng: Generator) -> None:
        if end <= start:
            raise ValueError("calendar_end must be after calendar_start.")
        self.start = start.astimezone(UTC)
        self.end = end.astimezone(UTC)
        self.rng = rng
        self.span_seconds = int((self.end - self.start).total_seconds())

    def instant(self) -> datetime:
        offset = int(self.rng.integers(0, self.span_seconds + 1))
        micros = int(self.rng.integers(0, 1_000_000))
        return self.start + timedelta(seconds=offset, microseconds=micros)

    def ordered_pair(self) -> tuple[datetime, datetime]:
        first = self.instant()
        extra = int(self.rng.integers(0, min(86_400 * 14, self.span_seconds) + 1))
        second = first + timedelta(seconds=extra)
        if second > self.end:
            second = self.end
        if second < first:
            second = first
        return first, second
