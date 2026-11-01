"""Live MySQL tests. These do not run unless explicitly enabled.

Set LEARNPATH_MYSQL_IT=1 and provide a real `.env` before expecting these to run.
They are skipped in Phase 1 because MySQL 8.4 LTS is not installed and no
application credentials were configured.
"""

from __future__ import annotations

import os

import pytest

from learnpath_ai.database import Database
from learnpath_ai.settings import SettingsError, load_database_settings

pytestmark = pytest.mark.mysql


def _enabled() -> bool:
    return os.environ.get("LEARNPATH_MYSQL_IT", "").strip() == "1"


@pytest.mark.skipif(not _enabled(), reason="LEARNPATH_MYSQL_IT is not set to 1")
def test_live_ping() -> None:
    try:
        settings = load_database_settings()
    except SettingsError as exc:
        pytest.skip(f"database settings unavailable: {exc}")
    db = Database(settings)
    try:
        result = db.ping()
    finally:
        db.dispose()
    assert result.server_version
    assert result.host
