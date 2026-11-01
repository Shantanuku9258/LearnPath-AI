"""Shared pytest fixtures.

Unit tests never require a live MySQL server.
Integration tests must opt in with LEARNPATH_MYSQL_IT=1.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "mysql: tests that open a real MySQL connection"
    )
