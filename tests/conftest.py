"""Shared pytest fixtures for the frustration_gpu test suite.

Centralises path resolution and exposes a ``pdb_dir`` session-scoped fixture.

Path-constant module: see ``tests/_paths.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure ``tests/`` is on sys.path so test modules can do
# ``from _paths import PDB_DIR``. pytest does not add the directory of
# the test file when no ``__init__.py`` is present, so we add it here.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import PDB_DIR  # noqa: E402

__all__ = ["PDB_DIR", "pdb_dir"]


@pytest.fixture(scope="session")
def pdb_dir() -> Path:
    """Directory containing reference PDB files for tests."""
    return PDB_DIR
