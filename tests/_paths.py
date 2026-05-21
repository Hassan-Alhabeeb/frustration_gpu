"""Shared path constants for the test suite.

``PDB_DIR`` is resolved by trying, in order:

1. The bundled ``tests/data/`` directory (committed to the repo so CI on
   any platform exercises the four-PDB validation panel without external
   downloads).
2. The ``FRUSTRATION_PDB_DIR`` environment variable, if set.
3. The legacy ``F:/research_plan/allosteric/data/pdb_files`` location
   used by the original developer machine.

``DUMP_ROOT`` (LAMMPS reference dump tree) is resolved analogously: the
``benchmark/cpu_baseline/`` directory at the repo root, with the
``FRUSTRATION_DUMP_ROOT`` environment override.

The first existing directory wins.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_pdb_dir() -> Path:
    here = Path(__file__).resolve().parent
    bundled = here / "data"
    if bundled.is_dir() and (bundled / "5AON.pdb").is_file():
        return bundled

    env = os.environ.get("FRUSTRATION_PDB_DIR")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate

    return Path("F:/research_plan/allosteric/data/pdb_files")


def _resolve_dump_root() -> Path:
    env = os.environ.get("FRUSTRATION_DUMP_ROOT")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate

    bundled = _REPO_ROOT / "benchmark" / "cpu_baseline"
    return bundled


PDB_DIR: Path = _resolve_pdb_dir()
DUMP_ROOT: Path = _resolve_dump_root()
