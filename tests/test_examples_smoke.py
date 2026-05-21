"""Smoke-test every ``examples/*.py`` script against the bundled four PDBs.

Finding #71: until now, QUICKSTART.md / examples/ claimed the example
scripts were run by the test suite, but no test actually executed them
end-to-end. This file fixes that.

Each example is invoked as a subprocess (so ``__main__`` semantics
match the documented `python examples/XX.py` invocation), with a 90 s
wall-clock timeout. The CI runner is CPU-only, so GPU-specific
examples (``05_gpu_vs_cpu.py``) gracefully degrade and assert no
exception is raised.

The examples already include fallback path resolution: if
``tests/data/<PDB>.pdb`` is present they use it, otherwise they fall
back to the developer-machine path. CI is the bundled-PDB branch.

Skipping rules:
* If the bundled PDB file is missing, the script will hard-error at
  the parse step; the smoke test then skips with a clear message.
* The 600 s overall timeout is well above the longest example (06_batch
  on 11BG + 5AON, ~5 s on CPU).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO / "examples"
BUNDLED_PDB_DIR = REPO / "tests" / "data"

# All example scripts shipped in examples/. Discover at collection
# time so adding a new example doesn't need a test edit.
EXAMPLE_SCRIPTS = sorted(
    p.name for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_")
)


@pytest.fixture(scope="module")
def bundled_pdbs_present() -> bool:
    """All four bundled PDBs must exist for the smoke tests to be meaningful."""
    needed = ["5AON.pdb", "11BG.pdb", "1O3S.pdb", "3F9M.pdb"]
    return all((BUNDLED_PDB_DIR / n).is_file() for n in needed)


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_runs_to_completion(
    script_name: str, bundled_pdbs_present: bool, tmp_path: Path,
) -> None:
    """Run ``python examples/<script_name>`` end-to-end on the bundled PDBs.

    Asserts:
      * subprocess exit code is 0
      * no traceback in stderr (some examples print warnings, those are OK)

    Examples that write output files (06_batch, 07_frustrapy_drop_in) are
    pointed at ``tmp_path`` via the ``FRUSTRATION_OUTPUT_DIR`` env var so
    the smoke run doesn't pollute the repo with stray ``.dat`` files.
    """
    if not bundled_pdbs_present:
        pytest.skip(
            f"bundled PDBs missing from {BUNDLED_PDB_DIR}; cannot smoke-test "
            "examples (this should not happen on a fresh CI checkout)"
        )

    script_path = EXAMPLES_DIR / script_name
    assert script_path.is_file(), f"example script not found: {script_path}"

    env = {
        # propagate the parent env, but route example output dirs into tmp_path
        # so we don't litter the working tree.
        **dict(_passthrough_env()),
        "FRUSTRATION_OUTPUT_DIR": str(tmp_path),
        # PYTHONUNBUFFERED for live capture in CI logs.
        "PYTHONUNBUFFERED": "1",
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"example {script_name} timed out (>90 s). This is far above the "
            "expected ~5 s budget for any bundled-PDB example; investigate."
        )

    if proc.returncode != 0:
        pytest.fail(
            f"example {script_name} exited with {proc.returncode}.\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr ---\n{proc.stderr[-1500:]}"
        )

    # Loose tracaback check on stderr — examples don't print "Traceback" in
    # the happy path. We allow UserWarning text because example 07
    # intentionally exercises the ``graphics=`` flag's warning.
    if "Traceback (most recent call last):" in proc.stderr:
        pytest.fail(
            f"example {script_name} surfaced a traceback in stderr:\n"
            f"{proc.stderr[-1500:]}"
        )


def _passthrough_env() -> dict[str, str]:
    """Copy a minimal env so the subprocess can find Python + packages."""
    import os
    keys = (
        "PATH", "PYTHONPATH", "TEMP", "TMP", "SYSTEMROOT", "USERPROFILE",
        "USERNAME", "HOME", "LANG", "LC_ALL",
        # CUDA env vars in case the runner has a GPU.
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
        # Frustration-side env vars (so the resolver chain picks the right dir).
        "FRUSTRATION_PDB_DIR", "FRUSTRATION_DUMP_ROOT",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}
