"""Example 07 - Drop-in replacement for frustrapy users.

Shows that `calculate_frustration` accepts the frustrapy kwarg surface
verbatim: `results_dir` (renamed to `output_dir` internally), `graphics`
(silently consumed), and so on.

For frustrapy users migrating: replace
    import frustrapy
    frustrapy.calculate_frustration(...)
with
    from frustration_gpu import calculate_frustration
    calculate_frustration(...)
and existing scripts should work unchanged.

The example writes outputs into a freshly-created temp directory so a
fresh-clone user sees a clean `git status` after running it. Override
the default output location by setting the ``FRUSTRATION_OUTPUT_DIR``
environment variable.

Run:
    python examples/07_frustrapy_drop_in.py
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frustration_gpu import calculate_frustration  # noqa: E402

_BUNDLED = Path(__file__).resolve().parent.parent / "tests" / "data" / "5AON.pdb"
if _BUNDLED.is_file():
    PDB_PATH = _BUNDLED
else:
    PDB_PATH = Path("F:/research_plan/allosteric/data/pdb_files/5AON.pdb")

_env_out = os.environ.get("FRUSTRATION_OUTPUT_DIR")
if _env_out:
    RESULTS_DIR = Path(_env_out)
else:
    RESULTS_DIR = Path(tempfile.mkdtemp(prefix="frustration_gpu_07_dropin_"))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Frustrapy-style call: results_dir, graphics, debug — all accepted.
    with warnings.catch_warnings():
        warnings.simplefilter("once")  # avoid "filterwarnings = error" in pytest envs
        result = calculate_frustration(
            pdb_file=str(PDB_PATH),
            mode="configurational",
            results_dir=str(RESULTS_DIR),   # frustrapy name, auto-mapped to output_dir
            graphics=False,                  # accepted, ignored
            debug=False,                     # accepted, ignored
            seed=42,
        )

    print(f"PDB: {PDB_PATH.name}")
    print(f"  mode = {result.metadata['mode']}")
    print(f"  results_dir mapped to output_dir = {result.metadata['output_dir']}")
    print(f"  n_pairs = {result.metadata['n_pairs']}")
    print(f"  n_residues = {result.metadata['n_residues']}")

    # The drop-in adapter writes the same LAMMPS-AWSEM-compatible files
    # frustrapy does.
    files = sorted(p.name for p in RESULTS_DIR.glob(f"{PDB_PATH.stem}_*.dat"))
    print(f"  Files emitted: {files}")

    # Show that an unknown kwarg raises a loud error (typo protection).
    try:
        calculate_frustration(pdb_file=str(PDB_PATH), unknwon_kwarg=42)
    except TypeError as e:
        print(f"\nTypo guard works: {type(e).__name__} -> {str(e)[:80]}...")


if __name__ == "__main__":
    main()
