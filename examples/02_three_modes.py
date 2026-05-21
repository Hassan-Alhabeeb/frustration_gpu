"""Example 02 — All three frustration modes on one PDB.

Runs configurational, mutational, and singleresidue frustration on the
same structure and reports a summary line per mode.

Run:
    python examples/02_three_modes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frustration_gpu import compute_frustration  # noqa: E402

_bundled = Path(__file__).resolve().parent.parent / "tests" / "data" / "11BG.pdb"
if _bundled.is_file():
    PDB_PATH = _bundled
else:
    PDB_PATH = Path("F:/research_plan/allosteric/data/pdb_files/11BG.pdb")
MODES = ("configurational", "mutational", "singleresidue")


def summarise_pair_mode(name: str, result) -> None:
    pr = result.pair_records
    n_high = (pr["FrstState"] == "highly").sum()
    n_min = (pr["FrstState"] == "minimally").sum()
    print(
        f"  {name:<15s} {len(pr):>5d} pairs   "
        f"highly={n_high:>4d}  minimally={n_min:>4d}   "
        f"min(FI)={pr['FrstIndex'].min():+.3f}  max(FI)={pr['FrstIndex'].max():+.3f}   "
        f"{result.metadata['wall_clock_ms']:>5.0f} ms"
    )


def summarise_singleresidue(name: str, result) -> None:
    sr = result.singleresidue_records
    n_high = (sr["FrstIndex"] <= -1.0).sum()
    n_min = (sr["FrstIndex"] >= 0.78).sum()
    print(
        f"  {name:<15s} {len(sr):>5d} resids  "
        f"highly={n_high:>4d}  minimally={n_min:>4d}   "
        f"min(FI)={sr['FrstIndex'].min():+.3f}  max(FI)={sr['FrstIndex'].max():+.3f}   "
        f"{result.metadata['wall_clock_ms']:>5.0f} ms"
    )


def main() -> None:
    print(f"PDB: {PDB_PATH.name}")
    for mode in MODES:
        result = compute_frustration(PDB_PATH, mode=mode)
        if mode == "singleresidue":
            summarise_singleresidue(mode, result)
        else:
            summarise_pair_mode(mode, result)


if __name__ == "__main__":
    main()
