"""Example 01 — Minimal use case.

Load 5AON, run configurational frustration analysis, print the top-5
most highly-frustrated native contacts.

Run:
    python examples/01_basic.py
"""
import sys
from pathlib import Path

# Make `src` importable when running this script directly from the
# repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frustration_gpu import compute_frustration  # noqa: E402

# Adjust this path if your local 5AON.pdb lives elsewhere.
_bundled = Path(__file__).resolve().parent.parent / "tests" / "data" / "5AON.pdb"
if _bundled.is_file():
    PDB_PATH = _bundled
else:
    PDB_PATH = Path("F:/research_plan/allosteric/data/pdb_files/5AON.pdb")


def main() -> None:
    result = compute_frustration(PDB_PATH, mode="configurational")
    meta = result.metadata
    print(
        f"5AON: {meta['n_residues']} residues, {meta['n_pairs']} native pairs, "
        f"{meta['wall_clock_ms']:.0f} ms on {meta['device']}"
    )

    # The 5 lowest FI values = the 5 most highly frustrated contacts.
    top5 = result.pair_records.nsmallest(5, "FrstIndex")
    print("\nTop-5 highly frustrated native contacts:")
    print(
        top5[
            [
                "Res1", "ChainRes1", "AA1",
                "Res2", "ChainRes2", "AA2",
                "r_ij", "FrstIndex", "FrstState",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
