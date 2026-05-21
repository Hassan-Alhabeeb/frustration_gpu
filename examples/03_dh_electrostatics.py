"""Example 03 — Toggle Debye-Hückel electrostatics on/off.

Run frustration analysis WITH and WITHOUT DH contributing to E_native,
and compare the top-10 highly-frustrated residue lists.

Note: by default `electrostatics_k=4.15` alone is METADATA-ONLY (matches
LAMMPS-AWSEM's analysis convention). The pair-energy column is unchanged
unless you also pass `include_dh_in_e_native=True`. See
`docs/lammps_compat_fixes.md` for the rationale.

Run:
    python examples/03_dh_electrostatics.py
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


def top_pairs_set(result, k: int = 10) -> set:
    top = result.pair_records.nsmallest(k, "FrstIndex")
    return {
        (int(r.Res1), r.ChainRes1, int(r.Res2), r.ChainRes2)
        for r in top.itertuples()
    }


def main() -> None:
    # DH off (default).
    off = compute_frustration(PDB_PATH, mode="configurational")
    # DH on, with the standard k_QQ = 4.15.
    on = compute_frustration(
        PDB_PATH,
        mode="configurational",
        electrostatics_k=4.15,
        include_dh_in_e_native=True,
    )

    print(f"PDB: {PDB_PATH.name}")
    print(f"  DH off:  min(FI) = {off.pair_records['FrstIndex'].min():+.4f}")
    print(f"  DH on:   min(FI) = {on.pair_records['FrstIndex'].min():+.4f}")

    set_off = top_pairs_set(off, k=10)
    set_on = top_pairs_set(on, k=10)
    shared = set_off & set_on
    only_off = set_off - set_on
    only_on = set_on - set_off

    print("\nTop-10 highly-frustrated pairs:")
    print(f"  Shared by both runs:    {len(shared):>2d}")
    print(f"  Unique to DH-off run:   {len(only_off):>2d}")
    print(f"  Unique to DH-on run:    {len(only_on):>2d}")

    if only_off:
        print("  DH-off-only pairs:", sorted(only_off))
    if only_on:
        print("  DH-on-only pairs:",  sorted(only_on))


if __name__ == "__main__":
    main()
