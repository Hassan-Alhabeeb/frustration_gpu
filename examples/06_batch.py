"""Example 06 - Batch process a directory of PDBs.

Walks a directory of `*.pdb` files, runs configurational frustration on
each, and writes a per-pair CSV next to each input PDB.

The example writes outputs into a freshly-created temp directory so a
fresh-clone user sees a clean `git status` after running it. Override
the default output location by setting the ``FRUSTRATION_OUTPUT_DIR``
environment variable.

Run:
    python examples/06_batch.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frustration_gpu import compute_frustration  # noqa: E402

# Tweak this to point at any directory of PDB files. Defaults to the
# bundled four-PDB validation panel committed to the repo for tests.
INPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
if not INPUT_DIR.is_dir():
    INPUT_DIR = Path("F:/research_plan/allosteric/data/pdb_files")
PDB_LIST = ["5AON.pdb", "11BG.pdb"]   # short list so the example finishes quickly

_env_out = os.environ.get("FRUSTRATION_OUTPUT_DIR")
if _env_out:
    OUTPUT_DIR = Path(_env_out)
else:
    OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="frustration_gpu_06_batch_"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pdb_name in PDB_LIST:
        pdb_path = INPUT_DIR / pdb_name
        if not pdb_path.exists():
            print(f"  SKIP {pdb_name} (not found)")
            continue

        t0 = time.perf_counter()
        result = compute_frustration(pdb_path, mode="configurational")
        elapsed = (time.perf_counter() - t0) * 1000.0

        csv_path = OUTPUT_DIR / f"{pdb_path.stem}_pairs.csv"
        result.pair_records.to_csv(csv_path, index=False)

        summary_rows.append({
            "pdb": pdb_path.stem,
            "n_residues": result.metadata["n_residues"],
            "n_pairs": result.metadata["n_pairs"],
            "wall_clock_ms": elapsed,
            "csv_path": str(csv_path),
        })
        print(
            f"  {pdb_path.stem}: {result.metadata['n_residues']:>4d} res, "
            f"{result.metadata['n_pairs']:>5d} pairs, "
            f"{elapsed:>6.0f} ms  ->  {csv_path.name}"
        )

    # Save a top-level summary CSV.
    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nSummary -> {OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
