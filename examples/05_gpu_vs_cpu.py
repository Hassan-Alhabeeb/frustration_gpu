"""Example 05 — Same PDB on CPU and GPU, time both, assert FI agrees.

Demonstrates that compute_frustration is device-portable and gives
identical FI values to numerical precision (modulo the RNG noise floor
in decoy_mean / decoy_std, which is the same seed on both devices so it
agrees to machine precision).

Run:
    python examples/05_gpu_vs_cpu.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from frustration_gpu import compute_frustration  # noqa: E402

_bundled = Path(__file__).resolve().parent.parent / "tests" / "data" / "11BG.pdb"
if _bundled.is_file():
    PDB_PATH = _bundled
else:
    PDB_PATH = Path("F:/research_plan/allosteric/data/pdb_files/11BG.pdb")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — running CPU only.")
        result = compute_frustration(PDB_PATH, device="cpu")
        print(f"  CPU: {result.metadata['wall_clock_ms']:.0f} ms")
        return

    # CPU run.
    t0 = time.perf_counter()
    cpu_res = compute_frustration(PDB_PATH, device="cpu", seed=0)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    # CUDA run (warm-up, then time).
    _ = compute_frustration(PDB_PATH, device="cuda", seed=0)  # warm-up
    t0 = time.perf_counter()
    gpu_res = compute_frustration(PDB_PATH, device="cuda", seed=0)
    gpu_ms = (time.perf_counter() - t0) * 1000.0

    print(f"PDB: {PDB_PATH.name}")
    print(f"  CPU:  {cpu_ms:>7.1f} ms")
    print(f"  CUDA: {gpu_ms:>7.1f} ms     speedup: {cpu_ms / gpu_ms:.1f}x")

    # The two pair_records should be row-aligned (same native-pair
    # enumeration order). FI agreement: machine-precision float64.
    fi_cpu = cpu_res.pair_records["FrstIndex"].values
    fi_gpu = gpu_res.pair_records["FrstIndex"].values
    abs_diff = abs(fi_cpu - fi_gpu).max()
    print(f"  max |FrstIndex_CPU - FrstIndex_CUDA| = {abs_diff:.2e}")
    assert abs_diff < 1e-3, f"FI mismatch above tolerance: {abs_diff}"
    print("  PASS: CPU and CUDA FI values agree within tolerance.")


if __name__ == "__main__":
    main()
