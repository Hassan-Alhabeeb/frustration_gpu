# Quickstart, 10 minutes from clone to first result

From a fresh clone to all three frustration modes on real PDBs, on CPU and (optionally) GPU. Code blocks below are excerpts adapted from `examples/`; each example script is smoke-tested in CI against the four bundled PDBs in `tests/data/` (see `tests/test_examples_smoke.py`), so the code paths shown here are kept honest.

## 1. Install

```bash
git clone https://github.com/Hassan-Alhabeeb/frustration_gpu
cd frustration_gpu
pip install -e .
```

This installs `torch`, `numpy`, `pandas`, and `biopython` (the only runtime dependencies). For CUDA, install the CUDA-matched `torch` wheel from the PyTorch index *before* `pip install -e .`:

```bash
# Example for CUDA 12.4:
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.0"
pip install -e .
```

The package then auto-detects CUDA via `torch.cuda.is_available()`. No further configuration is needed.

## 2. Get a PDB

Any standard PDB file works. The examples reference `5AON.pdb` and `11BG.pdb`, available from the [RCSB Protein Data Bank](https://www.rcsb.org/). For instance:

```bash
curl -O https://files.rcsb.org/download/5AON.pdb
```

## 3. Run `compute_frustration`

```python
from frustration_gpu import compute_frustration

result = compute_frustration("5AON.pdb", mode="configurational")
meta = result.metadata
print(
    f"5AON: {meta['n_residues']} residues, {meta['n_pairs']} native pairs, "
    f"{meta['wall_clock_ms']:.0f} ms on {meta['device']}"
)
```

On a 4070 the warm GPU number for 5AON is around 70 ms. First call includes CUDA warmup; the second call on the same process is much faster.

## 4. Inspect the result

`compute_frustration` returns a `FrustrationResult` dataclass with three dataframes (`pair_records`, `singleresidue_records`, `density_records`) plus a metadata dict.

```python
# The 5 lowest FI values = the 5 most highly frustrated contacts.
top5 = result.pair_records.nsmallest(5, "FrstIndex")
print(top5[["Res1", "ChainRes1", "AA1",
            "Res2", "ChainRes2", "AA2",
            "r_ij", "FrstIndex", "FrstState"]].to_string(index=False))
```

`FrstState` is the Ferreiro 2007 three-state classification: `highly` (FI <= -1.0), `neutral` (-1.0 < FI < 0.78), or `minimally` (FI >= 0.78).

## 5. Try all three modes

See `examples/02_three_modes.py` for the runnable version. The core loop is:

```python
from frustration_gpu import compute_frustration

for mode in ("configurational", "mutational", "singleresidue"):
    result = compute_frustration("11BG.pdb", mode=mode)
    if mode == "singleresidue":
        sr = result.singleresidue_records
        n_high = (sr["FrstIndex"] <= -1.0).sum()
        print(f"{mode}: {len(sr)} residues, {n_high} highly frustrated, "
              f"{result.metadata['wall_clock_ms']:.0f} ms")
    else:
        pr = result.pair_records
        n_high = (pr["FrstState"] == "highly").sum()
        print(f"{mode}: {len(pr)} pairs, {n_high} highly frustrated, "
              f"{result.metadata['wall_clock_ms']:.0f} ms")
```

Mode meanings:

- `configurational`: scalar decoy ensemble (one mean, one std for the whole protein). Cheapest; matches frustrapy's default mode.
- `mutational`: per-pair decoy statistics; each pair gets its own decoy mean/std. ~5-10x more expensive than configurational.
- `singleresidue`: per-residue FI; sums water-mediated contributions across all in-contact partners.

## 6. Enable GPU

If CUDA is available, `device="auto"` (the default) selects it automatically. To be explicit:

```python
result_cpu = compute_frustration("11BG.pdb", device="cpu", seed=0)
result_gpu = compute_frustration("11BG.pdb", device="cuda", seed=0)

# CPU and CUDA agree to machine precision at the same seed:
import numpy as np
diff = np.abs(
    result_cpu.pair_records["FrstIndex"].values
    - result_gpu.pair_records["FrstIndex"].values
).max()
print(f"max |FI_CPU - FI_CUDA| = {diff:.2e}")    # observed = 0.0
```

Both paths run in float64. CPU and CUDA share the same RNG seed and the same numerical algorithm. At the default `precision=3` (LAMMPS's `%8.3f`) the rounded FI values are exactly identical; the underlying high-precision FI differs only by reduction-order ULP drift in `decoy_mean` / `decoy_std` (~1e-15 in absolute terms, well below the rounded `0.000`). See `benchmark/phase5_spearman.csv` for the full 30-combo table.

## 7. Optional: Debye-Huckel electrostatics

Default behaviour matches frustratometeR's analysis convention: Debye-Huckel is NOT added to `E_native` even when LAMMPS-AWSEM ran with `huckel_flag=true` (verified empirically against the `electro_4p15` reference dumps). To opt in (verbatim from `examples/03_dh_electrostatics.py`):

```python
off = compute_frustration("11BG.pdb", mode="configurational")
on  = compute_frustration(
    "11BG.pdb",
    mode="configurational",
    electrostatics_k=4.15,
    include_dh_in_e_native=True,
)
print(f"DH off: min(FI) = {off.pair_records['FrstIndex'].min():+.4f}")
print(f"DH on:  min(FI) = {on.pair_records['FrstIndex'].min():+.4f}")
```

Setting `electrostatics_k` without `include_dh_in_e_native=True` is metadata-only: the value is recorded in `result.metadata["electrostatics_k"]` but not used in the per-pair energy.

## 8. Migrating from frustrapy

If you have existing frustrapy scripts, swap one import (full runnable form in `examples/07_frustrapy_drop_in.py`):

```python
# Before:
import frustrapy
result = frustrapy.calculate_frustration(
    pdb_file="X.pdb", mode="mutational",
    results_dir="out/", graphics=True,
)

# After:
from frustration_gpu import calculate_frustration
result = calculate_frustration(
    pdb_file="X.pdb", mode="mutational",
    results_dir="out/",   # automatically mapped to output_dir
    # graphics=True silently consumed (warns once)
)
```

`calculate_frustration` accepts the full frustrapy kwarg surface: `results_dir`, `graphics`, `visualization`, `debug`, `pbar`, `pdb_id`, `is_mutation_calculation`. Renamed kwargs are auto-translated. Unsupported kwargs (visualization, auto-download) are accepted and ignored with a one-time warning. Unknown kwargs raise `TypeError` to catch typos.

Full kwarg-by-kwarg comparison: [docs/frustrapy_vs_us.md](docs/frustrapy_vs_us.md).

## 9. Next

- More examples: [examples/](examples/) (`03_dh_electrostatics.py`, `04_chain_filter.py`, `05_gpu_vs_cpu.py`, `06_batch.py`, `07_frustrapy_drop_in.py`).
- Numerical validation evidence and reproduction commands: [VALIDATION.md](VALIDATION.md).
- Full API reference: [docs/API.md](docs/API.md).
- Phase-by-phase development log: [PHASES_ROADMAP.md](PHASES_ROADMAP.md).
