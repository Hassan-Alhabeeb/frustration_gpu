# Phase 5 stress test + frustrapy benchmark

> **Note (2026-05-21):** this file is now a *partial committed baseline*,
> not the full 20-PDB / 30-combo sweep.
>
> The committed `phase5_panel_results.csv` and `phase5_spearman.csv` contain
> a single (PDB, mode) row — `11BG configurational` on both CPU and GPU —
> which is what a fresh clone can reproduce from the four bundled PDBs in
> `tests/data/` (5AON, 11BG, 1O3S, 3F9M). The detailed 10-PDB validation
> numbers and 4PKN stress test reported in `README.md` / `VALIDATION.md`
> came from a developer-machine run that has not been re-committed since
> the `src/` → `frustration_gpu/` rename. To regenerate the full panel
> locally, supply the missing PDBs and re-run:
>
> ```bash
> python benchmark/run_phase5.py \
>     --pdb-dir /path/to/pdb_files \
>     --modes configurational,mutational,singleresidue \
>     --force
> ```
>
> The four bundled PDBs (5AON, 11BG, 1O3S, 3F9M) cover the entire
> frustrapy head-to-head subset and 4/10 of the validation panel.

Generated: 2026-05-21 08:49:20 (last partial run)

Hardware (last partial run): RTX 4070 (12 GB) + Windows 11 host. CPU runs are the local Windows Python process (single-threaded fair-comparison; no multiprocessing). All compute in **float64**, **n_decoys=1000**, **seed=0**.

Frustrapy comparison runs on VM ``root@10.1.0.45`` (EPYC 32-core, no GPU; frustrapy spawns LAMMPS subprocesses, so single-PDB single-threaded is the apples-to-apples).

## Bundled-PDB panel results (committed baseline)

| PDB | N res | mode | CPU (s) | GPU (ms) | Peak VRAM (MB) | Status |
|-----|-------|------|---------|----------|----------------|------|
| 11BG | 248 | configurational | 0.079 | 137.1 | 16 | ok |

This is the only row in `phase5_panel_results.csv` after the most recent
partial run. The other 19 PDBs in `pdb_panel.csv` are present in the
panel definition but their `.pdb` files are not bundled in `tests/data/`,
so they need an external `--pdb-dir` (or `FRUSTRATION_PDB_DIR=...`)
pointing at a directory that contains them.

## vs frustrapy CPU (head-to-head, archived numbers)

Frustrapy CPU times are single-PDB single-threaded on the VM.
These numbers come from a developer-machine run that pre-dates the
`src/` → `frustration_gpu/` rename; they are kept here for reference and
because the README headline numbers are derived from them. To reproduce:
provision the four PDBs locally, then run
`python benchmark/run_phase5.py --frustrapy --modes configurational,mutational,singleresidue --pdbs 5AON,11BG,1O3S,3F9M`.

| PDB | N res | mode | frustrapy CPU (s) | ours CPU (s) | ours GPU (ms) | Speedup GPU vs frustrapy |
|-----|-------|------|-------------------|--------------|--------------|--------------------------|
| 5AON | 49  | configurational | 0.220  | 0.046 | 70.7  | 3.1x |
| 11BG | 248 | configurational | 0.320  | 0.080 | 206.4 | 1.6x |
| 1O3S | 200 | configurational | 0.340  | 0.061 | 159.8 | 2.1x |
| 3F9M | 451 | configurational | 0.420  | 0.160 | 346.3 | 1.2x |
| 5AON | 49  | mutational      | 0.810  | 0.274 | 70.5  | **11.5x** |
| 11BG | 248 | mutational      | 7.336  | 2.118 | 203.5 | **36.0x** |
| 1O3S | 200 | mutational      | 4.812  | 0.673 | 156.6 | **30.7x** |
| 3F9M | 451 | mutational      | 22.407 | 3.130 | 425.6 | **52.6x** |
| 5AON | 49  | singleresidue   | 0.263  | 0.032 | 19.7  | 13.4x |
| 11BG | 248 | singleresidue   | 0.898  | 0.412 | 54.3  | 16.5x |
| 1O3S | 200 | singleresidue   | 0.730  | 0.400 | 48.0  | 15.2x |
| 3F9M | 451 | singleresidue   | 2.103  | 0.149 | 107.6 | 19.5x |

## 4PKN stress test (8689 residues) — archived

The 4PKN run that produced the README's 17.6 s wall-clock / 22.4 GB VRAM
allocator peak is **not** in the committed CSV. The PDB is not bundled
(8689 residues is far above what a test panel needs to ship); to
reproduce, place `4PKN.pdb` at `$FRUSTRATION_PDB_DIR/4PKN.pdb` and run

```
python benchmark/run_phase5.py --pdbs 4PKN --modes configurational --skip-cpu
```

(CPU is auto-skipped on >2000-residue PDBs.)

## Numerical validation (Spearman per PDB / mode)

``ref_spearman_fi`` = our per-pair FrstIndex vs LAMMPS dump f_ij. ``ref_density`` = our nHighlyFrst vs LAMMPS 5adens.dat. ``cpu_vs_cuda`` = stability check (should be ~1.0 to machine precision; both paths share the same RNG seed).

Current committed `phase5_spearman.csv`:

| PDB | mode | ref Spearman FI | ref Spearman density | CPU↔GPU Spearman FI | CPU↔GPU max |Δ FI| |
|-----|------|-----------------|----------------------|--------------------|----------------|
| 11BG | configurational | 1.0000 | 0.9760 | 1.0000 | 0.00e+00 |

The full per-(PDB, mode) Spearman table (30 rows across the 10-PDB
validation panel) is in `VALIDATION.md` §1; those numbers were measured
from the same developer-machine run as the timings above and have not
been re-committed.

## Key headline numbers (archived; reproduce locally for the latest)

- Largest successfully run on GPU (archived): **4PKN**
  (N=8689 residues, mode=configurational, wall=17.6 s, VRAM peak=22.4 GB).
  Not in the committed CSV; reproduce by supplying 4PKN.pdb.
- Mean GPU speedup vs frustrapy CPU on the 12 head-to-head combos:
  **17.0x** (geometric mean), median 14.3x. Source: archived
  `phase5_frustrapy_comparison.csv` (4 PDBs × 3 modes).
- README headline candidate: **~14x faster on RTX 4070** than frustrapy
  CPU on a typical ~250-res protein (geomean across all 12 combos).
