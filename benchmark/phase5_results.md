# Phase 5 stress test + frustrapy benchmark

Generated: 2026-05-21 08:49:20

Hardware: RTX 4070 (12 GB) + Windows 11 host. CPU runs are the local Windows Python process (single-threaded fair-comparison; no multiprocessing). All compute in **float64**, **n_decoys=1000**, **seed=0**.

Frustrapy comparison runs on VM ``root@10.1.0.45`` (EPYC 32-core, no GPU; frustrapy spawns LAMMPS subprocesses, so single-PDB single-threaded is the apples-to-apples).

## 20-PDB panel results

| PDB | N res | mode | CPU (s) | GPU (ms) | Peak VRAM (MB) | Status |
|-----|-------|------|---------|----------|----------------|------|
| 11BG | 248 | configurational | 0.079 | 137.1 | 16 | ok |

## vs frustrapy CPU (head-to-head)

Frustrapy CPU times are single-PDB single-threaded on the VM. Our CPU + GPU times are the timings from the panel table above.

| PDB | N res | mode | frustrapy CPU (s) | ours CPU (s) | ours GPU (ms) | Speedup GPU vs frustrapy |
|-----|-------|------|-------------------|--------------|--------------|--------------------------|
| 5AON | 49 | configurational | 0.220 | n/a | n/a | n/a |
| 11BG | 248 | configurational | 0.320 | 0.079 | 137.1 | 2.3× |
| 1O3S | 200 | configurational | 0.340 | n/a | n/a | n/a |
| 3F9M | 451 | configurational | 0.420 | n/a | n/a | n/a |
| 5AON | 49 | mutational | 0.810 | n/a | n/a | n/a |
| 11BG | 248 | mutational | 7.336 | n/a | n/a | n/a |
| 1O3S | 200 | mutational | 4.812 | n/a | n/a | n/a |
| 3F9M | 451 | mutational | 22.407 | n/a | n/a | n/a |
| 5AON | 49 | singleresidue | 0.263 | n/a | n/a | n/a |
| 11BG | 248 | singleresidue | 0.898 | n/a | n/a | n/a |
| 1O3S | 200 | singleresidue | 0.730 | n/a | n/a | n/a |
| 3F9M | 451 | singleresidue | 2.103 | n/a | n/a | n/a |

## 4PKN stress test (8689 residues)

(no 4PKN runs in this report)


## Numerical validation (Spearman per PDB / mode)

``ref_spearman_fi`` = our per-pair FrstIndex vs LAMMPS dump f_ij. ``ref_density`` = our nHighlyFrst vs LAMMPS 5adens.dat. ``cpu_vs_cuda`` = stability check (should be ~1.0 to machine precision; both paths share the same RNG seed).

| PDB | mode | ref Spearman FI | ref Spearman density | CPU↔GPU Spearman FI | CPU↔GPU max |Δ FI| |
|-----|------|-----------------|----------------------|--------------------|----------------|
| 11BG | configurational | 1.0000 | 0.9760 | 1.0000 | 0.00e+00 |

## Key headline numbers

- Largest successfully run on GPU: **11BG** (N=248 residues, mode=configurational, wall=137 ms, VRAM peak=16 MB).
- Mean GPU speedup vs frustrapy CPU on the panel (1 PDBs): **2.3×** (range 2.3× — 2.3×).
- README headline candidate: **2× faster on RTX 4070** than frustrapy CPU on a typical ~250-res protein.