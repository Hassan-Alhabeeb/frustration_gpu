# frustration-gpu

Pure-PyTorch reimplementation of LAMMPS-AWSEM frustration analysis. 14 to 53x faster on a single GPU, byte-comparable to frustratometeR / frustrapy.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-223_passing-brightgreen.svg)](tests/)

```python
from frustration_gpu import compute_frustration

result = compute_frustration("5AON.pdb", mode="configurational")
print(result.pair_records.head())
```

Same physics as frustratometeR. 14x faster (median) than frustrapy on CPU, 53x faster on GPU for a 451-residue protein in mutational mode. FI Spearman is >= 0.9975 against the LAMMPS-AWSEM C++ reference on every panel run.

## Why

frustratometeR / frustrapy is the canonical implementation of Ferreiro-Wolynes frustration analysis, but it wraps a precompiled LAMMPS-AWSEM binary that has no GPU support. On large proteins, mutational-mode analysis runs ~N x 20 LAMMPS subprocesses and takes tens of seconds per structure, prohibitive at proteome scale.

This package re-derives the AWSEM Hamiltonian (water + burial + Debye-Huckel) and the Ferreiro frustration index in pure PyTorch, batched as tensor operations that run identically on CPU and CUDA. The math is traceable line-by-line to the upstream `fix_backbone.cpp`: no shortcuts, no approximations, no alternative physics.

## Benchmarks (Phase 5, 20-PDB panel)

Hardware: AMD Ryzen 9 5900X + RTX 4070 (12 GB), Windows 11 host, PyTorch 2.6.0+cu124. Frustrapy runs on a 32-core EPYC VM (`root@10.1.0.45`), single-PDB single-threaded, the apples-to-apples comparison that frustrapy users experience.

### Head-to-head vs frustrapy CPU (4 PDBs x 3 modes)

| PDB | N res | mode | frustrapy CPU (s) | ours GPU (ms) | Speedup |
|---|---|---|---|---|---|
| 5AON | 49  | configurational | 0.220  | 70.7  | 3.1x |
| 11BG | 248 | configurational | 0.320  | 206.4 | 1.6x |
| 1O3S | 200 | configurational | 0.340  | 159.8 | 2.1x |
| 3F9M | 451 | configurational | 0.420  | 346.3 | 1.2x |
| 5AON | 49  | mutational      | 0.810  | 70.5  | **11.5x** |
| 11BG | 248 | mutational      | 7.336  | 203.5 | **36.0x** |
| 1O3S | 200 | mutational      | 4.812  | 156.6 | **30.7x** |
| 3F9M | 451 | mutational      | 22.407 | 425.6 | **52.6x** |
| 5AON | 49  | singleresidue   | 0.263  | 19.7  | 13.4x |
| 11BG | 248 | singleresidue   | 0.898  | 54.3  | 16.5x |
| 1O3S | 200 | singleresidue   | 0.730  | 48.0  | 15.2x |
| 3F9M | 451 | singleresidue   | 2.103  | 107.6 | 19.5x |

Configurational mode is already fast in frustrapy (one scalar decoy ensemble), so the GPU speedup there is modest (1.2-3.1x), workload dominated by Python and CUDA-launch overhead. Mutational mode is where the GPU port wins (30-53x): frustrapy fans out ~N x 20 LAMMPS subprocesses while we run one fused-alpha tensor pass on GPU.

Geometric-mean speedup across the 12 head-to-head runs: 17.0x (median 14.3x).

> **Hardware comparison footnote.** Our timings come from a Windows 11 host with an AMD Ryzen 9 5900X plus an RTX 4070 (12 GB). frustrapy timings come from a Linux EPYC VM (32 cores allocated, no GPU). Both runs use single-threaded execution and identical kwargs (`n_decoys=1000`, `seq_dist=12`, `electrostatics_k=None`). The CPU-vs-CPU column is therefore biased by hardware mismatch and is not directly comparable; the headline GPU-vs-CPU comparison (our RTX 4070 vs frustrapy on EPYC CPU) is fair as a real-world deployment comparison: both numbers reflect what users on the typical hardware see.

Raw numbers: `benchmark/phase5_panel_results.csv`, `benchmark/phase5_frustrapy_comparison.csv`. Full writeup: `benchmark/phase5_results.md`.

### Stress test: 4PKN (8,689 residues)

| metric | value |
|---|---|
| N residues parsed | 8,689 |
| N native pairs | 62,911 |
| GPU wall-clock (configurational) | 17.6 s |
| FrstState distribution | 4,350 highly / 41,469 neutral / 17,092 minimally |

`torch.cuda.max_memory_allocated` reported a 22.4 GB high-water mark, exceeding the RTX 4070's 12 GB physical VRAM. This is the PyTorch caching allocator's transient peak (caches plus intermediates that are not all alive simultaneously), not a sustained working set. Real WDDM paging would dominate the wall-clock; the observed 17.6 s is consistent with the effective resident set fitting in VRAM. Alpha-chunking the auxiliary (N, N) tensors built outside the alpha-loop will reduce the reported peak below 8 GB at this size, tracked as a Phase 6 polish item.

## Install

```bash
git clone https://github.com/Hassan-Alhabeeb/frustration_gpu
cd frustration_gpu
pip install -e .
```

For CUDA, install the matching `torch` wheel from the PyTorch index first (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)); the package then auto-detects CUDA via `torch.cuda.is_available()`.

## Quickstart

```python
from frustration_gpu import compute_frustration

# Default is byte-comparable to LAMMPS-AWSEM + frustratometeR on a clean PDB.
result = compute_frustration("5AON.pdb", mode="configurational", device="auto")

# Top-5 most highly frustrated contacts
top5 = result.pair_records.nsmallest(5, "FrstIndex")
print(top5[["Res1", "Res2", "AA1", "AA2", "r_ij", "FrstIndex", "FrstState"]])

# Per-residue 5adens density
print(result.density_records.head())
```

10-minute walkthrough: [QUICKSTART.md](QUICKSTART.md). Validation evidence: [VALIDATION.md](VALIDATION.md).

## Status

- Configurational, mutational, singleresidue modes: all three implemented and validated.
- FI Spearman >= 0.9975 vs LAMMPS reference on 30/30 panel runs (lowest is 0.9975 on 3F9M singleresidue).
- CPU <-> GPU agreement at machine precision: max \|delta-FI\| = 0.0 (literal) on all 30 reference combinations.
- Drop-in `calculate_frustration(...)` alias for frustrapy users (`results_dir`, `graphics`, etc. accepted).
- Multi-chain handling, chain filter, residue subset filter, opt-in Debye-Huckel electrostatics.
- LAMMPS-compat flags (`include_dna`, `lammps_compat_altloc`) reproduce frustratometeR's 5adens output byte-comparable on protein-DNA and alt-conformer PDBs.
- Scales to 4PKN (8,689 residues) end-to-end on a single RTX 4070.
- Alpha-chunking for sub-12 GB VRAM at N >= 4,000 (Phase 6 polish, in progress).
- VMD `.tcl` / PyMOL `.pml` visualization output (Phase 6 polish).

## Features

- **Pure PyTorch.** No LAMMPS binary, no OpenMM dependency, no R, no precompiled C++. Runs anywhere `torch` does.
- **Float64 throughout.** No precision compromises for speed; CPU and CUDA agree to machine precision on V_water, V_burial, V_DH.
- **Deterministic.** Reproducible RNG via `seed=...`; frustrapy uses unseeded `srand(NULL)`.
- **Three modes.** Configurational (scalar decoy ensemble), mutational (per-pair decoy stats), singleresidue (per-residue FI).
- **Byte-comparable output.** Default writes `<stem>_tertiary_frustration.dat`, `<stem>_5adens.dat`, etc. in the same column schema as LAMMPS-AWSEM.
- **Explicit opt-in flags** for known frustrapy bugs (`include_dna`, `lammps_compat_altloc`). Defaults are physically clean, opt-ins reproduce the LAMMPS reference exactly.

## Caveats

- For very small proteins (N < 100) CUDA launch overhead can make GPU slower than CPU; use `device="cpu"`.
- One panel cell (5N9R mutational, N=356) shows GPU 33% slower than CPU. See VALIDATION.md section 7.
- Large proteins (>5,000 residues) report a high VRAM allocator peak; alpha-chunking work in progress.
- Sparse or fragmented structures can trip the rejection-sampler fallback and emit a `frustration_gpu.decoys` warning; FI Spearman is unaffected on dense structures but per-pair decoy stats may be noisier.

## Documentation

- [QUICKSTART.md](QUICKSTART.md), 10-minute end-to-end walkthrough.
- [VALIDATION.md](VALIDATION.md), numerical validation evidence, reproduction instructions.
- [docs/API.md](docs/API.md), full public function reference.
- [docs/frustrapy_vs_us.md](docs/frustrapy_vs_us.md), kwarg-by-kwarg migration guide.
- [docs/lammps_compat_fixes.md](docs/lammps_compat_fixes.md), the four LAMMPS-compat flags and why they exist.
- [docs/awsem_hamiltonian_spec.md](docs/awsem_hamiltonian_spec.md), Hamiltonian specification.
- Examples: [examples/](examples/) (`01_basic.py`, `02_three_modes.py`, ..., `07_frustrapy_drop_in.py`).

## Citation

If you use this software, please cite both the original AWSEM / frustration work and this port. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata, and [NOTICE](NOTICE) for full algorithmic provenance.

Primary references:

- **AWSEM**: Davtyan, A. et al. (2012) *AWSEM-MD: protein structure prediction using coarse-grained physical potentials and bioinformatically based local structure biasing.* J. Phys. Chem. B 116, 8494-8503. doi:10.1021/jp212541y
- **Frustration index**: Ferreiro, D. U. et al. (2007) *Localizing frustration in native proteins and protein assemblies.* PNAS 104, 19819-19824. doi:10.1073/pnas.0709915104
- **frustratometeR**: Rausch, A. O. et al. (2021) *FrustratometeR: an R-package to compute local frustration in protein structures, point mutants and MD simulations.* Bioinformatics 37, 3038-3040. doi:10.1093/bioinformatics/btab176
- **Frustratometer 2 server**: Parra, R. G. et al. (2016) *Protein Frustratometer 2: a tool to localize energetic frustration in protein molecules, now with electrostatics.* Nucleic Acids Res. 44, W356-W360. doi:10.1093/nar/gkw304
- **LAMMPS engine**: Thompson, A. P. et al. (2022) *LAMMPS: a flexible simulation tool for particle-based materials modeling.* Comp. Phys. Comm. 271, 108171. doi:10.1016/j.cpc.2021.108171
- **PyTorch**: Paszke, A. et al. (2019) *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32, 8024-8035.
- **Biopython**: Cock, P. J. A. et al. (2009) *Biopython: freely available Python tools for computational molecular biology and bioinformatics.* Bioinformatics 25, 1422-1423. doi:10.1093/bioinformatics/btp163

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The upstream AWSEM-MD and frustratometeR projects use GPL / academic-use licenses; we re-implement the algorithms from published specifications rather than redistributing their source.
