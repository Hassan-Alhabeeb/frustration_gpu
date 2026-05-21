# Changelog

All notable changes to frustration_gpu will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `NOTICE` file at repo root enumerating algorithmic provenance (Apache-2.0 §4(d)).
- Citation upgrades in `CITATION.cff`: Parra 2016 (Frustratometer 2 server), Thompson 2022 (LAMMPS), Paszke 2019 (PyTorch), Cock 2009 (Biopython), and Rausch 2021 promoted from `software` to `article` with DOI.

### Removed
- `docs/reference_lammps_awsem/` (mirrored GPL-2.0 C++ source from adavtyan/awsemmd). The citation is preserved; the source is not redistributed. Inline `fix_backbone.cpp:NNN` references in the docs now read as upstream-file citations rather than pointers into this repo.

## [0.1.0] - 2026-05-21

### Added
- Pure-PyTorch implementation of LAMMPS-AWSEM frustration analysis
- All 3 modes: configurational, mutational, singleresidue
- GPU acceleration: 14-53x speedup vs frustrapy CPU on RTX 4070
- Top-level `compute_frustration()` API + `calculate_frustration()` drop-in alias for frustrapy users
- LAMMPS-compat opt-in flags: `lammps_compat_altloc`, `include_dna`, `keep_incomplete_backbone`, `include_dh_in_e_native`
- Chain filter, residue subset filter, opt-in DH electrostatics
- Byte-comparable dump emitters (tertiary_frustration.dat, singleresidue.dat, 5adens.dat)
- 223+ tests passing on the 10-PDB validation panel (4-PDB frustrapy head-to-head subset)
- Docs: API.md, QUICKSTART.md, VALIDATION.md, lammps_compat_fixes.md, frustrapy_vs_us.md
- 7 runnable example scripts
- Phase 5 benchmark harness (`benchmark/run_phase5.py`)

### Validated
- FI Spearman >= 0.9975 on 30/30 (PDB, mode) combos vs LAMMPS reference
- CPU<->CUDA max |DeltaFI| = 0.0 (literal, FI absorbs ULP drifts in decoy_mean - verified on all 30 (PDB, mode) combos)
- Configurational FI Spearman >= 0.99999 on every panel PDB (exact when E_native has no ties; lowest observed 0.9999981)
- LAMMPS-compat flags: 1O3S density Spearman 0.224 -> 0.9992, 3F9M 0.274 -> 0.9997
- Scales to 8,689-residue 4PKN end-to-end in 17.6s on RTX 4070

### Known limitations
- Large proteins (>5,000 residues) consume significant VRAM (~22 GB allocator high-water mark on 4PKN). Alpha-chunking the auxiliary (N,N) tensors will address this; tracked in `docs/optimization_opportunities.md`.
- LAMMPS' dump precision is `%8.3f` (3 decimal places); upstream fix requires recompiling LAMMPS-AWSEM (recipe in `docs/precision_upgrade_plan.md`).
- Rejection-sampler fallback bias on sparse/fragmented structures (logged warning, FI Spearman unaffected on dense structures).
- For very small proteins (N < 100) CUDA launch overhead can make GPU slower than CPU. The 5N9R (N=356) mutational row is the one panel cell where this leaks into a larger structure; CPU mode is recommended for N < 100.

[Unreleased]: https://github.com/Hassan-Alhabeeb/frustration_gpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Hassan-Alhabeeb/frustration_gpu/releases/tag/v0.1.0
