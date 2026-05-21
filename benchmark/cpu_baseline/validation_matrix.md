# CPU baseline validation matrix

10 PDBs × 3 modes + 4 param-sweep configs on 2 PDBs = 38 validation reference dumps.

Production-matched frustrapy calls in all cases. VM `root@10.1.0.45`, frustrapy 0.1.1.

## Modes covered (per PDB)

| PDB | residues | configurational | mutational | singleresidue |
|---|---|---|---|---|
| 5AON | 49 | ✓ | ✓ | ✓ |
| 11BG | 248 (dimer) | ✓ | ✓ | ✓ |
| 1O3S | 200 | ✓ | ✓ | ✓ |
| 5N9R | 356 | ✓ | ✓ | ✓ |
| 3F9M | 451 | ✓ | ✓ | ✓ |
| 4C8B | 560 | ✓ | ✓ | ✓ |
| 4HON | 675 | ✓ | ✓ | ✓ |
| 2SKE | 830 | ✓ | ✓ | ✓ |
| 1OS2 | 990 | ✓ | ✓ | ✓ |
| 6F56 | 1528 | ✓ | ✓ | ✓ (re-run after sequential-script kill) |

## Param sweeps (on 5AON + 11BG only)

| Param | Value | 5AON | 11BG |
|---|---|---|---|
| seq_dist | 3 | ✓ | ✓ |
| seq_dist | 12 | ✓ (= configurational default) | ✓ |
| seq_dist | 6 | ✗ — frustrapy hardcodes `seq_dist ∈ {3, 12}` (drops this from matrix) | ✗ |
| electrostatics_k | 4.15 (frustrapy default) | ✓ | ✓ |
| electrostatics_k | 17.3636 | ✓ | ✓ |
| chain | "A" only | ✓ (= configurational since 5AON is monomer) | ✓ (single-chain, vs the dimer default) |

## frustrapy API constraints discovered during dump

- `seq_dist` is **NOT a free parameter** — only accepts `3` or `12`. Documented in frustrapy as an enum, not a continuous int. Validation matrix accordingly restricted.
- `chain=None` (default) = process ALL chains. `chain="A"` = restrict to chain A, output dir gets `_A.done/` suffix (vs `.done/` for full-multimer runs).
- `electrostatics_k` defaults to 4.15 kcal·Å/mol per `fix_backbone_coeff.data`. Any float value accepted, no validation.

## Directory layout

```
benchmark/cpu_baseline/
├── configurational/   (10 PDBs × {tertiary, 5adens, configurational_classification, energy, timer} = 50+ files)
├── mutational/        (10 PDBs × {tertiary, 5adens, mutational, energy} = 40 files)
├── singleresidue/     (10 PDBs × {singleresidue, energy} = 20 files)
└── param_sweep/       (5AON + 11BG × {seq_dist_3, electro_4p15, electro_17p3636, chain_A_only} × {tertiary, energy} = 16 files)
```

Total: ~126 reference files for full Phase 3 validation.
