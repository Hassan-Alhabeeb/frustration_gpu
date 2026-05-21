# Full frustrapy API surface — what the PyTorch port must support

User requirement (2026-05-20): the GPU port must support **every config / parameter that normal frustrapy supports**, not just the minimum subset. This document tracks each parameter and its target phase.

## frustrapy.calculate_frustration() signature

From `inspect.signature(frustrapy.calculate_frustration)` on the VM:

```python
calculate_frustration(
    pdb_file: Optional[str] = None,
    pdb_id: Optional[str] = None,           # download from RCSB if pdb_file missing
    chain: Union[str, List[str], None] = None,
    residues: Optional[Dict[str, List[int]]] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = 'configurational',
    graphics: bool = True,
    visualization: bool = True,
    results_dir: Optional[str] = None,
    debug: bool = False,
    pbar: Optional[tqdm.asyncio.tqdm_asyncio] = None,
    is_mutation_calculation: Optional[bool] = False,
) -> Tuple[Pdb, Dict, Optional[FrustrationDensityResults], Optional[Dict]]
```

## Per-parameter coverage matrix

| Parameter | Type | Default | Numerical effect | Target phase | Status |
|---|---|---|---|---|---|
| `pdb_file` | path | None | core input | Phase 1 | ✅ supported (`parser.parse_pdb`) |
| `pdb_id` | str | None | auto-download from RCSB | Phase 5 | ⏳ deferred (UX, not numerical) |
| `chain` | str / list | None (all chains) | restricts the residue set processed | Phase 1 | ✅ partial — `parser.parse_pdb` returns all chains; chain-subset filter to add in Phase 4 |
| `residues` | dict[str, list[int]] | None | restricts to specific residue subset per chain | Phase 4 | ⏳ |
| `electrostatics_k` | float | None | overrides default DH k_QQ=4.15 | Phase 2c | ⏳ |
| `seq_dist` | int | 12 | min seq separation for contact inclusion in tertiary_frustration | Phase 2a/b | ⏳ — need to expose as param |
| `mode` | str | "configurational" | which decoy strategy: configurational / mutational / singleresidue | Phase 3 | ⏳ all 3 modes needed |
| `graphics` | bool | True | output VMD .tcl visualization files | Phase 5 | ⏳ (cosmetic, low priority) |
| `visualization` | bool | True | output PyMOL .pml visualization files | Phase 5 | ⏳ (cosmetic, low priority) |
| `results_dir` | path | None | where to write outputs | Phase 4 | ⏳ |
| `debug` | bool | False | preserves intermediate files | Phase 4 | ⏳ |
| `pbar` | tqdm | None | progress bar | Phase 5 | ⏳ (UX) |
| `is_mutation_calculation` | bool | False | flag for being called recursively from a mutation analysis | Phase 5 | ⏳ |

## Three frustration modes

All must be supported. Differences (from `fix_backbone.cpp:5249-5344`):

### `mode = "configurational"` (frustrapy default)

- Decoys: 1000, **cached once per structure**, shared across all (i,j) pairs
- Randomize: aa_i, aa_j, rij, rho_i, rho_j (all from protein's own distribution)
- All rows of `tertiary_frustration.dat` share the same `(decoy_mean, decoy_std)`
- This is what we just validated against on 5AON/11BG (Phase 1.5)

### `mode = "mutational"`

- Decoys: 1000 **per (i,j) pair** — NOT cached
- Randomize: only aa_i, aa_j; rij, rho_i, rho_j held at native values
- Includes (i,k) and (j,k) cross-terms in both native and decoy energies (different from configurational)
- Each row of `tertiary_frustration.dat` has its own `(decoy_mean, decoy_std)`
- N.B. This is computationally heavier than configurational (decoy reuse fails) — expect ~1.5×–2× slower on CPU; on GPU the batching saves us

### `mode = "singleresidue"`

- Per-residue (not per-pair) frustration
- Decoys: 1000 per residue
- Randomize: only aa_i (the focal residue); recompute total energy contribution
- Output: per-residue FI vector, NOT a per-pair matrix
- Output file: `<PDB>.pdb_singleresidue` (no `tertiary_frustration.dat`)

## Phase coverage of the modes

| Phase | Configurational | Mutational | Singleresidue |
|---|---|---|---|
| Phase 1 (✅) | parser + burial + virtual atoms — shared infrastructure | shared | shared |
| Phase 2 (in flight) | direct + mediated + DH — shared energy functions | shared | shared |
| Phase 3a | decoy sampling: cached-once pattern | per-pair regenerate pattern | per-residue regenerate pattern |
| Phase 3b | FI z-score per pair | FI z-score per pair | FI z-score per residue |
| Phase 4 | density aggregation per residue | density aggregation per residue | direct per-residue output |

## What we can defer

Cosmetic outputs (`graphics`, `visualization`, `pbar`) are Phase 5 / can-be-skipped. They generate VMD/PyMOL scripts for visualization but don't affect the numerical frustration values. The user's primary use case is per-residue density features for v44 input — cosmetic outputs are optional.
