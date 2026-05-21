# MISC bug-fix bundle (rev 2026-05-21)

Owner: misc-bugs agent.
Scope: 14 MEDIUM findings spanning `burial.py`, `frustration.py`, `density.py`,
`parameters.py`, `virtual_atoms.py`. Out of scope (other agents): `parser.py`,
`compute_frustration.py`, `_contact_common.py`, the contact-energy modules,
the decoy modules.

## Findings addressed

| # | File:func | Status | Behaviour change | Notes |
|---|-----------|--------|------------------|-------|
| 3 | `burial.compute_rho` (dense path) | FIXED | none for finite inputs | autograd-safe via double-where: NaN rows in cb_or_ca are replaced by a 1e6 filler before `vector_norm`, masked out at the end. Bit-identical rho on clean inputs (verified by `test_compute_rho_value_unchanged_for_clean_input`) |
| 13 | `virtual_atoms.compute_virtual_atoms` | FIXED | cis-proline coefficients now apply ONLY to PRO rows (`residue_types == ONE_TO_IDX["P"]`); other rows always use trans | was globally applied to every residue, silently corrupting non-proline N/H/C |
| 14 | `frustration._xb_coords` | already fixed | n/a | the CB-missing-fallback bug was patched in the QA-3 H-1 round; the docstring already documents the fallback. Parser-side half of #14 owned by parser agent |
| 15 | `frustration.classify_frustration` | FIXED | RAISES on non-finite FI (previously bucketed to neutral) | decision: loud > quiet; callers must filter NaN before classification |
| 27 | `density.compute_residue_density` | FIXED | RAISES on length-mismatch or non-finite FI | catches upstream pair/fi truncation that previously silently broadcast or zero-padded |
| 32 | `parameters.load_gamma_tables` | FIXED | strict: exactly 2 columns, exactly 420 rows, no NaN/inf | 1-column duplication path removed; extra-row truncation removed |
| 33 | `virtual_atoms.compute_virtual_atoms` | FIXED | i-1 / i+1 neighbour test now also requires `resnum[i] - resnum[i-1] == 1` | TER-separated same-chain breaks and missing-residue gaps now produce NaN virtual atoms at the boundary, matching chain-start/chain-end behaviour |
| 40 | `frustration.welltype_from_contact` | FIXED | RAISES on non-finite r_ij / rho_i / rho_j | was bucketing NaN to `long` |
| 41 | `burial.burial_density` | FIXED | RAISES on `residue_types < 0` (DNA sentinels) | reuses `_check_no_dna_sentinel` from `_contact_common`; `lammps_dump_rho` lives in `decoys.py` (out of scope) |
| 48 | `burial.compute_rho` | FIXED | RAISES when `residue_numbers` or `chain_index` shape != `(N,)` | early shape validation prevents silent broadcasting in the (N, N) masks |
| 49 | `parameters.load_gamma_tables`, `load_burial_gamma` | FIXED | RAISES on NaN/inf values in either parameter file | per-line filename:lineno error message |
| 51 | `frustration._aa_idx_to_letter` | FIXED | RAISES when any index is outside `[0, 20)` | catches DNA-sentinel `-1` that previously silently mapped to `V` |
| 52 | `density.compute_residue_density` | FIXED | RAISES on negative or `>= N` pair indices | combined with #27 in one validation block |
| 62 | `frustration.emit_singleresidue_dat` | FIXED | RAISES when `rho`, `e_native`, `decoy_mean`, `decoy_std`, or `fi` length != coords length | previously the writer iterated for `n = rho.numel()` and silently truncated |

## Decisions (loud vs quiet)

Across the bundle the choice for non-finite / shape-mismatched inputs is
**RAISE, never quiet sentinel**. Rationale:

1. The misc audit specifically called out cases where silent fallbacks
   (negative-index wrap to `V`, NaN-to-neutral, broadcast-on-shape-mismatch)
   were hiding upstream bugs. A loud exception forces the caller to
   confront the bad input.
2. The downstream consumers in `compute_frustration.py` and the decoy
   modules already filter DNA sentinels via `_subset_protein_only` and
   produce finite FI by construction. So in normal operation these
   exceptions never fire; they only surface when a low-level API is used
   incorrectly.
3. The "raise on non-finite" convention matches the existing
   `_check_no_dna_sentinel` guard in `_contact_common.py` — it is the
   precedent set by the QA-1 HIGH fix.

## Forward-compat: cis-proline (#13)

The new behaviour is gated on `parsed["residue_types"]` — if the dict
doesn't include `residue_types`, `use_cis_proline=True` is silently a
no-op (trans coefficients everywhere). This preserves the legacy
contract for synthetic test fixtures that build minimal coord dicts.
When/if the parser learns to preserve the `IPR` 3-letter code and emit
a dedicated index, the IPR check can replace the PRO-index test in one
line.

## Tests

New file `tests/test_misc_validation.py` (27 tests, all passing). Covers
every finding above plus a "real param files still load" sanity test so
the stricter loaders aren't allowed to regress on the shipped data.

## Validation gates

* `pytest tests/` — 305 passed, 1 warning (api_docs missing-signature
  warning on `emit_5adens_dat` / `chain_segments`, pre-existing).
* Bit-identity: `test_compute_rho_value_unchanged_for_clean_input`
  verifies the autograd-safe rho rewrite preserves values on clean input;
  the `test_real_gamma_dat_still_loads` test verifies the stricter
  parameter loaders accept the shipped tables unchanged.

## Files touched

* `frustration_gpu/burial.py` (#3, #41, #48)
* `frustration_gpu/frustration.py` (#15, #40, #51, #62)
* `frustration_gpu/density.py` (#27, #52)
* `frustration_gpu/parameters.py` (#32, #49)
* `frustration_gpu/virtual_atoms.py` (#13, #33)
* `tests/test_misc_validation.py` (NEW — 27 tests)
