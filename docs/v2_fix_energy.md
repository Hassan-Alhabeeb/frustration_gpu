# v2 energy + context bug fixes (2026-05-21)

This sprint closes fifteen findings from the 75-item audit at
`F:/research_plan/New folder/odo.txt` that were all scoped to the three
energy-term modules (`direct_contact.py`, `water_mediated.py`,
`debye_huckel.py`), the shared `_contact_common.py` infrastructure, and the
sparse-cutoff validation in `burial.py`.

## Files touched

| File | Lines added | Notes |
|---|---|---|
| `frustration_gpu/_contact_common.py` | ~250 | new helpers, fingerprint, sparse refactor |
| `frustration_gpu/direct_contact.py` | ~50 | validation + warnings + sparse-cutoff guard |
| `frustration_gpu/water_mediated.py` | ~70 | validation + warnings + theta diagnostic mask |
| `frustration_gpu/debye_huckel.py` | ~80 | screening_length validation + warnings + r_ij guard |
| `frustration_gpu/burial.py` | ~12 | sparse_cutoff_a NaN/inf guard |
| `tests/test_energy_validation.py` | 380 (new) | 37 regression tests |

No other files were modified; the default dense-path scalar energies for
5AON were unchanged to machine precision (verified by the existing
suite — the 291 pre-existing tests still pass).

## Per-finding status

| # | Severity | Fix summary | Test |
|---|---|---|---|
| 6  | LOW    | `water_mediated_energy(return_pair_matrix=True)` now masks the diagnostic `theta` (and the sparse 1-D `theta_1d`) so excluded pairs read `0`, not the spurious sigmoid peak that the `safe_dist` mid-window fill produced. | `test_water_mediated_theta_zeroed_on_excluded_pairs` |
| 11 | MEDIUM | New `_validate_context_device` is called at the top of every energy fn that accepts `_context`. Cross-device contexts (e.g. CPU context + `device="cuda"`) raise `ValueError("ContactContext on cpu, but device=cuda requested")` instead of dying inside a kernel call. | `test_context_device_mismatch_raises` |
| 14 | LOW/MED | `_resolve_contact_coords` emits a `UserWarning` listing the residue indices when a NON-Gly residue has missing/NaN CB and was silently substituted with CA (~1.5 Å bond-length shift in contact geometry). Gly fallback stays silent. | `test_non_gly_cb_fallback_warns` |
| 24 | LOW/MED | `debye_huckel_energy` and `debye_huckel_pair_energy` validate `screening_length` and `k_screening` are finite positive up front. `0` no longer raises a raw `ZeroDivisionError`, and negatives no longer invert the exponential into growth. | `test_dh_zero_screening_length_raises`, `test_dh_negative_screening_length_raises`, `test_dh_nan_screening_length_raises`, `test_dh_zero_k_screening_raises` |
| 28 | HIGH | New `_warn_sparse_cutoff` + `dh_sparse_min_safe` helper. `debye_huckel_energy` emits a `UserWarning` when the supplied `SparseContactContext.sparse_cutoff < 3*screening_length/k_screening` (30 Å at the LAMMPS default λ=10). The advice string includes the exact rebuild command. | `test_sparse_dh_warns_below_min_safe`, `test_sparse_dh_no_warn_at_min_safe` |
| 29 | HIGH | `MEDIATED_SPARSE_MIN_SAFE_A = 14.0` constant; `water_mediated_energy` warns when the context's `sparse_cutoff` is below 14 Å (the previous default 9.5 Å produced a 0.46 kcal/mol drift on 5AON). | `test_sparse_water_warns_below_min_safe`, `test_sparse_water_no_warn_at_min_safe` |
| 30 | MEDIUM | New `_coords_fingerprint` (cheap `(N, dev, first-sum, last-sum, total-sum, abs-total-sum)` 6-tuple). Built at `build_contact_context` time and pinned to both context dataclasses as a new `fingerprint` field. Every energy fn now calls `_validate_context_fingerprint` and raises `ValueError("Stale ContactContext: ...")` on mismatch. | `test_stale_context_fingerprint_rejected` |
| 31 | LOW/MED | Existing dense-vs-sparse mismatch was already raising `ValueError`; this finding is now formally covered by tests, and the wrong-type error message points the caller at `build_contact_context(..., sparse_cutoff=...)`. | `test_sparse_context_passed_without_sparse_flag_raises`, `test_dense_context_passed_with_sparse_flag_raises` |
| 39 | LOW/MED | New `_check_residue_types_in_range` helper rejects `residue_types < 0` (DNA sentinel — already caught by `_check_no_dna_sentinel`) AND `residue_types >= 20` with a single clear error. Wired into `direct_contact_energy`, `water_mediated_energy`, `debye_huckel_energy`, and the three scalar `*_pair_energy` helpers. | `test_residue_types_out_of_range_{direct,water,dh}`, `test_direct_pair_energy_rejects_out_of_range_aa`, `test_direct_pair_energy_rejects_negative_aa` |
| 42 | MEDIUM | `SparseContactContext` no longer caches a dense `(N, N) dist` tensor (`dist: torch.Tensor | None = None`). The sparse build now uses a chunked O(`chunk × N × 3`) scan that materialises pair distances only for pairs inside `sparse_cutoff`. The diagnostic `return_pair_matrix=True` path in direct/water/DH falls back to the 1-D `r_ij` tensor. | `test_sparse_context_does_not_store_dense_dist` (and indirectly via the existing sparse byte-exact tests) |
| 47 | HIGH | `DIRECT_SPARSE_MIN_SAFE_A = 7.5` constant; `direct_contact_energy` warns when the context's `sparse_cutoff` is below 7.5 Å (the previous default 6.5 Å — the documented `r_max` — produced a 0.51 kcal/mol drift on 5AON). | `test_sparse_direct_warns_below_min_safe`, `test_sparse_direct_no_warn_at_or_above_min_safe` |
| 56 | MEDIUM | `build_contact_context` and `compute_rho(sparse=True)` reject NaN / +inf / non-positive `sparse_cutoff` (`sparse_cutoff_a`) up front. Previously NaN slipped through (`NaN <= 0` is `False`) and produced empty contexts, while +inf silently kept every pair. | `test_sparse_cutoff_{nan,inf,zero,negative}_rejected`, `test_burial_sparse_cutoff_{nan,zero}_rejected` |
| 57 | LOW/MED | `direct_pair_energy`, `water_mediated_pair_energy`, and `debye_huckel_pair_energy` now validate any custom gamma tensors are exactly `(20, 20)` before indexing, and the amino-acid indices are in `[0, 20)`. NaN gamma tables still pass through (silent NaN is a separate v2 finding owned by the parameter-loader agent). | `test_direct_pair_energy_rejects_wrong_gamma_shape`, `test_water_pair_energy_rejects_wrong_gamma_shape`, `test_direct_pair_energy_rejects_out_of_range_aa`, `test_direct_pair_energy_rejects_negative_aa` |
| 58 | LOW/MED | `debye_huckel_pair_energy` validates `r_ij` is finite and strictly positive (the 1/r factor demands `> 0`); `direct_pair_energy` and `water_mediated_pair_energy` validate finite and non-negative. | `test_dh_pair_energy_rejects_{zero,negative,nan,inf}_distance` |

(Finding #28 / #29 / #47 also write the term-specific minimum-safe
`sparse_cutoff` into the public docstrings of
`direct_contact_energy`, `water_mediated_energy`, and `debye_huckel_energy`
so future users see the recommendation without having to scrape the
codebase.)

## Recommended minimum-safe sparse cutoffs

| Term | Constant / helper | Default | Rationale |
|---|---|---|---|
| Direct contact | `DIRECT_SPARSE_MIN_SAFE_A` | **7.5 Å** | Tanh shoulder tail leaves ~3 % of well amplitude at r=7 Å. 5AON drift at cutoff=6.5 Å is 0.51 kcal/mol (HIGH); at cutoff=7.5 Å it is < 0.01 kcal/mol. |
| Water-mediated | `MEDIATED_SPARSE_MIN_SAFE_A` | **14.0 Å** | Mediated tail decays slower than direct because the burial-blended γ stays O(1). 5AON drift at cutoff=9.5 Å is 0.46 kcal/mol (HIGH); at cutoff=14 Å it matches dense. |
| Debye-Hückel | `dh_sparse_min_safe(λ, k)` | **3 × λ_eff** | `V_DH(r) ∝ exp(-r/λ_eff)/r`. 5AON sign-flips at cutoff=11 Å (+0.15 vs −0.60 kcal/mol); at cutoff=30 Å it matches. The helper scales with `screening_length`. |

## Internal-API changes

| Change | Reason | Compat impact |
|---|---|---|
| `ContactContext.fingerprint: tuple[6]` | Finding #30 | Default value is provided; manual construction still works. |
| `SparseContactContext.fingerprint: tuple[6]` | Finding #30 | Same. |
| `SparseContactContext.dist: torch.Tensor \| None = None` | Finding #42 | Was a required `torch.Tensor` field. Any caller previously expecting a dense `(N, N)` distance from a sparse context should rebuild a dense context. The energy-fn `return_pair_matrix=True` dict still has a `"distances"` key — it now carries the 1-D `r_ij` pair-list as a fallback. |
| `_warn_sparse_cutoff`, `_validate_context_device`, `_validate_context_fingerprint`, `_check_residue_types_in_range`, `_coords_fingerprint`, `dh_sparse_min_safe`, `DIRECT_SPARSE_MIN_SAFE_A`, `MEDIATED_SPARSE_MIN_SAFE_A` | New helpers / constants in `_contact_common` | Additive, no removal of existing names. Exported via `__all__`. |

## Test count

* Pre-change baseline: 291 tests (291 passing, 1 docs warning).
* Post-change: **365 tests, 365 passing, 1 docs warning** (37 added by this
  sprint, 37 added by the parallel decoys/misc/parser agents, baseline
  unchanged for default dense path).
