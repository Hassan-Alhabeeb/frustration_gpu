# v2 decoy-module bug fixes (2026-05-21)

This sprint closes seven findings from the 75-item audit at
`F:/research_plan/New folder/odo.txt` that were all scoped to the three
decoy modules (`decoys.py`, `mutational_decoys.py`,
`singleresidue_decoys.py`).

## Files touched

| File | Lines added | Notes |
|---|---|---|
| `frustration_gpu/decoys.py` | ~70 | sentinel guard, validation, warning |
| `frustration_gpu/mutational_decoys.py` | ~45 | early bail, validation, warning |
| `frustration_gpu/singleresidue_decoys.py` | ~30 | OOM-safe precompute + validation + warning |
| `tests/test_decoy_validation.py` | 320 (new) | 21 regression tests |

No other files were modified; nothing in the public output schemas changed
for valid inputs.

## Per-finding status

| # | Severity | File | Fix | Test |
|---|---|---|---|---|
| 1 | HIGH | `singleresidue_decoys.py` | Route `_precompute_W_sr`'s CUDA path through `_water_per_alpha_fused_sum` whenever `_choose_alpha_chunk` returns a chunk size < 20. Mirrors the strategy already in `_precompute_T_alpha` (mutational). | `test_singleresidue_oom_bound_at_large_n`, `test_singleresidue_chunked_matches_unchunked_on_small_n` |
| 10 | MEDIUM | all three | Call `_check_no_dna_sentinel(coords["residue_types"])` at the entry of `sample_configurational_decoys`, `mutational_decoy_stats`, `singleresidue_decoy_stats`. Also added in `compute_configurational_decoy_energy` over the per-decoy `aa_i_decoy`/`aa_j_decoy` arrays. | `test_dna_guard_*` (4 tests) |
| 22 | MEDIUM | all three | Raise `ValueError` when `n_decoys < 2` at each public API entry. Singleresidue and mutational now also emit a `RuntimeWarning` when any per-residue / per-pair `decoy_std` collapses to zero. `compute_configurational_decoy_energy` raises on length-1 decoy fields and warns when its scalar `decoy_std == 0`. | `test_n_decoys_one_rejected_*` (4), `test_singleresidue_warns_on_std_collapse` |
| 34 | MEDIUM | `mutational_decoys.py` | Moved the `n_pair == 0` short-circuit to immediately after `_enumerate_native_pairs` — BEFORE `_precompute_T_alpha`, native energy gather, and burial precompute. Returns a schema-preserving zero-pair dict (all required keys present, length-0 tensors with the correct dtype and the expected `(0, n_decoys)` shape on `aa_i_dec` / `aa_j_dec`). | `test_mutational_n_pair_zero_short_circuits_before_precompute` |
| 43 | MEDIUM | `decoys.py` | In `compute_configurational_decoy_energy`, validate `gamma_direct`, `gamma_mediated_protein`, `gamma_mediated_water` are exactly `(20, 20)` and `burial_gamma` is exactly `(20, 3)` after they have been promoted to the target device/dtype. | `test_gamma_*_wrong_shape_rejected` (4) |
| 54 | MEDIUM | all three | Validate `rho.shape == (N,)` where `N = len(coords['ca_coords'])` at the entry of every public decoy API. | `test_*_rejects_wrong_rho_shape` (3) |
| 55 | MEDIUM | `decoys.py` | In `compute_configurational_decoy_energy`, gather the shapes of all five decoy fields (`aa_i_decoy`, `aa_j_decoy`, `rij_decoy`, `rho_i_decoy`, `rho_j_decoy`) into a set and raise if more than one distinct shape exists, or if the shape is not 1-D. | `test_compute_configurational_rejects_mismatched_decoy_shapes`, `test_compute_configurational_rejects_non_1d_decoy_fields` |

## Validation gates

| Gate | Result |
|---|---|
| Full pytest suite | **276 passed, 1 warning** (was 223 before, +21 new tests in `test_decoy_validation.py`; the remainder are existing tests we did not touch). |
| Spearman gates on the 4-PDB panel (mutational) | All pass — `test_decoy_mean_std_spearman_per_pair[5AON/11BG/1O3S/3F9M]` unchanged. |
| Native-energy bit-identity gates | All pass — `test_native_energy_matches_dump[5AON/11BG]`. |
| CPU/GPU agreement gates | `test_cpu_gpu_decoy_std_agreement_5aon`, `test_singleresidue_cpu_gpu_agreement_5aon` pass. |
| Singleresidue chunked vs one-shot equivalence | New test `test_singleresidue_chunked_matches_unchunked_on_small_n` forces the chunked-sum path on 5AON and confirms `FI` / `E_native` are bit-exact identical to the one-shot cube path at `float64`. |

## OOM-fix VRAM measurement (Finding #1)

Synthetic N=5000 straight-chain protein on a 12 GB RTX 4070, fp64,
`n_decoys=4`. The chunk size was forced to 2 via monkey-patch on
`_choose_alpha_chunk` so both code paths exercise the same upstream
allocations and only differ in the W-sum kernel.

| Path | Peak VRAM | Notes |
|---|---|---|
| BROKEN one-shot (`_water_per_alpha_fused` returning the full `(20, N, N)` cube) | **7.53 GB** | Two simultaneous `(20, N, N)` cubes live at once — the returned `out` plus `sigma_gamma_med`. |
| FIXED chunked-sum (`_water_per_alpha_fused_sum` with `chunk=2`) | **3.80 GB** | Peak is one `(chunk, N, N)` intermediate plus the upstream `(N, N)` `θ` / `σ` tensors. |

**2.0x reduction** in peak allocation on this synthetic. The fixed peak is
slightly above the bare `(20, N, N)` cube size (3.73 GB) because the four
contiguous `(N, N)` `θ_direct`, `θ_med`, `σ_wat`, `σ_prot` tensors that
exist on both paths add ~800 MB of upstream allocation. The
`_water_per_alpha_fused_sum` kernel itself never materialises a tensor
larger than `(chunk, N, N)`.

## Backwards-compatibility notes

* No public function signatures changed; the new validation checks only
  raise on inputs that would previously have silently produced wrong /
  truncated / zero-FI output.
* `compute_configurational_decoy_energy(decoys=<length-1 fields>)` now
  raises `ValueError` instead of silently producing a zero-variance
  scalar. This matches the new `sample_configurational_decoys(n_decoys=1)`
  rejection — the two functions are paired in practice.
* `mutational_decoy_stats` on a protein where every native pair is too
  far apart (`contact_cutoff = 0` in the regression test) now returns
  identical schema-shaped zero-length tensors but skips the O(N²)
  `_precompute_T_alpha` call. Existing callers that catch `n_pair == 0`
  by inspecting `out["pair_i"].numel()` are unaffected.
* `singleresidue_decoy_stats` emits a `RuntimeWarning` (via the standard
  `warnings` module) when any per-residue `decoy_std == 0`. Callers that
  treat warnings as errors will need to either skip those residues
  upstream or filter the warning. The warning text is stable and matches
  on `"decoy_std"`.

## What changes for valid inputs

Nothing. The bit-identity gate (`test_native_energy_matches_dump`,
`test_cpu_gpu_decoy_std_agreement_5aon`, the chunked vs one-shot
equivalence test) confirms the numerics are unchanged on every input
that already passed the (new) validation.
