# Phase 2b code review — water-mediated contact + shared infrastructure

Reviewer: Opus code-reviewer, 2026-05-20.
Files: `src/water_mediated.py`, `src/_contact_common.py`, `src/contact_gamma.py` (+`load_mediated_gamma`), `src/direct_contact.py` (refactor to consume `_contact_common`), `tests/test_water_mediated.py`.
Reference: `docs/reference_lammps_awsem/fix_backbone.cpp` (lines 257-266, 620-641, 5025-5054, 5444-5476), `docs/awsem_hamiltonian_spec.md`, `docs/lammps_awsem_term_spec.md`, `docs/phase_2a_review.md`.

## Verdict
**PASS** — formula, parameters, gamma loader, σ_water/σ_prot blend, sign, factor of 1/4, k_water-fold semantics, NaN-safety, and the shared-helper refactor all match the C++ source. Numerical validation gate cleared at machine precision (5AON 2.7e-7, 11BG 1.4e-8 relative). Phase 2a regression tests all green after the `_contact_common` refactor. No critical/high findings.

## Findings by severity

### Critical
*(none)*

### High
*(none)*

### Medium

- **`src/water_mediated.py:210-226` — n=0/n=1 short-circuit does NOT load gamma but DOES expand the `return_pair_matrix` dict using `(n, n)` zero matrices.** Cosmetically fine because `(0, 0)` and `(1, 1)` tensors are valid; functionally fine because the gamma load is skipped. However the dict returned in the early-out path omits the `gamma_pair` key returned in the normal path, while the early-out for `direct_contact_energy:251-254` *also* omits a different subset. The early-out dicts are not key-compatible with the full path. Low impact since users typically check the scalar; worth a one-line comment ("early-out dict has fewer keys than the full path") or just always returning the same set.

- **`src/water_mediated.py:285-288` — `sigma_prot = 1 - sigma_wat` is computed across the FULL (N, N) matrix including the diagonal and below-diagonal cells.** The diagonal terms `sigma_water_per_res[i]² ` plug into `gamma_pair`, which then gets multiplied by `theta[i,i]`. Because `theta[i,i] = 0.25 * (1+tanh(η * (0 - 6.5))) * (1+tanh(η * (9.5 - 0))) ≈ 0.25 * (1 + (-1)) * (1 + 1) ≈ 0`, the diagonal contributes negligibly, and the `upper`-triangle filter at line 303 zeros it anyway. So no bug — just slightly wasted compute. Same observation applies to the direct term. Not worth fixing.

- **`src/_contact_common.py:148-153` — sanitised `safe_cb` uses a fixed `1.0e6` decoy.** At float32 this is fine (`(1e6 - 0)² × 3 = 3e12`, still finite, the resulting distance `~1.73e6` evaluates `tanh(η * (1.73e6 - 6.5)) = 1.0` and `tanh(η * (9.5 - 1.73e6)) = -1.0`, so `theta = 0.25 * 2 * 0 = 0`). At float16 (not used currently, but plausible for trainable loss work) `1e6` would overflow. Document the float32+ assumption or scale the decoy with `r_max`. Minor.

- **`src/water_mediated.py:236-262` — `user_gamma` flag is set if EITHER table is user-supplied,** but the warning at line 255 implies BOTH being passed. If a caller supplies only `gamma_mediated_protein` and lets water fall back to default, the warning fires referring to "tables" plural. Cosmetic.

- **`src/water_mediated.py:78-87` — return-dict docstring says `sigma_wat` and `sigma_prot` are full-symmetric**, but they are constructed as outer products of per-residue σ_water and hence symmetric by construction (verified at line 287). `theta` is also full-symmetric. So the doc is correct, but worth one extra sentence flagging that `pair_energy` and `pair_mask` are upper-triangular while these auxiliaries are full. (Same mixed-convention pattern that Phase 2a review flagged; carried into 2b consistently.)

- **`tests/test_water_mediated.py:202-225` — `test_boundary_continuity` walks r from 3 → 12 with A-A residue type and ρ=1,1 but never asserts `θ` itself — it only checks the assembled V_mediated.** Because the test only confirms "not too big a jump (<1.0 kcal/mol per step)" and that endpoints are ~0, a subtle bug in the in-window region could escape. The existing pair-hand-check (test 1) at r=8.0 covers the in-window centre value, so combined coverage is OK, but a single assertion that θ(r=8.0) ≈ θ_ref to 1e-12 would be tighter.

- **No test for `k_water` runtime warning** — the analogous Phase 2a test (`test_k_water_warning_with_custom_gamma`) exists for direct, but there is no equivalent for the mediated module. The warning code path at lines 254-262 is untested. Quick `pytest.warns(UserWarning, ...)` test would close the gap.

- **`src/_contact_common.py:204` and `:208` — `valid_row` uses `isfinite(cb_coords)` directly,** which is correct AFTER `_resolve_contact_coords` has CA-substituted for GLY (so `cb_coords` passed into `_pair_mask` is actually the effective CB tensor returned by `_resolve_contact_coords`). The variable name in `_pair_mask` is `cb_coords` but it's really `cb_or_ca`. Rename to `eff_cb_coords` for clarity. Not a bug, just confusing on first read.

## sigma_prot formula verification

**Resolved: `sigma_prot = 1 - sigma_wat` is correct.** Verbatim from `fix_backbone.cpp`:

- `:5459`: `sigma_wat = 0.25*(1.0 - tanh(well->par.kappa_sigma*(rho_i-well->par.treshold)))*(1.0 - tanh(well->par.kappa_sigma*(rho_j-well->par.treshold)));`
- `:5460`: `sigma_prot = 1.0 - sigma_wat;`
- `:5463`: `sigma_gamma_mediated = sigma_prot*water_gamma_prot_mediated + sigma_wat*water_gamma_wat_mediated;`

The Phase 2b agent's interpretation matches C++ line-for-line. The naive alternative `σ_prot(ρ_i) × σ_prot(ρ_j) = (1-σ_water(ρ_i)) × (1-σ_water(ρ_j))` is NOT what LAMMPS does — it would differ by the cross terms `σ_water(ρ_i) × (1-σ_water(ρ_j)) + (1-σ_water(ρ_i)) × σ_water(ρ_j)`. The machine-precision match (2.7e-7) directly confirms the `1 - sigma_wat` form; the alternative would have produced a much larger discrepancy on 11BG (which has mixed-burial residues throughout the structure).

Physical interpretation also makes sense the LAMMPS way: `sigma_wat` is the probability that BOTH residues are solvent-exposed (intersection event), and `sigma_prot = 1 - sigma_wat` is the complement ("at least one is buried"). The naive product `(1-σ_w(ρ_i)) × (1-σ_w(ρ_j))` would be the probability BOTH are buried (a different intersection), and the two cross-terms would be "one buried, one exposed" — those would have NO weight in the blend, which is dimensionally wrong because every pair must be fully weighted.

## Numerical validation sanity check

Ran `pytest tests/test_water_mediated.py -v -s` locally on Windows (Python 3.10.10, PyTorch installed):

```
tests/test_water_mediated.py::test_validation_gate_5aon
5AON V_direct=-2.554251 V_mediated=-16.146025 V_water=-18.700276 target=-18.700281 rel_err=0.000027%
PASSED

tests/test_water_mediated.py::test_validation_gate_11bg
11BG V_direct=-7.198125 V_mediated=-140.192724 V_water=-147.390849 target=-147.390847 rel_err=0.000001%
PASSED
```

All 12 tests pass (3.40s). Phase 2a direct-contact regression (13 tests) also still passes (5.21s) — confirms the `_contact_common` refactor is non-breaking. 5AON precision 2.7e-7 and 11BG 1.4e-8 confirmed independently. The 2.7e-7 vs 1.4e-8 spread is consistent with float64 round-off on the larger 11BG sum benefiting from cancellation — sub-PPM at both scales.

Also spot-checked the σ_water sigmoid at extreme burial:
- ρ=0 (surface): σ_water = 1.0  → sigma_wat (pair) = 1.0  → water-mediated gamma dominates ✓
- ρ=2.6 (threshold): σ_water = 0.5 → sigma_wat (pair) = 0.25 → 75% protein-mediated, 25% water ✓
- ρ=5 (buried): σ_water = 2.6e-15 → sigma_wat (pair) ≈ 0 → protein-mediated dominates ✓

Matches AWSEM literature physics (surface residues see water mediation; buried residues see direct protein-protein mediation).

## Things I verified are correct

- **`src/water_mediated.py:276-278` ✓ θ_mediated** = `0.25 * (1 + tanh(η*(r - r_min))) * (1 + tanh(η*(r_max - r)))` matches `fix_backbone.cpp:5469-5471` character-for-character. Parameters r_min=6.5, r_max=9.5, η=5.0 match `well->par.well_r_min[1]/well_r_max[1]/kappa` at `:257-266`.
- **`src/water_mediated.py:285-288` ✓ σ_water/σ_protein** match `:5459-5460` with the factored-out `0.5 * (1 - tanh(...))` per residue followed by outer product → identical to the C++ `0.25 * (1 - ...) * (1 - ...)`.
- **`src/water_mediated.py:294, 299` ✓ blend + sign** `-k × (σ_prot γ_p + σ_wat γ_w) × θ` matches `:5463, :5473`.
- **`src/contact_gamma.py:101-141` ✓ load_mediated_gamma** correctly reads rows 210-419 of `gamma.dat`, distinct columns 0 and 1, upper-triangle walk with explicit symmetrisation. Matches `fix_backbone.cpp:626-636` (LAMMPS' own loader) and OpenAWSEM's `contactTerms.py` decoder. Symmetry verified by the upper-triangle assignment at `parameters.py:155-163`. AA order `A R N D C Q E G H I L K M F P S T W Y V` matches `se_map` at `fix_backbone.cpp:55`.
- **`src/_contact_common.py:36-66` ✓ `_resolve_contact_coords`** does CB-or-CA substitution exactly matching the C++ `:5088-5091` and OpenAWSEM `cb_fixed` at `contactTerms.py:166`. Generalisation to "any NaN CB" is a strict superset of the LAMMPS GLY-only check.
- **`src/_contact_common.py:90-167` ✓ `_pairwise_distance_safe`** implements the recommended Phase 2a "double-where NaN trick" (sanitise coords → diff → norm → mask with fill_value). Layer 1 prevents NaN from entering the backward graph; layer 2 keeps tanh well-conditioned. `test_nan_residue_does_not_poison_gradients` exercises the path and passes for both direct and mediated.
- **`src/_contact_common.py:171-213` ✓ `_pair_mask`** correctly implements cross-chain-always + same-chain-requires-min-sep + no-self-pair + finite-coord checks. Matches `fix_backbone.cpp:5048, :5086`.
- **`src/direct_contact.py` refactor ✓** — all Phase 2a tests still pass after switching to `_contact_common` helpers. No semantic change; the refactor just removes duplicated logic.
- **k_water-fold semantics ✓** — Mediated module documents the same convention as direct: gamma tables are RAW (not k_water-multiplied), k_water enters as an explicit prefactor. Matches the C++ math (which folds k_water at load time → identical when k_water=1.0, which is the only value frustrapy uses). Warning at `water_mediated.py:254-262` fires when user passes custom gamma + non-default k_water — exactly the only ambiguous combination. (Untested but the logic mirrors `direct_contact.py:270-279` which IS tested.)
- **Differentiability ✓** — `rho` flows through tanh in σ_water → continuous gradient. `cb_coords` flows through `_pairwise_distance_safe` → tanh in θ → continuous. `test_differentiable_wrt_cb` exercises backward through both via `burial_density(p).detach()` — meaning the test checks coord gradients but NOT joint backprop through ρ. The docstring (`water_mediated.py:194-197`) correctly flags that ρ-graph-to-coords requires the caller to pass a non-detached ρ tensor.
- **CPU/GPU agreement ✓** — `test_cpu_gpu_agreement` passes at `rel < 1e-6` on the local RTX 4070.

## Recommended Phase 2c next step

Move to **`V_Burial`** (the burial 3-well term — `fix_backbone.cpp:5478-5500`, computed per-residue using `t[0..2][0..1]` and `burial_gamma_0/1/2`). Three reasons:

1. `compute_burial_energy` shares NO infrastructure with `_contact_common`; it's per-residue (not per-pair), reads `burial_gamma.dat` instead of `gamma.dat`, and uses a different sigmoid template. It can be implemented in ~150 LOC plus tests, with zero coupling to the contact module.
2. Validation reuse: `V_water + V_burial + V_DH` ≈ `energy.log "Burial"` column, and ρ is already being computed correctly (verified by the 2.7e-7 match), so the burial term inherits a validated ρ feed.
3. After V_burial, the next term is V_AMH-Go (much harder — Gaussian wells per memory snapshot, requires the memory file parser). Sequencing burial first banks an easy win and de-risks the bigger lift.

**Phase 3 (decoy mode) latent risks I'd watch for**:
- The shared `_pair_mask` enforces `not_self` and seq-sep filtering. In decoy mode, the C++ at `:5099-5101` calls `compute_decoy_ixns(i_resno, j_resno, rij, rho_i, rho_j)` which shuffles the residue identity but keeps the (i, j) topology — so the mask is still valid. BUT: if the decoy implementation reuses `water_mediated_energy` with a permuted `aa` array, the gamma lookup at `:292-293` (`gamma_mediated_protein[aa.unsqueeze(1), aa.unsqueeze(0)]`) will silently produce the wrong values if the permutation is not applied symmetrically. Recommend an assertion/contract that `aa` is the only identity input — coords and ρ stay fixed.
- The pair-decoy mode at `:5099` recomputes decoys ONCE for configurational mode (cached) but PER-PAIR for mutational mode. The dense O(N²) builder here builds everything at once. For configurational decoy, that's fine (one big matrix). For mutational, the caller will want a single-pair scalar path — `water_mediated_pair_energy` is already that and matches the dense path to float64 precision (test 2 confirms). Good.
- `compute_decoy_ixns` will call this function ~1000 times per pair for mutational mode. The current implementation does NOT cache the gamma load (`load_mediated_gamma` is called once per `water_mediated_energy` call). For the dense path that's fine; for the scalar path it would be slow. Recommend a thin caching wrapper or just pre-loading gammas at the decoy-driver level.
- `rho` is treated as a fixed input. In configurational decoy mode this is correct (ρ doesn't change when identities are shuffled). In mutational mode it's ALSO correct because ρ depends only on coordinates, not identity. Both modes safe.
- The early-out `(n, n)` zero dict has fewer keys than the full path (noted under Medium). If Phase 3 decoy code branches on key presence, it'll crash on n=0/1 edges. Use `dict.get(key, default)`.

No critical/high regressions expected in Phase 3. The water-mediated implementation is decoy-ready as-is.
