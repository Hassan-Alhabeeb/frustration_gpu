# Phase 2a code review — direct contact term

Reviewer: Opus code-reviewer, 2026-05-20.
Code reviewed: `src/direct_contact.py`, `src/contact_gamma.py`, `tests/test_direct_contact.py`
Reference: `docs/reference_lammps_awsem/fix_backbone.cpp`, `docs/lammps_awsem_term_spec.md`

## Verdict
**CONDITIONAL-PASS** — formula and parameters match the C++ exactly; one latent autograd-NaN risk on GLY-with-no-CB pairs that the existing test does not exercise; a couple of medium-priority issues around the gamma loader's k_water-folding semantics and test coverage of edge cases.

## Findings by severity

### Critical (must fix before Phase 2b)
*(none)*

The formula, parameters, gamma table interpretation, seq-sep filter, CB/CA substitution, sign, factor of 1/4, and the deliberate omission of the spurious 1/2 are all correct against the C++ source. The Phase 2a coder's "no 1/2" call is correct (see "Verified" section below).

### High (should fix in Phase 2b or sooner)

- **`src/direct_contact.py:230-231, 261` — autograd NaN poisoning on `torch.where(mask, dist, const)`.**
  `dist` is built from `cb_or_ca.unsqueeze(0) - cb_or_ca.unsqueeze(1)` and `vector_norm`. For any residue whose effective-CB ends up NaN (fully missing residue with no CA fallback either), the corresponding rows/cols of `dist` are NaN. Forward is correctly masked by `torch.where`, but `torch.where`'s backward computes `grad_dist = grad_safe_dist * mask` — and since the masked-out positions had NaN values that participate in *both* tanh branches even when `mask == False` (mask is applied AFTER the `where`), the upstream gradient flowing back through the NaN-side path is `0 * NaN = NaN`, which poisons gradients on any finite residue that pairs with that NaN row through shared coordinates. The standard "double-where NaN trick" (replace `dist` with `where(mask, dist, dist.detach().nan_to_num(0.5*(rmin+rmax)))` BEFORE the tanh) would fix this. The existing test (`test_differentiable_wrt_cb`) only checks finite-CB rows of `cb_coords.grad`, so the failure mode is not exercised — it would only bite once we have a protein with a fully missing residue (no CA, no CB), which is rare but does occur in PDBs with disordered loops. **Low likelihood of triggering on 5AON/11BG, but a latent footgun before this becomes a training loss term.**

- **`tests/test_direct_contact.py:289-297` — CPU/GPU agreement test is gated behind `torch.cuda.is_available()` and thus skipped in CI without a GPU.** No verification that this test was actually run; the test exists but no recorded pass-output is in the test file's documentation. Run it explicitly on the local GPU before Phase 2b lands.

- **`tests/test_direct_contact.py` — no boundary tests for r_ij = r_min and r_ij = r_max.** The "smoothness at boundaries" check the prompt asked for is missing. At r = r_min, theta = (1/4)(1+0)(1+tanh(η·2)) = (1/4)(1)(~2.0) ≈ 0.5; at r = r_max, by symmetry, theta ≈ 0.5; at r in {3.5 Å, 7.5 Å} (outside the well) theta → 0. Worth a 5-line test that walks a single pair through the window and confirms continuity.

- **`tests/test_direct_contact.py` — no n=1 / empty-protein edge case.** The prompt asked for "single residue (n=1) — should give 0". This is the kind of bug a triu mask + sum naturally handles, but it's not asserted. A one-residue parsed dict (or empty) should return `tensor(0.0)`.

### Medium (cosmetic / future)

- **`src/contact_gamma.py:60-98` and `src/direct_contact.py:217` — silent semantic divergence from C++ on the k_water fold.** The C++ multiplies `water_gamma` by `k_water` at *load* time (`fix_backbone.cpp:632-633`), then uses `-(sigma_gamma * theta)` with no further prefactor at `:5473`. The PyTorch implementation does NOT fold k_water into the gamma table, and instead multiplies in the energy formula. This is **numerically identical** for `k_water = 1.0` (the only value frustrapy uses) and the contact_gamma.py docstring (lines 86-91) correctly flags it. But if a user passes `gamma_direct=...` from a custom source AND `k_water != 1.0`, they may not realize they need to *not* pre-multiply by k_water. Worth either (a) renaming the kwarg to `gamma_direct_raw` to make it explicit, or (b) adding a runtime warning when both `k_water != 1.0` and `gamma_direct is not None`.

- **`src/direct_contact.py:51-55` — docstring claim "advanced indexing with integer tensors is differentiable" is correct for the table values but may mislead readers about coordinate gradients.** The integer-index gather is differentiable w.r.t. the gamma TABLE, not w.r.t. positions. Positions flow through `dist → theta` only. Worth one sentence of disambiguation.

- **`src/direct_contact.py:282-286` — pair_energy is upper-triangularized but `mask` returned in the dict is NOT.** Inconsistent: `out["pair_energy"]` is upper triangle only (consistent with summing the full matrix giving the total), but `out["pair_mask"]` is `mask & upper`, which is upper-tri (good). However `out["distances"]` is the full symmetric matrix. Mixed conventions in one returned dict is mildly confusing. Either upper-triangularize everything, or document that `pair_energy` and `pair_mask` are upper-only while `distances` is full.

- **`tests/test_direct_contact.py:81-106` — `_load_dump_rows` returns 10-tuple but is parsed positionally; small magic-number risk.** Using a NamedTuple or @dataclass here would make the test more maintainable. Not blocking.

- **`tests/test_direct_contact.py:198-200` — tolerance of 0.01 for the reconstructed E_native is loose.** The dump prints to 3 decimals, but ours is float64. The C++ value at line 5473 is a single multiplication; 0.005 would be the tightest principled tolerance. 0.01 is fine for now but consider tightening once Phase 2b lands.

- **`src/direct_contact.py:60-62` — O(N²) memory claim "fine up to ~8k residues at float32 (~256 MB)".** At N=8689 (4PKN), the dense `diff` tensor is 8689² × 3 × 4 bytes ≈ 907 MB at float32 (just `diff`), plus `dist`, `theta`, `gamma_pair`, `pair_energy`, `safe_dist` — easily 3-5 GB transient. The "256 MB" figure undercounts. This is informational rather than a bug, but the comment should be corrected so users don't get surprised by OOMs on the largest test protein.

### Things I verified are correct

- **`src/direct_contact.py:265` ✓ — `theta = 0.25 * (1 + tanh(η(r - r_min))) * (1 + tanh(η(r_max - r)))`** matches `fix_backbone.cpp:5465-5467` to the character.
- **`src/direct_contact.py:276` ✓ — `V_direct = -k_water * γ * θ` (no 1/2)** matches `fix_backbone.cpp:5473` once you account for the C++ folding `k_water` into `water_gamma` at load (`:632-633`). The "1/2" at `:5462` is `sigma_gamma = (γ[0] + γ[1]) / 2` and both columns hold the same number for direct entries (verified at `:6819` "water_gamma_0_direct and water_gamma_1_direct should be equivalent (relic of compute_water_potential)"), so the average is the identity. The Phase 2a coder's call to omit the 1/2 from the user-facing formula is CORRECT and the docstring at `direct_contact.py:22-31` explains it.
- **`src/direct_contact.py:78-81` ✓ — parameters r_min=4.5, r_max=6.5, η=5.0, k_water=1.0, contact_min_seq_sep=2** match `awsem_hamiltonian_spec.md:15-16` and `lammps_awsem_term_spec.md:205-209`, which both quote the C++ `[Water]` block exactly.
- **`src/direct_contact.py:246` ✓ — `sep_ok = (~same_chain) | (seq_diff >= contact_min_seq_sep)`** matches `fix_backbone.cpp:5086` and `:5048` exactly: `(abs(i-j)>=contact_cutoff || i_chno != j_chno)`. Cross-chain pairs always included; same-chain requires `|i-j| ≥ 2`.
- **`src/direct_contact.py:99-102` ✓ — CB-substitution-for-GLY** mirrors `fix_backbone.cpp:3404-3407` and `:5088-5091`: GLY (or any NaN CB) uses CA. The PyTorch generalisation to "any NaN CB" is a strict superset of the C++ behaviour and is required for robust PDB handling.
- **`src/contact_gamma.py:62-98` ✓ — gamma loader** correctly takes column 0 of the first 210 rows of `gamma.dat`, builds a symmetric (20,20), and uses AA order `A R N D C Q E G H I L K M F P S T W Y V`. Verified against the C++ `se_map` array at `fix_backbone.cpp:55` which decodes to exactly the same ordering.
- **`src/direct_contact.py:283-286` ✓ — upper-triangular sum (i<j) instead of full-matrix sum (i,j)** correctly avoids the factor-of-2 double count that would otherwise happen on a symmetric pair_energy. Matches the C++ loop structure at `fix_backbone.cpp:5076` (`for j = i+1`).
- **`src/direct_contact.py:271` ✓ — `gamma_pair = gamma_direct[aa.unsqueeze(1), aa.unsqueeze(0)]`** is standard advanced indexing, differentiable w.r.t. the gamma table values. Confirmed in PyTorch docs that index-tensor indexing on a leaf tensor preserves grad.
- **`tests/test_direct_contact.py:157-200` ✓ — pair-level reconstruction of E_native = -1.003 for (i=1, j=3, S-R, r=5.065 Å)** is a strong end-to-end check. Adds a tiny `_ref_mediated_pair_kcal` (≈ 10 LOC) so the test does NOT depend on un-written Phase 2b code. This is the right way to validate direct in isolation.

## Recommended next steps for Phase 2b

1. **Apply the double-where NaN trick** in `direct_contact.py` before tackling Phase 2b — the mediated module will inherit the same NaN-on-missing-residue pattern, so fix it once now and copy the pattern. Add a regression test using a synthetic protein with a fully-missing residue (NaN CA + NaN CB) and assert grad is finite on its neighbours.
2. **Add the boundary + n=1 edge-case tests** before Phase 2b lands so direct_contact + mediated_contact can share a common test harness.
3. **Run the CPU/GPU agreement test on the local GPU** and record the relative diff in `PHASE_1_STATUS.md`. The current `< 1e-6` tolerance is right for float64; document it.
4. **Decide on the k_water convention for the public API** — either (a) keep "user-facing k_water is a separate knob, gamma table is raw" (current behaviour, easier to explain), or (b) fold k_water into the loaded gamma to match C++ exactly (so frustrapy's exported tables drop in without a divide). Document the decision once.
5. **Phase 2b's mediated term re-uses everything in this module** — the gamma loader (which already returns `mediated_protein` and `mediated_water`), the seq-sep filter, the CB-substitution helper, the `safe_dist` pattern. Lift `_resolve_contact_coords` and `_build_chain_index` into a shared internal module (`src/_contact_common.py` or similar) so direct and mediated don't fork. Same for the dense (N, N) distance matrix — build it once and reuse for both wells.
6. **Tighten the dump-reconstruction tolerance from 0.01 to 0.005** once mediated is in. The current loose tolerance can hide a real bug in the eventual mediated implementation.
