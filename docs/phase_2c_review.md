# Phase 2c code review — DebyeHuckel electrostatics

Reviewer: Opus code-reviewer, 2026-05-20.
Files: `src/debye_huckel.py`, `src/__init__.py`, `tests/test_debye_huckel.py`

## Verdict
**PASS**

The implementation is faithful to `fix_backbone.cpp:5502-5547` line by line, the
test suite covers every load-bearing invariant (charge assignment, sign,
linearity in `k_QQ`, sequence-separation gate, cross-chain bypass, CPU/GPU
parity, differentiability), and the module reuses the Phase 2b
`_contact_common` helpers correctly. No critical or high-severity issues
found. A small number of cosmetic / Phase-3 forward-looking notes are listed
under "Medium".

## Findings by severity

### Critical
None.

### High
None.

### Medium

1. **`k_QQ` sign-pair fan-out is collapsed into a single scalar.**
   `fix_backbone.cpp:5534-5542` keeps three independent constants
   (`k_PlusPlus`, `k_MinusMinus`, `k_PlusMinus`) and dispatches on the sign of
   `charge_i × charge_j`. The current implementation uses a single
   `k_QQ` scalar — which is correct **only because the default
   `fix_backbone_coeff.data` sets all three to 4.15** (confirmed at
   `awsem_hamiltonian_spec.md:32`). If a user ever supplies a coeff file
   where the three differ (e.g. asymmetric salt-bridge biasing), this code
   will silently regress. Recommend: keep the scalar default for the
   `electrostatics_k` API, but expose `k_PlusPlus / k_MinusMinus / k_PlusMinus`
   as optional kwargs that fall back to `k_QQ` when unset. Track for a
   future phase — not blocking 2c.

2. **`fill_value = 1000` is fine but undertested at boundaries.**
   The double-where trick uses `fill = 1000.0`; with `λ_eff = 10`, this gives
   `exp(-100) ≈ 3.7e-44` and `1/r ≈ 1e-3`, so masked-out pairs contribute
   well below float64 epsilon. Worth a single explicit test that verifies
   `pair_energy[mask == False]` is exactly zero after the final `torch.where`
   — currently inferred but not asserted. Not blocking.

3. **`debye_huckel_min_sep` is an int, but the C++ uses `abs(...) <` so the
   semantics are "drop pairs strictly closer than the threshold in sequence."**
   The implementation passes `min_seq_sep` into `_pair_mask`, which (per
   `_contact_common.py:171-216`) drops same-chain pairs with `|i-j| < min_seq_sep`.
   This matches `fix_backbone.cpp:5504` (and the in-pair-list path at 6441)
   semantically. Verified.

## Charge assignment verification

`fix_backbone.cpp:5510-5531` (cited in the implementation docstring at
`debye_huckel.py:36-46`):

```
if (one_letter_code[ires_type]=='R' || one_letter_code[ires_type]=='K') {
  charge_i = 1.0;
}
else if (one_letter_code[ires_type]=='D' || one_letter_code[ires_type]=='E') {
  charge_i = -1.0;
}
else {
  return 0.0;   // ← HIS lands here, returns 0
}
```

Mapped against `src/parser.py:46-51` (`ONE_TO_IDX`) the implementation
hard-codes the 20-tuple `DH_CHARGES_FLOAT` at `debye_huckel.py:119-140`:

| idx | AA | C++ branch              | DH_CHARGES_FLOAT[idx] | OK |
|-----|----|-------------------------|-----------------------|----|
| 0   | A  | else → 0                | 0.0                   | yes |
| 1   | R  | R/K → +1                | +1.0                  | yes |
| 2   | N  | else → 0                | 0.0                   | yes |
| 3   | D  | D/E → -1                | -1.0                  | yes |
| 4   | C  | else → 0                | 0.0                   | yes |
| 5   | Q  | else → 0                | 0.0                   | yes |
| 6   | E  | D/E → -1                | -1.0                  | yes |
| 7   | G  | else → 0                | 0.0                   | yes |
| **8** | **H** | **else → 0** (NOT +1) | **0.0**             | **yes — load-bearing** |
| 9   | I  | else → 0                | 0.0                   | yes |
| 10  | L  | else → 0                | 0.0                   | yes |
| 11  | K  | R/K → +1                | +1.0                  | yes |
| 12  | M  | else → 0                | 0.0                   | yes |
| 13  | F  | else → 0                | 0.0                   | yes |
| 14  | P  | else → 0                | 0.0                   | yes |
| 15  | S  | else → 0                | 0.0                   | yes |
| 16  | T  | else → 0                | 0.0                   | yes |
| 17  | W  | else → 0                | 0.0                   | yes |
| 18  | Y  | else → 0                | 0.0                   | yes |
| 19  | V  | else → 0                | 0.0                   | yes |

`tests/test_debye_huckel.py:76-98` (`test_charge_vector_against_cpp`) plus
the explicit HIS check at line 95 cover this exhaustively. **The
surprising-but-correct HIS = 0 convention is implemented and tested.**

## Screening formula sign + scaling verification

`fix_backbone.cpp:5535-5545`:

```c
if   ((charge_i > 0.0) && (charge_j > 0.0))                    term_qq_by_r = k_PlusPlus  * charge_i*charge_j / rij;
else if (charge_i < 0.0 &&  charge_j < 0.0)                    term_qq_by_r = k_MinusMinus* charge_i*charge_j / rij;
else if ((charge_i < 0.0 && charge_j > 0.0) || ...)            term_qq_by_r = k_PlusMinus * charge_i*charge_j / rij;
...
return epsilon * term_qq_by_r * exp(-k_screening*rij/screening_length);
```

The prefactor is **positive** (`+epsilon`). All sign asymmetry comes from
`charge_i × charge_j`:

* Like-sign (++ or --): `q_i × q_j = +1` → V > 0 (repulsive)
* Opposite (+ -):       `q_i × q_j = -1` → V < 0 (attractive)

`debye_huckel.py:330-331`:

```python
k_QQ_t = torch.as_tensor(k_QQ * epsilon, dtype=dtype, device=device)
full_pair_energy = k_QQ_t * q_outer * decay * inv_r
```

Matches exactly. The `epsilon` factor is folded into `k_QQ_t` (line 330) and
the exponential decay uses `-r * inv_lambda_eff` where
`inv_lambda_eff = k_screening / screening_length` (line 313-315) — identical
to `exp(-k_screening * rij / screening_length)` in C++.

Hand-computed witness in tests:
* `test_pair_value_hand_check_D_K` (`r=10, q=(-1)(+1)` → `-4.15 × e⁻¹ / 10
  ≈ −0.152671`). Implementation matches to `1e-12`.
* `test_pair_value_hand_check_like_charge` (`E-D, r=5, q=(-1)(-1) → +1`)
  asserts `v > 0` AND matches the hand value. **The sign convention is
  correctly attractive for opposite charges and repulsive for like.**

Linear scaling in `k_QQ` is verified at `test_linear_scaling_in_k_QQ_5aon`
and `test_linear_scaling_in_k_QQ_11bg`:
`V_DH(k=17.3636) / V_DH(k=4.15) == 17.3636 / 4.15 = 4.184`
to `1e-10` relative tolerance. This is the `electrostatics_k` API parity
guarantee — passing it through `k_QQ` does the same thing as passing
`electrostatics_k=4.184 × default` would in frustrapy.

The cross-chain bypass at `_pair_mask` (verified via
`test_cross_chain_no_seq_sep_filter`) mirrors `fix_backbone.cpp:6441`
(`abs(i-j) >= debye_huckel_min_sep || i_chno != j_chno`) which is the
inclusive form used by the pair-list iterator.

## "Electro. column = 0" question

I investigated this directly. The summary:

1. **DH is not gated off at the C++ level** — `compute_electrostatic_energy`
   is always compiled in, and `fix_backbone.cpp:5545` always runs when the
   `huckel_flag` block is active. The all-zero `Electro.` column observed in
   `energy.log` for the canonical and `param_sweep` runs is therefore NOT
   "DH was never computed." There are two more likely explanations, both
   compatible with the test suite's docstring assertion:

   a. **`huckel_flag = false` in the LAMMPS `fix_backbone` invocation** the
      frustrapy `awsem.in` script uses. `huckel_flag` is set from the
      `fix_backbone` input flag list at module load (see C++ line 1059 —
      `if (huckel_flag) pair_list_cutoff = MAX(...)`). When the flag is
      false, the `compute_total_energy` dispatch never accumulates the DH
      pair-energy into `Electro.` and it stays at its initialised 0.0.

   b. **(Less likely) The flag IS on but the dump column reports a
      different per-pair accumulator** that wasn't populated in the
      iteration mode frustrapy uses (note the `param_sweep` `electrostatics_k`
      kwarg modifies `k_QQ` even when the energy column is 0 — it sets the
      *value* used should DH ever be on, not whether DH is on).

   Either way: **frustrapy does not include `V_DH` in its native energy or
   decoy energy when running the default configurational mode.** The
   reviewer's empirical observation (`Electro. = 0.0` in param-sweep
   `energy.log`) is consistent with case (a) — which is the case the
   test-file docstring asserts.

2. **Implication for Phase 3 (frustration index validation):**
   The frustrapy native energy that we'll match against is
   `V_Direct + V_Mediated + V_Burial` (the three terms whose columns are
   non-zero in `energy.log`) — NOT `V_DH`. Per-pair native energies in the
   frustration index also exclude DH. So:

   - **Phase 3a target:** compare our `V_Direct + V_Mediated + V_Burial`
     total against the LAMMPS `V_Total` column. We've already established
     `V_Water + V_Burial = -60.500` on 5AON matches the dump within
     reported tolerance — Phase 3 is matching the per-pair decomposition,
     not adding more terms.
   - **Do NOT add V_DH to the per-pair frustration energy** for the Phase 3
     end-to-end validation — that would create a deliberate mismatch with
     frustrapy's output.
   - **V_DH IS available** (and tested) as an independent term that we can
     surface separately (a "predicted electrostatic contribution if DH were
     enabled" diagnostic), but we should not stir it into the totals being
     compared to frustrapy.

3. **The alternative validation routes used in `test_debye_huckel.py` are
   sound**: (a) per-pair scalar formula vs hand-computed value, (b)
   polyalanine = 0, (c) linear scaling in `k_QQ` (proves the API kwarg
   plumbing), (d) per-pair sum == dense sum (proves the masking +
   triangulation logic), (e) CPU/GPU parity, (f) differentiability. These
   collectively pin down every degree of freedom in the formula without
   needing an end-to-end column to compare against.

## Things I verified are correct

* `DH_CHARGES_FLOAT` (20-tuple) matches `fix_backbone.cpp:5511-5527` for
  every AA in `ONE_TO_IDX`, including HIS = 0.
* Sign convention: `+epsilon × k_QQ × q_i × q_j × exp(-r/λ_eff) / r` —
  positive prefactor; sign of energy carried by `q_i × q_j`. Matches
  `fix_backbone.cpp:5545`.
* Default constants: `k_QQ = 4.15`, `λ = 10.0 Å`, `k_screening = 1.0`,
  `min_seq_sep = 1`, `epsilon = 1.0`. All match the `[DebyeHuckel]` block
  in `awsem_hamiltonian_spec.md:31-35` and `fix_backbone.cpp:467-477`.
* `min_seq_sep = 1` excludes only the self-pair (`|i-j| = 0`); same-chain
  `|i-j| = 1` neighbours **do** contribute (verified by the
  per-pair-sum equality test, which iterates `j = i+1`).
* Cross-chain pairs always contribute, irrespective of `min_seq_sep`
  (verified by `test_cross_chain_no_seq_sep_filter`).
* `electrostatics_k` API parity: passing `k_QQ = X × default` linearly
  scales the total by `X` to machine precision.
* `_contact_common` helpers (`_pair_mask`, `_pairwise_distance_safe`,
  `_resolve_contact_coords`, `_build_chain_index`) are reused — no
  re-implementation of pair-mask or NaN-safe distance logic.
* The `q = 0` early-mask optimisation is purely a perf trick: any pair
  where `q_i == 0` or `q_j == 0` would multiply `full_pair_energy` by 0
  anyway. The test
  `test_dense_equals_per_pair_sum` confirms the optimised dense path
  reproduces the explicit per-pair sum to `1e-9`, so the optimisation is
  numerically lossless.
* NaN-safety: `safe_dist` carries a finite fill (1000 Å) into the
  `exp` and `1/r`, and the final `torch.where(mask, ...)` zeroes
  out the contribution. Gradient backprop is verified by
  `test_differentiable_wrt_cb` (finite, non-zero grad on charged residues).
* `n < 2` short-circuit returns 0 with the correct dict shape.
* `return_pair_matrix` dict keys + upper-triangular shape are correct.

## Recommended Phase 3a next step

1. **Implement the per-pair `V_Direct + V_Mediated + V_Burial` native-energy
   matrix** matching frustrapy's per-pair output format. **Do NOT include
   `V_DH`** in this matrix — `Electro. = 0.0` in the LAMMPS dump means
   frustrapy's reference per-pair native energy excludes the DH term, and
   adding it would create a deliberate mismatch.

2. **Surface `V_DH` as a separate auxiliary diagnostic** (e.g. an optional
   column in the per-pair output, gated by an `include_electrostatics`
   flag that defaults to False). This preserves the `V_DH` math for any
   future use case (e.g. when comparing against an OpenMM-AWSEM run that
   *does* have DH on) without poisoning the frustrapy parity check.

3. **Promote the three independent `k_PlusPlus / k_MinusMinus / k_PlusMinus`
   constants** to `debye_huckel_energy` kwargs, defaulting to `k_QQ`. This
   future-proofs the API for non-default coeff files. (Low priority — does
   not block Phase 3a.)

4. **Add the explicit "masked entries are exactly zero in `pair_energy`"
   assertion** mentioned under Medium #2. One-line `assert
   torch.where(~out["pair_mask"], out["pair_energy"], torch.tensor(0.)) == 0`
   style check in `test_return_pair_matrix_shape`.

5. **Phase 3a primary target:** end-to-end `V_Total` reconstruction against
   `energy.log` columns `Direct`, `Water`, `Burial` (which sum to the
   `V_Total` reported by LAMMPS, given `Electro. = 0`). We already have
   `V_Water + V_Burial = -60.500` matching on 5AON; adding `V_Direct` is
   the last column needed before we move to per-pair frustration index
   computation.
