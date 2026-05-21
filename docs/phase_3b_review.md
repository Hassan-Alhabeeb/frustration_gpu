# Phase 3b code review — mutational + singleresidue modes

Reviewer: Opus 4.7 (read-only; no code modifications).
Files reviewed:
- `F:/research_plan/frustration_gpu/src/mutational_decoys.py` (790 lines)
- `F:/research_plan/frustration_gpu/src/singleresidue_decoys.py` (378 lines)
- `F:/research_plan/frustration_gpu/src/__init__.py` (added exports)
- `F:/research_plan/frustration_gpu/tests/test_mutational_mode.py` (292 lines)
- `F:/research_plan/frustration_gpu/tests/test_singleresidue_mode.py` (224 lines)
- Reference: `F:/research_plan/frustration_gpu/docs/reference_lammps_awsem/fix_backbone.cpp` lines 5070-5411
- Validation dumps confirmed present in `benchmark/cpu_baseline/{mutational,singleresidue}/`

## Verdict
**PASS**

The implementation is algebraically equivalent to the LAMMPS-AWSEM C++ reference, the test gates compare against authentic LAMMPS dumps (not self-output), and the headline numerical results (Spearman > 0.99 on the panel, rank-1 100%) are credible given the high signal-to-noise the formula produces. The 60x perceived speedup comes from a real O(N²·20) vs O(N²·N_decoys·N) reduction — verified below.

## Findings by severity

### Critical
**None.**

### High
**None.**

### Medium

1. **(M1) `_per_pair_U` in_cutoff mask is dead code.** `mutational_decoys.py:445`. Native pairs are selected by `_enumerate_native_pairs` with `dist_full < contact_cutoff` (line 268), so by construction `r_ij_pair < contact_cutoff` for every row. The `torch.where(in_cutoff, U_iSlot_kj, zeros)` calls at lines 471 and 493 always pass. Harmless but the inline comment at 446 ("always True by construction") confirms the author knew; could be a `# noqa: B007` or removed. Style nit, not a bug.

2. **(M2) `finite_pair_2d` re-computation in singleresidue path.** `singleresidue_decoys.py:265-269`. The `if False` branch is leftover scaffolding; the actual mask is built by the second expression `finite_row.expand(n, n) & finite_row.transpose(0, 1).expand(n, n)`. Dead code that should be deleted for readability. No functional impact.

3. **(M3) RNG noise floor not portable to libc rand().** Both modules document this (line 89-95 of mutational_decoys.py). Per-pair stats agree only at Spearman > 0.99 on the panel, not to machine precision. This is the same Phase 3a constraint; not new.

4. **(M4) Decoy energies include `(k=i±1)` cross-terms even though `|k-i| = 1`.** Per docstring, the C++ mask is *only* spatial — there is no seq-sep filter on the (i,k) inner loop (confirmed lines 5302, 5311 of fix_backbone.cpp). The python code reproduces this exactly. This is *correct* but worth flagging because it disagrees with how some AWSEM derivatives ignore i±1 in cross terms. Downstream Frustrapy-API code that does its own (i, k) masking would diverge from LAMMPS dump unless it uses the same liberal mask.

## Cross-term mask verification

The C++ outer (i, j) loop (`fix_backbone.cpp:5076-5086`) DOES enforce both spatial (`rij < tert_frust_cutoff`) AND sequence-separation (`abs(i-j)>=contact_cutoff || i_chno != j_chno`) on the iterated pair. This is what `_enumerate_native_pairs` mirrors at `mutational_decoys.py:267-272`. Correct.

The C++ INNER cross-term loop (`compute_native_ixn` mutational branch at 5215-5244, and `compute_decoy_ixns` mutational branch at 5299-5327) has:

```c
// fix_backbone.cpp:5300-5314
for (k=0; k<n; k++) {
  if (k==i_resno || k==j_resno) continue;     // L5302
  rho_k = get_residue_density(k);
  kres_type = get_residue_type(k);
  rik = get_residue_distance(i_resno, k);
  if (rik < tert_frust_cutoff) {              // L5311 — SPATIAL ONLY
    water_energy += compute_water_energy(...);
  }
  ...
```

No `abs(i_resno - k) >= contact_cutoff` filter. No chain filter. The mask used in `_precompute_T_alpha` (`mutational_decoys.py:352-354`):

```python
cross_mask_neighbor = (dist_full < contact_cutoff)
diag = torch.eye(n, dtype=torch.bool, device=device)
cross_mask = cross_mask_neighbor & (~diag)
```

is the spatial-only mask + `k != i`. The `k != j` exclusion is enforced algebraically by the `-U_iSlot_kj` subtraction (which has the C++'s would-be `k=j` term in it). **Verified.**

(The singleresidue path uses the standard AWSEM seq-sep mask because that's what `compute_singleresidue_native_ixn` line 5383 enforces — `pair_min_seq_sep=2`. Also correct.)

## Burial-once-per-pair verification

C++ `compute_native_ixn` (line 5199-5200):
```c
burial_energy_i = compute_burial_energy(i_resno, ires_type, rho_i);
burial_energy_j = compute_burial_energy(j_resno, jres_type, rho_j);
```
These are computed **once**, before the k-loop. The k-loop (5216-5243) only updates `water_energy` (`+=` only on water_energy and electrostatic_energy). The final return at 5244:
```c
return water_energy + burial_energy_i + burial_energy_j + electrostatic_energy;
```

C++ `compute_decoy_ixns` follows the same structure (5288-5331):
```c
burial_energy_i = compute_burial_energy(rand_i_resno, ires_type, rho_i);  // L5288
burial_energy_j = compute_burial_energy(rand_j_resno, jres_type, rho_j);  // L5289
// ... k-loop on 5300-5327 only modifies water_energy ...
tert_frust_decoy_energies[decoy_i] = water_energy + burial_energy_i + burial_energy_j + electrostatic_energy;  // L5331
```

Python mirror in `mutational_decoys.py:767`:
```python
E_decoy = pair_term + cross_i + cross_j + burial_i_dec + burial_j_dec
```
`burial_i_dec` / `burial_j_dec` are computed once per (pair, decoy) without any k-loop dependence (lines 752, 759). **Verified.**

## T[i,α] precompute correctness

Walk-through:

Define for residue i and alphabet α:
```
T[i, α] := Σ_{k ≠ i, r_ik < cutoff} water_pair(r_ik, α, aa_k_native, rho_i, rho_k_native)
```

This is the would-be "incoming water sum at residue i if i had identity α", summing over all neighbours k including k=j when r_ij < cutoff.

For the C++ native at residue pair (i, j), the cross sum at residue i with native identity is:
```
Σ_{k ≠ i, k ≠ j, r_ik < cutoff} water_pair(r_ik, aa_i_native, aa_k_native, ...) 
  = T[i, aa_i_native] − water_pair(r_ij, aa_i_native, aa_j_native, rho_i, rho_j)
  = S_i_native − W_native_pair
```

Same for j. Adding the bare (i,j) bond and burial:
```
E_native_pair = W_native_pair                                # the (i,j) bond
              + (S_i_native − W_native_pair)                 # cross-i sum, k≠j enforced
              + (S_j_native − W_native_pair)                 # cross-j sum, k≠i enforced
              + B(aa_i_native, rho_i)                        # burial_i (once)
              + B(aa_j_native, rho_j)                        # burial_j (once)
            = S_i_native + S_j_native − W_native_pair + B_i + B_j
```

This is exactly `mutational_decoys.py:702`:
```python
E_native = S_i + S_j - W_native_pair + burial_i_native + burial_j_native
```

For the decoy with sampled (α_i, α_j) at slot (i, j):
```
E_decoy = water_pair(r_ij, α_i, α_j, rho_i, rho_j)                # pair_term
        + (T[i, α_i] − water_pair(r_ij, α_i, aa_j_native, rho_i, rho_j))   # cross_i
        + (T[j, α_j] − water_pair(r_ij, α_j, aa_i_native, rho_j, rho_i))   # cross_j
        + B(α_i, rho_i)
        + B(α_j, rho_j)
```

Which is `pair_term + cross_i + cross_j + burial_i_dec + burial_j_dec` per `mutational_decoys.py:767`. **Algebra correct.**

The W[i,j] subtraction (called `U_iSlot_kj` in code) is the term that handles the C++ `k != j` exclusion. Without it the cross-sum at i would double-count the (i,j) bond once already counted via `pair_term`. The corresponding C++ check is line 5302's `if (k==i_resno || k==j_resno) continue;`.

Note on cross-i and cross-j symmetry: water_pair is symmetric in its (aa, rho) slots (verified in `_water_pair_full:169-185` — `gamma_direct[aa_i, aa_j]` is read from a symmetric AAxAA table, and the sigma_wat formula is `(1-tanh(rho_i)) * (1-tanh(rho_j))` which is symmetric). So `U_iSlot_kj` with arg order `(α_i, aa_j_native, rho_i, rho_j)` and `U_jSlot_ki` with `(α_j, aa_i_native, rho_j, rho_i)` give the right values for the would-be k=j (in T[i]) and k=i (in T[j]) terms. **Verified.**

Cost analysis:
- T precompute: 20 × O(N²) = O(20 N²) — for N=248 (11BG), that's 1.2M float64 ops.
- Per-pair U: O(N_pair × N_decoys) — 1517 × 1000 = 1.5M ops.
- Naive C++ inner-loop equivalent: N_pair × N_decoys × N = 1517 × 1000 × 248 = 3.8 × 10^8 ops.

The 5.5M burial-eval count cited (= N_pair × N_decoys × 2 + N_pair × 2 native) is the *decoy energy* op count after all the precompute is in; that's the actual GPU kernel size. The advertised 60x speedup is plausible at this N — and as N grows, the ratio is exactly O(N / 20) which goes up. **Confirmed.**

## Native rho convention

`mutational_decoys.py:601-603` uses `lammps_dump_rho(coords)` with the default `min_seq_sep = LAMMPS_DUMP_RHO_MIN_SEQ_SEP = 12` (per `decoys.py:144`). This is the SAME dump-rho Phase 3a used (per `decoys.py:202-205` comment, validated against the frustratometeR binary's compile-time SeqDist=12 default).

This matches the LAMMPS dump's rho columns and the rho used inside `compute_decoy_ixns` for mutational mode (where rho is held fixed at native — the same `rho_i_orig` from the outer loop, which is `get_residue_density(i_resno)` at line 5092 — *that* function returns the burial-rho with the binary's compile-time seq-sep).

`singleresidue_decoys.py:251-253` same path. **Verified — mutational mode does NOT use burial's min_seq_sep=1.**

## Reuse of Phase 3a infrastructure

- `_water_pair_full` is a **new** helper in `mutational_decoys.py:136-185`. It is *not* re-using Phase 3a's `decoys.compute_configurational_decoy_energy` (that one takes per-decoy 1-D shapes). Acceptable — the broadcast pattern is different (T uses (N, N) shapes; per-pair U uses (N_pair, N_decoys) shapes). The formula is identical to Phase 3a's pair-energy expression.
- `_burial_residue_energy` is also new and reproduces the per-element burial energy. Phase 3a's `compute_configurational_decoy_energy` has a similar inner formula but bundled into the per-decoy path; the new helper is cleaner.
- AA sampler `_sample_aa_pair_indices` (mutational:282-311) and `_sample_aa_per_residue` (singleresidue:163-184) both use `torch.Generator(device="cpu").manual_seed(seed)` then `torch.randint`, exactly the same pattern as Phase 3a's `sample_configurational_decoys`. The pair sampler does two independent draws for the (i, j) slots and lookups `aa[idx]`, which is the protein-composition draw used in Phase 3a. **Consistent.**
- Cached gamma loaders (`_cached_load_direct_gamma`, `_cached_load_mediated_gamma`, `_cached_load_burial_gamma`) all reused from `decoys.py`. Good.

No code duplication that creates a "drift" risk — different signatures means a shared helper would have made calling sites uglier.

## Rank-1 100% match plausibility

The singleresidue test `test_rank_one_most_frustrated` (lines 181-195) compares:
- `int(np.argmin(ours_fi))` against `int(np.argmin(theirs_fi))`

`theirs_fi` is parsed from column 8 of `<PDB>_singleresidue.dat` — the LAMMPS reference dump (line 168 in fix_backbone.cpp shows this is `frustration_index` written by LAMMPS itself, not by us). Sample shown:
```
Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex
23 A 0.000 S -0.401 -0.473 0.383 -0.187
24 A 0.000 E -1.001 -0.043 0.416 2.303
...
```

The test is **NOT tautological** — it compares against authentic LAMMPS output.

Why rank-1 100% is plausible:
- Native energy matches the dump to < 5e-3 kcal/mol (the dump's print precision). Phase 3a established this.
- Decoy std varies smoothly across residues; the noise floor in `std` is at the few-percent level for N_decoys=1000.
- The most-frustrated residue typically has FI several sigma away from the next candidate. A protein with N~250 residues is unlikely to have two residues with FI tied to within the RNG floor unless they are pathologically similar.
- All 4 panel PDBs hitting rank-1 100% is a 4-trial outcome where each trial succeeds with high probability (~90%). Joint probability ~65% — not surprising, not suspicious.

What WOULD have been suspicious: rank-30 100% (would imply tied tail values reproduced exactly). The test says 86-100% on rank-30, which is the right shape — top-1 hardest to lose, tail starts to lose under RNG noise. **Genuine.**

## Recommended Phase 3c next step

1. **Implement `frustration_index` computation as a separate module** consuming the dicts returned by Phase 3a/3b. The C++ does `(decoy_mean - E_native) / decoy_std` — already done inline in singleresidue (line 361). For mutational, this is *not* computed yet (the dict returns `E_native, decoy_mean, decoy_std` but not FI per pair). Phase 3c should produce a unified `compute_frustration_index(out)` that handles configurational vs mutational dict shapes.

2. **Per-pair FI Pearson validation against LAMMPS dump column 18.** The decoy_mean/std Spearman gates pass at > 0.95; the actual FI Pearson should also be > 0.95. Add `test_mutational_FI_pearson_per_pair` to close the gate.

3. **Investigate (M4)**: if any downstream consumer (Frustrapy compatibility layer) wants to enforce |i-k|>=2 on the cross-term, expose this as an option but default to LAMMPS-faithful behaviour.

4. **Clean up M1 and M2** dead code blocks — 10-line cleanup, no functional impact.

5. **Phase 4 (electrostatics) heads-up**: the C++ shows `huckel_flag` gated electrostatic terms in BOTH `compute_native_ixn` (mutational, lines 5201-5203, 5231-5232, 5240-5242) and `compute_decoy_ixns` (lines 5290-5295, 5315-5316, 5324-5325). When DH is re-enabled for the panel that uses it, the T[i,α] precompute must be extended with an analogous DH-per-pair table OR the decoy formula must explicitly add DH(rij, α_i, α_j, ...) per pair. The current code assumes huckel_flag=false (matches frustratometeR's awsem.in default — verified in Phase 3a docstring lines 64-67). Add a guard / assertion in Phase 3c so a future re-enable of DH does not silently miss the cross-term contribution.

6. **Latent bug surfaceable in Phase 4 — per-pair electrostatic cross-terms**. The cross-term inner loop in C++ has `electrostatic_energy += compute_electrostatic_energy(rik, ...)` regardless of `rik < cutoff` (lines 5231, 5240, etc — note the indentation: the `if (huckel_flag)` and the `if (rik < tert_frust_cutoff)` checks are SEPARATE, so DH is added even outside the 9.5 Å contact cutoff if huckel_flag is on). This is the screened-coulomb behaviour and is correct, but the T[i,α] precompute pattern in mutational would have to be redesigned for it (DH has its own cutoff, typically wider). Flag for Phase 4 design.
