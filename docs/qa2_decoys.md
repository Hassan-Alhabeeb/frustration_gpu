# QA-2 code review — decoy machinery (configurational, mutational, singleresidue)

Reviewer: Opus 4.7 (read-only).
Date: 2026-05-21.
Files reviewed:
- `F:/research_plan/frustration_gpu/src/decoys.py` (729 LOC, configurational mode)
- `F:/research_plan/frustration_gpu/src/mutational_decoys.py` (982 LOC, mutational mode)
- `F:/research_plan/frustration_gpu/src/singleresidue_decoys.py` (399 LOC, singleresidue mode)

Reference cross-checked against:
- `docs/reference_lammps_awsem/fix_backbone.cpp:5070-5411`
- `docs/phase_3a_review.md`, `phase_3b_review.md`, `phase_3c_review.md`
- `docs/optimization_sprint_results.md` (135/135 passing, Spearman ≥ 0.997)

## Verdict

**PASS** — the modules are algebraically correct, the optimization-sprint
refactor preserves correctness (the (r, ρ) hoisting + fused (20, N, N) build
is provably equivalent to the per-α loop, and Phase 3c's Spearman ≥ 0.997
gate verifies this numerically), and the rejection sampler / RNG seeding
behave as documented. The findings below are maintainability cruft plus
two latent performance / memory issues that will only bite under
non-default kwargs or on very large proteins (4PKN-scale N=8689). No
sampling-distribution bugs, no wrong-energy bugs, no symmetry bugs on
(i, j) vs (j, i).

Counts: **0 CRITICAL, 0 HIGH, 3 MEDIUM, 4 LOW.**

## Findings by severity

### Critical
**None.**

### High
**None.**

### Medium

**M-1 — `_burial_residue_energy` still has the Python `for w_idx in range(3)` loop that Phase-3a M-3 was supposed to kill.** `mutational_decoys.py:362-374`. The fix in `decoys.py:_burial_total` (configurational, lines 637-643) correctly vectorises across the 3 burial wells with broadcast tensors; the corresponding helper in `mutational_decoys.py:_burial_residue_energy` (which serves BOTH mutational and singleresidue, imported into singleresidue at line 86) still iterates with a Python loop + `torch.stack`. For mutational mode this helper runs FOUR times per call (native i, native j, decoy i, decoy j), and the last two are called on (N_pair, n_decoys) tensors — at 3F9M N_pair = 3349 that's a 3.3M-element tensor going through three Python iterations + a stack op per call. Not a correctness bug (output is identical), but it directly contradicts the optimization-sprint claim that this was vectorised, and is the same anti-pattern the 3a reviewer flagged for performance. Fix is the same 4-line refactor used in `decoys.py:_burial_total`.

**M-2 — `_per_pair_U` recomputes the (r, ρ)-only ingredients 3 times.** `mutational_decoys.py:_per_pair_U` calls `_water_pair_full` three times (lines 645, 667, 688) — once for `U_iSlot_kj`, once for `U_jSlot_ki`, once for the bare pair_term. Each call re-builds `theta_direct`, `theta_med`, `sigma_wat`, `sigma_prot` from the SAME `r_ij_full, rho_i_b, rho_j_b` tensors (or the swapped `rho_j_b, rho_i_b` for `U_jSlot_ki` — but `sigma_wat` is symmetric in (rho_i, rho_j), and so are the θ's which depend only on `r`, so the rho-swap is a noop for these four ingredients). The optimization sprint's Idea-1 (`_water_rho_terms`) was added precisely to hoist these out and is used inside `_precompute_T_alpha`/`_precompute_W_sr`, but `_per_pair_U` was not updated to use it. Cost: 3× redundant tanh evaluations on a (N_pair, n_decoys) tensor — at 11BG roughly 4.5M extra `tanh` ops per call. Not on the critical path of the 60× algorithmic win (`T[i, α]` precompute is unaffected), but a free ~2× drop in `_per_pair_U` wall-clock if you call `_water_rho_terms(r_ij_full, rho_i_b, rho_j_b, ...)` once and pass the four ingredients into a thin per-α gather.

**M-3 — Cross-mask on `decoys.py:374` and again at `mutational_decoys.py:419-420` and `singleresidue_decoys.py:287-292` triplicates the NaN-poisoned-distance handling logic.** Three near-identical hand-rolled "build pairwise distance, replace NaN-row coords with 1e6, then force NaN pairs to +inf" blocks. `_contact_common.py:_pairwise_distance_safe` exists for exactly this and is autograd-safe; the decoy modules predate that helper and copy-pasted the pattern. Not a bug (the three implementations are equivalent), but a known drift surface — if `_pairwise_distance_safe` gets a fix it will not flow here. Recommend consolidating into a single helper in `_contact_common.py` and calling from all three modules.

### Low

**L-1 — Dead `if False:` branch in singleresidue.** `singleresidue_decoys.py:287-289`:
```python
finite_pair = (finite_row & finite_row.transpose(0, 1)).squeeze(-1) if False else (
    finite_row & finite_row.transpose(0, 1)
)
```
The `if False` branch is leftover scaffolding; `finite_pair` is then never used because line 291 immediately overwrites with `finite_pair_2d`. Delete lines 287-290 entirely. Same finding as Phase-3b M2; never cleaned up.

**L-2 — `in_cutoff` mask in `_per_pair_U` is dead code by construction.** `mutational_decoys.py:637-638, 663, 685`. The pair enumeration in `_enumerate_native_pairs` selects only pairs with `dist_full < contact_cutoff`, so `r_ij_b < contact_cutoff` is identically `True` for every row. The `torch.where(in_cutoff, X, zeros)` calls are no-ops. The inline comment on line 638 ("but kept explicit for symmetry with the cross-term mask logic") acknowledges this; same finding as Phase-3b M1, still present. Either delete or escalate the comment to a `# noqa` with a one-liner explaining why future maintenance shouldn't drop it.

**L-3 — Configurational gamma-loader cache key is `str(device)`, not a normalised device tuple.** `decoys.py:165-188`. `torch.device("cuda")` and `torch.device("cuda:0")` are functionally identical on a single-GPU box but `str()` differently, so the lru_cache will hold two entries for the same physical device. Trivial waste; matters only at multi-GPU scale. Same finding as Phase-3a L-2; never fixed. Suggested fix: `device_str = f"{d.type}:{d.index if d.index is not None else 0}"`.

**L-4 — Per-pair scratch tensors `.contiguous()`'d unnecessarily.** `mutational_decoys.py:641-644, 666, 942-943`. Each `(N_pair, 1)` tensor is `.expand(N_pair, n_decoys).contiguous()`'d before being passed to `_water_pair_full` / `_burial_residue_energy`. At 11BG (N_pair=1517, n_decoys=1000) each contiguous copy is 12 MB float64; there are 5 such copies (`aa_j_nat_b`, `rho_i_b`, `rho_j_b`, `r_ij_full`, `aa_i_nat_b`). Total ~60 MB scratch per call that didn't need materialising — `_water_pair_full` consumes them through `gamma[...][aa]` indexing which works on non-contiguous views. At 4PKN scale (projected N_pair ≈ 150k) this same pattern would allocate ~1.2 GB per scratch tensor × 5 ≈ 6 GB just for the broadcast scratch, before any real work. The peak-memory headline of `_choose_alpha_chunk` (which bounds the (20, N, N) tensor) is the dominant cost on big proteins, but these scratch buffers add a constant `5 × N_pair × n_decoys × 8 B` on top that nothing in the chunker accounts for.

## What I verified is still correct (post-optimization sprint)

1. **`T[i, α]` math is identical pre- and post-sprint.** `mutational_decoys.py:485-579` (CPU + GPU branches both produce (N, 20) via different reduction orderings of the same float64 sum tree). The `_water_per_alpha_fused → .sum(dim=2).transpose(0,1).contiguous()` chain on the GPU is mathematically equivalent to the CPU's per-α `w.sum(dim=1)` accumulator (associativity of finite float64 sums isn't strictly guaranteed, but the Phase-3c Spearman 0.998-0.999 gate empirically confirms machine-noise drift only).

2. **Symmetry on (i, j) vs (j, i).** Water-pair energy is invariant under (aa_i, rho_i) ↔ (aa_j, rho_j) (verified algebraically: γ table is symmetric, θ is `r`-only, σ_wat is symmetric in ρ_i, ρ_j). `_per_pair_U` exploits this — `U_iSlot_kj` and `U_jSlot_ki` swap (α, aa_native, ρ) consistently. The native formula `S_i + S_j − W_native_pair + B_i + B_j` is symmetric on input swap (S_i, S_j swap, W is symmetric, B_i + B_j swap). No asymmetry bug.

3. **Cross-residue mask `k != i AND k != j` enforcement.** Both anchors handled correctly: `k != i` enforced by the diagonal mask in `_precompute_T_alpha:520-521`; `k != j` enforced by subtracting `U_iSlot_kj` (i.e. T includes the would-be `k=j` term, U removes it). Symmetric reasoning for the j-anchor. This is the algebraic trick that makes the 60× win possible; verified to match `fix_backbone.cpp:5302` semantics (only spatial cutoff on inner loop, no seq-sep filter).

4. **Configurational rejection sampler.** `decoys.py:404-441` correctly resamples until `p != q AND r_pq < cutoff`; matches C++ `||` → `&` accept-mask after De Morgan. Fallback branch at lines 442-459 now emits a `RuntimeWarning` per the Phase-3a L-3 fix instead of silently tiling — verified at `decoys.py:430-441`.

5. **RNG determinism across devices.** All three modules use a CPU `torch.Generator(seed)` + `torch.randint` + `aa_cpu[idx]` then `.to(device)`. Cross-device reproducibility holds (Phase-3a confirmed CPU/GPU agree to 2.6e-9 on 5AON).

6. **Alpha-chunking adapts to free VRAM.** `_choose_alpha_chunk` calls `torch.cuda.mem_get_info(device)` at the moment of decoy entry; uses 0.25 × free as the budget; falls back to `chunk=1` when memory is tight. Floor `max(1, …)` is correct (`budget // per_alpha` can never go to 0 without the `max`). Ceiling `min(chunk, 20)` prevents a useless oversize chunk. The heuristic is conservative — at 4PKN-projected N=8689 with 10 GB free this gives chunk size 4, well inside the VRAM envelope.

7. **`finite_pair` NaN-poisoned coordinate masking.** All three modules force NaN-containing pairs to `+inf` distance, which guarantees they fail the `< cutoff` rejection in the configurational sampler and the `cross_mask = (dist_full < contact_cutoff)` mask in mutational/singleresidue. Defensive and consistent across the three implementations (even though the same code is rewritten three times — see M-3).

8. **Configurational caching contract.** `configurational_decoy_stats` is documented as stateless; the "cache once per structure" is enforced at the caller architecture level (Phase-3a docstring "The cache is at the architectural level"). No lru_cache on the function itself — correct, because the seed parameter would defeat the cache anyway. Gamma loaders are cached (`_cached_load_*`), which is the only persistent state. Verified: no leaked tensors, no growing globals.

9. **Population std (ddof=0).** All three modules use `.std(unbiased=False)` to match LAMMPS' `compute_array_std` which divides by N. Verified at `decoys.py:652`, `mutational_decoys.py:962`, `singleresidue_decoys.py:379`.

10. **n=0 / n<2 edge cases.** Configurational raises on `n<2` (correct; decoy sampling is undefined for monomers). Mutational has an `n_pair == 0` short-circuit returning empty tensors at lines 897-909. Singleresidue has an `n == 0` short-circuit at 358-367. All three exit safely with correctly-shaped empties.

## Top 3 findings

1. **M-1**: `_burial_residue_energy` still has the Python well-loop that Phase-3a M-3 already fixed in `decoys.py` — the configurational fix never propagated to the mutational/singleresidue shared helper despite being the original rationale for the M-3 patch.
2. **M-2**: `_per_pair_U` recomputes `theta_direct, theta_med, sigma_wat, sigma_prot` three times when one `_water_rho_terms` call suffices. The hoisting helper exists in the same file; the per-pair code path just doesn't use it. ~2× speedup available on the hot mutational path.
3. **L-4**: Five `.contiguous()` broadcasts on (N_pair, n_decoys) scratch tensors that `_water_pair_full` would accept as views. At 11BG ~60 MB wasted; at 4PKN scale projects to ~6 GB, outside what `_choose_alpha_chunk` budgets for.

## Anything that surprised me

- **The dead `in_cutoff` mask** in `_per_pair_U` is a noop by construction yet has been carried forward through two reviews. The author's inline comment acknowledges this, but the dead `torch.where` is still emitted at every call — the right move would be to drop it (3 lines) and put the rationale in a comment, not leave the dead op in the hot path.
- **`_per_pair_U` was the obvious place** for the optimization-sprint's Idea-1 hoisting, given that the helper does three back-to-back water-pair evaluations with identical (r, ρ). The sprint inlined Idea-1 in `_precompute_T_alpha` but missed the second-largest beneficiary in the same file. Easy ~2× win left on the table.
- **Three independent copies of "NaN-safe pairwise distance" math** (decoys.py:363-378, mutational_decoys.py:415-420, singleresidue_decoys.py:283-292) when `_contact_common._pairwise_distance_safe` is right there. The decoy modules predate the helper and never adopted it; refactoring would shrink each by ~10 LOC and remove a drift surface.
- No CRITICAL or HIGH findings — the math is solid. All issues are maintainability and latent performance under non-default kwargs or 4PKN-scale N.
