# Speculative optimization opportunities — 2026-05-20

Reviewer: Opus 4.7. Code: `src/*` (Phases 1-3b complete, 108/108 tests).
Status: **read-only audit, no code modified**.
Reference baseline: 11BG (N=248) wall-clock 0.99 s CPU / 82 ms GPU; 4PKN (N=8689) untested.

The audit covers `parser.py`, `virtual_atoms.py`, `parameters.py`, `burial.py`, `_contact_common.py`, `direct_contact.py`, `water_mediated.py`, `contact_gamma.py`, `debye_huckel.py`, `decoys.py`, `mutational_decoys.py`, `singleresidue_decoys.py`, `frustration.py`. Confidence rating for each idea: H/M/L.

---

## Summary

| # | Idea | Module | Estimated speedup | Engineering cost | Accuracy risk | Confidence |
|---|---|---|---|---|---|---|
| 1 | Factor identity-independent terms out of the alpha-loop in `_precompute_T_alpha` / `_precompute_W_sr` | mutational, singleresidue | **3–8×** on the precompute (the dominant CPU cost on small N) | ~25 LOC | Zero (algebraic refactor) | H |
| 2 | Single fused (N, N, 20) tensor build replacing the Python `for alpha in range(20)` loop | mutational, singleresidue | **3–10× on GPU** via launch-overhead amortisation | ~40 LOC | Zero | H |
| 3 | Sparse contact-list representation for direct + water-mediated + DH | direct_contact, water_mediated, debye_huckel | **5–50×** on large N (8689 res in particular); also enables 4PKN to run at all | ~150 LOC, schema change | Zero (mask is the same set) | M |
| 4 | Share dist + safe_dist + mask + chain_idx across the three contact terms (single pass) | _contact_common + direct + mediated + DH | **1.5–2×** on the contact-term half of every run | ~80 LOC | Zero | H |
| 5 | Drop the (N, N, 3) `diff` intermediate via cdist or chunked vector_norm | _contact_common | **2× peak memory**, modest speed | ~30 LOC | Zero | M |
| 6 | Batched-protein dispatch (run K PDBs in one kernel) | top-level wrapper | **10–50×** throughput on small proteins (N ≤ 500) | ~100 LOC plus a new module | Zero | M |
| 7 | Avoid `.contiguous()` after `expand` when shape is consumed elementwise | mutational, singleresidue | **1.5–3×** memory drop in precompute; some speedup | ~10 LOC | Zero | H |
| 8 | Cache T_alpha / W_sr across modes (or across decoy seeds) | mutational + singleresidue | If both modes run on same PDB: **2×** the W_sr part | ~30 LOC | Zero | M |
| 9 | Inline cheap reductions; remove dead `torch.where(in_cutoff, …)` in `_per_pair_U` | mutational | ~5–10 µs per pair on GPU launch overhead; ~20% on 11BG | ~5 LOC | Zero (review M1 confirms dead code) | H |
| 10 | Precompute the (20, 20) `gamma_*` tables ALWAYS into `(20, 20, 1, 1)` shape for broadcast | water_mediated | Negligible compute, but readability + ~5% less reshape cost | ~5 LOC | Zero | L |
| 11 | Switch from per-decoy `expand().contiguous()` in `_per_pair_U` to broadcast-only | mutational | ~2× the memory for the cross-term U build; modest speed | ~15 LOC | Zero | H |
| 12 | Replace the per-`alpha` Python loop with a `bmm`/`einsum` against gamma stacked as (20, 20, 20) | mutational, singleresidue | **2–4× over Idea 1**, but Idea 1+2 already capture most of it | ~30 LOC | Zero | L |
| 13 | Use a single chained `torch.compile` decorator on the hot module-level functions | mutational, contact terms | **Unknown, possibly 2× on GPU**; risk of breaking float64 contract or pytest determinism | ~5 LOC | Possible — torch.compile rewrites numerics | L |

Top picks are **1, 2, 4, 3** in that order (see Recommendation below).

---

## Idea 1 — Factor identity-independent expressions out of the alpha loop

**Module**: `mutational_decoys.py:_precompute_T_alpha` (lines 318–387) and `singleresidue_decoys.py:_precompute_W_sr` (lines 96–156).

**Current cost**: For N=248 (11BG), the alpha-loop runs `_water_pair_full` 20 times on (N, N) tensors. Inside that function (lines 173–185 of `mutational_decoys.py`):

```
theta_direct = water_theta(r, direct_r_min, direct_r_max, eta)    # depends only on r
theta_med    = water_theta(r, mediated_r_min, mediated_r_max, eta) # depends only on r
sigma_wat    = 0.25 * (1 - tanh(η_σ(ρ_i - ρ_0))) * (1 - tanh(η_σ(ρ_j - ρ_0))) # depends only on rho
sigma_prot   = 1 - sigma_wat
```

None of these depend on `aa_i`. Yet the alpha-sweep at `mutational_decoys.py:364-386` recomputes them on every one of 20 iterations. The recomputed work per alpha is (per (N,N) tensor): 2× tanh (8 N² ops each), 2× theta combine (~10 N² ops), 1× sigma_wat (~10 N² ops). At N=248, that's ~3M ops × 20 = 60M ops just in identity-independent terms.

**Proposed cost**: Precompute the four (N, N) tensors once; reuse across the alpha sweep. Per alpha iteration only the 3 gamma gathers (`g_dir[α, aa_col]`, `g_mp[α, aa_col]`, `g_mw[α, aa_col]`) and the final 4-multiply-and-add remain — about ~6 N² ops per alpha. Net work drops from ~16·N²·20 to ~16·N² + 6·N²·20, i.e. ~136·N² instead of ~320·N². **~2.3× compute** for the precompute. On GPU this is amplified by the kernel-launch reduction (every torch op is at least one launch).

**Math walkthrough**: `_water_pair_full(r, α, aa_col, ρ_i, ρ_k) = -k_w·(g_dir[α, aa_col]·θ_dir(r) + (σ_p·g_mp[α, aa_col] + σ_w·g_mw[α, aa_col])·θ_med(r))`. The terms `θ_dir(r)`, `θ_med(r)`, `σ_w(ρ_i, ρ_k)`, `σ_p = 1 - σ_w` are identical for every value of α. Lifting them out of the loop is a pure algebraic re-grouping. The bit-exact result is identical because tanh/multiply are float64-deterministic when applied to the same inputs in the same order — which they are.

For W_sr in singleresidue: identical refactor; even more impactful since singleresidue is dominated by the precompute (no per-pair U/cross-term stage).

**Why machine precision preserved**: Pure algebraic factoring; no chunking, no float32, no reordering of additions that could expose floating-point non-associativity (we never SUM across alpha — alpha is an output dimension).

**Tests that would gate it**: `test_mutational_mode.py::test_native_pair_match_lammps` (machine-precision native energy), `test_mutational_mode.py::test_decoy_stats_spearman_vs_lammps` (Spearman > 0.99), `test_singleresidue_mode.py::test_native_per_residue_match_lammps`. All should pass unchanged.

**Engineering cost**: ~25 lines total. Move the four tensor computations from inside `_water_pair_full` to outside the loop; create a new helper or inline.

**Confidence**: H. The refactor is straightforward and the savings are immediate.

---

## Idea 2 — Replace the alpha Python loop with a single fused (N, 20, N) tensor build

**Module**: same as Idea 1.

**Current cost**: Python `for alpha in range(20)` loop dispatches 20 separate kernel launches per torch operation inside `_water_pair_full`. On GPU each launch is ~5–80 µs of fixed overhead regardless of N. The body of `_water_pair_full` contains ~8 torch ops (tanh, mul, add, gather, etc), so per alpha that's ~8 launches × 20 alpha = 160 launches just for the precompute. At ~30 µs/launch that's ~5 ms of pure overhead per call on RTX 4070 — non-trivial on the 82 ms total runtime.

**Proposed cost**: Build `aa_alpha_all` as `arange(20).view(20, 1, 1).expand(20, N, N)`, then broadcast against `aa_col` (1, 1, N) once. Result `g_dir_all` is (20, N, N). Sum and accumulate via a single `(20, N, N)` reduction. **Number of kernel launches: ~10 instead of ~160.** Plus, the inner work is identical to what was already happening (same flops); we just amortise the launch overhead. On CPU there is no launch overhead so the gain is smaller (just ~10–20% from reduced Python interpreter overhead).

Alternatively, use `torch.einsum`:
- `g_dir_pair_all = torch.einsum('ab,bn->abn', gamma_direct, F.one_hot(aa_native, 20))` — gathers all 20 alpha rows at once.

Or simply:
- `gamma_direct[:, aa_col]` returns (20, N, N) in one shot.

**Why machine precision preserved**: No reduction across alpha; alpha is a stored axis. The eventual `.sum(dim=2)` over N (the neighbour axis) is the same partial-sum tree it always was. Algebraically identical.

**Tests that would gate it**: same as Idea 1.

**Engineering cost**: ~40 LOC. Slightly trickier than Idea 1 because the (20, N, N) tensor for 4PKN at N=8689 is 12 GB float64. The fix is to use float64 only at the reductions and float32 for the alpha-expanded tensors — but that breaks machine precision, so a chunked alpha approach (alpha=0..9 then alpha=10..19) is the safer move. For all panel proteins with N ≤ ~1500, the (20, N, N) tensor at float64 is < 1.5 GB and fits comfortably.

**Confidence**: H for N ≤ 2000 proteins; M for 4PKN (needs chunking to avoid OOM). Compose with Idea 1 — together they realistically deliver **3–10× on the precompute on GPU**.

---

## Idea 3 — Sparse contact-list representation for the three pairwise terms

**Module**: `direct_contact.py`, `water_mediated.py`, `debye_huckel.py`, `_contact_common.py:_pairwise_distance_safe`.

**Current cost**: All three modules build the full (N, N) distance + theta + gamma matrix. At N=8689 (4PKN):
- `diff` tensor in `_pairwise_distance_safe`: 8689² × 3 × 8 B = 1.81 GB
- `dist`, `safe_dist`, `theta`, `gamma_pair`, `pair_energy`, `mask`: ~6 × 8689² × 8 B = 3.62 GB
- Per term, transient peak: ~5.5 GB. Three terms back-to-back: GPU memory likely exceeds RTX 4070's 12 GB.

The actual contact list (`dist < 9.5`) for a real protein has < 30·N entries (each residue has < 30 contacts). At N=8689 that's ~260k pairs — **0.4% of the dense matrix**.

**Proposed cost**: One initial dense distance scan (cheap, single big matmul-like op) to extract the contact list (i, j, r_ij) for r_ij < cutoff_max (9.5 Å for water terms, 30+ Å for DH where exp(-r/λ) decays). Then all downstream computation in `direct_contact_energy`, `water_mediated_energy`, `debye_huckel_energy` runs on a 1-D tensor of length N_pair ≈ 30·N instead of (N², ). Memory drops from O(N²) to O(N·k) where k ≈ 30. For 4PKN: from 5.5 GB to 60 MB.

Compute: dense version was O(N²) tensor ops on tensors that are 99.7% masked-out zeros — wasted work. Sparse is O(N_pair) with no waste. At N=8689 the ratio is ~30·N / N² = 30/N ≈ 290× — but the initial scan is still O(N²). Effective speedup is bounded by the scan; in practice **5–50×** depending on whether the dense terms are the bottleneck.

**Why machine precision preserved**: The contact list is built by `r < cutoff` which is the same predicate as the dense mask. The sum over an upper-triangular sparse list of N_pair entries is the same sum as the upper-triangle of the dense matrix (same addends, same order if we sort by (i, j)).

**Tests that would gate it**: `test_direct_contact.py`, `test_water_mediated.py`, `test_debye_huckel.py` — all should be unchanged because the public API (returning a scalar or a per-pair dict) can preserve its shape. The internal mechanism changes.

Caveat: if any consumer expects the full (N, N) `pair_energy` matrix as a return, the sparse implementation must reconstruct or document the change. Looking at the code, `return_pair_matrix=True` returns full (N, N) matrices used in tests — would need a `.to_dense()` path for backward compat, which negates some of the memory win. But for the SCALAR return (the default), we're fine.

**Engineering cost**: ~150 LOC. Schema-touching. This is the biggest engineering effort in the list. Worth it ONLY if 4PKN scaling is on the critical path.

**Confidence**: M. The math is right; the engineering is the work. Also addresses the headline "we haven't even run 4PKN" concern.

---

## Idea 4 — Single-pass contact-term pipeline (share dist + mask + chain_idx)

**Module**: `direct_contact.py`, `water_mediated.py`, `debye_huckel.py` all call `_resolve_contact_coords` + `_build_chain_index` + `_pairwise_distance_safe` + `_pair_mask` independently. Three identical (N, N) distance matrix builds.

**Current cost**: Each contact term independently builds:
- `cb_or_ca = _resolve_contact_coords(coords, device=device)` — (N, 3) — small
- `chain_idx = _build_chain_index(...)` — (N,) — small, but Python dict loop in `_contact_common.py:80-86`
- `_pair_mask(...)` — (N, N) bool
- `_pairwise_distance_safe(...)` — (N, N) distance via (N, N, 3) intermediate

The three-term pipeline does this work **three times**. At N=8689, each redundant pass costs ~2.5 GB transient and ~50 ms GPU time.

**Proposed cost**: Build the dist + mask + chain_idx **once** in a top-level orchestrator (or memoize on a `coords` id). Each term takes these as inputs. Total wall-clock saving: 2/3 of the contact-prep stage = ~50% of the dense-build cost. **1.5–2× on the contact-term half** of the run.

**Why machine precision preserved**: We pass the same dist tensor to all three terms — they all use it identically (it IS the same data, byte for byte). No algebraic change.

**Tests that would gate it**: any test that calls direct_contact_energy + water_mediated_energy + debye_huckel_energy in sequence. The top-level orchestrator pattern would change API slightly (could be optional via a `_precomputed` kwarg). Internal tests unchanged.

**Engineering cost**: ~80 LOC. Add a `ContactContext` dataclass / NamedTuple holding `(dist_full, safe_dist, mask, chain_idx, cb_or_ca)`. Each term gets a private `_apply_to_context` path; the public API auto-builds the context if absent.

**Confidence**: H. This is a standard refactor; well-defined and contained.

---

## Idea 5 — Avoid (N, N, 3) `diff` intermediate via `torch.cdist`

**Module**: `_contact_common.py:_pairwise_distance_safe` (lines 89-167).

**Current cost**: `diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)` creates an (N, N, 3) intermediate. At N=8689 float64 that's 1.81 GB. Then `torch.linalg.vector_norm(diff, dim=-1)` produces (N, N).

**Proposed cost**: `torch.cdist(safe_cb, safe_cb)` — same result, no (N, N, 3) intermediate (it's computed in-stream by the cuBLAS / MKL kernel). On GPU, cdist is also typically faster because it uses optimized GEMM-based distance. Memory drops by ~3× for the intermediate.

But: `cdist` may produce slightly different float64 values vs the explicit subtract+norm path because the BLAS-optimized formula uses `||x||² + ||y||² - 2·x·y` and the sqrt at the end. The error is bounded by machine epsilon but it's a different rounding path. Could fail the bit-exact gate.

**Why machine precision preserved (qualified)**: `cdist` uses a different floating-point reduction path; results agree to ~1e-15 relative, not bit-exact. The downstream tanh sigmoid smooths this out heavily — the energy comparison tests use 1e-3 tolerance — but the FI-Pearson tests use much tighter tolerances. Recommend gating with `torch.use_deterministic_algorithms` and a careful regression test.

**Tests that would gate it**: all distance-sensitive tests. Risk of LOW failure but worth verifying.

**Engineering cost**: ~30 LOC including the safety wrapper. Just swap `safe_dist_raw = torch.cdist(safe_cb, safe_cb)`.

**Confidence**: M. Bit-exactness is the open question.

---

## Idea 6 — Batched-protein processing

**Module**: New top-level wrapper.

**Current cost**: For a proteome-scale run on small proteins (e.g. 1000 PDBs of N=200 each), each call has fixed Python interpreter overhead + ~10 ms of CUDA setup. On RTX 4070 the per-protein 80 ms wall-clock contains ~20–40 ms of overhead that doesn't scale with N.

**Proposed cost**: Pad all PDBs in the batch to N_max, pack into (B, N_max, 3) tensors with mask. All contact / decoy ops are elementwise + matmul — trivially batched. **10–50× throughput** on small proteins. Most of the existing code already operates on (N, N) shapes that broadcast naturally to (B, N, N).

Caveats:
- The Python dict-based `_build_chain_index` doesn't batch easily (needs a vectorized approach with `torch.unique`).
- The rejection sampler in `sample_configurational_decoys` is per-protein (different in-contact masks); harder to batch but possible with mask-aware rejection.
- The mutational `_enumerate_native_pairs` returns variable-length lists per protein (N_pair varies). Needs ragged-batching pattern (similar to `torch.nested` or a per-protein loop on top of a batched precompute).

The wins compound with Ideas 1, 2, 4 — the precompute is batchable but the per-pair loop is harder.

**Why machine precision preserved**: Padding values masked out before any reduction; same arithmetic per protein.

**Tests that would gate it**: New batched tests; existing single-protein tests unchanged (treat batch=1 as a special case).

**Engineering cost**: ~100 LOC for the batching wrapper. Harder than it sounds because of variable-length outputs.

**Confidence**: M. Real win, real engineering cost.

---

## Idea 7 — Drop `.contiguous()` after `expand()` on consumed-as-broadcast tensors

**Module**: `mutational_decoys.py:357-359, 449-452, 474, 750-751` and `singleresidue_decoys.py:129-131, 320-321`.

**Current cost**: Patterns like `rho.unsqueeze(1).expand(n, n).contiguous()` allocate a new (N, N) tensor every time. `.expand()` is a zero-copy view; `.contiguous()` forces a memory copy. For 4PKN at float64: each `.contiguous()` after `expand(8689, 8689)` is a ~600 MB allocation + copy.

These tensors are then consumed by element-wise ops (`-`, `tanh`, `gather`) — none of which require the input to be contiguous. PyTorch handles strided views natively for those.

**Proposed cost**: Drop the `.contiguous()` calls. The tensor remains a view; subsequent torch ops handle the stride. Saves the allocation + memcopy on each.

Specifically:
- `mutational_decoys.py:357-359`: `rho_row`, `rho_col`, `aa_col` all `.contiguous()` after `.expand()`. Used in `_water_pair_full` for `tanh(eta_sigma * (rho_i - rho_0))` — elementwise, doesn't need contig.
- `mutational_decoys.py:449-452, 474`: `aa_j_nat_b`, `rho_i_b`, `rho_j_b`, `r_ij_full`. Used in elementwise ops.

**Why machine precision preserved**: `.contiguous()` does not change values — it changes memory layout. Removing it does not change any output.

**Tests that would gate it**: zero algebraic change — all existing tests should pass without modification.

**Engineering cost**: ~10 lines deleted. Trivial.

**Confidence**: H. The only caveat is that `gather()` MAY require contig input on some PyTorch versions — needs a quick experimental check, but I'd be surprised.

---

## Idea 8 — Cache T_alpha / W_sr across modes

**Module**: `mutational_decoys.py:_precompute_T_alpha` and `singleresidue_decoys.py:_precompute_W_sr`.

**Observation**: When the user runs all three modes (config + mut + sr) on the same PDB:
- `T_alpha` (mutational) uses cutoff-only mask: `r < 9.5 ∧ k ≠ i` (no seq-sep filter).
- `W_sr` (singleresidue) uses the standard contact mask: `r < 9.5 ∧ (|i-j| ≥ 2 OR cross-chain) ∧ i ≠ j`.

These differ ONLY in the same-chain seq-sep filter (|i-j| = 1 contacts are included in mutational mode, excluded in singleresidue). The arithmetic per alpha is identical otherwise.

**Proposed cost**: Compute both at once by:
1. Computing the dense per-alpha `w_alpha = water_pair(r, α, aa_col, rho_row, rho_col)` (shape (N, N), one per alpha — see Idea 2).
2. Apply two different masks → `T_alpha` (sum over `cross_mask`) and `W_sr` (sum over `contact_pair_mask`).

This is ~2× cheaper than computing the two precomputes separately. Combined with Ideas 1+2, this becomes a single fused kernel that produces both outputs.

**Why machine precision preserved**: Pure algebraic factor.

**Tests that would gate it**: both mutational and singleresidue test suites.

**Engineering cost**: ~30 LOC. Best done via a shared `_water_per_alpha_dense(coords, …) → (20, N, N)` helper.

**Confidence**: M. Useful only if both modes are run on the same coords (which the test panel does).

---

## Idea 9 — Remove dead `in_cutoff` masks

**Module**: `mutational_decoys.py:_per_pair_U` (lines 445-446, 471, 493).

**Observation**: Per phase_3b_review M1 (cited in docs/phase_3b_review.md:28), the `in_cutoff` mask in `_per_pair_U` is dead code because every native pair satisfies `r_ij < 9.5` by construction.

**Proposed cost**: Remove the three `torch.where(in_cutoff, ...)` calls. Removes 3 kernel launches per call on GPU. At 11BG that's ~150 µs of avoided overhead (~0.2% of 82 ms).

**Why machine precision preserved**: The mask is always-True by construction (see phase_3b_review.md:28 confirmation).

**Tests that would gate it**: existing mutational tests.

**Engineering cost**: ~5 lines deleted.

**Confidence**: H (already validated as dead code in phase_3b_review).

---

## Idea 11 — Broadcast instead of `expand().contiguous()` in `_per_pair_U`

**Module**: `mutational_decoys.py:449-477`.

**Observation**: `_per_pair_U` does `aa_i_nat_pair.unsqueeze(1).expand(n_pair, n_decoys).contiguous()` repeatedly to get full (N_pair, n_decoys) tensors that are constant along the n_decoys axis. The `.contiguous()` forces a (N_pair × n_decoys × 8) byte allocation per such expansion.

For 11BG: N_pair = 1517, n_decoys = 1000 → 12 MB per tensor × ~4 tensors = 48 MB unnecessary alloc on each call.

**Proposed cost**: `_water_pair_full` broadcasts cleanly; passing the (N_pair, 1) tensor and letting torch broadcast against the (N_pair, n_decoys) aa_dec tensor works identically. ~2× less memory allocation in the per-pair stage.

**Why machine precision preserved**: Same arithmetic, different memory layout.

**Tests that would gate it**: same.

**Engineering cost**: ~15 lines.

**Confidence**: H.

---

## Things investigated but rejected

### Mixed precision (float32 in compute, float64 at reductions)
**Why rejected**: Mutational mode's per-pair `decoy_std` is computed from ~1000 floats with typical spread of 0.5 kcal/mol and mean ~−1 kcal/mol. The std calculation `√(Σ (x − μ)² / N)` loses ~3–4 digits in catastrophic cancellation under float32. The downstream FI = (mean − native) / std then loses another 1–2 digits. Total: float32 would push the noise floor from 5% (current libc/torch RNG floor) to ~10%, breaking the Spearman > 0.99 gate. Not safe.

### Custom CUDA / Triton kernel for the alpha loop
**Why rejected**: User constraint #4 — no new dependencies. Even though a fused Triton kernel could in theory hit ~10× over the current loop, the project rules out adding Triton.

### Numba / Cython / TorchScript JIT
**Why rejected**: Same dependency rule. `torch.compile` (Idea 13) is in-tree PyTorch so technically allowed, but it carries determinism + bit-exactness risks that need careful validation; I've put it on the speculative list rather than recommended.

### Replacing `tanh` with a polynomial approximation
**Why rejected**: At the AWSEM windows (r ∈ [4.5, 9.5] Å, η = 5 Å⁻¹) the argument can reach ±25, where any polynomial approx loses 5+ digits. The native tanh on CPU is already as fast as multiplication on modern hardware. No win, big risk.

### Caching gamma table on a per-call basis with `lru_cache`
**Already done** — see `decoys.py:165-183`. Re-verified that the cache hits on subsequent calls; not a missed win.

### Symmetric AA-pair (only iterate upper triangular on (α_i, α_j))
**Why rejected**: gamma tables are AA-symmetric but the per-pair (α_i, α_j) decoy products are NOT free of (i, j) asymmetry because rho_i ≠ rho_j and aa_i_native ≠ aa_j_native. The (N_pair, n_decoys) tensors don't admit a triangular shortcut.

### Single-precision FFT-based distance computation
**Why rejected**: at N=8689, the FFT distance for ~3 GB would be slower than cdist and lose precision. Not applicable.

### `torch.compile` on the hot precompute
**Why "L" confidence, listed but not recommended**: would change the FP execution path; possibility of non-bit-exact divergence; gate is fragile.

---

## Recommendation

**Phase 5 (optimization sprint, ~1 week)** is justified by 4PKN sitting un-run and 1517 pairs × 1000 decoys / pair clearly room to grow. Recommended order:

1. **Idea 7 (drop redundant `.contiguous()`)** — ~10 minutes work, free win, near-zero risk. Land first as a confidence-builder.
2. **Idea 9 (remove dead `in_cutoff` masks)** — ~5 minutes, free win.
3. **Idea 1 (factor identity-independent terms out of alpha loop)** — half a day, biggest single algorithmic win for the precompute. Validate against test panel.
4. **Idea 2 (fused (20, N, N) tensor)** — 1 day. Compose with #3. Together they should deliver the 3–10× headline.
5. **Idea 4 (single-pass contact-term pipeline)** — 1 day. The 1.5–2× contact-term speedup directly closes the GPU runtime on the 0.99s → 82ms numbers.
6. **Idea 3 (sparse contact lists)** — 2 days. **Only do this if 4PKN refuses to run.** Otherwise the engineering cost is heavy relative to the win for the typical N=250 panel.
7. **Idea 6 (batched-protein)** — 2 days. Worth doing if proteome-scale runs are on the roadmap; defer if not.

**Skip in Phase 5**: Idea 5 (cdist) — too much accuracy risk; Idea 8 — niche; Idea 12 — incremental over Idea 2; Idea 13 — too speculative.

**Total Phase 5 wins if everything lands**: expected **5–15× on mutational+singleresidue on small N**, **enables 4PKN to run** under 4 GB GPU memory, and **2–3× on the contact-term setup half** of every run. Worth a sprint.
