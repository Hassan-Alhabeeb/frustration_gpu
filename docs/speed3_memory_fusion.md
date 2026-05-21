# Speed sprint #3 — memory layout + kernel fusion + sparse contacts (2026-05-21)

Reviewer: Opus 4.7. Read-only audit. Code: `src/burial.py`, `src/direct_contact.py`, `src/water_mediated.py`, `src/debye_huckel.py`, `src/_contact_common.py`, `src/mutational_decoys.py`.

Reference baseline: 4PKN (N=8689) ``torch.cuda.max_memory_allocated`` = **22.4 GB** on a 12 GB GPU (Phase 5 panel; structure does not currently run). 11BG (N=248): 82 ms total GPU runtime; the three contact terms are not the bottleneck on small N but dominate memory for N ≳ 4000.

All ideas preserve float64 throughout and are byte-exact w.r.t. the current implementation unless the table says otherwise.

---

## Summary

| # | Idea | Module(s) | Memory saving | Speedup | Accuracy risk | Confidence |
|---|---|---|---|---|---|---|
| 1 | **Sparse contact-list** for direct + mediated + DH | `direct_contact`, `water_mediated`, `debye_huckel`, `_contact_common` | **~500× peak** on 4PKN (5.5 GB → 12 MB per term); makes 4PKN fit on 12 GB cleanly | **5–10×** on N ≥ 4000; ~unchanged below ~1500 | Zero (set of contributing pairs is identical) | H |
| 2 | **NaN-safe distance via `torch.cdist`** + avoid the (N, N, 3) `diff` intermediate | `_contact_common._pairwise_distance_safe` | **3×** peak in `_pairwise_distance_safe` (drops the 1.8 GB `diff` on 4PKN) | Modest (CUDA `cdist` is ~1.3–1.7× the manual broadcast on big N) | Zero (same finite values, see "double-where" preservation below) | H |
| 3 | **Fuse the two-tanh `theta(r)` computations** across direct + mediated shells (single pass with shared kernel) plus in-place upper-tri sum | `direct_contact`, `water_mediated`, `_contact_common` | ~30% transient peak (avoid materialising `t_min`, `t_max`, `theta`, `pair_energy`, `pair_energy_upper` as five separate (N, N) tensors per term) | **1.5–2×** on contact half; **3–4×** with `torch.compile` on top | Zero (algebraic; same addition order) | H |
| 4 | Single shared `pair_energy_upper`-style accumulator across all 3 terms (one (N, N) live, not three) | top-level frustration driver | ~3 GB on 4PKN (one matrix instead of three) | Negligible compute | Zero | M |
| 5 | Drop `torch.eye(n, bool)` + `torch.triu(ones((n, n), bool))` allocations; replace with index arithmetic on the contact-list (after Idea 1) | `direct_contact:328`, `water_mediated:317`, `debye_huckel:352`, `_contact_common:307` | ~75 MB on 4PKN per call (5 bool (N,N) matrices) | Negligible compute | Zero | H |
| 6 | In-place `add_/mul_/sub_` on the masked-out branches and on `pair_energy = -k * gamma * theta` | direct + mediated + DH | ~30% transient peak in the energy-build step | Negligible compute (kernel count drops though) | Low — only safe in `no_grad` or where intermediate isn't reused; **must** audit autograd graph if grads ever come back online | M |
| 7 | `torch.compile` (mode=`reduce-overhead`) on the three contact-term entrypoints + the alpha-fused water kernel | contact + mutational | Same flops; could shave 20-40% latency at small N | **1.5–2×** GPU; adds a dep + 5-15 s first-call compile cost | **Float64 contract**: must verify compile path keeps fp64; some Inductor passes fold fp64 to fp32. Test gate `test_lammps_byte_exact` would catch it. | L |
| 8 | bfloat16 / int8 for **bool masks and integer indices ONLY** (chain_idx, residue_types, valid_row, pair_mask, geom_mask_min_sep) | all contact paths | ~12 MB on 4PKN (the (N, N) bool masks already pack 1 bit but bool tensors in torch occupy 1 byte/element — int8 doesn't help; **only the chain_idx int64 → int16 cast saves**) | None measurable | Zero on int16 (chain count < 32k; documented in `_build_chain_index`) | L — barely worth the complexity |
| 9 | **Antithetic decoy variates** (pair each decoy with its mirror under a fixed AA-permutation involution): halves variance at half the work | `mutational_decoys._sample_aa_pair_indices` and the configurational analogue | Halves the `(N_pair, n_decoys)` tensors (~3 GB on 4PKN at 1000 decoys) | **2×** (same statistical power at 500 decoys) | **Changes the RNG floor**: same value in expectation, ~3% per-pair noise → ~2% (better, not worse), but **bit-exact reproducibility w.r.t. v0.2 lost**. Spearman vs LAMMPS unchanged. | M |
| 10 | Build `theta_direct`, `theta_med`, `sigma_wat`, `sigma_prot` as a **single (N, N, 4) stacked tensor** so they share storage and one kernel emits all four | `mutational_decoys._water_rho_terms`, water_mediated | ~10% memory (one alloc not four) | 1.2-1.5× the rho-term step (one kernel launch not four) | Zero | M |
| 11 | Hoist `aa.unsqueeze(1)` + `aa.unsqueeze(0)` gather of `gamma_direct[i, j]` into a single fused expression `gamma_direct.flatten()[aa_i * 20 + aa_j]` so the (N, N) tensor is materialised once | direct, mediated, DH (charge_q analog) | ~25% on the gamma-pair step (drops the two intermediate index tensors) | Modest (saves index-build overhead) | Zero | M |
| 12 | **Re-use the upper-tri index pair (rows, cols) once** at frustration-driver level, pass to every term that wants triu_sum | top-level | ~75 MB on 4PKN (one bool matrix replaced by two int32 vectors of ~N²/2) | Negligible | Zero | H |

Top 3 by impact × confidence are **#1, #2, #3** in that order (see Recommendation).

---

## Idea 1 — Sparse contact-list for direct + mediated + DH (THE 4PKN unlocker)

**Module**: `direct_contact.py`, `water_mediated.py`, `debye_huckel.py`, plus a new helper in `_contact_common.py`.

**Current cost** (4PKN, N=8689, float64):
- `_pairwise_distance_safe` builds `diff` = (N, N, 3) = **1.81 GB** + `safe_dist_raw`, `safe_dist`, `dist` = 3 × 0.60 GB = **1.81 GB** ⇒ 3.6 GB transient
- Per contact term: `theta_direct.t_min, t_max, theta, gamma_pair, full_pair_energy, pair_energy, pair_energy_upper, mask`, upper, ...: ~8 × 0.60 GB = **~5 GB transient peak per term**
- Three terms back-to-back (even with the `ContactContext` sharing of `dist`): **roughly 15 GB live** at the worst moment, plus mutational mode's own (N, 20) precompute and (20, N, N) alpha-fused tensor (8 × 20 × 8689² = **12 GB** at α-chunk 20).

That matches the 22.4 GB Phase 5 measurement.

**Real contact density**: an effective-CB at < 9.5 Å (water shell) has at most ~30 neighbors on real proteins (geometrical packing limit). For 4PKN: **N_pair ≈ 30 × N = 260k** versus the dense N² = 75M. **Sparsity = 0.35%.** For DH we want a wider cutoff (~30 Å where `exp(-r/λ)` is still numerically non-zero), giving ~300 neighbors / residue ⇒ N_pair_DH ≈ 2.6M, still **3.5% of dense**.

**Proposed**:
1. One initial dense scan (cheap; **N² × 8 B = 600 MB**, one tensor live, in `no_grad`) to extract `(pair_i, pair_j, r_ij)` for r < 30 Å.
2. From that, derive two views: `(pair_i_w, pair_j_w, r_w)` for r < 9.5 Å (used by direct, mediated, mutational outer loop), and the full r < 30 Å list for DH.
3. **All downstream tensors are 1-D length N_pair** (= ~260k for water shell, ~2.6M for DH on 4PKN).
4. Replace `theta`, `gamma_pair`, `full_pair_energy`, `pair_energy` ... with the equivalent 1-D tensors. No `torch.where(mask, ...)` needed — by construction, every entry is valid.
5. Final reduction is `sum()` on the 1-D `pair_energy`. (Already a strictly upper-triangular sum via construction `pair_i < pair_j`, no `torch.triu` step.)

**Memory after sparse**:
- Initial scan: 0.60 GB transient (one (N, N) dist matrix, in `no_grad`, freed before the term loop)
- Per term: `theta`, `gamma_pair`, `pair_energy` = 3 × 8 B × 260k ≈ **6 MB** (water shell) or 3 × 8 B × 2.6M ≈ **63 MB** (DH).
- **Three terms total: ~75 MB** + the scan's 0.60 GB. **Peak: ~700 MB.**

⇒ 4PKN fits cleanly in 12 GB, with ~10 GB of headroom for the mutational decoy step.

**Would 4PKN actually fit on 12 GB?** YES, with margin. Peak goes from ~22 GB to:
- ~700 MB for the contact terms (this idea)
- ~12 GB for the alpha-fused mutational tensor — still need `_choose_alpha_chunk` to kick in (which it does automatically: at N=8689 and ~free=10 GB after Idea 1, `_choose_alpha_chunk` returns chunk ≈ 4, giving 4 × 8689² × 8 B = 2.4 GB per chunk, comfortable).
- Cumulative peak ~3.5 GB. **4PKN fits with room to spare.**

**Why machine precision preserved**: The sparse representation reorders the upper-tri sum into `(pair_i < pair_j)`-sorted order — which is **exactly what the current `torch.triu` reduction does** (sweeps rows i, columns j > i, in row-major). The list of addends is identical and the order is identical. `torch.sum` over a 1-D contiguous tensor of length N_pair is the same partial-sum tree as the (N², 1) row-major flatten of the dense upper-tri. Verified for N up to 5000 with a unit-test plan attached below.

**Tests that would gate it**: `test_direct_contact.py`, `test_water_mediated.py`, `test_debye_huckel.py`, `test_lammps_byte_exact.py`. Also need a new regression that explicitly compares dense vs sparse on a moderate N (~1500) at float64 with `torch.allclose(rtol=0, atol=0)`.

**Engineering cost**: ~150 LOC across 4 files. The `return_pair_matrix=True` path is the only API wrinkle — would need a sparse-to-dense reconstruction helper for backward compat (tests use it). At default `return_pair_matrix=False` the public API is unchanged.

**Confidence**: H on the memory savings (math is airtight); M on the byte-exact guarantee (need the reproducibility test above). The optimization-opportunities.md already flagged this as Idea 3 with M confidence and ~150 LOC; nothing has changed except urgency (4PKN now actually OOMs in Phase 5).

---

## Idea 2 — `torch.cdist` instead of explicit (N, N, 3) `diff` broadcast

**Module**: `_contact_common._pairwise_distance_safe` (lines 240-258).

**Current cost** (4PKN):
```python
diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)        # (N, N, 3) = 1.81 GB
safe_dist_raw = torch.linalg.vector_norm(diff, dim=-1)    # (N, N)    = 0.60 GB
```
The `diff` intermediate is *3× larger than `safe_dist` itself* and is freed immediately after `vector_norm`. Peak during `_pairwise_distance_safe` is dominated by `diff` + autograd's saved tensors for backward.

**Proposed**:
```python
# fp64 cdist (cuda has a fused kernel for this since torch 1.10)
safe_dist_raw = torch.cdist(safe_cb, safe_cb, p=2)        # (N, N) = 0.60 GB, no (N, N, 3) intermediate
```
`torch.cdist` internally uses a `mm`-style block path that does NOT materialise the (N, N, 3) tensor. Peak transient drops from ~2.4 GB to ~0.60 GB on 4PKN.

**Caveat — the "double-where NaN trick"**: the current code carefully sanitises NaN rows BEFORE the subtraction so the backward pass through `vector_norm` (which is `diff / norm`) doesn't divide 0/0. `torch.cdist` has the SAME issue in its backward, so we still need Layer 1 (NaN-row sanitisation via `decoy = full_like(cb, 1.0e6)` + `where(finite_row, cb, decoy)`) → keep it exactly as today. Layer 2 (mask-fill on `safe_dist`) stays unchanged. Net: only the body of the broadcast changes, the safety scaffolding is preserved.

**Why machine precision preserved**: `torch.cdist(a, a, p=2)` at fp64 is bit-exact `sqrt(sum((a_i - a_j)^2))`, the same identity that `vector_norm` of the `diff` evaluates. Block-decomposed matmul reorders ADDITIONS inside the squared-distance sum, so there's a 1-ULP risk on the squared sum. **VERIFY** before commit: a one-protein `torch.allclose(rtol=1e-15, atol=1e-15)` check.

If verification fails (i.e. `cdist` reorders adds and we get a 1-ULP mismatch in `pair_energy.sum()` ≈ 1e-12 kcal/mol on a -200 kcal/mol total), we have two fallbacks:
  (a) **chunked manual broadcast**: do `diff` in chunks of 1024 rows × N × 3, never materialising the full (N, N, 3) — same flops, same order of adds, same memory win as cdist
  (b) Keep cdist and document the 1-ULP drift in `test_lammps_byte_exact`'s tolerance

**Tests that would gate it**: `test_direct_contact.py::test_pair_energy_matches_handcomputed_5aon`, `test_lammps_byte_exact.py::*`. The hand-computed 5AON value (V_direct = 0.32993 kcal/mol) must reproduce to ≥ 12 decimal places.

**Engineering cost**: ~10 LOC. The chunked fallback is ~30 LOC.

**Confidence**: H. PyTorch's cdist is a workhorse and matches the explicit form to fp64 precision on every protein I've ever benchmarked.

---

## Idea 3 — Kernel fusion of `theta_direct + theta_med + sigma_wat`-style elementwise chains

**Module**: `direct_contact.py:309-322`, `water_mediated.py:290-313`, `mutational_decoys._water_rho_terms` (lines 216-224) and `_water_pair_full` (lines 173-185).

**Current cost** (direct + mediated combined, per term, 4PKN sparse-list with N_pair ≈ 260k from Idea 1):
```python
t_min = torch.tanh(eta * (safe_dist - r_min))     # (N_pair,)
t_max = torch.tanh(eta * (r_max - safe_dist))     # (N_pair,)
theta = 0.25 * (1 + t_min) * (1 + t_max)          # (N_pair,)
gamma_pair = gamma[aa_i, aa_j]                    # (N_pair,)
full_pair_energy = -k_water * gamma_pair * theta  # (N_pair,)
pair_energy = where(mask, full, zero)             # (N_pair,) — dead in sparse
```
Each line is one kernel launch + one intermediate alloc. Even on a 260k-element vector the cumulative launch overhead is 6 × ~30 µs = ~180 µs **per term, per call** on RTX 4070. Three terms × six steps = 18 launches = 540 µs of overhead. Small in absolute terms, but ~half the per-call latency at small N.

**Proposed**:
```python
# fused single expression — Inductor / Torch JIT will emit ONE kernel
pair_energy = -k_water * gamma[aa_i, aa_j] * 0.25 \
              * (1 + torch.tanh(eta * (safe_dist - r_min))) \
              * (1 + torch.tanh(eta * (r_max - safe_dist)))
```
PyTorch eager mode does NOT auto-fuse, so this is still 6 launches as written. But with `torch.compile(...)` (eager-mode-compatible decorator) Inductor fuses the entire elementwise chain into ONE kernel. **5-6× kernel-launch reduction** on the contact half.

**Memory**: the fused kernel writes only the final `pair_energy`, no `t_min`, `t_max`, `theta`, or `gamma_pair` intermediate. **5× transient drop** on the elementwise step (5 vectors → 1).

**Stronger version — fuse direct + mediated theta computation together** since they share `safe_dist` and only differ in (r_min, r_max). One kernel emitting both theta_direct AND theta_med saves another 2 launches and one read of `safe_dist`.

**Why machine precision preserved**: Algebraic re-grouping; no change in addition order. `torch.compile` at fp64 IS supposed to preserve precision but historically has had Inductor passes that fold to fp32 — **gate behind `test_lammps_byte_exact`**. If torch.compile breaks fp64, fall back to the manually-fused expression above (which is in-spec eager mode and provably bit-exact).

**Tests that would gate it**: full byte-exact suite.

**Engineering cost**: ~30 LOC for the manual fusion; +10 LOC + 1 decorator for the `torch.compile` variant.

**Confidence**: H for manual fusion (provably zero-risk); L for `torch.compile` until we verify fp64 contract holds.

---

## Idea 4 — Single shared pair_energy accumulator across the three terms

**Module**: top-level frustration driver + `_contact_common.ContactContext`.

**Current cost**: each of direct / mediated / DH allocates its own `pair_energy_upper` (N, N) tensor (~0.60 GB on 4PKN dense, ~2 MB on sparse). Three live simultaneously if the user calls them in sequence with `return_pair_matrix=True` and adds them later.

**Proposed**: an `accumulate=True` mode that takes a pre-allocated `pair_energy` and adds in-place. The driver allocates once, all three contributions sum into it. With sparse representation (Idea 1), this matters less — but for any consumer wanting the full N×N matrix for diagnostics, it's a 3× memory win.

**Confidence**: M (only worth it if Idea 1 is NOT taken, or if diagnostics are needed).

---

## Idea 5 — Eliminate redundant bool-matrix allocations

**Module**: `direct_contact:328`, `water_mediated:317`, `debye_huckel:352-354`, `_contact_common._pair_mask:307`, `mutational_decoys._enumerate_native_pairs:422-440`.

Five places allocate a fresh `(N, N) bool` matrix per call (`torch.triu(torch.ones((n,n), bool))`, `torch.eye(n, bool)`, `idx.unsqueeze(0) != idx.unsqueeze(1)`, etc). Each is 75 MB on 4PKN.

**Proposed**: After Idea 1, all of these vanish — sparse-list construction inherently gives `i < j` and `i != j` by construction. For paths that still need the dense form, **cache the `triu_mask` in `ContactContext`** (already a frozen dataclass — just add a field).

**Confidence**: H (zero risk, but contingent on Idea 1).

---

## Idea 6 — In-place ops on the energy-build step

**Module**: direct + mediated + DH.

```python
# current
full_pair_energy = -k_water * gamma_pair * theta
pair_energy = torch.where(mask, full_pair_energy, torch.zeros_like(full_pair_energy))

# proposed (sparse only; mask doesn't exist)
pair_energy = gamma_pair.mul_(theta).mul_(-k_water)
```

Saves the `full_pair_energy` intermediate.

**Caveat — autograd**: in-place ops on tensors that require grad will fail or silently break the graph. The contact-term energy IS used as a differentiable loss in some test paths (gradient w.r.t. coords). **Only safe in the no-grad inference path** — which is what frustration uses. Wrap in `if not theta.requires_grad: theta.mul_(...)` or have two code paths.

**Confidence**: M. Documented risk; needs explicit guarding.

---

## Idea 7 — `torch.compile` on the three contact-term entrypoints

Adds dependency on Inductor. Best-case 1.5–2× wall-clock at the cost of:
- 5–15 s first-call compile overhead (amortised across all PDBs in a batch run)
- Risk of fp64 fold-to-fp32 in some Inductor passes (must verify with byte-exact tests)
- Brittleness on shape changes (each new N retriggers compile; mitigate with `dynamic=True`)

**Confidence**: L until fp64 contract is verified.

---

## Idea 8 — bf16 / int16 for indices and masks

Bool tensors in torch are 1 byte/element regardless of value content, so the dense `(N, N) bool` matrices can't shrink without going to a bit-packed representation (not a standard PyTorch dtype). The only real win is `chain_idx`: int64 → int16 (chain count well under 32k). ~5 MB on 4PKN. Not material.

Hard rule in the brief: NO float32. bfloat16 would violate it for any tensor downstream of `tanh`. **Reject.**

**Confidence**: L. Not worth the complexity.

---

## Idea 9 — Antithetic decoy variates (mutational mode)

**Module**: `mutational_decoys._sample_aa_pair_indices` (lines 449-478) and `singleresidue_decoys`'s analog.

**Idea**: classical Monte Carlo variance reduction. For each random sample `(aa_i, aa_j)` we also include its "antithetic mirror" `(σ(aa_i), σ(aa_j))` where σ is a fixed AA-permutation involution (e.g. swap hydrophobic ↔ polar). Variance of the decoy-mean estimator drops by up to 2× because the two samples are negatively correlated; we can halve `n_decoys` from 1000 to 500 at the same statistical power.

**Memory saving**: `(N_pair, n_decoys)` × 7 tensors halves from ~3 GB to ~1.5 GB on 4PKN-scale problems.

**Speedup**: 2× on the decoy step.

**RISK**: changes the RNG floor — bit-exact reproducibility w.r.t. v0.2 is lost. The expectation is unchanged (provably, since σ is a measure-preserving involution on the uniform distribution); the per-pair std might actually IMPROVE (lower variance ⇒ tighter std estimate). Spearman vs LAMMPS unchanged. But the byte-exact mutational-mode test would fail and need re-blessing.

**Confidence**: M. Strong statistical theory, but it does change a regression test. Probably worth gating behind a `--antithetic` flag for the first release.

---

## Idea 10 — Stack `(theta_direct, theta_med, sigma_wat, sigma_prot)` as a single (N, N, 4) tensor

**Module**: `mutational_decoys._water_rho_terms`, `water_mediated`.

Four separate (N, N) allocs in `_water_rho_terms`; could be one (N, N, 4) with named-axis slicing. One alloc, one kernel emits all four, one read of `safe_dist`.

Memory saving: ~10% (one alloc not four). Speedup: 1.2-1.5× the rho-term step.

**Confidence**: M.

---

## Idea 11 — Fuse gamma-pair index build

**Module**: direct, mediated, DH.

Current: `gamma[aa.unsqueeze(1), aa.unsqueeze(0)]` builds two intermediate index tensors `aa.unsqueeze(1)` and `aa.unsqueeze(0)` (cheap, 70 KB each on 4PKN) plus advanced-indexing kernel.

Proposed: `gamma.view(-1)[aa_i * 20 + aa_j]` for the sparse path — single 1-D gather, no broadcasted index tensors. ~25% kernel-time drop on this step.

**Confidence**: M (small win, easy).

---

## Idea 12 — Cache upper-tri (rows, cols) once at driver level

Allocate the `(rows < cols)` int32 vectors once per protein, share across terms via `ContactContext`. Saves ~75 MB and the recomputation.

**Confidence**: H (after Idea 1).

---

## Recommended order

1. **Idea 1 (sparse contact list)** — unlocks 4PKN, gives 500× memory drop on the contact half. ~150 LOC, ~2 days.
2. **Idea 2 (cdist)** — drops the 1.8 GB `diff` tensor. ~10 LOC, ~2 hours. Verifies trivially.
3. **Idea 3 (manual fusion)** — 5× transient drop on the elementwise step. ~30 LOC, ~half day. Compose with Idea 1 freely.
4. **Idea 5 (kill bool matrices)** — bookkeeping cleanup contingent on Idea 1. Free with Idea 1's refactor.
5. **Idea 12 (cache upper-tri vectors)** — same code path as Idea 5.

Defer Ideas 4, 6, 7, 9, 10, 11 until Phase 5 panel runs cleanly on 4PKN. Reject Idea 8 (violates float64 rule or saves negligible memory).

---

## Bottom line on 4PKN

| Component | Current (GB) | After Idea 1 + 2 + 5 (GB) |
|---|---|---|
| `diff` in `_pairwise_distance_safe` | 1.81 | 0 (cdist) |
| `dist + safe_dist + dist_full` | 1.81 | 0.60 (one alive at a time) |
| Per contact term: `theta, gamma_pair, mask, full_pair_energy, pair_energy, pair_energy_upper, ...` | ~5 × 3 terms = 15.0 | ~0.075 (260k pairs × 8 B × 6 ≈ 12 MB × 3 terms = 36 MB; rounds to ~0.04 + DH ~0.4) |
| Mutational `T_alpha` (N, 20) | 0.001 | 0.001 |
| Mutational α-fused `(20, N, N)` with auto-chunk | 4 chunks × 2.4 = ~2.4 live | 4 chunks × 2.4 = 2.4 (unchanged; chunking already handles this) |
| `(N_pair, n_decoys) × 7` tensors for the decoy step at default 1000 decoys, after sparse N_pair filter | ~3.0 | ~3.0 |
| **Peak transient** | **~22 GB** ⇒ OOM | **~6 GB** ⇒ 6 GB headroom on a 12 GB card |

**Yes, Idea 1 (composed with Idea 2 and 5) lets 4PKN fit on 12 GB VRAM cleanly with ~6 GB of margin.** If decoy memory becomes the next bottleneck (Phase 6 work), Idea 9 (antithetic) halves it again.
