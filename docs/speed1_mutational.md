# SPEED-1: Mutational mode speedup ideas

Read-only audit of `src/mutational_decoys.py` (~982 LOC post Opt-Sprint).
Hard rule: NO proposed change may alter numerical output. Float64
throughout, no new deps (no Triton/JIT/custom kernels). Spearman gate
must stay ≥ 0.9986, and `max |ΔFI|` must equal exactly `0.0`.

Baseline reference (RTX 4070 best-of-10, Opt-Sprint head):

| PDB  | N    | mut GPU (ms) | bottleneck (Phase 5 §M2/footnote) |
|---|---|---|---|
| 5AON | 49   | 19.0 | sampler + per-pair U  |
| 11BG | 248  | 50.4 | per-pair U + (1517,1000) reductions |
| 1O3S | 200  | ~45  | per-pair U |
| 3F9M | 451  | 94.2 | per-pair U + reductions |

A100 estimate at 50-150 ms for 3F9M (4070 FP64-gimped); the bottleneck
ranking is the same on A100 because all three remaining hot spots are
launch-overhead/algorithmic, not throughput-bound on FP64 ALUs.

## Summary

| # | Idea | Est. speedup (11BG mut GPU) | LOC | Accuracy risk | Confidence |
|---|---|---|---|---|---|
| 1 | Replace `_per_pair_U` three `_water_pair_full` calls with **one** stacked-batch call (3× pair-term/U_iSlot_kj/U_jSlot_ki fused) | 1.10-1.25× | ~25 | **none — same float64 ops, just stacked along an extra axis** | high |
| 2 | Sampler off the critical path: pre-sample `(2, N_pair, n_decoys)` once on **GPU** with `torch.Generator(device='cuda')`, drop the CPU↔GPU index round-trip | 1.05-1.15× (≈ 10 ms saved on 11BG) | ~15 | **NONE provided sampler stays seeded the same way and we don't change the RNG sequence**. If the RNG sequence changes, decoy energies move by O(σ_decoy). Hard gate: must keep the CPU `torch.Generator(seed)` path as the default and route via a private kwarg; only adopt GPU RNG if a regenerated reference panel reproduces. **In its strict form (CPU sampler, async host→device copy with `pin_memory()` + `non_blocking=True`) it preserves the sequence exactly.** Drop to that strict form before claiming it. | high (strict) / med (full GPU RNG variant) |
| 3 | Build U-tensor once via the **fused (20, N, N) cube** that already exists in `_precompute_T_alpha`. Pair-term and U_iSlot_kj/U_jSlot_ki are all gathers from the same cube. | 1.5-2.0× (and removes 3 of the 4 remaining hot kernels) | ~40 | **none — identical math, just memoized** | med-high |
| 4 | Reduction layout: store `E_decoy` row-contiguous, use `torch.var(..., unbiased=False, dim=1).sqrt_()` and skip the temporary | <1.05× | ~5 | none | high |
| 5 | Burial-per-decoy memoization keyed on `aa_dec` cube | 1.05-1.10× | ~20 | none | high |
| 6 | **Idea 3 + Idea 5 combined** (single (20, ...) pass for water + burial cubes) | 1.6-2.1× (replaces the per-pair work entirely with two gathers and one add) | ~55 | none | med |

Realistic stack (Ideas 1+2+3+4+5): **2.0-2.6× on 11BG mut GPU → ~20-25
ms**, close to the 30 ms stretch goal Opt-Sprint missed. Ideas 1/4/5
are mechanical, low-risk; Idea 2 needs the strict CPU-sampler form;
Idea 3 needs careful peak-memory accounting on N>1800.

---

## Idea 1: Stack the three `_water_pair_full` calls in `_per_pair_U`

### Where
`_per_pair_U` (lines 582-706). Today it calls `_water_pair_full` three
times with shape `(N_pair, n_decoys)` inputs, each time recomputing
`theta_direct`, `theta_med`, `sigma_wat`, `sigma_prot` — which are
**identical** across all three calls (same `r_ij`, same `rho_i, rho_j`
per pair).

### Math
For each pair `(i, j)` and decoy `d`, all three terms compute
`-k_water · (γ_dir · θ_dir + (σ_prot·γ_mp + σ_wat·γ_mw) · θ_med)`
where `θ_dir, θ_med, σ_prot, σ_wat` are **identity-independent** (same
`(r_ij, ρ_i, ρ_j)` triple). Only the `(α_i, α_j)`-dependent γ lookup
changes:

- `U_iSlot_kj` : γ at `(α_i_dec[d], aa_j_native)`
- `U_jSlot_ki` : γ at `(α_j_dec[d], aa_i_native)` with rho swapped — but
  the `r, ρ` ingredients are **symmetric under (i↔j)** so `θ_*` are the
  same, `σ_wat = σ_wat(ρ_i, ρ_j)` is symmetric (the `0.25·(1-tanh)·(1-tanh)`
  factor is symmetric in its two args).
- `pair_term` : γ at `(α_i_dec[d], α_j_dec[d])`.

Stack the three γ-gathers as a single `(3, 20, 20)`→`(3, N_pair, n_decoys)`
fancy-index and run one fused water-pair op.

### Algebraic identity
The result is bit-identical because:
1. `θ_dir, θ_med, σ_wat, σ_prot` are computed **once** instead of three
   times. Each computation uses the same `tanh`/`switch` op tree, so we
   evaluate the same float64 value to the last bit. Hoisting an
   expression out of repeated callsites doesn't change the float result.
2. The γ-gathers are three independent lookups; stacking them along an
   extra leading axis preserves each row's value exactly.
3. The final multiply-add is per-element — pulling three separate
   `-k_water·(γ_dir·θ_dir + σ_med_γ·θ_med)` evaluations into one
   broadcast-shaped `(3, N_pair, n_decoys)` expression evaluates the
   **same arithmetic** at each (l, p, d) index. Float64 addition order
   per-element is unchanged.

### Why machine precision is preserved
No reductions or accumulators introduced — purely a stack-the-broadcast
refactor. Per-element value identity guaranteed by IEEE-754
determinism of `+`/`*`/`tanh` (PyTorch CUDA elementwise ops are
deterministic for these primitives).

### Gates
- Numerical: `max |ΔE_decoy| == 0.0` exactly per element (assert in
  a unit test on 5AON + 11BG).
- Spearman: trivially ≥ 0.9986 (same numbers).
- Bench: 11BG mut GPU best-of-10 ≤ 46 ms (≥ 1.10× over current 50.4 ms).

### Cost
~25 LOC. Replaces three calls + their per-call `_water_pair_full`
prologue with one shared prologue + one fused gamma-stack op.

### Confidence: HIGH

---

## Idea 2: Off-load the AA sampler off the critical path

### Where
`_sample_aa_pair_indices` (lines 449-478). Today: `torch.randint`
on CPU → `aa.cpu()[idx]` (a 1.5M-element gather on CPU) →
`.to(device=device)` (a synchronous H2D copy).

### Two variants

**2a (strict, preserves RNG sequence exactly)**

Keep the CPU `torch.Generator(seed)`. Change three things:
1. Compute both `randint`s in one call (already done — but interleave
   them so they overlap with the gamma-table caching).
2. Issue the H2D copy as **`non_blocking=True`** with a pre-pinned
   buffer. While the copy runs, the CPU can do gamma-table caching and
   `_enumerate_native_pairs` finalisation.
3. Replace `aa_cpu[idx_i]` (CPU gather of 1.5M elements) with a GPU
   gather: copy the tiny `aa` tensor (N int64, ~2 KB) to device once,
   then index on device. This drops the 1.5M-element CPU gather and
   the corresponding H2D copy.

   Worked example, 11BG: `idx_i` is `(1517, 1000)` int64 = 12.1 MB.
   Today we pay one CPU-side gather (~1.5M loads from `aa_cpu`) then
   12.1 MB H2D. New: 2 KB `aa.to(device)` once, then 12.1 MB H2D of
   `idx_i` only, then on-device gather. The on-device gather is
   ~free (already done implicitly in `T_alpha.gather(1, aa_i_dec)`).
   Net: **same H2D bandwidth, but skips the CPU gather** (~3-5 ms on
   11BG, more on 3F9M).

**2b (full GPU RNG — accuracy risk if sequence diverges)**

Use `torch.Generator(device='cuda')`. **REJECTED unless we accept
re-establishing the reference panel** — `torch.randint` on CUDA uses a
philox4x32-10 PRNG with a different state evolution than the CPU
mersenne-twister; same seed produces different sequences. Every
downstream decoy energy changes by O(σ_decoy), Spearman would still be
~0.9985 but `max |ΔFI| ≠ 0`. **HARD-RULE VIOLATION.** Do not adopt.

### Algebraic identity (variant 2a)
The sampler still uses the same `torch.Generator("cpu").manual_seed(seed)`
to draw the same `int64` index tensors. The only change is **where the
gather happens** (GPU instead of CPU), which is an integer gather
operation — for int64 indexing into an int64 buffer, the result is
bit-identical regardless of device.

### Why machine precision is preserved (variant 2a)
The output of the sampler is integer indices into `aa`. Integer gather
is exact. No FP arithmetic involved.

### Gates (variant 2a)
- `aa_i_dec` and `aa_j_dec` must compare `torch.equal()` against the
  current CPU-gathered output on 5AON + 11BG + 3F9M.
- All downstream `E_decoy` values bit-identical.
- Bench: ≥ 5 ms saved on 11BG (≥ 1.10×).

### Cost
~15 LOC. Single function rewrite; no callers change.

### Confidence: HIGH (variant 2a). Variant 2b is REJECTED.

---

## Idea 3: Memoize per-pair U via the existing (20, N, N) fused cube

### Where
The fused water cube `w_all` built at `_water_per_alpha_fused`
(lines 547-559) inside `_precompute_T_alpha` already contains
**every** value needed for `_per_pair_U`:

- `w_all[α, i, j] = water_pair(r_ij, α, aa_j_native, ρ_i, ρ_j) · 1{r<cutoff, i≠j}`
- We currently reduce it as `T_alpha = w_all.sum(dim=2)` and then
  **discard** `w_all` (line 560).

But `_per_pair_U` then recomputes:
- `U_iSlot_kj[p, d] = w_all[α_i_dec[p,d], pair_i[p], pair_j[p]]` ✓ exactly
  what's in the cube (mask already applied).
- `U_jSlot_ki[p, d] = w_all[α_j_dec[p,d], pair_j[p], pair_i[p]]` ✓ symmetric
  partner.
- `pair_term[p, d] = water_pair(r_ij_pair[p], α_i_dec[p,d], α_j_dec[p,d],
  ρ_i, ρ_j)` — **NOT** in `w_all` because the cube fixes `α_col = aa_j_native`,
  not arbitrary `α_col`. However, build a sibling cube
  `w_all2[α_row, α_col, p]` of shape `(20, 20, N_pair)` indexed by the
  pair endpoints — this is the cross-residue identity:
  `pair_term[p, d] = w_all2[α_i_dec[p,d], α_j_dec[p,d], p]`.
  Memory: `20 × 20 × 1517 × 8 B = 4.86 MB` on 11BG; `4.86 × (3F9M N_pair /
  11BG N_pair) ≈ 11 MB` on 3F9M. Negligible.

### Math
The fused cube already exists in VRAM during `_precompute_T_alpha`.
Currently it's reduced and freed. Keep it (rename to `W_full_cube`,
return alongside `T_alpha`), then in `_per_pair_U`:

```
U_iSlot_kj = W_full_cube[aa_i_dec, pair_i[:,None], pair_j[:,None]]   # gather
U_jSlot_ki = W_full_cube[aa_j_dec, pair_j[:,None], pair_i[:,None]]   # gather
pair_term  = pair_cube[aa_i_dec, aa_j_dec, arange(N_pair)[:,None]]   # gather
```

Three gathers replace three `_water_pair_full` calls (~12 elementwise
ops each at `(N_pair, n_decoys)` shape).

### Algebraic identity
Indexed gather from a precomputed cube returns the **exact same
float64** as recomputing the formula — both paths evaluate the same
expression at the same `(α, i, j)` indices. The cube is built with the
same `_water_per_alpha_fused` op tree currently used for T_alpha;
recomputing it on-the-fly in `_per_pair_U` evaluates the same op tree.
Per-element float identity guaranteed.

For `pair_cube`: this is a `(20, 20, N_pair)` tensor built by sweeping
both `α_i` and `α_j` over `(N_pair,)` worth of `(r_ij, ρ_i, ρ_j)`
triples — total `400 × N_pair` elementwise evals vs the current
`N_pair × n_decoys = 1.5M` evals on 11BG. With 1000 decoys per pair
this is a 2.5× compute reduction on the pair-term branch alone, **and**
each evaluation produces the same float as the current path because
the same `_water_pair_full` formula is evaluated with the same inputs.

### Why machine precision is preserved
Per-element identity of gather-vs-recompute is guaranteed when both
sides evaluate the same arithmetic with the same inputs at the same
precision. No reduction order changes (the only reduction was
`T_alpha = w_all.sum(dim=2)`, which is unchanged).

### Peak-memory accounting (gate)
- `w_all` cube: `20 × N² × 8 B` = 9.84 MB at 11BG (N=248), 32.5 MB at
  1O3S (N=200, lower), 65.2 MB at 3F9M (N=451), **604 MB at 4PKN
  (N=8689)**. The existing `_choose_alpha_chunk` already handles 4PKN;
  if we hold `w_all` past `T_alpha` we lose the chunking benefit. **Gate:**
  α-chunked path must materialise the cube once, gather from it, then
  free — i.e. fold `_per_pair_U` into the α-chunk loop body (per-chunk
  partial U updates). LOC cost is in the +40 estimate.
- `pair_cube`: `(20, 20, N_pair)` worst-case at 3F9M `N_pair ≈ 3349` →
  10.7 MB. Trivial.

### Gates
- Bit-identical `U_iSlot_kj`, `U_jSlot_ki`, `pair_term` vs current
  implementation on 5AON + 11BG + 3F9M (assert `torch.equal`).
- Peak VRAM unchanged on 4PKN (must stay under 12 GB).
- Bench: 11BG mut GPU best-of-10 ≤ 30 ms (≥ 1.65×).

### Cost
~40 LOC. Refactors `_precompute_T_alpha` to return the cube alongside
`T_alpha`, modifies `_per_pair_U` to gather, adds the small `pair_cube`
builder.

### Confidence: MEDIUM-HIGH. The peak-memory interaction with
α-chunking is the only thing that could derail this; on N<2000 PDBs
the cube fits trivially.

---

## Idea 4: Reduction layout — combine `mean`/`std` into one pass

### Where
Lines 961-962:
```
decoy_mean = E_decoy.mean(dim=1)
decoy_std = E_decoy.std(dim=1, unbiased=False)
```

Currently issues two reductions over `E_decoy.shape == (N_pair, n_decoys)`.
PyTorch's `std` recomputes the mean internally.

### Math
Welford-merged single pass via `torch.var_mean(E_decoy, dim=1, unbiased=False)`
which returns both in one kernel. Mathematically `var_mean` is defined
to produce **exactly the same** values as `mean` and `var` would
produce separately (PyTorch contract). The `sqrt` for std is one
elementwise op.

### Algebraic identity
`torch.var_mean(unbiased=False)` is documented to produce the same
result as `(mean, var)` computed separately, modulo a single SVD/Welford
implementation choice. On CUDA both `var` and `var_mean` use the same
two-pass algorithm by default, so the result is bit-identical. Verify
on 5AON+11BG+3F9M.

### Caveat
If `var_mean` internally uses a different reduction (single-pass vs
two-pass), the last-ULP value could change. Empirical check needed
before commit — but this is one of the cheapest gates to run.

### Gates
- `torch.equal(decoy_mean_new, decoy_mean_old)` and
  `torch.equal(decoy_std_new, decoy_std_old)` on 5AON+11BG+3F9M.
  If not bit-identical, **REJECT** (since `max |ΔFI| = 0` is the rule).
- Bench: <1.05× — this is a small win.

### Cost
~5 LOC.

### Confidence: HIGH that the speedup is small; MEDIUM that `var_mean`
matches `mean`+`std` bit-identically across PyTorch versions. The
fallback is "leave it alone".

---

## Idea 5: Memoize burial per (α, ρ_anchor) cube

### Where
Lines 943-957. `_burial_residue_energy(aa_i_dec, rho_i_b, ...)` evaluates
the burial formula at `(N_pair, n_decoys)` = (1517, 1000) per-pair
positions. But `aa_i_dec` has only **20 distinct values** and
`rho_i_b` is broadcast from `rho_i_p` which has only `N_pair = 1517`
distinct values (one ρ per anchor residue, replicated across decoys).

### Math
Build a `(N_pair, 20)` cube `B_i[p, α] = burial(α, ρ_i_p[p])` (and
similarly `B_j[p, α]`). Then:
```
burial_i_dec = B_i.gather(1, aa_i_dec)
burial_j_dec = B_j.gather(1, aa_j_dec)
```

Same trick as `T_alpha`. Compute time: `(N_pair, 20)` evals vs current
`(N_pair, n_decoys)` = 50× reduction on 11BG (1517·20 / 1517·1000).
Memory: `1517 × 20 × 8 B = 243 KB` on 11BG. Trivial.

Better still: each unique `(α, ρ)` is one of `20 × N` combinations.
Build `B[α, k] = burial(α, ρ_k)` once at shape `(20, N)` (already
implicitly in the residue-level summation), then
`B_i = B.T[pair_i]`, `B_j = B.T[pair_j]`. Even smaller setup cost.

### Algebraic identity
Burial energy is a closed-form function of `(α, ρ)` with no sums over
other indices. Memoizing and gathering returns the same float64 as
recomputing.

### Why machine precision is preserved
Burial uses `burial_switch` (a closed-form tanh-based sigmoid) and a
gamma-table lookup. No reductions. Per-element identity preserved by
gather-vs-recompute.

### Gates
- `torch.equal(burial_i_dec_new, burial_i_dec_old)` on
  5AON+11BG+3F9M.
- Bench: ~3-5 ms saved on 11BG (~1.05-1.10×).

### Cost
~20 LOC.

### Confidence: HIGH.

---

## Idea 6: Idea 3 + Idea 5 combined — eliminate the per-pair recompute branch entirely

After Ideas 3+5, `_per_pair_U` becomes three integer-gathers on cubes
that were already built (or are tiny). The only remaining elementwise
work in the decoy branch is the final `E_decoy = pair_term + cross_i +
cross_j + burial_i_dec + burial_j_dec` — a single fused add over
`(N_pair, n_decoys)`. That's ~10 ms on 11BG today, less if Idea 4
saves a fused-mul reduction.

Expected combined: **2.0-2.6× on 11BG mut GPU → 20-25 ms**.

LOC cost: bundle 3+5 carefully; total ~55 LOC and one new top-level
"cube context" passed from `_precompute_T_alpha` to `_per_pair_U`.

### Confidence: MEDIUM (compounding 3 and 5; the only risk is that
peak VRAM regresses on 4PKN. Gate that with `max_memory_allocated`
before/after).

---

## Cross-cutting opportunities considered

### Cross-residue (i, j, k) terms — structural insight
The cross-term mask is **spatial-only** (no seq-sep). Per the docstring
"the (i, k=i±1) contribution IS included". This means there is no
free-lunch sparse-pattern reduction: every (i, j) pair sees ~N_neighbors
contributing k's, and they are pre-summed into `T[i, α]` already.
**No further work to be saved here.** The `_precompute_T_alpha` step is
already the optimal O(N²·20) reduction.

### Batched matmul opportunities
The per-pair gamma lookup is fundamentally a gather, not a matmul.
Trying to turn it into a matmul (e.g., one-hot-encoded `α` × `γ`)
would replace ~12 element-wise ops per `(p, d)` with a 20-way
contraction — strictly slower and **not** bit-identical (matmul
reduces in a different order). REJECTED.

### GPU launch-overhead merging
After Ideas 1+3+5 the mutational forward has roughly these kernels:
1. NaN-safe pairwise distance (`_enumerate_native_pairs`)
2. `_water_rho_terms` (4 elementwise ops fused into ~2 kernels)
3. `_water_per_alpha_fused` (1 big kernel, returns cube)
4. T-reduce (`sum(dim=2)`)
5. Per-pair gather + add (1 kernel each, 3 gathers + 1 fused add)
6. Burial cube + 2 gathers
7. `var_mean` (1 kernel)
8. Sampler (1 H2D copy + 1 gather, async)

Total ~10 kernels for the decoy stage. Hard to shrink further without
custom kernels (banned by the hard rule).

---

## Rejected ideas

- **Pre-sample 20×20 unique-AA-pair grid once and gather** (instead of
  N_pair × n_decoys decoys). Rejected: the sampler is **per-pair**
  with independent draws; reusing samples across pairs would alter the
  PRNG sequence and shift `decoy_mean`/`decoy_std`. HARD-RULE
  VIOLATION.

- **Reduce n_decoys from 1000 to 500** for speed. Rejected: `decoy_std`
  has finite-sample noise that scales as 1/√n_decoys; halving n shifts
  the FI by ~3-5% per pair. HARD-RULE VIOLATION.

- **Use the empirical 20-AA composition vector to compute analytic
  decoy_mean and decoy_std** (instead of sampling). This is the
  "analytic mutational" idea — it would replace the (1517, 1000)
  reductions with closed-form expressions. Mathematically correct in
  the limit n_decoys → ∞, but our finite-sample reference panel
  (n_decoys=1000) is what the LAMMPS dump compares against. The
  analytic answer differs from the n_decoys=1000 sampler by ~1-3%,
  which **does** move FI values. HARD-RULE VIOLATION. Keep as a
  separate research direction (would need its own validation panel).

- **Replace `torch.where(mask, x, 0)` with `x * mask`** in pre-mask
  step at `_precompute_T_alpha:543-545`. Mathematically `x * 0 = 0`
  for all finite `x`, but if `x = inf` (the NaN-safe distance trick
  outputs `inf` for non-finite pairs) then `inf * 0 = NaN` whereas
  `torch.where(False, inf, 0) = 0`. Float identity NOT preserved on
  non-finite inputs. HARD-RULE VIOLATION at the boundary.

- **fp32 mixed precision for the water-pair tensor** (cast cube to
  fp32, gather and reduce in fp32, cast back). HARD-RULE VIOLATION
  (float64 throughout is required).

- **CUDA graphs to amortise launch overhead** across the ~10 kernels.
  Rejected by "no Triton/JIT/custom kernels" — CUDA graphs are
  capture-and-replay (not a custom kernel), but the typical
  implementation requires `torch.cuda.graph()` wrapping which is a
  graph compile step. The Phase 5 review noted single-run timings are
  the headline; graph capture would help repeat runs but not
  cold-cache, and the gate is `tests/...` correctness which graphs
  can disturb on first-call. PARK — revisit if the no-JIT rule is
  relaxed.

- **Stream-overlap sampler with `_precompute_T_alpha`** on separate
  CUDA streams. Possible, but PyTorch streams require explicit
  `torch.cuda.Stream()` plumbing; adds complexity for ~5 ms win.
  Idea 2a (async H2D + on-device gather) captures most of the
  available overlap without stream plumbing. PARK.
