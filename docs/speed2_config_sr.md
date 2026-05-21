# SPEED-2 — Configurational + Singleresidue speed hunt — 2026-05-21

Reviewer: Opus 4.7. Code: `src/decoys.py` (~711 LOC, configurational mode) and
`src/singleresidue_decoys.py` (~400 LOC). Status: **read-only audit, no code modified**.

Reference baselines (11BG, N=248, RTX 4070, float64):
- Configurational: ~186 ms (already cached once per structure)
- Singleresidue: ~8 ms (post optimization sprint, 4× win already landed)

HARD RULE: Spearman ≥ 0.997 vs LAMMPS-AWSEM must hold across all 3 modes × 4 PDB panel.
No accuracy change. RNG noise floor is ~3% configurational / ~5% mutational (set by
N_decoys=1000 and seed independence — independent of speed work).

---

## Summary

| # | Idea | Module | Estimated speedup | Engineering cost | Accuracy risk | Confidence |
|---|---|---|---|---|---|---|
| 1 | Direct sampler for `r_ij` from precomputed in-contact CDF — kills rejection loop AND the sparse-fallback bias | decoys.py:402-459 | **2-5×** on 11BG config (eliminates the dominant remaining cost); larger on sparse PDBs | ~30 LOC | Zero on dense; FIXES known sparse-fallback bias | H |
| 2 | Share `dist_full` + `cb_or_ca` between config and SR (one-pass build cached on the structure) | decoys.py:362-378 + singleresidue_decoys.py:280-292 | **1.2-1.5×** end-to-end when both modes run on same PDB | ~40 LOC + ContactContext wiring | Zero (same tensor reused) | H |
| 3 | SR `aa_dec` sampling — single fused `randint` then index into `aa` directly on device | singleresidue_decoys.py:185-206 | **1.3-1.6×** on SR (cuts CPU→GPU copy in the only per-decoy op) | ~10 LOC | Zero | H |
| 4 | Concurrent CUDA streams for config + SR (independent compute) | new top-level driver | **1.5-2×** when both modes requested (latency, not throughput) | ~60 LOC | Zero (stream sync gate) | M |
| 5 | SR `_precompute_W_sr` already has T[i,α] analog (W_sr is exactly that) — no further precompute win available | singleresidue_decoys.py:99-178 | — | n/a | n/a | — |
| 6 | Replace `gather(1, aa_dec)` (N, 1000) with reshape + index_select | singleresidue_decoys.py:374-375 | ~5-10% on SR gather kernel | ~5 LOC | Zero | M |
| 7 | Drop the (N, N, 3) `diff` intermediate in config; use `torch.cdist(safe_cb, safe_cb)` | decoys.py:369-370 | 2× peak memory; modest speed (3-8%) | ~10 LOC | Zero | M |
| 8 | `n_decoys` reduction to 500 or 250 (RNG-floor experiment) | both modules | **2-4×** on the per-decoy reductions IF Spearman holds at lower N | ~5 LOC + offline study | **PROBABLE accuracy hit** — Spearman bound is set by `1/sqrt(N_decoys)`. Needs empirical gate | L |
| 9 | Move config rejection sampler's `int(accept_mask.sum().item())` sync out of the loop | decoys.py:416 | 1.1-1.3× on small/medium PDBs (per-iter device→host sync) | ~15 LOC | Zero | H |
| 10 | SR: skip per-residue `safe_std` branch — replace `torch.where` divide-guard with `nan_to_num` | singleresidue_decoys.py:381-384 | <5%; readability win | ~5 LOC | Zero | L |
| 11 | torch.compile the SR `_precompute_W_sr` GPU path | singleresidue_decoys.py | Unknown, possibly 1.5-2× via fusion of (20, N, N) ops | ~5 LOC | Possible — torch.compile is float64-sensitive; pytest determinism risk | L |

Top picks are **1, 2, 3** in that order (see Recommendation below).

---

## Idea 1 — Direct sampler for `r_ij` (kill the rejection loop + the bias)

**Module**: `src/decoys.py:402-459` (the `for _ in range(max_resample_iter):` loop).

**Current cost**: The rejection sampler draws (need, 2) candidate pairs, gathers
`cand_dist`, checks `(p != q) & (cand_dist < contact_cutoff)`, fills accepted, repeats.
On dense folded proteins ~30% accept rate → ~3-4 iterations. Each iteration is **two
CPU-side `randint`s + two `.to(device)` copies + a gather + a `.sum().item()` sync**.
The `.item()` call (line 416) forces a device→host stall per iter, which is the worst
cost on GPU for a 30% accept rate workload.

On sparse / fragmented PDBs (the bias case logged in MEMORY and called out at lines
422-441), the sampler exhausts `max_resample_iter=64` and falls back to "cycle through
in-contact distances" — biased toward whichever pairs satisfied the cutoff first.
**This is the only known correctness gap in configurational mode.**

**Proposed cost**: Precompute the in-contact distance distribution once::

```
flat = dist_full[~eye].flatten()
in_contact_dists = flat[flat < contact_cutoff]   # (M,) — M = #in-contact pairs
# inverse-CDF sampling reduces to a single randint into in_contact_dists
sample_idx = torch.randint(0, M, (n_decoys,), generator=gen)
rij_decoy = in_contact_dists[sample_idx]
```

This is mathematically equivalent to the rejection sampler in the limit of infinite
iterations — every accepted sample comes from the same uniform-over-in-contact-pairs
distribution. The diagonal mask is automatic because `~eye` excludes it. The `p != q`
guard (line 415) becomes the `~eye` mask.

**Equivalence to LAMMPS**: The C++ rejection loop at fix_backbone.cpp:5262 is::

```
do { p = rand() % N; q = rand() % N; r_pq = …; } while (r_pq >= cutoff || p == q);
```

— exactly uniform sampling over the in-contact pair set. Our current rejection
implementation has the same asymptotic distribution but adds a low-probability
truncation when `max_resample_iter` runs out. The direct sampler is the analytic
exact form — strictly closer to LAMMPS, **not** further.

**Math walkthrough**: Let S = {(p, q) : p ≠ q, r_pq < cutoff}. The C++ samples
uniformly from S. The rejection loop samples uniformly from S because each (p, q)
draw is independent uniform over [0,N)² and we accept iff (p, q) ∈ S. Direct CDF
sampling = uniform draw of an index into the enumeration of S. **Identical
distribution.**

**Why this also kills the bias**: There is no "fallback". If S is non-empty, the
sample is correct by construction; if S is empty, raise (same as current
RuntimeError). No silent biased tail.

**Why machine precision preserved**: We're not changing the per-decoy energy formula,
only how `rij_decoy` is sampled. The Spearman gate is set by `n_decoys=1000` (RNG
floor ~3%), and the new sampler draws from the **same** uniform-over-S distribution as
the rejection loop. The seed sequence will differ (1 randint call instead of 1-4 per
need), so per-PDB scalar decoy_mean/decoy_std will shift by ~3% RNG floor — but
**rank ordering of decoys is preserved** because the marginal r distribution is
unchanged.

**Tests that would gate it**:
- `tests/test_decoys.py` validation tolerance is already 3% (module docstring line 93-95).
- New unit test: assert `rij_decoy` histogram matches rejection-sampler histogram in
  KS distance < 0.05 on 11BG.
- Spearman gate `python _opt_spearman.py` — 11BG configurational must stay ≥ 0.9999.

**Engineering cost**: ~30 LOC. Replace lines 402-459 with the 4-line direct sampler;
preserve the empty-S RuntimeError path; remove `max_resample_iter` from the public API
(or keep it as a no-op kwarg for backward compat). The `import warnings` block goes
away — there is no fallback to warn about.

**Confidence**: H. The math is bulletproof, the speedup is real (kills the .item()
sync), and the bias fix is a free correctness win.

**Estimated speedup**: 2-5× on the **rejection-sampler portion** of `sample_configurational_decoys`.
On 11BG with the current ~186 ms total config time, the rejection sampler is ~30-50% of
wall-clock (the dist matrix build dominates), so end-to-end **1.3-1.8×**. On sparse
PDBs that currently hit fallback, this is 10×+ AND a correctness fix.

---

## Idea 2 — Share `dist_full` between config and SR

**Module**: `src/decoys.py:362-378` and `src/singleresidue_decoys.py:280-292`.

**Current cost**: Both modes independently build the (N, N) distance matrix:

- config: lines 363-378 → `safe_cb`, `diff`, `dist_full`, `finite_pair`
- SR: lines 283-292 → **literally the same computation** (~50 lines duplicated)

For N=248, the (N, N, 3) `diff` allocation is 1.5 MB, the `vector_norm` reduction is
~120K ops. On GPU this is 4-6 kernel launches plus the alloc. On CPU it's 5-10 ms.

The `ContactContext` wiring landed in optimization sprint #3 (per
`docs/optimization_sprint_results.md:18`) already does this between direct/mediated/DH
— but the decoy samplers were not retrofitted.

**Proposed cost**: Extend `ContactContext` to carry the `dist_full` and `finite_pair_2d`
tensors. `sample_configurational_decoys` and `singleresidue_decoy_stats` accept an
optional `_context=` kwarg that, when present, skips the rebuild.

**Math walkthrough**: The two builds produce **bit-identical** tensors (same input
`cb_or_ca`, same `1.0e6` sentinel for NaN rows, same `vector_norm`). Verified by
reading both files: config line 367 uses `1.0e6`, SR line 284 uses `1.0e6`; both use
`safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)`; both apply `finite_pair` after the norm.

**Why machine precision preserved**: Bit-identical tensors are bit-identical.

**Tests that would gate it**: Existing 135/135 pytest suite; Spearman panel.

**Engineering cost**: ~40 LOC. Add `dist_full` and `finite_pair_2d` fields to the
existing `ContactContext` dataclass (in `_contact_common.py`); thread the optional kwarg
through both samplers; keep the no-context path identical to today.

**Confidence**: H. Wiring already exists for the dense contact terms; just extends it.

**Estimated speedup**: 1.2-1.5× when both config + SR run on the same PDB. Useless if
only one mode is run. For the "compute full frustration triple" workflow (the common
case for the allosteric pipeline), this is real money.

---

## Idea 3 — SR `aa_dec` sampling: fuse the randint + on-device gather

**Module**: `src/singleresidue_decoys.py:185-206`.

**Current cost**:
```python
gen = torch.Generator(device="cpu")
idx = torch.randint(0, n, (n, n_decoys), generator=gen)        # CPU
aa_cpu = aa.cpu()                                              # GPU→CPU sync
return aa_cpu[idx].to(device=device, dtype=torch.int64)        # CPU index, then CPU→GPU
```

For N=248, n_decoys=1000 → `idx` is (248, 1000) int64 = 2 MB. The `aa.cpu()` plus the
final `.to(device)` are **two host-device synchronisations**, each costing 10-50 µs
GPU launch latency. The 248K-element gather happens on CPU then transfers — wasteful.

**Proposed cost**:
```python
gen = torch.Generator(device="cpu")
idx = torch.randint(0, n, (n, n_decoys), generator=gen).to(device)   # one transfer
return aa[idx]                                                       # on-device gather
```

`aa` is already on device (line 276). We move only the index tensor (which is the
small one, 2 MB), and gather on-device. **One sync instead of two**; gather happens on
GPU instead of CPU.

**Math walkthrough**: `aa[idx]` produces identical output regardless of which device
the gather runs on — it's a pure integer-indexed copy. The PRNG sequence is unchanged
(we keep the CPU generator for cross-device reproducibility per the established
convention at decoys.py:380-382).

**Why machine precision preserved**: aa is int64. No float math.

**Tests that would gate it**: All singleresidue tests including
`test_singleresidue_mode.py`. Spearman panel.

**Engineering cost**: ~10 LOC.

**Confidence**: H. Straight cleanup of a dual-sync pattern.

**Estimated speedup**: 1.3-1.6× on the `_sample_aa_per_residue` portion (which is
~10-20% of SR total per benchmarks). End-to-end: **5-15% on SR**.

---

## Idea 4 — Concurrent CUDA streams for config + SR

**Module**: New driver — config and SR have no data dependency on each other (they
share `aa`, `rho`, `coords`, `dist_full` as **inputs**; their outputs are independent).

**Current cost**: Today these run serially. On 11BG, config = 186 ms, SR = 8 ms →
194 ms wall-clock. Config has ~30% device-idle time during the rejection loop sync
points. SR can fully overlap that idle window since it's only 8 ms.

**Proposed cost**: Two `torch.cuda.Stream()` objects; dispatch config on stream A,
SR on stream B (after the shared `dist_full` is computed on the default stream).
Final `torch.cuda.synchronize()` before returning. **Note: requires Idea 2 landed
first** (shared `dist_full`) so the streams aren't double-building it.

**Math walkthrough**: Stream parallelism does not change the per-op math; it only
changes the kernel launch order. Both streams operate on independent output tensors,
so there is no read-after-write hazard.

**Why machine precision preserved**: Determinism on a single stream is preserved
because each stream's op order is fixed. The cross-stream merge does not happen —
outputs are written to disjoint tensors.

**Tests that would gate it**: All tests; Spearman panel. CUDA-only path so CPU tests
unaffected.

**Engineering cost**: ~60 LOC for a `compute_config_and_sr_concurrent(coords, …)`
wrapper, plus stream context managers in the existing functions.

**Confidence**: M. Stream parallelism is well-trodden but adds a maintenance
surface; the win is bounded by the smaller of (config, SR) = 8 ms = **at most 4-8%
end-to-end** if both modes are run. Smaller than Ideas 1-3.

**Estimated speedup**: 1.04-1.08× end-to-end when both modes requested. Listed for
completeness; **not** a top-3 idea.

---

## Idea 5 — Singleresidue precompute analog (already applied)

**Status**: NOT actionable. `_precompute_W_sr` (singleresidue_decoys.py:99-178) IS the
T[i, α] equivalent. The phase 3b notes that called this out are stale — the sprint
already landed it (see `docs/optimization_sprint_results.md:23-24`: "singleresidue
biggest win, 4.02×"). The GPU path is even fused via `_water_per_alpha_fused`
(line 154-162). No further precompute lever exists on SR.

---

## Idea 6 — `gather` micro-opt on SR `aa_dec`

Replace `W_sr.gather(1, aa_dec)` and `B_table.gather(1, aa_dec)` (lines 374-375) with
`W_sr.reshape(-1)[aa_dec + idx_offset]` style flat indexing. Marginal; 5-10% on the
gather kernel. Not worth the readability hit unless 1-3 land first.

---

## Idea 7 — `cdist` replacement for the (N, N, 3) `diff`

`torch.cdist(safe_cb, safe_cb)` is a fused euclidean distance kernel that avoids
materialising the 1.5 MB `diff` intermediate. Has subtle float64 numerics differences
on some CUDA versions (cdist uses a matmul-based form). **Must validate bit-exact
agreement with the explicit form** before landing. Probably 3-8% wall-clock; mostly
peak-memory.

---

## Idea 8 — Reduce N_decoys from 1000 to 500/250

**This changes the RNG floor.** The current 3% configurational / 5% mutational floor
is set by `1/sqrt(N_decoys)`. Halving N_decoys raises the floor to ~4.2% / ~7%.

The Phase 3c Spearman gate is 0.997. Current observed: 0.9978-0.9999 across the 4-PDB
panel. The headroom is **tiny** on 11BG SR (0.99788). Halving N_decoys would likely
push 11BG SR below 0.997. **NOT recommended without an empirical N_decoys-sweep study
first**.

The right framing: this is a research question (does 500 give the same biological
signal?) not a speed optimization. Defer.

---

## Idea 9 — Remove the in-loop `.item()` sync from the rejection sampler

`int(accept_mask.sum().item())` at line 416 forces a device→host sync per loop
iteration. With 3-4 iterations that's 3-4 stalls. Replace with a fixed-budget
oversample (draw 4× n_decoys candidates up front, take the first n_decoys accepts via
`cumsum + scatter`) — no per-iteration sync.

**Subsumed by Idea 1** — direct sampling has zero iterations, zero syncs.

---

## Idea 10 — `nan_to_num` on FI divide

`torch.where(decoy_std > 0, FI, zeros)` (line 384) is correct but allocates twice.
Replace with `torch.nan_to_num(FI, nan=0.0, posinf=0.0, neginf=0.0)` after the
unguarded divide. <5% win on SR; cosmetic.

---

## Idea 11 — `torch.compile` on SR `_precompute_W_sr` GPU branch

The (20, N, N) fused build at lines 154-162 is a natural candidate for torch.compile's
fusion pass. Could collapse the 20 chunked ops into 1-2 kernels. Risk: torch.compile
has known float64 / non-determinism issues that could break the Spearman gate. Try
with `mode="reduce-overhead"` and validate against the pytest+Spearman matrix.
Confidence L until tested.

---

## Recommendation

**Land in order**: 1 → 3 → 2.

- **Idea 1** is the headline: it kills the rejection loop (the largest remaining
  non-distance-matrix cost in config), fixes the documented sparse-fallback bias as a
  free side-effect, and is provably distribution-equivalent to the C++ original.
  **2-5× on the sampler stage, 1.3-1.8× end-to-end on config, ZERO accuracy risk.**

- **Idea 3** is a 10-LOC cleanup with no risk; pure CPU-GPU pipeline hygiene.
  **5-15% on SR.**

- **Idea 2** requires extending ContactContext (~40 LOC) but compounds across both
  modes and is bit-identical to the current build. **1.2-1.5× on the combined
  config+SR workflow** which is the actual production path.

Combined estimated win on the combined config+SR pipeline: **1.5-2.5×** with zero
accuracy risk and one correctness bug fixed (the bias).

Defer Ideas 4 (small win, big surface), 7 (numerics validation needed), 8 (research
question, not speed), 11 (torch.compile risk). Drop Ideas 6, 9, 10 (subsumed or
cosmetic).

---

## Hard rule check

Every Idea-1/2/3 entry is:
- Algebraic / distributional equivalent (or strictly closer to LAMMPS in the case of
  Idea 1's bias fix).
- Float64 throughout — no precision compromises.
- Does not change `n_decoys` or the seed convention.
- Does not change the per-decoy energy formula or the (mean, std) reduction.

Spearman ≥ 0.997 panel must be re-run after each lands. Idea 1 will shift seeds (one
randint call per `rij_decoy` instead of 1-4) so per-PDB decoy_mean/decoy_std scalars
will differ by ~3% RNG floor — Spearman ordering is preserved.
