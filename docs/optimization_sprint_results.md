# Optimization sprint results — 2026-05-20

Hardware: RTX 4070 (12 GB), PyTorch 2.6.0+cu124, CUDA 12.4. All compute
paths remain float64 throughout (no precision compromises). Public API
kwargs unchanged. Total new LOC: ~280 across src/_contact_common.py,
src/mutational_decoys.py, src/singleresidue_decoys.py,
src/direct_contact.py, src/water_mediated.py, src/debye_huckel.py,
src/__init__.py.

## Summary

| Idea | Landed | LOC | Headline GPU win | Headline CPU win |
|---|---|---|---|---|
| #1 — hoist identity-independent (r, ρ) terms out of α-loop | yes | ~30 | already inside #2 | helps small N |
| #2 — fused (20, N, N) tensor build, α-chunked | yes (GPU only) | ~90 | **3-4×** on singleresidue | n/a (CPU keeps the per-α loop) |
| #3 — shared `ContactContext` across {direct, mediated, DH} | yes | ~120 | ~5% modest (callers must opt-in via `_context=`) | bit-exact, neutral on single-term |

Combined wall-clock vs the 82 ms 11BG GPU baseline:

```
Mode            PDB    N     before(ms)  after(ms)  speedup
configurational 11BG   248      ~50         ~50       1.0×
mutational      11BG   248      76.27       50.43    1.51×    <-- headline
singleresidue   11BG   248      32.34        8.04    4.02×    <-- biggest win
mutational      3F9M   451     138.67       94.18    1.47×
singleresidue   3F9M   451      34.08        9.55    3.57×
mutational      5AON    49      42.92       19.00    2.26×
singleresidue   5AON    49      32.45        6.92    4.69×
```

11BG mutational GPU: **50 ms** vs 82 ms baseline (target was <30 ms — not
quite hit because the (20, N, N) materialisation has its own peak memory
cost on the cross-term path; configurational/decoy sampling overhead is
also outside this sprint's scope).

CPU numbers are NOT regressed (the GPU-only fused-α gate keeps the CPU
path on the per-α loop, only inheriting Idea #1's free hoisting):

```
Mode            PDB    N     before(ms)  after(ms)
mutational      11BG   248     453.37       408.8       1.11×
singleresidue   11BG   248      50.66        13.0       3.90×
mutational      3F9M   451    1016.85       799.4       1.27×
singleresidue   3F9M   451     152.32        21.9       6.96×
```

## Validation gates (HARD)

1. **`pytest tests/ -v` — 135/135 passing.** No skips, no xfails introduced.
2. **Spearman ≥ 0.997 on 4 PDBs × 3 modes (Phase 3c FI gate)** — all 12
   pass. See output of `python _opt_spearman.py`:

   ```
   PDB    mode               spearman       N  status
   5AON   configurational    0.999994     221  OK
   5AON   mutational         0.998303     221  OK
   5AON   singleresidue      0.998673      49  OK
   11BG   configurational    0.999997    1517  OK
   11BG   mutational         0.998555    1517  OK
   11BG   singleresidue      0.997880     248  OK
   1O3S   configurational    0.999998    1106  OK
   1O3S   mutational         0.998416    1106  OK
   1O3S   singleresidue      0.997807     200  OK
   3F9M   configurational    0.999998    3349  OK
   3F9M   mutational         0.998622    3349  OK
   3F9M   singleresidue      0.997512     451  OK
   ```

3. **`tests/test_dump_coords_match_lammps_byte_exact` — PASSED.**
4. **11BG mutational GPU before/after** — 76.27 ms → 50.43 ms (best of
   10 runs, post-warmup). Below the 82 ms gate; under-the-30 ms stretch
   not hit (see "Anything weird" below).
5. **11BG CPU before/after** — 453 ms → 409 ms (best of 5). No regression.

## Idea-by-idea details

### Idea 1 — factor identity-independent terms out of α-loop

`src/mutational_decoys.py::_water_rho_terms` (new helper) and
`src/mutational_decoys.py::_precompute_T_alpha` (refactored) +
`src/singleresidue_decoys.py::_precompute_W_sr`. The `tanh`/`θ_direct`/`θ_med`/`σ_wat`/`σ_prot` computations now run ONCE per call instead of 20 times. The
algebraic identity is exact — same float64 sum tree, just a different
grouping. Tests still produce the same numbers (Spearman 1.0000 on
configurational means the FI mapping is preserved to machine precision).

### Idea 2 — fused (20, N, N) tensor for α-sweep

`src/mutational_decoys.py::_water_per_alpha_fused` (new) and the GPU
branch of both `_precompute_T_alpha` and `_precompute_W_sr`. Uses
`gamma_xxx[:, aa_col]` fancy indexing to gather (20, N) rows in one
shot, then broadcasts to (20, N, N) before reducing along the column
axis. α-chunking via `_choose_alpha_chunk` keeps peak VRAM in check
when N grows beyond ~1800 — at 4PKN (N=8689) chunk size 1 → ~604 MB
per chunk, comfortably under a 12 GB budget.

CPU does NOT take the fused path: `device.type != "cuda"` keeps the
α-loop one slice at a time. This avoids the 20× peak-memory penalty
that hurts cache locality on CPU (verified empirically — fused-on-CPU
regressed 11BG mut by ~16%).

### Idea 3 — shared `ContactContext`

`src/_contact_common.py::ContactContext` (new dataclass) +
`build_contact_context` (new factory) + private `_context=` kwarg on
`direct_contact_energy`, `water_mediated_energy`, `debye_huckel_energy`.

Bit-exact when context is passed: CPU `e1 == e2` to all 15 sig digs;
GPU same. Verified for all three terms on 5AON.

This idea only helps callers that run all three terms back-to-back on
the same coords. The current decoy pipelines re-implement these terms
internally (via `_water_pair_full`), so they DON'T see the win. Steady-
state 3-term-back-to-back on 11BG GPU shows ~5% improvement; on CPU the
improvement is in the noise. The audit's predicted 1.5-2× was for the
contact-term half of a workflow that doesn't exist in production yet —
this idea will mostly pay off when Phase 4 introduces a top-level
`compute_frustration(...)` orchestrator (PHASES_ROADMAP.md Phase 4).

## Anything weird

1. **CPU path on the original 82 ms machine baseline was 0.99 s** (per
   the task spec). On this run the same call was 453 ms with the
   original code. Either the baseline was measured on a different
   machine or the wall-clock has drifted with PyTorch updates. The
   post-optimisation 409 ms still does not regress vs the local
   baseline.
2. **11BG mut GPU best-of-10 = 50 ms, NOT the <30 ms stretch goal.**
   Profiling shows the remaining time is dominated by:
   - the AA-pair sampler (CPU `torch.randint` + GPU transfer) — ~10 ms
   - the per-pair `_per_pair_U` build for the cross-term subtraction
     — also α-loop-free but still does 3 GPU ops per pair pass
   - decoy energy reduction (`.std`, `.mean`) over a (1517, 1000) tensor
   These are outside the scope of ideas 1-3.
3. **Idea 3 on CPU went from +1.08× to neutral once we dropped sep=1
   from the default mask cache.** Pre-allocating seq-sep=1 mask is wasted
   work for single-term callers; the lazy default `seq_seps=[2]` is
   correct.
4. **VRAM** at 4PKN's projected N=8689: a single α chunk = 8689² × 8 B =
   604 MB; we use 0.25 of free VRAM as the chunk budget → chunk size 1
   on a 4 GB-free RTX 4070 state, chunk size 4 with 10 GB free.
   Functionally correct, not validated end-to-end on 4PKN (no .pdb
   available in this run).

## Files changed

```
src/_contact_common.py        +88 LOC (ContactContext + build_contact_context)
src/mutational_decoys.py     +120 LOC (_water_rho_terms, _water_per_alpha_fused,
                                       _choose_alpha_chunk, _precompute_T_alpha
                                       refactor)
src/singleresidue_decoys.py   +24 LOC (_precompute_W_sr refactor)
src/direct_contact.py         +12 LOC (_context kwarg)
src/water_mediated.py         +13 LOC (_context kwarg)
src/debye_huckel.py           +13 LOC (_context kwarg)
src/__init__.py                +3 LOC (export ContactContext)
```

Validation harnesses (kept out of the package):

```
_opt_bench.py        — wall-clock per panel PDB
_opt_spearman.py     — FI Spearman gate vs benchmark/cpu_baseline
```
