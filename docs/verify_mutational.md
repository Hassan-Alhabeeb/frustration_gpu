# VERIFY-MUTATIONAL — final-pass audit of mutational mode

Reviewer: Opus 4.7, 2026-05-21
Scope (READ-ONLY):
- `src/mutational_decoys.py` (1009 LOC after FIX-4)
- `src/compute_frustration.py` (1239 LOC, mutational orchestrator path)
- `src/frustration.py` (585 LOC, FI + emit)
- `benchmark/cpu_baseline/mutational/` — 10 LAMMPS reference dumps

Predecessor work: `docs/speed_fix4_results.md` (FIX-4 sprint, 2026-05-21) +
`docs/qa3_orchestrator.md` (QA3 orchestrator audit, 2026-05-21).

---

## Verdict

**PASS.** Mutational mode is end-to-end correct and matches LAMMPS to within
the documented ~5% RNG floor on the four-PDB panel. FIX-4's three landed
changes (SPEED-1 Idea 2 stack-hoist, SPEED-1 Idea 3 on-device gather, QA-2
M-1 burial-loop vectorization) were verified bit-identical to their
pre-fix equivalents on synthetic + real inputs. The T-cube factorisation
in `_precompute_T_alpha` is mathematically rigorous (verified to machine
precision: max |cube_E - naive_E| = 1.776e-15 on 100 random decoys per
pair). All 223 tests still pass.

The audit found **0 CRITICAL**, **0 HIGH**, **2 MEDIUM**, and **5 LOW**
findings. The MEDIUMs are scalability concerns at 4PKN-scale N=8689 that
have not been exercised on the current hardware panel; none affect the
panel results. No CPU-vs-CUDA bit-identicality, contrary to the brief's
ask of "max |ΔFI| = 0" — the differences are at parallel-reduction-order
floor (~5e-7 on FI, ~1e-6 on decoy_mean/std).

### Severity counts

| Sev      | Count | Items |
|----------|------:|-------|
| CRITICAL | 0     | — |
| HIGH     | 0     | — |
| MEDIUM   | 2     | M-1 α-chunked path still pre-allocates full (20, N, N) → 4PKN OOM; M-2 CPU vs CUDA same-seed FI differs by ~5e-7, NOT bit-identical |
| LOW      | 5     | L-1 `_per_pair_U` accepts unused `contact_cutoff` kwarg; L-2 all-same-AA chains → decoy_std=0 → FI=nan/inf, undocumented; L-3 `residues` post-filter uses .iterrows; L-4 `mediated_r_max`/`direct_r_max` parameters threaded through `_per_pair_U` but C++ uses fixed 6.5/9.5; L-5 the docstring at lines 56-69 swaps the cross_i/cross_j sign conventions when describing the cube algebra (mathematically correct, but reads confusingly) |

### Top 3 findings

1. **(MEDIUM, M-1)** `_water_per_alpha_fused` chunking does NOT bound peak
   output VRAM. At N=8689 (4PKN-scale), the `torch.empty((20, N, N), ...)`
   allocation at `mutational_decoys.py:308-310` requests 12.08 GB
   regardless of `alpha_chunk_size`, which exceeds the 11.65 GB free VRAM
   observed on this dev box. The chunking saves only the intermediate
   `(chunk, N, N)` working tensors, not the output. A correct
   implementation would accumulate directly into `T_alpha` (N, 20) and
   free each chunk after summing. This is the exact 4PKN VRAM gate the
   FIX-4 sprint deferred — confirmed unresolved.

2. **(MEDIUM, M-2)** Brief asked for "CPU vs CUDA same seed → max |ΔFI|
   should be 0". Observed: max|ΔFI| = 4.96e-07 on 5AON, 5.25e-07 on 11BG;
   decoy_mean max|d| ≈ 1-2e-6; decoy_std max|d| ≈ 1e-6. These are at the
   parallel-reduction-order floor for float64. The existing test
   `test_cpu_gpu_decoy_std_agreement_5aon` documents this as a 1e-5
   tolerance — the actual delta is 100× tighter than the test gate, but
   it is NOT zero. The PRNG sequence IS bit-identical across devices
   (CPU `torch.Generator`, then `.to(device, non_blocking=True)`), so the
   discrepancy comes from reduction order in `T_alpha.sum`,
   `decoy_mean.mean(dim=1)`, etc.

3. **(LOW, L-2)** All-same-AA chains (homo-polymer, e.g. all-Gly or
   all-Cys) silently produce `decoy_std = 0` and `FI = nan or ±inf`.
   `compute_frustration_index` has an `eps` kwarg that would clamp the
   denominator, but it defaults to `0.0` (true division). The `FI`
   value is mathematically undefined for a degenerate single-AA
   ensemble — but downstream consumers (welltype/classify_frustration,
   the dataframe emit) propagate NaN/inf without complaint. This is a
   user-error class of input, but a defensive `eps` default or an
   explicit error would help.

### T-precompute math rigour

**RIGOROUS — verified to machine precision.** Picked native pair (i=2, j=6)
on 5AON, generated 100 random `(α_i, α_j)` decoy AA assignments, computed
the decoy energy two ways:

- **Cube path** (`_precompute_T_alpha` factorisation):
  `E = T[i, α_i] + T[j, α_j] - U_iSlot_kj - U_jSlot_ki + pair_term + burial_i + burial_j`
- **Naive path**: explicit `Σ_{k≠i, r_ik<9.5}` and `Σ_{k≠j, r_jk<9.5}`
  via `_water_pair_full` for every (i, k) and (j, k), then add
  `pair_term + burial`.

Result: `max |cube_E - naive_E| = 1.776e-15` (i.e., one ULP at ~10
kcal/mol scale, which is float64 round-off). The factorisation is
algebraically exact — no approximation, no shortcut.

### FIX-4 bit-identity preservation

| Change | Verification | Result |
|--------|-------------|--------|
| SPEED-1 Idea 2 (stack 3 water_pair calls) | 11BG synthetic: 3 stacked outputs vs 3 unstacked `_water_pair_full` calls | **max\|d\| = 0.0** on all three tensors |
| SPEED-1 Idea 3 (on-device gather, CPU Generator preserved) | Same seed CPU vs CUDA `aa_i_dec` / `aa_j_dec` byte-equal (integer gather) | PRNG-sequence bit-identical (decoy means differ only by reduction-order floor — see M-2) |
| QA-2 M-1 (vectorised burial 3-well loop) | 50×1000 synthetic input: vectorised vs explicit `for w_idx in range(3): stack` | **max\|d\| = 0.0** |

All three FIX-4 changes preserve bit-identity to the pre-FIX baseline at
the per-element math level. The pre-existing `speed_fix4_results.md`
also documented `max|d| = 0.0` for 8/8 mutational + 8/8 singleresidue
combos in `_fix4_capture_fi.py` — independently re-verified here.

### α-chunking at 4PKN-scale

`_choose_alpha_chunk` (lines 321-344) returns the right *intent* but the
caller `_water_per_alpha_fused` defeats it (see M-1):

| N    | device   | full (20,N,N) | chosen chunk | peak working alloc | comment |
|------|----------|--------------:|-------------:|-------------------:|---------|
| 248 (11BG)   | cuda | 9.8 MB | **0** (full) | 9.8 MB | comfortable |
| 451 (3F9M)   | cuda | 33 MB | **0** (full) | 33 MB | comfortable |
| 2000 | cuda | 640 MB | **0** | 640 MB | still one-shot |
| 5000 | cuda | 4.0 GB | **14** | ~2.8 GB intermediates + 4.0 GB output | OK if free > ~7 GB |
| 8689 (4PKN)  | cuda | 12.08 GB | **4** | 2.42 GB intermediates + 12.08 GB output | **OOM** on 11.65 GB free |
| any  | cpu  | any | **0** | n/a | CPU path loops α serially, no fused tensor |

For 11BG (N=248) on a 12 GB GPU, chunking is correctly bypassed (chunk=0,
full one-shot). For 4PKN-scale, chunking kicks in but the output
allocation is the real ceiling; the function will OOM at line 308 before
processing the first chunk. This is the same gate that deferred SPEED-1
Idea 1 (T-cube memoization) — both share the 4PKN VRAM unresolved issue.

CPU path is unaffected (it does not build `(20, N, N)` at all; loops α
into `T_alpha[:, alpha]` one column at a time, lines 587-595).

---

## Detailed findings

### M-1 — α-chunked path pre-allocates full (20, N, N) output

**File / line:** `src/mutational_decoys.py:308-310`

```python
out = torch.empty(
    (20, n, n), dtype=theta_direct.dtype, device=theta_direct.device
)
```

The `torch.empty((20, n, n), ...)` runs **before** the chunked write
loop. At N=8689 in float64, this requests 12.08 GB. The chunking saves
the `(chunk, N, N)` intermediates (`sigma_gamma_med_c` etc.), but the
output is the real ceiling. The caller `_precompute_T_alpha` immediately
sums to (N, 20) — so the (20, N, N) tensor never needs to exist
materially; it could be eliminated entirely.

**Refactor sketch (NOT applied):** in `_water_per_alpha_fused`, replace
the (20, N, N) allocation with `T_partial = torch.zeros((20, N), ...)`
and accumulate the column-sums per chunk: `T_partial[s:e] += w_chunk.sum(dim=2)`.
That bounds peak working memory to `chunk × N × N × elem_bytes` + `(20, N)`,
which at N=8689, chunk=4 = 2.42 GB + 1.4 MB instead of 14.5 GB.

**Confidence:** 95%. The audit panel has no N>451 PDB to gate this
empirically. The FIX-4 sprint document explicitly noted this 4PKN gate
as unresolved. Severity MEDIUM because the panel runs perfectly fine —
no current consumer is affected.

### M-2 — CPU vs CUDA same-seed FI differs by ~5e-7

**File / line:** `src/mutational_decoys.py:576, 989-990`

Observed on 5AON (49 res, 221 pairs) seed=0:
- `max |decoy_mean_cpu - decoy_mean_cuda| = 1.03e-06`
- `max |decoy_std_cpu  - decoy_std_cuda|  = 8.61e-07`
- `max |FI_cpu - FI_cuda| = 4.96e-07`

Observed on 11BG (248 res, 1517 pairs):
- `max |decoy_mean_cpu - decoy_mean_cuda| = 1.96e-06`
- `max |decoy_std_cpu  - decoy_std_cuda|  = 1.01e-06`
- `max |FI_cpu - FI_cuda| = 5.25e-07`

PRNG sequence IS preserved across devices (CPU `torch.Generator(device='cpu')`
draws → `.to(device, non_blocking=True)` for indexing — integer gather is
bit-exact). The discrepancy is reduction-order:

- `T_alpha = w_all.sum(dim=2)` (GPU path) vs the explicit α-loop with
  `w.sum(dim=1)` (CPU path): different tile/block-reduction tree.
- `E_decoy.mean(dim=1)` and `.std(dim=1)` over a 1000-element axis: GPU
  uses a parallel pairwise reduction; CPU uses a serial Kahan-free sum.

These are NOT bugs — they are intrinsic to float64 floating-point
non-associativity. But the brief asked for `max |ΔFI| = 0`, which is not
achievable without enforcing single-thread reduction on CUDA (would
defeat the speedup).

The existing test `test_cpu_gpu_decoy_std_agreement_5aon` (`tests/
test_mutational_mode.py:264-281`) sets the tolerance at 1e-5 — actual
delta is 100× tighter than that gate. Status: documented behaviour, not
a regression.

**Confidence:** 100%. Severity MEDIUM only because the brief explicitly
asked for bit-identity.

### L-1 — `_per_pair_U` accepts unused `contact_cutoff` kwarg

**File / line:** `src/mutational_decoys.py:610` (signature),
called from line 947.

The kwarg is documented (lines 637-644) as "guaranteed True for every
native pair (the outer enumeration only emits pairs that already satisfy
it)" — i.e., the function REQUIRES the guarantee but doesn't enforce it
internally. It's a vestigial parameter from earlier drafts. Confidence
100%. Recommend dropping from the signature.

### L-2 — Single-AA-composition inputs produce `decoy_std = 0`, `FI = nan/inf`

**File / line:** `src/mutational_decoys.py:990`,
`src/frustration.py:147-151`.

Verified on synthetic all-Gly (n=5) and all-Cys (n=4) chains:
- all-Gly: `decoy_mean == E_native` (decoy AAs are all Gly, identical to
  native), `decoy_std = 0`, `FI = [inf, nan, inf]`.
- all-Cys: same pattern.

`compute_frustration_index` (`frustration.py:113-151`) has an `eps`
kwarg that would clamp `decoy_std`, but mutational mode calls it via
`compute_frustration.py:827` **without setting `eps`** (defaults to 0).
The downstream `welltype_from_contact` and `classify_frustration` both
inherit NaN/inf into the dataframe. No error is raised.

This is a degenerate input (no real protein has uniform AA composition),
so severity is LOW. Recommendation: either pass `eps=1e-9` from the
orchestrator OR raise a clear ValueError when the protein has fewer than
2 distinct AAs.

### L-3 — `residues` post-filter still uses `df.iterrows()`

**File / line:** `src/compute_frustration.py:921-944` (3 loops).

Already flagged in `qa3_orchestrator.md` M-3. Not specific to mutational
mode, but reachable via `compute_frustration(mode='mutational',
residues={...})`. For the 5AON × `residues={"A":[25]}` test (8 rows kept)
it's fine; on a 10K-pair PDB with a 100-residue filter the iterrows
quadratic feel would dominate.

### L-4 — `mediated_r_max` / `direct_r_max` are exposed as kwargs but C++ uses fixed 6.5 / 9.5

**File / line:** `src/mutational_decoys.py:754-757`,
`src/decoys.py:159` (`MEDIATED_R_MAX_A = 9.5`).

`mutational_decoy_stats` exposes `direct_r_min/max` and
`mediated_r_min/max` as overridable kwargs. The C++
`fix_backbone::compute_decoy_ixns` uses hardcoded 4.5/6.5 (direct)
and 6.5/9.5 (mediated). If a user passes a non-default `mediated_r_max`,
the cross-mask in `_precompute_T_alpha` STILL uses `contact_cutoff=9.5`
(line 535), so the result diverges from a self-consistent recalculation.
The kwargs are dangerous to expose. Confidence: 90% — would need user
brief to confirm if non-default values are supported.

### L-5 — Cross-term sign-convention docstring at lines 56-69 is mathematically correct but confusing

**File / line:** `src/mutational_decoys.py:56-69`.

The docstring describes the per-pair decoy energy as:
```
cross_i[d] = T[i, α_i[d]] - U[i, j, α_i[d]]
cross_j[d] = T[j, α_j[d]] - U[j, i, α_j[d]]
pair[d]    = water_pair(r_ij, α_i[d], α_j[d], rho_i, rho_j)
burial[d]  = burial_pair(α_i[d], rho_i) + burial_pair(α_j[d], rho_j)
E_decoy(i, j, d) = pair[d] + cross_i[d] + cross_j[d] + burial[d]
```

This is correct — the `U[i, j, α]` subtraction removes the k=j
contribution from T[i, α], avoiding double-counting with `pair[d]`. The
implementation matches at lines 964-987. But the implicit assumption
that `T[i, α]` INCLUDES the k=j contribution (so we have to subtract it)
is buried in the cross_mask logic at line 537. A reader would benefit
from one explicit sentence "T[i, α] includes all k≠i with r_ik<9.5,
INCLUDING k=j" alongside the existing exposition.

Not a bug. LOW severity, doc-only.

---

## A. Mode correctness — row-by-row vs LAMMPS dumps

Each panel PDB: full mutational run, all pairs matched against
`benchmark/cpu_baseline/mutational/<PDB>_tertiary_frustration.dat`.

| PDB  | N_pair | pair-count match | i<j invariant | r_ij max\|d\| | rho max\|d\| | E_native max\|d\| | decoy_mean Spearman | decoy_std Spearman |
|------|-------:|:----------------:|:-------------:|--------------:|-------------:|------------------:|--------------------:|-------------------:|
| 5AON | 221    | 221/221 ✓        | ✓             | 5.00e-04      | 4.70e-04     | 4.96e-04          | 0.99577             | 0.99738            |
| 11BG | 1517   | 1517/1517 ✓      | ✓             | 5.00e-04      | 5.00e-04     | 5.06e-04          | 0.99328             | 0.99786            |
| 1O3S | 1106   | 1106/1106 ✓      | ✓             | 5.00e-04      | 5.00e-04     | 5.04e-04          | 0.99294             | 0.99783            |
| 3F9M | 3349   | 3349/3349 ✓      | ✓             | 5.00e-04      | 5.00e-04     | 5.06e-04          | 0.99137             | 0.99861            |

- **r_ij, rho_i, rho_j**: byte-equal at the LAMMPS 3-decimal print
  precision (max|d| = 0.0005 = half-LSB at f=3).
- **E_native**: max|d| < 5.07e-04 — within the print-precision floor.
  E_native is deterministic (no RNG), so we expect exact agreement.
- **decoy_mean / decoy_std**: stochastic. Spearman > 0.99 on all 4
  PDBs. Mean |Δ decoy_mean| = 0.1-0.17 kcal/mol — that's the ~5% RNG
  floor documented in the module docstring (lines 90-95).
- **i<j invariant**: holds (Phase 5 P1 fix at `mutational_decoys.py:446`
  is correct — was a real bug before the fix).

## A.3 — AA-pair coverage check

For each native pair, count distinct `(a_i_dec, a_j_dec)` cells across
the 1000 decoys.

| PDB  | unique a_i in protein | unique a_j in protein | distinct (a_i, a_j) per native pair (min/mean/max) | total (a_i, a_j) over corpus |
|------|----------------------:|----------------------:|---------------------------------------------------:|-----------------------------:|
| 5AON | 16/20                 | 16/20                 | 195 / 206.6 / 219                                  | 16×16 = 256 (the protein only contains 16 AAs) |
| 3F9M | 20/20                 | 20/20                 | (not measured) — sampled all 400 (a_i, a_j) cells  | **400/400** ✓ |

5AON only contains 16 of 20 AA identities, so the maximum reachable
(a_i, a_j) is 256. The empirical "200 per native pair" reflects
composition-weighted sampling — exactly the documented behaviour
(C++ samples `aa[randint(0,N)]`, NOT `randint(0,20)` uniform). 3F9M is
large enough (451 res) to contain all 20 AAs and the corpus covers all
400 (a_i, a_j) cells. ✓

## B. T-precompute optimization correctness

Selected native pair (i=2, j=6) on 5AON (5N9R skipped — too large for
the naive comparison). Generated 100 random `(α_i, α_j)` decoy pairs;
for each, computed `E_decoy` two ways:

- **Cube**: `T[i, α_i] + T[j, α_j] - U_iSlot_kj - U_jSlot_ki + pair_term + burial`
- **Naive**: explicit double-loop over k (with `k ≠ i, k ≠ j, r_ik<9.5`
  / `r_jk<9.5` filters) calling `_water_pair_full` for each (i,k) and
  (j,k), then add `pair_term + burial`.

**Result: `max |cube_E - naive_E| = 1.776e-15`** (one ULP at ~10 kcal/mol
scale). This is the float64 representation limit. The cube factorisation
is rigorous.

## C. FIX-4 changes — bit-identity

### Idea 2 (stack 3 water_pair calls in `_per_pair_U`)

Verification: ran 11BG mutational with 1000 decoys. Stacked path
(current FIX-4) vs explicit-3-`_water_pair_full`-calls path.

| tensor       | max\|d\| |
|--------------|---------:|
| U_iSlot_kj   | **0.0**  |
| U_jSlot_ki   | **0.0**  |
| pair_term    | **0.0**  |

Confirmed bit-identical. The hoisted θ/σ tensors carry the same
floating-point values; the three weighted sums execute the same FMA
sequence as the unstacked calls.

### Idea 3 (on-device aa gather)

Verification: PRNG sequence preserved (CPU `torch.Generator`,
`idx_i/j` drawn on CPU, then `.to(device)`). Integer gather is bit-exact
across devices. The same `aa_i_dec`, `aa_j_dec` tensors are produced
whether `aa` is materialised on CPU or CUDA before indexing.

Caveat: see M-2 — the downstream sum/std reductions on these tensors
produce ~1e-6 differences between CPU and CUDA due to reduction-order
non-associativity. That is NOT this idea's fault.

### M-1 (burial 3-well vectorization)

Verification: 50×1000 synthetic input (`aa = randint(0,20)`,
`rho = rand × 5`), compared vectorised `_burial_residue_energy` (current
FIX-4) against an explicit `for w_idx in range(3): wells.append(...);
stack(wells, dim=-1)` reference.

**max|d| = 0.0**. The vectorised version performs the same per-element
`tanh(κ(ρ - ρ_min[w])) + tanh(κ(ρ_max[w] - ρ))` operation in a fused
broadcast; PyTorch dispatches the same elementwise FMA pattern.

## D. Alpha-chunking under VRAM pressure

`_choose_alpha_chunk` (lines 321-344): correct heuristic but the caller
defeats it — see M-1.

Behaviour table (12.5 GB total VRAM, 11.65 GB free at test time):

| N         | full (20,N,N) | chosen chunk | comment                                |
|----------:|--------------:|-------------:|----------------------------------------|
| 248 (11BG)|       9.8 MB  |       0      | full-α, no chunking needed             |
| 451 (3F9M)|        33 MB  |       0      | full-α                                 |
| 1000      |       160 MB  |       0      | full-α                                 |
| 2000      |       640 MB  |       0      | full-α                                 |
| 5000      |       4.0 GB  |      14      | chunked (intermediates ~2.8 GB)        |
| **8689 (4PKN)** | **12.08 GB** | **4**  | **OOM at output alloc, see M-1**     |
| CPU any   |           n/a |       0      | CPU path loops α serially              |

11BG decision: chunk=0 (full one-shot). 4PKN decision: chunk=4, but the
12.08 GB output `torch.empty` will fail to allocate. This is the same
4PKN gate that deferred SPEED-1 Idea 1.

## E. Kwarg combinations

15 strategic combos on 5AON (mode='mutational' fixed):

| combo                            | n_pairs | result |
|----------------------------------|--------:|--------|
| plain (seed=0, cpu)              | 221     | ✓      |
| seed=7                           | 221     | ✓ (different stochastic output) |
| chain="A" (single chain)         | 221     | ✓      |
| chain=["A"] (list)               | 221     | ✓ (parser-level filter, byte-identical to chain="A") |
| residues={"A":[25]}              | 8       | ✓ (post-filter keeps only pairs touching resnum 25) |
| device="cuda"                    | 221     | ✓      |
| precision=5                      | 221     | ✓ (5-decimal output)            |
| n_decoys=100                     | 221     | ✓      |
| seq_dist=3                       | 221     | ✓ (alternate rho cutoff)        |
| pair_min_seq_sep=3               | 174     | ✓ (fewer short-range pairs)     |
| electrostatics_k=4.15 (metadata only) | 221 | ✓                            |
| electrostatics_k=4.15, include_dh_in_e_native=True | 221 | ✓ (DH added to E_native) |
| dtype=float32                    | 221     | ✓ (lower precision floor)       |
| keep_incomplete_backbone=True    | 221     | ✓      |
| lammps_compat_altloc=True        | 221     | ✓ (no altloc on 5AON, no-op)    |

Multi-chain: `chain=["A","B"]` on 4HON → 4298 pairs spanning both chains
(parser-level filter; rho correctly reflects both chains, per QA3 H-2
fix). ✓

DNA: `include_dna=True` on 1O3S → math runs on protein-only subset
(1106 pairs, all chain A), DNA rows preserved for density emission. ✓

3F9M altloc: `lammps_compat_altloc=True` → 3349 pairs (matches LAMMPS
dump). ✓

## F. Edge cases

| edge case                      | behaviour | notes |
|--------------------------------|-----------|-------|
| Single residue (n=1, no pairs) | `n_pair = 0`, zero-shape tensors returned cleanly | `mutational_decoys.py:923-935` handles this branch |
| All-Gly chain (n=5, all CB=CA) | 3 pairs, runs without error | `decoy_std = 0` because single-AA-protein → see L-2 |
| All-Cys chain (n=4)            | 3 pairs, runs without error | same `decoy_std = 0` issue → L-2 |
| DNA+protein (1O3S, include_dna=True) + mutational | works; math runs on protein-only via `_subset_protein_only` | DNA rows preserved for density emit |
| `residues={"A":[10]}` (single residue) | filters dataframe to pairs touching res 10; full mutational still runs on all residues | correct — cross-terms need full structure |
| empty chain filter             | not tested directly; `parse_pdb` raises if chain missing | defensive at parse level |

## G. Determinism

| run                          | max\|d FI\| | gate         |
|------------------------------|------------:|--------------|
| CPU seed=0 vs CPU seed=0     | **0.0**     | bit-identical ✓ |
| CPU seed=0 vs CUDA seed=0    | 5.25e-07    | reduction-order floor (M-2) |
| CPU seed=0 vs CPU seed=7     | > 0.5       | genuinely different RNG ✓ |

Same-seed CPU twice → byte-equal. Same-seed CPU vs CUDA → 5e-7 (not
zero; explained by reduction order). Different seed → meaningfully
different stats (test `test_mutational_seed_reproducibility` already
gates this).

## Test suite

Full pytest run: **223 passed, 1 warning** in 15.87 s (the warning is the
known `APIDocsCoverageWarning` for emit_5adens_dat / chain_segments, not
mutational-related). All 20 mutational-specific tests pass:

```
tests/test_mutational_mode.py .................... [100%]
20 passed in 6.25s
```

## Cross-reference to prior reviews

- `docs/speed_fix4_results.md` (2026-05-21): documents the FIX-4 sprint
  + 16/16 mutational+SR bit-identical claim. This audit independently
  re-verified the bit-identity on synthetic + real inputs.
- `docs/qa3_orchestrator.md` (2026-05-21): orchestrator code review. H-1
  (`_xb_coords` per-element NaN mix), H-2 (calculate_frustration
  multi-chain rho), M-1 (no `cuda.synchronize()` around wall-clock) — all
  HIGH/MEDIUM findings have been fixed in `frustration.py:265`,
  `compute_frustration.py:639-650`, and `compute_frustration.py:629-635 /
  944-950` respectively (verified in this audit).
- `docs/qa2_decoys.md` (FIX-4 source) and `docs/speed1_mutational.md`
  (Idea 1-3 specs) — directly applied in FIX-4, all verified bit-
  identical here.

---

## Recommendations (NOT applied — READ-ONLY audit)

1. **M-1 fix** (high impact for 4PKN-scale users): refactor
   `_water_per_alpha_fused` to accumulate `(20, N)` partial sums rather
   than materialising `(20, N, N)`. Removes the 12 GB peak alloc for
   N=8689. Same effort: ~10 LOC.
2. **L-2 fix** (defensive): pass `eps=1e-9` into
   `compute_frustration_index` from the mutational orchestrator branch,
   or raise a clear ValueError when `len(set(aa_native)) < 2`. ~3 LOC.
3. **L-1 cleanup**: drop the unused `contact_cutoff` kwarg from
   `_per_pair_U`. ~1 LOC.
4. **L-4 audit**: decide whether `mediated_r_max` / `direct_r_max` are
   user-tunable. If yes, propagate to `cross_mask` in
   `_precompute_T_alpha`. If no, drop from the kwarg list. ~5 LOC either
   way.
5. **M-2 (informational)**: document the CPU vs CUDA ~5e-7 FI floor in
   `mutational_decoys.py:90-95` docstring. The current text discusses
   the LAMMPS-vs-ours 5% floor but not the CPU-vs-CUDA float64-reduction
   floor.

End of audit.
