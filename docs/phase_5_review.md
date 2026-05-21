# Phase 5 review — stress test + bench fairness

Audit of `benchmark/run_phase5.py`, `phase5_results.md`,
`phase5_panel_results.csv`, `phase5_frustrapy_comparison.csv`,
`phase5_spearman.csv`. Read-only. Methodology focus.

## Verdict

**CONDITIONAL** — headline speedups (median 14×, max 52.6× on 3F9M mutational,
4PKN end-to-end 17.6 s) are defensible. CPU↔GPU bit-exactness is rigorous on
deterministic modes but the "max abs diff = 0.0" cells in the results MD are
**not actually present in `phase5_spearman.csv`** (the CSV has only 3 empty
rows). The MD table is hand-curated or copy-pasted from a console log, not
machine-generated. Fix that before publication. 4PKN VRAM "paging" is a
**measurement quirk, not a real paging slowdown**.

## Findings by severity

### CRITICAL (1)

**C1. `phase5_spearman.csv` is essentially empty** — `wc -l` = 4 lines,
header plus three rows for 5AON, all five numeric columns blank. The
`phase5_results.md` shows a 30-row Spearman table with values like
`5AON configurational ref FI = 1.0000`, `3F9M singleresidue FI = 0.9975`,
`CPU↔GPU max |ΔFI| = 0.000000` for all 30 combos. Those numbers do not
match what the harness wrote to disk. Either:

- (a) the Spearman validation pass at `run_phase5.py:504-520` never
  populated the cache (the cache is populated in `_run_one` line 265 but
  only persists if both CPU and GPU were run in the same process — the
  `--force` rerun or `--skip-cpu`/`--skip-gpu` flag would empty it), or
- (b) someone re-ran with a subset and the MD was generated from a prior
  run's in-memory state.

The CSV file on disk doesn't substantiate the MD's claims. The MD is
plausible (and likely correct based on Phase 4's prior numbers) but
needs a regenerated CSV before it can be cited as evidence.

### HIGH (2)

**H1. CPU↔GPU "bit-exact" claim is defensible BUT the validation
artifact is missing.** Looking at `phase5_panel_results.csv`, the
`decoy_mean` / `decoy_std` columns for configurational mode agree to
~14 significant digits between CPU and CUDA (e.g. 5AON: CPU
-1.2442272905621115, CUDA -1.2442272905621117 — last-bit float64 differ
in dot-product reduction order, NOT bit-exact). The same holds for
singleresidue and mutational (FrstState counts are bit-identical
because they're discrete classifications post-quantile). The MD claim
"max FI diff = 0.000000 on every (PDB, mode)" is correct **at 6-decimal
print precision** but is not literally bit-exact at float64 precision.
The actual claim should be "max |ΔFI| < 1e-14" or "agreement to
machine precision after reduction reordering".

**H2. Frustrapy hardware mismatch is undocumented in the MD.** Frustrapy
runs on VM EPYC 32-core (Linux); "ours CPU" runs on local Windows.
These are different CPUs with different memory bandwidth and clock.
The MD says frustrapy is single-threaded and ours is single-threaded so
"single-PDB single-threaded is apples-to-apples" — but per-core
EPYC vs the local Windows CPU is NOT apples-to-apples. The "ours CPU
is 2-10× faster than frustrapy CPU" claim conflates algorithm speedup
with hardware difference. **The GPU-vs-frustrapy-CPU comparison is
still fair** (users run frustrapy on whatever CPU they have, and our
GPU on whatever GPU they have — the typical real-world comparison
frustrapy users care about is "what does this take on my machine vs
this GPU implementation"). But the CPU↔CPU comparison in column 4 of
the head-to-head table should be flagged.

### MEDIUM (4)

**M1. GPU warmup only runs once per process** (`_run_one._cuda_warm`
flag). The 5N9R mutational footnote (2290 ms vs comparable 3F9M
mutational 425 ms) is honest about this — first-call-on-this-PDB
allocator-state cost. But the harness could do per-mode warmup, not
just once. The MD's footnote is acceptable disclosure.

**M2. Single-run timings, no best-of-N.** `_run_one` does one timed
forward pass. For small GPU runs (5AON 70 ms) launch jitter ±5-10 ms is
~10% of wall time. Best-of-3 or median-of-3 would tighten the small-N
numbers. Less critical for the headline (3F9M mutational 425 ms) where
jitter is <1%.

**M3. 4PKN VRAM 22.4 GB on 12 GB card — measurement quirk, NOT real
paging.** `torch.cuda.max_memory_allocated()` returns the
**high-water mark of CUDA-allocator-tracked allocations**, not physical
VRAM in use. PyTorch's caching allocator can hold pointers to memory
that was never simultaneously resident. Running in 17.6 s for 8689
residues is consistent with the math actually fitting (probably ~6-8 GB
peak resident, with `max_memory_allocated` over-counting because of
allocator fragmentation + transient peaks released before the next
allocation). If it were truly paging through Windows shared memory,
8689² float64 = ~604 MB per α-slice but with WDDM paging the run would
be **minutes, not 17.6 s** (PCIe round-trips at ~16 GB/s would dominate
heavily — a 22 GB working set would need many seconds just to page-in
once). The MD over-claims paging. Real story: `max_memory_allocated`
reporting is the allocator's bookkeeping, not the resident set size.

**M4. Spearman computed correctly (scipy.stats.spearmanr) but per-pair
vs per-residue-density distinction is muddied.** Lines 144-157 use
`spearmanr` (which uses `rankdata(method='average')` for tie-handling
— correct). Per-pair FI is computed at line 327-330 from `f_ij` column
of LAMMPS dump vs our `FrstIndex` column. Per-residue density is at
line 347-350 from `nHighlyFrst` after a `merge` on `(Res, ChainRes)`.
These two signals are computed correctly. But the MD groups them
under a single "FI Spearman ≥ 0.9975" headline; the density-Spearman
column has many values < 0.5 (1O3S DNA bug, 3F9M altloc bug). The
"30/30 PASSED" claim is FI-only; the density gate is conditional on
`include_dna=True`/`lammps_compat_altloc=True` flags. This is
disclosed in the MD notes — acceptable, but slightly buried.

### LOW (3)

**L1. `time.perf_counter()` used correctly (monotonic).** ✓

**L2. `torch.cuda.synchronize()` before stopping timer at line 226.** ✓

**L3. Peak VRAM uses `max_memory_allocated` correctly.** ✓ (though see
M3 — the reported value over-counts in absolute terms).

## Frustrapy fairness analysis

| Check | Verdict | Note |
|---|---|---|
| Same `n_decoys` (1000)? | ✓ | Our `N_DECOYS = 1000`; frustrapy default is 1000. |
| Same `seed`? | ✗ MITIGATED | Our `seed=0` uses torch RNG; frustrapy LAMMPS uses libc `rand` seeded from `time()` or a default. **Decoy ensembles are different** — but FI rank correlation against LAMMPS reference dumps is 0.9975-1.0000, meaning the seed doesn't matter at the rank-level. ✓ for FI rank; ✗ for decoy energy bit-exact. |
| Same `seq_dist=12`? | ✓ | Frustrapy default = 12; our `compute_frustration.py:474` default = 12; harness uses default. |
| Same `electrostatics_k=None`? | ✓ | Our default = None (DH off); frustrapy default also off. |
| Frustrapy single-threaded? | ✓ | Per-PDB subprocess on VM with no `Pool`. Frustrapy internally spawns LAMMPS subprocesses per residue for mutational/singleresidue, which the speedup math correctly attributes as the source of the 30-53× wins. |
| Wall-clock includes parse/import? | ✗ FAIR | The frustrapy timer at `_run_frustrapy_on_vm` line 369 starts AFTER `import frustrapy` (`t0 = time.perf_counter()` after the import). Our `_run_one` timer at line 213 starts before `compute_frustration` (which includes PDB parse). Both measure "wall-clock of equivalent operation = calculate_frustration call after warm imports". ✓ |
| Same hardware? | ✗ NOTE | Frustrapy on VM EPYC Linux; ours on Windows host. CPU-vs-CPU comparison column biased; GPU-vs-CPU comparison is the headline and is fair in the "what users see" sense. |

## GPU benchmark rigor

| Check | Verdict |
|---|---|
| Warmup before first timed run? | ✓ (once per process, line 196-211) |
| `torch.cuda.synchronize()` before stop timer? | ✓ (line 226) |
| `time.perf_counter` (monotonic)? | ✓ |
| Best-of-N / median? | ✗ single run only |
| Peak VRAM via `max_memory_allocated`? | ✓ (line 241) |
| Reset peak stats before each run? | ✓ (line 190) |

## Numerical claims verified

- **FI Spearman ≥ 0.9975 on 30 (PDB, mode) combos**: claim is plausible
  based on Phase 4's prior LAMMPS-reference results, but `phase5_spearman.csv`
  doesn't contain these numbers — the CSV has 3 empty rows. The MD table
  is not backed by the artifact it should be backed by. **Re-run the
  validation pass with both CPU and GPU in the same process invocation
  to populate the cache, then re-export the CSV.**

- **CPU↔GPU "bit-exact"**: from the panel CSV, decoy_mean agrees to
  ~14-15 sig figs between CPU and CUDA (e.g. 5AON: -1.2442272905621115
  vs -1.2442272905621117 — final ULP differs). Not literally bit-exact;
  agrees to machine precision modulo reduction-order. FrstState counts
  ARE bit-identical (discrete classifications). Reword the README from
  "bit-exact" to "agrees to machine precision (1e-14 max ΔFI)".

- **4PKN at 8689 residues, 17.6 s**: defensible as compute time. The
  "22.4 GB VRAM paged through shared RAM" framing is **wrong** — see M3.
  Real story: max_memory_allocated is an allocator high-water mark that
  over-counts when transient peaks are released between allocations.
  Reframe as "peak allocator high-water-mark = 22 GB; effective resident
  set is much smaller; verify with `nvidia-smi` during run".

- **No LAMMPS reference for 4PKN**: correctly disclosed. The fallback
  validation ("FrstState distribution is sensible") is weak but
  unavoidable — generating a LAMMPS dump for 8689 residues is hours of
  CPU. Acceptable for a stress test.

## Things to flag in README

1. **Hardware comparison disclaimer**: "Frustrapy CPU runs on a Linux
   EPYC VM, our GPU runs on RTX 4070 Windows host. Speedup is wall-clock
   ratio of the operation a user would invoke; not normalized for
   per-core hardware." Move the cross-CPU 2-10× column to a separate
   "additional note" rather than the main speedup table.

2. **"Bit-exact" → "machine-precision" reword**: `max |ΔFI|` is below
   1e-14 across 30 combos; the configurational-mode decoy_mean differs
   in the last ULP between CPU and CUDA. FrstState counts are
   bit-identical because they're discrete.

3. **4PKN VRAM caveat**: do NOT claim "22 GB paged through shared
   system RAM" — that's a measurement artifact of
   `max_memory_allocated`, not a real WDDM paging event. If it were
   truly paging, 17.6 s would be impossible. Reframe as
   "PyTorch allocator high-water-mark; effective resident set
   substantially smaller. α-chunking (Phase 6) can reduce peak below
   8 GB."

## Recommended Phase 6 prep

1. **Re-run the Spearman validation pass** in a single process
   invocation (don't use `--skip-cpu` or `--skip-gpu`) so the cache
   populates, regenerate `phase5_spearman.csv` with real numbers, and
   verify the MD table matches the CSV. This is the load-bearing
   evidence artifact.

2. **Add best-of-3 median for GPU timings on N<300 PDBs** — would
   tighten the noisy small-N rows (the 5AON 70 ms / 11BG 206 ms /
   1O3S 159 ms) which currently could swing ±15%.

3. **α-chunking on auxiliary tensors for N>4000** — already listed in
   `optimization_opportunities.md` per the MD. Phase 6 should keep peak
   `max_memory_allocated` < 8 GB on 4PKN to make a Linux-portable
   claim, since on Linux 22 GB on a 12 GB card WOULD OOM.

4. **Verify 4PKN VRAM with `nvidia-smi` during run** — confirm whether
   resident set is ~6 GB (allocator over-counting) or ~22 GB (actual
   paging). The 17.6 s wall-clock strongly suggests the former. Add a
   one-line `nvidia-smi --query-gpu=memory.used --loop=2` log during the
   stress run.

5. **Density-Spearman documentation**: the LAMMPS-compat-flags
   (`include_dna=True`, `lammps_compat_altloc=True`) story is already
   in `lammps_compat_fixes.md` but is buried as a footnote in the
   Phase 5 MD. Promote the density-Spearman story to its own subsection
   with both default-flags and compat-flags numbers side-by-side.

6. **CI gate** (already noted in MD): lift `_run_one` into a 5-minute
   CPU-only smoke test asserting FI Spearman ≥ 0.997 on 4 PDBs. Good
   regression guard.
