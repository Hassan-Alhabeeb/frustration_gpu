# Changelog

All notable changes to frustration_gpu will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-05-21

This release closes 75 findings from a heavy code audit covering correctness,
input validation, sparse-mode accuracy, and reproducibility. Pre-1.0 versioning:
this is technically a patch number but contains documented breaking changes
(flagged with **BREAKING** below). Migration notes inline.

### Added
- `tests/test_examples_smoke.py` — smoke-tests every `examples/*.py`
  script end-to-end against the bundled 4-PDB panel
  (5AON, 11BG, 1O3S, 3F9M). Adding a new example automatically picks
  up a smoke test. Closes the QUICKSTART claim that "examples are run
  by the test suite".
- `benchmark/run_phase5.py` now accepts a `--pdb-dir` CLI argument
  and a `FRUSTRATION_PDB_DIR` env var, mirroring `tests/_paths.py`.
  Resolution order: CLI → env → bundled `tests/data/` → legacy
  developer path. First directory containing `<pdb_id>.pdb` wins.
- `benchmark/compare.py` is now a working tool: diffs two
  `phase5_panel_results.csv` files into a Markdown report and exits
  non-zero on regressions (status flips, > 2x slowdowns, missing
  rows). Previously raised `NotImplementedError`.
- `tests/test_api_docs.py` gained
  `test_no_undocumented_kwargs_anywhere_in_api_docs`, which
  reverse-checks **every** documented function (not just
  `compute_frustration`) for code-but-not-docs kwarg drift. Closes
  the "ship a kwarg without documenting it" gap.
- ORCHESTRATOR: `metadata["v_dh"]` — scalar Debye-Hückel energy on
  native pairs, populated whenever `electrostatics_k` is set. Lets
  callers report the DH contribution without re-running the term.
- ORCHESTRATOR: `metadata["n_pairs_unfiltered"]` and
  `metadata["n_residues_unfiltered"]` — original counts before the
  `residues=` filter is applied. The post-filter counts remain in
  `n_pairs` / `n_residues`.
- ENERGY: `ContactContext` and `SparseContactContext` now carry a
  `fingerprint` field (cheap tuple over `(N, device, first/last/total/
  abs-total)` of the cb_or_ca tensor). Mismatch between context
  fingerprint and live coords raises `ValueError` at every energy /
  decoy entry point.
- ENERGY: minimum-safe sparse-cutoff constants per term, exported
  from `frustration_gpu._contact_common`:
  `DIRECT_SPARSE_MIN_SAFE_A = 7.5`,
  `MEDIATED_SPARSE_MIN_SAFE_A = 14.0`,
  `dh_sparse_min_safe(screening_length, k_screening)` (returns
  `max(30, 3 * λ_eff)`). When `sparse_cutoff` is below the term's
  safe minimum a clear `UserWarning` is emitted naming the term and
  the recommended cutoff.
- ENERGY: public helpers re-exported from `_contact_common`:
  `_warn_sparse_cutoff`, `_validate_context_device`,
  `_validate_context_fingerprint`,
  `_check_residue_types_in_range`. Energy / decoy modules call
  these uniformly so the validation contract is centralised.
- PARSER: `HETATM`-encoded MSE / SEC / PYL residues are now
  accepted and promoted to their canonical (MET / CYS / LYS) amino
  acids instead of being silently dropped.
- PARSER: `OXT` is promoted to the `O` slot when a residue is missing
  its `O` record (common C-terminal convention).
- DECOYS: DNA sentinel guard at every public decoy API entry —
  `sample_configurational_decoys`, `mutational_decoy_stats`,
  `singleresidue_decoy_stats` — so negative `residue_types` cannot
  silently wrap into the `(20, 20)` / `(20, 3)` tables.

### Changed

> **BREAKING — flagged per backward-compatibility rule.** Real RCSB
> PDBs and the bundled 4-PDB panel are unaffected; pathological /
> synthetic / mis-formed inputs that previously merged silently now
> raise with a clear migration message.

- PARSER: Blank PDB chain ID is now preserved as `""`. Previously
  it was silently coerced to `"A"`, which could collide with a real
  chain A elsewhere in the file. Migration: filter explicitly by
  `chain == ""` (or pre-sanitise) if you relied on the coercion.
- PARSER: A `TER` record followed by a same-letter chain restart
  now creates separate chain segments labelled `A`, `A#2`, `A#3`, …
  rather than merging into one chain `A`. The `chain=` filter still
  matches all suffixed segments. Migration: use the parser's segment
  labels (or `chain_segments()`) when segment identity matters.
- PARSER: Two different `resname`s at the same `(chain, resnum,
  icode)` now raise `ValueError("conflicting residue names …")`.
  Previously, the first-encountered resname won and the second was
  silently dropped. Migration: clean / pre-validate the PDB upstream
  (real microheterogeneous residues use altloc, which is unchanged).
- PARSER: `END` records and a second `MODEL` block now terminate
  coordinate parsing. Previously, atoms after `END` (or in the
  second `MODEL`) entered the structure.
- PARSER: Non-finite coordinates (`NaN` / `Inf`) in an `ATOM`/`HETATM`
  record are now rejected at the line level rather than retained as
  valid residue coordinates.
- PARSER: Duplicate atom records for the same `(chain, resnum,
  icode, atom_name)` are now resolved by highest occupancy. Previously
  the first-encountered record won regardless of occupancy.
- ORCHESTRATOR: `n_decoys < 2` now raises `ValueError`. Previously
  it returned a silent `NaN` FI (std undefined with a single decoy).
- ORCHESTRATOR: `dtype` must be a floating-point dtype
  (`float16` / `float32` / `float64` / `bfloat16`). Integer / complex /
  bool dtypes are rejected with a clear error. `precision` must be a
  non-negative integer.
- ORCHESTRATOR: `chain=["A", "Z"]` on a single-chain PDB now raises
  `ValueError` if any requested chain is absent. Previously, missing
  chains were silently dropped and the run continued on the survivor.
  Partial misses (some present, some absent) still raise; the warning
  path is reserved for `residues=` partial misses.
- ORCHESTRATOR: `include_dh_in_e_native=True` is **deprecated** —
  see the *Deprecated* section.
- MISC: `classify_frustration` and `welltype_from_contact` now raise
  `ValueError` on non-finite FI input. Previously, a `NaN` FI was
  silently classified as neutral / a real well type, hiding upstream
  bugs.
- MISC: Negative `residue_types` in the writers (`emit_*_dat`) now
  raise instead of being mapped to `"V"` (the last alphabet letter
  reached by Python wrap-around).
- MISC: `parameters.load_gamma_tables` and `load_burial_gamma` are
  strict: exactly 2 columns, exactly 420 / 60 rows, no `NaN` / `Inf`
  values. Custom gamma tables that previously loaded silently with
  trailing whitespace / extra columns / missing rows will now fail
  with `ValueError("filename:line: …")`. Migration: re-export the
  table cleanly from the upstream toolchain.
- MISC: `compute_residue_density` and `compute_rho` now validate
  input shapes and length. A mismatched FI / coords length raised an
  `IndexError` (or silently broadcast) before; now it raises
  `ValueError` with explicit shapes in the message.
- ENERGY: `build_contact_context(sparse_cutoff=...)` no longer
  stores the dense `(N, N)` distance matrix on the resulting
  `SparseContactContext`. `ctx.dist` is now `None` for sparse
  contexts. Callers that consumed the diagnostic `"distances"` key
  from `return_pair_matrix=True` on sparse contexts now receive a
  1-D `(N_pair,)` `r_ij` tensor in that slot. (Dense
  `ContactContext.dist` is unchanged.)
- ENERGY: Scalar pair-energy helpers (`direct_pair_energy`,
  `water_mediated_pair_energy`) now reject custom gamma tables
  containing `NaN` / `Inf` values (in addition to the existing
  shape and amino-acid-range checks).
- ENERGY: `debye_huckel_energy` validates `screening_length` and
  `k_screening` as finite positive scalars at API entry.
  Previously, `screening_length = 0` died with `ZeroDivisionError`
  deep in the kernel; non-finite values produced `NaN` energies.
- ENERGY: A `ContactContext` built on CPU but called with
  `device="cuda"` (or vice versa) now raises `ValueError` with a
  rebuild-on-device hint. Previously the call died with an opaque
  cross-device kernel error.
- ENERGY: Non-glycine residues with a missing CB (CB → CA fallback)
  now emit a single `UserWarning` listing the affected residue
  indices. Previously the fallback was silent.
- `docs/API.md`: chain type widened to `str | list[str] | None`
  (matches live signature); `calculate_frustration` signature block
  adds `overwrite` and `n_cpus`; gamma-table loader row ranges
  corrected (rows 0–209 direct, 210–419 mediated — the previous
  "lines 1–20 / 21–60" was wrong); all `src/...` references
  rewritten to `frustration_gpu/...`; chain-list semantics clarified
  as parser-level filter (was: "run full pipeline then post-filter").
- `README.md`, `VALIDATION.md`, `QUICKSTART.md`: replaced the
  "literal 0.0 max |ΔFI| on CPU vs CUDA" wording with "rounded
  outputs (default `precision=3`) match exactly; high-precision
  values agree to ~1e-15 ULP drift". The previous wording was an
  artifact of the rounded outputs and was technically misleading.
- `README.md`, `VALIDATION.md`: removed mentions of the
  rejection-sampler fallback warning. The current sampler is
  inverse-CDF with a hard `RuntimeError` when no in-contact pairs
  exist (FIX-4, `frustration_gpu/decoys.py:425-449`).
- `README.md`, `VALIDATION.md`: the "30/30 (PDB, mode) combos"
  headline is qualified — the bundled `benchmark/
  phase5_panel_results.csv` is the **4-PDB CI subset** (5AON, 11BG,
  1O3S, 3F9M); the full 10-PDB / 30-combo panel was run on a
  development machine, the headline numbers (FI Spearman ≥ 0.9975,
  14×–53× speedup) are reproduced locally by supplying the missing
  PDBs to `benchmark/run_phase5.py --pdb-dir ...`. Bit-identity for
  the 4 bundled PDBs is hard-gated by the test suite.
- `VALIDATION.md`: test-coverage table now lists scope without
  hard-coded counts, and clarifies the "4 PDB bundled / 10 PDB
  archived" distinction so the test-count claim cannot drift.
- `benchmark/phase5_results.md`: top-of-file note flags that the
  committed `phase5_panel_results.csv` / `phase5_spearman.csv`
  contain a single (PDB, mode) baseline (11BG configurational); the
  full 10-PDB / 30-combo numbers are archived from a
  developer-machine run, reproducible by supplying the missing PDBs.
- `docs/verify_api.md`, `docs/verify_config.md`: top-of-file banner
  notes these are historical 2026-05-21 audit snapshots; the LOW
  findings about missing `overwrite`/`n_cpus` are now obsolete.
- `benchmark/run_phase5.py`: `--modes` now hard-errors on typos
  (was: silently filtered to an empty mode list).

### Deprecated
- ORCHESTRATOR: `include_dh_in_e_native=True` parameter now emits a
  `DeprecationWarning` and is scheduled for removal in v0.3.0. The
  flag was a LAMMPS-compat shim that folded the DH energy into
  `E_native` for the decoy denominator; physically, that means decoys
  see an inconsistent energy function (DH on native, no DH on decoys),
  which biases FI z-scores. Migration: drop the flag and use
  `electrostatics_k` alone — DH is now reported separately in
  `metadata["v_dh"]` while leaving the byte-comparable `E_native`
  bit-identical to LAMMPS reference dumps.

### Fixed
- DECOYS: Singleresidue CUDA OOM at `N ≥ 4000` — the per-residue
  decoy energy now flows through a chunked
  `_water_per_alpha_fused_sum` path (re-used from mutational mode),
  removing the transient `(20, N, N)` cube. Verified ~2× VRAM
  reduction at N=5000 (7.5 GB → 3.8 GB) and bit-identical FI vs the
  one-shot path on N=49 / 248 / 451.
- ENERGY: Sparse-context contact terms could be materially wrong at
  the user's natural cutoff choices (e.g. direct at 6.5 Å, DH at 11
  Å with λ=10). Each term now warns at its physically-motivated
  minimum and the warning message includes the recommended cutoff.
- ENERGY: A stale `ContactContext` (built on coords A, then used
  with coords B) silently produced wrong energies. The fingerprint
  guard now raises `ValueError("Stale ContactContext …")` at every
  energy and decoy entry point.
- ENERGY: `SparseContactContext` previously stored the full
  `(N, N)` distance matrix in `dist`, defeating the sparse-memory
  benefit. The builder is now a chunked `O(chunk × N × 3)` scan;
  `dist` is `None` by default on the sparse path.
- ENERGY: `screening_length = 0` raised a cryptic
  `ZeroDivisionError`. Now validated as finite positive at API
  entry, with a clear message.
- ENERGY: A `ContactContext` on CPU called with `device="cuda"`
  produced an opaque cross-device kernel error. The validator now
  raises `ValueError` with a "rebuild on requested device" hint.
- ENERGY: `water_mediated_energy(return_pair_matrix=True)`
  populated the diagnostic `theta` matrix with the mid-window
  sentinel value on excluded (cross-chain / sequence-skipped /
  NaN-row) pairs. Diagnostic theta is now zero on excluded pairs.
- MISC: `compute_rho` autograd produced `NaN` gradients on every
  row when any coord was `NaN`. A double-where rewrite preserves
  valid gradients on the finite rows.
- ORCHESTRATOR: `residues=` filter was applied AFTER the per-PDB
  dump files were written, so on-disk dumps contained unfiltered
  data while the in-memory DataFrame was filtered. Now filtered
  before emission for both modes.
- ORCHESTRATOR: Configurational mode crashed with
  `RuntimeError("no in-contact pairs")` on zero-contact valid
  inputs (highly extended / disordered structures). The orchestrator
  now catches and returns a schema-preserving empty result so
  downstream callers do not need to special-case the empty path.
- ORCHESTRATOR: Mutational mode's zero-pair branch returned a
  shape `(0, 0)` DataFrame missing the standard columns. Schema is
  now preserved like configurational.
- PARSER: HETATM-encoded MSE / SEC / PYL residues were dropped
  entirely. Now correctly promoted to MET / CYS / LYS.
- PARSER: `END` records and second-`MODEL` records were silently
  ignored, so post-`END` atoms entered the structure. Both now
  terminate coordinate parsing.
- PARSER: Non-finite coordinates were accepted as valid residue
  positions. Now rejected at line level.
- PARSER: Duplicate atom-name records resolved as first-coord-wins
  regardless of `occupancy`. Now resolved by highest-occupancy.
- PARSER: Virtual-atom construction (CB ↔ neighbour interpolation)
  used the wrong neighbour residue at sequence-numbering gaps,
  producing geometrically nonsensical CBs. The virtual-atom helper
  now emits `NaN` at the gap and the downstream "non-Gly CB
  fallback" warning lists the affected indices.
- `benchmark/run_phase5.py`: the default rerun after a panel sweep no
  longer clobbers `phase5_spearman.csv` with a single-row file from
  an empty in-process cache. The script now preserves the existing
  CSV and auto-refills the cache for any (PDB, mode) flagged "ok" in
  the timing CSV but missing from the in-process cache.
- `tests/test_api_docs.py`: signature parser now tolerates a trailing
  inline `# comment` on the last kwarg line (previously the closing
  `):` could be swallowed into the comment), and the
  `calculate_frustration` signature block is now discovered by the
  drift check.

### Internal
- `docs/API.md` `output_dir` comment edited from `# our name (wins
  if both set)` to `# our name; wins if both set` so the parenthesis
  in the comment doesn't break the regression test's signature
  parser.

## [0.1.1] - 2026-05-21

### Added
- ORCID identifier for author in CITATION.cff (https://orcid.org/0009-0001-4944-5567)
- Zenodo integration: GitHub releases are now archived to Zenodo, which mints a permanent DOI per release

### Fixed
- Hardware footnote in README/VALIDATION correctly identifies the local machine as AMD Ryzen 9 5900X (was "Intel CPU")

### Internal
- PyPI Trusted Publishing workflow added (publishes on every `v*` tag push, requires manual approval through GitHub `pypi` environment)
- CI: tests/_paths.py and tests/__init__.py now tracked (were accidentally excluded by gitignore in v0.1.0); ruff workflow rewired to lint `frustration_gpu/` after the src/ rename

### Added
- `NOTICE` file at repo root enumerating algorithmic provenance (Apache-2.0 §4(d)).
- Citation upgrades in `CITATION.cff`: Parra 2016 (Frustratometer 2 server), Thompson 2022 (LAMMPS), Paszke 2019 (PyTorch), Cock 2009 (Biopython), and Rausch 2021 promoted from `software` to `article` with DOI.

### Removed
- `docs/reference_lammps_awsem/` (mirrored GPL-2.0 C++ source from adavtyan/awsemmd). The citation is preserved; the source is not redistributed. Inline `fix_backbone.cpp:NNN` references in the docs now read as upstream-file citations rather than pointers into this repo.

## [0.1.0] - 2026-05-21

### Added
- Pure-PyTorch implementation of LAMMPS-AWSEM frustration analysis
- All 3 modes: configurational, mutational, singleresidue
- GPU acceleration: 14-53x speedup vs frustrapy CPU on RTX 4070
- Top-level `compute_frustration()` API + `calculate_frustration()` drop-in alias for frustrapy users
- LAMMPS-compat opt-in flags: `lammps_compat_altloc`, `include_dna`, `keep_incomplete_backbone`, `include_dh_in_e_native`
- Chain filter, residue subset filter, opt-in DH electrostatics
- Byte-comparable dump emitters (tertiary_frustration.dat, singleresidue.dat, 5adens.dat)
- 223+ tests passing on the 10-PDB validation panel (4-PDB frustrapy head-to-head subset)
- Docs: API.md, QUICKSTART.md, VALIDATION.md, lammps_compat_fixes.md, frustrapy_vs_us.md
- 7 runnable example scripts
- Phase 5 benchmark harness (`benchmark/run_phase5.py`)

### Validated
- FI Spearman >= 0.9975 on 30/30 (PDB, mode) combos vs LAMMPS reference
- CPU<->CUDA max |DeltaFI| = 0.0 (literal, FI absorbs ULP drifts in decoy_mean - verified on all 30 (PDB, mode) combos)
- Configurational FI Spearman >= 0.99999 on every panel PDB (exact when E_native has no ties; lowest observed 0.9999981)
- LAMMPS-compat flags: 1O3S density Spearman 0.224 -> 0.9992, 3F9M 0.274 -> 0.9997
- Scales to 8,689-residue 4PKN end-to-end in 17.6s on RTX 4070

### Known limitations
- Large proteins (>5,000 residues) consume significant VRAM (~22 GB allocator high-water mark on 4PKN). Alpha-chunking the auxiliary (N,N) tensors will address this; tracked in `docs/optimization_opportunities.md`.
- LAMMPS' dump precision is `%8.3f` (3 decimal places); upstream fix requires recompiling LAMMPS-AWSEM (recipe in `docs/precision_upgrade_plan.md`).
- Rejection-sampler fallback bias on sparse/fragmented structures (logged warning, FI Spearman unaffected on dense structures).
- For very small proteins (N < 100) CUDA launch overhead can make GPU slower than CPU. The 5N9R (N=356) mutational row is the one panel cell where this leaks into a larger structure; CPU mode is recommended for N < 100.

[Unreleased]: https://github.com/Hassan-Alhabeeb/frustration_gpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Hassan-Alhabeeb/frustration_gpu/releases/tag/v0.1.0
