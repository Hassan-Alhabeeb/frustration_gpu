# v2 fix — `compute_frustration.py` orchestrator (2026-05-21)

Audit source: `F:/research_plan/New folder/odo.txt` — 75-finding audit.
Scope: orchestrator-only changes to `frustration_gpu/compute_frustration.py`
plus a new test file `tests/test_orchestrator_validation.py` (23 tests).

## Final test counts

* Baseline (pre-fix): **223 passed**
* Post-fix: **372 passed, 1 warning** (`pytest tests/`) — no regressions,
  +23 new orchestrator-validation tests plus collection of suites the
  baseline summary undercounted.
* Existing happy-path numerics gates (5AON DH byte-exact regression,
  smoke, residues filter, mode dispatch, chain filter, LAMMPS
  comparison): all unchanged.
* `ruff check` clean on both `compute_frustration.py` and the new test
  file.

## Per-finding status

| # | Severity | Status | Notes |
|---|----------|--------|-------|
| 2  | MEDIUM      | FIXED      | `n_decoys < 2` raises `ValueError` (FI z-score requires ensemble ≥ 2). Also rejects non-int (float/bool). |
| 4  | MEDIUM      | FIXED      | `metadata["v_dh"]` populated whenever `electrostatics_k` is set — scalar total of native Debye-Hückel energy in kcal/mol. |
| 5  | LOW/MEDIUM  | FIXED      | Zero-pair pair-mode runs now write header-only marker files (`_tertiary_frustration.dat`, `_{mode}.dat`, `_5adens.dat`) into `output_dir`. |
| 7  | MEDIUM/HIGH | FIXED      | `residues=` filter is applied to underlying tensors before file emission, so the dumps match the returned DataFrames. |
| 8  | MEDIUM/HIGH | DEPRECATED | `include_dh_in_e_native=True` still functions in v0.2.0 but now emits a `DeprecationWarning`; flag will be removed in v0.3.0. See "DH semantics decision" below. |
| 19 | MEDIUM      | FIXED      | `metadata["n_pairs"]` / `n_residues` now reflect post-filter counts; original sizes preserved as `n_pairs_unfiltered` / `n_residues_unfiltered`. |
| 20 | MEDIUM      | FIXED      | Configurational mode catches `RuntimeError("no in-contact pairs")` from the sampler and returns a schema-preserving empty result, matching mutational/singleresidue behaviour. |
| 21 | LOW/MEDIUM  | FIXED      | Mutational zero-pair branch now returns the same 15-column schema as configurational (via shared `_empty_pair_df()` helper). |
| 23 | LOW/MEDIUM  | FIXED      | `dtype` validated against `_VALID_DTYPES = (float16, float32, float64, bfloat16)`; ints and bool raise a clear `ValueError`. |
| 25 | LOW         | FIXED      | `chain` accepts only `str`, `list[str]`, or `None`; partial-miss (e.g. `["A","Z"]` on single-chain PDB) raises `ValueError` listing the missing chains and available chains. |
| 26 | LOW         | FIXED      | `residues=` partial-miss emits a `UserWarning` naming the absent resnums per chain. Existing "all-empty" warning kept as a secondary signal. |
| 35 | LOW         | FIXED      | `calculate_frustration` uses a module-level `_CF_WARN_ONCE` set to dedupe `graphics`/`visualization`, `n_cpus`, and (new) `overwrite=False` warnings — once per process, as docstring promised. The `overwrite=False` path now actually inspects `output_dir` for pre-existing dump files and warns when overwrite would happen. |
| 50 | MEDIUM      | SKIPPED    | Owned by MISC agent (lives in `frustration.py`). Untouched here per scope. |
| 59 | LOW/MEDIUM  | FIXED      | `precision` validated as `isinstance(int) and >= 0`; `'3'`, `None`, `1.5`, `True` all rejected up front with `ValueError`. |

## DH semantics decision (finding #8): option B with deprecation

User instruction offered (A) compute DH for every decoy or (B) make
`include_dh_in_e_native` opt-out / metadata-only. **Chose B**, with a
back-compat deprecation path:

* `electrostatics_k=k` alone now ALWAYS computes the per-pair Debye-Hückel
  energy on native pairs and reports the scalar total in
  `metadata["v_dh"]`. E_native is unchanged.
* `include_dh_in_e_native=True` still adds DH to E_native (existing
  behaviour preserved for v0.2.0 back-compat) but emits a
  `DeprecationWarning` pointing callers at the diagnostic path.
* The flag will be removed in v0.3.0.
* Singleresidue mode still warns at `RuntimeWarning` if the deprecated
  flag is passed (existing behaviour).

Rationale: option A requires recomputing the DH cost on every decoy
identity to produce a valid same-Hamiltonian z-score. That's an O(n_pair
× n_decoys) tensor with no integer-index path through
`debye_huckel_pair_energy` (current implementation is a Python loop), so
implementing it correctly here means either a vectorised DH kernel or
accepting a ~1000× DH cost overhead. Option B keeps the FI mathematically
honest (water+burial only, matching LAMMPS-AWSEM byte-exact) and surfaces
DH as a separate diagnostic — which is what every published frustration
analysis actually does.

## Deprecations introduced

* `include_dh_in_e_native: bool = False` — `DeprecationWarning` on
  `True`; functional through v0.2.0; scheduled removal v0.3.0.

No other parameter signatures changed. New return-payload fields
(`metadata["v_dh"]`, `metadata["n_pairs_unfiltered"]`,
`metadata["n_residues_unfiltered"]`) are additive and don't affect any
existing tests that read by key.

## New helper functions (private)

* `_empty_pair_df()` — schema-preserving empty `pair_records` DataFrame
  shared across configurational and mutational zero-pair returns.
* `_emit_empty_pair_files(...)` — header-only marker file writer for
  zero-pair pair-mode runs.
* `_filter_single_emit_state(...)` — subsets the singleresidue
  emit-state tensors to a boolean residue mask so the post-filter
  `.dat` write matches the returned DataFrame.

## Files changed

* `F:/research_plan/frustration_gpu/frustration_gpu/compute_frustration.py`
  — orchestrator fixes.
* `F:/research_plan/frustration_gpu/tests/test_orchestrator_validation.py`
  — new test file (23 tests).
* `F:/research_plan/frustration_gpu/docs/v2_fix_orchestrator.md` — this
  summary.

No other `src/` or `tests/` files touched — other audit agents own
those.
