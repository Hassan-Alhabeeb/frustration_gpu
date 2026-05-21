# QA-4: Test suite review

Reviewer: Opus 4.7. Scope: all `tests/test_*.py` files. 187 tests collected,
~25-60 s wall-clock.

## Verdict

YELLOW — the suite is genuinely good at gating the headline math (decoy
stats, FI Spearman, byte-exact LAMMPS column comparisons, charge vectors,
NaN-safety), but it has 1 CRITICAL "claims-byte-exact-but-isn't" check, 3
HIGH coverage gaps on public-API surface, and 4 MEDIUM flakiness/fairness
risks that will bite later. No test is outright vacuous (no `assert True`
junk), but several quantitative gates are looser than their docstrings
claim.

## Counts

| Severity | Count |
|----------|-------|
| CRITICAL | 1     |
| HIGH     | 5     |
| MEDIUM   | 6     |
| LOW      | 4     |

## Critical (1)

### C1. `test_emit_tertiary_dat_byte_diff_against_lammps` claims byte-exact comparison but only checks header lines + row count

`tests/test_frustration.py:646-683`. Docstring says "How small is the text
diff between our raw dump and LAMMPS' for 5AON configurational? Document
the answer; we expect ~3% RNG-floor difference in the decoy stats columns
+ exact match elsewhere." The body emits a dump, reads both files, and
then only asserts:

```python
assert ours[0] == them[0]          # header line 1
assert ours[1] == them[1]          # header line 2
assert len(ours) == len(them)      # row count
```

It NEVER iterates over the 221 data rows, never compares any columns,
never quantifies the diff that the docstring promises to "document". A
20% drift in coord columns or AA letters on every row would pass this
test. The deterministic columns (coords, AA letters, r_ij, rho, E_native)
ARE checked in `test_dump_coords_match_lammps_byte_exact` and
`test_mutational_dump_byte_exact_against_lammps_5AON` so the suite isn't
blind overall — but this specific test's name and docstring make it
look like a strong gate when it isn't. Either delete it or extend the
loop to actually compare columns.

## High (5)

### H1. `compute_rho`, `chain_segments`, `density_to_dataframe`, `compute_virtual_atoms` have no behavioural unit tests

These are in `src.__all__` and listed in `EXPECTED_DOCUMENTED`
(`test_api_docs.py:66-83`), but `grep` for their names in `tests/`
returns only `test_api_docs.py` (which checks the doc-vs-signature
contract, not behaviour). `compute_virtual_atoms` does have one test
(`test_virtual_atoms_5aon` at `tests/test_burial.py:79`) but ONLY checks
the virtual-N atom for a single residue and the NaN-at-chain-start
contract; virtual-H and virtual-C (used by the HB term, which is not
yet implemented but the helper is exported) are untested. `compute_rho`
is the lower-level kernel under `burial_density` and is used directly
by callers per the API.md — no direct call appears in any test.

### H2. `burial_energy` has no quantitative validation against LAMMPS

`tests/test_burial.py:135-141` (`test_burial_energy_5aon`) only asserts:
- output dict has 3 keys
- energy is finite
- `abs(energy) > 1e-3`

No comparison to a target number (e.g. the `V_Burial` column from
`benchmark/cpu_baseline/configurational/5AON_energy.log`), no
sign-convention check, no per-residue cross-check against the
LAMMPS dump. Burial is one of three energy terms summing to `V_total`
and is the principal driver of frustration on buried residues. The only
indirect numerical gate is via `test_pair_value_first_contact`
(`test_direct_contact.py:157-201`), which uses an INLINE 14-line
reference `_burial_pair_kcal` (`test_direct_contact.py:135-153`) — i.e.
the test re-implements burial and compares to it, which doesn't catch a
shared-bug regression. A V_Burial scalar gate against the energy.log
column is a 5-line addition that would close this hole.

### H3. No coverage for several edge-case structural inputs

The corpus tests 4 PDBs (5AON, 11BG, 1O3S, 3F9M) covering homodimer +
protein-DNA + alt-conformer. The user-listed missing edges are not
exercised:
- **Multi-model NMR PDB** — `parse_pdb` defaults / behaviour on multi-MODEL
  PDB files is untested. NMR ensembles are common input.
- **Single-residue protein** — `direct_contact_energy` and `water_mediated_energy`
  have `n=1` early-out tests; `parse_pdb` itself, `compute_frustration`, and
  the singleresidue / mutational drivers do not.
- **All-Gly protein** — virtual-atom path for missing-CB is critical;
  burial-density coords pick CA for Gly. No test constructs an all-Gly
  fixture (the closest is the polyalanine DH test at
  `test_debye_huckel.py:149-163`).
- **Empty chain / single-chain PDB with chain filter that matches nothing**
  (e.g. `compute_frustration(chain="Z")` on 5AON). No test exercises this.
- **Header-only PDB / parser failure modes** — what happens on a corrupted
  file is undocumented and untested.

### H4. `ContactContext` / `build_contact_context` are in `__all__` but have zero tests

`grep -r ContactContext tests/` → no matches. These are exported from
`src/__init__.py:16` so they're part of the public API contract.

### H5. `test_emit_postprocessed_matches_lammps_welltype_column` feeds LAMMPS' own decoy stats back in, then asserts agreement

`tests/test_frustration.py:454-511`. The test loads `(rij, rho_i, rho_j,
e_native, decoy_mean, decoy_std, fi)` from the LAMMPS dump and feeds them
into our writer, then asserts our writer produces the same Welltype +
FrstState columns. Since Welltype is a pure function of (rij, rho_i,
rho_j) and FrstState is a pure function of FI, this test ONLY exercises
our `welltype_from_contact` and `classify_frustration` (already covered
by tests at lines 171-210), plus formatting in `emit_postprocessed_pair_dat`.
It does NOT validate our pipeline's choice of decoy stats / FI — the
LAMMPS values are wired in directly. The 100% match assertion is
tautological given the prior unit tests pass. The name suggests stronger
coverage than it provides.

## Medium (6)

### M1. `test_compute_frustration_dh_byte_exact_against_lammps_5AON` skips the LAMMPS-pair comparison when the dump file is missing, with no warning

`tests/test_compute_frustration.py:294-296`. The test silently
`pytest.skip`s when `5AON_electro_4p15_tertiary_frustration.dat` is
missing. CI without the param_sweep dumps gets a green run that
"passes" 187 tests despite the headline DH byte-exact gate being
skipped. Same pattern in `test_compute_frustration_chain_filter`
(`test_compute_frustration.py:134-135`). Suggest at least one
non-skippable smoke check that the dump dir is intact.

### M2. The "3% RNG floor" tolerance is hand-tuned

`tests/test_decoys.py:70` sets `RTOL = 0.035` ("3.5% relative — slightly
above the spec's 3%"). On seed=0 5AON gives (-1.253, 0.491) target — a
seed-0 drift to (-1.296, 0.501) would still pass (3.4% on mean). The
config-decoy stats are deterministic given seed; freezing them to within
1e-9 vs a captured reference would catch any decoy-sampling regression
without depending on the LAMMPS dump. Currently a real RNG-handling
regression that happens to land within 3.5% would slip by.

### M3. `test_mutational_wall_clock_cpu_11bg` is a wall-clock gate that can flake on a busy machine

`tests/test_mutational_mode.py:192-201` asserts `< 60.0 s`. The whole
suite is supposedly ~25-60 s; a single test approaching the 60 s limit is
inherently flaky under CI load or thermal throttling. The GPU variant at
:206-220 has a tighter 5 s gate. Either mark `@pytest.mark.slow` and gate
it out of default runs, or accept the flake. (`pyproject.toml:88-90`
declares the `slow` marker but no test uses it.)

### M4. `test_compute_frustration_gpu_timing_11bg_under_5s` is a flakable performance gate

`tests/test_compute_frustration.py:511-524`. Same risk class as M3 — a
5 s wall-clock gate on a GPU test is reasonable as a smoke check but
makes the suite non-deterministic when run on shared GPUs.

### M5. `test_decoy_aa_composition_tracks_protein_5aon` sets L1 < 0.05 but writes "0.02" in the docstring

`tests/test_decoys.py:168-175`. Docstring says "threshold 0.02 is
generous" but the actual `assert l1 < 0.05` is 2.5× looser. A real bias
that doubles the L1 distance would still pass. Either tighten to 0.02
(match the doc) or fix the docstring.

### M6. `tests/test_frustration.py:65-72` `_spearman` doesn't tie-correct

The simple version `np.argsort(np.argsort(a))` puts ties at distinct
ranks (instead of average rank), which artificially deflates the Spearman
ρ on FI columns that contain duplicate values. The density test file
DOES use the correct tie-handling (`tests/test_density.py:69-84`,
`_rankdata_avg`). Two different implementations across the suite is a
maintenance smell, and on protein dumps with many integer pair counts
this can pull a true 0.99 down to 0.94, landing right at the 0.95 gate.

## Low (4)

### L1. Tolerances inconsistent across files

- `test_water_mediated.py:155` uses `< 1e-3` relative
- `test_debye_huckel.py:331` uses `< 1e-12` relative
- `test_direct_contact.py:298` uses `< 1e-6` relative
- `test_burial.py:152` uses `< 1e-8` absolute

All are sensible but the lack of a project-wide tolerance constant
makes future audit harder. Suggest a `TOLS = {'cpu_gpu_rel': 1e-6, ...}`
in `conftest.py`.

### L2. No `conftest.py` — fixtures duplicated across files

Every file redefines `parsed_5aon`, `parsed_11bg`, `_has_pdb`, and a
private `_spearman` helper. A `tests/conftest.py` with module-scoped
fixtures and a shared `spearman()` would cut ~150 lines and centralise
the tie-handling fix from M6.

### L3. `tests/__init__.py` is empty (0 bytes)

Not harmful but unnecessary; pytest doesn't need it and removing it
makes the test tree non-importable as a package, which is what you want
to prevent ambiguous imports.

### L4. `@pytest.mark.slow` and `@pytest.mark.gpu` declared but never used

`pyproject.toml:88-90` declares both markers under `--strict-markers`,
but no test in the suite is tagged with either. The wall-clock tests
(M3, M4) and the GPU agreement tests across files are the obvious
candidates. Without `@pytest.mark.gpu` there's no way to do
`pytest -m "not gpu"` on a CPU-only machine — currently they fall back
to `pytest.skip(not torch.cuda.is_available())` which still goes
through full collection + parsed fixture setup.

## Coverage gaps — top 3 missing areas

1. **`compute_rho`, `chain_segments`, `density_to_dataframe`,
   `ContactContext` / `build_contact_context`** — public API surface with
   zero behavioural tests (H1, H4).
2. **`burial_energy` quantitative validation** — no LAMMPS-target
   comparison; only "is finite and nonzero" (H2).
3. **Edge-case PDB inputs** — multi-model NMR, single-residue,
   all-Gly, empty-chain-filter, corrupted/header-only files (H3).

## Tests that don't actually test what they claim

- **C1**: `test_emit_tertiary_dat_byte_diff_against_lammps`
  (`test_frustration.py:646`) advertises a byte-diff comparison but
  only checks header lines + row count.
- **H5**: `test_emit_postprocessed_matches_lammps_welltype_column`
  (`test_frustration.py:454`) tests our column writer against itself
  after feeding LAMMPS' own decoy stats in — the headline 100% match
  is tautological.
- **H2 partial**: `test_burial_energy_5aon` (`test_burial.py:135`)
  claims "Phase 1 burial pipeline" coverage but only asserts shape +
  finiteness + `>1e-3` magnitude.

## Flakiness risks — top 3

1. **M2 + M6 combined**: hand-tuned 3.5% RTOL on decoy stats + buggy
   tie-handling Spearman could produce a borderline ~0.95 Spearman that
   flaps green/red depending on which tied FI values reorder. Tightening
   the floor against a frozen reference and using the average-rank
   Spearman across both files would eliminate this.
2. **M3 (CPU mutational 11BG < 60s)**: wall-clock gate at the very edge
   of the suite's runtime budget. A single GC pause or CI thermal
   throttling event flips it red. Mark `@pytest.mark.slow` and exclude
   from default runs, or budget 120 s.
3. **`test_mutational_dump_byte_exact_against_lammps_5AON` median 8%
   drift floor** (`test_frustration.py:819-823`). Floor of 0.08 on
   `np.median(dm_rel)` is empirical; a future change to `torch.rand`
   semantics across PyTorch versions could push the median to 0.09
   without anything physical changing.

## `test_api_docs.py` robustness check

The regression-test at `tests/test_api_docs.py:1-343` does what it
claims: ast-parses every `python` code block in `docs/API.md`, locates
the matching `src.<name>` function, asserts every documented kwarg
exists with the documented default. Resolves symbolic defaults
(`RHO_MIN_SEQ_SEP`, `torch.float64`) via `src.parameters`. The
parametrise spans all DOC_SIGS keys, so adding a new documented function
auto-extends the gate. Two minor caveats:

- `test_expected_functions_are_documented` (line 310) uses a custom
  `APIDocsCoverageWarning` subclassing bare `Warning` to dodge the
  `filterwarnings = ["error::UserWarning"]` rule. Working as intended,
  but means the soft-warning is only visible with `pytest -v` — easy to
  miss in CI logs.
- `_defaults_match` (line 237-266) has three fallback strategies. The
  third (`repr(live_value) == doc_literal`) makes the test mildly
  tolerant of formatting drift (`torch.float64` vs `torch.float64`),
  which is reasonable. No issue.

This test is the strongest single piece of plumbing in the suite.

## Recommendations (prioritised)

1. **Fix C1**: extend `test_emit_tertiary_dat_byte_diff_against_lammps`
   to iterate row-by-row, or delete it and rely on the dedicated
   `test_dump_coords_match_lammps_byte_exact`.
2. **Close H1/H4**: add behavioural unit tests for `compute_rho`,
   `chain_segments`, `density_to_dataframe`, `ContactContext`.
3. **Close H2**: add a V_Burial scalar gate against
   `5AON_energy.log` / `11BG_energy.log`.
4. **Close H3**: add an all-Gly synthetic and a multi-MODEL fixture;
   tighten n=1 / empty-chain coverage.
5. **Fix M6**: switch `test_frustration.py:_spearman` to use the
   tie-correcting `_rankdata_avg` from `test_density.py`; move both into
   a shared `tests/conftest.py` per L2.
6. **Mark slow/gpu**: tag M3, M4, and the wall-clock tests so they can
   be excluded from default runs (`pytest -m "not slow"`).
7. **Freeze decoy stats** against a captured reference (M2) — replace
   `RTOL = 0.035` with an exact match to a stored `(seed=0, mean, std)`
   tuple for each panel PDB.
