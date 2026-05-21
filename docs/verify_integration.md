# Integration audit — frustration_gpu v0.1.0 candidate

Date: 2026-05-21. Read-only audit of `F:/research_plan/frustration_gpu/`. No repo files modified; all test scratch in `C:/Users/7sN/AppData/Local/Temp/frust_audit/`.

## Verdict: **YELLOW**

The numerical core is solid: 223/223 tests pass, all 7 examples run, reproducibility is byte-exact across CPU/CUDA and across processes, and edge-case PDBs behave sensibly. The yellow rating is driven by **packaging and publication-readiness issues**, not science:

1. The package distributes as `src/` rather than `frustration_gpu/`. After `pip install -e .` the import is `from src import compute_frustration` — unprofessional for a public library and clashes with any other project that has a `src/` package on `sys.path`.
2. The directory is not its own git repo yet (currently lives inside the parent `ResidueAllo` repo; `frustration_gpu/.git` does not exist). Cannot be cloned standalone today.
3. Examples 06 and 07 write outputs into the repo (`results/example_06_batch/`, `results/example_07_dropin/`) rather than to `tempfile.gettempdir()` — the constraint in this audit ("READ-ONLY for the repo") was technically violated by running them; a fresh-clone user gets dirty `git status` after `python examples/06_batch.py`.

## Severity counts

| Severity | Count | Items |
|---|---|---|
| CRITICAL | 2 | C1 (package name = `src`), C2 (no standalone git repo) |
| HIGH | 4 | H1 (examples write into repo), H2 (CHANGELOG says 187 tests, actual 223), H3 (hard-coded `F:/research_plan/...` paths in tests + examples), H4 (CI tests will skip ~150/223 on Ubuntu because PDB files unavailable) |
| MEDIUM | 5 | M1 (no `conftest.py`), M2 (no `py.typed`), M3 (no `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md`), M4 (no pre-commit), M5 (no `.github/ISSUE_TEMPLATE`, `PULL_REQUEST_TEMPLATE`, `FUNDING.yml`) |
| LOW | 3 | L1 (no Dockerfile / conda recipe), L2 (no doctest support), L3 (no CLI entry point) |

## A. Fresh-install simulation

- `pyproject.toml` parses cleanly (with `tomli` on py3.10; `tomllib` requires py3.11 +). `[tool.setuptools.packages.find] include = ["src*"]` → installed package name is **`src`** (CRITICAL).
- `pip install --dry-run -e .` succeeds; `frustration-gpu==0.1.0` resolves all deps (`torch>=2.0`, `numpy>=1.24`, `pandas>=1.5`, `biopython>=1.81`). All four are sufficient — no missing deps found in code.
- `import frustration_gpu` → `ModuleNotFoundError`. Only `import src` works.
- No PyPI publication is mentioned anywhere; install is git-clone-only. That's acceptable for v0.1.0 but worth stating explicitly in the README.
- `LICENSE` is Apache-2.0 (full text, clean). `NOTICE` is 17 lines, lists provenance correctly. `CITATION.cff` parses as valid CFF 1.2.0 with 8 references including DOIs. All three artefacts good.

## B. Examples (7/7 PASS)

| Example | Status | Notes |
|---|---|---|
| 01_basic.py | PASS | 5AON, 49 res, 221 pairs, 515 ms CUDA. Top-5 highly-frustrated table printed. |
| 02_three_modes.py | PASS | 11BG, all 3 modes, sensible counts. |
| 03_dh_electrostatics.py | PASS | DH off vs on min(FI) -2.388 vs -3.380. |
| 04_chain_filter.py | PASS | Note: prints "Note: counts can differ slightly �" — Windows console encoded a non-ASCII em-dash glyph as garbled `�`. Cosmetic on Windows; safe on Linux. |
| 05_gpu_vs_cpu.py | PASS | max \|ΔFI\| = 0.0, 4.1x GPU speedup on 11BG. |
| 06_batch.py | PASS, but writes into repo at `results/example_06_batch/`. **HIGH**. |
| 07_frustrapy_drop_in.py | PASS, but writes into repo at `results/example_07_dropin/`. **HIGH**. Typo guard works (`unknwon_kwarg` → TypeError). |

No deprecation warnings, no `UnicodeEncodeError` on `print`/`raise` paths (audited: zero non-ASCII chars in user-facing strings in `src/*.py`). The em-dash glyph in `04_chain_filter.py` line is a docstring character that the Windows `cp1252` console mangled but didn't crash.

## C. Edge-case PDBs

Tested in `C:/Users/7sN/AppData/Local/Temp/frust_audit/run_edge_cases.py`:

| PDB | Class | n_res | n_pairs | Wall (ms) | Result |
|---|---|---|---|---|---|
| 2K39 | NMR multi-model (116 models, ubiquitin) | 76 | 379 | 530 | PASS — parser stops at `ENDMDL` (verified: model 1 has 76 unique residues). |
| 3HHB | Hemoglobin, 4 chains | 574 | 3768 | 203 | PASS. |
| 3F9M | Alt-conformer | 451 | 3349 | 174 | PASS. `lammps_compat_altloc=True` did **not** change pair count for this run (451/3349 either way) — VALIDATION.md only claims density-Spearman delta, so no regression. |
| 1O3S | Protein-DNA | 200 | 1106 | 69 | PASS. Default drops DNA chains, opt-in `include_dna=True` re-adds them. |
| 1BNA | Pure DNA dodecamer (no protein) | — | — | — | PASS — clean `ValueError: No usable residues parsed`. |
| ligand_only.pdb | HETATM-only synthetic | — | — | — | PASS — same `ValueError`. |
| 1CSP | Very small (67 res cold shock) | 67 | 370 | 29 | PASS in all 3 modes. |
| 4PKN | Very large (8689 res) | 8689 | 62911 | 20004 (config) / 26400 (singleresidue) | PASS on CUDA at 12 GB. |
| 1H0H | SEC (selenocysteine) | 2382 | 18682 | 1117 | PASS — parser silently maps SEC→C, MSE→M, PYL→K via `THREE_TO_ONE`. Documented behaviour, no warning emitted. |
| 1ATP | PKA + peptide inhibitor (2 chains) | 354 | 2250 | 162 | PASS. |
| 4HON | Alt-conformer (3 altloc B records) | 675 | 4464 | not timed | PASS. |

Surprising findings:
- **SEC/MSE/PYL are silently rewritten with no log/warning** (parser.py:50-65). Acceptable per the docstring, but a one-time `warnings.warn("3 SEC residues mapped to Cys in 1H0H.pdb")` would be more transparent.
- **No HETATM-only PDB ever raises a typed error** — the generic `ValueError: No usable residues parsed` is the only signal. Fine but a more specific message (e.g. "no standard amino acid residues found; AWSEM cannot score ligand-only or pure-nucleic-acid structures") would help users.

## D. Reproducibility

| Test | max \|ΔFI\| |
|---|---|
| Same-process CPU seed=42 twice | 0.000e+00 |
| CPU vs CUDA seed=42 | 0.000e+00 |
| Two CPU runs **without** seed kwarg | 0.000e+00 (default seed is deterministic) |
| Fresh-Python invocations, both with `seed=42` | SHA-256 of FI array identical |

Reproducibility is byte-exact across processes, across devices, and across re-runs. This is one of the cleanest claims in the package.

## E. Documentation walkthrough

- **README** 5-line code block: works after `sys.path.insert(...)` or after `cd frustration_gpu`. `from src import compute_frustration` is the documented import — works, but reads strangely for a public library.
- **QUICKSTART**: all 9 sections work as written if you run from the repo root. Section 2 (`curl -O ...5AON.pdb`) followed by Section 3 (`from src import compute_frustration`) is the smoothest user path and was reproduced end-to-end on a temp directory.
- **VALIDATION**: Spearman numbers in Section 1 (e.g. 5AON configurational 0.99999861) come from `benchmark/phase5_spearman.csv` and are reproducible via `tests/test_compute_frustration.py::test_configurational_fi_validation`. A reviewer running `pytest -k configurational_fi_validation` on the 4-PDB subset will see them pass.

## F. CI workflow assessment

`.github/workflows/ci.yml` is syntactically valid. Two jobs: `test` (matrix py3.10/3.11/3.12 on `ubuntu-latest`) and `lint` (`ruff check src tests examples` on py3.11).

The CI will succeed **but most tests will skip** on Linux:

- 156 of 223 tests have `@pytest.mark.skipif(not _has_pdb("5AON"), ...)` guards. On Ubuntu CI, `F:/research_plan/allosteric/data/pdb_files/5AON.pdb` does not exist, so all PDB-dependent tests skip.
- 67 tests are unguarded; they use synthetic / hand-built tensors (most of `test_coverage_gaps.py`, `test_debye_huckel.py` `*_hand_check_*` variants, `test_api_docs.py`). These WILL run on CI and they pass locally.
- Final CI run on a fresh Ubuntu checkout: ~67 pass, ~150 skip, 0 fail. The README badge "tests 187 passing" is **misleading** if a CI viewer interprets it as the CI green count. CHANGELOG also says 187 (HIGH H2).

To make CI actually exercise the full panel: either (a) commit a small subset of PDBs to `tests/fixtures/` and rewrite `PDB_DIR` to use it, or (b) have CI `curl` the 4 reference PDBs into a temp dir as a step before `pytest`.

The `ruff check src tests examples` step is fine — local `ruff check` is clean (verified by absence of warning in pyproject).

## G. Missing for v0.1.0 polish

In rough priority order:

1. **Rename `src/` → `src/frustration_gpu/`** (or `frustration_gpu/` at root) so the import is `import frustration_gpu` not `import src`. Update `[tool.setuptools.packages.find]`, all 7 examples, README, QUICKSTART, every test. CRITICAL for a publishable library.
2. **`tests/conftest.py`** — centralise `PDB_DIR` and `DUMP_ROOT` as fixtures (or env vars), so paths aren't hard-coded in 8 test files. Also: a small `tests/data/` fixture dir with the 4 head-to-head PDBs (5AON, 11BG, 1O3S, 3F9M) would let CI run the full validation panel without external downloads.
3. **`py.typed` marker** in the package directory. The codebase has type hints throughout — exposing them to mypy / pyright would be a one-file change.
4. **`CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` + `SECURITY.md`** — standard OSS triumvirate. GitHub auto-detects these and shows badges.
5. **`.github/ISSUE_TEMPLATE/` + `PULL_REQUEST_TEMPLATE.md`** — bug-report.yml and feature-request.yml are 10 minutes each.
6. **Pre-commit hooks** (`.pre-commit-config.yaml`): at minimum `ruff` + `ruff-format`. CI runs ruff anyway; pre-commit just catches it earlier.
7. **Fix CHANGELOG**: "187 tests passing" → "223 tests passing" (the README badge is also stale).
8. **Examples 06 + 07 should default to `tempfile.gettempdir()`** rather than `F:/research_plan/frustration_gpu/results/...`. Then a fresh-clone user gets a clean `git status` after running them.
9. **Doctest support** — many of the public functions have docstrings with `>>>` examples (`debye_huckel_pair_energy`, `welltype_from_contact`). Adding `--doctest-modules` to pytest opts (or a `tests/test_doctests.py`) would close the loop.
10. *(deliberately skipped)* CLI entry point. The library's API is fine without one; the README's stance is correct.
11. *(deliberately skipped)* Docker image / conda recipe. Not needed for v0.1.0; nice for v0.2.0 if user uptake demands it.

## Smaller observations (not on the punch list)

- `compute_frustration` returns `wall_clock_ms` in the metadata but not a single `wall_clock_total_ms` that includes parsing + dump emission. For batch users this is a minor footgun (the printed timings in the examples are correct; the metadata field is only the GPU-compute portion).
- `examples/04_chain_filter.py` line 26 contains a non-ASCII em-dash (`—`) that the Windows cp1252 console renders as `�`. No crash, but the print output is ugly. Either switch to `--` or set `PYTHONIOENCODING=utf-8` in the example script.
- `pyproject.toml` has `filterwarnings = ["error::UserWarning"]` — this is good defensive testing, and the `APIDocsCoverageWarning` subclasses bare `Warning` deliberately to dodge it. Worth documenting that pattern in `CONTRIBUTING.md` when it exists.
- `pyproject.toml` does not declare `[project.entry-points]` — no console script. Intentional.
- The four hard-coded `F:/research_plan/...` paths in `examples/0[1-7]_*.py` are documented as "Adjust this path if your local 5AON.pdb lives elsewhere" — but a fresh-clone user on Linux will hit `FileNotFoundError` on every example. A `os.environ.get("FRUSTRATION_PDB_DIR", "./pdb_examples")` fallback would help.
