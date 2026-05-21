# v0.2.0 docs/benchmark/tests fix pass — summary

Date: 2026-05-21. Scope: findings #63–#75 (medium / low / docs / tests
/ benchmark severity) from the 75-finding audit at
`F:/research_plan/New folder/odo.txt`. No `frustration_gpu/*.py` source
file was modified (source ownership belongs to other agents). Test
baseline before: 223 passing; after: 372 passing (+149 from the new
parametrised api-docs reverse-check and the 7 example smoke tests).

## Files modified

* `README.md` — accuracy claims (#70, #72)
* `QUICKSTART.md` — example-CI claim + ULP wording (#70, #71)
* `VALIDATION.md` — accuracy claims, reproduction, bundled-vs-archived (#65, #70, #72, #75)
* `CHANGELOG.md` — new `[Unreleased]` block under Keep a Changelog
* `docs/API.md` — chain type, calculate_frustration kwargs, src→frustration_gpu, gamma.dat rows, chain-list semantics (#67, #68, #69)
* `docs/verify_api.md` — historical-snapshot banner (#73)
* `docs/verify_config.md` — historical-snapshot banner (#73)
* `benchmark/run_phase5.py` — PDB resolver chain, mode-typo rejection, Spearman cache repair (#63, #64, #74)
* `benchmark/compare.py` — implemented as panel-CSV diff tool (#66)
* `benchmark/phase5_results.md` — top-of-file note + archived/committed split (#65)
* `tests/test_api_docs.py` — broadened reverse-check + signature-parser fix
* `tests/test_examples_smoke.py` — NEW, smoke-tests all 7 examples (#71)

## Per-finding status

| # | Status | Fix |
|---|--------|-----|
| #63 reproducibility | Done | `--pdb-dir` CLI + `FRUSTRATION_PDB_DIR` env + bundled `tests/data/` fallback chain in `_pdb_path`. |
| #64 silent CSV clobber | Done | `phase5_spearman.csv` preserved across runs; auto-rerun for completed timing rows missing from the in-process cache; explicit row count printed. |
| #65 committed CSV vs README | Done | Top-of-file note on `phase5_results.md` declares committed numbers (11BG only) vs archived 10-PDB / 4PKN runs. README/VALIDATION call out the same. |
| #66 compare.py placeholder | Done | Working `argparse` CLI that diffs two `phase5_panel_results.csv` files; flags status flips and ≥2× slowdowns; non-zero exit on regression. |
| #67 reverse-check too narrow | Done | `test_no_undocumented_kwargs_anywhere_in_api_docs` parametrised across every API.md-documented function; explicit `DOC_DRIFT_TOLERATED` allowlist with inline justifications; `_x` private kwargs excluded; signature-parser bug fixed (was swallowing `):` when last line had an inline comment). API.md `chain` widened, `overwrite`/`n_cpus` added. |
| #68 chain semantics wrong | Done | API.md now describes chain-list as parser-level filter and explains why `["A","B"]` differs from running `"A"` then `"B"` separately. |
| #69 src/ + gamma row ranges | Done | All `src/...` → `frustration_gpu/...`; gamma.dat rows corrected to "0–209 direct, 210–419 mediated (C(21,2)=210 unordered pairs per block)". |
| #70 "literal 0.0" claim | Done | Replaced with "rounded outputs (default precision=3) match exactly; high-precision values agree to ~1e-15 ULP drift" in README/VALIDATION/QUICKSTART. Spearman-table rows updated to `0.000` (rounded). |
| #71 examples-in-CI claim | Done | New `tests/test_examples_smoke.py` subprocesses every `examples/*.py` against the 4 bundled PDBs (90 s timeout, `FRUSTRATION_OUTPUT_DIR` redirected to tmp_path). QUICKSTART claim now true. |
| #72 rejection-sampler wording | Done | README/VALIDATION mention the inverse-CDF sampler (FIX-4) and the hard `RuntimeError` when zero in-contact pairs. Historical phase-review docs left untouched (out of scope, they describe a phase where the fallback was real). |
| #73 verify_* doc tense | Done | Top banners on `verify_api.md` + `verify_config.md` flag them as historical 2026-05-21 snapshots and point readers to current `API.md` / `CHANGELOG.md`. The obsolete `overwrite`/`n_cpus` LOW findings are called out as resolved. |
| #74 --modes typo silent | Done | `argparse.error` with the offending mode name(s) and the valid set before any work starts. |
| #75 tests/data scope | Done | VALIDATION test-coverage section clarifies 4-PDB bundled vs 10-PDB archived, notes `FRUSTRATION_PDB_DIR` opt-in for the wider panel. Test-count claim ("187") replaced with the running 223+ figure plus a scope explanation. |

## Decisions made

* **`compare.py`**: implemented as a working diff tool rather than deleted.
  Argument is "the placeholder was confusing but the use case (compare
  baseline vs candidate phase5 CSVs) is real and ~150 lines of code".
* **Examples in CI**: implemented the smoke test. Argument is "the
  QUICKSTART claim that examples are run by tests should be made true,
  not retracted".
* **`docs/verify_*.md`**: chose the banner-as-historical-snapshot route
  rather than rewriting. Argument is "they accurately describe a
  point-in-time audit and are useful as audit trail; updating them
  in-place would lose that signal. Banner makes the snapshot status
  explicit, links to the current source of truth."

## Validation gates

* `pytest tests/ -v` — 372 passing (was 223). The +149 delta is +7 example
  smoke tests and +142 from the parametrised `test_api_docs.py` (each
  documented function now has a forward-drift and a reverse-drift case).
* No "literal 0.0" / "rejection-sampler fallback" phrases remain in
  README/VALIDATION/QUICKSTART (grep clean).
* `benchmark/run_phase5.py --help` and `benchmark/compare.py` both run
  cleanly on Windows CP1252 (Unicode hazards removed from help text).
* `--modes config,bogus` exits 2 with a clear error listing the valid
  modes.

## Known follow-ups (not in this scope)

* `emit_5adens_dat` and `chain_segments` still trigger the soft
  `APIDocsCoverageWarning` because their API.md sections have no
  defaulted kwargs to parse (the test deliberately ignores no-default
  signatures). Either add a defaulted kwarg in the docs or remove the
  names from `EXPECTED_DOCUMENTED` to silence. Out of scope here.
* `PHASES_ROADMAP.md` and `docs/phase_3a_review.md` still mention the
  pre-FIX-4 rejection sampler. Those are historical phase logs; not
  rewriting them here.
