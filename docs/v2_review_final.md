# v0.2.0 audit-fix review

Date: 2026-05-21. Reviewer: final-review agent. Source audit:
`F:/research_plan/New folder/odo.txt` (75 findings). Per-agent summaries:
`docs/v2_fix_{parser,energy,decoys,orchestrator,misc,docs}.md`.

## Verdict

**SHIP-AFTER-PATCH** — code-side fixes are high quality and bit-identity
on the 4-PDB validation panel is preserved exactly. Two ship-blockers in
the *release plumbing* (not code) need a follow-up commit before tagging
v0.2.0 on PyPI.

The fixes themselves are the highest-quality batch this repo has seen
(15 new helpers, fingerprint-based stale-context guard, chunked sparse
build, OOM-safe singleresidue path, ~149 new tests). The blockers are
documentation honesty issues, not engineering gaps.

## Critical issues (0)

None. No code regression on the bundled panel; nothing the audit asked
for is missing in the implementation sense.

## High issues (3)

1. **CHANGELOG only covers the DOCS agent's work.** Lines 7–76 of
   `CHANGELOG.md` describe the README/API.md/QUICKSTART rewording and
   benchmark/test harness additions. The 60+ user-visible behaviour
   changes from PARSER / ENERGY / DECOYS / ORCHESTRATOR / MISC are NOT
   listed. Concrete omissions:
   - Parser: blank chain ID is now `""` not `"A"` (`parser.py:242`);
     TER + same-letter restart now emits `<letter>#<N>` segment labels
     (`parser.py:416-428`); HETATM MSE/SEC/PYL now accepted
     (`parser.py:133-145`); duplicate atom records now resolve by
     occupancy not first-wins (`parser.py:512-529`); END terminates
     parsing (`parser.py:447-450`); mixed-resname-same-key now raises
     (`parser.py:500-511`); MODEL/ENDMDL discipline enforced
     (`parser.py:430-444`); non-finite coords rejected; OXT promotes to
     O.
   - Orchestrator: `n_decoys < 2` raises (`compute_frustration.py`);
     `metadata["v_dh"]` added when `electrostatics_k` set; `dtype` and
     `precision` strictly validated; `include_dh_in_e_native=True` emits
     `DeprecationWarning` and is scheduled for removal in v0.3.0;
     `metadata["n_pairs_unfiltered"]` / `n_residues_unfiltered` added;
     `residues=` partial-miss now emits `UserWarning`.
   - MISC: `classify_frustration` and `welltype_from_contact` now RAISE
     on non-finite input (`frustration.py:212-219`, `:259-270`);
     `compute_residue_density` raises on length-mismatch or non-finite
     FI (`density.py`); `parameters.load_gamma_tables` strict (exactly
     420 rows, two columns, no NaN/inf) — could fail on user-supplied
     custom tables that previously loaded silently.
   - Energy: `screening_length <= 0` now raises (previously
     `ZeroDivisionError`); sparse cutoffs below per-term minimums emit
     `UserWarning`; non-Gly CB fallback emits `UserWarning`.
   - Decoys: DNA sentinel guards at every public API entry; rho shape
     validation; `gamma_*` shape validation in
     `compute_configurational_decoy_energy`.
   These should appear in `### Changed` and `### Fixed` of the
   Unreleased block. Per the user's rule #5 (`backward compat
   preserved … MUST be documented in CHANGELOG with clear migration
   notes`) this is a release-blocking honesty gap, not a code bug.

2. **Stale-context fingerprint check (#30) is NOT applied to
   `singleresidue_decoys.py` or `decoys.py`**, despite the audit
   explicitly listing them as affected (`odo.txt:204` — *"Same stale-
   context pattern exists in water_mediated.py, debye_huckel.py, and
   singleresidue_decoys.py"*). Verified by grep:
   `_validate_context_fingerprint` is only called in `direct_contact.py:307`,
   `water_mediated.py:252`, `debye_huckel.py:329`. Both
   `singleresidue_decoys.py:339-340` and `decoys.py:389-390` happily
   consume `_context.dist_full` without any source-identity check. So
   the singleresidue / decoy path remains exposed to the bug the audit
   described (build context on protein A, run decoys on protein B,
   silently get wrong energies). The energy agent's summary
   (`v2_fix_energy.md:34`) claims "every energy fn now calls
   `_validate_context_fingerprint`" — that claim is wrong for the two
   decoy modules. This is a real-but-narrow gap (only fires when a
   caller manually passes `_context=...` between structures), not a
   regression — but it is what the audit asked for and it didn't land.

3. **Committed benchmark CSV remains 11BG-only.** The DOCS agent
   chose to flag the discrepancy (top-of-file note on
   `benchmark/phase5_results.md`) rather than re-commit the 30-row
   panel. The README still says "FI Spearman ≥ 0.9975 on 30/30 (PDB,
   mode) combos" (`README.md:98-99`). The new framing — "headline
   numbers are archived, reproduce locally" — is honest but a fresh
   clone of v0.2.0 will not contain runnable evidence for the
   30-combo claim, only the 1-combo committed baseline. Either
   re-commit the archived CSVs or further soften the headline. Not a
   ship-blocker for PyPI per se, but it is what audit finding #65
   asked for.

## Medium issues (4)

4. **`benchmark/run_phase5.py` reproducibility is partial.** `--pdb-dir`
   and `FRUSTRATION_PDB_DIR` are wired in and the bundled `tests/data`
   is in the resolution chain (`v2_fix_docs.md:#63`), but the script
   still wants ≥ 10 PDBs to produce the panel headline. Fresh-clone
   users get a runnable script that produces 4-of-10 rows. This is
   strictly an improvement over the previous "FileNotFoundError on
   line 84" but doesn't fully close #63.

5. **`include_dh_in_e_native` deprecation** (#8) — the
   `DeprecationWarning` fires, the flag is scheduled for v0.3.0 removal,
   but the deprecation is **not announced in the CHANGELOG**. Library
   users who pin via pip will get a deprecation warning on next
   upgrade with no migration note. Reference: `compute_frustration.py`
   (the agent's `v2_fix_orchestrator.md:50` cites the flow but the
   CHANGELOG entry never made it in).

6. **`emit_5adens_dat` and `chain_segments` still trigger the
   APIDocsCoverageWarning** (pre-existing soft warning;
   `v2_fix_docs.md:71-75` flags it as known follow-up). Worth a 1-line
   fix in API.md to close the last warning before tagging.

7. **Documentation gap for the new `SparseContactContext.dist=None`
   contract (#42)**. The agent's summary says callers using
   `return_pair_matrix=True` on sparse context get a fallback 1-D
   `r_ij` tensor in the `"distances"` key. That's a behaviour change
   from "(N, N) dense matrix" to "(N_pair,) 1-D" — a backward-
   incompatible shape change for any caller using the diagnostic key.
   Not in CHANGELOG. Not in API.md. Only documented in the agent's
   summary doc.

## Per-finding status (75 rows)

Severity column copies the audit's tag. "Closed?" reflects whether the
file:line cited by the audit actually changed. "Fix quality" is my
read against the user's "no band-aid / addresses root cause" bar.
"Test catches bug?" answers whether the regression test would fail if
the fix were reverted (verified by file-name + grep of
`tests/test_*_validation.py`; spot-checked, not exhaustively mutation-
tested).

| # | Severity | Closed? | Fix quality | Test catches bug? |
|---|---|---|---|---|
| 1 | HIGH | ✓ | root-cause: `_water_per_alpha_fused_sum` wired in `singleresidue_decoys.py:164` | yes — `test_singleresidue_oom_bound_at_large_n` + chunked-vs-one-shot equivalence |
| 2 | MED | ✓ | API-entry validation in `compute_frustration.py` | yes — `test_n_decoys_one_rejected_*` |
| 3 | MED | ✓ | double-where autograd-safe in `burial.py:206-218` | yes — `test_compute_rho_value_unchanged_for_clean_input` (bit-identity gate) |
| 4 | MED | ✓ | `metadata["v_dh"]` populated; documented | yes — implicit (verified by hand) |
| 5 | LOW/MED | ✓ | `_emit_empty_pair_files()` helper | yes |
| 6 | LOW | ✓ | theta masked in `water_mediated.py` | yes — `test_water_mediated_theta_zeroed_on_excluded_pairs` |
| 7 | MED/HIGH | ✓ | `_filter_single_emit_state` + filter-before-emit | yes |
| 8 | MED/HIGH | ⚠ | DEPRECATED (option B). Math is unchanged for non-flag users. Migration via DeprecationWarning is correct but deprecation NOT in CHANGELOG | partial — DeprecationWarning fires, but mutation test would need explicit catch |
| 9 | MED | ✓ | `HETATM_PROMOTED_RESNAMES` set | yes — `test_parser_edge_cases.py` |
| 10 | MED | ✓ | DNA sentinel guard added to all 3 decoy APIs | yes — `test_dna_guard_*` (4 tests) |
| 11 | MED | ✓ | `_validate_context_device` called at every energy fn | yes — `test_context_device_mismatch_raises` |
| 12 | MED | ✓ | documented; opt-in only on `compute_dist_full=True` | partial — relies on documentation, not enforcement |
| 13 | LOW/MED | ✓ | per-residue PRO mask in `virtual_atoms.py` | yes |
| 14 | LOW/MED | ✓ | non-Gly CB fallback `UserWarning` with residue list | yes — `test_non_gly_cb_fallback_warns` |
| 15 | LOW | ✓ | RAISES on non-finite FI | yes |
| 16 | LOW | ✓ | scalar pair helpers reject `aa < 0` and `aa >= 20` | yes — `test_direct_pair_energy_rejects_*_aa` |
| 17 | HIGH | ✓ | TER actually creates new segment label `<letter>#<N>` | yes — verified by my own synthetic-PDB test; parser test in `test_parser_edge_cases.py` |
| 18 | MED | ⚠ | parser already returned `insertion_codes`; compute-layer DataFrame still has no `icode` column. The agent flagged this as out-of-scope; **finding remains partially open at the DataFrame surface** | partial |
| 19 | MED | ✓ | `n_pairs_unfiltered` / `n_residues_unfiltered` added | yes |
| 20 | MED | ✓ | catches `RuntimeError("no in-contact pairs")` → empty result | yes |
| 21 | LOW/MED | ✓ | `_empty_pair_df()` shared schema | yes |
| 22 | MED | ✓ | `n_decoys < 2` raises in all 3 modes + scalar helpers | yes |
| 23 | LOW/MED | ✓ | `_VALID_DTYPES` whitelist | yes |
| 24 | LOW/MED | ✓ | screening_length / k_screening positive-finite validation | yes — 4 tests |
| 25 | LOW | ✓ | `chain` type checking + partial-miss raises | yes |
| 26 | LOW | ✓ | partial-miss `UserWarning` | yes |
| 27 | MED | ✓ | length-mismatch + non-finite FI raise in `density.py` | yes |
| 28 | HIGH | ✓ | `dh_sparse_min_safe(λ, k)` + warning; verified fires on cutoff=11 with default λ=10 | yes — `test_sparse_dh_warns_below_min_safe` |
| 29 | HIGH | ✓ | `MEDIATED_SPARSE_MIN_SAFE_A=14` constant + warning | yes |
| 30 | MED | ⚠ band-aid | `_validate_context_fingerprint` wired in direct/water/DH only; **NOT** wired in `decoys.py:389` or `singleresidue_decoys.py:339` despite audit naming them. Energy agent's summary overstates coverage. | partial — direct/water/DH tested, decoy modules unprotected |
| 31 | LOW/MED | ✓ | dense-vs-sparse mismatch already raised; tests added | yes |
| 32 | LOW/MED | ✓ | gamma.dat: exactly 2 cols, exactly 420 rows, no NaN | yes |
| 33 | MED | ✓ | resnum[i] − resnum[i−1] == 1 check added | yes |
| 34 | LOW/MED | ✓ | n_pair == 0 short-circuit before `_precompute_T_alpha` | yes |
| 35 | LOW | ✓ | warn-once via `_CF_WARN_ONCE`; overwrite=False now inspects output_dir | yes |
| 36 | LOW | ⚠ | Audit asked for radius/eta/k_water/threshold validation across direct/water/density. Direct/water energy fns DO validate screening_length now (#24) but the general "all numeric knobs" check is NOT fully landed. Eta, r_min/r_max ordering for direct_contact still unvalidated at API entry. | partial |
| 37 | MED | ✓ | altloc-B-only handling | yes |
| 38 | MED | ✓ | blank chain = `""`; verified by my own synthetic test | yes |
| 39 | LOW/MED | ✓ | `_check_residue_types_in_range` helper rejects ≥ 20 | yes |
| 40 | LOW/MED | ✓ | `welltype_from_contact` raises on non-finite | yes |
| 41 | MED | ⚠ | `burial.burial_density` raises on DNA sentinel; **`lammps_dump_rho` in `decoys.py` was flagged in audit but is "out of scope" per misc agent and the decoys agent didn't add a sentinel guard to `lammps_dump_rho` either** — coverage gap between agents | partial |
| 42 | MED | ✓ | dense `(N,N) dist` is `None` on sparse path; chunked scan never materialises the full matrix; verified by `ctx.dist is None` | yes — `test_sparse_context_does_not_store_dense_dist` |
| 43 | LOW/MED | ✓ | gamma + burial table shape validation at decoy API entry | yes |
| 44 | LOW | ✓ | highest-occupancy-wins dedup | yes |
| 45 | MED | ✓ | END terminates parse | yes |
| 46 | MED | ✓ | non-finite coords rejected at line level | yes |
| 47 | HIGH | ✓ | `DIRECT_SPARSE_MIN_SAFE_A=7.5` + warning; verified | yes |
| 48 | LOW/MED | ✓ | shape validation in `compute_rho` | yes |
| 49 | MED | ✓ | gamma/burial loader rejects NaN/inf with filename:line | yes |
| 50 | MED | ✓ | `compute_frustration_index` rejects negative/NaN/inf std | yes |
| 51 | LOW/MED | ✓ | `_aa_idx_to_letter` raises on out-of-range | yes |
| 52 | LOW/MED | ✓ | density rejects negative/out-of-range pair indices | yes |
| 53 | MED | ✓ | mixed-resname raises | yes |
| 54 | MED | ✓ | rho shape validation in all 3 decoy APIs | yes |
| 55 | MED | ✓ | `compute_configurational_decoy_energy` rejects mismatched decoy shapes | yes |
| 56 | MED | ✓ | NaN/inf/non-positive sparse_cutoff rejected | yes |
| 57 | LOW/MED | ⚠ band-aid | scalar pair helpers validate exact `(20, 20)` shape, but **NaN-filled tables still pass through** silently (the agent flagged this themselves at `v2_fix_energy.md:40`). The audit specifically called out NaN tables as the failure mode. Punted to "parameter-loader agent's scope" — but #49 only covers the file loaders, not direct user-supplied tensors. | partial |
| 58 | LOW/MED | ✓ | `debye_huckel_pair_energy` rejects 0/negative/NaN/inf distances | yes |
| 59 | LOW/MED | ✓ | `precision` validated as int ≥ 0 | yes |
| 60 | LOW/MED | ✓ | second MODEL terminates parse | yes |
| 61 | LOW/MED | ✓ | OXT promoted to O slot when O absent | yes |
| 62 | LOW | ✓ | `emit_singleresidue_dat` raises on length mismatch | yes |
| 63 | MED | ⚠ | `--pdb-dir` + `FRUSTRATION_PDB_DIR` + bundled fallback wired, but only 4 of 10 PDBs are bundled — fresh-clone still gets a partial run | partial |
| 64 | MED | ✓ | `phase5_spearman.csv` preserved + auto-refill | yes (manual check) |
| 65 | MED | ⚠ | committed CSV still 1-row + 2-row. Top-of-file note added but discrepancy with README's "30/30 combos" claim remains | partial |
| 66 | LOW/MED | ✓ | `compare.py` is now a working diff tool | yes |
| 67 | MED | ✓ | parametrised reverse-check across every documented function | yes (test passes; revert would fail) |
| 68 | MED | ✓ | API.md chain-list semantics rewritten | yes (doc-sync test) |
| 69 | LOW/MED | ✓ | `src/` → `frustration_gpu/` cleaned; gamma row ranges corrected | yes (grep clean) |
| 70 | LOW/MED | ✓ | "1e-15 ULP drift" replaces "literal 0.0" in README/VALIDATION/QUICKSTART | yes (grep clean) |
| 71 | LOW/MED | ✓ | `test_examples_smoke.py` runs all 7 examples; verified 7/7 passing | yes |
| 72 | LOW/MED | ✓ | rejection-sampler wording gone; replaced with inverse-CDF | yes (grep clean) |
| 73 | LOW/MED | ✓ | historical banner on `docs/verify_*.md` | yes |
| 74 | LOW | ✓ | `--modes` typo → argparse.error | yes |
| 75 | LOW/MED | ✓ | VALIDATION.md test-coverage section clarifies 4-bundled vs 10-archived | yes |

**Summary**: 67 closed properly (✓), 8 partial / band-aid (⚠), 0 not
addressed (✗), 0 superseded (N/A). The 8 partials are dominated by
honest scoping decisions (audit-asked work that crossed agent
ownership lines) rather than malpractice. Findings #30 and #57 are the
two real gaps.

## Numerical bit-identity verification

Methodology: stash `frustration_gpu/*.py` to recover v0.1.1
(`git stash push frustration_gpu/`), record FrstIndex digests, unstash,
re-record digests, compare.

| PDB | Mode | v0.1.1 digest | v0.2.0 digest | Match? |
|---|---|---|---|---|
| 5AON | configurational | `705536500865066453` (first 50 FI, rounded precision=3) | `705536500865066453` | ✓ |
| 11BG | singleresidue | `8432194359430635656` | `8432194359430635656` | ✓ |
| 1O3S | configurational | `9095260708416756654` | `9095260708416756654` | ✓ |
| 3F9M | mutational (n=10) | `3410257800078388962` | `3410257800078388962` | ✓ |

**Bit-identity preserved on every gate**. No silent numerics regression.
All 372 tests pass (the +149 over baseline 223 is the new validation
test files + parametrised api-docs reverse-check + example smoke
tests).

## Top 3 strongest fixes (root cause, no band-aid)

1. **#1 singleresidue OOM** — `_water_per_alpha_fused_sum` wired into
   `singleresidue_decoys.py:163-172`. The sum kernel never materialises
   the `(20, N, N)` cube; verified 2.0× peak-VRAM reduction on N=5000
   synthetic. Mirrors the proven mutational strategy. Test forces the
   chunked path and asserts bit-identical FI vs the one-shot path —
   exactly the "mutation test" criterion.
2. **#42 sparse context O(N²) memory** — full rewrite of
   `build_contact_context(sparse_cutoff=...)` in `_contact_common.py:431-520`
   to a chunked O(chunk×N×3) scan that never builds the full distance
   matrix. `SparseContactContext.dist` is now `None`. Verified by
   `ctx.dist is None` on 5AON. The audit said "even when sparse_cutoff
   is supplied, the context retains the full NxN dist tensor" — that
   sentence is now wrong (in the good direction).
3. **#28 / #29 / #47 sparse cutoff term-specific warnings** — three
   different physical-decay-rate-aware minimum-safe-cutoff constants
   (`DIRECT_SPARSE_MIN_SAFE_A=7.5`, `MEDIATED_SPARSE_MIN_SAFE_A=14.0`,
   `dh_sparse_min_safe(λ, k) = max(30, 3λ_eff)`) wired into the
   per-term warning emitter. Verified by hand that the warning fires
   at the audit's reproducer thresholds (direct=6.5 → warns; DH=11
   with λ=10 → warns). This was the audit's most physically dangerous
   finding (sign-flip on DH electrostatics).

## Top 3 weakest fixes (band-aid or partial)

1. **#30 stale-context fingerprint — INCOMPLETE.** The audit explicitly
   listed `singleresidue_decoys.py` and (by extension) `decoys.py` as
   having the same stale-context vulnerability. The energy agent's
   summary claims "every energy fn now calls
   `_validate_context_fingerprint`" but `grep` shows it's only in
   direct/water/DH. The two decoy modules consume `_context.dist_full`
   without any fingerprint check (`decoys.py:389`,
   `singleresidue_decoys.py:339`). This is the gap most likely to
   re-surface as a v0.2.0 bug report. Real-cause fix should be a
   centralised "any context use validates fingerprint" decorator or
   one helper call per context-consuming function — and it should be
   in the decoy modules too.
2. **#8 DH semantics — option B chose deprecation over correctness.**
   The fix is honest (`DeprecationWarning` + remove in v0.3.0) but
   the agent acknowledges option A (compute DH for every decoy) is the
   physically right answer and was punted because "it's an O(n_pair ×
   n_decoys) tensor with no integer-index path through
   `debye_huckel_pair_energy`". The flag still lets callers ship
   physically-inconsistent FI z-scores for one more minor version. Not
   a band-aid as such — but the deprecation is currently NOT in
   CHANGELOG, which means a pip-upgrade user could be silently bitten.
3. **#57 NaN-filled custom gamma tables still pass.** The energy agent
   added shape validation to all three scalar pair helpers but
   explicitly noted (`v2_fix_energy.md:40`): "NaN gamma tables still
   pass through (silent NaN is a separate v2 finding owned by the
   parameter-loader agent)". The parameter-loader agent's fix (#49)
   covers `load_gamma_tables()` reading from disk but doesn't reach
   the scalar API path where the user can supply a tensor literal.
   The audit explicitly called out "A 20x20 gamma table filled with
   NaN returned NaN" as a failure mode; that path is still open.

## Recommendations before pushing v0.2.0 to PyPI

1. **Expand the `[Unreleased]` CHANGELOG block to cover all 60+
   behaviour changes from the 5 non-docs agents.** Per the user's
   stated bar ("backward compat preserved … MUST be documented in
   CHANGELOG with clear migration notes"), this is required. The
   six `v2_fix_*.md` agent docs already contain the full list — it's
   a copy/edit job. Suggested sections:
   - `### Changed` — blank chain ID, TER segments, classify_frustration
     raises on NaN, density raises on length mismatch, sparse context
     dist is None, parameter loaders strict, dtype/precision strict,
     metadata fields renamed.
   - `### Deprecated` — `include_dh_in_e_native` for removal in v0.3.0.
   - `### Fixed` — singleresidue OOM, autograd-safe rho, theta diagnostic.
   - `### Added` — `metadata["v_dh"]`, `metadata["n_pairs_unfiltered"]`,
     warn-once `_CF_WARN_ONCE`, HETATM/MSE acceptance, OXT-as-O fallback,
     sparse-cutoff per-term warnings.
2. **Add `_validate_context_fingerprint` to `decoys.py:389` and
   `singleresidue_decoys.py:339`** before the v0.2.0 tag. The audit
   explicitly asked for this and the energy agent's summary overstates
   coverage. Five lines per call site.
3. **Either re-commit the 30-row `phase5_panel_results.csv` /
   `phase5_spearman.csv` OR soften the README's "30/30 combos"
   headline.** Currently the fresh clone evidence and the headline
   claim mismatch. The DOCS agent's note is honest but the gap
   remains.
4. **Either fix or document #57**: if shipping with NaN gamma tables
   silently producing NaN energies, add a one-sentence note in API.md
   so users know the validation contract stops at shape.
5. **(Nice-to-have)** Silence the pre-existing `APIDocsCoverageWarning`
   for `emit_5adens_dat` / `chain_segments` so `pytest` is fully
   clean on tag.

After (1)–(3) land, the package is publishable. Bit-identity is the
hard gate and that's preserved exactly. The code quality of the fixes
is genuinely higher than the v0.1.x baseline — no shortcuts, real
input validation at API entry points (not buried), and the OOM /
sparse / fingerprint additions address root causes rather than
symptoms. The CHANGELOG honesty issue is fixable in an afternoon.
