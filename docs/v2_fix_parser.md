# Parser hardening fixes (2026-05-21)

Reference: `F:/research_plan/New folder/odo.txt`. This pass addresses the
PARSER-only audit findings from the 75-finding 2026-05-21 audit. Every fix
lives in `frustration_gpu/parser.py` and is covered by a regression test in
`tests/test_parser_edge_cases.py`.

## Per-finding summary

| # | Severity | Finding | Where in parser.py |
|---|----------|---------|--------------------|
| 9 | MED | HETATM MSE/SEC/PYL accepted instead of dropped | `_parse_atom_record` lines 199-209 — added `HETATM_PROMOTED_RESNAMES` (lines 133-145); HETATM lines whose resname is in this set are treated exactly like ATOM lines |
| 17 | HIGH | TER + same-letter restart now creates a NEW chain segment (label `<letter>#<n>`) | `parse_pdb` lines 408-468 — new `chain_segment_counter` + `_segment_label` helper bumps the segment number on each TER; the residue dedup key uses the suffixed label, so the downstream `_build_chain_index` sees a fresh chain |
| 18 | MED | Insertion codes already in `insertion_codes` output — documented in module docstring + new regression test pins the parser-level behaviour (the compute-layer DataFrame fix is owned by another agent) | docstring lines 37-41; test `test_insertion_codes_distinguish_residues` |
| 37 | MED | Altloc-B-only residues no longer dropped / no IndexError | `_parse_atom_record` line 236 (accept B at line level); `parse_pdb` lines 526-552 (default-mode B-only retention); lines 583-596 (lammps_compat-mode B-only retagged as A when no matching A exists) |
| 38 | MED | Blank chain ID kept as `""` instead of being coerced to `"A"` (no silent collision with real chain A) | `_parse_atom_record` line 242 — `chain_id = line[21:22].strip()` (no `or "A"`) |
| 44 | LOW | Duplicate-atom-name records: highest-occupancy coord wins, not first-encountered | `_parse_atom_record` lines 264-275 (parse occupancy); `parse_pdb` lines 507-524 (occupancy-aware atom dedup) |
| 45 | MED | `END` record terminates coordinate parsing | `parse_pdb` lines 449-450 |
| 46 | MED | Non-finite coords (NaN / Inf) rejected at line level | `_parse_atom_record` lines 256-260 |
| 53 | MED | Mixed resnames at the same `(chain, resnum, icode)` raise `ValueError` instead of silent merge | `parse_pdb` lines 500-506 |
| 60 | MED | Second `MODEL` record without `ENDMDL` stops parsing | `parse_pdb` lines 438-444 |
| 61 | LOW/MED | OXT promoted to the `O` slot when residue has no `O` record | `_parse_atom_record` line 225 (accept OXT at parse time); `parse_pdb` lines 555-557 (post-grouping OXT-to-O fallback) |

## Breaking-behaviour changes flagged for the v0.2.0 changelog

Three changes alter parser output on inputs that were previously buggy. Real
RCSB PDBs are very unlikely to trigger them, but downstream code should be
aware:

1. **Blank chain ID is now `""` not `"A"`** (#38). If a caller's
   downstream pipeline assumes every chain string is non-empty, it must
   either filter out empty-string chains or upgrade to handle them. The
   four bundled validation PDBs (5AON, 11BG, 1O3S, 3F9M) have no blank
   chain records — they are unaffected.
2. **TER + same-letter segments emit suffixed chain labels** (#17). A PDB
   that wrote `chain A 1...n TER chain A 1...m` now emits chains `A`
   and `A#2`. The chain-list filter (`chains=["A"]`) was widened to match
   both segments so end-user code still gets the same residue set, but
   `chain_ids` strings differ. Same caveat for the four validation PDBs:
   no segment repeats, so visible output is identical.
3. **Mixed resnames at the same `(chain, resnum, icode)` now raise**
   (#53). Previously silent merge of ALA + GLY at A:1 into one ALA;
   now `ValueError("conflicting residue names ...")`. Real PDBs with
   microheterogeneous residues use altloc, so this should only fire on
   malformed synthetic inputs.

No changes to the `ca_coords`, `n_coords`, `c_coords`, `o_coords`,
`cb_coords`, `residue_types`, `residue_numbers`, `is_gly`, `is_dna`,
`is_altloc_b_shadow`, or `lammps_emit_rows` outputs on the four golden-anchor
validation PDBs (5AON, 11BG, 1O3S, 3F9M); same residue counts as before
(49 / 248 / 200 / 451 respectively).

## Test count delta

- Before this pass: `tests/` reported 223 passing.
- New tests added by this pass: 20 (in `tests/test_parser_edge_cases.py`).
- After this pass: parser-relevant tests still all pass; the validation
  golden anchors are pinned by parameterized tests at
  `test_validation_pdbs_unchanged[{5AON,11BG,1O3S,3F9M}-...]`.

Note: a few unrelated test failures in `test_compute_frustration.py`,
`test_decoy_validation.py`, and `test_api_docs.py` existed before this pass
(verified by stashing `frustration_gpu/parser.py` and re-running them).
Those belong to other agents' scopes.

## Findings deferred / out of scope

- **#18 downstream surface**: the audit finding also calls out
  `compute_frustration.py` building DataFrames without an `icode` column.
  That sits in another agent's scope (compute_frustration.py is forbidden
  for this pass). The parser already returns `insertion_codes` correctly;
  no change is needed at the parser layer beyond the documented test.

## Unexpected issues found while implementing

- The original `_parse_atom_record` short-circuit `accepted_altlocs` test
  blocked altloc-B records at line level, which made #37 impossible to fix
  cleanly. The fix routes B records through the line-level parse and lets
  the grouping logic decide what to do based on whether a matching altloc-A
  exists in the same `(chain, resnum, icode)` group.
- `_weave_altloc_b_shadows` + `_build_lammps_emit_rows` could index a
  stale `eq_list` when an altloc-B-only residue has no matching A. Added a
  pre-weave re-tag pass that promotes A-orphaned B groups to A so they
  behave as primary residues instead of shadows.
- The 1-line `chain_id = line[21:22].strip() or "A"` "or A" was load-bearing
  in two senses (silently making blank chains "valid" and merging them with
  real "A"). Removing it required the rest of the grouping logic to be OK
  with the empty-string chain label — turned out fine because the
  downstream `_build_chain_index` keys on string equality.
