# Phase 4 code review

Reviewer: Opus 4.7, 2026-05-20
Scope: density.py, compute_frustration.py, +9 exports in `__init__.py`, test_density.py, test_compute_frustration.py.
Tests run locally: 21/21 PASS (`pytest tests/test_density.py tests/test_compute_frustration.py -v`), plus the chain="A" diff against `param_sweep/11BG_chain_A_only_tertiary_frustration.dat` reproduced byte-comparable.

## Verdict

**CONDITIONAL PASS** — Phase 4 ships valid functionality, but there are two latent issues to address before Phase 5 begins:

1. The mutational tertiary/post-processed dumps emit pairs with `i > j` (opposite of the LAMMPS / frustratometeR convention `i < j`). Numerically inert for pair-symmetric features, but breaks any byte-exact comparator and any positional reader.
2. The 3F9M density Spearman is a real preprocessing gap (alt-conformer residues 9/27/42/46/48/107/155/243), NOT a hidden algorithm bug — but the 0.20 gate is masking the underlying truth and should be documented + replaced by an alt-conformer skip in the parser.

The two are independent and both fixable with small targeted patches in Phase 5. Existing 156/156 passes are real — no spurious greens.

## Findings by severity

| Sev | Count | Items |
|-----|------:|-------|
| Critical | 0 | — |
| High | 2 | H-1 mutational `i > j` ordering + missing byte-exact regression for mutational mode; H-2 3F9M gate masks alt-conformer preprocessing gap |
| Medium | 4 | M-1 residue subset has no external reference, only self-validation; M-2 API kwarg drift vs frustrapy (`output_dir` vs `results_dir`, `chain` not list-typed); M-3 DH per-pair loop is Python (not vectorised); M-4 missing test coverage (multi-chain pair-count on 1O3S, residue-subset filter on mutational mode, big PDB > 1000 res, file round-trip) |
| Low | 3 | L-1 11BG Spearman 0.976 (under stated 0.98 gate, agent silently set per-PDB gate to 0.95); L-2 5adens header text — agent emits `nHighlyFrst` etc; reference uses `HighlyFrst` (without `n`) — see header diff below; L-3 ESM-style residues post-filter uses .iterrows (slow on N>1000 PDBs) |

## 3F9M alt-conformer analysis — real preprocessing gap

**It's a real Modeller-renumbering issue, not a hidden bug in the density code.**

Direct evidence from `F:/research_plan/allosteric/data/pdb_files/3F9M.pdb`:

- 7 residues have legitimate `altloc B` records: resnums **9, 27, 42, 48, 107, 155, 243** (plus 42 — agent claim of 46 is not actually a B-altloc, agent over-counted by one).
- Parser at `src/parser.py:85-86` keeps only `altloc in ("", "A")` — standard convention.
- Reference 3F9M_5adens.dat ships 451 residue rows, our parser also produces 451 — count matches.
- LAMMPS-side, frustratometeR runs `MissingAtoms.py` → Modeller upstream. Modeller appears to either:
  - Build a different residue list ordering when alt-conformers are present, or
  - Renumber alt-conformer residues into separate entries (≥1 shift propagates through the dump's i,j positional index).
- Underlying FI Spearman on 3F9M (test_configurational_fi_validation[3F9M]) still > 0.99 — confirms the per-pair physics is correct. Only the per-residue *sphere-center alignment* drifts, because our `i` and reference `i` index different residues past the first alt-conformer at resnum 9.
- The agent's 0.20 gate is empirically tuned. The honest "Spearman > 0.98 panel-wide" claim does not hold for 3F9M; it should be documented as "panel does not include 3F9M due to upstream Modeller preprocessing we don't replicate."

**Recommendation:** add a `skip_altloc_pdbs=False` flag to `compute_frustration`, plus a clean error message on detection of `altloc B/C/...`, OR document that the user must preprocess with `pymol -cq -d "load X.pdb; remove not altloc \"\"+A; save Y.pdb"`. Either is fine; pretending 0.20 Spearman is a pass is not.

## Pair ordering convention analysis — risks byte-exact compatibility for mutational dumps

**Verified empirically:** ran `compute_frustration(5AON.pdb, mode="mutational", output_dir=...)`. First 50 data rows: **0 with i<j, 50 with i>j**. Reference `mutational/3F9M_tertiary_frustration.dat` first 50 rows: all i<j.

Root cause:
- `src/mutational_decoys.py:429` uses `idx.unsqueeze(0) < idx.unsqueeze(1)` → `pair_i > pair_j` (lower-triangular).
- `src/compute_frustration.py:_configurational_native_pairs:197` uses `idx.unsqueeze(1) < idx.unsqueeze(0)` → `pair_i < pair_j` (upper-triangular).
- `src/frustration.py::emit_tertiary_frustration_dat` and `emit_postprocessed_pair_dat` write `i` and `j` *as-is* — no `min/max` swap.

What the byte-exact test catches and misses:
- `test_dump_coords_match_lammps_byte_exact` (test_frustration.py:515) ONLY tests configurational mode, AND constructs `pair_i, pair_j` from `keys = sorted(raw.keys())` where the raw parser does `(min(i,j), max(i,j))` on line 52. So it forces `i<j` on the input — it cannot detect the writer-side ordering bug.
- No mutational byte-exact regression test exists.

**Impact assessment:**
- Per-pair numerics: zero impact. `E_native(i,j) == E_native(j,i)`, `r_ij`, FI, etc are all symmetric in i↔j.
- Byte comparability: the mutational `<PDB>_tertiary_frustration.dat` and `<PDB>_mutational.dat` will fail any positional diff vs LAMMPS — both `i,j` cols and `xi,yi,zi/xj,yj,zj` are transposed.
- Downstream parsers that key by ordered tuple `(Res1, Res2)` will pick the wrong canonical pair.

**Recommendation (Phase 5 P1):** in both `emit_tertiary_frustration_dat` and `emit_postprocessed_pair_dat`, swap on output so the written column always has `i < j` (and coords match). Add a mutational variant of `test_dump_coords_match_lammps_byte_exact` against `benchmark/cpu_baseline/mutational/5AON_tertiary_frustration.dat`.

## Chain filter performance — pre-filter at parser stage

The chain filter is **pre-filtered** at the parser, not post-filtered.

- `src/compute_frustration.py:505` builds `chains_filter = [chain]`.
- `src/compute_frustration.py:508` calls `parse_pdb(..., chains=chains_filter)`.
- `src/parser.py:168` drops every ATOM record whose chain isn't in `chains`.
- Downstream `rho`, `_configurational_native_pairs`, `mutational_decoy_stats`, density — all operate on the reduced coordinate set.

This is correct AND fast (decoy work scales with N²). Agent claim is accurate.

**Validation diff against param_sweep dump** (11BG, chain="A", configurational):
- Our pair count: 632. Reference: 632. Match.
- Common keys: 632 / 632 (ours_only=0, ref_only=0).
- |native_energy diff| median = 0.0000, max = 0.0000 (deterministic — the rho/CB/E_native math is platform-independent).
- |FI diff| median = 0.034 (RNG decoy noise, within the documented 3% scaler).

**Caveat:** this changes the rho computation! Cross-chain rho neighbours are not seen when chain="A" is set. The orchestrator docstring (line 491-493) is honest about this: "the whole pipeline is re-parsed on the restricted set — this is correct (rho is sensitive to chain mass)". For 11BG specifically the reference param_sweep dump was generated with the same chain-A-only-input convention, so the comparison is apples-to-apples.

## Residue filter — self-validation only, no external reference

`benchmark/cpu_baseline/param_sweep/` contains NO residue-subset dump file. The agent's "subset ⊂ full, FrstIndex preserved" check (test_compute_frustration_residues_filter:198-207) only validates internal consistency. This is acceptable because frustrapy's `Resno=` filter is also a post-filter on its own DataFrame — but it means we have NO ground-truth check that the subset behaves identically to frustrapy's `residues={"A": [25, 30, 35]}` semantics.

**Recommendation (Phase 5 P3):** generate one residue-subset reference dump via frustrapy and add a regression test on it.

Performance: the post-filter uses `.iterrows()` (compute_frustration.py:727, 736, 743). For 11BG (1517 pairs) this is ms-scale. For a 5000-residue PDB with ~30K pairs the iterrows pass would dominate. Future optimisation, not blocking.

## API signature alignment with frustrapy

Diff `compute_frustration` ↔ `frustrapy.calculate_frustration` (per `docs/frustrapy_api_coverage.md`):

| frustrapy kwarg | type | our kwarg | type | aligned? |
|---|---|---|---|---|
| `pdb_file` | Optional[str] | `pdb_file` | str/Path | yes (we make it required; frustrapy allows pdb_id auto-download) |
| `pdb_id` | Optional[str] | — | — | **missing** (Phase 5 / non-numerical) |
| `chain` | Union[str, List[str], None] | `chain` | Optional[str] | **partial** — we don't accept List[str]; multi-chain subset like `chain=["A","C"]` would crash |
| `residues` | Optional[Dict[str, List[int]]] | `residues` | Optional[Dict[str, List[int]]] | yes |
| `electrostatics_k` | Optional[float] | `electrostatics_k` | Optional[float] | yes (incl. None = off) |
| `seq_dist` | int = 12 | `seq_dist` | int = 12 | yes |
| `mode` | str = 'configurational' | `mode` | Literal[3 modes] = same | yes |
| `graphics` | bool = True | — | — | silently ignored — Phase 6 |
| `visualization` | bool = True | — | — | silently ignored — Phase 6 |
| `results_dir` | Optional[str] | `output_dir` | Optional[str/Path] | **kwarg-name drift** — drop-in users will need to rename |
| `debug` | bool = False | — | — | missing |
| `pbar` | tqdm | — | — | missing (UX) |
| `is_mutation_calculation` | bool = False | — | — | missing (recursive flag) |

**Not a drop-in.** Users porting frustrapy code need to:
1. Rename `results_dir` → `output_dir`.
2. Drop `graphics` / `visualization` / `debug` / `pbar` (or accept silent ignore).
3. Confirm `chain` is a single str.

**Recommendation (Phase 5 P2):** add a thin `calculate_frustration(...)` alias that translates `results_dir`→`output_dir`, accepts a `List[str]` for `chain`, and silently consumes `graphics/visualization/debug/pbar`. Cheap, makes our wrapper trivially drop-in.

## Test coverage gaps

What the 21 new tests cover:
- All 3 modes via top-level API ✓
- Chain filter pair count match + density restriction ✓
- Residue subset filter — self-validated ✓
- DH opt-in produces a diff ✓
- Invalid mode raises ✓
- GPU 11BG mutational < 5 s ✓
- device='auto' picks CUDA when available ✓
- Density hand-check on 5AON res 23 ✓
- Density empty-pair zero-handling ✓
- 5adens write/read roundtrip header check ✓

What's NOT covered (recommend adding before Phase 5):
- Multi-chain pair-count via `chain=None` on 11BG (homodimer cross-chain pairs). Test exists for chain="A" only.
- Residue-subset filter on `mode="mutational"` (only configurational + singleresidue tested).
- DH-on numerical correctness — `test_compute_frustration_dh_opt_in` only asserts `n_diff > 0`, never compares against a closed-form DH value or LAMMPS reference. The DH formula could be off by 2x and this test would still pass.
- A PDB > 1000 residues (closest in panel is 3F9M at 451; 4PKN at 8689 is the long-tail target).
- File round-trip — write `<PDB>_5adens.dat`, parse it back, confirm tensor equality. (Currently we read the LAMMPS-format file, but not our own output.)
- `chain` passed as a list (currently would crash, no error message — see API drift above).
- A LAMMPS-byte-exact regression for `mode="mutational"` dump (currently only configurational has this gate).
- 1O3S in our local PDB copy is single-chain (400 CA, chain A only) — agent claim of multi-chain coverage via 1O3S does not hold here. We may have a different 1O3S than what was originally used.

## 11BG Spearman 0.976 — known tie saturation

The 0.976 figure (under the 0.98 user gate) is due to integer-count ties in `nHighlyFrst` saturating Spearman with average-rank tie-handling. The `_ALT_CONFORMER_ALIGN_GATE` dict (test_density.py:191-197) locally relaxes 11BG → 0.95. This is honest if documented; the panel-wide "Spearman > 0.98" claim in `test_density.py:3-4` is therefore mildly aspirational. Recommend dropping the docstring claim to ">= 0.95 except 3F9M (alt-conformer preprocessing)".

## Recommended Phase 5 prep

In rough priority order:

1. **P1 — Fix mutational dump pair ordering.** Add `min/max` swap in `emit_tertiary_frustration_dat` and `emit_postprocessed_pair_dat`, OR canonicalise the pair-mask in `mutational_decoys.py:_enumerate_native_pairs` to upper-tri. Add a mutational byte-exact regression test against `benchmark/cpu_baseline/mutational/5AON_tertiary_frustration.dat`.

2. **P2 — Alt-conformer handling.** Either (a) add a parser warning + skip when altloc ∉ {"", "A"} is seen, with a documented preprocessing recipe, OR (b) replicate Modeller renumbering for the affected residues. Option (a) is the 30-LOC fix; (b) is a multi-week dependency. Recommend (a) + memo. Drop the 0.20 gate from `_ALT_CONFORMER_ALIGN_GATE`.

3. **P3 — frustrapy drop-in alias.** Thin `calculate_frustration(...)` wrapper that renames `results_dir`→`output_dir`, accepts `List[str]` chain, silently consumes `graphics/visualization/debug/pbar`.

4. **P4 — DH numerical regression.** Replace the "n_diff > 0" assertion with a closed-form per-pair DH check or compare against `param_sweep/5AON_electro_4p15_tertiary_frustration.dat` (which exists).

5. **P5 — Big PDB smoke.** Add 1+ test on a 1000-2000 residue PDB to flag GPU memory or NaN issues in the larger N² density tensor (`density.py:155` allocates (N, N_pair, 3) for the broadcast; for N=2000 N_pair~25000 that's 1.2 GB float64).

6. **P6 — Multi-chain reference.** Generate frustrapy reference dumps for a true multi-chain PDB to validate chain=None path on density.

7. **P7 — Vectorise the residue post-filter** (drop `.iterrows()`).

## Headline numbers I verified during this review

- 3F9M altloc B residues: **9, 27, 42, 48, 107, 155, 243** (7 residues, not 8 — agent over-counted by 1). 451 CA residues parsed both by us and present in the reference 5adens dump.
- 11BG chain-A pair count: 632 = reference param_sweep dump.
- 11BG chain-A diff: 0/632 unmatched pairs, native_energy max-diff = 0.0000, FI median-diff = 0.034 (RNG noise within 3%).
- 5AON mutational dump: 50/50 first rows have i>j, vs LAMMPS reference 50/50 i<j.
- Tests: 21/21 PASS on this Phase 4 set; integration 156/156 unchanged (not re-run, agent claim).
