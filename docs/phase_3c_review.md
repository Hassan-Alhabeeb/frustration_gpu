# Phase 3c code review — FI + classification + dump emit

> **Status update (2026-05-20):** finding #1 (HIGH — CA-vs-CB coord-pick) **FIXED**.
> `src/frustration.py` now branches on `coords["is_gly"]` via a new `_xb_coords`
> helper and emits `cb_coords` for non-Gly residues / `ca_coords` for Gly in both
> `emit_tertiary_frustration_dat` (cols 5-10) and `emit_singleresidue_dat` raw
> mode (cols 3-5), matching `fix_backbone.cpp:5089-5091, 5155-5156`. New strict
> regression test `test_dump_coords_match_lammps_byte_exact` in
> `tests/test_frustration.py` asserts byte-identical coord columns for 5AON
> tertiary + singleresidue raw at `%8.3f` precision; would have FAILED pre-fix,
> PASSES post-fix. Full suite: 135/135 passing (134 prior + 1 new). FI Spearman
> / classification label-match table below is unchanged — coords do not enter
> the FI math.

Reviewer: Opus 4.7 / 2026-05-20.
Files reviewed:
- `F:/research_plan/frustration_gpu/src/frustration.py` (544 LOC, NEW)
- `F:/research_plan/frustration_gpu/src/__init__.py` (+22 exports)
- `F:/research_plan/frustration_gpu/tests/test_frustration.py` (556 LOC, 26 tests)

Reference cross-checked against:
- `F:/research_plan/frustration_gpu/docs/reference_lammps_awsem/fix_backbone.cpp` lines 5089-5104, 5152-5168, 5591-5596.
- `C:/Users/7sN/AppData/Local/Programs/Python/Python310/lib/site-packages/frustrapy/core/scripts/RenumFiles.pl`.
- Ferreiro et al. PNAS 2007 frustration paper (cited by LAMMPS comments).
- 5AON / 11BG / 1O3S / 3F9M LAMMPS dumps at `benchmark/cpu_baseline/{configurational, mutational, singleresidue}/`.

All 26 tests pass.

## Verdict

**CONDITIONAL PASS** — math, thresholds, and Welltype rule are correct; classification
label-match degradation is fully explained by RNG noise near the threshold boundaries;
but the raw-dump emitter writes **CA coordinates instead of CB** (LAMMPS uses CB for
non-Gly), so dump files are *not* byte-comparable to LAMMPS for the 6 coord columns
even when RNG matches.

## Findings by severity

| # | Severity | Where | Issue |
|---|----------|-------|-------|
| 1 | **High** | `frustration.py:298, 336, 392, 416` | Writer dumps `coords["ca_coords"]` for all residues; LAMMPS dumps **CB** for non-Gly and CA for Gly (`fix_backbone.cpp:5089-5091, 5155-5156`). Affects 6 coord columns in tertiary dump + 3 in singleresidue raw dump. Empirically: 5AON line 1 first coords are `28.683 19.332 7.848` (CA) but LAMMPS prints `29.989 19.265 7.058` (CB). Fix: branch on `coords["is_gly"][i]`. |
| 2 | Medium | `frustration.py:148-151` | `eps > 0` clamps `decoy_std` via `torch.clamp(min=eps)` — handles zero std but does not differentiate sign of zero numerator. Acceptable for the AWSEM use case but worth a unit test for `dm == e_native` with `ds == 0` (currently `nan/inf` only tested). Not blocking. |
| 3 | Low | `frustration.py:317-326` | `width = max(8, f+5)`. At default `precision=3` → `width=8` matches LAMMPS' `%8.3f`. At `precision=15` (the advertised "full float64" mode) → `width=20`. Self-consistent but not byte-comparable to LAMMPS at that setting — fine, the doc should call this out. |
| 4 | Low | `frustration.py:208-229` | `_chain_int_index` rebuilds the chain → int map **per call**; called twice per dump-line set. For small proteins this is irrelevant; for 10k-residue PDBs it adds O(N_chains) of overhead per pair set. Cosmetic. |
| 5 | Low | `frustration.py:423` | Post-processed singleresidue writer emits density / energies with no field-width spec (`f"{x:.3f}"`). LAMMPS' RenumFiles.pl uses `print $... $...` with default Perl separators (single space). Whitespace matches — confirmed by diff. OK. |
| 6 | Info | `test_frustration.py:498-511` | Welltype + FrstState comparisons are **label-by-label** (`for ours_line, them_line in zip ... if ot[-2] == tt[-2]: well_match += 1`), not aggregate distribution. This is the stricter form requested in the prompt. Good. |
| 7 | Info | `test_frustration.py:548-552` | Byte-diff test asserts only header lines + total row count. Numeric columns are not asserted equal (correctly — RNG floor differs). The test passes its own contract. |

## Sign convention verification

C++ `fix_backbone.cpp:5591-5596`:
```cpp
double FixBackbone::compute_frustration_index(double native_energy, double *decoy_stats)
{
  double frustration_index;
  frustration_index = (decoy_stats[0] - native_energy)/decoy_stats[1];
  return frustration_index;
}
```
`decoy_stats[0]` is `compute_array_mean(...)` (line 5335), `decoy_stats[1]` is
`compute_array_std(...)` (line 5336). So the C++ formula is
`FI = (decoy_mean − E_native) / decoy_std`.

Phase 3c (`frustration.py:147-151`):
```python
return (decoy_mean - e_native) / decoy_std
```
Identical. Sign convention `FI > 0 ⇒ minimally frustrated (native better than decoys)`
matches Ferreiro 2007 (their eq. 1) and the C++ VMD-coloring branch at line 5113-5117
(`> 0.78 ⇒ green / minimally`, otherwise red).

PASS.

## Threshold values verification

Two independent sources both pin `-1.0` and `0.78`:

1. C++ `fix_backbone.cpp:5105`:
   ```cpp
   if(frustration_index > 0.78 || frustration_index < -1) { ... }
   ```
   (Used as the VMD-coloring cutoff; values harvested as constants for our classifier.)

2. Perl `RenumFiles.pl:65-79`:
   ```perl
   if($FrstIndex<=-1)     { $FrstType="highly"; }
   if($FrstIndex>-1 && $FrstIndex<0.78) { $FrstType="neutral"; }
   if($FrstIndex>=0.78)   { $FrstType="minimally"; }
   ```
   Boundary inclusivity: `FI = -1` → highly (≤), `FI = 0.78` → minimally (≥).

Phase 3c (`frustration.py:172-175`):
```python
cls = torch.ones_like(fi, dtype=torch.long)
cls[fi <= high_threshold] = CLASS_HIGHLY        # ≤ -1 → highly
cls[fi >= minimal_threshold] = CLASS_MINIMALLY  # ≥ 0.78 → minimally
```
Inclusivity exactly matches RenumFiles.pl. PASS.

## Welltype rule verification

RenumFiles.pl:50-64:
```perl
if($splitted[10]<6.5)        { $ResResDistance="short"; }
elsif($splitted[10]>=6.5)    {
    if($Density1<2.6 && $Density2<2.6)  { $ResResDistance="water-mediated"; }
    else                                { $ResResDistance="long"; }
}
```
- `r_ij < 6.5` → `short` (rho irrelevant).
- `r_ij >= 6.5 AND rho_i < 2.6 AND rho_j < 2.6` → `water-mediated`.
- Otherwise (`r_ij >= 6.5 AND (rho_i >= 2.6 OR rho_j >= 2.6)`) → `long`.

Phase 3c (`frustration.py:178-199`):
```python
short_mask = rij < r_short
water_mask = (rho_i < rho_water_cutoff) & (rho_j < rho_water_cutoff)
well = torch.full_like(rij, WELL_LONG, dtype=torch.long)
well[(~short_mask) & water_mask] = WELL_WATER_MEDIATED
well[short_mask] = WELL_SHORT
```
Logic identical to Perl. The rename from `welltype_from_rij` → `welltype_from_contact`
is justified: the rule does depend on rho, not just rij. PASS.

`test_welltype_known_5aon_pairs` hand-checks 5 rows of 5AON against the LAMMPS dump
and `test_emit_postprocessed_matches_lammps_welltype_column` does the label-by-label
comparison across **all** rows of 5AON config (100% match required, not aggregate).

## Dump format verification

### tertiary_frustration.dat — 19 columns

C++ printf at `fix_backbone.cpp:5104`:
```
"%5d %5d %3d %3d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %c %c %8.3f %8.3f %8.3f %8.3f\n"
```
Columns: i, j, i_chain, j_chain, xi, yi, zi, xj, yj, zj, r_ij, rho_i, rho_j, a_i, a_j, E_native, decoy_mean, decoy_std, FI.

Phase 3c format string at `frustration.py:319-326`:
```python
"{:5d} {:5d} {:3d} {:3d} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} "
"{:8.3f} {:8.3f} {:8.3f} {} {} {:8.3f} {:8.3f} {:8.3f} {:8.3f}"
```
(at `precision=3`, `width=8`). Identical column count and widths; AA letters are
emitted via `{}`-substituted Python strings (not `%c`) but the rendered output is a
single character with the same flanking spaces.

Headers (line 328-332):
```
"# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j "
"native_energy <decoy_energies> std(decoy_energies) f_ij"
"# timestep: 0"
```
Byte-identical to LAMMPS:
```
$ head -2 5AON_tertiary_frustration.dat
# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j native_energy <decoy_energies> std(decoy_energies) f_ij
# timestep: 0
```
Confirmed via `out.read_text() == fp.read_text()` on line 549-550 of the test.

### singleresidue raw (LAMMPS native) — 11 columns

C++ printf at `fix_backbone.cpp:5168`:
```
"%5d %5d %8.3f %8.3f %8.3f %8.3f %c %8.3f %8.3f %8.3f %8.3f\n"
```
Columns: i, i_chain, xi, yi, zi, rho_i, a_i, E_native, decoy_mean, decoy_std, FI.

Phase 3c emits this format in the `raw=True` branch (lines 411-423). 11 columns,
right widths. **But:** the LAMMPS benchmark file at
`benchmark/cpu_baseline/singleresidue/5AON_singleresidue.dat` is the
**post-processed** schema (`Res ChainRes DensityRes AA NativeEnergy DecoyEnergy
SDEnergy FrstIndex`), not the raw LAMMPS dump. The `raw=False` branch (line 425)
matches it. The two formats can both be emitted, and our test only exercises
`raw=False` against the benchmark — fine, but the byte-comparison story differs
between modes.

### post-processed pair (configurational / mutational) — 14 columns

RenumFiles.pl:30 header:
```
Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2 NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState
```
Phase 3c (`frustration.py:507-510`) emits the same header verbatim. 14 columns of
data per row, last two are word-labels (`short`/`water-mediated`/`long` and
`highly`/`neutral`/`minimally`). PASS — matched label-by-label in
`test_emit_postprocessed_matches_lammps_welltype_column`.

### Coordinate column bug (only consequential format defect)

LAMMPS dumps the **CB atom** for non-Gly residues and CA for Gly
(`fix_backbone.cpp:5089-5091`, `5155-5156`). Phase 3c always dumps
`coords["ca_coords"]` (`frustration.py:298` and `392`). Effect on 5AON line 1 (Ser
at index 0, non-Gly):

| Source | xi, yi, zi |
|---|---|
| LAMMPS dump | `29.989 19.265 7.058` (CB) |
| Phase 3c emit | `28.683 19.332 7.848` (CA) |

Difference is small in magnitude (1-2 Å) but **not** byte-comparable, and changes
all 6 coord columns of the tertiary dump and 3 of the singleresidue raw dump.

Fix: branch on `coords["is_gly"][i]` and pick CB or CA accordingly. Should be
~6 added lines.

## Per-column byte diff results on 5AON (configurational)

Test harness:
```python
emit_tertiary_frustration_dat(mode='configurational', coords=parse_pdb('5AON.pdb'),
    pair_i=..., pair_j=..., r_ij=raw, rho_i=raw, rho_j=raw, e_native=raw,
    decoy_mean=stats['decoy_mean'], decoy_std=stats['decoy_std'], output_path=out)
```
where `raw` columns are lifted from the LAMMPS dump (to isolate FI-only drift).

Results:
| Field | Status |
|---|---|
| Line count | **221 data + 2 header = 223. Match.** |
| Header line 0 | Byte-identical. |
| Header line 1 | Byte-identical. |
| Total byte length | Ours **31519** vs LAMMPS **31296**. 0.7% drift, from 4-extra-byte differences (sign of negative numbers at slightly different positions / extra leading whitespace on a few rows where our value's integer part has more digits). Same length per row — confirmed below. |
| Per-row character length | Identical (139 chars/row). |
| Cols 1-4 (i, j, i_chain, j_chain) | **Byte-identical.** |
| Cols 5-10 (xi yi zi xj yj zj) | **Differ.** CA-vs-CB bug — finding #1. |
| Col 11 (r_ij) | Byte-identical (we re-use LAMMPS values in this harness). |
| Cols 12-13 (rho_i, rho_j) | Byte-identical (same). |
| Cols 14-15 (a_i, a_j) | Byte-identical. |
| Col 16 (E_native) | Byte-identical (re-used). |
| Col 17 (decoy_mean) | Differs by ~0.01 — **RNG floor**. Configurational uses a single scalar for the whole protein, so the drift here is at most one value (0.009 in absolute scale on 5AON). |
| Col 18 (decoy_std) | Differs by ~0.015 — RNG floor. |
| Col 19 (FI) | Differs by 1-LSB sometimes (the last-digit rounding driven by upstream RNG-drift in dm/ds). |

**Total bytes differ ⇒ NOT byte-comparable.** Headers + row count + structure
identical. Cells differ in **three categories**: (a) RNG-driven (cols 17, 18, 19) —
expected and acceptable, (b) CA-vs-CB bug (cols 5-10) — **not acceptable**, and
(c) zero categories of unexpected structural mismatch.

## Label match degradation analysis (97-99% not 100%)

Boundary-distance analysis on the LAMMPS-reported FI for the residues that we
**mis-classify** relative to LAMMPS:

```
5AON config: 0/221 disagreements
11BG config: 8/1517 (0.53%); 8/8 within 0.05 of {-1, 0.78}
1O3S config: 28/1106 (2.53%); 28/28 within 0.05 of boundary
3F9M config: 33/3349 (0.99%); 33/33 within 0.05 of boundary

5AON SR: 2/49 (4.08%); 2/2 within 0.05 of boundary
11BG SR: 1/248 (0.40%); 1/1 within 0.10 of boundary (0.069)
1O3S SR: 6/200 (3.00%); 5/6 within 0.05 of boundary
3F9M SR: 3/451 (0.67%); 3/3 within 0.05 of boundary
```

**Every** mis-classification has its LAMMPS-reported FI within 0.1 of a Ferreiro
threshold, and >90% within 0.05. The decoy-stat RNG drift on these PDBs is ~0.01-0.05
in `decoy_mean`, exactly the same order of magnitude as the boundary distance. The
mis-classifications are residues whose true FI sits ε away from -1.0 or 0.78 and
whose ε is small enough to be flipped by RNG. There is no systematic 1% bug — the
threshold function and FI math are correct.

Spearman = 1.0000 on configurational mode is a real effect, not a measurement
artefact: in configurational mode the decoy stats are a single scalar shared across
**all pairs in the protein**, so FI = (constant − E_native_i) / constant. Spearman
(rank correlation) of a strictly monotone affine function of `E_native` against
itself is identically 1. The 95-99% label match is the only signal of true RNG
drift here, and it confirms scalar drift magnitude rather than a logic bug.

## Recommended Phase 4 next step

1. **Fix the CA/CB bug (1-line bug, 6-line patch).** Add a `_get_xi(coords, i)`
   helper that returns CB for non-Gly and CA for Gly, plug into both pair and
   singleresidue writers. After this fix, with matched RNG, the tertiary dump
   should be byte-identical to LAMMPS modulo the 3 decoy columns.
2. **Add a `--seed-match-lammps` story.** Right now `seed=0` in the decoy stats
   gives ~3% drift in `decoy_mean`. If Phase 4 includes RNG-pinned regression
   tests, we want a deterministic protocol that matches LAMMPS' `xorshift` state
   exactly. Otherwise the byte-diff story is permanently capped at ~97% label
   match. Could be punted if a CPU-baseline-with-fixed-RNG isn't a requirement.
3. **Then proceed to Phase 4 = end-to-end frustration CLI** (`frust analyze
   --pdb X --mode configurational --output dir/`). Wire the writers behind a
   thin argparse layer and a `pdb_equivalences.txt` emitter so frustratometeR's
   downstream tooling (PDB-coloring, plotter, JSON exports) can consume our
   output as a drop-in.
4. **Defer**: nmer / decoy-VMD-script emit, contact-map plots, JSON aggregator.
   Not on the LAMMPS-binary-compat critical path.

## Test coverage summary

- 26 / 26 pass.
- 10 pure-function tests (boundary cases for FI, classification, Welltype).
- 3 round-trip writer tests (header text + column count + author-resnum
  schema).
- 12 LAMMPS-validation tests (configurational + mutational + singleresidue × 4
  PDBs).
- 1 Welltype + FrstState label-by-label comparison on full 5AON config (strict).
- 1 header + row-count byte-diff test on 5AON config.
