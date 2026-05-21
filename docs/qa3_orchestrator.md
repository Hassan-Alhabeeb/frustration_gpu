# QA3 — Orchestrator + Emit + Density code review

Reviewer: Opus 4.7, 2026-05-21
Scope:
- `F:/research_plan/frustration_gpu/src/compute_frustration.py` (1237 LOC)
- `F:/research_plan/frustration_gpu/src/frustration.py` (579 LOC) — FI + Ferreiro + dump writers
- `F:/research_plan/frustration_gpu/src/density.py` (272 LOC)
- `F:/research_plan/frustration_gpu/src/__init__.py` — public exports

Mode: READ-ONLY. No code changes. >80% confidence on each finding.

## Verdict

**CONDITIONAL PASS.** The post-P1+P3+P4 sprint repaired the two big numerical / byte-comparability issues (DH-in-E_native semantics, mutational `i<j` ordering), and the new LAMMPS-compat flags (DNA, altloc-B, incomplete-backbone) wire through cleanly in both `compute_frustration` and the `calculate_frustration` adapter. The math layer correctly subsets to protein-only via `_subset_protein_only`, with a stable `math_to_full_idx` projection for density emission. No CRITICAL findings. Two HIGH and several MEDIUM findings, all isolatable patches; no design rework required.

## Findings by severity

| Sev | Count | Items |
|-----|------:|-------|
| CRITICAL | 0 | — |
| HIGH | 2 | H-1 `_xb_coords` per-element NaN-mask can mix CA/CB axes; H-2 calculate_frustration multi-chain path silently differs from a true multi-chain rho |
| MEDIUM | 5 | M-1 wall_clock_ms unreliable on CUDA (no synchronize); M-2 DH per-pair Python loop; M-3 residues post-filter uses .iterrows; M-4 missing residues-resnum existence check; M-5 _build_pair_dataframe ignores compute_frustration's full v60.5-style residue-subset rho |
| LOW | 3 | L-1 math_to_full_idx returned but never used in compute_frustration; L-2 dataclass dataframe fields are typed `Any` (no pandas import-time dependency, OK but loses IDE help); L-3 minor docstring drift |

---

## H-1 — `_xb_coords` NaN handling is per-element, not per-row (compute_frustration / frustration / density all affected)

**File / line:** `src/frustration.py:259`

```python
mask = is_gly.unsqueeze(1) | torch.isnan(cb)
return torch.where(mask, ca, cb)
```

`is_gly.unsqueeze(1)` is `(N, 1)`; `torch.isnan(cb)` is `(N, 3)`. The broadcast OR produces an `(N, 3)` mask, then `torch.where(mask, ca, cb)` element-wise picks from CA or CB **per axis**. Consequence: if a residue's CB row has NaN only in some axes (theoretically possible if a downstream Engh-Huber NaN propagated through one axis only, or if a user constructed coords by hand), the output coord would be a Frankenstein `(CB.x, CA.y, CB.z)`. This violates the "fall back to CA when CB is missing" contract.

In the current parser, NaN-fill is whole-row (`src/parser.py:354-358` writes `cb = torch.full((n_res, 3), nan, ...)`), so the bug is **latent** — but `src/_contact_common.py:_resolve_contact_coords` is correct (uses `~torch.isfinite(cb).all(dim=-1, keepdim=True)` → per-row mask). The two modules disagree on convention.

Why this matters: `_xb_coords` is the LAMMPS-CB-on-non-Gly, CA-on-Gly source used by **both** the tertiary/singleresidue dump writers (`frustration.py:332,429`) **and** the density module (`density.py:141`). A subtle mix could shift sphere midpoints and emitted coords by sub-Å.

Confidence: 99%. Fix: replace with the same `nan_row = ~torch.isfinite(cb).all(dim=-1, keepdim=True)` pattern as `_resolve_contact_coords`.

---

## H-2 — `calculate_frustration` multi-chain path bypasses chain-level rho

**File / line:** `src/compute_frustration.py:1153-1167`

When the adapter receives `chain=["A", "B"]` (list of >1 entries) it sets `chain_arg = None` and runs the FULL pipeline (all chains), then post-filters the result dataframes on the `keep_chains` set (lines 1213-1227). The orchestrator docstring (lines 599-608) is honest that chain filter affects rho because rho is sensitive to chain mass — but it documents this only for the `compute_frustration(chain="A")` case.

The adapter's behaviour is therefore: `chain=["A"]` (length 1) → parser-level filter (rho excludes other chains). `chain=["A","B"]` (length ≥ 2) → full-pipeline rho **then** filter. Two callers asking "give me chains A and B only" get numerically different rho, FI, density depending on whether they passed `chain="A"` (single) vs `chain=["A"]` (length-1 list — handled correctly) vs `chain=["A","B"]` (handled differently).

The adapter docstring at lines 1112-1114 mentions this is a widening but does NOT mention the rho semantics divergence.

Confidence: 90%. Fix: either (a) widen `compute_frustration`'s `chain` kwarg to accept `list[str]` and forward through to `parse_pdb(..., chains=...)`, or (b) add a loud `UserWarning` in the adapter's multi-chain branch.

---

## M-1 — `wall_clock_ms` unreliable on CUDA, no `torch.cuda.synchronize()` before `time.perf_counter()`

**File / line:** `src/compute_frustration.py:617, 916`

The orchestrator brackets the whole pipeline with `start = time.perf_counter()` and `wall_ms = (time.perf_counter() - start) * 1000.0`. There is **no** `torch.cuda.synchronize()` before either bracket. On CUDA, kernels execute asynchronously; the timer captures CPU launch time. Some forced syncs happen via `.item()` (`decoy_mean.item()`, `decoy_std.item()` at lines 702-703) and the eventual `.cpu().tolist()` in `_build_pair_dataframe`, but the actual GPU completion time isn't guaranteed at the end-bracket if any post-stats kernel is still in flight.

In practice the dataframe construction forces a host-side sync (the dm_per_pair tolist call) so the timing is approximately right, but the metadata field is documented (line 930) as a real wall-clock — it isn't, for CUDA. CPU runs are fine.

Confidence: 85%. Fix: bracket the body with `if dev.type == "cuda": torch.cuda.synchronize(dev)` before both `time.perf_counter()` calls.

---

## M-2 — DH per-pair contribution uses a Python loop

**File / line:** `src/compute_frustration.py:357-366` (`_add_dh_to_e_native`)

```python
for k in range(n_pair):
    v = debye_huckel_pair_energy(float(rij_l[k]), int(aa_i_idx[k]), ...)
```

For 11BG (1517 pairs) this is ms-scale, but the per-call overhead of `debye_huckel_pair_energy` (which itself creates 0-d tensors at line 397-403) compounds. A 5000-res PDB with ~30K pairs would dominate the orchestrator wall time. The function header (line 342) already calls this out as a future optimisation.

Confidence: 100% (the loop is right there). Vectorised version: gather `q_i[pair_i]`, `q_j[pair_j]`, multiply by `exp(-r/lambda)/r` in a single tensor op. ~10 lines.

---

## M-3 — Residue subset post-filter uses `.iterrows()` (slow)

**File / line:** `src/compute_frustration.py:891-914`

Three pandas `for _, row in df.iterrows()` loops to build keep-masks for pair_df / sr_df / density_df. iterrows is the slowest possible pandas iteration. Fine for 1517 pairs; quadratic-feeling for 100K+ pair PDBs (e.g. multi-domain Hsp90).

Confidence: 100%. Vectorised replacement: build per-residue containment masks via `df["ChainRes1"].map(...).isin(...)` etc., one boolean expression per dataframe.

---

## M-4 — `residues` subset doesn't validate the user's resnums exist

**File / line:** `src/compute_frustration.py:891-914`, no early validation.

A user passing `residues={"A": [9999]}` (typo / wrong PDB numbering) gets back **empty** `pair_records`, `singleresidue_records`, `density_records` with no warning. Combined with the lack of `output_dir` validation (silent skip when n_pairs == 0), debugging is painful: the user assumes their predicted residue was scored 0, when actually it was never in the structure.

Confidence: 100%. Fix: at the top of compute_frustration, after parse, intersect `residues` with the actual `(chain, resnum)` set and raise/warn on missing entries.

---

## M-5 — `_build_pair_dataframe` uses tolist() in tight loops

**File / line:** `src/compute_frustration.py:402-403` and parallel loops in `_build_singleresidue_dataframe`.

```python
def _round(t: torch.Tensor) -> List[float]:
    return [round(float(v), precision) for v in t.detach().cpu().tolist()]
```

For each of 7 columns it forces a `cpu().tolist()` round-trip plus a Python `round()`. Could be `t.round(decimals=precision).cpu().numpy()` to pandas Series, one shot per column. Same complaint applies to `rho_list = rho.tolist()` in the dump writers — not a correctness issue, just wasted host-side work proportional to n_pair.

Confidence: 100%.

---

## L-1 — `math_to_full_idx` returned but never used

**File / line:** `src/compute_frustration.py:646`

```python
coords, math_to_full_idx = _subset_protein_only(coords_full)
```

`math_to_full_idx` is captured and never referenced. The actual emit-row projection uses `coords_full["lammps_emit_rows"]` instead (line 983). If the index is intended for future per-residue audits, fine; otherwise the unused return adds noise. Drop or use.

Confidence: 100%.

---

## L-2 — `FrustrationResult` dataclass uses `Any` for dataframe fields

**File / line:** `src/compute_frustration.py:115-118`

```python
pair_records: Optional[Any] = None        # pandas.DataFrame
```

The lazy-pandas-import is intentional and good (the rest of the module is pandas-free at import time), but `Any` defeats type checkers / IDE help. A `TYPE_CHECKING` guarded import would give `pd.DataFrame` typing for free. Cosmetic.

Confidence: 100%.

---

## L-3 — Minor docstring drift

- `src/compute_frustration.py:115-118` docstring of `FrustrationResult` lists field `r_ij`, `NativeEnergy`, etc; the actual `_build_pair_dataframe` emits the same set — consistent. OK.
- `src/frustration.py:13-14` header docstring says `FI <= -1.0 → highly`, `-1.0 < FI < 0.78 → neutral`, `FI >= 0.78 → minimally`. The code at line 173-174 implements `<=` and `>=` — consistent with the header. Note the per-function docstring of `classify_frustration` (lines 165-167) says `-1.0 < FI < 0.78 → neutral` (strict on both sides); the implementation makes `FI == -1.0 → highly` and `FI == 0.78 → minimally` (i.e. boundaries are NOT in neutral). This is correct per Ferreiro 2007 and matches RenumFiles.pl. OK.
- `src/density.py:107` mentions the R script's asymmetric inclusive/exclusive convention `<= (-1) / > (-1) & < (0.78) / >= 0.78`. Implementation at lines 164-166 is `fi <= high_thr`, `fi >= minimal_thr`, with `neutral = (~highly) & (~minimally)` — boundaries on -1 are "highly" and boundaries on 0.78 are "minimally". Matches R script. OK.

---

## Per-checklist responses

1. **API consistency — `compute_frustration` vs `calculate_frustration`.**
   - Defaults match: `mode="configurational"`, `electrostatics_k=None`, `include_dh_in_e_native=False`, `seq_dist=12`, `n_decoys=1000`, `device="auto"`, `seed=0`, `precision=3`, all compat flags False. ✓
   - All compat flags forwarded (line 1205-1207). ✓
   - **Drift**: `dtype` (default `torch.float64`) is NOT exposed on the adapter — frustrapy callers can't pass `dtype=torch.float32`. Minor; documented frustrapy doesn't expose it either, so acceptable.
   - **Drift**: `pair_min_seq_sep` not exposed on adapter (compute_frustration default `2` is used silently). Minor.
   - Multi-chain handling differs (see H-2).

2. **Four LAMMPS-compat flags wired correctly.**
   - `lammps_compat_altloc`, `include_dna`, `keep_incomplete_backbone` all forwarded to `parse_pdb` at lines 623-631. ✓
   - `include_dh_in_e_native` consumed at lines 680-690 (configurational) and 784-794 (mutational). ✓
   - Singleresidue case correctly emits a RuntimeWarning when both `electrostatics_k` and `include_dh_in_e_native` are set (lines 845-859). ✓
   - **Interaction check**: when `include_dna=True` and `lammps_compat_altloc=True` are both on, `_subset_protein_only` strips DNA + altloc-B from math, then `_project_density_to_lammps_emit` re-projects density onto the full emit list. The `or` at line 731 / 747 / 816 / 838 correctly enables the projection on either flag. ✓
   - **Interaction check**: `keep_incomplete_backbone=True` combined with `lammps_compat_altloc=True` is plausible — altloc-B shadows inherit missing backbone from altloc-A (`_inherit_backbone_to_altloc_b` at parser.py:523), so the strict backbone filter doesn't kill the shadow. ✓
   - **Metadata reports all 4 flags** (lines 935-937). ✓

3. **DH-in-E_native semantics.** Default OFF (lines 680-690 require `include_dh_in_e_native=True` AND `electrostatics_k is not None` AND `n_pairs > 0` to fire). Matches the LAMMPS-AWSEM analysis-pipeline convention documented in `lammps_compat_fixes.md`. ✓

4. **Pair ordering `i < j`.**
   - Configurational orchestrator: `idx.unsqueeze(1) < idx.unsqueeze(0)` at line 267 (rows < cols) → pair_i < pair_j. ✓
   - Mutational decoys via `_enumerate_native_pairs` in `decoys.py:437`: same form, pair_i < pair_j. ✓
   - Singleresidue: per-residue, no pair ordering relevant. ✓
   - The P1 fix landed correctly in both pipelines.

5. **Dump emit correctness.**
   - CB-vs-CA on Gly: `_xb_coords` at `frustration.py:251-260` picks CB for non-Gly, CA for Gly. ✓ (modulo H-1 latent NaN-axis bug)
   - Welltype rule (`r_ij < 6.5 → short`; `r_ij >= 6.5 AND rho_i < 2.6 AND rho_j < 2.6 → water-mediated`; else `long`): implemented at `frustration.py:178-199`. Matches `RenumFiles.pl`. ✓
   - Column count + order: tertiary-frustration dump at lines 350-358 produces 19 fields (matches LAMMPS `fix_backbone.cpp:5104`). ✓
   - Post-processed pair dump produces 14 fields (Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2 NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState) — matches frustratometeR. ✓
   - Singleresidue raw vs post-processed both implemented (lines 444-466). ✓

6. **Density aggregation.**
   - Sphere radius: `DEFAULT_DENSITY_RATIO_A = 5.0`. ✓ Matches XAdens default.
   - Distance metric: midpoint between effective-CB (xb) coords of pair i and pair j, sphere CENTER = CA of residue i. Both match the R reference (`density.py:131-132` honours `CA_xyz` not `xb`). ✓
   - Tie-handling: classification is `<= -1.0 → highly`, `>= 0.78 → minimally`, else neutral. Boundary on the high side is inclusive-highly; boundary on the minimal side is inclusive-minimally. Matches R script. ✓
   - Ratio guard for `Total == 0`: line 177-188, returns 0.0. Matches R `if(total_density > 0)`. ✓

7. **`FrustrationResult` dataclass.**
   - configurational mode → `pair_records`, `density_records` populated; `singleresidue_records=None`. ✓
   - mutational mode → same. ✓
   - singleresidue mode → `singleresidue_records` populated; `pair_records=None` (line 658 `pair_df = None`), `density_records=None` (line 660). ✓ — matches docstring at lines 100-108.
   - Empty `n_pairs == 0` branches build empty DataFrames (line 738-743 for configurational, line 834 `pd.DataFrame()` for mutational — **note the asymmetry**: configurational gets a column-named empty DF, mutational gets a column-less one. Cosmetic; would surface if caller tries `df["FrstIndex"]` on a mutational empty result. Latent edge case.)

8. **Error handling.**
   - Mode validation: `mode not in _VALID_MODES → ValueError` at line 610-611. ✓
   - Negative precision: `ValueError` at line 612-613. ✓
   - Empty parse: `ValueError` at lines 633-637 (full) and 648-652 (protein subset). ✓
   - **Missing chain**: silently passes `chains=[chain]` to parser; parser raises if no residues match. Adequate.
   - **Nonexistent residues**: silently filters to empty. See M-4.
   - **Unknown kwargs in adapter**: `TypeError` at lines 1182-1190. ✓

9. **Public API surface.**
   - `__init__.py` exports `compute_frustration`, `calculate_frustration`, `FrustrationResult` plus all dump writers, classification helpers, density helpers. ✓
   - Three private helpers (`_density_to_df`, `_project_density_to_lammps_emit`, `_emit_pair_files`) correctly kept module-private. ✓
   - No leaking helpers like `_subset_protein_only`, `_aa_letters`, `_configurational_native_pairs`, `_add_dh_to_e_native`, `_build_pair_dataframe`, `_build_singleresidue_dataframe` — all stay module-private. ✓
   - `density_to_dataframe` (density.py:203) is exported but `_density_to_df` (compute_frustration.py:951) is a near-duplicate with slightly different schema (both produce identical columns — duplicates the work). Minor opportunity for consolidation.

10. **Threading / async.**
   - No `torch.cuda.synchronize()` anywhere in the orchestrator. See M-1.
   - All tensor ops are single-stream; no `cuda.Stream()` usage; no async copies. Single-threaded execution model — safe but leaves perf on the table on CUDA.

## Surprises (good and bad)

- **Good**: the `_subset_protein_only` / `math_to_full_idx` / `_project_density_to_lammps_emit` separation between math view and emit view is **clean** — math runs on protein-only, density is reprojected back onto the LAMMPS-bug-emit-list. This is the kind of architectural decision that prevents the next dozen bugs.
- **Good**: the singleresidue + DH guard at lines 845-859 emits a RuntimeWarning rather than silently ignoring `include_dh_in_e_native=True`. Defensive and well-justified.
- **Bad**: H-1's per-axis NaN mask in `_xb_coords` is a latent footgun. The parser hides it today, but two modules (`_resolve_contact_coords` vs `_xb_coords`) use inconsistent conventions for the same problem.
- **Bad**: the timer metadata is documented as wall-clock but isn't CUDA-synchronised.
- **Bad**: the multi-chain adapter path silently changes rho semantics — would surprise a user comparing `chain="A"` to `chain=["A","B"]` outputs.
- **Surprising**: configurational mode re-implements its native-pair loop inline (`_configurational_native_pairs`) rather than reusing `mutational_decoy_stats(n_decoys=0)`. The author justifies this in the docstring (line 220-223) as a perf win — fair, but worth a regression test that both paths produce the same `(pair_i, pair_j, r_ij, E_native)` for identical inputs (test panel only checks output-level Spearman, not per-pair byte equality between modes).

## Recommended next-sprint priorities

1. **H-1 (5 LOC)** — replace per-element NaN mask in `_xb_coords` with the per-row pattern from `_resolve_contact_coords`. Risk: zero (latent today). Adds defensive consistency.
2. **H-2 (10 LOC)** — widen `compute_frustration`'s `chain` kwarg to accept `list[str]`, forward through to `parse_pdb`. Eliminates the silent rho-semantics divergence.
3. **M-4 (10 LOC)** — validate `residues` subset against actual `(chain, resnum)` set at entry; warn on missing entries.
4. **M-1 (4 LOC)** — `torch.cuda.synchronize()` around the timer.
5. **M-2/M-3/M-5** — vectorisation cleanups, batched together when someone profiles a 5000-res PDB.

No CRITICAL fixes needed. Code is shippable as-is.
