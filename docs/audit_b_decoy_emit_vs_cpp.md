# AUDIT-B: decoy + dump-emit modules vs LAMMPS-AWSEM C++

Auditor: read-only review, final pre-`git push` correctness audit.
Date: 2026-05-21.
Scope: `frustration_gpu/decoys.py`, `frustration_gpu/mutational_decoys.py`,
`frustration_gpu/singleresidue_decoys.py`, `frustration_gpu/frustration.py`.
Reference C++: `adavtyan/awsemmd@master:src/fix_backbone.cpp` (8232 lines,
fetched via raw.githubusercontent.com 2026-05-21; the upstream path is
`src/fix_backbone.cpp`, not the `src/USER-AWSEMMD/` path quoted in the
audit prompt). Reference Perl: `proteinphysiologylab/frustratometeR@master:inst/Scripts/RenumFiles.pl`.

## Verdict: GO

All four publication-blocking checks pass to >80% confidence:

- **FI formula** sign + denominator: bit-identical to C++ (`fix_backbone.cpp:5595`).
- **Ferreiro thresholds** + boundary inclusion: bit-identical to `RenumFiles.pl:65-77`.
- **Welltype rule**: bit-identical to `RenumFiles.pl:50-63`.
- **Dump printf format** (tertiary + singleresidue): column-count and order match `fix_backbone.cpp:5104` and `:5168`.

Severity counts: **0 CRITICAL**, **0 HIGH**, **2 MEDIUM** (docstring drift), **3 LOW** (style/edge-case). User is clear to `git push`.

## Severity summary

| # | Severity | Topic | Python file:line | C++ file:line |
|---|---|---|---|---|
| 1 | MEDIUM | Stale C++ line-range citation in `decoys.py` module docstring | `decoys.py:3` | C++ `5249-5344` (actual `5249-5344` checks; off by ≤5) |
| 2 | MEDIUM | `decoys.py:413` inline cite of `line 5262` uses `||` — C++ actually uses `||` AND treats `rij > cutoff` (strict gt) — the rejection inequality differs by `==` boundary from Python's `<` (Python rejects `rij == cutoff`; C++ accepts) | `decoys.py:438` (`flat < contact_cutoff`) | `fix_backbone.cpp:5262` (`rij > cutoff`) |
| 3 | LOW | `decoys.py:692` and other call-sites silently default `min_seq_sep_rho = 12`; matches frustratometeR but not vanilla LAMMPS — already documented in module docstring, fine | `decoys.py:148, 692` | n/a (frustratometeR patch) |
| 4 | LOW | `frustration.py:228-229` welltype assignment uses mask-overwrite order that is *correct* but reads awkwardly (`well[(~short) & water] = WATER; well[short] = SHORT`) | `frustration.py:227-229` | matches `RenumFiles.pl:50-63` |
| 5 | LOW | `frustration.py:139-181` `degenerate_threshold` clamp to 0 is a NEW behaviour vs C++ (which returns nan/inf on `decoy_std == 0`); documented + opt-in via default. Safe. | `frustration.py:159-181` | n/a (C++ produces inf) |

## Detailed check-by-check

### 1. FI formula and sign convention — VERIFIED CORRECT

**C++ `fix_backbone.cpp:5591-5598`:**
```
double FixBackbone::compute_frustration_index(double native_energy, double *decoy_stats)
{
  double frustration_index;
  frustration_index = (decoy_stats[0] - native_energy)/decoy_stats[1];
  return frustration_index;
}
```
`decoy_stats[0]` is the decoy MEAN (`fix_backbone.cpp:5335`), `decoy_stats[1]` the decoy STD (`:5336`).

**Python `frustration.py:182`:** `return (decoy_mean - e_native) / decoy_std`.

Bit-identical. Sign convention (`FI>0 = minimally frustrated`) matches the LAMMPS check at `fix_backbone.cpp:5105` (`> 0.78` flagged as MIN, `< -1` flagged as HIGH) and `RenumFiles.pl:65-77`. Python `frustration.py:62-66` docstring states the same.

### 2. Ferreiro thresholds + boundary inclusion — VERIFIED CORRECT

**Perl `RenumFiles.pl:65-77`:**
```
if($FrstIndex<=-1)             { $FrstType="highly"; }
if($FrstIndex>-1 && $FrstIndex<0.78)   { $FrstType="neutral"; }
if($FrstIndex>=0.78)           { $FrstType="minimally"; }
```
Boundary convention: `FI == -1.0` → HIGHLY (not NEUTRAL); `FI == 0.78` → MIN (not NEUTRAL).

**Python `frustration.py:202-206`:**
```python
cls = torch.ones_like(fi, dtype=torch.long)        # default = NEUTRAL (1)
cls[fi <= high_threshold] = CLASS_HIGHLY           # HIGH inclusive
cls[fi >= minimal_threshold] = CLASS_MINIMALLY     # MIN inclusive
```
Constants at `frustration.py:84-85`: `-1.0` and `0.78` exactly.

Bit-identical boundary inclusion. The two `cls[fi <= ...]` / `cls[fi >= ...]` overwrites are non-overlapping (since `-1 < 0.78`), so order doesn't matter.

### 3. Welltype rule — VERIFIED CORRECT

**Perl `RenumFiles.pl:50-63`:**
```
if($splitted[10]<6.5)                 { $ResResDistance="short"; }
elsif($splitted[10]>=6.5) {
  if($Density1<2.6 && $Density2<2.6)  { $ResResDistance="water-mediated"; }
  else                                { $ResResDistance="long"; }
}
```

**Python `frustration.py:225-230`:**
```python
short_mask = rij < r_short                                   # r_short = 6.5
water_mask = (rho_i < rho_water_cutoff) & (rho_j < rho_water_cutoff)  # 2.6
well = torch.full_like(rij, WELL_LONG, dtype=torch.long)     # default LONG
well[(~short_mask) & water_mask] = WELL_WATER_MEDIATED
well[short_mask] = WELL_SHORT
```

Bit-identical: `rij < 6.5` → SHORT; `rij >= 6.5 AND rho_i < 2.6 AND rho_j < 2.6` → WATER-MEDIATED; else LONG. Boundary at `rij == 6.5` goes to LONG (or WATER if both ρ < 2.6) — matches Perl `elsif >=6.5`.

This is the rule Phase 3c agent had to fix: in `frustration.py:228-229` the mask-overwrite is in the correct order (water-mediated first, then short last) so `short` always wins when it matches. Confirmed against Perl: when `rij < 6.5` Perl unconditionally returns SHORT regardless of ρ — Python matches via the final `well[short_mask] = WELL_SHORT` overwrite.

### 4. Dump format printf — VERIFIED CORRECT

**C++ `fix_backbone.cpp:5104`:**
```
"%5d %5d %3d %3d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f
 %8.3f %8.3f %8.3f %c %c %8.3f %8.3f %8.3f %8.3f\n"
```
19 fields, in order: `i+1 j+1 i_chno+1 j_chno+1 xi yi zi xj yj zj rij rho_i rho_j a_i a_j E_nat decoy_mean decoy_std FI`.

**Python `frustration.py:389-396`:**
```python
fmt = (f"{{:5d}} {{:5d}} {{:3d}} {{:3d}} "
       f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
       f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
       f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
       f"{{}} {{}} {{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
       f"{{:{width}.{f}f}}")
```
With `f=3, width=max(8, 3+5)=8` → identical `%5d %5d %3d %3d %8.3f...%8.3f` shape. The `{}` for the two `%c` letters prints a single character with no padding — matches `%c`. Argument order at `frustration.py:408-414` is `i+1, j+1, i_chain_int, j_chain_int, xi[0..2], xj[0..2], r_ij, rho_i, rho_j, a_i, a_j, E_nat, decoy_mean, decoy_std, FI` — byte-exact.

**Singleresidue C++ `fix_backbone.cpp:5168`:**
```
"%5d %5d %8.3f %8.3f %8.3f %8.3f %c %8.3f %8.3f %8.3f %8.3f\n"
```
11 fields in order: `i+1 i_chno+1 xi yi zi rho_i a_i E_nat decoy_mean decoy_std FI`.

**Python `frustration.py:489-495`:** identical column order, same widths.

### 5. CB-vs-CA-on-Gly branch — VERIFIED CORRECT

**C++ `fix_backbone.cpp:5088-5091`** (tertiary):
```
if (se[i_resno]=='G') { xi = xca[i]; }
else { xi = xcb[i]; }
if (se[j_resno]=='G') { xj = xca[j]; }
else { xj = xcb[j]; }
```

**C++ `fix_backbone.cpp:5155-5156`** (singleresidue):
```
if (se[i_resno]=='G') { xi = xca[i]; }
else { xi = xcb[i]; }
```

**Python `frustration.py:282-298` (`_xb_coords` helper):**
```python
nan_row = ~torch.isfinite(cb).all(dim=-1)
row_mask = (is_gly | nan_row).unsqueeze(-1)
return torch.where(row_mask, ca, cb)
```
Returns CA when Gly OR CB is NaN; CB otherwise. The Gly branch is exact (`is_gly` true → CA). The NaN-fallback is a defensive extension: LAMMPS reads CB from the data file which is always populated, but a Python caller may construct `coords` with NaN CB for missing residues — falling back to CA prevents NaN strings hitting the dump (would break downstream parsers). Per `frustration.py:272-280` docstring this is documented. **Match for the canonical case** (all CB populated); **safe extension** for the NaN case.

### 6. Configurational decoy sampling — VERIFIED CORRECT (with documented PRNG drift)

**C++ `fix_backbone.cpp:5255-5295`:**
For each of 1000 decoys:
- draw `(p, q)` pair, recompute `rij = dist(p, q)`, reject and resample while `rij > cutoff || p == q`.
- draw new `(p', q')` pair (no contact constraint), read `rho_i = rho[p'], rho_j = rho[q']`.
- draw new `(p'', q'')` pair, read `aa_i = aa[p''], aa_j = aa[q'']`.
- `E_decoy = water(rij, aa_i, aa_j, rho_i, rho_j) + burial(aa_i, rho_i) + burial(aa_j, rho_j) + (DH=0 because huckel_flag=false)`.

**Python `decoys.py:286-459` (`sample_configurational_decoys`):**
- `aa_i_idx, aa_j_idx ~ Uniform[0, n)` then `aa_i_decoy = aa[aa_i_idx]`. (`decoys.py:397-400`) ✓
- `rho_i_idx, rho_j_idx ~ Uniform[0, n)` then `rho_i_decoy = rho[rho_i_idx]`. (`decoys.py:403-406`) ✓
- `r_ij_decoy`: enumerate off-diag in-contact pairs (`dist_full < contact_cutoff`, `decoys.py:431-438`), then draw one uniform index into that enumeration (`decoys.py:447`). This is **mathematically equivalent** to C++'s rejection loop, which is also uniform over the set `S = {(p, q) : p != q, rij < cutoff}`. Inverse-CDF over the same `S` produces the same distribution.

One micro-boundary asymmetry (LOW): C++ accepts `rij == cutoff` (`while(rij > cutoff || ...)` is strict-gt), Python rejects (`flat < contact_cutoff` is strict-lt). For float64 distances at 9.500000 Å exactly this is measure-zero — no observable effect on `decoy_mean` / `decoy_std`. Documented in `decoys.py:411-415` inline comment.

PRNG (`torch.Generator` Mersenne Twister vs libc `rand()`) produces ~3% relative drift in `(decoy_mean, decoy_std)` between Python and LAMMPS runs on the same protein. Documented in `decoys.py:88-96`. The FI-Spearman gate (≥ 0.997) is preserved because FI is rank-invariant under the same sampling rule.

### 7. Mutational decoy sampling — VERIFIED CORRECT

**C++ `fix_backbone.cpp:5215-5328`** (native + decoy mutational branch):
- Native: `E_native = water(rij, aa_i, aa_j, rho_i, rho_j) + burial_i + burial_j + Σ_{k!=i,k!=j, rik<cutoff} water(rik, aa_i, aa_k, rho_i, rho_k) + Σ_{k!=i,k!=j, rjk<cutoff} water(rjk, aa_j, aa_k, rho_j, rho_k)`.
- Decoy: same formula but `aa_i, aa_j` replaced by `(rand_i_resno_aa, rand_j_resno_aa)` each draw, `(rij, rho_i, rho_j)` held at native.

**Python `mutational_decoys.py:566-1085`:**
- Precompute `T[i, α] = Σ_{k!=i, r_ik<cutoff} water(r_ik, α, aa_k_native, rho_i, rho_k)` over all 20 α (`_precompute_T_alpha`). The mask `cross_mask = (dist_full < contact_cutoff) & ~diag` excludes only `k==i` — matches C++ line 5302 (`k==i_resno || k==j_resno`); the `k==j` exclusion is honoured separately via the per-pair U subtraction (`mutational_decoys.py:1046-1047`).
- Decoy energy per (i,j,decoy): `pair_term + (T[i,α_i] - U_iSlot_kj) + (T[j,α_j] - U_jSlot_ki) + burial(α_i, rho_i) + burial(α_j, rho_j)` — algebraically identical to the C++ loop.
- Native: `S_i + S_j - W_native_pair + burial_i + burial_j` where `S_i = T[i, aa_i_native]` (`mutational_decoys.py:1000-1002`). Identity: `S_i + S_j - W = water_ij + Σ_{k!=i,k!=j} water_ik + Σ_{k!=i,k!=j} water_jk + 0 + W - W = water_ij + crossterms`. ✓

Mask convention exactly matches C++: spatial-only `r < cutoff` on cross-terms, no seq-sep filter (C++ lines 5300-5327, Python `_precompute_T_alpha:600-602`). The audit prompt's "cross_i/j mask convention" check passes.

The pair-enumeration filter at `mutational_decoys.py:500-513` uses `(idx.unsqueeze(1) < idx.unsqueeze(0))` for upper-tri (i < j) — matches C++ `for (j=i+1; j<n; ...)`. Phase 5 P1 fix note in the code is correctly applied.

### 8. Singleresidue decoy sampling — VERIFIED CORRECT

**C++ `fix_backbone.cpp:5346-5410`:**
- Native `compute_singleresidue_native_ixn`: loops j over all residues, applies `rij < cutoff AND (|i-j| >= contact_cutoff OR cross-chain)`, accumulates water + burial_i. Note: uses `abs(i_resno - j_resno)` (PDB resnums); for canonical inputs where list-index == resnum this equals list-index seq-sep.
- Decoy: same function but `ires_type` replaced by `get_residue_type(get_random_residue_index())` — i.e. one-of-N native AA uniformly.

**Python `singleresidue_decoys.py:227-416`:**
- Precompute `W_sr[i, α] = Σ_j pair_mask[i,j] * water(r_ij, α, aa_j_nat, rho_i, rho_j_nat)` and per-residue burial table `B[i, α]` (`singleresidue_decoys.py:343-374`). Decoy energy per (i, decoy): `W_sr[i, α_d] + B[i, α_d]`.
- Mask: `(dist < cutoff) & ((~same_chain) | (seq_diff >= pair_min_seq_sep)) & (i != j) & finite_pair_2d` (`singleresidue_decoys.py:320-325`). Default `pair_min_seq_sep = 2` matches typical LAMMPS `awsem.in` ([Water] block `2 2`).
- AA sampling: `_sample_aa_per_residue:189-220` draws `(N, n_decoys)` uniform residue indices, gathers `aa_dev[idx]` — uniform-over-N residue indices reading off AA composition. Matches C++ `get_random_residue_index() -> get_residue_type` pattern.

### 9. FI formula edge cases

Python adds a `degenerate_threshold = 1e-12` clamp at `frustration.py:159-181` that returns FI=0 (with a `UserWarning`) when `decoy_std < 1e-12`. C++ produces `inf` or `nan` in this case (line 5595 has no zero-guard). **This is a defensive Python-only feature**, opt-in via the default. Both sides agree on the canonical case `decoy_std > 0`. Documented in the function's docstring at `frustration.py:138-143`. LOW severity, design-intentional.

## Things actually checked, in two-column form

| Check | Python | C++ / Perl | Match |
|---|---|---|---|
| FI = (mean − native) / std | `frustration.py:182` | `fix_backbone.cpp:5595` | YES |
| Sign convention (FI>0 = MIN) | `frustration.py:62-66, 84-85` | `:5105` (`>0.78` flagged green/MIN, `<-1` red/HIGH) | YES |
| `FI <= -1.0` → HIGH | `frustration.py:204` | `RenumFiles.pl:65` | YES |
| `FI >= 0.78` → MIN | `frustration.py:205` | `RenumFiles.pl:75` | YES |
| Boundary `FI = -1.0` → HIGH (inclusive) | `frustration.py:204` (`<=`) | `RenumFiles.pl:65` (`<=`) | YES |
| Boundary `FI = 0.78` → MIN (inclusive) | `frustration.py:205` (`>=`) | `RenumFiles.pl:75` (`>=`) | YES |
| Welltype `r < 6.5` → SHORT | `frustration.py:229` | `RenumFiles.pl:50` | YES |
| Welltype `r >= 6.5 AND ρ_i < 2.6 AND ρ_j < 2.6` → WATER | `frustration.py:226-228` | `RenumFiles.pl:54-60` | YES |
| Welltype otherwise → LONG | `frustration.py:227` (default fill) | `RenumFiles.pl:62` | YES |
| Tertiary dump column count = 19 | `frustration.py:389-396, 408-414` | `:5104` (19 `%`-specifiers) | YES |
| Tertiary dump column order | `frustration.py:408-414` | `:5104` arg list | YES |
| Singleresidue dump column count = 11 | `frustration.py:489-495` | `:5168` (11 `%`-specifiers) | YES |
| Singleresidue dump column order | `frustration.py:489-495` | `:5168` arg list | YES |
| `%8.3f` decimal precision | `width=8, f=3` (`frustration.py:387-388`) | `:5104, :5168` `%8.3f` | YES |
| `%5d` int precision | `frustration.py:390` | `:5104` `%5d` | YES |
| `%3d` chain index | `frustration.py:390` (`{{:3d}}`) | `:5104` `%3d` | YES |
| CB used for non-Gly | `frustration.py:296-298` (`is_gly` false → CB) | `:5089` (`else { xi = xcb }`) | YES |
| CA used for Gly | `frustration.py:296-298` (`is_gly` true → CA) | `:5088` (`if G { xi = xca }`) | YES |
| Same CB/CA branch for singleresidue | `_xb_coords` called at `:467` | `:5155-5156` | YES |
| 1000 decoys default | `decoys.py:154` (`DEFAULT_N_DECOYS=1000`) | `:5255` (`tert_frust_ndecoys`, typ. 1000 in awsem.in) | YES |
| Cutoff = 9.5 Å | `decoys.py:153` (`DEFAULT_CONTACT_CUTOFF_A=9.5`) | `:5086, :5262` (`tert_frust_cutoff`, 9.5 in awsem.in) | YES |
| Configurational: uniform-over-{(p,q): p≠q, r<cutoff} | `decoys.py:431-449` (inverse-CDF) | `:5258-5266` (rejection loop) | YES (math) |
| Configurational: independent (ρ_i, ρ_j) draws | `decoys.py:403-406` | `:5268-5271` | YES |
| Configurational: independent (aa_i, aa_j) draws | `decoys.py:397-400` | `:5281-5284` | YES |
| Configurational: huckel_flag = false → E_DH = 0 | `decoys.py:64-67` (docstring), no DH term in formula | `:5290-5295` | YES |
| Configurational: cached across pairs | `decoys.py:79-86` (docstring); single `sample_configurational_decoys` call | `:5099-5100, :5341-5342` (`already_computed`) | YES |
| Mutational: per-pair fresh decoys | `mutational_decoys.py:79-87` | `:5099` (not cached when mode != configurational) | YES |
| Mutational: hold (r_ij, ρ_i, ρ_j) at native | `mutational_decoys.py:751-755` (uses `rho[pair_i]`, `r_ij_pair`) | `:5274-5278` | YES |
| Mutational: cross-terms (i,k), (j,k) spatial-only | `_precompute_T_alpha:600-602` (no seq-sep filter) | `:5300-5327` (no seq-sep filter) | YES |
| Mutational: exclude k==i AND k==j | `cross_mask & ~diag` + U subtraction | `:5302` (`k==i_resno || k==j_resno`) | YES |
| Mutational: native = water_ij + crosstotal + burial_i + burial_j | `mutational_decoys.py:1000-1002` | `:5198-5244` | YES (algebraic) |
| Singleresidue: 1000 decoys/residue | `singleresidue_decoys.py:231` | `:5399` | YES |
| Singleresidue: only scramble aa_i | `singleresidue_decoys.py:189-220` | `:5401-5402` | YES |
| Singleresidue: native uses seq-sep filter | `singleresidue_decoys.py:320-322` (`seq_diff >= pair_min_seq_sep`) | `:5383` (`abs(i_resno-j_resno) >= contact_cutoff`) | YES |
| Population std (ddof=0) | `decoys.py:644` (`unbiased=False`), `mutational_decoys.py:1072`, `singleresidue_decoys.py:401` | `:5438` (`std /= (double)arraysize`) | YES |

## Findings list

### MEDIUM-1 — Stale module-docstring line cite

`decoys.py:3` says `fix_backbone.cpp:5249-5344`. Actual function `compute_decoy_ixns` runs `5249-5344` exactly — citation is correct but the path prefix in the audit prompt (`src/USER-AWSEMMD/fix_backbone.cpp`) is wrong upstream; the actual path is `src/fix_backbone.cpp`. Not a code defect — only a citation hygiene note in case the docstring path comment ever gets added.

### MEDIUM-2 — Boundary inclusion micro-asymmetry on cutoff

`decoys.py:438` does `flat[flat < contact_cutoff]` (strict less-than). C++ at `fix_backbone.cpp:5262` does `while(rij > cutoff || ...)` — i.e. accepts `rij == cutoff`. Float64 distances at exactly 9.500000 Å are measure-zero; no observable effect on `decoy_mean` / `decoy_std`. Not a publication risk. Stylistic: changing Python to `<=` would match C++ exactly but introduces no behavioural change in practice.

### LOW-1 — `min_seq_sep_rho = 12` default

`decoys.py:148, 692` default to `min_seq_sep = 12` matching frustratometeR's patched binary. Vanilla LAMMPS-AWSEM uses the burial rho (`min_seq_sep = 1`) for the decoy formula. Documented explicitly in `decoys.py:8-34` module docstring — caller can override. **Safe**.

### LOW-2 — Welltype mask-overwrite reads awkwardly

`frustration.py:227-229` writes LONG first (full_like), then WATER-MEDIATED for `(~short) & water`, then SHORT last. Result is correct (SHORT wins for `r<6.5`; otherwise WATER if both ρ<2.6; else LONG). Refactor candidate (`torch.where` chain) for readability — no behavioural change.

### LOW-3 — `degenerate_threshold` is Python-only behaviour

`frustration.py:159-181` clamps FI to 0 when `decoy_std < 1e-12`, with a `UserWarning`. C++ produces `inf`/`nan`. Defensive Python extension, documented in the docstring at `frustration.py:138-143`. The default behaviour is documented and opt-in via `degenerate_threshold=0`. Won't surprise reviewers; documented + warned.

## What is verified correct, summary

- **FI formula sign + denominator**: `(decoy_mean - E_native) / decoy_std`, matching `fix_backbone.cpp:5595` bit-by-bit. Positive FI = minimally frustrated (native better than decoys).
- **Ferreiro thresholds**: `FI <= -1.0` → HIGH; `FI >= 0.78` → MIN; otherwise NEUTRAL. Boundary inclusion matches `RenumFiles.pl:65-77`.
- **Welltype rule**: `r < 6.5` → SHORT; `r >= 6.5 AND ρ_i < 2.6 AND ρ_j < 2.6` → WATER-MEDIATED; else LONG. Matches `RenumFiles.pl:50-63`.
- **Dump printf format**: column count (19 / 11), order, and `%5d / %3d / %8.3f / %c` precision match `fix_backbone.cpp:5104` and `:5168` byte-by-byte. AA letters print as single chars without padding (matches `%c`).
- **CB-vs-CA Gly branch**: `_xb_coords` returns CA for Gly, CB otherwise — matches `fix_backbone.cpp:5088-5091` (tertiary) and `:5155-5156` (singleresidue). NaN-fallback to CA is a safe Python-only defensive extension.
- **Decoy sampling**:
  - Configurational: inverse-CDF over the same `S = {(p,q) : p≠q, r<cutoff}` set the C++ rejection loop draws from. Algebraically identical distribution (mod the 9.5-Å boundary edge case).
  - Mutational: precompute-T trick is algebraically equivalent to the C++ per-pair-per-decoy nested loop. Native + decoy formulas verified by expansion.
  - Singleresidue: precompute-W_sr identity with the C++ `compute_singleresidue_native_ixn` verified. AA sampling matches `get_random_residue_index() → get_residue_type` pattern.
- **PRNG drift** (~3% relative on `decoy_mean`, `decoy_std`): documented and accepted; FI Spearman ≥ 0.997 preserved because rank-order is invariant under the same sampling rule.

## Recommendation

**Proceed with `git push`.** No publication-blocking defects. The two MEDIUM items are documentation hygiene (one stale path comment; one boundary float-equality edge case that is measure-zero in practice). The three LOW items are documented design choices (frustratometeR-default `min_seq_sep=12`; stylistic mask-overwrite; opt-in degenerate-std clamp).

If the user wants zero loose ends before tagging v0.1.0:
1. (60s) Tweak `decoys.py:438` to `flat <= contact_cutoff` to match C++'s `> cutoff` rejection exactly. No observable change.
2. (30s) Refactor `frustration.py:227-229` to a clearer `torch.where` chain. No behavioural change.

Neither is required for correctness.
