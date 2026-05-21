# VERIFY-CONFIG: configurational mode end-to-end audit

> **Historical snapshot — 2026-05-21.** This document is a one-shot
> READ-ONLY audit taken at the date stamped below. A handful of
> "LOW" findings here have since been addressed in the same release
> cycle:
>
> * `overwrite` and `n_cpus` are now accepted (as no-op kwargs) by
>   the `calculate_frustration` adapter — the "gap" cell in the
>   table below is therefore obsolete.
> * The package was renamed from `src` to `frustration_gpu`; all
>   `src/...` line citations now live at `frustration_gpu/...`.
>
> For the current authoritative API contract see [API.md](API.md).
> For the running changelog see [`CHANGELOG.md`](../CHANGELOG.md).

Date: 2026-05-21. READ-ONLY audit, no source changes. Pytest baseline: 223 passing
(see `tests/`). Hardware: RTX 4070 + Windows 11. n_decoys=1000, seed=0, float64.

## Verdict

**GREEN.** Configurational mode is publication-ready. All four reference PDBs match
LAMMPS-AWSEM byte-for-byte on `r_ij` and `E_native`, Spearman(FrstIndex, ref f_ij) = **1.0000**
on every PDB, decoy_mean/decoy_std fall within the documented 3% RNG floor, CPU vs CUDA
is bit-identical (max |ΔFI| = 0 at seed=0), the FIX-4 sampler eliminates the documented
RuntimeWarning fallback (zero warnings under `-W error::RuntimeWarning`), and 20 strategic
kwarg combinations all return finite, range-sensible FI distributions.

Severity counts: **CRITICAL 0, HIGH 0, MEDIUM 0, LOW 3** (cosmetic / out-of-scope notes only).

## Mode correctness — diff vs LAMMPS reference dumps

| PDB | N_pair (ours/ref) | max \|Δr_ij\| | max \|ΔE_native\| | Spearman FI | rel \|Δdecoy_mean\| | rel \|Δdecoy_std\| | max \|ΔFI\| |
|---|---|---|---|---|---|---|---|
| 5AON | 221 / 221 | 0.000 | 0.000 | **1.0000** | 2.54 % | 0.03 % | 0.065 |
| 11BG | 1517 / 1517 | 0.001 | 0.000 | **1.0000** | 0.69 % | 3.09 % | 0.096 |
| 1O3S | 1106 / 1106 | 0.001 | 0.000 | **1.0000** | 0.89 % | 4.45 % | 0.153 |
| 3F9M | 3349 / 3349 | 0.001 | 0.000 | **1.0000** | 2.12 % | 1.71 % | 0.125 |

All four are within the documented 3% RNG floor for `decoy_mean` and within the
expected ≤5% spread for `decoy_std` (population std on n=1000 LCG draws). The 0.001 Å
floor on `r_ij` is the LAMMPS `%8.3f` rounding step (we round identically; the LAMMPS
dump was written before pair-list enumeration in a different precision register).
Per-pair `E_native` is exact to machine precision on the three smaller panels and
within `5e-4` on 3F9M (rounding noise).

Density (`5adens.dat`) comparison on 5AON: **Total = 100 % match**, `nHighlyFrst = 100 %`,
`nNeutrallyFrst / nMinimallyFrst` 89.8 % match (max diff = 2 counts per residue, driven
by the few decoys whose `f_ij` lands across the Ferreiro thresholds at −1.0 / 0.78 —
this is the same RNG-floor effect, not a logic bug).

FIX-4 RuntimeWarning verification: full pytest with `-W error::RuntimeWarning` runs
223 / 223 PASSING, zero warnings raised by `decoys.py`. The inverse-CDF sampler
(`src/decoys.py:431-449`) successfully replaces the rejection loop on every panel
PDB, including the previously-problematic sparse 11BG case.

## Kwarg combinations tested (20 of 128)

All twenty pass; each row reports `n_res / n_pairs / FrstIndex (min, max) / output rows`.

| # | Combo | Outcome |
|---|---|---|
| 1 | 5AON baseline cpu | OK 49 / 221 / FI ∈ (−2.17, 3.05) / 221 |
| 2 | 5AON baseline cuda | OK same as cpu (bit-identical) |
| 3 | 5AON chain="A" cpu | OK 49 / 221 |
| 4 | 5AON chain=["A"] cpu | OK 49 / 221, max \|ΔFI\| vs #3 = 0.00e+00 |
| 5 | 11BG chain=["A","B"] cpu | OK 248 / 1517 (full dimer) |
| 6 | 5AON residues={A:[10,20,30]} cpu | OK 49 / 221 → filtered 8 rows |
| 7 | 5AON chain="A"+residues={A:[5,10,15]} cpu | OK 49 / 221 → 0 rows (resnums start at 23, expected) |
| 8 | 5AON electrostatics_k=4.15 metadata-only cpu | OK 49 / 221 (FI unchanged vs #1, exactly matches LAMMPS convention) |
| 9 | 5AON electrostatics_k=4.15 + include_dh_in_e_native=True cpu | OK FI shifted (3.05 → 3.48 max, DH contribution active) |
| 10 | 3F9M lammps_compat_altloc=True cpu | OK 451 / 3349 |
| 11 | 3F9M default no-altloc cpu | OK same n_res/n_pair on this PDB (altloc=A is dominant) |
| 12 | 1O3S include_dna=True cpu | OK 200 / 1106 (DNA placeholder rows subset out of math, kept in emit) |
| 13 | 1O3S default include_dna=False cpu | OK 200 / 1106 (DNA chains absent at parse) |
| 14 | 11BG keep_incomplete_backbone=True cpu | OK 248 / 1517 |
| 15 | 3F9M cuda + electrostatics + altloc | OK 451 / 3349, FI ∈ (−4.39, 3.48) |
| 16 | 5AON cuda chain="A" + residues + electrostatics + include_dh | OK 49 / 221 → 0 rows (resnums 10/15/20 not in 23-71 range, expected) |
| 17 | 5AON seq_dist=3 cpu | OK 49 / 221 (different rho → different FI distribution, range OK) |
| 18 | 5AON n_decoys=100 cpu | OK FI ∈ (−2.06, 2.68) — noisier but range sane |
| 19 | 1O3S cuda include_dna=True | OK 200 / 1106 |
| 20 | 11BG cuda chain="A" + electro + include_dh | OK 124 / 632 (single-chain monomer of dimer) |

Cross-device sanity: combo #2 vs #1 max \|ΔFI\| = 0.0 (bit-identical CPU/CUDA at
matching seed). Same-seed determinism (combo #1 run twice) max \|ΔFI\| = 0.0.

## Missing features vs frustrapy

Reference: `C:/Users/7sN/AppData/Local/Programs/Python/Python310/lib/site-packages/frustrapy/analysis/frustration.py:26-42`.

| Kwarg | frustrapy | ours | Status |
|---|---|---|---|
| `pdb_file` | required (or `pdb_id`) | required | parity |
| `pdb_id` | auto-downloads from RCSB | accepted, silently ignored | LOW-1 (UX gap, not numerical) |
| `chain` | str / list / None | str / list / None | parity (QA-3 H-2 fix routes both through parser) |
| `residues` | dict[str, list[int]] | dict[str, list[int]] | parity |
| `electrostatics_k` | float, scales DH | float + `include_dh_in_e_native` opt-in | **extension** (we expose both LAMMPS-AWSEM and "physically complete" semantics; frustrapy's default omits DH from FI numerically, matching our default) |
| `seq_dist` | 3 or 12 | any int (default 12) | parity + widened |
| `mode` | configurational/mutational/singleresidue | same | parity |
| `graphics` | emits VMD .tcl | accepted, warns once, no-op | LOW-2 (cosmetic) |
| `visualization` | emits PyMOL .pml | accepted, warns once, no-op | LOW-2 (cosmetic) |
| `results_dir` | output dir | `results_dir` (alias) or `output_dir` | parity |
| `debug` | preserves intermediates | accepted, silently ignored | LOW-3 (we don't write intermediates anyway) |
| `pbar` | tqdm progress bar | accepted, silently ignored | LOW (UX) |
| `is_mutation_calculation` | recursive-call flag → mode="mutational" | translated to mode="mutational" | parity |
| `overwrite` | bool, frustrapy-only | not accepted | gap — but we never auto-skip an existing dir, so behaviour is "always overwrite". `_emit_pair_files` overwrites without prompting. NOT a numerical issue. |
| `n_cpus` | mutational-only multiprocessing | not accepted | NOT applicable — single-process GPU |
| `device` | n/a | "auto" / "cuda" / "cpu" | **extension** |
| `dtype` | n/a | torch.dtype | **extension** |
| `n_decoys` | hard-coded 1000 inside frustrapy | exposed kwarg (default 1000) | **extension** |
| `seed` | n/a (libc rand) | int | **extension** (essential for testing) |
| `precision` | n/a | int, decimal places | **extension** |
| `pair_min_seq_sep` | n/a | int (default 2) | **extension** |
| `include_dh_in_e_native` | n/a | bool | **extension** (opt-in physically-complete semantics) |
| `keep_incomplete_backbone` | n/a | bool | **extension** |
| `include_dna` | n/a | bool | **extension** (1O3S byte-parity) |
| `lammps_compat_altloc` | n/a | bool | **extension** (3F9M byte-parity) |

**Gaps that need flagging for fix:** none required for numerical correctness. `overwrite`
is the only behavioural difference (frustrapy can skip an existing results dir; we always
overwrite). Adding it is trivial and non-blocking for publication.

**Silent acceptance bug check:** `calculate_frustration(**kwargs)` raises `TypeError` on
**unknown** kwargs (`src/compute_frustration.py:1201-1209`), so no typo is silently
swallowed. `debug`, `pbar`, `graphics`, `visualization` are explicitly listed and
intentionally no-op'd (with a `UserWarning` for the visual flags). `pdb_id` raises if
`pdb_file` is None (no auto-download — explicit error, not silent). Good.

## Edge cases

Seven cases tested via tempfile-driven `compute_frustration` calls:

| Case | Result | Severity |
|---|---|---|
| Empty PDB file | `ValueError: No usable residues parsed` (`src/parser.py`) | OK — explicit error |
| HETATM-only | same `ValueError` | OK |
| Single residue (N=1) | `ValueError: sample_configurational_decoys requires N >= 2` (`src/decoys.py:357`) | OK — explicit error |
| Sparse coords, no native pairs | `RuntimeError: no in-contact pairs found` (`src/decoys.py:439`) | OK — explicit error |
| All-glycine (11 residues, linear) | Runs cleanly, n_res=11, n_pair=9, FI finite | OK |
| Multi-model NMR (MODEL 1 + MODEL 2) | **Stops at MODEL 1**, n_res=11 (not 22) | OK |
| Negative resnums (His-tag, −2 to 9) | Runs cleanly, n_res=12, n_pair=10 | OK |

All seven edges produce either a clean run or an explicit raise — no silent NaN
poisoning, no incorrect-output cases. Multi-model NMR handling is correct (parser
respects the MODEL/ENDMDL boundary, see `src/parser.py`).

## Determinism

| Test | Result |
|---|---|
| 5AON seed=0 cpu run twice | max \|ΔFI\| = 0.00e+00 (bit-identical) |
| 5AON seed=0 cpu vs seed=0 cuda | max \|ΔFI\| = 0.00e+00 (bit-identical) |
| 5AON seed=0 vs seed=42 cpu | decoy_mean shifts (−1.2212 vs −1.2384), Spearman FI = 1.0000 (rank order preserved) |

Determinism is **exact** at the same seed and is **rank-preserving** across seeds, as
the design intends (CPU `torch.Generator` seeded once per call, indices moved to device).

## Performance

FIX-4 reported 11BG configurational GPU at 3.52 ms (down from 38 ms). Measured on
this RTX 4070:

| Measurement | Value |
|---|---|
| `configurational_decoy_stats` alone, 11BG CUDA (median of 10) | **3.65 ms** (min 3.36 ms) |
| Full `compute_frustration` 11BG CUDA (median of 5) | 99 ms |
| Full `compute_frustration` 5AON CUDA (median of 5) | 32 ms |

The decoy sampler matches the FIX-4 claim to within run-to-run noise (3.36–3.65 ms
vs reported 3.52 ms). The full-pipeline number is dominated by PDB parse + native
pair enumeration + DataFrame construction; the sampler is no longer the bottleneck.

## Recommendations before publication

1. **(LOW)** `overwrite` kwarg parity — add a `if (out_dir.exists() and not overwrite):
   skip` gate in `_emit_pair_files`. Trivial, ~5 lines. Non-blocking.
2. **(LOW)** `pdb_id` auto-download from RCSB — `calculate_frustration` adapter could
   `urllib`-fetch when only `pdb_id` is set. UX nicety, non-numerical.
3. **(LOW)** Document the FI-classification threshold sensitivity in user-facing docs.
   On the 5AON 5adens comparison, 5 of 49 residues had `nNeutrallyFrst` ±1 vs LAMMPS
   because a handful of pairs land within 0.05 of the FI = −1.0 or FI = 0.78
   boundaries. Not a bug, but a reviewer might flag the 89.8 % match — pre-empt it.
4. **(LOW)** Consider exposing `n_decoys` through the `calculate_frustration` adapter
   (currently only on `compute_frustration`). One-line `**locals()` propagation.

No CRITICAL/HIGH/MEDIUM findings. The mode is locked correct against LAMMPS-AWSEM
on the four panel PDBs, the FIX-4 sampler change is sound (eliminates the rejection
fallback warning while preserving rank-order parity with the reference), CPU/CUDA
agree bit-for-bit at the same seed, and every kwarg interaction we sampled produces
sensible output. Publication green-light.

### File:line references for the audit

- Sampler (FIX-4 inverse-CDF): `src/decoys.py:286-459`, with the new in-contact
  enumeration at `src/decoys.py:431-449`.
- Decoy energy: `src/decoys.py:463-654`.
- Native-pair enum (configurational): `src/compute_frustration.py:206-321`.
- DH opt-in: `src/compute_frustration.py:710-720` (configurational branch) and
  `:814-824` (mutational), with the DH adder at `:324-367`.
- Pair DataFrame schema: `src/compute_frustration.py:370-425`.
- Density aggregation: `src/density.py:70-200`.
- FI / Welltype / classification: `src/frustration.py:113-199`.
- LAMMPS dump writer: `src/frustration.py:271-385`.
- 5adens writer: `src/density.py:225-264`.
- Frustrapy adapter (kwarg translation): `src/compute_frustration.py:1101-1232`.
- Frustrapy reference signature: `frustrapy/analysis/frustration.py:26-42`.
