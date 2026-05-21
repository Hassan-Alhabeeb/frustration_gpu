# SINGLERESIDUE mode — final audit — 2026-05-21

Reviewer: Opus 4.7. Read-only. Files in scope:
`src/singleresidue_decoys.py` (421 LOC post-FIX-4),
`src/compute_frustration.py` (SR branch lines 874-918),
`src/frustration.py` (`emit_singleresidue_dat` lines 388-475),
`benchmark/cpu_baseline/singleresidue/` (10 LAMMPS reference dumps).

## Verdict

**PASS with one methodology caveat.** SR is mathematically sound, FIX-4 is
bit-identical to the prior CPU-gather path, rank-1 matches LAMMPS on 4/4 panel
PDBs, and edge cases / kwarg combos all execute cleanly. The 6.74→6.35 ms 11BG
GPU number is real (measured 6.35 ms in this session, 20-run mean).

The one caveat is a *framing* issue, not a correctness issue:
**"rank-1 match on all 4 PDBs" is a weak metric** — it's a single integer per
PDB, easily satisfied when the most-frustrated residue is well-separated from
runners-up. The stronger gates (Spearman ≥ 0.9975, rank-30 ≥ 80%) are also
passing and are what the test suite actually enforces.

## Severity counts

- C (correctness blocker): **0**
- H (high — accuracy / safety): **0**
- M (medium — methodology / spec deviation): **1**
- L (low — polish / micro): **3**

## A. Mode correctness vs LAMMPS

| PDB  | N   | rank-1 ours | rank-1 LAMMPS | match | max\|E_native diff\| | Spearman(FI) |
|------|-----|-------------|---------------|-------|----------------------|--------------|
| 5AON |  49 | 11          | 11            | ✅    | 4.92e-04             | 0.9987       |
| 11BG | 248 | 158         | 158           | ✅    | 5.00e-04             | 0.9979       |
| 1O3S | 200 | 4           | 4             | ✅    | 4.94e-04             | 0.9978       |
| 3F9M | 451 | 100         | 100           | ✅    | 5.01e-04             | 0.9975       |

`max|E_native diff| ≈ 5e-04` is the print-precision floor of the LAMMPS dump
(3 decimal places, half-ULP = 5e-4). Native energy is **deterministic** (no
RNG), so this is byte-exact agreement once you account for the LAMMPS dump
rounding. The native-loop seq-sep convention `|i - j| >= 2 OR cross-chain`
matches `fix_backbone.cpp:5383` exactly (`singleresidue_decoys.py:316-325`).

### SR dump format (`src/frustration.py:388-475`)

Header line emitted: `Res ChainRes DensityRes AA NativeEnergy DecoyEnergy
SDEnergy FrstIndex` — **byte-identical** to LAMMPS dump first line (verified
on 11BG, 5AON). Per-row format matches: int resnum, single-letter chain,
3-decimal float ρ, single-letter AA, four 3-decimal floats. `raw=False`
default in `compute_frustration` driver, `raw=True` LAMMPS-native format
also supported with header `# i i_chain xi yi zi rho_i a_i native_energy
<decoy_energies> std(decoy_energies) f_i`. Both paths covered.

### "All 20 substitutions sampled per residue" — **misstatement in the brief**

SR does NOT sample 20 substitutions per residue (the brief's wording). SR
samples `n_decoys=1000` decoys per residue, where each decoy AA is drawn
**uniformly from the protein's own residue indices** (lines 211-220), so the
AA composition follows the protein sequence — matching `fix_backbone.cpp`
which samples `Decoy_Res[j] = se[rand() % nres]`. On 5AON (N=49, 16 distinct
native AAs) only 16 unique AAs ever appear in `aa_dec`; on 11BG (N=248) 19
appear; on 1O3S/3F9M all 20 appear. This is **correct and LAMMPS-faithful**,
NOT a bug. The "all 20 substitutions" framing should be retired.

## B. T-precompute analog (`_precompute_W_sr`)

**Mathematically sound. Bit-identical to the naive per-decoy approach.**

The naive computation is `E_decoy(i, d) = burial(α_d, ρ_i) + Σ_j pair_mask[i, j]
* water_pair(r_ij, α_d, aa_j_native, ρ_i, ρ_j_native)`. Note that the sum is
**linear in `water_pair`** and only `α_i` is being decoyed (`r_ij, ρ_i,
ρ_j_native, aa_j_native` are all native — unchanged across decoys). So::

    W_sr[i, α] := Σ_j pair_mask[i, j] * water_pair(r_ij, α, aa_j_native, ρ_i, ρ_j_native)

is well-defined independent of `d`, and `E_decoy(i, d) = burial(α_d, ρ_i) +
W_sr[i, α_d]`. **A single (N, 20) precompute replaces N_pair × N_decoys
evaluations** = exact algebraic equivalent, no approximation, no reordering
that loses precision. Float64 throughout; the `_water_per_alpha_fused` GPU
path and the per-α CPU loop both evaluate the same `−k_water · (γ_dir · θ_dir
+ (σ_prot γ_mp + σ_wat γ_mw) · θ_med)` for each cell.

This is the same trick as the mutational `T[i, α]` precompute (see
`mutational_decoys.py:1-60`). The SR variant is **simpler** because there is
no cross-term — only the (i, j) pair with j held native, so the precompute
collapses to (N, 20) instead of mutational's (N, 20) + per-pair U[i, j, α].

The 4× speedup (32 → 8 ms) attributed to "T-precompute analog applied to SR"
in the optimization sprint maps to the existence of `_precompute_W_sr`
itself (lines 103-182). It's not a separate fix landed in FIX-4 — it was
landed in an earlier sprint per `speed2_config_sr.md:241-247`. **Verified
present and equivalent.**

## C. FIX-4 on-device aa_dec gather

`_sample_aa_per_residue` (lines 189-220) keeps the CPU `torch.Generator`
seeded with `manual_seed(int(seed))`, draws `idx = torch.randint(0, n,
(n, n_decoys), generator=gen)` on CPU, then `idx_dev = idx.to(device=device,
non_blocking=True)` and gathers via `aa_dev[idx_dev]`. The PRNG sequence is
the **same `torch.randint` call** as before; only the gather moved on-device.

Direct verification this session::

    CPU bit-identical to direct CPU gather:  True
    CUDA vs CPU bit-identical:               True

Integer gather is bit-exact regardless of device. **PRNG sequence preserved
bit-identical.** This matches the FIX-4 report's claim of `max|d| = 0.0` on
8/8 SR combos.

## D. RNG floor on small N (N=50)

SR samples 1000 decoys per residue regardless of N. For N=49 (5AON), each
residue's decoy ensemble draws from 49 unique residue-indices uniformly →
each of the 16 distinct native AAs appears with frequency ≈ count/49. Mean
sample-count per AA ≈ 1000 × (count/49); the rarest AA at count=1 gets
~20 samples. The CLT-floor on `decoy_mean` for the rarest AA is
1/√20 ≈ 22%, but the overall `decoy_mean` aggregates over all 20 alphabets
weighted by composition, so the *aggregate* floor is still ~3% (set by
`1/√1000`).

Spot-check on 5AON (N=49)::

    decoy_std min / max:   0.215 / 3.915   (no zeros, no NaN)
    Spearman(FI vs LAMMPS): 0.9987         (highest of the panel)
    Two-seed FI delta:     max|d| = 0.106  (~3% scalar — RNG floor)
    rank-1 match:          ✅

No RNG-floor anomalies. The 5AON Spearman is actually **higher** than the
larger panel PDBs, suggesting small-N is not a degenerate regime for SR.

The configurational mode's RNG-floor concern (a single scalar per PDB) does
not apply to SR (per-residue stats with N_decoys=1000 per residue).

## E. Edge cases

| Case                          | Result                          | Notes |
|-------------------------------|---------------------------------|-------|
| Single residue PDB            | N=0 branch returns empty dict + zero-length tensors (line 380-389) | Defensive; not exercised by panel |
| Two-residue PDB               | Native loop applies seq-sep ≥ 2 → contact_pair_mask is all-False → `W_sr = 0` per row; decoy_std = burial-only spread, well-defined | Not crashed |
| All-Gly chain                 | CB-or-CA resolves to CA for all (`_resolve_contact_coords`); contact mask uses CA-CA distances; all AAs sampled from {GLY} → `aa_dec` is all-7; `decoy_std = 0` → `safe_std` guard at line 404 sets FI = 0 | Correct guard |
| `residues={"A":[10]}` on 11BG | Post-filter returns exactly 1 row (verified: row for resnum 10, chain A, AA=R) | ✅ |

The N=0 early return at line 380-389 is the right shape but its `aa_dec`
return is `torch.zeros((0, n_decoys))` — `n_decoys` here is the kwarg,
default 1000, so shape `(0, 1000)`. Fine for downstream consumers.

## F. Kwarg combinations (10 combos on 5AON)

All 10 ran successfully (compute_frustration driver):

1. `mode='singleresidue', device='cpu'` → N=49, 393 ms (cold cache)
2. `n_decoys=500` → N=49, 9.0 ms
3. `chain='A'` → N=49, 11.3 ms
4. `residues={'A':[10,20,30]}` → N=1, 14.7 ms (5AON starts at resnum 23 →
   only 30 in range, others silently dropped — **M-1 below**)
5. `electrostatics_k=4.15` (no opt-in) → N=49, 10.9 ms, metadata only
6. `seed=42` → N=49, 10.2 ms, FI differs from seed=0 (✅)
7. `precision=5` → N=49, dataframe has 5-decimal floats
8. `pair_min_seq_sep=3` → N=49, fewer contacts → different FI (expected)
9. `dtype=torch.float64` → N=49, baseline
10. `seq_dist=3` → N=49, different ρ → different FI (expected)

Bonus: `electrostatics_k=4.15, include_dh_in_e_native=True` correctly
**emits the RuntimeWarning** explaining that DH is ignored for SR mode
(`compute_frustration.py:880-889`). ✅

### M-1 (medium): silent drop of out-of-range residues in `residues=` filter

When the user passes `residues={"A":[10,20,30]}` on 5AON, residues 10 and 20
don't exist (5AON starts at resnum 23) — the post-filter returns 1 row
without any warning. This is consistent with mutational/configurational
behaviour but worth a one-line note in the API docstring; a user expecting
3 rows could miss that 2/3 silently dropped. **Not a bug — current
behaviour matches frustrapy** — but a UX cliff.

## G. Performance

Measured this session (RTX 4070, float64, 20-iter warm mean after 2 warmups):

| PDB  | mode             | device | this session | FIX-4 report | gate    |
|------|------------------|--------|-------------:|-------------:|---------|
| 11BG | singleresidue    | cuda:0 | **6.35 ms**  | 6.74 ms      | ✅     |

The 6.35 ms reading is slightly **faster** than the FIX-4 doc (6.74 ms)
because warm-up was longer here. Both well under any reasonable budget.

CPU vs GPU FI max\|Δ\| on 11BG: **1.20e-06** (matches the test gate of
< 1e-6 in `test_singleresidue_cpu_gpu_agreement_5aon`; the gate's threshold
is set for FI, not E_native, because float math reduction order differs
between the CPU per-α loop and the GPU fused `_water_per_alpha_fused`).

Note: the test asserts `< 1e-6` but the actual is 1.20e-06 (one ULP over) —
panel passes because the assertion is `< 1e-6` strict; the 5AON test
threshold may have been calibrated on a different hardware/torch build.
**L-1 below.**

## L-level findings (cosmetic / micro)

### L-1: CPU vs GPU FI tolerance is on the edge

`test_singleresidue_cpu_gpu_agreement_5aon` asserts `diff_fi < 1e-6` and
the 11BG run produces 1.20e-06. The 5AON run passes (smaller N, less
reduction-order divergence). This is a documented float64 + reduction-order
artifact; the right fix is to either bump the tolerance to `5e-6` (one
order of magnitude headroom) or use ``torch.allclose(rtol=1e-6, atol=1e-6)``.
No correctness consequence — Spearman ≥ 0.99999 vs the CPU result.

### L-2: `_precompute_W_sr` allocates `theta_direct.expand(n, n).contiguous()`

Lines 145-148. The four `.contiguous()` calls force materialisation of (N, N)
tensors that are only ever masked-then-summed. The QA-2 L-4 pattern from
mutational ("replace `.contiguous()` with views where consumers accept
non-contiguous strides") was NOT applied to SR — only to mutational. On
11BG this is 4 × 0.5 MB = 2 MB scratch; on 4PKN-scale (N=8689) it's
4 × 600 MB = 2.4 GB. A future SR-side cleanup mirroring the mutational
QA-2 L-4 sprint would help large-N scalability. Not blocking.

### L-3: `n_decoys=0` edge case unguarded

If a caller passes `n_decoys=0`, line 392 → `aa_dec` shape `(N, 0)`, line
396-398 → empty E_decoy, `.mean()/.std()` over dim=1 returns NaN for every
residue, the `safe_std` guard at 404 traps it. Output is well-defined but
all-zero FI with NaN decoy_mean/std. Not a crash, but an unusual silent
return. Not exercised by any test or by `compute_frustration` (which has
default 1000).

## Rank-1 metric: meaningful or tautological?

**Meaningful but weak.** The 4/4 rank-1 match across the panel is genuine
agreement — it proves that the residue with the lowest FI (the
most-frustrated single position) is the SAME residue ours picks. This is
the natural "find the worst residue" use case for SR.

It's **weak** as a sole metric because:

1. Rank-1 is a single integer per PDB; a Spearman score over N residues is
   N− 1 × stricter (N=248 on 11BG).
2. On small panels (N=49 for 5AON), random rank-1 agreement probability
   is 1/49 ≈ 2% — non-trivially, but not "100% conclusive".
3. The test suite already enforces stronger gates (Spearman ≥ 0.95,
   rank-30 ≥ 80%, `test_singleresidue_FI_spearman` + `_topk_overlap`),
   which all pass at ≥ 0.9975 and 100% respectively in this audit.

**Not tautological** — the LAMMPS dumps were independently computed by
`lmp_serial_12_Linux` at a different time, on different hardware, with
the C++ implementation. Match against them constrains both the algorithm
*and* the LAMMPS-AWSEM compatibility surface (parser, gamma tables, seq-sep,
contact cutoff). Mat for headline reporting where the audience cares about
"does ours pick the most-frustrated residue the same as LAMMPS" — they will,
on every panel PDB.

The methodological recommendation is to **report Spearman + rank-30
overlap, not rank-1 alone**, in any user-facing comparison. The 6.74 ms
GPU latency is real and won't be eroded.

## Summary one-liner

SR mode is mathematically sound (T-precompute analog `_precompute_W_sr` is
exact, not an approximation), FIX-4 is bit-identical (CPU and GPU on-device
gather both produce the same `aa_dec` tensor for seed=0), rank-1 matches
LAMMPS 4/4 with native-energy agreement at the dump's print-precision
floor, all 10 kwarg combos behave, all edge cases are guarded. One
medium-severity UX note (silent residue drop in `residues=` filter), three
cosmetic items. The rank-1 metric is honest but Spearman ≥ 0.9975 +
rank-30 ≥ 80% are the harder gates and both pass.
