# CPU baseline dump — 2026-05-20

VM: `root@10.1.0.45` — frustrapy 0.1.1 installed at `/root/pyenvs/tuhnon/lib/python3.13/site-packages/frustrapy/`.

## Call signature (production-matched)

Identical to `extract_local_frustration.py:370-375`:

```python
frustrapy.calculate_frustration(
    pdb_file=...,
    mode="configurational",
    results_dir=...,
    graphics=False,
    debug=True   # for energy.log/timer.log preservation only — no numerical effect
)
```

Verified `debug=True` does NOT change numerical output (5AON tertiary_frustration.dat byte-identical between debug=True and debug=False runs).

## Files produced per PDB

| file | purpose |
|---|---|
| `<PDB>_tertiary_frustration.dat` | per-pair FI z-scores (one row per native contact). Columns: `i j chain_i chain_j x_i y_i z_i x_j y_j z_j r_ij ρ_i ρ_j ss_i ss_j E_native mean_decoy std_decoy FI` |
| `<PDB>_configurational.dat` | per-pair HIGH / NEU / MIN classification + PDB B-factor format |
| `<PDB>_5adens.dat` | per-residue 5 Å density features |
| `<PDB>_energy.log` | native LAMMPS-AWSEM per-term energy breakdown (Step 0) |
| `<PDB>_timer.log` | LAMMPS internal timing |

## Reference numbers (the values our PyTorch port must reproduce)

### 5AON (49 residues, monomer, chain A only)

```
Native AWSEM energy decomposition (kcal/mol, epsilon=1.0):
  V_Water  = -18.700281
  V_Burial = -41.799488
  V_Total  = -60.499769

Contact count (rows in tertiary_frustration.dat): 222
LAMMPS Frust_Analysis time: 0.99 ms
LAMMPS Total time:          1.16 ms
```

### 11BG (124 residues × 2 chains = 248 residues, homodimer)

```
Native AWSEM energy decomposition (kcal/mol):
  V_Water  = -147.390847
  V_Burial = -210.591584
  V_Total  = -357.982431

Contact count: 1,518
LAMMPS Frust_Analysis time: ~? ms (need to extract from 11BG_timer.log)
LAMMPS Total time:          6.20 ms
```

## Units

V_Water and V_Burial are in **kcal/mol** (LAMMPS-AWSEM default with `epsilon=1.0`). Note: previous Phase 1 status mentioned OpenAWSEM default of `4.184 kJ/mol` per contact — **that converts to 1.0 kcal/mol**, so units ARE consistent across implementations. The unit uncertainty from Phase 1 status (uncertainty #1) is now resolved: **use kcal/mol with k_contact = 1.0**.

## Format of tertiary_frustration.dat (header reproduced)

```
# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j native_energy <decoy_energies> std(decoy_energies) f_ij
# timestep: 0
```

So `FI = (mean_decoy − native_energy) / std_decoy_energies` is column 19 (the last one, `f_ij`). Columns 16-17-18 are native_energy, mean_decoy_energy, std_decoy_energy respectively.

## Wall-clock observation (smaller than expected)

LAMMPS itself is very fast on small proteins:
- 5AON (49 res): 1.2 ms total LAMMPS time
- 11BG (248 res dimer): 6.2 ms total LAMMPS time

The "minutes per protein" we kept worrying about must come from (a) frustrapy Python orchestration overhead, (b) LAMMPS subprocess startup, OR (c) much larger proteins. We won't know how it scales without timing on the bigger panel PDBs (4PKN 8689 res is the stress test).

The frustrapy Python wall-clock reported was 0.23-0.32s per PDB on the VM (most is subprocess + file I/O overhead, not LAMMPS itself).

## Re-verification 2026-05-20 (after C++ audit)

Re-ran both PDBs cleanly (`time.py` on VM). End-to-end wall-clock with
`debug=False`:

| PDB | residues | wall-clock |
|---|---|---|
| 5AON | 49 | **214 ms** |
| 11BG | 248 | **301 ms** |

Pair counts re-confirmed: 5AON = 221 contacts, 11BG = 1517 contacts.

Decoy stats verified constant across all rows (confirms the
`already_computed_configurational_decoys` cache flag in `fix_backbone.cpp:5341`):

* 5AON: `(decoy_mean, decoy_std) = (-1.253, 0.491)` kcal/mol — same for all 221 rows.
* 11BG: `(decoy_mean, decoy_std) = (-1.513, 0.454)` kcal/mol — same for all 1517 rows.

## Phase 2 numeric validation targets

From the locally-stored energy.log files:

| PDB | V_Water (kcal/mol) | V_Burial (kcal/mol) | V_Total |
|---|---|---|---|
| 5AON | -18.700281 | -41.799488 | -60.499769 |
| 11BG | -147.390847 | -210.591584 | -357.982431 |

These are the per-PDB targets the PyTorch port must reproduce to within 0.1%
relative error (per `awsem_hamiltonian_spec.md` validation criterion).

For per-pair: Pearson r > 0.99 on `f_ij` across all rows of
`tertiary_frustration.dat`. The 1000-decoy mean/std contributes ~3% relative
run-to-run noise from the C++ RNG (libc `rand()`), so the PyTorch port's
allowed deviation floor is set by that, not by any deterministic mismatch.
