# AWSEM Hamiltonian — reference for PyTorch port

This is the canonical spec the pure-PyTorch implementation must match. Numerical outputs must agree with LAMMPS-AWSEM (which frustrapy wraps) to within tolerance set by the validation tests.

## Hamiltonian terms enabled in frustration mode

From `frustrapy/core/scripts/AWSEMFiles/fix_backbone_coeff.data`:

```
[Epsilon] 1.0          # global energy multiplier on all k_* terms
[ABC]
  0.483 0.703 -0.186   # an bn cn  → N(i)  = an·CA(i-1) + bn·CA(i) + cn·O(i-1)
  0.444 0.235  0.321   # ap bp cp  → C(i)  = ap·CA(i)   + bp·CA(i+1) + cp·O(i)
  0.841 0.893 -0.734   # ah bh ch  → H(i)  = ah·CA(i-1) + bh·CA(i) + ch·O(i-1)
[Water] 1.0            # k_water (then ×= epsilon = 1.0 → 1.0 kcal/mol)
  5.0 7.0              # water_kappa (rij sigmoid 1/Å), water_kappa_sigma (rho sigmoid 1/ρ-unit)
  2.6                  # treshold = ρ_0 threshold for buried/exposed
  2 2                  # contact_cutoff, min seq separation (USED IN compute_tert_frust)
  4.5 6.5 1            # well 0 (direct): r_min, r_max, on-flag
  6.5 9.5 1            # well 1 (mediated): r_min, r_max, on-flag
[Burial] 1.0           # k_burial (then ×= epsilon = 1.0 → 1.0 kcal/mol)
  4.0                  # burial_kappa
  0.0 3.0              # well 0: ρ_min, ρ_max (exposed)
  3.0 6.0              # well 1: ρ_min, ρ_max (medium)
  6.0 9.0              # well 2: ρ_min, ρ_max (buried)
[Tertiary_Frustratometer]
  9.5                  # tert_frust_cutoff: CB-CB pair cutoff (Å)
  1000                 # tert_frust_ndecoys per (computed ONCE per structure in configurational mode)
  1                    # tert_frust_output_freq (NOT a seq-sep filter; the seq-sep gate is contact_cutoff=2 from [Water])
  configurational      # tert_frust_mode — frustrapy's default and what we run
[DebyeHuckel]
  4.15 4.15 4.15       # k_PlusPlus, k_MinusMinus, k_PlusMinus (Coulomb constants, kcal·Å/mol)
  1.0                  # k_screening (1/λ scale factor)
  10.0                 # screening_length λ (Å) — was incorrectly labeled "screening cutoff" earlier
  1                    # debye_huckel_min_sep (residues with |i-j|<1 excluded; only self)
```

## Parameter table sources

All from `openawsem/parameters/` (byte-identical files are shipped by frustrapy
inside `frustrapy/core/scripts/AWSEMFiles/`):
- `gamma.dat` — direct + mediated contact gamma (20×20). 420 rows total:
  rows 0-209 are direct (two identical columns: `g g`); rows 210-419 are
  mediated with two columns (protein-mediated, water-mediated). Iteration is
  upper triangle: for i in 0..19, for j in i..19. AA index order is
  A R N D C Q E G H I L K M F P S T W Y V (standard 3-letter alphabetical:
  ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO,
  SER, THR, TRP, TYR, VAL).
- `burial_gamma.dat` — 20 rows × 3 cols (low / med / high density wells).
  Same AA row order as gamma.dat.
- `anti_HB`, `para_HB`, `anti_NHB` — H-bond gamma tables (NOT used for frustration, only for folding)
- `anti_one`, `para_one` — H-bond directional terms (NOT used for frustration)

For frustration analysis we need: `gamma.dat` + `burial_gamma.dat` only.

## Resolved unit and AA-mapping conventions (from C++ direct reading, 2026-05-20)

These were previously flagged uncertain in `PHASE_1_STATUS.md`; both are now
resolved against `fix_backbone.cpp`:

1. **`k_burial` units = kcal/mol** (not kJ/mol). LAMMPS uses `units real` by
   default. The value in `fix_backbone_coeff.data` (`[Burial] 1.0`) is read
   literally as kcal/mol then multiplied by `epsilon = 1.0`. There is NO
   `4.184 kJ/mol` factor in the C++ path — that 4.184 is OpenAWSEM/OpenMM
   specific. **Action for `src/burial.py`**: change default `k_contact=4.184`
   to `k_contact=1.0` if matching LAMMPS/frustrapy; keep 4.184 for OpenAWSEM.
   See `fix_backbone.cpp:131, 270, 507-508`.

2. **AA index ordering for burial AND contact gamma = same single
   convention**: `A R N D C Q E G H I L K M F P S T W Y V` → 0..19. This is
   what `se_map[]` produces (`fix_backbone.cpp:55`) and what both
   `burial_gamma[i_resno][well]` (line 689) and `water_gamma[well][i][j][k]`
   (line 629) use as their row index. Current `src/parameters.py` uses the
   same convention — PASS.

3. **rho seq-sep convention**: C++ uses `|res_no[i]-res_no[j]| > 1`
   (i.e. excludes self and immediate ±1 neighbours). Current
   `src/burial.py:RHO_MIN_SEQ_SEP=2` with `> min_seq_sep` excludes self +
   ±1 + ±2. **This is an off-by-one bug** — fix `RHO_MIN_SEQ_SEP=1` for
   LAMMPS parity. See `smart_matrix_lib.h:638`.

## Decoy strategy (CONFIGURATIONAL mode — frustrapy's default and what we run)

Confirmed against `fix_backbone.cpp:5249-5344` (`compute_decoy_ixns`).

The configurational and mutational modes differ in **what is randomized**:

- **Configurational** (frustrapy default): randomize rij, rho_i, rho_j, aa_i, aa_j
  ALL independently from the structure's distribution. **The 1000 decoys are
  computed ONCE per structure** and reused across all (i,j) pairs (the cache
  flag `already_computed_configurational_decoys` at line 5341). So every row of
  `tertiary_frustration.dat` shares the same `<decoy_E>` and `std(decoy_E)`
  values.

- **Mutational**: keep rij, rho_i, rho_j fixed at native values; randomize only
  aa_i, aa_j. Decoys are recomputed for each (i,j) pair. Includes (i,k) and
  (j,k) cross-terms in both native and decoy energies.

For each native contact (i, j) with `|r_ij| < 9.5 Å` and (|i-j| >= contact_cutoff=2
OR different chains):

1. Compute `E_native = E_water(rij, aa_i, aa_j, rho_i, rho_j) + E_burial(aa_i, rho_i) + E_burial(aa_j, rho_j) [+ E_DH(rij, aa_i, aa_j)]`.
2. (Once per structure) Generate 1,000 decoys by:
   - Uniformly drawing a random residue index `k ∈ {0,...,n-1}` and reading off
     its AA identity. Repeat for `aa_i`, `aa_j`. Net effect: AA distribution
     follows the **protein's own composition**, NOT uniform 1/20.
   - Independently drawing `rij` from a random in-contact pair (reject-and-resample
     until `rij < 9.5`).
   - Independently drawing `rho_i`, `rho_j` from yet another random pair (no
     contact constraint).
   - Evaluate `E_decoy = E_water + E_burial(i) + E_burial(j) [+ E_DH]` on the
     scrambled inputs.
3. `decoy_mean = mean(1000 E_decoys)`, `decoy_std = std(1000 E_decoys)`.
4. Per-pair frustration index `FI[i,j] = (decoy_mean - E_native) / decoy_std`.
   **Sign convention**: FI > 0 means native is BETTER than average decoy
   (minimally frustrated, good). FI < 0 means native is WORSE than decoys
   (highly frustrated, bad).
5. Classify per pair:
   - FI < -1.0 → highly frustrated (red)
   - -1.0 ≤ FI ≤ 0.78 → neutrally frustrated
   - FI > 0.78 → minimally frustrated (green)
6. Aggregate per residue: 5 Å sphere around each CA, density of each class
   computed by `frustration_calculator.py:_calculate_frustration_density`. The
   per-residue file is `<pdb>.pdb_<mode>_5adens`.

## Reference implementations

- **LAMMPS-AWSEM C++** (canonical, what frustrapy runs):
  - Local copies in `docs/reference_lammps_awsem/`:
    - `fix_backbone.cpp` (8,031 LOC — main Hamiltonian + tertiary_frustratometer fix)
    - `fix_backbone.h`
    - `atom_vec_awsemmd.cpp` / `.h`
    - `pair_excluded_volume.cpp` / `.h`
    - `pair_ex_gauss_coul_cut.cpp` / `.h`
    - `smart_matrix_lib.h`
    - `fragment_memory.cpp` / `.h` (NOT needed for frustration — H-bond / folding only)
- **OpenAWSEM Python** (Wolynes-lab reimplementation via OpenMM):
  - At `C:/Users/7sN/AppData/Local/Programs/Python/Python310/lib/site-packages/openawsem/`
  - Same Hamiltonian, different engine. Use as a Python reference for the math; do NOT depend on at runtime (uses Linux-only `stride`, OpenMM, multiple Linux deps).

## Validation criteria

Per-PDB:
- **Native AWSEM energy** matches LAMMPS-AWSEM to relative error < 0.1%
- **Per-pair FI z-score matrix** Pearson r > 0.99 with frustrapy output
- **Per-residue density** (high / neutral / minimal fractions) Spearman ρ > 0.98
- **Top-30 most-frustrated residue list** overlap > 28/30

Test set: 20 PDBs in `benchmark/pdb_panel.csv`.

## Things that are explicitly NOT in scope

- Fragment-memory term (not used for frustration)
- H-bond / secondary-structure terms (not used for frustration)
- AMH-Go (not used for frustration)
- Q-bias (not used for frustration)
- Membrane-protein terms
- Long-range solvation (covered by water-mediated term already)

Only need: Burial, Direct contact, Water-mediated contact, ABC virtual atoms, DebyeHuckel electrostatics, Tertiary_Frustratometer analyzer.
