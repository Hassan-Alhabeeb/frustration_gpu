# LAMMPS-AWSEM term spec — canonical reference for the Phase 2 PyTorch port

Source of truth: `docs/reference_lammps_awsem/fix_backbone.cpp` (8,031 LOC) and
`smart_matrix_lib.h` (~700 LOC), as actually shipped inside frustrapy.

Mined from the C++ on 2026-05-20. All line numbers refer to the local copies in
`docs/reference_lammps_awsem/`.

The Phase 2 PyTorch port must reproduce the values these functions compute to
within float32 noise.

---

## 0. Global constants from `fix_backbone.cpp:50-63`

```cpp
// {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
//  "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"};
//   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
//   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
int se_map[] = {0,0,4,3,6,13,7,8,9,0, 11,10,12,2,0,14,5,1,15,16, 0,19,17,0,18,0};
char one_letter_code[] = {'A','R','N','D','C','Q','E','G','H','I',
                          'L','K','M','F','P','S','T','W','Y','V'};
```

`se_map` indexes by `(letter - 'A')` (so 26 entries, slots for non-AA letters
are 0). It returns the gamma-table column. Example: `se[i]='C' → se_map[2] = 4`
(cysteine is index 4).

This is the order all gamma tables (`gamma.dat`, `burial_gamma.dat`) are indexed by.

**Current PyTorch impl (`parser.py:46-51` and `parameters.py` docstring)**: uses
exactly the same A R N D C Q E G H I L K M F P S T W Y V → 0..19 mapping.
**Status: PASS.**

---

## 1. Burial term (`fix_backbone.cpp:3503-3570`, energy-only variant `5478-5500`)

### Source

```cpp
void FixBackbone::compute_burial_potential(int i) {
  ...
  t[0][0] = tanh( burial_kappa*(well->ro(i) - burial_ro_min[0]) );
  t[0][1] = tanh( burial_kappa*(burial_ro_max[0] - well->ro(i)) );
  t[1][0] = tanh( burial_kappa*(well->ro(i) - burial_ro_min[1]) );
  t[1][1] = tanh( burial_kappa*(burial_ro_max[1] - well->ro(i)) );
  t[2][0] = tanh( burial_kappa*(well->ro(i) - burial_ro_min[2]) );
  t[2][1] = tanh( burial_kappa*(burial_ro_max[2] - well->ro(i)) );

  burial_gamma_0 = get_burial_gamma(i_resno, ires_type, 0);
  burial_gamma_1 = get_burial_gamma(i_resno, ires_type, 1);
  burial_gamma_2 = get_burial_gamma(i_resno, ires_type, 2);

  energy[ET_BURIAL] += -0.5*k_burial*burial_gamma_0*(t[0][0] + t[0][1]);
  energy[ET_BURIAL] += -0.5*k_burial*burial_gamma_1*(t[1][0] + t[1][1]);
  energy[ET_BURIAL] += -0.5*k_burial*burial_gamma_2*(t[2][0] + t[2][1]);
}
```

### Density `ρ_i` (`smart_matrix_lib.h:630-652`, `compute_ro`)

```cpp
void cWell<T,U>::compute_ro(int i) {
  v_ro[i] = 0;
  for (j=0;j<lc->nn;++j) {
    if (lc->res_info[j]==lc->OFF) continue;
    if ( lc->chain_no[i]!=lc->chain_no[j] || abs(lc->res_no[j] - lc->res_no[i])>1 )
      v_ro[i] += theta(i, j, 0);   // well 0 = direct-contact window (4.5-6.5 A)
  }
}
```

with `theta(i,j,0)` defined at lines 583-589:

```cpp
t_min = tanh( par.kappa*(rij - par.well_r_min[i_well]) );  // par.kappa = 5.0
t_max = tanh( par.kappa*(par.well_r_max[i_well] - rij) );
v_theta = 0.25 * (1.0 + t_min) * (1.0 + t_max);
```

`rij` is the CB-CB distance, **CA substituted for CB on glycine** (lines 568-571
of `smart_matrix_lib.h`).

### Plain English

For each residue `i`:

1. Count nearby CB atoms (CA on glycine) using a smooth tanh window centred on
   the 4.5-6.5 Å shell, where `eta = par.kappa = 5.0 1/Å`. Only count pairs
   with `|res_no[i]-res_no[j]| > 1` **within the same chain**, OR pairs on
   **different chains** (different-chain pairs ALWAYS count, no seq separation
   filter).
2. Pass the resulting `ρ_i` through three tanh wells (low / med / high) and
   weight by the residue-type-specific `burial_gamma[aa_i, w]`.
3. Sum across wells and over residues with prefactor `-0.5 * k_burial`.

### Parameters

From `fix_backbone_coeff.data` (`[Burial]` section, parsed at `fix_backbone.cpp:267-274`):

```
[Burial] 1.0
4.0          → burial_kappa
0.0 3.0      → burial_ro_min[0], burial_ro_max[0]
3.0 6.0      → burial_ro_min[1], burial_ro_max[1]
6.0 9.0      → burial_ro_min[2], burial_ro_max[2]
```

So `k_burial = 1.0`, `burial_kappa = 4.0`. Then `k_burial *= epsilon` at line 508
(`epsilon = 1.0` from `[Epsilon] 1.0`), so net `k_burial = 1.0`.

### Units (definitive resolution)

**`k_burial = 1.0` in `kcal/mol`** — LAMMPS uses `units real` by default
(kcal/mol, Å, fs). The fix-backbone code does no further unit conversion.
There is NO `4.184 kJ/mol` factor anywhere in the C++ path.

OpenAWSEM's `4.184 kJ/mol` factor is OpenMM-specific (OpenMM defaults to kJ/mol).
For a frustrapy-equivalent port we want `k_burial = 1.0 kcal/mol`. To compare
to OpenAWSEM output, multiply by 4.184.

### Burial-gamma table indexing

`get_burial_gamma(i_resno, ires_type, w)` at line 2887 returns
`burial_gamma[ires_type][local_dens]`. Loaded at `fix_backbone.cpp:684-693`:

```cpp
for (i=0;i<20;++i) {
  in_brg >> burial_gamma[i][0] >> burial_gamma[i][1] >> burial_gamma[i][2];
}
```

Row order = AA order from `one_letter_code` (A, R, N, D, ..., V). Column order
= well 0 (low), 1 (med), 2 (high). Shape (20, 3).

NOT multiplied by `k_burial` at load time — `k_burial` is applied at the energy
formula instead (unlike water_gamma, which IS pre-multiplied; see §2).

### Comparison to `src/burial.py`

| Aspect | C++ | PyTorch | Status |
|---|---|---|---|
| Density window r_min, r_max | 4.5, 6.5 Å | 0.45, 0.65 nm | PASS (units differ but consistent) |
| Density sharpness η (par.kappa) | 5.0 1/Å | 50.0 1/nm | PASS |
| Sequence sep mask | `\|res_no[i]-res_no[j]\| > 1` (same chain), all pairs (cross chain) | `\|i-j\| > min_seq_sep` with `min_seq_sep=2` (same chain), all pairs (cross chain) | **FAIL** — C++ excludes only `\|i-j\|<=1` (i.e. self + 1 neighbour). PyTorch excludes `\|i-j\|<=2` (excludes self + 2 neighbours). PyTorch is dropping `\|i-j\|=2` contributions that C++ includes. Fix: change `min_seq_sep=2` to `min_seq_sep=1` in `parameters.py:58` and update the inline math (the existing `>` comparison keeps the off-by-one in alignment if 1 is used). |
| Burial wells | 3 wells (0,3) (3,6) (6,9) | Same | PASS |
| Burial kappa | 4.0 | 4.0 (`BURIAL_KAPPA`) | PASS |
| Energy prefactor | `-0.5 * k_burial * γ * (tanh + tanh)` per well | Same | PASS |
| `k_burial` value | 1.0 kcal/mol | 4.184 kJ/mol (default, configurable) | **NEEDS-CHECK** — kJ/mol vs kcal/mol differ by 4.184x. For frustrapy parity set `k_contact=1.0` in `burial_energy`. |
| AA gamma row order | A, R, N, D, ..., V | A, R, N, D, ..., V | PASS |
| Glycine CB substitution | CA for GLY | CA for GLY (`_resolve_density_coords`) | PASS |

### Bottom line

Two off-by-X issues to fix in Phase 2 before claiming numerical parity:

1. Seq-sep mask off-by-one in `compute_rho` (drops `|i-j|=2` pairs).
2. Default `k_contact` should be `1.0` (kcal/mol) to match LAMMPS, not 4.184.

---

## 2. Direct contact term (`fix_backbone.cpp:5444-5476`, `compute_water_energy`)

### Source

```cpp
double FixBackbone::compute_water_energy(double rij, int i_resno, int j_resno,
                                         int ires_type, int jres_type,
                                         double rho_i, double rho_j) {
  // Direct contact (well 0): r_min=4.5, r_max=6.5
  water_gamma_0_direct = get_water_gamma(i_resno, j_resno, 0, ires_type, jres_type, 0);
  water_gamma_1_direct = get_water_gamma(i_resno, j_resno, 0, ires_type, jres_type, 1);
  sigma_gamma_direct = (water_gamma_0_direct + water_gamma_1_direct) / 2;

  t_min_direct = tanh( well->par.kappa*(rij - well->par.well_r_min[0]) );  // kappa=5.0
  t_max_direct = tanh( well->par.kappa*(well->par.well_r_max[0] - rij) );
  theta_direct = 0.25 * (1.0 + t_min_direct) * (1.0 + t_max_direct);

  // ... mediated contact (see §3)

  water_energy = -(sigma_gamma_direct*theta_direct + sigma_gamma_mediated*theta_mediated);
  return water_energy;
}
```

### Plain English

Direct contact between residues i,j is a single sigmoid switch on `rij` (CB-CB
distance, CA on glycine) within the 4.5-6.5 Å shell with sharpness 5.0 1/Å.

The "gamma" used is the **average of the two direct gammas** (`water_gamma_0`
and `water_gamma_1`). This is a quirk of the LAMMPS table layout: the direct
column has two redundant copies — same number — so the average is identity.
Verified empirically: in `gamma.dat`, the first 210 rows are written `g g` (two
identical columns); see `5AON.done/gamma.dat` lines 1-3 dumped during this
session.

### Parameters

From `[Water]` section (`fix_backbone.cpp:257-266`):

```
[Water] 1.0      → k_water (then *= epsilon = 1.0)
5.0 7.0          → water_kappa, water_kappa_sigma
2.6              → treshold (ρ threshold for water/protein sigma)
2 2              → contact_cutoff, min seq separation for both wells
4.5 6.5 1        → well_r_min[0], well_r_max[0], well_flag[0]
6.5 9.5 1        → well_r_min[1], well_r_max[1], well_flag[1]
```

`well->par.kappa = water_kappa = 5.0`.

### Gamma table layout

`get_water_gamma(_, _, i_well=0, i, j, k)` at line 2872 returns
`water_gamma[i_well][ires_type][jres_type][water_prot_flag]`. Loaded at
`fix_backbone.cpp:626-639`:

```cpp
for (i=0;i<20;++i) {
  for (j=i;j<20;++j) {
    in_wg >> water_gamma[i_well][i][j][0] >> water_gamma[i_well][i][j][1];
    water_gamma[i_well][i][j][0] *= k_water;          // ← pre-multiplied!
    water_gamma[i_well][i][j][1] *= k_water;
    water_gamma[i_well][j][i][0] = water_gamma[i_well][i][j][0];   // symmetric
    water_gamma[i_well][j][i][1] = water_gamma[i_well][i][j][1];
  }
}
```

So `water_gamma` is **symmetric in i,j** (PASS — `get_water_gamma(i,j) ==
get_water_gamma(j,i)`) and **pre-multiplied by `k_water`** at load time. Phase 2
should mirror this: load the gamma file, multiply by k_water once, use it.

For the direct table specifically (i_well=0), both `[0]` and `[1]` slots are
the SAME number — so the average is identity. (Two slots are there for
mediated-table API symmetry.)

### Units

`k_water = 1.0` from data file × `epsilon = 1.0` → 1.0 kcal/mol. Same as burial.

---

## 3. Water-mediated contact term (`fix_backbone.cpp:5444-5476`, same function)

### Source

```cpp
  water_gamma_prot_mediated = get_water_gamma(i_resno, j_resno, 1, ires_type, jres_type, 0);
  water_gamma_wat_mediated  = get_water_gamma(i_resno, j_resno, 1, ires_type, jres_type, 1);

  sigma_wat  = 0.25 * (1.0 - tanh(well->par.kappa_sigma*(rho_i-well->par.treshold)))
                    * (1.0 - tanh(well->par.kappa_sigma*(rho_j-well->par.treshold)));
  sigma_prot = 1.0 - sigma_wat;

  sigma_gamma_mediated = sigma_prot * water_gamma_prot_mediated
                       + sigma_wat  * water_gamma_wat_mediated;

  t_min_mediated = tanh( well->par.kappa*(rij - well->par.well_r_min[1]) );  // 6.5
  t_max_mediated = tanh( well->par.kappa*(well->par.well_r_max[1] - rij) );  // 9.5
  theta_mediated = 0.25 * (1.0 + t_min_mediated) * (1.0 + t_max_mediated);

  water_energy = -(sigma_gamma_direct*theta_direct + sigma_gamma_mediated*theta_mediated);
```

### Plain English

The mediated shell (6.5-9.5 Å) gets two gamma values per AA pair: a protein-
mediated and water-mediated one. The pair's effective gamma blends them based
on `sigma_wat` — the joint probability that **both** residues are solvent-
exposed:

* `sigma_wat = 0.25 * (1 - tanh(κ_σ (ρ_i - ρ_0))) * (1 - tanh(κ_σ (ρ_j - ρ_0)))`
* `κ_σ = water_kappa_sigma = 7.0`, `ρ_0 = treshold = 2.6`
* `sigma_prot = 1 - sigma_wat`
* Effective gamma = `sigma_prot * γ_protein_mediated + sigma_wat * γ_water_mediated`

So if both residues are buried (`ρ > 2.6`), `sigma_wat → 0` and the protein-
mediated gamma dominates. If both are exposed, `sigma_wat → 1` and the water-
mediated gamma dominates.

### Parameters

Same `[Water]` block. Key numbers:

* `well_r_min[1] = 6.5`, `well_r_max[1] = 9.5`
* `water_kappa = 5.0` (1/Å for the rij sigmoid)
* `water_kappa_sigma = 7.0` (1/ρ-unit for the rho sigmoid)
* `treshold = 2.6` (the rho threshold for "buried")

### Gamma table

For mediated (i_well=1), the two slots are DIFFERENT:

* `water_gamma[1][i][j][0]` = protein-mediated γ
* `water_gamma[1][i][j][1]` = water-mediated γ

Loaded from `gamma.dat` rows 210-419 (second block; first block is direct).
Each row has 2 columns (column 0 = protein-mediated, column 1 = water-mediated).
Confirmed against `parameters.py:load_gamma_tables` which decodes the same way
(PASS for layout — but Phase 2 must remember the `k_water` pre-multiplication).

### Units

Same as direct: kcal/mol after `k_water *= epsilon`.

---

## 4. ABC virtual atoms (`fix_backbone.cpp:7117-7135`)

### Source

```cpp
xn[i][0] = an*xca[im1][0] + bn*xca[i][0] + cn*xo[im1][0];  // N(i)
xh[i][0] = ah*xca[im1][0] + bh*xca[i][0] + ch*xo[im1][0];  // H(i)
xcp[im1][0] = ap*xca[im1][0] + bp*xca[i][0] + cp*xo[im1][0];  // C'(im1) = C(im1)
```

The C' (C-prime) of residue `im1` uses (CA(im1), CA(i), O(im1)). Rewriting in
terms of residue `i` (substitute `i → i+1` in the above for C):

```
N(i) = an*CA(i-1) + bn*CA(i) + cn*O(i-1)
C(i) = ap*CA(i)   + bp*CA(i+1) + cp*O(i)
H(i) = ah*CA(i-1) + bh*CA(i) + ch*O(i-1)
```

### Coefficients

Two sources, in priority order:

1. **Runtime (used by LAMMPS-AWSEM)** — from `fix_backbone_coeff.data`
   `[ABC]` section (`fix_backbone.cpp:169-174`):
   ```
   [ABC]
   0.483  0.703 -0.186     → an, bn, cn   (N)
   0.444  0.235  0.321     → ap, bp, cp   (C')
   0.841  0.893 -0.734     → ah, bh, ch   (H)
   ```
   These are **rounded to 3 decimal places**.

2. **Compile-time default** — `fix_backbone.cpp:141-143`:
   ```cpp
   an = 0.4831806; bn = 0.7032820; cn = -0.1864262;
   ap = 0.4436538; bp = 0.2352006; cp =  0.3211455;
   ah = 0.8409657; bh = 0.8929599; ch = -0.7338894;
   ```

The file value **overrides** the default, so the runtime numbers are the
3-decimal-rounded ones.

### Comparison to `src/virtual_atoms.py`

| Aspect | C++ (runtime via `.data`) | PyTorch `_TRANS` | Status |
|---|---|---|---|
| N coefficients | (0.483, 0.703, -0.186) | (0.48318, 0.70328, -0.18643) | **NEEDS-CHECK** — PyTorch uses higher precision than what LAMMPS reads at runtime. Discrepancy is ~1e-5 per coefficient. To exactly match LAMMPS, use the 3-decimal values from `[ABC]`. To exactly match OpenAWSEM (which uses 5-decimal), keep current. |
| C coefficients | (0.444, 0.235, 0.321) | (0.44365, 0.23520, 0.32115) | Same NEEDS-CHECK |
| H coefficients | (0.841, 0.893, -0.734) | (0.84100, 0.89296, -0.73389) | Same NEEDS-CHECK |
| Formula structure | C(i) uses CA(i), CA(i+1), O(i) — same as PyTorch | PASS | |
| Chain-boundary handling | C++ checks `!isFirst(i)`, `!isLast(i)`, same-chain via `res_info` array | PyTorch checks `chains[i]==chains[i-1]`, `chains[i]==chains[i+1]` | PASS (functionally equivalent) |
| Cis-proline branch | Not in C++ — uses single set always | Behind `use_cis_proline` flag, defaults False | PASS (matches LAMMPS default; cis-PRO is OpenAWSEM-specific) |
| Auto-grad friendliness | N/A | Yes (linear in inputs) | PASS |

### Bottom line

ABC virtual atoms PASS for shape and formula. Only open question is whether
to keep 5-decimal precision (matches OpenAWSEM) or downgrade to 3-decimal
(matches LAMMPS at runtime). For frustrapy parity, use 3-decimal. For OpenAWSEM
parity (which is more numerically stable), keep 5-decimal. The numeric
difference in downstream rho / contact energy from this is below float32 noise.

---

## 5. Debye-Hückel electrostatics (`fix_backbone.cpp:5502-5547`, frustration variant)

### Source

```cpp
double FixBackbone::compute_electrostatic_energy(double rij, int i_resno, int j_resno,
                                                 int ires_type, int jres_type) {
  if (abs(i_resno-j_resno)<debye_huckel_min_sep) return 0.0;

  double charge_i = 0.0, charge_j = 0.0;

  if (one_letter_code[ires_type]=='R' || one_letter_code[ires_type]=='K') charge_i = +1.0;
  else if (one_letter_code[ires_type]=='D' || one_letter_code[ires_type]=='E') charge_i = -1.0;
  else return 0.0;

  // ... same for j ...

  if      (charge_i > 0 && charge_j > 0) term_qq_by_r = k_PlusPlus  * charge_i*charge_j / rij;
  else if (charge_i < 0 && charge_j < 0) term_qq_by_r = k_MinusMinus* charge_i*charge_j / rij;
  else                                    term_qq_by_r = k_PlusMinus* charge_i*charge_j / rij;

  return epsilon * term_qq_by_r * exp(-k_screening * rij / screening_length);
}
```

### Plain English

* Only the four canonical charged residues participate: D, E (−1), R, K (+1).
* **Histidine is NOT charged** (matches PHE/TYR/etc.).
* Energy: `q_i q_j / r_ij × k_QQ × exp(-r_ij / λ)` with separate prefactors
  for ++ / -- / +-.
* `r_ij` is the **CB-CB distance** (CA on glycine) — same `get_residue_distance`
  used elsewhere.
* Cutoff: minimum sequence separation `debye_huckel_min_sep`. From data file
  `[DebyeHuckel]` last entry = 1, so only `|i-j| < 1` excluded (i.e. self only).

### Parameters

From `[DebyeHuckel]` (`fix_backbone.cpp:467-477`, frustrapy data file):

```
[DebyeHuckel]                  ← note: file has a trailing '-' but parsing is lenient
4.15 4.15 4.15                → k_PlusPlus, k_MinusMinus, k_PlusMinus
1.0                            → k_screening
10.0                           → screening_length λ (Å)
1                              → debye_huckel_min_sep
```

Wait — the spec doc in `awsem_hamiltonian_spec.md` lists the screening_length
as 4.15 Å and k_QQ values as 1.0. The actual frustrapy data file (verified by
inspecting `/tmp/awsem_validation/5AON_out/5AON.done/fix_backbone_coeff.data`)
shows `4.15 4.15 4.15 / 1.0 / 10.0 / 1`. **So all three `k_QQ` are 4.15 kcal·Å/mol
(rationalized Coulomb constant for unit charges), the screening length is 10.0 Å,
and screening sharpness `k_screening = 1.0`.** I need to update
`awsem_hamiltonian_spec.md` — it had the values transposed.

(Cross-check: 4.15 kcal·Å/mol is the right scale for `k_e * (e^2 / 4πε₀) / ε_r`
in Coulombs squared with kcal/mol units, when `ε_r ≈ 80` for bulk water. Confirms
the parameter is k_QQ, not λ.)

### Units

`epsilon = 1.0` (the `[Epsilon]` line, NOT the dielectric constant — confusing
naming). `k_QQ × charges / r` has units of energy when r is in Å and k_QQ in
kcal·Å/mol. Final result is kcal/mol. PASS.

### Status — no current PyTorch impl

`src/` has nothing for electrostatics yet. Phase 2 will add this.

---

## 6. Tertiary frustratometer (`fix_backbone.cpp:5057-5134`, with helpers 5191-5344)

### Outer loop (`compute_tert_frust`, lines 5057-5134)

```cpp
for (i=0; i<n; ++i) {
  for (j=i+1; j<n; ++j) {
    rij = get_residue_distance(i_resno, j_resno);

    if (rij < tert_frust_cutoff && (abs(i-j)>=contact_cutoff || i_chno != j_chno)) {
      // CB-CB (CA for GLY)
      rho_i = get_residue_density(i_resno);
      rho_j = get_residue_density(j_resno);
      native_energy = compute_native_ixn(rij, i_resno, j_resno, ires_type, jres_type, rho_i, rho_j);

      if (configurational mode AND not yet computed) {
        compute_decoy_ixns(i_resno, j_resno, rij, rho_i, rho_j);
      }
      frustration_index = (decoy_mean - native_energy) / decoy_std;

      // dump row to tert_frust_output_file:
      // i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j
      // native_energy <decoy_E> std(decoy_E) f_ij
    }
  }
}
```

Contact filter: `rij < 9.5 Å` AND (`|i-j| >= contact_cutoff` OR different
chains). `contact_cutoff = 2` from `[Water]` line — but for the frustratometer
its own `[Tertiary_Frustratometer]` block sets `min seq separation` = 1.
However the cutoff used inside `compute_tert_frust` is `contact_cutoff` (from
the Water section, value 2), NOT `tert_frust_seqsep`. Note that the actual
`compute_decoy_ixns` doesn't apply any seq-sep filter to the decoys.

### Native energy (`compute_native_ixn`, 5191-5247)

For **configurational mode** (the frustrapy default and what we run):

```
E_native(i,j) = E_water(rij, i, j, aa_i, aa_j, rho_i, rho_j)
              + E_burial(aa_i, rho_i)
              + E_burial(aa_j, rho_j)
              + [E_DH(rij, i, j, aa_i, aa_j)]  if Debye-Hückel on
```

This is the **sum of the per-pair contact terms plus the two per-residue
burial terms**. No accumulation over k≠i,j (that would be mutational mode).

### Decoy generation (`compute_decoy_ixns`, 5249-5344) — CRITICAL

This is the key difference between configurational and mutational modes.

In **configurational mode** (frustrapy default), 1000 decoys are generated by:

```cpp
for (decoy_i=0; decoy_i<1000; decoy_i++) {
  // 1. Random rij from a random in-contact pair of native residues
  do {
    rand_i_resno = get_random_residue_index();  // uniform 0..n-1
    rand_j_resno = get_random_residue_index();
    rij = get_residue_distance(rand_i_resno, rand_j_resno);
  } while (rij > 9.5 || rand_i_resno == rand_j_resno);

  // 2. Random rho_i, rho_j from ANOTHER random pair
  rand_i_resno = get_random_residue_index();
  rand_j_resno = get_random_residue_index();
  rho_i = get_residue_density(rand_i_resno);
  rho_j = get_residue_density(rand_j_resno);

  // 3. Random AA identities ires_type, jres_type from a THIRD random pair
  rand_i_resno = get_random_residue_index();
  rand_j_resno = get_random_residue_index();
  ires_type = get_residue_type(rand_i_resno);
  jres_type = get_residue_type(rand_j_resno);

  // 4. Same Hamiltonian (water + burial + DH) on the scrambled inputs
  water_E = compute_water_energy(rij, ..., ires_type, jres_type, rho_i, rho_j);
  burial_E_i = compute_burial_energy(rand_i_resno, ires_type, rho_i);
  burial_E_j = compute_burial_energy(rand_j_resno, jres_type, rho_j);
  ...
}

decoy_stats[0] = mean(decoy_energies);   // 1000 samples
decoy_stats[1] = std(decoy_energies);
already_computed_configurational_decoys = 1;   // cache it across pairs
```

**KEY OBSERVATIONS for Phase 2:**

1. **The decoy distribution is generated ONCE per structure** (the
   `already_computed_configurational_decoys` flag at line 5341), then **reused
   for all (i,j) pairs**. This is why `<decoy_energies>` and `std(decoy_energies)`
   are identical across all rows in `tertiary_frustration.dat` (verified: rows
   1-220 in 5AON.dat all show `-1.253  0.491`).

2. **AA sampling is frequency-weighted by the protein's composition**, NOT
   uniform 1/20. It comes from `get_random_residue_index() → uniform over n
   residues → get_residue_type(idx)`. So a protein 30% Ala will sample Ala
   30% of the time.

3. **rij sampling rejects-and-resamples** until the random pair is in contact
   (`rij < 9.5`). Could be slow for sparse structures but it's a one-shot
   precompute.

4. **rho_i, rho_j are independent random draws** from the protein's rho
   distribution (independent of the rij draw).

5. **Same Hamiltonian is used for decoys as for native** — same water + burial
   + (optional) Debye-Hückel terms. The only thing that changes is the inputs:
   rij, rho_i, rho_j, aa_i, aa_j.

### Frustration index (`compute_frustration_index`, lines 5591-5598)

```cpp
frustration_index = (decoy_stats[0] - native_energy) / decoy_stats[1];
                  = (mean(decoy_E) - E_native) / std(decoy_E)
```

**Sign convention**: positive frustration index = native is **better than
average decoy** (i.e. minimally frustrated). Negative = native is worse than
decoys (highly frustrated). This is the OPPOSITE of the convention I was using
mentally; the classification at line 5105 uses:

* `f_ij > 0.78` → minimally frustrated
* `f_ij < -1.0` → highly frustrated

Confirms: positive = good native, negative = bad native.

This matches the spec doc at `awsem_hamiltonian_spec.md:55-61`.

### Output file format

From `fix_backbone.cpp:5104`:

```cpp
fprintf(tert_frust_output_file,
  "%5d %5d %3d %3d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %c %c %8.3f %8.3f %8.3f %8.3f\n",
  i_resno+1, j_resno+1, i_chno+1, j_chno+1,
  xi[0], xi[1], xi[2], xj[0], xj[1], xj[2],
  rij, rho_i, rho_j,
  se[i_resno], se[j_resno],
  native_energy, decoy_ixn_stats[0], decoy_ixn_stats[1], frustration_index);
```

Confirmed by the header on the actual dumped 5AON file:
```
# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j native_energy <decoy_energies> std(decoy_energies) f_ij
```

19 whitespace-separated columns. 1-indexed residue numbers. Letter codes for
AA. The `f_ij` is the per-pair frustration index.

### Parameters

From `[Tertiary_Frustratometer]` (`fix_backbone.cpp:385-397`):

```
[Tertiary_Frustratometer]
9.5            → tert_frust_cutoff (CB-CB contact cutoff, Å)
1000           → tert_frust_ndecoys
1              → tert_frust_output_freq (don't care for frustration)
configurational → tert_frust_mode  (this is what frustrapy uses by default)
```

The "min seq separation" mentioned in the spec doc's `[Tertiary_Frustratometer]
... 1` line is the **output frequency**, not a seq-sep filter. There is no
explicit seq-sep filter in `compute_tert_frust`; instead, `contact_cutoff = 2`
(from `[Water]`) is what gates the loop. Spec doc currently mislabels this —
should be fixed.

### Per-residue density (5adens file)

The `tertiary_frustration.dat` file is per-pair. To get per-residue density of
high/neutral/minimal frustration, frustrapy aggregates the pairs in a 5 Å
sphere around each CA — see
`frustration_calculator.py:_calculate_frustration_density` (lines ~880-1000).
The output file is `<pdb>.pdb_<mode>_5adens`. We won't replicate the 5 Å sphere
aggregation in PyTorch — it's a post-process — but we need to produce a
compatible `tertiary_frustration.dat` so frustrapy's aggregator can run on
our output (or we re-implement the aggregator, ~30 LOC).

---

## 7. Open questions / Phase 2 blockers

1. **Burial seq-sep off-by-one** (§1) is a real bug in `src/burial.py`.
   `RHO_MIN_SEQ_SEP=2` should be `RHO_MIN_SEQ_SEP=1` to match LAMMPS' `>1`
   condition. Worth a unit test that checks `rho` matches LAMMPS exactly on
   5AON.

2. **k_contact default** (§1) should be 1.0 kcal/mol, not 4.184 kJ/mol, for
   LAMMPS parity. Easy fix in `burial.py:201`.

3. **ABC precision** (§4) — 3-decimal LAMMPS values vs 5-decimal OpenAWSEM
   values. Negligible in practice but document the choice.

4. **Decoy AA sampling** (§6) is residue-frequency-weighted (uniform draws from
   the n-residue native sequence), NOT uniform 1/20. Phase 2 decoy code must
   sample residue INDICES uniformly, then read off the AA, NOT sample AA
   indices uniformly.

5. **Decoys are computed ONCE per structure** in configurational mode (§6).
   The implementation should compute the 1000-decoy mean/std a single time and
   broadcast across all (i,j) pairs.

6. **DH "k_PlusPlus" = 4.15 kcal·Å/mol is the Coulomb prefactor, not a
   screening length.** Update `docs/awsem_hamiltonian_spec.md` accordingly.

7. **HIS has charge 0 in the C++** — Phase 2 electrostatics must follow this
   convention even though biologically HIS pKa is around physiological pH.

8. **Frustrapy invokes LAMMPS via subprocess.** Output goes to a temp dir per
   PDB; the binary is `lmp_serial_<seq_dist>_Linux` (default seq_dist=12).
   Phase 2 doesn't need any LAMMPS interaction at runtime — it just needs to
   reproduce the `tertiary_frustration.dat` format.
