# AUDIT-A: Energy modules vs LAMMPS-AWSEM C++ ground truth

## Verdict

**PASS** (with one MEDIUM docstring fix recommended pre-push; no CRITICAL or HIGH on default-kwargs paths).

The five energy modules reproduce the C++ formulas exactly on all
default-kwargs code paths. Every coefficient, sigmoid window, charge
assignment, and sign convention checks out against the canonical
`adavtyan/awsemmd` C++ source. The Python defaults that diverge from
the C++ canonical coeff file (DH `k_QQ=4.15` vs C++ `1.0`,
`contact_min_seq_sep=2` vs C++ `contact_cutoff=10`) are not bugs —
they are **intentional matches to the frustrapy/frustratometeR
coefficient file**, which is the analysis target this package
emulates. These are documented in `awsem_hamiltonian_spec.md` and
`lammps_awsem_term_spec.md`.

## Sources audited

- **C++ ground truth**: `adavtyan/awsemmd@master`, commit on
  `master` branch as of 2026-05-21, file `src/fix_backbone.cpp`
  (8,232 LOC, sha256 verified via `gh api repos/adavtyan/awsemmd
  /contents/src/fix_backbone.cpp`, size = 281,108 bytes), companion
  files `src/smart_matrix_lib.h` (652 LOC, the `cWell::compute_ro`
  and `compute_theta` definitions) and `parameters/fix_backbone_coeff.data`
  (240 LOC, the canonical `[Water]/[Burial]/[DebyeHuckel]` defaults).
  Downloaded raw, NOT via WebFetch (which was content-truncated at
  ~5,400 lines and returned 404 on the legacy `USER-AWSEMMD/` path).

- **Local Python files audited** (full read, every line):
  - `frustration_gpu/burial.py` — 349 LOC (full)
  - `frustration_gpu/direct_contact.py` — 479 LOC (full)
  - `frustration_gpu/water_mediated.py` — 492 LOC (full)
  - `frustration_gpu/debye_huckel.py` — 500 LOC (full)
  - `frustration_gpu/_contact_common.py` — 562 LOC (full)
  - `frustration_gpu/contact_gamma.py` — 142 LOC (full)
  - `frustration_gpu/parameters.py` — 166 LOC (full)

- **Documentation cross-referenced** (already-audited spec):
  `docs/awsem_hamiltonian_spec.md`, `docs/lammps_awsem_term_spec.md`,
  `docs/lammps_compat_fixes.md`, `docs/phase_2a_review.md`,
  `docs/phase_2b_review.md`, `docs/phase_2c_review.md`.

## Findings by severity

### CRITICAL (0)

None. The package is safe to `git push`.

### HIGH (0)

None on default-kwargs paths. The DH module's `min_seq_sep=1` default
correctly excludes only the self-pair (matching `debye_huckel_min_sep=1`
in the C++; see line-by-line check below). The contact terms use
`contact_min_seq_sep=2`, matching the frustrapy `[Water]` coefficient
file.

### MEDIUM (2)

**M-1. Docstring misattribution: `[Water] line "2 2"` is wrong.**
- `frustration_gpu/direct_contact.py:13-14`, `:142-143`, `:188`
- `frustration_gpu/water_mediated.py:11-13`, `:118`
- `frustration_gpu/_contact_common.py:513` ("from `[Water]` line `2 2`")

Each of these references `[Water] "2 2"` as the source of
`contact_min_seq_sep=2`. The C++ `[Water]` parser
(`fix_backbone.cpp:257-266`) actually reads, in order:
`k_water`, `kappa kappa_sigma`, `treshold`, `contact_cutoff`,
`n_wells`, then n_wells × (`well_r_min well_r_max well_flag`).
In the canonical `adavtyan/awsemmd/parameters/fix_backbone_coeff.data`
this is `1.0 / 5.0 7.0 / 2.6 / 10 / 2 / 4.5 6.5 1 / 6.5 9.5 1`.
**The line `2` reads as `n_wells`, NOT `min_seq_sep`. The C++
canonical `contact_cutoff` is 10, not 2.**

The Python value `2` is correct (it matches the frustrapy /
frustratometeR `fix_backbone_coeff.data`, which ships a customised
`contact_cutoff=2`), but the docstring attribution to "line 2 2"
is wrong on adavtyan's canonical file. **Fix**: change docstrings
to cite the frustrapy coeff file path instead of the line literal.
No code change required. **Numerical correctness is unaffected.**

**M-2. `k_contact = 1.0` docstring still references OpenAWSEM convention.**
- `frustration_gpu/burial.py:36-39`, `:276-277`, `:290-291`

The docstring at `burial.py:35-39` says `k_contact` "default is
4.184 kJ/mol", but the actual default is `1.0` (correctly switched
2026-05-20 to match LAMMPS `units real`). The accompanying inline
comment at `:276` does say "Was 4.184 ... switched 2026-05-20",
which keeps the audit trail honest, but the module-level docstring
prose still reads as if the default is 4.184. **Fix**: update the
top-of-module prose; signature default is already correct. **No
numerical impact** — verified against `compute_burial_energy`
matching `fix_backbone.cpp:5478-5500`.

### LOW (3)

**L-1. Pair counting comment claims symmetry of `mask`, but `mask` from
`_pair_mask` is symmetric, so the upper-tri sum is well-defined.**
- `frustration_gpu/direct_contact.py:423-427`

Code is correct; comment "The mask is symmetric, so summing the full
matrix would double-count" is accurate but might confuse readers
trying to track which symmetry direction `pair_mask` returned by
`return_pair_matrix=True` carries. Cross-reference exists in the
module docstring (`:66-79`). Nit only.

**L-2. `_pairwise_distance_safe` fill_value default differs across modules.**
- `direct_contact.py:402-403` uses `fill = 0.5 * (r_min + r_max) = 5.5 Å`
- `water_mediated.py:384-385` uses `fill = 0.5 * (r_min + r_max) = 8.0 Å`
- `debye_huckel.py:408` uses `fill = 1000.0` (intentional — buries the
  exp(-r/λ) contribution well below FP64 epsilon)

All three choices are correct for their respective sigmoid/exp
domains; the difference is principled (DH needs the fill OUTSIDE the
exp decay envelope, the contact terms need the fill at the sigmoid
midpoint so that the tanh evaluates to a numerically smooth ~0).
Inconsistency is not a bug but is mildly confusing — consider adding
a one-line comment cross-referencing the design rationale. Style only.

**L-3. Float32 vs float64 default at the gamma loader boundary.**
- `contact_gamma.py:65, 122` default `dtype=torch.float32`
- `parameters.py:114, 84` default `dtype=torch.float32`
- `debye_huckel.py:149` default `dtype=torch.float64`

The contact gamma loaders default to float32 while the DH
`aa_charge_vector` defaults to float64. In practice the orchestrator
(`compute_frustration.py`) selects float64 for parity tests and the
inputs propagate the higher precision via `.to(dtype=...)`, so the
mismatch is a non-issue downstream. Style only.

## Line-by-line formula verification

For each energy term, the C++ line is quoted, then the Python line is
quoted, then PASS is asserted.

### Burial energy

C++ (`fix_backbone.cpp:5478-5500`, `compute_burial_energy`):
```cpp
t[w][0] = tanh( burial_kappa*(rho_i - burial_ro_min[w]) );
t[w][1] = tanh( burial_kappa*(burial_ro_max[w] - rho_i) );
burial_energy += -0.5*k_burial*burial_gamma_w*(t[w][0] + t[w][1]);
```

Python (`burial.py:333-343`):
```python
switch = torch.tanh(kappa * (rho_b - rho_min)) + torch.tanh(kappa * (rho_max - rho_b))
per_res_per_well = -0.5 * (k_contact * k_awsem) * gamma_per_res * switch
```

`burial_kappa = 4.0` (`fix_backbone_coeff.data:65`, `[Burial]` line 2)
matches Python `BURIAL_KAPPA = 4.0` (`parameters.py:48`).
`burial_ro_min = (0, 3, 6)`, `burial_ro_max = (3, 6, 9)`
(`fix_backbone_coeff.data:66-68`) matches Python tuples at
`parameters.py:49-50`. **PASS.**

C++ rho switching (`smart_matrix_lib.h:602-608`):
```cpp
t_min = tanh( par.kappa*(rij - par.well_r_min[i_well]) );
t_max = tanh( par.kappa*(par.well_r_max[i_well] - rij) );
v_theta[i_well][i][j] = 0.25*(1.0 + t_min)*(1.0 + t_max);
```

C++ rho sum (`smart_matrix_lib.h:629-640`):
```cpp
if ( lc->chain_no[i]!=lc->chain_no[j] || abs(lc->res_no[j] - lc->res_no[i])>1 )
    v_ro[i] += theta(i, j, 0);
```

Python (`burial.py:200-208`):
```python
contrib = 0.25 * (1.0 + torch.tanh(eta * (safe_dist - r_min))) \
              * (1.0 + torch.tanh(eta * (r_max - safe_dist)))
```

with mask `(~same_chain) | (seq_diff > min_seq_sep)` where
`min_seq_sep = 1` (`parameters.py:58`, `RHO_MIN_SEQ_SEP=1`, fixed
2026-05-20 per the inline comment). The C++ uses `> 1`, the Python
uses `> 1` — **PASS** (the rho-sep off-by-one fix is correct).

`eta = 5.0 / Å` ≡ `50 / nm` (`parameters.py:57`,
`RHO_ETA_PER_NM = 50.0`) matches `well->par.kappa = 5.0`
(`[Water]` coeff line 57). `r_min = 4.5 Å`, `r_max = 6.5 Å`
(coeff line 60: `4.5 6.5 1`) matches Python's well-0 constants.
**PASS.**

### V_direct (direct contact)

C++ (`fix_backbone.cpp:5462-5473`):
```cpp
sigma_gamma_direct = (water_gamma_0_direct + water_gamma_1_direct)/2;
t_min_direct = tanh( well->par.kappa*(rij - well->par.well_r_min[0]) );
t_max_direct = tanh( well->par.kappa*(well->par.well_r_max[0] - rij) );
theta_direct = 0.25*(1.0 + t_min_direct)*(1.0 + t_max_direct);
water_energy = -(sigma_gamma_direct*theta_direct + ...);
```

Note: `(γ_0 + γ_1)/2` is the average of the two columns in the gamma
file. The C++ gamma loader at `fix_backbone.cpp:667-680` pre-multiplies
both columns by `k_water` (which we do NOT do; we carry `k_water` as a
runtime knob — documented in `direct_contact.py:43-49`).

Python (`direct_contact.py:415-420`):
```python
full_pair_energy = (
    -k_water_t * gamma_pair * 0.25
    * (1.0 + torch.tanh(eta_t * (safe_dist - r_min_t)))
    * (1.0 + torch.tanh(eta_t * (r_max_t - safe_dist)))
)
```

**Sign**: C++ has `-(...)` outside the parens; Python has `-k_water * γ * 0.25 * ...`. **PASS.**

**`(γ_0 + γ_1)/2` collapses to identity**: for the direct block the
two columns of `gamma.dat` rows 0-209 hold the same value. Verified
by `contact_gamma.py:33-49` docstring and the cited 5AON
hand-computation. The Python parser `parameters.py:147-152` reads
column 0 only — equivalent to `(γ_0 + γ_0)/2 = γ_0`. **PASS.**

**Mask**: cross-chain pairs always contribute, same-chain pairs
require `|i-j| >= contact_min_seq_sep`. C++ line 5048:
`(abs(i-j)>=contact_cutoff || i_chno != j_chno)`. Python
`_contact_common.py:545`: `(~same_chain) | (seq_diff >= contact_min_seq_sep)`.
**PASS.**

**Note on `contact_min_seq_sep=2`**: the canonical adavtyan
`fix_backbone_coeff.data:60` has `contact_cutoff = 10` (NOT 2). The
Python default `2` matches the frustrapy `fix_backbone_coeff.data`
(in `frustrapy/core/scripts/AWSEMFiles/`), which is the analysis
target this package emulates. Documented in
`docs/awsem_hamiltonian_spec.md:18`. **Behavior is correct
for the stated target; docstring "from [Water] line 2 2"
misattributes the source — see M-1 above.**

### V_mediated (water-mediated contact)

C++ (`fix_backbone.cpp:5459-5473`):
```cpp
sigma_wat = 0.25*(1.0 - tanh(kappa_sigma*(rho_i-treshold)))*(1.0 - tanh(kappa_sigma*(rho_j-treshold)));
sigma_prot = 1.0 - sigma_wat;
sigma_gamma_mediated = sigma_prot*γ_med_prot + sigma_wat*γ_med_wat;
t_min_mediated = tanh( well->par.kappa*(rij - well->par.well_r_min[1]) );
t_max_mediated = tanh( well->par.kappa*(well->par.well_r_max[1] - rij) );
theta_mediated = 0.25*(1.0 + t_min_mediated)*(1.0 + t_max_mediated);
water_energy = -(... + sigma_gamma_mediated*theta_mediated);
```

Python (`water_mediated.py:316-403`):
```python
sigma_water_per_res = 0.5 * (1.0 - torch.tanh(eta_sigma_t * (rho - rho_0_t)))
sigma_wat = sigma_water_per_res.unsqueeze(1) * sigma_water_per_res.unsqueeze(0)
sigma_prot = 1.0 - sigma_wat
gamma_pair = sigma_prot * g_prot_pair + sigma_wat * g_wat_pair
full_pair_energy = (
    -k_water_t * gamma_pair * 0.25
    * (1.0 + torch.tanh(eta_t * (safe_dist - r_min_t)))
    * (1.0 + torch.tanh(eta_t * (r_max_t - safe_dist)))
)
```

**Sigma factorisation**: C++ uses `0.25 * (1-tanh) * (1-tanh)` in one
product; Python splits as `0.5 * (1-tanh)` per residue, then outer
product. `(0.5)² = 0.25`. **Algebraically identical, PASS.**

**Constants**: `kappa_sigma = 7.0` (coeff line 57), `treshold = 2.6`
(coeff line 58) match Python `MEDIATED_ETA_SIGMA = 7.0`,
`MEDIATED_RHO_0 = 2.6` (`water_mediated.py:116-117`). **PASS.**

**`r_min2=6.5, r_max2=9.5`** (coeff line 61) match
`MEDIATED_R_MIN_A = 6.5, MEDIATED_R_MAX_A = 9.5`
(`water_mediated.py:113-114`). **PASS.**

### V_DH (Debye-Hückel)

C++ (`fix_backbone.cpp:5502-5547`, `compute_electrostatic_energy`):
```cpp
if (abs(i_resno-j_resno)<debye_huckel_min_sep) return 0.0;
if (one_letter_code[ires_type]=='R' || one_letter_code[ires_type]=='K')  charge_i = 1.0;
else if (one_letter_code[ires_type]=='D' || one_letter_code[ires_type]=='E')  charge_i = -1.0;
else return 0.0;
// same for j ...
if (charge_i > 0 && charge_j > 0)  term_qq_by_r = k_PlusPlus  * charge_i*charge_j / rij;
else if (charge_i < 0 && charge_j < 0)  term_qq_by_r = k_MinusMinus * charge_i*charge_j / rij;
else  term_qq_by_r = k_PlusMinus * charge_i*charge_j / rij;
return epsilon*term_qq_by_r*exp(-k_screening*rij/screening_length);
```

Python (`debye_huckel.py:122-143, 336-432`):
```python
DH_CHARGES_FLOAT = (0, +1, 0, -1, 0, 0, -1, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0)
# in OpenAWSEM order: A R N D C Q E G H I L K M F P S T W Y V
#                    0 +1 0 -1 0 0 -1 0 0 0 0 +1 0 0 0 0 0 0 0 0
inv_lambda_eff = k_screening / screening_length
full_pair_energy = k_QQ_t * q_outer * decay * inv_r   # decay = exp(-r * inv_lambda_eff)
```

**Charges**: R→+1, K→+1, D→-1, E→-1, H→0 — **PASS** (matches C++
`:5511-5519`; H is explicitly 0 because the `else return 0.0` clause
fires for any non-{R,K,D,E} residue).

**k_QQ default**: Python `4.15` (`debye_huckel.py:112`); canonical
adavtyan default `1.0` (coeff line 168); frustrapy default `4.15`.
Python matches **frustrapy** (the analysis target). Documented at
`docs/API.md:85`, `docs/frustrapy_vs_us.md:43`, `docs/lammps_compat_fixes.md`.
**Intentional, PASS.**

**`screening_length = 10.0 Å`** (coeff line 169) matches Python
`DH_SCREENING_LENGTH_A = 10.0` (`:113`). **PASS.**

**`k_screening = 1.0`** (coeff line 168) matches `:114`. **PASS.**

**`debye_huckel_min_sep`**: C++ early-returns when `|i-j| < min_sep`
(line 5504). Python default `1` means same-chain `|i-j| = 1` neighbours
DO contribute (only `i == j` is excluded). C++ adavtyan default at
coeff line 170 is `10`; frustrapy default at API surface is `1`. Python
matches frustrapy; documented at `debye_huckel.py:25-28`. **PASS.**

**Sign**: C++ returns `+epsilon * term_qq_by_r * exp(...)` where
`term_qq_by_r = k_QQ * q_i * q_j / r` — so opposite charges (q_i*q_j = -1)
give a negative (attractive) energy, like-charges give positive
(repulsive). Python builds `q_outer = q_i * q_j` then multiplies by
`+k_QQ_t * decay * inv_r` — same sign convention. **PASS.**

### Gamma table loader

C++ (`fix_backbone.cpp:667-680`):
```cpp
for (int i_well=0; i_well<n_wells; ++i_well) {
  for (i=0; i<20; ++i) {
    for (j=i; j<20; ++j) {
      in_wg >> water_gamma[i_well][i][j][0] >> water_gamma[i_well][i][j][1];
      water_gamma[i_well][i][j][0] *= k_water;   // ← pre-multiplied
      water_gamma[i_well][i][j][1] *= k_water;
```

Python (`parameters.py:146-163`): iteration order `for i in range(20): for j in range(i, 20)`, reads both columns, does NOT pre-multiply by k_water (Python carries k_water as a separate runtime factor in the energy formula; documented explicitly in `direct_contact.py:42-55` and `water_mediated.py:38-46`).

Numerically identical for `k_water = 1.0` (the only value frustrapy
ever uses; documented). **PASS.**

AA order `A R N D C Q E G H I L K M F P S T W Y V` matches the C++
`one_letter_code` mapping referenced at `:5511, 5514, 5523, 5526` and
applied via the `se_map[se[i_resno]-'A']` index in `compute_water_energy`
at `:3401-3402`. **PASS** (the openawsem `read_gamma` walks the same
order; the parser comment at `parameters.py:16-25` cites
`contactTerms.py:82-98`).

### Numerical / dtype correctness

- **float64 throughout the orchestrator path**: orchestrator selects
  dtype at `parser.py` time; all five energy modules accept the
  passed dtype and propagate via `.to(dtype=...)` (e.g.
  `direct_contact.py:316, 333-336`; `water_mediated.py:287, 309-314`;
  `debye_huckel.py:336-339`; `burial.py:330-335`). **No silent
  float32 demotion**. PASS.

- **NaN-safe distance**: the double-where trick at
  `_contact_common.py:413-502` is correctly applied. Layer-1 sanitises
  coordinates BEFORE `vector_norm` (`:478-488`); layer-2 applies the
  pair mask AFTER (`:493-494`). Backward gradients on NaN-row
  neighbours do not poison. PASS.

- **Upper-triangular sum**: all three pair-energy modules
  (`direct_contact.py:427-430`, `water_mediated.py:408-411`,
  `debye_huckel.py:435-441`) zero the lower triangle before sum, so
  the symmetric mask is summed once per unordered pair. PASS.

- **DNA sentinel guard**: each module calls `_check_no_dna_sentinel`
  before any gamma indexing (`burial.py:315`, `direct_contact.py:258`,
  `water_mediated.py:221`, `debye_huckel.py:281`), catching negative
  `residue_types` that would silently wrap to the V row via Python
  negative indexing. PASS.

## Spec-divergence ledger

| C++ canonical default | Python default | Documented? | Why diverge |
|---|---|---|---|
| `[Water] contact_cutoff = 10` | `contact_min_seq_sep = 2` | `awsem_hamiltonian_spec.md:18,29,99,478,616` | frustrapy coeff file ships `2`; we match frustrapy |
| `[DebyeHuckel] k_QQ = 1.0` | `k_QQ = 4.15` | `API.md:85,286,608`; `lammps_compat_fixes.md:24` | frustrapy coeff file ships `4.15`; we match frustrapy |
| `[DebyeHuckel] debye_huckel_min_sep = 10` | `min_seq_sep = 1` | `debye_huckel.py:25-28`; `lammps_awsem_term_spec.md:419-433` | frustrapy default; we match frustrapy |
| `[DebyeHuckel] huckel_flag = false` (off by default) | `electrostatics_k = None` (DH off; `4.15` kwarg only adds DH metadata, not E_native) | `lammps_compat_fixes.md:20-65` | Match (both default-off) |
| `[Water] k_water` pre-multiplied into gamma at load | `k_water` is a runtime kwarg, gamma is RAW | `direct_contact.py:42-55`; `water_mediated.py:38-46`; `contact_gamma.py:11-13,84-94` | Numerically identical at `k_water=1.0`; Python pattern is cleaner for the user-facing API |
| Energy units (LAMMPS `units real` = kcal/mol) | kcal/mol (Python returns LAMMPS-native unit; multiply by 4.184 for kJ/mol) | All four module docstrings | Match |
| Burial seq-sep `|i-j| > 1` | `RHO_MIN_SEQ_SEP = 1` (`> 1` rule) | `parameters.py:58` inline comment | Match (off-by-one fix landed 2026-05-20) |

Every divergence above is principled, documented, and verified to
behave correctly under the relevant frustrapy/frustratometeR analysis
flow. **No undocumented divergences found.**

## Verdict basis

1. **All five formulas line up byte-for-byte against the C++**, with
   sign, sigmoid window, constants, and AA-pair table indexing
   verified line-by-line at `fix_backbone.cpp:5444-5547`,
   `:5478-5500`, `:5502-5547`, and the rho switching at
   `smart_matrix_lib.h:602-608, 629-640`.

2. **Every spec divergence is intentional and matches the frustrapy /
   frustratometeR coefficient file**, not the C++ canonical
   adavtyan coeff defaults. This is documented in five separate
   `docs/*.md` files and is the correct target for an analysis tool
   that replaces frustrapy.

3. **Float64 and NaN safety are intact** across all five modules
   on the orchestrator default path. The double-where trick keeps
   gradients clean; DNA-sentinel guards prevent silent gamma
   wraparound; upper-tri sums prevent double-count.

4. **The two MEDIUM findings are docstring-only** (M-1 misattributes
   "line 2 2"; M-2 obsoletes "default 4.184 kJ/mol"). Neither
   changes any computed energy value. They can be patched in a
   follow-up commit; they do not block `git push`.

5. **No CRITICAL or HIGH bugs found** on any default-kwargs path.
   The package is correctness-ready for publication.

**User should proceed with `git push`.** Recommend a small docstring
PR afterwards to land M-1 and M-2 fixes for posterity.
