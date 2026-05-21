# frustrapy vs. our PyTorch port — kwarg / behaviour comparison

Last updated: 2026-05-20 (post LAMMPS-compat fix pass)

Mapping our top-level `compute_frustration(...)` and `calculate_frustration(...)`
API onto frustrapy 0.1.1's `calculate_frustration()`.

## Kwarg parity table

| frustrapy kwarg | type | our kwarg | type | aligned? | notes |
|---|---|---|---|---|---|
| `pdb_file` | `Optional[str]` | `pdb_file` | `Optional[str/Path]` | yes | We require it; frustrapy can auto-download from RCSB via `pdb_id` (we don't). |
| `pdb_id` | `Optional[str]` | `pdb_id` (in adapter) | accepted, IGNORED | partial | We don't auto-download. Adapter accepts the kwarg for API parity but raises TypeError if `pdb_file` is None. |
| `chain` | `Union[str, List[str], None]` | `chain` (in adapter) | same | yes | Adapter accepts list. `compute_frustration` itself only takes `Optional[str]`; lists fall through to a post-filter. |
| `residues` | `Optional[Dict[str, List[int]]]` | `residues` | same | yes | Post-filter on the result dataframes (decoys still computed on the full structure). |
| `electrostatics_k` | `Optional[float]` | `electrostatics_k` | same | **semantics differ** | See Fix 1 in `lammps_compat_fixes.md`. We default to **NOT adding DH to E_native** (matches LAMMPS analysis pipeline). Pass `include_dh_in_e_native=True` to opt-in. |
| — | — | `include_dh_in_e_native` | `bool` default `False` | NEW | Our opt-in DH inclusion flag. No frustrapy equivalent. |
| `seq_dist` | `int = 12` | `seq_dist` | same | yes | |
| `mode` | `str = 'configurational'` | `mode` | `Literal[3 modes]` same | yes | |
| `graphics` | `bool = True` | — | accepted, IGNORED in adapter | partial | We emit no VMD/PyMOL files (Phase 6 polish). |
| `visualization` | `bool = True` | — | accepted, IGNORED | partial | Same as `graphics`. |
| `results_dir` | `Optional[str]` | `output_dir` | `Optional[str/Path]` | **renamed** | Adapter accepts `results_dir` and maps it to `output_dir`. |
| `debug` | `bool = False` | — | accepted, IGNORED | partial | No debug-mode preservation of intermediate files. |
| `pbar` | tqdm flag | — | accepted, IGNORED | partial | No progress bar yet. |
| `is_mutation_calculation` | `bool = False` | — | accepted, → `mode='mutational'` | yes | Legacy synonym handled in adapter. |
| `n_decoys` | (hard-coded 1000 in frustrapy) | `n_decoys` | `int = 1000` | better | We expose it. |
| `device` | (CPU only) | `device` | `'auto'/'cuda'/'cpu'` | better | We support CUDA. |
| `seed` | (unseeded `srand(NULL)`) | `seed` | `int = 0` | better | We make decoy sampling reproducible. |
| `precision` | (fixed %8.3f) | `precision` | `int = 3` | better | We expose decimal places. |
| `dtype` | (fixed float32 internally) | `dtype` | `torch.dtype = float64` | better | We default to float64 for analytical precision. |
| — | — | `keep_incomplete_backbone` | `bool` default `False` | NEW | Drop residues missing N/CA/C/O. Default matches LAMMPS-AWSEM. |
| — | — | `include_dna` | `bool` default `False` | NEW | Opt-in DNA placeholder rows for byte-comparable parity on protein-DNA complexes. |
| — | — | `lammps_compat_altloc` | `bool` default `False` | NEW | Opt-in altloc-B duplication for byte-comparable parity on PDBs with alt-conformers. |

## Default-behaviour comparison table

Each row gives the default behaviour of `compute_frustration(pdb)` vs.
`frustrapy.calculate_frustration(pdb)` on identical input.

| Aspect | frustrapy default | our default | After our LAMMPS-compat flags on |
|---|---|---|---|
| **`E_native` includes DH** | NO (analysis omits DH even when sim used it) | NO (matches) | Toggleable via `include_dh_in_e_native=True` |
| **`E_native` on `electrostatics_k=4.15`** | byte-identical to `k=0` run | byte-identical to `k=None` run | Adds DH when `include_dh_in_e_native=True` |
| **Multi-model PDB** | first model | first model (stop at `ENDMDL`) | unchanged |
| **altloc handling** | BioPython picks highest occupancy (tie → first added → 'A') | accept '' or 'A' (matches BioPython on ties) | With `lammps_compat_altloc=True`: emit altloc-B as duplicate-density rows in 5adens |
| **DNA chains** | Listed in `pdb.equivalences` but `pdb.atom[atom_name="CA"]` has no rows → zip-bug truncates protein output | Drop entirely (clean) | With `include_dna=True`: emit DNA placeholder rows in 5adens (reproduces frustrapy's zip-bug) |
| **Residues missing backbone** | `PDBToCoordinates.py` skips if N/CA/C/O missing | Drop unless `keep_incomplete_backbone=True` | unchanged |
| **Modified residues (MSE, M3L, CAS)** | Mapped to MET/LYS/CYS via `code` dict | Mapped via `THREE_TO_ONE` | identical |
| **HID/HIE/HIP, CYX, ASH, GLH, LYN** | Not in frustrapy's code dict (mapped to '?' = filtered out) | Mapped to canonical AA | We're more permissive on protonation variants |
| **Sphere radius for 5adens** | hard-coded 5.0 Å | `DEFAULT_DENSITY_RATIO_A = 5.0` | identical |
| **FI thresholds for 5adens classification** | -1.0 (highly), 0.78 (minimally) | same | identical |
| **Pair `i < j` ordering in dumps** | Always i < j (BioPython iteration) | Configurational: i < j; Mutational/Singleresidue: emitter swaps so i < j (Phase 4 fix) | identical |
| **CB-or-CA for non-Gly midpoint** | CB for non-Gly, CA for Gly | same (`src/frustration._xb_coords`) | identical |
| **Decoy count** | 1000 (hard-coded) | 1000 (kwarg) | identical |
| **Decoy sampling distribution** | Native AA frequency (rejection sampler on `rij < 9.5 && p ≠ q`) | Same | identical |
| **Welltype classification rule** | short / water-mediated / long via `r < 6.5` and `ρ < 2.6` | same (verified 100% match on 5AON) | identical |
| **Output device** | CPU only (libc rand) | CPU / CUDA / auto | we support GPU |
| **Per-pair FI Pearson vs. dump** | n/a (reference) | > 0.99 on all 4 panel PDBs | identical |

## Tests that gate these behaviours

| Behaviour | Test |
|---|---|
| DH NOT added to E_native by default | `test_compute_frustration_dh_byte_exact_against_lammps_5AON` (assertion 1) |
| DH added when opted in | `test_compute_frustration_dh_byte_exact_against_lammps_5AON` (assertion 3), `test_compute_frustration_dh_opt_in` |
| 5AON / 11BG default unchanged | `test_density_spearman_against_5adens[5AON]`, `test_density_spearman_against_5adens[11BG]` |
| 1O3S compat flags improve density Spearman | `test_density_spearman_lammps_compat_flags[1O3S]` (gate 0.90) |
| 3F9M compat flags improve density Spearman | `test_density_spearman_lammps_compat_flags[3F9M]` (gate 0.90) |
| 5AON / 11BG unchanged with compat flags on | `test_density_spearman_lammps_compat_flags[5AON, 11BG]` (gate 0.95-0.98) |
| Per-pair FI ≥0.99 Pearson | `test_configurational_fi_validation[*]` (existing, Phase 3c) |
| Welltype 100% match on 5AON | `test_welltype_5AON_byte_match` (existing, Phase 3c) |

## Things we deliberately don't reproduce

The following frustrapy behaviours are bugs or non-physical artefacts
we choose NOT to reproduce by default (opt-in flags are provided for
parity when needed):

1. **`pdb.atom["atom_name"]=="CA"` vs. `pdb.equivalences` zip-mismatch
   on protein-DNA complexes** (1O3S). Available as `include_dna=True`.

2. **Disordered-atom iteration yielding altloc-B as a separate CA
   record** on PDBs with alt-conformers (3F9M). Available as
   `lammps_compat_altloc=True`.

3. **Adding DH to E_native when `electrostatics_k` is set**. NEW
   default: we don't, matching frustratometeR's analysis convention.
   Available as `include_dh_in_e_native=True`.

These are exposed as `metadata["..."]` fields after each
`compute_frustration` call so downstream users can audit which flags
were active.

## Migration cheatsheet (frustrapy → us)

```python
# Before (frustrapy)
import frustrapy
result = frustrapy.calculate_frustration(
    pdb_file="X.pdb", mode="configurational",
    results_dir="out/", graphics=True, debug=True,
)

# After (PyTorch port — DEFAULT behaviour, no compat flags)
from frustration_gpu import calculate_frustration
result = calculate_frustration(
    pdb_file="X.pdb", mode="configurational",
    results_dir="out/",  # renamed kwarg accepted by adapter
    # graphics=True silently consumed (warns once)
    # debug=True silently consumed
)

# After (PyTorch port — FULL frustrapy-compatible 5adens emission)
result = calculate_frustration(
    pdb_file="X.pdb", mode="configurational",
    results_dir="out/",
    include_dna=True,            # match frustrapy's DNA-row emission on protein-DNA PDBs
    lammps_compat_altloc=True,   # match frustrapy's altloc-B duplicate rows on PDBs w/ alt-conformers
    keep_incomplete_backbone=False,  # match LAMMPS PDBToCoordinates.py (default already)
)
```

The defaults are intentionally chosen so that scientifically clean
output is the default and frustrapy's known bugs require explicit
opt-in. Tests cover both code paths.

## Migrating from frustrapy: return-type change

frustrapy returns a 4-tuple:

```python
# frustrapy 0.1.1
pdb, plots, density_results, mutational_data = frustrapy.calculate_frustration(...)
#   ^Pdb    ^Dict   ^Optional[FrustrationDensityResults]  ^Optional[Dict]
```

This PyTorch port returns a single `FrustrationResult` dataclass:

```python
from frustration_gpu import calculate_frustration

result = calculate_frustration("X.pdb", mode="configurational")
result.pair_records           # pandas.DataFrame, schema documented in compute_frustration docstring
result.singleresidue_records  # pandas.DataFrame, populated only when mode='singleresidue'
result.density_records        # pandas.DataFrame, the 5adens equivalent
result.metadata               # dict[str, Any] — kwargs, wall-clock, decoy stats, etc.
```

Unpacking pattern for code that was written against frustrapy's 4-tuple:

```python
# Before
pdb, plots, dens, _ = frustrapy.calculate_frustration("X.pdb", mode="configurational")

# After
result = calculate_frustration("X.pdb", mode="configurational")
# `pdb` and `plots` have no direct equivalent (we return tabular results, not a
# parsed Pdb object or matplotlib figures). The closest analogues are:
pair_df = result.pair_records       # what frustrapy users want from `pdb.atom`
dens_df = result.density_records    # what frustrapy returns as FrustrationDensityResults.density
```

There is no plan to reintroduce the 4-tuple. The single-dataclass return
is intentional: it is type-checkable, pickleable, and avoids the
positional-unpacking footgun frustrapy users hit when a new field is
added at index 0 in a future release.
