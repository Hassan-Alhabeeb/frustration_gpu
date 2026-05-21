# frustration_gpu — Public API Reference

Last updated: 2026-05-20 (Phase 4.5 — LAMMPS-compat fixes shipped)

This document covers every public symbol exported from `src.__init__.py`.
For the conceptual overview see `PHASES_ROADMAP.md`; for the full kwarg-by-kwarg
comparison against frustrapy see `docs/frustrapy_vs_us.md`.

---

## Table of contents

1. [Top-level API](#top-level-api)
   - [`compute_frustration`](#compute_frustration)
   - [`calculate_frustration`](#calculate_frustration-frustrapy-drop-in)
   - [`FrustrationResult`](#frustrationresult)
2. [LAMMPS-compat flags — when to use which](#lammps-compat-flags)
3. [Building blocks](#building-blocks)
   - [Parser — `parse_pdb`, `chain_segments`](#parser)
   - [Burial — `burial_density`, `burial_energy`, `compute_rho`](#burial)
   - [Contact terms — `direct_contact_energy`, `water_mediated_energy`](#contact-terms)
   - [Debye-Hückel — `debye_huckel_energy`, `debye_huckel_pair_energy`](#debye-hückel)
   - [Decoy machinery](#decoy-machinery)
   - [Frustration index + classification](#frustration-index--classification)
   - [Per-residue density](#per-residue-density)
   - [LAMMPS-compatible writers](#lammps-compatible-writers)
   - [Virtual atoms](#virtual-atoms)
4. [Gamma-table loaders](#gamma-table-loaders)
5. [Constants](#constants)
6. [Differences from frustrapy](#differences-from-frustrapy)

---

## Top-level API

### `compute_frustration`

Single-call orchestrator that runs parse → burial → decoys → frustration index
→ classification → optional output emission, returning a
[`FrustrationResult`](#frustrationresult) dataclass.

```python
from frustration_gpu import compute_frustration

result = compute_frustration(
    "5AON.pdb",
    mode="configurational",
    device="auto",
)
```

#### Signature

```python
def compute_frustration(
    pdb_file: str | Path,
    *,
    mode: Literal["configurational", "mutational", "singleresidue"] = "configurational",
    chain: str | None = None,
    residues: dict[str, list[int]] | None = None,
    electrostatics_k: float | None = None,
    include_dh_in_e_native: bool = False,
    seq_dist: int = 12,
    pair_min_seq_sep: int = 2,
    n_decoys: int = 1000,
    device: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 0,
    precision: int = 3,
    dtype: torch.dtype = torch.float64,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
) -> FrustrationResult
```

#### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `pdb_file` | `str` or `Path` | required | Path to input PDB file. |
| `mode` | `"configurational" / "mutational" / "singleresidue"` | `"configurational"` | Decoy ensemble convention. |
| `chain` | `str` or `None` | `None` | Restrict the *whole* pipeline to one chain (parser-level filter). |
| `residues` | `dict[str, list[int]]` or `None` | `None` | Post-filter map `chain → [resnum, ...]` applied to result dataframes. Decoys still computed on the full structure. |
| `electrostatics_k` | `float` or `None` | `None` | Debye-Hückel prefactor `k_QQ`. `None` → DH skipped. `4.15` reproduces stock `fix_backbone_coeff.data`. By default DH is metadata-only; see `include_dh_in_e_native`. |
| `include_dh_in_e_native` | `bool` | `False` | Add DH to per-pair `E_native`. Default `False` matches LAMMPS analysis convention (DH in dynamics only, not analysis). |
| `seq_dist` | `int` | `12` | Sequence-separation cutoff used by `lammps_dump_rho`. `12` matches `lmp_serial_12_Linux`; pass `3` for `lmp_serial_3_Linux`. |
| `pair_min_seq_sep` | `int` | `2` | Outer-loop `|i-j|` requirement for native pairs. |
| `n_decoys` | `int` | `1000` | Decoy ensemble size. |
| `device` | `"auto" / "cuda" / "cpu"` | `"auto"` | Compute device. `"auto"` picks CUDA when available. |
| `output_dir` | `str / Path` or `None` | `None` | If set, write LAMMPS-AWSEM-compatible dump files into this directory. |
| `seed` | `int` | `0` | Master seed for the decoy sampler (reproducible). |
| `precision` | `int` | `3` | Decimal places in DataFrame / `.dat` output (LAMMPS uses `%8.3f`). |
| `dtype` | `torch.dtype` | `torch.float64` | Working precision for energy math. Recommend `float64`. |
| `keep_incomplete_backbone` | `bool` | `False` | If False (LAMMPS-AWSEM default), drop residues missing any of N/CA/C/O. |
| `include_dna` | `bool` | `False` | If True, emit DNA placeholder rows in 5adens output (parity with frustrapy's zip-mismatch bug on protein-DNA PDBs). |
| `lammps_compat_altloc` | `bool` | `False` | If True, insert altloc-B records as shadow residues (parity with frustrapy on alt-conformer PDBs). |

#### Returns

A [`FrustrationResult`](#frustrationresult).

#### Output files (when `output_dir` is set)

| Mode | Files written |
|---|---|
| `configurational` | `<stem>_tertiary_frustration.dat`, `<stem>_configurational.dat`, `<stem>_5adens.dat` |
| `mutational` | `<stem>_tertiary_frustration.dat`, `<stem>_mutational.dat`, `<stem>_5adens.dat` |
| `singleresidue` | `<stem>_singleresidue.dat` |

`<stem>` is `Path(pdb_file).stem`.

#### Example

```python
from frustration_gpu import compute_frustration

# Default = byte-comparable to LAMMPS-AWSEM + frustratometeR on a clean PDB.
result = compute_frustration("11BG.pdb", mode="configurational", device="cuda")

print(f"{result.metadata['n_residues']} residues, "
      f"{result.metadata['n_pairs']} native pairs, "
      f"{result.metadata['wall_clock_ms']:.0f} ms")

# Pair_records is a pandas DataFrame
top5 = result.pair_records.nsmallest(5, "FrstIndex")
print(top5[["Res1", "Res2", "ChainRes1", "ChainRes2", "FrstIndex", "FrstState"]])
```

#### See also

- [`calculate_frustration`](#calculate_frustration-frustrapy-drop-in) — drop-in
  adapter for frustrapy users.
- [`FrustrationResult`](#frustrationresult) — the returned dataclass.

---

### `calculate_frustration` (frustrapy drop-in)

Translates the frustrapy kwarg surface onto `compute_frustration`. Use this
when migrating existing frustrapy code.

#### Signature

```python
def calculate_frustration(
    pdb_file: str | Path | None = None,
    *,
    mode: str = "configurational",
    chain: str | list[str] | None = None,
    residues: dict[str, list[int]] | None = None,
    electrostatics_k: float | None = None,
    include_dh_in_e_native: bool = False,
    seq_dist: int = 12,
    n_decoys: int = 1000,
    device: str = "auto",
    results_dir: str | Path | None = None,   # frustrapy name
    output_dir: str | Path | None = None,    # our name (wins if both set)
    seed: int = 0,
    precision: int = 3,
    graphics: bool = False,                  # accepted, ignored
    debug: bool = False,                     # accepted, ignored
    pbar: bool = False,                      # accepted, ignored
    visualization: bool = False,             # accepted, ignored
    pdb_id: str | None = None,               # accepted, no auto-download
    is_mutation_calculation: bool | None = None,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
) -> FrustrationResult
```

#### Kwarg translations vs frustrapy

| frustrapy | this adapter | Notes |
|---|---|---|
| `results_dir` | `results_dir` (mapped to `output_dir`) | If both set, `output_dir` wins. |
| `chain` (str or list) | `chain` | Accepts list; multi-chain runs the full pipeline then post-filters. |
| `is_mutation_calculation=True` | → `mode="mutational"` | Legacy synonym. |
| `graphics` / `visualization` | accepted, ignored (one-time warn) | VMD/PyMOL emission not yet implemented. |
| `debug`, `pbar` | accepted, ignored | No-op. |
| `pdb_id` | accepted, ignored | We do NOT auto-download from RCSB; pass a local `pdb_file`. |

Unknown kwargs raise `TypeError` — typos surface loudly.

#### Example

```python
# Before (frustrapy)
import frustrapy
r = frustrapy.calculate_frustration(
    pdb_file="X.pdb", mode="mutational",
    results_dir="out/", graphics=True,
)

# After (this package — drop-in)
from frustration_gpu import calculate_frustration
r = calculate_frustration(
    pdb_file="X.pdb", mode="mutational",
    results_dir="out/",  # automatically mapped to output_dir
    # graphics=True silently consumed
)
```

#### See also

- [`compute_frustration`](#compute_frustration) — the underlying single-entry API.

---

### `FrustrationResult`

Container for the three result DataFrames and a metadata dict.

```python
@dataclass
class FrustrationResult:
    pair_records: pd.DataFrame | None        # configurational / mutational
    singleresidue_records: pd.DataFrame | None  # singleresidue
    density_records: pd.DataFrame | None     # configurational / mutational
    metadata: dict[str, Any]
```

#### `pair_records` columns

| Column | Type | Meaning |
|---|---|---|
| `Res1`, `Res2` | int | Author residue numbers (from the PDB ATOM record). |
| `ChainRes1`, `ChainRes2` | str | Chain letters. |
| `DensityRes1`, `DensityRes2` | float | Local CB-density rho (LAMMPS-dump-compatible, `min_seq_sep=seq_dist`). |
| `AA1`, `AA2` | str | One-letter codes (OpenAWSEM order). |
| `r_ij` | float | Effective-CB distance in Å. |
| `NativeEnergy` | float | Per-pair AWSEM energy (water + burial, optionally + DH). |
| `DecoyEnergy` | float | Decoy mean (scalar across all pairs in configurational; per-pair in mutational). |
| `SDEnergy` | float | Decoy std (same scalar/per-pair distinction). |
| `FrstIndex` | float | `(DecoyEnergy − NativeEnergy) / SDEnergy`. |
| `Welltype` | str | `"short"` / `"water-mediated"` / `"long"`. |
| `FrstState` | str | `"highly"` / `"neutral"` / `"minimally"` (Ferreiro thresholds). |

#### `singleresidue_records` columns

`Res`, `ChainRes`, `DensityRes`, `AA`, `NativeEnergy`, `DecoyEnergy`, `SDEnergy`, `FrstIndex`.

#### `density_records` columns

5adens schema: `Res`, `ChainRes`, `Total`, `nHighlyFrst`, `nNeutrallyFrst`,
`nMinimallyFrst`, `relHighlyFrustrated`, `relNeutralFrustrated`,
`relMinimallyFrustrated`.

#### `metadata` keys

| Key | Type | Meaning |
|---|---|---|
| `mode` | str | The mode that was run. |
| `chain` | str/list/None | Effective chain filter. |
| `residues` | dict/None | Effective residue post-filter. |
| `electrostatics_k` | float/None | The `k_QQ` that was used (None = DH skipped). |
| `include_dh_in_e_native` | bool | Whether DH was added to E_native. |
| `seq_dist` | int | Cutoff used for `lammps_dump_rho`. |
| `pair_min_seq_sep` | int | Native-pair `|i-j|` cutoff. |
| `n_decoys` | int | Decoy ensemble size. |
| `device` | str | Effective torch device. |
| `dtype` | str | Effective torch dtype. |
| `seed` | int | RNG seed. |
| `wall_clock_ms` | float | Total runtime. |
| `n_residues` | int | Protein residue count post-filter. |
| `n_pairs` | int | Native pair count (None for singleresidue). |
| `pdb_file` | str | Original input path. |
| `output_dir` | str/None | Emit directory. |
| `keep_incomplete_backbone` | bool | Parser flag. |
| `include_dna` | bool | Parser flag. |
| `lammps_compat_altloc` | bool | Parser flag. |
| `decoy_mean`, `decoy_std` | float | Configurational-mode scalars (omitted for other modes). |

---

## LAMMPS-compat flags

These flags exist for byte-comparable parity with the LAMMPS-AWSEM +
frustratometeR reference pipeline. Defaults produce scientifically clean
output (DNA dropped, altloc-A only, DH not in analysis energy). Each flag
has a single canonical "use when" trigger.

| Flag | Default | Use when... |
|---|---|---|
| `electrostatics_k` (float) | `None` (DH off) | You want DH reported in metadata. Set to `4.15` to reproduce stock `fix_backbone_coeff.data`. |
| `include_dh_in_e_native` | `False` | You explicitly want DH added to per-pair `E_native`. By default LAMMPS-AWSEM's analysis pipeline excludes DH from `tertiary_frustration.dat` even when dynamics used `huckel_flag=true`. |
| `keep_incomplete_backbone` | `False` | You're working with a custom PDB where some residues legitimately lack one of N/CA/C/O and you want them retained anyway (math will see NaNs). |
| `include_dna` | `False` | You're processing a protein-DNA complex (e.g. 1O3S) and need byte-comparable 5adens output to frustratometeR's broken-zip emission. |
| `lammps_compat_altloc` | `False` | You're processing a PDB with alt-conformers (e.g. 3F9M) and need byte-comparable 5adens output — adds altloc-B as a shadow row. |

Full mechanism for each flag in `docs/lammps_compat_fixes.md`.

---

## Building blocks

These are the lower-level functions that `compute_frustration` composes. Most
users will never need them, but they're exported for advanced users who want
to inspect individual terms or customise the pipeline.

### Parser

#### `parse_pdb`

Parse a PDB to PyTorch tensors. See [parser.py](../src/parser.py) docstring
for the full list of returned tensor keys.

```python
parse_pdb(
    pdb_path: str | Path,
    *,
    chains: list[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
) -> dict[str, torch.Tensor | list]
```

Returns a dict with keys `ca_coords`, `n_coords`, `c_coords`, `o_coords`,
`cb_coords`, `residue_types`, `chain_ids`, `residue_numbers`,
`insertion_codes`, `is_gly`, `is_dna`, `is_altloc_b_shadow`,
plus an internal `lammps_emit_rows` list when LAMMPS-compat flags are on.

#### `chain_segments`

```python
chain_segments(chain_ids: list[str]) -> list[tuple[int, int]]
```

Return list of `(start, end_exclusive)` index ranges, one per contiguous chain
in `chain_ids`. Used internally by virtual-atom construction.

### Burial

#### `burial_density`

```python
burial_density(parsed: dict) -> torch.Tensor   # shape (N,)
```

Convenience wrapper around `compute_rho` — substitutes CA for missing CB
(GLY etc.), builds a per-residue chain index, uses internal residue index
0..N−1 for the sequence-separation rule.

#### `compute_rho`

```python
compute_rho(
    cb_or_ca_coords: torch.Tensor,    # (N, 3)
    residue_numbers: torch.Tensor,    # (N,) int
    chain_index: torch.Tensor,        # (N,) int
    *,
    r_min_nm: float = RHO_R_MIN_NM,
    r_max_nm: float = RHO_R_MAX_NM,
    eta_per_nm: float = RHO_ETA_PER_NM,
    min_seq_sep: int = 1,
    coord_units: str = "angstrom",
) -> torch.Tensor                     # (N,)
```

Lower-level density computation. Useful for custom AWSEM-style densities.

#### `burial_energy`

```python
burial_energy(
    parsed: dict,
    *,
    k_contact: float = 1.0,
    k_awsem: float = 1.0,
    burial_gamma: torch.Tensor | None = None,
    return_per_residue: bool = True,
) -> dict[str, torch.Tensor]
```

Three-well burial energy. Returns `{"energy": scalar, "rho": (N,), "per_residue": (N,)}`.

### Contact terms

#### `direct_contact_energy`

```python
direct_contact_energy(
    coords: dict,
    *,
    gamma_direct: torch.Tensor | None = None,
    k_water: float = 1.0,
    r_min: float = 4.5,
    r_max: float = 6.5,
    eta: float = 5.0,
    contact_min_seq_sep: int = 2,
    return_pair_matrix: bool = False,
) -> dict
```

V_direct on all `i < j` pairs satisfying `|i-j| >= contact_min_seq_sep` (same-chain) or any cross-chain pair, with the sigmoid window `[r_min, r_max]` controlling the cutoff.

#### `water_mediated_energy`

```python
water_mediated_energy(
    coords: dict,
    *,
    rho: torch.Tensor,
    gamma_mediated_protein: torch.Tensor | None = None,
    gamma_mediated_water: torch.Tensor | None = None,
    k_water: float = 1.0,
    ...
) -> dict
```

V_mediated. Both direct and mediated are pre-shaped by the same effective-CB
convention.

#### `direct_pair_energy`, `water_mediated_pair_energy`

Scalar per-pair versions for hand-checks.

### Debye-Hückel

#### `debye_huckel_energy`

```python
debye_huckel_energy(
    coords: dict,
    *,
    k_QQ: float = 4.15,
    screening_length: float = 10.0,
    k_screening: float = 1.0,
    min_seq_sep: int = 1,
    epsilon: float = 1.0,
    return_pair_matrix: bool = False,
) -> dict
```

Total electrostatic energy + (optional) per-pair contributions.

#### `debye_huckel_pair_energy`

```python
debye_huckel_pair_energy(
    r_ij: torch.Tensor | float,
    aa_i: int,
    aa_j: int,
    *,
    k_QQ: float = 4.15,
    screening_length: float = 10.0,
    k_screening: float = 1.0,
    epsilon: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor
```

Scalar V_DH for one charged pair. Useful for hand-checks; returns a
0-d tensor that is exactly 0.0 if either residue is not R/K/D/E.
First positional was renamed from `r` to `r_ij` and now accepts either
a scalar or a 0-d / 1-d tensor (Phase 5 polish).

### Decoy machinery

| Function | Mode | Purpose |
|---|---|---|
| `sample_configurational_decoys` | configurational | Sample 1000 (default) AA/rij/rho-triples for the cached configurational decoys. |
| `compute_configurational_decoy_energy` | configurational | Energy on a batch of decoys. |
| `configurational_decoy_stats` | configurational | Cache + return `{"decoy_mean", "decoy_std"}` scalars. |
| `mutational_decoy_stats` | mutational | Per-pair `(decoy_mean, decoy_std)` plus native pair info. |
| `singleresidue_decoy_stats` | singleresidue | Per-residue `E_native`, `decoy_mean`, `decoy_std`, `FI`. |
| `lammps_dump_rho` | — | Compute rho with `min_seq_sep=12` (or other), matching LAMMPS's dump-rho convention. |
| `water_theta`, `burial_switch` | — | Building-block sigmoids reused by all decoy paths. |

### Frustration index + classification

#### `compute_frustration_index`

```python
compute_frustration_index(
    *,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,
    decoy_std: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor
```

Returns `(decoy_mean - e_native) / decoy_std`. Sign convention: positive =
minimally frustrated. Pass `eps > 0` to clamp `decoy_std` for stability on
flat distributions.

#### `classify_frustration`

```python
classify_frustration(
    fi: torch.Tensor,
    *,
    high_threshold: float = -1.0,
    minimal_threshold: float = 0.78,
) -> torch.LongTensor
```

Three-state classification per Ferreiro 2007. Returns `0` (highly), `1`
(neutral), `2` (minimally).

#### `welltype_from_contact`

```python
welltype_from_contact(
    rij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    *,
    r_short: float = 6.5,
    rho_water_cutoff: float = 2.6,
) -> torch.LongTensor
```

Per-contact `Welltype`: `0`=short, `1`=water-mediated, `2`=long. Rules
extracted from `frustratometeR/inst/Scripts/RenumFiles.pl`.

### Per-residue density

#### `compute_residue_density`

```python
compute_residue_density(
    *,
    coords: dict,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    fi: torch.Tensor,
    ratio: float = 5.0,
    classification_thresholds: tuple[float, float] = (-1.0, 0.78),
) -> dict
```

Aggregates per-pair FI into per-residue 5adens counts.

#### `density_to_dataframe`

Convenience: `dict → pandas.DataFrame` with the 5adens column schema.

#### `emit_5adens_dat`

```python
emit_5adens_dat(*, density: dict, output_path: str | Path) -> None
```

Write the LAMMPS-AWSEM `<PDB>_5adens.dat` file.

### LAMMPS-compatible writers

| Function | File written |
|---|---|
| `emit_tertiary_frustration_dat` | `<stem>_tertiary_frustration.dat` (LAMMPS raw format, 1-indexed) |
| `emit_postprocessed_pair_dat` | `<stem>_{configurational,mutational}.dat` (frustratometeR post-processed format) |
| `emit_singleresidue_dat` | `<stem>_singleresidue.dat` |
| `emit_5adens_dat` | `<stem>_5adens.dat` |

All four use the `precision` kwarg (default 3) to match LAMMPS's `%8.3f`.

### Virtual atoms

#### `compute_virtual_atoms`

```python
compute_virtual_atoms(
    parsed: dict,
    *,
    use_cis_proline: bool = False,
) -> dict
```

Engh-Huber Cβ / amide-N / amide-H construction for AWSEM's virtual atoms.
Takes the parsed-PDB dict from `parse_pdb` (uses its `n_coords`, `ca_coords`,
`c_coords`, `chain_ids` keys). Used internally; rarely needed by end users.

---

## Gamma-table loaders

| Function | Returns | Source |
|---|---|---|
| `load_gamma_tables(...)` | `GammaTables` dataclass holding direct + mediated tables | `src/data/gamma.dat` |
| `load_direct_gamma(...)` | direct gamma `(20, 20)` | `src/data/gamma.dat` (lines 1–20) |
| `load_mediated_gamma(...)` | `(gamma_protein, gamma_water)`, each `(20, 20)` | `src/data/gamma.dat` (lines 21–60) |
| `load_burial_gamma(...)` | burial gamma `(20, 3)` | `src/data/burial_gamma.dat` |

All four accept `device` and `dtype` kwargs and return tensors on the
requested device. The decoy-driver level wraps these in `functools.lru_cache`
keyed by `(device_str, dtype_str)` so repeated calls during a sweep don't
re-load from disk.

---

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `BURIAL_KAPPA` | `4.0` | Burial-well steepness (`fix_backbone.cpp`). |
| `BURIAL_RHO_MIN` | `(0.0, 3.0, 6.0)` | Lower edges of the three burial wells. |
| `BURIAL_RHO_MAX` | `(3.0, 6.0, 9.0)` | Upper edges. |
| `RHO_R_MIN_NM` | `0.45` | rho switching window lower edge (nm). |
| `RHO_R_MAX_NM` | `0.65` | rho switching window upper edge (nm). |
| `RHO_ETA_PER_NM` | `50.0` | rho sigmoid steepness. |
| `DH_K_QQ_DEFAULT` | `4.15` | Default `k_QQ` from `fix_backbone_coeff.data`. |
| `DH_SCREENING_LENGTH_A` | `10.0` | Debye length. |
| `DH_K_SCREENING` | `1.0` | DH inverse-screening prefactor. |
| `DH_EPSILON` | `1.0` | Global LAMMPS-AWSEM energy scale. |
| `DH_MIN_SEQ_SEP` | `1` | DH `|i-j|` cutoff (i.e. only self-pair excluded). |
| `DH_CHARGES_FLOAT` | `(0, +1, 0, -1, 0, 0, -1, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0)` | Charge per AA (OpenAWSEM index order). |
| `DEFAULT_CONTACT_CUTOFF_A` | `9.5` | Native-pair contact cutoff. |
| `DEFAULT_N_DECOYS` | `1000` | Default decoy count. |
| `LAMMPS_DUMP_RHO_MIN_SEQ_SEP` | `12` | Default `seq_dist` for the LAMMPS-dump rho. |
| `PAIR_MIN_SEQ_SEP` | `2` | Outer-loop `|i-j|` for native pairs. |
| `WELLTYPE_R_SHORT_A` | `6.5` | Short / long-contact split. |
| `WELLTYPE_RHO_WATER` | `2.6` | Water-mediated rho cutoff. |
| `HIGHLY_FRUSTRATED_THRESHOLD` | `-1.0` | Ferreiro highly-frustrated cutoff. |
| `MINIMALLY_FRUSTRATED_THRESHOLD` | `0.78` | Ferreiro minimally-frustrated cutoff. |
| `DEFAULT_DENSITY_RATIO_A` | `5.0` | 5adens sphere radius. |
| `WELL_SHORT`, `WELL_WATER_MEDIATED`, `WELL_LONG` | `0, 1, 2` | Welltype labels. |
| `CLASS_HIGHLY`, `CLASS_NEUTRAL`, `CLASS_MINIMALLY` | `0, 1, 2` | FrstState labels. |
| `ONE_TO_IDX`, `THREE_TO_ONE` | dicts | AA code mappings (OpenAWSEM gamma order). |

---

## Differences from frustrapy

See `docs/frustrapy_vs_us.md` for the full kwarg-by-kwarg comparison. Summary:

### Renamed kwargs

| frustrapy | this package |
|---|---|
| `results_dir` | `output_dir` (or `results_dir` accepted by `calculate_frustration` adapter) |
| `is_mutation_calculation=True` | `mode="mutational"` |

### New kwargs (no frustrapy equivalent)

| Kwarg | Purpose |
|---|---|
| `include_dh_in_e_native` | Add DH to E_native (default off, matching LAMMPS analysis). |
| `keep_incomplete_backbone` | Match LAMMPS-AWSEM `PDBToCoordinates.py` filter. |
| `include_dna` | Opt-in DNA emission for protein-DNA PDBs. |
| `lammps_compat_altloc` | Opt-in altloc-B shadow rows for alt-conformer PDBs. |
| `n_decoys` | Configurable (frustrapy hard-codes 1000). |
| `device` | CPU / CUDA / auto (frustrapy is CPU-only). |
| `seed` | Reproducible RNG (frustrapy uses libc `srand(NULL)`). |
| `precision` | Decimal places in `.dat` output (frustrapy fixed at `%8.3f`). |
| `dtype` | torch.float64 (frustrapy fixed at internal float32). |

### Accepted-but-ignored kwargs (adapter only)

`graphics`, `visualization`, `debug`, `pbar`, `pdb_id`. Phase 6 will wire
the visualisation flags to real VMD/PyMOL output.

### Numerical-behaviour differences

| Aspect | frustrapy | this package |
|---|---|---|
| Default `E_native` includes DH | NO (analysis omits DH) | NO (matches) |
| Multi-model PDB | first model only | same |
| altloc handling | BioPython picks A on tied occupancy | accept `''` or `'A'` (matches) |
| DNA chains | listed in `equivalences` but `pdb.atom[atom_name=="CA"]` empty → zip-bug truncates protein | drop entirely (clean); opt-in via `include_dna=True` |
| HID/HIE/HIP, CYX, ASH, GLH, LYN | not in code dict (filtered) | mapped to canonical AA |
| Modified residues (MSE, M3L, CAS) | mapped via `code` dict | mapped via `THREE_TO_ONE` (same outcome on the standard set) |

### Output-format differences

`.dat` files are byte-comparable in header/column count + on the
`i/j/chain/r_ij/rho/aa/E_native` fields. Per-row differences confined to
RNG-noise columns (`decoy_mean`, `decoy_std`, `FrstIndex`) and the
documented PDB-coord transformation that LAMMPS applies internally
(`tertiary_frustration.dat` coord columns differ by ~1 Å — both correct,
just different origins).

---

## Cross-links

- Hamiltonian specification: `docs/awsem_hamiltonian_spec.md`
- LAMMPS-AWSEM term-by-term C++ citations: `docs/lammps_awsem_term_spec.md`
- LAMMPS-compat fix details: `docs/lammps_compat_fixes.md`
- frustrapy comparison: `docs/frustrapy_vs_us.md`
- Phase reviews: `docs/phase_{1,2a,2b,2c,3a,3b,3c,4}_review.md`
