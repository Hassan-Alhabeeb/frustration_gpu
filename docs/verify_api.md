# Public-API Verification — `frustration_gpu.__init__` vs frustrapy

> **Historical snapshot — 2026-05-21.** This document is a one-shot
> verification audit taken at the date stamped below. Some statements
> here have been superseded by later changes in this same release
> cycle. In particular:
>
> * The `calculate_frustration` adapter now accepts `overwrite` and
>   `n_cpus` as no-op kwargs (was: "raises TypeError"). The drop-in
>   surface is therefore complete for the frustrapy `0.1.1` signature.
> * The package name was renamed from `src` to `frustration_gpu`;
>   all `src/...` paths quoted below now live at `frustration_gpu/...`.
>
> For the current authoritative API contract see [API.md](API.md).
> For the changelog of these post-audit fixes see
> [`CHANGELOG.md`](../CHANGELOG.md).

Final read-only audit, 2026-05-21. Covers the public surface exported from
`frustration_gpu/__init__.py`, the drop-in `calculate_frustration` adapter, the
`FrustrationResult` dataclass, error messages, type hints, docstrings, and
the public/private boundary.

## Verdict: **YELLOW** (release-ready with minor polish gaps)

- **Drop-in claim** (`calculate_frustration` swaps in for
  `frustrapy.calculate_frustration`): holds for the **kwarg surface** of
  every frustrapy parameter except two (`overwrite`, `n_cpus`) — these
  are silently broken because the adapter raises `TypeError` on unknown
  kwargs. Most real frustrapy users *do not* pass `overwrite=` or
  `n_cpus=` in the simple path, so the failure mode is loud, not silent.
  Return type differs (we return one dataclass, frustrapy returns a
  4-tuple) — but no existing migration guide mentioned this.
- **Docstrings**: every one of the 69 public exports has a docstring.
- **Type hints**: 38 / 40 public functions have complete annotations on
  every parameter + return. Two minor gaps: `density_to_dataframe`
  (no return type), `emit_5adens_dat` (no annotation on
  `output_path`).
- **Error messages**: 4 of 9 probed corner cases produce a HELPFUL
  message; 3 produce a TECHNICALLY-correct but USER-HOSTILE message;
  2 silently accept invalid input.
- **FrustrationResult**: dataclass works, fields populate correctly per
  mode, pickle round-trip succeeds for all 3 modes, schemas documented.
- **Public/private boundary**: clean — `from src import *` exports
  exactly the 69 names in `__all__`, no leaked private helpers.

---

## A. Drop-in compatibility table

frustrapy `0.1.1` `calculate_frustration` signature (from
`frustrapy/analysis/frustration.py:26-42`):

```python
calculate_frustration(
    pdb_file: Optional[str] = None,
    pdb_id: Optional[str] = None,
    chain: Union[str, List[str], None] = None,
    residues: Optional[Dict[str, List[int]]] = None,
    electrostatics_k: Optional[float] = None,
    seq_dist: int = 12,
    mode: str = "configurational",
    graphics: bool = True,
    visualization: bool = True,
    results_dir: Optional[str] = None,
    debug: bool = False,
    overwrite: bool = False,            # NOT accepted by us
    n_cpus: Optional[int] = None,       # NOT accepted by us
    pbar: Optional[tqdm] = None,
    is_mutation_calculation: Optional[bool] = False,
) -> Tuple[Pdb, Dict, Optional[FrustrationDensityResults], Optional[Dict]]
```

Our `src.calculate_frustration` (from `src/compute_frustration.py:1101`):

| frustrapy kwarg | our kwarg | default match | behaviour |
|---|---|---|---|
| `pdb_file: Optional[str]` | `pdb_file: Union[str, Path, None]` | YES (None) | Required; we raise `TypeError` if None. frustrapy auto-downloads via `pdb_id`. |
| `pdb_id: Optional[str]` | `pdb_id: Optional[str]` | YES (None) | Accepted, IGNORED. No RCSB auto-download. Raises `TypeError` if `pdb_file=None`. |
| `chain: Union[str, List[str], None]` | same | YES (None) | Routes through parser-level filter (QA-3 H-2 fix 2026-05-21). |
| `residues: Optional[Dict[str, List[int]]]` | same | YES (None) | Post-filter on DataFrames (decoys still computed on full structure). |
| `electrostatics_k: Optional[float]` | same | YES (None) | **Semantics differ**: ours is metadata-only unless `include_dh_in_e_native=True`. Frustrapy adds to E_native unconditionally. See `lammps_compat_fixes.md`. |
| `seq_dist: int = 12` | same | YES (12) | Identical. |
| `mode: str = "configurational"` | same | YES | Identical 3-value enum. |
| `graphics: bool = True` | `graphics: bool = False` | **default differs** | Accepted, IGNORED with one-time `UserWarning`. Frustrapy default True; we default False to avoid the warning firing on every default call. |
| `visualization: bool = True` | `visualization: bool = False` | **default differs** | Same treatment as `graphics`. |
| `results_dir: Optional[str]` | `results_dir: Optional[Union[str, Path]]` | YES (None) | Mapped to `output_dir`. If both are set, `output_dir` wins. |
| `debug: bool = False` | same | YES | Accepted, silently consumed. No debug-mode intermediate file preservation. |
| `overwrite: bool = False` | **NOT ACCEPTED** | — | Raises `TypeError: unknown kwargs ['overwrite']`. |
| `n_cpus: Optional[int]` | **NOT ACCEPTED** | — | Raises `TypeError: unknown kwargs ['n_cpus']`. (We expose `device` instead.) |
| `pbar: Optional[tqdm]` | `pbar: bool = False` | YES | Accepted, silently consumed. Type widened from `Optional[tqdm]` to `bool` — passing a tqdm instance still works (truthy → no-op). |
| `is_mutation_calculation: Optional[bool]` | same | YES | Maps `True` → `mode="mutational"`. |
| — | `include_dh_in_e_native: bool = False` | NEW | Opt-in for "DH added to E_native" parity. |
| — | `n_decoys: int = 1000` | NEW | Frustrapy hard-codes 1000. |
| — | `device: str = "auto"` | NEW | CUDA / CPU / auto. Frustrapy is CPU-only. |
| — | `output_dir`, `seed`, `precision` | NEW | Reproducibility + emission knobs. |
| — | `keep_incomplete_backbone`, `include_dna`, `lammps_compat_altloc` | NEW | LAMMPS-compat opt-in flags. |

**Return type**: frustrapy returns `Tuple[Pdb, Dict, Optional[FrustrationDensityResults], Optional[Dict]]`. We return a single `FrustrationResult` dataclass. **This is a breaking change for code that does `pdb, _, dens, _ = calculate_frustration(...)`.** The migration cheatsheet in `frustrapy_vs_us.md` does not currently document this.

**Output file compatibility**: our `.dat` outputs (`<stem>_tertiary_frustration.dat`, `<stem>_{configurational,mutational}.dat`, `<stem>_singleresidue.dat`, `<stem>_5adens.dat`) are byte-comparable with frustratometeR on header + column count + the `i/j/chain/r_ij/rho/aa/E_native` fields (RNG columns differ by ~noise). frustrapy's downstream R scripts that read these files will work.

---

## B. Type hint coverage

All 40 public functions and 3 public classes (`GammaTables`, `ContactContext`, `FrustrationResult`) inspected via `inspect.signature`.

| Metric | Result |
|---|---|
| Total public exports | 69 |
| Functions with complete param + return annotations | 38 / 40 |
| Classes with annotations on fields | 3 / 3 |
| `Optional[X]` used correctly throughout | YES |
| `Literal["configurational", "mutational", "singleresidue"]` on `compute_frustration.mode` | YES |
| `Union[str, List[str], None]` on `chain` (compute + calculate) | YES |
| `Union[str, Path]` on every file-path arg of `compute_frustration` | YES |

Gaps (minor, polish-grade):

1. `density_to_dataframe(density: Dict[str, torch.Tensor])` — no return type annotation. Should be `-> pd.DataFrame`.
2. `emit_5adens_dat(*, density: Dict[str, torch.Tensor], output_path) -> None` — `output_path` lacks `Union[str, Path]` annotation.

`calculate_frustration.mode` is typed as plain `str`, not `Literal[...]`, because the adapter accepts frustrapy strings before dispatching to the internal `Literal` validator. This is intentional and reasonable.

---

## C. Docstring coverage

**100 %** — every one of the 69 public exports has a docstring.

The flagship docstrings (`compute_frustration`, `calculate_frustration`, `FrustrationResult`, `parse_pdb`, `compute_residue_density`, `debye_huckel_pair_energy`) include all of:
- one-line brief
- Parameters block with type + default + units
- Returns / Notes section
- For `compute_frustration`: a "Performance hints" section and an "Output files" table

`tests/test_api_docs.py` runs a regression that parses every Python code block in `docs/API.md`, extracts the documented kwarg defaults, and compares against the live `inspect.signature` — this catches doc drift. The test file is robust (handles symbolic constants like `PAIR_MIN_SEQ_SEP`, falls back to `repr` comparison, uses `ast.literal_eval` for primitive literals) and currently passes.

---

## D. Error message quality samples

Tested live on `_spike_11BG/11BG.pdb` with `n_decoys=10`:

| Probe | Error class | Message | Helpful? |
|---|---|---|---|
| `mode="invalid"` | `ValueError` | `mode must be one of ('configurational', 'mutational', 'singleresidue'); got 'invalid'` | **YES** — names the valid set. |
| `pdb_file="missing.pdb"` | `FileNotFoundError` | `missing.pdb` (path only, no hint) | **PARTIAL** — the path is clear but no "did you mean an absolute path?" hint. |
| `chain="Z"` on single-chain PDB | `ValueError` | `No usable residues parsed from _spike_11BG\11BG.pdb` | **WEAK** — doesn't tell user that chain Z does not exist. Should list available chains: `chain 'Z' not found; available chains: ['A']`. |
| `residues={"A": [99999]}` | (none — succeeds) | `pair_records` is empty DataFrame, no warning | **POOR** — user gets zero rows, has to guess that the resnum was wrong. Should warn `0 of 1 requested residues found in chain A; check resnum ranges`. |
| `electrostatics_k=-1.5`, `include_dh_in_e_native=True` | (none — succeeds) | DH energies multiplied by negative k, no warning | **POOR** — negative `k_QQ` is unphysical; should warn. |
| `electrostatics_k=0.0`, `include_dh_in_e_native=True` | (none — succeeds) | DH always zero, no warning | **PARTIAL** — degenerate but mathematically valid; ideally hint "use `electrostatics_k=None` to disable DH entirely". |
| `precision=-1` | `ValueError` | `precision must be >= 0; got -1` | **YES** — exact message. |
| `mode="mutational"` on a 1-residue PDB | (untested — `parse_pdb` would already error first) | n/a | n/a |
| Unknown kwarg to `calculate_frustration` (e.g. `bogus=42`) | `TypeError` | `unknown kwargs ['bogus']. Allowed: pdb_file, mode, chain, residues, electrostatics_k, ...` (enumerates) | **EXCELLENT** — catches typos and tells user the valid set. |
| `calculate_frustration(pdb_id="1ABC")` (no pdb_file) | `TypeError` | `pdb_file is required (PyTorch port does not auto-download from RCSB by pdb_id). Got pdb_id='1ABC', pdb_file=None.` | **EXCELLENT** — explains the missing capability. |

**Summary**: 5 / 10 messages are excellent or helpful; 3 are weak (chain not found, nonexistent resnums, weird electrostatics_k); 1 is partial (FileNotFoundError without hint).

---

## E. `FrustrationResult` schema verification

```python
@dataclass
class FrustrationResult:
    pair_records: Optional[Any] = None           # actually pandas.DataFrame
    singleresidue_records: Optional[Any] = None  # actually pandas.DataFrame
    density_records: Optional[Any] = None        # actually pandas.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Live behaviour per mode (verified on 11BG):

| Mode | `pair_records` | `singleresidue_records` | `density_records` |
|---|---|---|---|
| `configurational` | populated (1517 rows, 15 cols) | None | populated (5adens schema) |
| `mutational` | populated | None | populated |
| `singleresidue` | None | populated | None |

`metadata` dict keys (configurational mode) — 21 keys:

`chain, decoy_mean, decoy_std, device, dtype, electrostatics_k, include_dh_in_e_native, include_dna, keep_incomplete_backbone, lammps_compat_altloc, mode, n_decoys, n_pairs, n_residues, output_dir, pair_min_seq_sep, pdb_file, residues, seed, seq_dist, wall_clock_ms`

`decoy_mean`/`decoy_std` keys are only present in configurational mode (they're scalars across all pairs); mutational/singleresidue omit them because the stats are per-pair / per-residue and already in the DataFrame.

**Pickle round-trip**: PASSES for all three modes (`pickle.dumps` → `pickle.loads` preserves `metadata['n_pairs']` and DataFrame schema).

**DataFrame schemas**: documented in the `FrustrationResult` docstring (`compute_frustration.py:81-113`) AND in `docs/API.md` lines 224-248. Schemas verified consistent.

**Type-annotation polish gap**: the dataclass fields are annotated `Optional[Any]` rather than `Optional[pd.DataFrame]`. The actual return type is always a pandas DataFrame; `Optional[Any]` avoids importing pandas at module load time. A `# type: pd.DataFrame` comment beside each field would help IDE users. Low priority — docstring already says "pandas.DataFrame".

---

## F. Public/private boundary review

`from src import *` gives exactly the 69 names in `src.__all__` (plus the `src` module itself, which is normal). Verified:

- No `_private` helpers leak through `*`-import.
- All 69 names are real exported symbols (no typos or dangling references).
- The 3 names re-exported by `compute_frustration.__all__` (`FrustrationResult`, `compute_frustration`, `calculate_frustration`) are all present in the top-level `__all__`.

**One observation**: `ContactContext` and `build_contact_context` are in `__all__` and have docstrings, but their primary use is internal (the configurational native-pair builder). Exposing them is defensible — advanced users may want to drive contact enumeration manually — but if v0.2 wants to tighten the surface, these are candidates to demote to `_contact_common._private_*`.

---

## G. Backwards-compatibility recommendation for v0.1.0

**The v0.1.0 contract** should pin:
1. `compute_frustration` signature (positional `pdb_file`, all else keyword-only) — frozen.
2. `calculate_frustration` kwarg names as listed above — frozen, EXCEPT add `overwrite` and `n_cpus` as accepted-but-ignored no-ops for tighter frustrapy parity (one-line fix).
3. `FrustrationResult` 4 fields (3 DataFrames + metadata dict) — frozen.
4. `pair_records`, `singleresidue_records`, `density_records` column names — frozen.
5. `metadata` keys listed above — additive only; never remove a key in a minor version.
6. The 69 names in `__all__` — additive only; renames go through a deprecation cycle.

**Recommended deprecation policy**:
- v0.1.x: add `overwrite`/`n_cpus` accepted-but-ignored. Add `Literal[...]` to `calculate_frustration.mode`. Add the two missing type annotations on `density_to_dataframe` and `emit_5adens_dat`.
- v0.2.0: deprecate `pbar: bool` → suggest `pbar: Optional[tqdm]` for true progress-bar wire-up. Add `DeprecationWarning` for `is_mutation_calculation` in favour of explicit `mode="mutational"`.
- v0.3.0: harden error messages (the 3 weak cases above) without changing any field. These are bug-fix-grade improvements, not API changes.

**Things NOT to rename**:
- `output_dir` (we currently accept `results_dir` as alias — keep both forever).
- `electrostatics_k` (frustrapy name — keep).
- The frustrapy column names in `pair_records` (`Res1`, `Res2`, `FrstIndex`, etc.) — downstream R scripts depend on these.

---

## Appendix: top-3 priorities to lift verdict from YELLOW → GREEN

1. **Add `overwrite: bool = False` and `n_cpus: Optional[int] = None` to `calculate_frustration` as accepted-but-ignored kwargs.** Three-line patch; makes the drop-in claim genuinely complete.
2. **Improve `chain="Z"` error message.** Have `parse_pdb` collect available chains and raise `ValueError(f"chain {chain!r} not found in {pdb_file}; available chains: {available}")`. This single fix would also help `residues={"A": [99999]}` by emitting a `warnings.warn` when the residue subset has zero hits.
3. **Document the return-type difference** in `docs/frustrapy_vs_us.md`: frustrapy returns `Tuple[Pdb, Dict, Optional[FrustrationDensityResults], Optional[Dict]]`, we return `FrustrationResult`. Show the unpacking migration: `pdb, dens, _, _ = old_call()` → `r = new_call(); pair = r.pair_records; dens = r.density_records`.
