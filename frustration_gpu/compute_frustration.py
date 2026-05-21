"""Top-level orchestrator for AWSEM frustration analysis (Phase 4).

Single public entry point :func:`compute_frustration` that:

1. Parses the PDB.
2. Computes burial density (LAMMPS-dump-compatible rho).
3. Runs the configured decoy machinery (configurational / mutational /
   singleresidue).
4. Computes per-pair (or per-residue) frustration index, Welltype, FrstState.
5. Optionally adds the Debye-Hückel pair energy to the per-pair native
   energy when ``electrostatics_k`` is not None.
6. Aggregates per-residue density (configurational + mutational only).
7. Returns a :class:`FrustrationResult` dataclass with three dataframes +
   a metadata dict; optionally writes the LAMMPS-AWSEM-compatible dump
   files into ``output_dir``.

Subset filters
--------------
* ``chain="A"`` restricts the whole pipeline to a single chain — parsed
  via :func:`src.parser.parse_pdb`'s ``chains=`` kwarg, so the burial,
  decoy, and emitted file contents all see only that chain. This
  reproduces frustratometeR's ``Pdb$Chain`` filter behaviour and
  matches the ``param_sweep/<PDB>_chain_A_only_*.dat`` dumps.
* ``residues={"A": [...]}`` is a post-filter applied AFTER stats are
  computed (the heavy work runs on the full coordinate set so cross-
  chain water/burial contributions stay numerically correct). The
  pair_records DataFrame is filtered to rows where *either* residue is
  in the user's subset. The density DataFrame is filtered to residues
  in the subset.

API parity with frustrapy
-------------------------
``calculate_frustration(pdb_file, mode=..., chain=..., results_dir=...,
electrostatics_k=..., graphics=False)`` maps directly onto our
``compute_frustration(pdb_file, mode=..., chain=..., output_dir=...,
electrostatics_k=...)``. Differences:

* ``graphics`` is silently ignored — we don't emit VMD/PyMOL scripts
  (Phase 6 polish work).
* We return data structures the caller can introspect/save; frustrapy
  returns a wrapper R-object-style namespace. The DataFrames carry the
  same columns as the frustratometeR dumps.
* ``electrostatics_k=None`` (our default) gates DH OFF, matching the
  LAMMPS-AWSEM ``huckel_flag = false`` default. Pass a float to enable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from .debye_huckel import debye_huckel_pair_energy
from .decoys import configurational_decoy_stats, lammps_dump_rho
from .density import compute_residue_density, emit_5adens_dat
from .frustration import (
    classify_frustration,
    compute_frustration_index,
    emit_postprocessed_pair_dat,
    emit_singleresidue_dat,
    emit_tertiary_frustration_dat,
    welltype_from_contact,
)
from .mutational_decoys import PAIR_MIN_SEQ_SEP, mutational_decoy_stats
from .parser import ONE_TO_IDX, parse_pdb
from .singleresidue_decoys import singleresidue_decoy_stats

# Inverse mapping idx → one-letter (built from ONE_TO_IDX, length 20).
_IDX_TO_ONE: list[str] = [""] * 20
for _aa, _i in ONE_TO_IDX.items():
    _IDX_TO_ONE[_i] = _aa


_VALID_MODES = ("configurational", "mutational", "singleresidue")

# Valid floating dtypes for the energy math. Reject ints / complex / bool
# up front with a clean ValueError (audit finding #23 — 2026-05-21).
_VALID_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)

# Module-level dedup set for `calculate_frustration` one-shot UserWarnings.
# Audit finding #35 (2026-05-21): the docstring promises a single warning
# per kwarg (graphics/visualization/overwrite=False/n_cpus) over the
# module's lifetime, but the previous implementation warned on every call.
_CF_WARN_ONCE: set[str] = set()


def _empty_pair_df():
    """Schema-preserving empty pair_records DataFrame.

    Audit findings #20 + #21 (2026-05-21): every zero-pair return path must
    use the SAME columns so concatenation / column lookups don't silently
    drop schema in mutational mode while working in configurational mode.
    """
    import pandas as pd
    return pd.DataFrame(columns=[
        "Res1", "Res2", "ChainRes1", "ChainRes2",
        "DensityRes1", "DensityRes2", "AA1", "AA2", "r_ij",
        "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex",
        "Welltype", "FrstState",
    ])


@dataclass
class FrustrationResult:
    """Container for everything :func:`compute_frustration` produces.

    Fields
    ------
    pair_records : pandas.DataFrame | None
        One row per native (i, j) contact. Columns:
            ``Res1``, ``Res2`` — author residue numbers
            ``ChainRes1``, ``ChainRes2`` — chain letters
            ``DensityRes1``, ``DensityRes2`` — dump-compatible rho
            ``AA1``, ``AA2`` — one-letter codes
            ``r_ij`` — pair distance Å
            ``NativeEnergy`` — per-pair E_native (kcal/mol)
            ``DecoyEnergy`` — per-pair decoy mean (== scalar in configurational)
            ``SDEnergy`` — per-pair decoy std
            ``FrstIndex`` — frustration index
            ``Welltype`` — "short" / "water-mediated" / "long"
            ``FrstState`` — "highly" / "neutral" / "minimally"
        ``None`` for ``mode="singleresidue"``.
    singleresidue_records : pandas.DataFrame | None
        One row per residue for singleresidue mode. Columns:
            ``Res``, ``ChainRes``, ``DensityRes``, ``AA``,
            ``NativeEnergy``, ``DecoyEnergy``, ``SDEnergy``, ``FrstIndex``
        ``None`` for configurational / mutational.
    density_records : pandas.DataFrame | None
        Per-residue density aggregation (5adens schema). ``None`` for
        ``mode="singleresidue"``.
    metadata : dict
        Diagnostic info: mode, chain, residues filter, electrostatics_k,
        device, dtype, seed, wall_clock_ms, n_residues, n_pairs,
        n_residues_unfiltered, n_pairs_unfiltered, decoy_mean / decoy_std
        (configurational only — scalars), output_dir, v_dh (scalar total
        of native Debye-Hueckel energy when ``electrostatics_k`` is set).
        ``n_residues`` / ``n_pairs`` reflect the post-``residues=`` filter
        counts; the ``_unfiltered`` variants are the original computation
        sizes.
    """

    pair_records: Any | None = None        # pandas.DataFrame
    singleresidue_records: Any | None = None
    density_records: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_device(spec: str) -> torch.device:
    """Map ``"auto"`` / ``"cuda"`` / ``"cpu"`` to a real ``torch.device``."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _subset_protein_only(coords_full: dict[str, Any]) -> tuple:
    """Subset the parsed-coord dict to **protein** residues only.

    DNA placeholder rows (``residue_types == -1``) and altloc-B shadow
    rows (``is_altloc_b_shadow == True``) are removed; the math layer
    (burial / contact / decoy) only ever sees clean protein residues.

    Returns
    -------
    coords_subset : dict
        Same schema as :func:`src.parser.parse_pdb` output, with all
        tensor / list fields subset along the residue axis.
    math_to_full_idx : torch.LongTensor
        ``math_to_full_idx[k]`` gives the original index in ``coords_full``
        for the kth row of ``coords_subset``. Used by the density
        emission to align with full-list resnums + altloc-B rows.

    Notes
    -----
    Pre-2026-05-20 builds did not subset (because DNA/altloc-B couldn't
    enter the residue list in the first place). With the new opt-in
    flags this function preserves the legacy "math runs on protein only"
    behaviour while keeping the FULL residue list available for the
    LAMMPS-compatible density emission.
    """
    is_dna = coords_full.get("is_dna")
    is_altb = coords_full.get("is_altloc_b_shadow")
    n_full = int(coords_full["ca_coords"].shape[0])
    if is_dna is None and is_altb is None:
        # Old parser output: nothing to subset.
        return coords_full, torch.arange(n_full, dtype=torch.int64)

    keep = torch.ones(n_full, dtype=torch.bool)
    if is_dna is not None:
        keep &= ~is_dna.to(keep.device)
    if is_altb is not None:
        keep &= ~is_altb.to(keep.device)
    # Also drop any residue with the -1 sentinel residue_type (defensive
    # — DNA mask should already catch them, but mismatched flags would
    # leak otherwise).
    keep &= coords_full["residue_types"].to(keep.device) >= 0
    if bool(keep.all().item()):
        # Nothing to subset.
        return coords_full, torch.arange(n_full, dtype=torch.int64)

    idx = torch.nonzero(keep, as_tuple=True)[0].to(torch.int64)

    def _sub_tensor(t):
        return t[idx.to(t.device)]

    def _sub_list(L):
        return [L[int(i)] for i in idx.tolist()]

    coords_subset = {
        "ca_coords": _sub_tensor(coords_full["ca_coords"]),
        "n_coords": _sub_tensor(coords_full["n_coords"]),
        "c_coords": _sub_tensor(coords_full["c_coords"]),
        "o_coords": _sub_tensor(coords_full["o_coords"]),
        "cb_coords": _sub_tensor(coords_full["cb_coords"]),
        "residue_types": _sub_tensor(coords_full["residue_types"]),
        "chain_ids": _sub_list(coords_full["chain_ids"]),
        "residue_numbers": _sub_tensor(coords_full["residue_numbers"]),
        "insertion_codes": _sub_list(coords_full["insertion_codes"]),
        "is_gly": _sub_tensor(coords_full["is_gly"]),
        "is_dna": _sub_tensor(coords_full["is_dna"]),
        "is_altloc_b_shadow": _sub_tensor(coords_full["is_altloc_b_shadow"]),
    }
    return coords_subset, idx


def _aa_letters(aa_idx: torch.Tensor) -> list[str]:
    return [_IDX_TO_ONE[int(a)] for a in aa_idx.tolist()]


def _configurational_native_pairs(
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor,
    *,
    pair_min_seq_sep: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Enumerate native pairs + per-pair E_native for configurational mode.

    Configurational E_native = V_water(i,j) + V_burial(i) + V_burial(j),
    NO cross-term contributions (matches ``fix_backbone.cpp:5208-5211``).

    The mutational pipeline already builds these primitives. We re-use its
    helpers by calling ``mutational_decoy_stats`` with n_decoys=0? No — that
    re-computes too much. Instead we inline the (lighter) configurational
    native loop here.
    """
    # We build the same components mutational uses but skip cross-terms.
    from ._contact_common import _build_chain_index, _resolve_contact_coords
    from .decoys import (
        DEFAULT_CONTACT_CUTOFF_A,
        DIRECT_R_MAX_A,
        DIRECT_R_MIN_A,
        MEDIATED_R_MAX_A,
        MEDIATED_R_MIN_A,
        WATER_ETA_PER_A,
        WATER_ETA_SIGMA,
        WATER_RHO_0,
        _cached_load_burial_gamma,
        _cached_load_direct_gamma,
        _cached_load_mediated_gamma,
        _dtype_to_str,
    )
    from .mutational_decoys import _burial_residue_energy, _water_pair_full
    from .parameters import BURIAL_KAPPA, BURIAL_RHO_MAX, BURIAL_RHO_MIN

    cb_or_ca = _resolve_contact_coords(coords, device=device)
    n = cb_or_ca.shape[0]
    chain_idx = _build_chain_index(coords["chain_ids"], device=device)

    finite_row = torch.isfinite(cb_or_ca).all(dim=-1, keepdim=True)
    safe_cb = torch.where(finite_row, cb_or_ca, torch.full_like(cb_or_ca, 1.0e6))
    diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)
    dist_full = torch.linalg.vector_norm(diff, dim=-1).to(dtype=dtype)
    finite_pair = finite_row & finite_row.transpose(0, 1)
    dist_full = torch.where(
        finite_pair, dist_full, torch.full_like(dist_full, float("inf"))
    )

    same_chain = chain_idx.unsqueeze(0) == chain_idx.unsqueeze(1)
    idx = torch.arange(n, device=device)
    seq_diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    # Upper-triangular pair enumeration (i < j). Note:
    #   idx.unsqueeze(1)  → (N, 1) rows
    #   idx.unsqueeze(0)  → (1, N) cols
    # So `rows < cols` selects upper-tri (i_row < j_col); torch.nonzero
    # then returns (pair_i = row, pair_j = col) with pair_i < pair_j.
    pair_mask = (
        (dist_full < DEFAULT_CONTACT_CUTOFF_A)
        & ((~same_chain) | (seq_diff >= pair_min_seq_sep))
        & (idx.unsqueeze(1) < idx.unsqueeze(0))
        & finite_pair
    )
    pair_i, pair_j = torch.nonzero(pair_mask, as_tuple=True)
    r_ij_pair = dist_full[pair_i, pair_j]

    aa_native = coords["residue_types"].to(device=device, dtype=torch.int64)
    rho_dev = rho.to(device=device, dtype=dtype)
    rho_i_p = rho_dev[pair_i]
    rho_j_p = rho_dev[pair_j]
    aa_i_p = aa_native[pair_i]
    aa_j_p = aa_native[pair_j]

    device_str = str(device)
    dtype_str = _dtype_to_str(dtype)
    gamma_direct = _cached_load_direct_gamma(device_str, dtype_str)
    gamma_med_prot, gamma_med_wat = _cached_load_mediated_gamma(device_str, dtype_str)
    burial_gamma = _cached_load_burial_gamma(device_str, dtype_str)

    eta_t = torch.as_tensor(WATER_ETA_PER_A, dtype=dtype, device=device)
    eta_sigma_t = torch.as_tensor(WATER_ETA_SIGMA, dtype=dtype, device=device)
    rho_0_t = torch.as_tensor(WATER_RHO_0, dtype=dtype, device=device)
    k_water_t = torch.as_tensor(1.0, dtype=dtype, device=device)
    k_burial_t = torch.as_tensor(1.0, dtype=dtype, device=device)
    burial_kappa_t = torch.as_tensor(BURIAL_KAPPA, dtype=dtype, device=device)

    v_pair = _water_pair_full(
        r_ij_pair, aa_i_p, aa_j_p, rho_i_p, rho_j_p,
        gamma_direct, gamma_med_prot, gamma_med_wat,
        direct_r_min=DIRECT_R_MIN_A, direct_r_max=DIRECT_R_MAX_A,
        mediated_r_min=MEDIATED_R_MIN_A, mediated_r_max=MEDIATED_R_MAX_A,
        eta=eta_t, eta_sigma=eta_sigma_t, rho_0=rho_0_t,
        k_water=k_water_t,
    )
    b_i = _burial_residue_energy(
        aa_i_p, rho_i_p, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=BURIAL_RHO_MIN, burial_rho_max=BURIAL_RHO_MAX,
        k_burial=k_burial_t,
    )
    b_j = _burial_residue_energy(
        aa_j_p, rho_j_p, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=BURIAL_RHO_MIN, burial_rho_max=BURIAL_RHO_MAX,
        k_burial=k_burial_t,
    )
    e_native = v_pair + b_i + b_j
    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "r_ij": r_ij_pair,
        "rho_i": rho_i_p,
        "rho_j": rho_j_p,
        "E_native": e_native,
    }


def _add_dh_to_e_native(
    e_native: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    aa_native: torch.Tensor,
    r_ij: torch.Tensor,
    chain_ids: list[str] | None = None,
    *,
    electrostatics_k: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Add per-pair Debye-Hückel contribution to E_native (opt-in).

    Honours the same DH min-seq-sep + cross-chain rules as the
    scalar :func:`src.debye_huckel.debye_huckel_pair_energy`. This is a
    Python loop over ``n_pair`` — fine for the test panel (1517 pairs on
    11BG ≈ ms-scale), but a future optimisation could vectorise via the
    full dense V_DH(N,N) matrix and indexing.
    """
    n_pair = int(pair_i.numel())
    if n_pair == 0:
        return e_native
    _ = chain_ids  # reserved for future cross-chain DH handling
    aa_i_idx = aa_native[pair_i].cpu().tolist()
    aa_j_idx = aa_native[pair_j].cpu().tolist()
    rij_l = r_ij.detach().cpu().tolist()

    # The DH min-seq-sep gate: same-chain pairs with |i - j| < DH_MIN_SEQ_SEP
    # contribute zero. ``DH_MIN_SEQ_SEP = 1`` by default, so only the self
    # pair (i==j) is excluded — which never appears in native pairs anyway.
    # All native pairs already pass |i - j| >= 2 (pair_min_seq_sep), so the
    # DH gate is satisfied for every native pair. We do not pre-filter here.
    dh_l: list[float] = []
    for k in range(n_pair):
        v = debye_huckel_pair_energy(
            float(rij_l[k]),
            int(aa_i_idx[k]),
            int(aa_j_idx[k]),
            k_QQ=float(electrostatics_k),
        )
        dh_l.append(float(v))
    dh = torch.tensor(dh_l, dtype=dtype, device=device)
    return e_native + dh


def _build_pair_dataframe(
    *,
    coords: dict[str, torch.Tensor],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    r_ij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    e_native: torch.Tensor,
    dm_per_pair: torch.Tensor,
    ds_per_pair: torch.Tensor,
    fi: torch.Tensor,
    welltype_arr: torch.Tensor,
    cls_arr: torch.Tensor,
    precision: int,
):
    """Build the pair_records pandas DataFrame in the post-processed schema."""
    import pandas as pd

    aa = coords["residue_types"]
    chain_ids = coords["chain_ids"]
    resnums = coords["residue_numbers"]

    aa_i_letters = _aa_letters(aa[pair_i])
    aa_j_letters = _aa_letters(aa[pair_j])
    pi_l = pair_i.cpu().tolist()
    pj_l = pair_j.cpu().tolist()
    res1 = [int(resnums[k].item()) for k in pi_l]
    res2 = [int(resnums[k].item()) for k in pj_l]
    c1 = [chain_ids[k] for k in pi_l]
    c2 = [chain_ids[k] for k in pj_l]

    def _round(t: torch.Tensor) -> list[float]:
        return [round(float(v), precision) for v in t.detach().cpu().tolist()]

    well_names = {0: "short", 1: "water-mediated", 2: "long"}
    cls_names = {0: "highly", 1: "neutral", 2: "minimally"}

    df = pd.DataFrame({
        "Res1": res1,
        "Res2": res2,
        "ChainRes1": c1,
        "ChainRes2": c2,
        "DensityRes1": _round(rho_i),
        "DensityRes2": _round(rho_j),
        "AA1": aa_i_letters,
        "AA2": aa_j_letters,
        "r_ij": _round(r_ij),
        "NativeEnergy": _round(e_native),
        "DecoyEnergy": _round(dm_per_pair),
        "SDEnergy": _round(ds_per_pair),
        "FrstIndex": _round(fi),
        "Welltype": [well_names[int(w)] for w in welltype_arr.detach().cpu().tolist()],
        "FrstState": [cls_names[int(c)] for c in cls_arr.detach().cpu().tolist()],
    })
    return df


def _build_singleresidue_dataframe(
    *,
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,
    decoy_std: torch.Tensor,
    fi: torch.Tensor,
    precision: int,
):
    """Build the singleresidue_records DataFrame in the post-processed schema."""
    import pandas as pd

    aa = coords["residue_types"]
    aa_letters = _aa_letters(aa)
    resnums = coords["residue_numbers"]
    chain_ids = coords["chain_ids"]
    n = int(rho.numel())

    def _round(t: torch.Tensor) -> list[float]:
        return [round(float(v), precision) for v in t.detach().cpu().tolist()]

    return pd.DataFrame({
        "Res": [int(resnums[k].item()) for k in range(n)],
        "ChainRes": [chain_ids[k] for k in range(n)],
        "DensityRes": _round(rho),
        "AA": aa_letters,
        "NativeEnergy": _round(e_native),
        "DecoyEnergy": _round(decoy_mean),
        "SDEnergy": _round(decoy_std),
        "FrstIndex": _round(fi),
    })


# ---------------------------------------------------------------------------
# top-level API
# ---------------------------------------------------------------------------

def compute_frustration(
    pdb_file: str | Path,
    *,
    mode: Literal["configurational", "mutational", "singleresidue"] = "configurational",
    chain: str | list[str] | None = None,
    residues: dict[str, list[int]] | None = None,
    electrostatics_k: float | None = None,
    include_dh_in_e_native: bool = False,
    seq_dist: int = 12,
    pair_min_seq_sep: int = PAIR_MIN_SEQ_SEP,
    n_decoys: int = 1000,
    device: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 0,
    precision: int = 3,
    dtype: torch.dtype = torch.float64,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
) -> FrustrationResult:
    """End-to-end AWSEM frustration analysis on a single PDB.

    Mirrors frustrapy's ``calculate_frustration`` signature as closely as
    practical.

    Parameters
    ----------
    pdb_file : str or Path
        Path to the input PDB file.
    mode : {'configurational', 'mutational', 'singleresidue'}
        Decoy ensemble convention. Default ``'configurational'`` (matches
        frustrapy's default).
    chain : str or list[str], optional
        If set, restrict the *whole* pipeline to this chain (or set of
        chains). The filter is applied at the parser level — burial /
        decoys / output all see only the selected chains. Use this to
        reproduce ``param_sweep/<PDB>_chain_A_only_*.dat`` dumps.

        Semantics (canonical): a single ``"A"`` and a list ``["A"]``
        both run the pipeline on chain A residues only. A list
        ``["A", "B"]`` runs on both chains. Because rho / FI / density
        are sensitive to total chain mass, ``chain=["A", "B"]`` and
        ``chain="A"`` followed by ``chain="B"`` produce DIFFERENT numeric
        values for the same chain-A residue — the multi-chain run sees
        cross-chain contacts. This is the LAMMPS-AWSEM convention.
        QA-3 H-2 fix (2026-05-21): both single-chain and multi-chain
        paths now go through the parser filter; the prior adapter that
        ran the full pipeline and post-filtered was numerically
        inconsistent.
    residues : dict[str, list[int]], optional
        Post-filter on the pair_records / singleresidue_records / density
        DataFrames. Maps chain letter → list of author residue numbers to
        keep. For pair_records, a row is kept if EITHER ``Res1`` or
        ``Res2`` matches the subset. NOT applied during decoy sampling —
        the model still sees the full structure (matches frustrapy's
        ``get_frustration(Resno=)`` semantics).
    electrostatics_k : float, optional
        Scales the Debye-Hückel pair-energy term. When set, DH is
        computed and reported in the metadata (``metadata["v_dh"]``)
        but BY DEFAULT it is NOT added to per-pair E_native.

        This default reproduces frustratometeR's behaviour: even when
        the LAMMPS-AWSEM run was launched with ``huckel_flag = true``
        and ``k_QQ = 4.15``, the ``energy.log`` Electro column is
        0.000000 and the ``tertiary_frustration.dat`` ``native_energy``
        column is identical to a ``k_QQ = 0`` run (verified empirically
        against ``benchmark/cpu_baseline/param_sweep/5AON_electro_4p15_*``
        and the matching ``configurational/`` dump). The frustration
        analysis pipeline scores the WATER + BURIAL Hamiltonian only,
        even when DH was active during dynamics.

        To opt-in to the "physically complete" semantics where DH
        contributes to E_native, pass ``include_dh_in_e_native=True``.
        Until that flag is set, ``electrostatics_k`` is a metadata-only
        knob and the per-pair native energy is byte-comparable to
        the LAMMPS-AWSEM dump.

        Default ``None`` → DH not computed at all.
        ``electrostatics_k=4.15`` reproduces the stock
        ``fix_backbone_coeff.data`` value and matches the
        ``param_sweep/<PDB>_electro_4p15_*.dat`` reference dumps.
    include_dh_in_e_native : bool, default False
        .. deprecated:: 0.2.0
            Will be removed in v0.3.0. The DH term is added to ``E_native``
            but NOT to decoy energies, so the resulting FI is NOT a valid
            same-Hamiltonian z-score (audit finding #8, 2026-05-21).
            Set ``electrostatics_k`` alone to get DH as a diagnostic via
            ``metadata['v_dh']`` without polluting the FI.

        Add the per-pair Debye-Hückel term to ``E_native`` when
        ``electrostatics_k`` is set. Off by default to match the
        LAMMPS-AWSEM/frustratometeR analysis convention (DH is part of
        the simulation Hamiltonian but is excluded from frustration's
        ``native_energy``).
    n_decoys : int
        Number of decoys. Default ``1000``. Must be ``>= 2`` (the FI is
        a z-score, so an ensemble of one is degenerate; an ensemble of
        zero produces NaN). Audit finding #2 + #22 (2026-05-21).
    keep_incomplete_backbone : bool, default False
        Forwarded to :func:`src.parser.parse_pdb`. When False (the
        default + LAMMPS-AWSEM convention), drop residues missing
        ANY of N / CA / C / O.
    include_dna : bool, default False
        Forwarded to :func:`src.parser.parse_pdb`. Opt-in compat flag
        for byte-comparable parity with frustratometeR on protein-DNA
        complexes (e.g. 1O3S). Adds DNA pseudo-residues using C1' as
        a CA proxy. **Not physically meaningful for AWSEM frustration**
        — see the parser docstring for limitations.
    lammps_compat_altloc : bool, default False
        Forwarded to :func:`src.parser.parse_pdb`. Opt-in compat flag
        for byte-comparable parity on PDBs with alt-conformers
        (e.g. 3F9M). Inserts altloc-B records as shadow residues
        immediately after their altloc-A counterpart.
    seq_dist : int
        Sequence-separation cutoff used by ``lammps_dump_rho``. Default
        ``12`` matches frustratometeR's bundled ``lmp_serial_12_Linux``.
        Pass ``3`` for the SeqDist=3 binary. Does NOT change the
        ``pair_min_seq_sep`` used for native-pair enumeration.
    pair_min_seq_sep : int
        Outer-loop sequence-separation requirement ``|i - j|`` for native
        pairs. Default ``2``.
    device : {'auto', 'cuda', 'cpu'}
        Compute device. ``'auto'`` chooses CUDA when available.
    output_dir : str or Path, optional
        If set, write LAMMPS-AWSEM-compatible dump files to this directory.
        Files written depend on ``mode`` (see Notes).
    seed : int
        Master seed for the decoy sampler.
    precision : int
        Decimal places used when populating DataFrames + emitted .dat
        files. Default ``3`` (matches LAMMPS ``%8.3f``).
    dtype : torch.dtype
        Working precision for the decoy / energy math. Default
        ``torch.float64`` (recommended — matches Phase 2/3 precision floor).

    Returns
    -------
    FrustrationResult — see the dataclass docstring.

    Notes
    -----
    Output files written (when ``output_dir`` is set):

    * configurational mode: ``<basename>_tertiary_frustration.dat``,
      ``<basename>_configurational.dat``, ``<basename>_5adens.dat``.
    * mutational mode: ``<basename>_tertiary_frustration.dat``,
      ``<basename>_mutational.dat``, ``<basename>_5adens.dat``.
    * singleresidue mode: ``<basename>_singleresidue.dat``.

    Where ``<basename>`` is ``pdb_file.stem``. The post-processed pair
    file is the frustratometeR-style dump (``Res1 Res2 ChainRes1 ...
    FrstIndex Welltype FrstState``); the ``tertiary_frustration.dat`` is
    the LAMMPS raw format (1-indexed positions, int chains, raw coords).

    Performance hints
    -----------------
    * ``device='cuda'`` typically gives 10-50× speedup on n=200-500 PDBs.
    * ``electrostatics_k`` adds a Python loop for the per-pair DH term;
      negligible for ``n_pair < 5000`` but worth vectorising for
      large-PDB sweeps (Phase 5).
    * For the chain filter, the whole pipeline is re-parsed on the
      restricted set — this is correct (rho is sensitive to chain mass)
      but means full vs chain-A runs cannot reuse cached state. Multi-
      chain runs should call once with ``chain=None``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
    # Audit finding #59 (2026-05-21): precision must be a non-negative int,
    # not just any "ordered" value. The previous code did `precision < 0`,
    # which raises a raw TypeError for precision='3'/None/etc. and accepts
    # float values like 1.5 that later fail inside round() with another
    # raw TypeError. Validate as int up front.
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise ValueError(
            f"precision must be a non-negative int; got {precision!r} "
            f"(type {type(precision).__name__})."
        )
    if precision < 0:
        raise ValueError(f"precision must be >= 0; got {precision}")
    if electrostatics_k is not None and electrostatics_k <= 0:
        raise ValueError(
            f"electrostatics_k must be positive (got {electrostatics_k}); "
            f"use None to disable Debye-Huckel electrostatics."
        )
    # Audit finding #2 + #22 (2026-05-21): n_decoys must be int >= 2. An
    # ensemble of 0 produced NaN decoy_mean/std/FI with only a PyTorch
    # warning; an ensemble of 1 produced FI=0 for every pair via the
    # degenerate-std clamp. Both are nonsense; reject before sampling.
    if not isinstance(n_decoys, int) or isinstance(n_decoys, bool):
        raise ValueError(
            f"n_decoys must be int >= 2; got {n_decoys!r} "
            f"(type {type(n_decoys).__name__})."
        )
    if n_decoys < 2:
        raise ValueError(
            f"n_decoys must be >= 2 for a meaningful decoy z-score; "
            f"got {n_decoys}."
        )
    # Audit finding #23 (2026-05-21): dtype must be a floating torch.dtype.
    # parse_pdb fills NaN into coord tensors of this dtype, so int dtypes
    # fail with a raw RuntimeError. Reject up front with a clear message.
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"dtype must be a floating torch.dtype "
            f"(one of {[str(d) for d in _VALID_DTYPES]}); got {dtype!r}."
        )
    # Audit finding #8 + deprecation (2026-05-21):
    # ``include_dh_in_e_native=True`` is mathematically inconsistent —
    # DH is added to E_native but NOT to decoy energies, so the resulting
    # FI is not a valid same-Hamiltonian z-score. We keep the flag
    # functional in v0.2.0 for backward compatibility, but emit a
    # DeprecationWarning so callers migrate before v0.3.0 removes it.
    if include_dh_in_e_native:
        import warnings as _w
        _w.warn(
            "include_dh_in_e_native=True is deprecated and will be "
            "removed in v0.3.0. The Debye-Hueckel term is added to "
            "E_native but NOT to the decoy energies, so the resulting "
            "FrstIndex is not a valid same-Hamiltonian z-score. "
            "Pass electrostatics_k alone (no flag) to compute DH as a "
            "diagnostic — its scalar total appears in metadata['v_dh'].",
            DeprecationWarning,
            stacklevel=2,
        )

    dev = _resolve_device(device)
    pdb_path = Path(pdb_file)
    # QA-3 M-1 fix (2026-05-21): synchronise CUDA before sampling the
    # wall-clock timer so the metadata field reflects actual completed
    # work, not async kernel launch time. Cheap on CPU runs (no-op).
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    start = time.perf_counter()

    # --- parse PDB (chain filter at parser level) -------------------------
    # Accept str OR list[str]; canonical semantic is parser-level filter,
    # regardless of cardinality. The pre-2026-05-21 adapter ran the full
    # pipeline for multi-chain lists and post-filtered the dataframes,
    # which silently changed rho/FI for the same residue depending on
    # whether the caller passed "A" or ["A", "B"]. Now both go through
    # the parser, so chain="A" and chain=["A"] are byte-identical, and
    # chain=["A", "B"] correctly includes cross-chain contacts.
    if chain is None:
        chains_filter = None
    elif isinstance(chain, list):
        # Audit finding #25 (2026-05-21): validate element types so the
        # ambiguous parser.py:309 string-membership behaviour can never
        # leak through. Elements must be single-letter strings.
        if not all(isinstance(c, str) for c in chain):
            raise ValueError(
                f"chain list must contain only str elements; got {chain!r}."
            )
        chains_filter = list(chain)
    elif isinstance(chain, str):
        chains_filter = [chain]
    else:
        raise ValueError(
            f"chain must be None, a str, or list[str]; got {chain!r} "
            f"(type {type(chain).__name__})."
        )
    # Parser supports float32/float64; pass `dtype` directly so downstream
    # tensor ops don't need to upcast.
    coords_full = parse_pdb(
        pdb_path,
        chains=chains_filter,
        device=dev,
        dtype=dtype,
        keep_incomplete_backbone=keep_incomplete_backbone,
        include_dna=include_dna,
        lammps_compat_altloc=lammps_compat_altloc,
    )
    n_residues_full = int(coords_full["ca_coords"].shape[0])
    if n_residues_full == 0:
        raise ValueError(
            f"compute_frustration: no residues parsed from {pdb_path} "
            f"with chain={chain!r}."
        )
    # Audit finding #25 (2026-05-21): partial chain miss should not be
    # silent. parse_pdb only raises when ALL requested chains are missing;
    # chain=["A", "Z"] on a single-chain PDB silently ran A and dropped Z.
    # Reject mismatches loudly so typos in batch scripts surface as errors.
    if chains_filter is not None:
        loaded_chains = set(coords_full["chain_ids"])
        requested_chains = set(chains_filter)
        missing_chains = requested_chains - loaded_chains
        if missing_chains:
            raise ValueError(
                f"compute_frustration: requested chain(s) "
                f"{sorted(missing_chains)} not found in {pdb_path}; "
                f"available chains with usable residues: "
                f"{sorted(loaded_chains)}."
            )

    # --- Split into "math" subset and "emit" view -------------------------
    # AWSEM math (burial / contact / decoys) is defined on protein residues
    # with valid residue_types only. DNA placeholder rows (residue_type=-1)
    # and altloc-B shadows participate in NEITHER LAMMPS-AWSEM's tertiary
    # frustration calc — verified empirically: 1O3S tertiary has only
    # chain A indices (no DNA), 3F9M tertiary has 451 unique indices (no
    # altloc-B). Strip them out before any math.
    coords, math_to_full_idx = _subset_protein_only(coords_full)
    n_residues = int(coords["ca_coords"].shape[0])
    if n_residues == 0:
        raise ValueError(
            f"compute_frustration: no PROTEIN residues parsed from {pdb_path} "
            f"with chain={chain!r}."
        )

    # --- LAMMPS-dump-compatible rho ---------------------------------------
    rho = lammps_dump_rho(coords, min_seq_sep=seq_dist, device=dev)

    # --- mode dispatch ----------------------------------------------------
    pair_df = None
    sr_df = None
    density_df = None
    dm_scalar: float | None = None
    ds_scalar: float | None = None
    n_pairs = 0
    # Audit finding #4 (2026-05-21): the docs promised metadata["v_dh"]
    # for electrostatics_k != None but the code never set it. Compute the
    # native total DH energy whenever electrostatics_k is supplied (it's
    # deterministic and cheap relative to decoy sampling) and report it
    # as a diagnostic scalar regardless of include_dh_in_e_native.
    v_dh_total: float | None = None
    # For configurational empty-contact recovery (audit finding #20):
    # remember whether the decoy sampler raised so we can fall back to a
    # schema-preserving empty result instead of crashing the caller.
    _configurational_empty_fallback = False
    # Audit findings #7 + #19 (2026-05-21): we MUST defer file emission
    # until after the residues= filter is applied, otherwise output_dir
    # contains the full unfiltered dump while the returned DataFrames are
    # filtered. Save the per-mode emit state here and write at the end.
    _emit_state: dict[str, Any] | None = None

    if mode == "configurational":
        # native pairs + per-pair E_native
        native = _configurational_native_pairs(
            coords, rho,
            pair_min_seq_sep=pair_min_seq_sep,
            device=dev, dtype=dtype,
        )
        pair_i = native["pair_i"]
        pair_j = native["pair_j"]
        r_ij = native["r_ij"]
        rho_i = native["rho_i"]
        rho_j = native["rho_j"]
        e_native = native["E_native"]
        n_pairs = int(pair_i.numel())

        # DH diagnostic: compute once and remember total (regardless of
        # whether the user opted into adding it to E_native).
        if electrostatics_k is not None and n_pairs > 0:
            dh_vec = _add_dh_to_e_native(
                torch.zeros_like(e_native), pair_i, pair_j,
                coords["residue_types"], r_ij, coords["chain_ids"],
                electrostatics_k=float(electrostatics_k),
                device=dev, dtype=dtype,
            )
            v_dh_total = float(dh_vec.sum().item())
            if include_dh_in_e_native:
                e_native = e_native + dh_vec
        elif electrostatics_k is not None and n_pairs == 0:
            v_dh_total = 0.0

        # scalar decoy stats (cached once per structure). Audit finding
        # #20 (2026-05-21): on zero-contact valid structures the sampler
        # raises RuntimeError; fall back to a clean empty result so
        # configurational matches mutational/singleresidue behaviour.
        try:
            stats = configurational_decoy_stats(
                coords, rho=rho,
                n_decoys=n_decoys,
                seed=seed,
                device=dev,
                dtype=dtype,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "no in-contact pairs" in msg:
                _configurational_empty_fallback = True
                stats = None
            else:
                raise
        if stats is not None:
            dm = stats["decoy_mean"]
            ds = stats["decoy_std"]
            dm_scalar = float(dm.item())
            ds_scalar = float(ds.item())
        else:
            dm = torch.zeros((), dtype=dtype, device=dev)
            ds = torch.zeros((), dtype=dtype, device=dev)
            dm_scalar = float("nan")
            ds_scalar = float("nan")

        if n_pairs > 0 and not _configurational_empty_fallback:
            dm_per_pair = dm.expand(n_pairs)
            ds_per_pair = ds.expand(n_pairs)
            fi = compute_frustration_index(
                e_native=e_native, decoy_mean=dm_per_pair, decoy_std=ds_per_pair,
            )
            welltype_arr = welltype_from_contact(r_ij, rho_i, rho_j)
            cls_arr = classify_frustration(fi)

            pair_df = _build_pair_dataframe(
                coords=coords,
                pair_i=pair_i, pair_j=pair_j,
                r_ij=r_ij, rho_i=rho_i, rho_j=rho_j,
                e_native=e_native,
                dm_per_pair=dm_per_pair, ds_per_pair=ds_per_pair,
                fi=fi,
                welltype_arr=welltype_arr, cls_arr=cls_arr,
                precision=precision,
            )

            # density aggregation
            density = compute_residue_density(
                coords=coords, pair_i=pair_i, pair_j=pair_j, fi=fi,
            )
            # If LAMMPS-compat flags are on, project onto the file-order
            # emit rows (DNA + altloc-B shadows interleaved).
            if include_dna or lammps_compat_altloc:
                density = _project_density_to_lammps_emit(
                    density, coords_full, n_protein=n_residues,
                )
            density_df = _density_to_df(density)
            _emit_state = {
                "kind": "pair",
                "coords": coords,
                "pair_i": pair_i, "pair_j": pair_j,
                "r_ij": r_ij, "rho_i": rho_i, "rho_j": rho_j,
                "e_native": e_native,
                "dm": dm, "ds": ds, "fi": fi,
                "density": density,
            }
        else:
            # Audit findings #20 + #21 (2026-05-21): schema-preserving
            # empty for both natural zero-pair and the zero-contact
            # RuntimeError fallback.
            pair_df = _empty_pair_df()
            density = compute_residue_density(
                coords=coords, pair_i=pair_i, pair_j=pair_j, fi=torch.zeros(0),
            )
            if include_dna or lammps_compat_altloc:
                density = _project_density_to_lammps_emit(
                    density, coords_full, n_protein=n_residues,
                )
            density_df = _density_to_df(density)
            # Make sure n_pairs reflects the no-output case for metadata
            # consumers that read it before the post-filter recount.
            n_pairs = 0
            # Empty emit state: when output_dir is set we still want to
            # write header-only marker files (audit finding #5).
            _emit_state = {
                "kind": "pair_empty",
                "coords": coords,
                "density": density,
            }

    elif mode == "mutational":
        stats = mutational_decoy_stats(
            coords, rho=rho,
            n_decoys=n_decoys,
            pair_min_seq_sep=pair_min_seq_sep,
            seed=seed,
            device=dev,
            dtype=dtype,
        )
        pair_i = stats["pair_i"]
        pair_j = stats["pair_j"]
        r_ij = stats["r_ij"]
        rho_i = stats["rho_i"]
        rho_j = stats["rho_j"]
        e_native = stats["E_native"]
        dm = stats["decoy_mean"]
        ds = stats["decoy_std"]
        n_pairs = int(pair_i.numel())

        # DH diagnostic (always populated when electrostatics_k is set).
        if electrostatics_k is not None and n_pairs > 0:
            dh_vec = _add_dh_to_e_native(
                torch.zeros_like(e_native), pair_i, pair_j,
                coords["residue_types"], r_ij, coords["chain_ids"],
                electrostatics_k=float(electrostatics_k),
                device=dev, dtype=dtype,
            )
            v_dh_total = float(dh_vec.sum().item())
            if include_dh_in_e_native:
                e_native = e_native + dh_vec
        elif electrostatics_k is not None and n_pairs == 0:
            v_dh_total = 0.0

        if n_pairs > 0:
            fi = compute_frustration_index(
                e_native=e_native, decoy_mean=dm, decoy_std=ds,
            )
            welltype_arr = welltype_from_contact(r_ij, rho_i, rho_j)
            cls_arr = classify_frustration(fi)

            pair_df = _build_pair_dataframe(
                coords=coords,
                pair_i=pair_i, pair_j=pair_j,
                r_ij=r_ij, rho_i=rho_i, rho_j=rho_j,
                e_native=e_native,
                dm_per_pair=dm, ds_per_pair=ds,
                fi=fi,
                welltype_arr=welltype_arr, cls_arr=cls_arr,
                precision=precision,
            )
            density = compute_residue_density(
                coords=coords, pair_i=pair_i, pair_j=pair_j, fi=fi,
            )
            if include_dna or lammps_compat_altloc:
                density = _project_density_to_lammps_emit(
                    density, coords_full, n_protein=n_residues,
                )
            density_df = _density_to_df(density)
            _emit_state = {
                "kind": "pair",
                "coords": coords,
                "pair_i": pair_i, "pair_j": pair_j,
                "r_ij": r_ij, "rho_i": rho_i, "rho_j": rho_j,
                "e_native": e_native,
                "dm": dm, "ds": ds, "fi": fi,
                "density": density,
            }
        else:
            # Audit finding #21 (2026-05-21): use the same schema-preserving
            # empty DataFrame as configurational mode (previously a bare
            # pd.DataFrame() with shape (0,0)).
            pair_df = _empty_pair_df()
            density = compute_residue_density(
                coords=coords, pair_i=pair_i, pair_j=pair_j, fi=torch.zeros(0),
            )
            if include_dna or lammps_compat_altloc:
                density = _project_density_to_lammps_emit(
                    density, coords_full, n_protein=n_residues,
                )
            density_df = _density_to_df(density)
            _emit_state = {
                "kind": "pair_empty",
                "coords": coords,
                "density": density,
            }

    else:  # singleresidue
        if electrostatics_k is not None and include_dh_in_e_native:
            # Singleresidue doesn't easily expose per-residue cross-term DH;
            # frustrapy's singleresidue mode also omits DH. Flag this clearly
            # only when the opt-in was explicitly requested (a bare
            # `electrostatics_k=4.15` is metadata-only and not a misuse).
            import warnings as _w
            _w.warn(
                "include_dh_in_e_native=True is ignored for "
                "mode='singleresidue' — AWSEM's singleresidue native energy "
                "does not include a DH contribution (the per-residue scalar "
                "is integrated over all contacts, no clean way to add "
                "pairwise DH). Pass mode='configurational' or 'mutational' "
                "to actually use the DH opt-in.",
                RuntimeWarning, stacklevel=2,
            )
        stats = singleresidue_decoy_stats(
            coords, rho=rho,
            n_decoys=n_decoys,
            pair_min_seq_sep=pair_min_seq_sep,
            seed=seed,
            device=dev,
            dtype=dtype,
        )
        e_native = stats["E_native"]
        dm = stats["decoy_mean"]
        ds = stats["decoy_std"]
        fi = stats["FI"]
        sr_df = _build_singleresidue_dataframe(
            coords=coords, rho=rho,
            e_native=e_native, decoy_mean=dm, decoy_std=ds, fi=fi,
            precision=precision,
        )
        _emit_state = {
            "kind": "single",
            "coords": coords,
            "rho": rho,
            "e_native": e_native,
            "dm": dm, "ds": ds, "fi": fi,
        }

    # --- post-filter on residue subset ------------------------------------
    # Capture the pre-filter counts so metadata can expose BOTH the filtered
    # n_pairs/n_residues (audit #19) and the original counts. The dump
    # files written below reflect the filtered state (audit #7).
    n_pairs_unfiltered = n_pairs
    n_residues_unfiltered = n_residues

    if residues is not None:
        # Remember pre-filter sizes to detect zero-hit specs.
        _pre_pair = 0 if pair_df is None else len(pair_df)
        _pre_sr = 0 if sr_df is None else len(sr_df)
        _pre_dens = 0 if density_df is None else len(density_df)

        # Pair-mode filter: build a boolean mask we can reuse for both
        # the DataFrame and the emit-state tensors.
        pair_mask = None
        if pair_df is not None and len(pair_df) > 0:
            mask_keep = []
            for _, row in pair_df.iterrows():
                want_i = row["ChainRes1"] in residues and \
                    int(row["Res1"]) in residues[row["ChainRes1"]]
                want_j = row["ChainRes2"] in residues and \
                    int(row["Res2"]) in residues[row["ChainRes2"]]
                mask_keep.append(want_i or want_j)
            pair_df = pair_df[mask_keep].reset_index(drop=True)
            pair_mask = mask_keep
        sr_mask = None
        if sr_df is not None and len(sr_df) > 0:
            mask_keep = []
            for _, row in sr_df.iterrows():
                want = row["ChainRes"] in residues and \
                    int(row["Res"]) in residues[row["ChainRes"]]
                mask_keep.append(want)
            sr_df = sr_df[mask_keep].reset_index(drop=True)
            sr_mask = mask_keep
        density_mask = None
        if density_df is not None and len(density_df) > 0:
            mask_keep = []
            for _, row in density_df.iterrows():
                want = row["ChainRes"] in residues and \
                    int(row["Res"]) in residues[row["ChainRes"]]
                mask_keep.append(want)
            density_df = density_df[mask_keep].reset_index(drop=True)
            density_mask = mask_keep

        # Audit findings #7 (2026-05-21): apply the same filter to the
        # emit-state tensors so any output files written below reflect
        # the filtered subset, not the unfiltered computation.
        if _emit_state is not None and _emit_state.get("kind") == "pair":
            if pair_mask is not None:
                mask_t = torch.as_tensor(
                    pair_mask, dtype=torch.bool, device=dev,
                )
                _emit_state = {
                    **_emit_state,
                    "pair_i": _emit_state["pair_i"][mask_t],
                    "pair_j": _emit_state["pair_j"][mask_t],
                    "r_ij": _emit_state["r_ij"][mask_t],
                    "rho_i": _emit_state["rho_i"][mask_t],
                    "rho_j": _emit_state["rho_j"][mask_t],
                    "e_native": _emit_state["e_native"][mask_t],
                    "fi": _emit_state["fi"][mask_t],
                }
                dm_state = _emit_state["dm"]
                ds_state = _emit_state["ds"]
                # For configurational mode dm/ds are 0-D scalars (broadcast
                # to per-pair); for mutational they're per-pair tensors.
                if dm_state.ndim > 0 and dm_state.numel() == len(pair_mask):
                    _emit_state["dm"] = dm_state[mask_t]
                    _emit_state["ds"] = ds_state[mask_t]
            # density: filter to requested residues using density_mask.
            density = _emit_state["density"]
            if density_mask is not None:
                density_mask_t = torch.as_tensor(
                    density_mask, dtype=torch.bool, device=density["Total"].device,
                )
                _emit_state["density"] = {
                    "residue_numbers": density["residue_numbers"][
                        density_mask_t.to(density["residue_numbers"].device)
                    ],
                    "chain_ids": [
                        c for c, k in zip(density["chain_ids"], density_mask) if k
                    ],
                    "Total": density["Total"][density_mask_t],
                    "nHighlyFrst": density["nHighlyFrst"][density_mask_t],
                    "nNeutrallyFrst": density["nNeutrallyFrst"][density_mask_t],
                    "nMinimallyFrst": density["nMinimallyFrst"][density_mask_t],
                    "relHighlyFrustrated": density["relHighlyFrustrated"][density_mask_t],
                    "relNeutralFrustrated": density["relNeutralFrustrated"][density_mask_t],
                    "relMinimallyFrustrated": density["relMinimallyFrustrated"][density_mask_t],
                }
        elif _emit_state is not None and _emit_state.get("kind") == "single":
            if sr_mask is not None:
                mask_t = torch.as_tensor(
                    sr_mask, dtype=torch.bool, device=dev,
                )
                # Singleresidue emit writes per-residue rows from coords +
                # rho + e_native arrays of length n_residues. We need to
                # subset coords to the kept rows too.
                _emit_state = _filter_single_emit_state(_emit_state, mask_t)

        # Update n_pairs / n_residues to reflect the filtered subset.
        n_pairs = 0 if pair_df is None else len(pair_df)
        if sr_df is not None:
            n_residues = len(sr_df)
        elif density_df is not None:
            n_residues = len(density_df)

        # Audit finding #26 (2026-05-21): warn for ANY requested resnum
        # missing from the result tables. Previously we only warned when
        # EVERY table was zeroed, so residues={"A": [25, 9999]} on 5AON
        # silently returned rows for 25 with no flag that 9999 was a typo.
        # Collect the resnums that DO appear in any returned table for the
        # requested chains, then compare against the request.
        present: dict[str, set] = {ch: set() for ch in residues}
        if pair_df is not None and len(pair_df) > 0:
            for _, row in pair_df.iterrows():
                if row["ChainRes1"] in present:
                    present[row["ChainRes1"]].add(int(row["Res1"]))
                if row["ChainRes2"] in present:
                    present[row["ChainRes2"]].add(int(row["Res2"]))
        if sr_df is not None and len(sr_df) > 0:
            for _, row in sr_df.iterrows():
                if row["ChainRes"] in present:
                    present[row["ChainRes"]].add(int(row["Res"]))
        if density_df is not None and len(density_df) > 0:
            for _, row in density_df.iterrows():
                if row["ChainRes"] in present:
                    present[row["ChainRes"]].add(int(row["Res"]))
        missing_resnums: dict[str, list] = {}
        for ch, want in residues.items():
            miss = [r for r in want if r not in present.get(ch, set())]
            if miss:
                missing_resnums[ch] = miss
        if missing_resnums:
            import warnings as _w
            _w.warn(
                f"residues= requested resnums that were not present in "
                f"the result tables (typo or out-of-range?): "
                f"{missing_resnums!r}. Note: pair-mode rows require a "
                f"contact to a *kept* residue, so requested-but-isolated "
                f"residues can also appear here.",
                UserWarning,
                stacklevel=2,
            )

        # If the post-filter zeroed every table even though pre-filter had
        # rows, the caller almost certainly passed resnums that don't exist
        # in any chain. Keep the historical "fully empty" warning too — it
        # carries the chain-filter reminder which is otherwise lost.
        _post_pair = 0 if pair_df is None else len(pair_df)
        _post_sr = 0 if sr_df is None else len(sr_df)
        _post_dens = 0 if density_df is None else len(density_df)
        had_input = (_pre_pair + _pre_sr + _pre_dens) > 0
        all_zero = (_post_pair + _post_sr + _post_dens) == 0
        if had_input and all_zero:
            import warnings as _w
            _requested = sum(len(v) for v in residues.values())
            _w.warn(
                f"residues filter retained 0 of {_requested} requested resnums; "
                f"check that the resnums {residues!r} exist in the structure. "
                f"Note: chain filter is applied first, so the requested chains "
                f"must also be present.",
                UserWarning,
                stacklevel=2,
            )

    # --- write output files (post-filter) --------------------------------
    # Audit findings #5 + #7 (2026-05-21):
    # * #5: write header-only marker files for pair-mode runs that
    #   produced zero native pairs, so batch users see a deliberate empty
    #   result instead of a missing file.
    # * #7: emit files AFTER the residues= filter is applied so the
    #   written .dat reflects the same subset as the returned DataFrames.
    if output_dir is not None and _emit_state is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if _emit_state["kind"] == "pair":
            # n_pairs may now be zero (filter wiped them out) — fall
            # through to the header-only path so the file still exists.
            if int(_emit_state["pair_i"].numel()) > 0:
                _emit_pair_files(
                    out_dir=out_dir, basename=pdb_path.stem, mode=mode,
                    coords=_emit_state["coords"],
                    pair_i=_emit_state["pair_i"], pair_j=_emit_state["pair_j"],
                    r_ij=_emit_state["r_ij"],
                    rho_i=_emit_state["rho_i"], rho_j=_emit_state["rho_j"],
                    e_native=_emit_state["e_native"],
                    dm=_emit_state["dm"], ds=_emit_state["ds"],
                    fi=_emit_state["fi"],
                    density=_emit_state["density"],
                    precision=precision,
                )
            else:
                _emit_empty_pair_files(
                    out_dir=out_dir, basename=pdb_path.stem, mode=mode,
                )
        elif _emit_state["kind"] == "pair_empty":
            _emit_empty_pair_files(
                out_dir=out_dir, basename=pdb_path.stem, mode=mode,
            )
        elif _emit_state["kind"] == "single":
            emit_singleresidue_dat(
                coords=_emit_state["coords"],
                rho=_emit_state["rho"],
                e_native=_emit_state["e_native"],
                decoy_mean=_emit_state["dm"], decoy_std=_emit_state["ds"],
                fi=_emit_state["fi"],
                output_path=out_dir / f"{pdb_path.stem}_singleresidue.dat",
                raw=False,
                precision=precision,
            )

    # QA-3 M-1 fix (2026-05-21): drain any in-flight CUDA work before
    # stopping the timer. Without this, wall_clock_ms on a GPU run is
    # CPU launch time, not actual completion time.
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    wall_ms = (time.perf_counter() - start) * 1000.0

    metadata: dict[str, Any] = {
        "mode": mode,
        "chain": chain,
        "residues": residues,
        "electrostatics_k": electrostatics_k,
        "include_dh_in_e_native": include_dh_in_e_native,
        "seq_dist": seq_dist,
        "pair_min_seq_sep": pair_min_seq_sep,
        "n_decoys": n_decoys,
        "device": str(dev),
        "dtype": str(dtype),
        "seed": seed,
        "wall_clock_ms": wall_ms,
        # Audit finding #19 (2026-05-21): expose filtered counts here and
        # the original (pre-residues-filter) counts as *_unfiltered. This
        # keeps batch users sizing arrays from `n_pairs`/`n_residues`
        # consistent with the returned DataFrames.
        "n_residues": n_residues,
        "n_pairs": n_pairs,
        "n_residues_unfiltered": n_residues_unfiltered,
        "n_pairs_unfiltered": n_pairs_unfiltered,
        "pdb_file": str(pdb_path),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "keep_incomplete_backbone": keep_incomplete_backbone,
        "include_dna": include_dna,
        "lammps_compat_altloc": lammps_compat_altloc,
    }
    if dm_scalar is not None:
        metadata["decoy_mean"] = dm_scalar
        metadata["decoy_std"] = ds_scalar
    # Audit finding #4 (2026-05-21): expose the native DH energy total
    # whenever electrostatics_k was supplied (regardless of whether DH
    # was added to E_native). The promise was in the docstring; we now
    # honour it.
    if v_dh_total is not None:
        metadata["v_dh"] = v_dh_total

    return FrustrationResult(
        pair_records=pair_df,
        singleresidue_records=sr_df,
        density_records=density_df,
        metadata=metadata,
    )


def _density_to_df(density: dict[str, Any]):
    """Build the density DataFrame from the dict returned by
    :func:`compute_residue_density`."""
    import pandas as pd
    return pd.DataFrame({
        "Res": density["residue_numbers"].detach().cpu().tolist(),
        "ChainRes": density["chain_ids"],
        "Total": density["Total"].detach().cpu().tolist(),
        "nHighlyFrst": density["nHighlyFrst"].detach().cpu().tolist(),
        "nNeutrallyFrst": density["nNeutrallyFrst"].detach().cpu().tolist(),
        "nMinimallyFrst": density["nMinimallyFrst"].detach().cpu().tolist(),
        "relHighlyFrustrated": density["relHighlyFrustrated"].detach().cpu().tolist(),
        "relNeutralFrustrated": density["relNeutralFrustrated"].detach().cpu().tolist(),
        "relMinimallyFrustrated": density["relMinimallyFrustrated"].detach().cpu().tolist(),
    })


def _project_density_to_lammps_emit(
    density: dict[str, Any],
    coords_full: dict[str, Any],
    n_protein: int,
) -> dict[str, Any]:
    """Re-project protein-only density onto the LAMMPS-compatible emit row
    list (which may include DNA / altloc-B shadow rows in PDB-file order).

    Returns a dict with the same keys as :func:`compute_residue_density`
    but with len = ``min(n_protein, len(emit_rows))`` and values pulled
    from ``density`` at the math_protein_idx specified by each emit row.

    The emit row list is taken from ``coords_full["lammps_emit_rows"]``
    (built by :func:`src.parser._build_lammps_emit_rows`).
    """
    emit_rows = coords_full.get("lammps_emit_rows")
    if not emit_rows:
        return density
    # Apply the zip-truncation: stop at N_protein rows. This reproduces
    # the frustratometeR zip-cut behaviour on PDBs where the equivalences
    # stream is longer than the protein ca_xyz list.
    n_out = min(n_protein, len(emit_rows))
    out_rows = emit_rows[:n_out]
    # Gather density fields at the math_protein_idx of each emit row.
    import torch as _torch
    math_idx_t = _torch.as_tensor(
        [int(r[2]) for r in out_rows], dtype=_torch.int64,
    )
    # Clamp defensive: math_idx within [0, n_protein - 1].
    math_idx_t = math_idx_t.clamp(0, n_protein - 1)
    # Density tensors are on the same device as the protein math.
    dev = density["Total"].device

    def _gather(t):
        return t[math_idx_t.to(dev)]

    return {
        # The labels (resnum + chain) come from the emit rows, NOT from
        # the math view's residue_numbers / chain_ids.
        "residue_numbers": _torch.as_tensor(
            [int(r[1]) for r in out_rows], dtype=_torch.int64,
        ),
        "chain_ids": [r[0] for r in out_rows],
        "Total": _gather(density["Total"]),
        "nHighlyFrst": _gather(density["nHighlyFrst"]),
        "nNeutrallyFrst": _gather(density["nNeutrallyFrst"]),
        "nMinimallyFrst": _gather(density["nMinimallyFrst"]),
        "relHighlyFrustrated": _gather(density["relHighlyFrustrated"]),
        "relNeutralFrustrated": _gather(density["relNeutralFrustrated"]),
        "relMinimallyFrustrated": _gather(density["relMinimallyFrustrated"]),
    }


def _emit_empty_pair_files(
    *,
    out_dir: Path,
    basename: str,
    mode: str,
) -> None:
    """Write header-only marker files for pair-mode runs with zero pairs.

    Audit finding #5 (2026-05-21): batch users expect a deliberate empty
    file (header only, no data rows) rather than a missing file when the
    pipeline completes successfully but produces no pair / density rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{basename}_tertiary_frustration.dat").write_text(
        "# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j "
        "native_energy <decoy_energies> std(decoy_energies) f_ij\n"
        "# timestep: 0\n"
        "# (no native pairs after computation / filter — header-only marker)\n"
    )
    (out_dir / f"{basename}_{mode}.dat").write_text(
        "Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2 "
        "NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState\n"
    )
    (out_dir / f"{basename}_5adens.dat").write_text(
        "Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst "
        "relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated\n"
    )


def _filter_single_emit_state(
    state: dict[str, Any], mask_t: torch.Tensor,
) -> dict[str, Any]:
    """Subset the singleresidue emit-state tensors to a boolean residue mask.

    Audit finding #7 (2026-05-21): when residues= filters singleresidue
    output, the written dump must match the returned DataFrame. We need
    to subset coords + rho + per-residue energy arrays consistently.
    """
    coords = state["coords"]
    coords_idx = mask_t.to(coords["ca_coords"].device)
    coords_sub = {
        "ca_coords": coords["ca_coords"][coords_idx],
        "n_coords": coords["n_coords"][coords_idx],
        "c_coords": coords["c_coords"][coords_idx],
        "o_coords": coords["o_coords"][coords_idx],
        "cb_coords": coords["cb_coords"][coords_idx],
        "residue_types": coords["residue_types"][coords_idx],
        "chain_ids": [c for c, k in zip(coords["chain_ids"], mask_t.tolist()) if k],
        "residue_numbers": coords["residue_numbers"][coords_idx],
        "insertion_codes": [
            c for c, k in zip(coords["insertion_codes"], mask_t.tolist()) if k
        ],
        "is_gly": coords["is_gly"][coords_idx],
    }
    # Optional fields the parser may or may not have populated.
    for opt_key in ("is_dna", "is_altloc_b_shadow"):
        if opt_key in coords:
            coords_sub[opt_key] = coords[opt_key][coords_idx]
    rho_dev = state["rho"].device
    rho_sub = state["rho"][mask_t.to(rho_dev)]
    en_dev = state["e_native"].device
    return {
        "kind": "single",
        "coords": coords_sub,
        "rho": rho_sub,
        "e_native": state["e_native"][mask_t.to(en_dev)],
        "dm": state["dm"][mask_t.to(state["dm"].device)],
        "ds": state["ds"][mask_t.to(state["ds"].device)],
        "fi": state["fi"][mask_t.to(state["fi"].device)],
    }


def _emit_pair_files(
    *,
    out_dir: Path,
    basename: str,
    mode: str,
    coords: dict[str, torch.Tensor],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    r_ij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    e_native: torch.Tensor,
    dm: torch.Tensor,
    ds: torch.Tensor,
    fi: torch.Tensor,
    density: dict[str, Any],
    precision: int,
) -> None:
    """Write the three output files for configurational/mutational mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_tertiary_frustration_dat(
        mode=mode,
        coords=coords,
        pair_i=pair_i, pair_j=pair_j,
        r_ij=r_ij, rho_i=rho_i, rho_j=rho_j,
        e_native=e_native, decoy_mean=dm, decoy_std=ds,
        output_path=out_dir / f"{basename}_tertiary_frustration.dat",
        fi=fi,
        precision=precision,
    )
    emit_postprocessed_pair_dat(
        coords=coords,
        pair_i=pair_i, pair_j=pair_j,
        r_ij=r_ij, rho_i=rho_i, rho_j=rho_j,
        e_native=e_native, decoy_mean=dm, decoy_std=ds,
        output_path=out_dir / f"{basename}_{mode}.dat",
        fi=fi,
        precision=precision,
    )
    emit_5adens_dat(
        density=density,
        output_path=out_dir / f"{basename}_5adens.dat",
    )


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
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 0,
    precision: int = 3,
    graphics: bool = False,
    debug: bool = False,
    pbar: bool = False,
    visualization: bool = False,
    pdb_id: str | None = None,
    is_mutation_calculation: bool | None = None,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
    overwrite: bool = False,
    n_cpus: int | None = None,
    **kwargs: Any,
) -> FrustrationResult:
    """frustrapy ``calculate_frustration`` drop-in adapter (Phase 5 P3).

    Translates the frustrapy kwarg surface onto :func:`compute_frustration`.
    Designed so that user code written for frustrapy can swap modules::

        # before
        import frustrapy
        r = frustrapy.calculate_frustration("X.pdb", mode="mutational",
                                            results_dir="out/", graphics=True)

        # after
        from frustration_gpu import calculate_frustration
        r = calculate_frustration("X.pdb", mode="mutational",
                                  results_dir="out/", graphics=True)

    Kwarg translations
    ------------------
    * ``results_dir`` (frustrapy) → ``output_dir`` (ours). If both passed,
      ``output_dir`` wins.
    * ``chain`` accepts ``str`` or ``list[str]`` — if list, parsed via
      :func:`parse_pdb`'s native ``chains=`` filter (supports >1 chain).
      (Underlying ``compute_frustration`` only accepted ``str``; this
      adapter widens the type.)
    * ``is_mutation_calculation=True`` → ``mode="mutational"`` (frustrapy
      historical synonym).
    * ``graphics`` / ``visualization`` / ``debug`` / ``pbar`` are accepted
      but ignored (with a single ``UserWarning`` on the first call that
      sets ``graphics=True`` or ``visualization=True``). Phase 6 will wire
      these to real outputs.
    * ``pdb_id`` is accepted for API parity but ignored — we don't auto-
      download from RCSB; the user must provide a local ``pdb_file``.
    * ``overwrite`` is accepted for frustrapy parity. The PyTorch port
      always overwrites existing output files; passing ``overwrite=False``
      emits a one-time ``UserWarning`` so the caller is aware.
    * ``n_cpus`` is accepted for frustrapy parity but ignored — we run a
      single-process GPU/CPU pipeline. Use ``device="cuda"`` or
      ``device="cpu"`` to control execution. Passing a non-``None`` value
      emits a one-time ``UserWarning``.

    Notes
    -----
    Any unknown kwargs raise ``TypeError`` (via ``**kwargs`` capture below).
    """
    # Handle is_mutation_calculation legacy synonym
    if is_mutation_calculation is True:
        mode = "mutational"

    # Warn for unsupported visual flags (don't error — frustrapy users
    # often pass these by reflex). Audit finding #35 (2026-05-21): the
    # docstring promises a SINGLE UserWarning per kwarg over the module's
    # lifetime; we dedup via the module-level _CF_WARN_ONCE set.
    if (graphics or visualization) and "graphics" not in _CF_WARN_ONCE:
        import warnings as _w
        _w.warn(
            "graphical output (graphics/visualization) is not supported by "
            "the PyTorch port — flag ignored. Phase 6 will add VMD/PyMOL "
            "script emission. (This warning fires only once per process.)",
            UserWarning,
            stacklevel=2,
        )
        _CF_WARN_ONCE.add("graphics")
    # debug + pbar are silently consumed.
    _ = debug
    _ = pbar

    # overwrite: accepted for frustrapy parity. The PyTorch port always
    # overwrites existing output files. Audit finding #35 (2026-05-21):
    # the docstring says overwrite=False should warn when an existing
    # file would be overwritten. Surface that here (once per process).
    out_dir_for_check = output_dir if output_dir is not None else results_dir
    if overwrite is False and out_dir_for_check is not None:
        if pdb_file is not None:
            stem = Path(pdb_file).stem
            existing = [
                p for p in (
                    Path(out_dir_for_check) / f"{stem}_tertiary_frustration.dat",
                    Path(out_dir_for_check) / f"{stem}_configurational.dat",
                    Path(out_dir_for_check) / f"{stem}_mutational.dat",
                    Path(out_dir_for_check) / f"{stem}_singleresidue.dat",
                    Path(out_dir_for_check) / f"{stem}_5adens.dat",
                ) if p.is_file()
            ]
            if existing and "overwrite_false" not in _CF_WARN_ONCE:
                import warnings as _w
                _w.warn(
                    f"overwrite=False is not enforced by the PyTorch port — "
                    f"existing output file(s) {[p.name for p in existing]} "
                    f"in {out_dir_for_check!r} WILL be overwritten. "
                    f"(This warning fires only once per process.)",
                    UserWarning,
                    stacklevel=2,
                )
                _CF_WARN_ONCE.add("overwrite_false")
    _ = overwrite

    # n_cpus: not applicable to the GPU/single-process port. Warn once when
    # explicitly supplied so the user knows it had no effect. Audit
    # finding #35 (2026-05-21): dedup via the module-level set so the
    # warning fires once over the process, matching the docstring.
    if n_cpus is not None and "n_cpus" not in _CF_WARN_ONCE:
        import warnings as _w
        _w.warn(
            "n_cpus is not supported by the PyTorch port (single-process "
            "GPU/CPU pipeline); kwarg ignored. Use device='cuda' or "
            "device='cpu' to control execution. "
            "(This warning fires only once per process.)",
            UserWarning,
            stacklevel=2,
        )
        _CF_WARN_ONCE.add("n_cpus")
    _ = n_cpus

    # results_dir → output_dir translation
    out_dir = output_dir if output_dir is not None else results_dir

    # `compute_frustration` now natively accepts str OR list[str] and
    # routes both through the parser-level filter, so we pass `chain`
    # through unchanged. QA-3 H-2 fix (2026-05-21).
    chain_arg: str | list[str] | None = chain

    # The pdb_id arg is accepted for API parity (frustrapy auto-downloads
    # from RCSB when only pdb_id is given). We don't auto-download.
    if pdb_file is None:
        raise TypeError(
            "calculate_frustration: pdb_file is required (PyTorch port does "
            "not auto-download from RCSB by pdb_id). "
            f"Got pdb_id={pdb_id!r}, pdb_file=None."
        )
    _ = pdb_id  # accepted for parity

    # Unknown kwargs → loud error (don't silently swallow typos).
    if kwargs:
        raise TypeError(
            f"calculate_frustration: unknown kwargs {list(kwargs)}. "
            f"Allowed: pdb_file, mode, chain, residues, electrostatics_k, "
            f"include_dh_in_e_native, seq_dist, n_decoys, device, "
            f"results_dir/output_dir, seed, precision, graphics, debug, "
            f"pbar, visualization, pdb_id, is_mutation_calculation, "
            f"keep_incomplete_backbone, include_dna, lammps_compat_altloc, "
            f"overwrite, n_cpus."
        )

    result = compute_frustration(
        pdb_file,
        mode=mode,  # type: ignore[arg-type]
        chain=chain_arg,
        residues=residues,
        electrostatics_k=electrostatics_k,
        include_dh_in_e_native=include_dh_in_e_native,
        seq_dist=seq_dist,
        n_decoys=n_decoys,
        device=device,
        output_dir=out_dir,
        seed=seed,
        precision=precision,
        keep_incomplete_backbone=keep_incomplete_backbone,
        include_dna=include_dna,
        lammps_compat_altloc=lammps_compat_altloc,
    )

    # No post-filter needed — `compute_frustration` now routes list[str]
    # through the parser, so the pipeline only ever sees the selected
    # chains' residues. QA-3 H-2 fix (2026-05-21).
    return result


__all__ = [
    "FrustrationResult",
    "compute_frustration",
    "calculate_frustration",
]
