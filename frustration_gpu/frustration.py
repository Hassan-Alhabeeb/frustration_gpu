"""AWSEM frustration index + classification + LAMMPS-compatible writers (Phase 3c).

Thin wrapper that takes decoy statistics from Phase 3a/3b and:

1. Computes the per-pair (or per-residue) frustration index::

       FI = (decoy_mean - E_native) / decoy_std

2. Classifies it via the Ferreiro (2007) thresholds reproduced in
   ``inst/Scripts/RenumFiles.pl`` of frustratometeR::

       FI <= -1.0   →  highly  frustrated   (label index 0)
       -1.0 < FI < 0.78  →  neutral         (label index 1)
       FI >= 0.78   →  minimally frustrated (label index 2)

3. Assigns a contact ``Welltype`` matching the same Perl post-processor::

       r_ij < 6.5                                    →  'short'
       r_ij >= 6.5 AND rho_i < 2.6 AND rho_j < 2.6   →  'water-mediated'
       r_ij >= 6.5 AND (rho_i >= 2.6 OR rho_j >= 2.6) →  'long'

4. Emits text dump files in the LAMMPS-AWSEM raw format and in
   frustratometeR's post-processed format.

LAMMPS raw printf format (``fix_backbone.cpp:5104``)::

    "%5d %5d %3d %3d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
    " %8.3f %8.3f %8.3f %c %c %8.3f %8.3f %8.3f %8.3f\\n"

Columns (1-indexed):
    1   i (1-indexed)
    2   j
    3   i_chain (1-indexed integer)
    4   j_chain
    5-7 xi yi zi
    8-10 xj yj zj
    11  r_ij
    12  rho_i
    13  rho_j
    14  a_i (one-letter AA code)
    15  a_j
    16  E_native
    17  decoy_mean
    18  decoy_std
    19  FI

Singleresidue raw printf format (``fix_backbone.cpp:5168``)::

    "%5d %5d %8.3f %8.3f %8.3f %8.3f %c %8.3f %8.3f %8.3f %8.3f\\n"

Columns:
    1   i
    2   i_chain
    3-5 xi yi zi
    6   rho_i
    7   a_i
    8   E_native
    9   decoy_mean
    10  decoy_std
    11  FI

Sign convention
---------------
FI > 0  →  minimally frustrated (native better than decoys)
FI < 0  →  highly frustrated    (native worse than decoys)

LOC budget: ~200 lines + this docstring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .parser import ONE_TO_IDX

# Inverse of the gamma-column AA ordering (idx -> one-letter code).
_IDX_TO_ONE: list[str] = [None] * 20
for _aa, _i in ONE_TO_IDX.items():
    _IDX_TO_ONE[_i] = _aa

# Thresholds (Ferreiro 2007 + frustratometeR RenumFiles.pl)
HIGHLY_FRUSTRATED_THRESHOLD: float = -1.0
MINIMALLY_FRUSTRATED_THRESHOLD: float = 0.78
WELLTYPE_R_SHORT_A: float = 6.5
WELLTYPE_RHO_WATER: float = 2.6

# Class label codes
CLASS_HIGHLY: int = 0
CLASS_NEUTRAL: int = 1
CLASS_MINIMALLY: int = 2

# Welltype codes
WELL_SHORT: int = 0
WELL_WATER_MEDIATED: int = 1
WELL_LONG: int = 2

_CLASS_NAMES: dict[int, str] = {
    CLASS_HIGHLY: "highly",
    CLASS_NEUTRAL: "neutral",
    CLASS_MINIMALLY: "minimally",
}

_WELL_NAMES: dict[int, str] = {
    WELL_SHORT: "short",
    WELL_WATER_MEDIATED: "water-mediated",
    WELL_LONG: "long",
}


# --- core math ---------------------------------------------------------------
def compute_frustration_index(
    *,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,
    decoy_std: torch.Tensor,
    eps: float = 0.0,
    degenerate_threshold: float = 1e-12,
) -> torch.Tensor:
    """Compute the per-pair (or per-residue) frustration index.

    ``FI = (decoy_mean - E_native) / decoy_std``

    Parameters
    ----------
    e_native : tensor
        Per-pair (or per-residue) native energy. Any shape; must broadcast
        against ``decoy_mean`` and ``decoy_std``.
    decoy_mean, decoy_std : tensors
        Decoy ensemble statistics. ``decoy_std`` must be non-negative.
    eps : float
        If > 0, clamp ``decoy_std`` to ``max(decoy_std, eps)`` before the
        division. Default 0 (true division — zero ``decoy_std`` returns
        ``+/- inf`` or NaN). Pass a small positive value if the caller
        prefers stable zeros over NaNs on flat-distribution decoys.
    degenerate_threshold : float
        When ``eps`` is 0, any ``decoy_std`` below this threshold is
        treated as "decoy ensemble is degenerate" — FI is forced to 0
        (matching the Spearman-rank tie behaviour) and a single
        ``UserWarning`` is emitted. This catches uniform-AA chains (every
        decoy identical to native, ``decoy_std == 0``) without poisoning
        the dataframe with NaN/inf. Set to 0 to disable the guard.

    Returns
    -------
    Tensor with the broadcast shape of the inputs.

    Notes
    -----
    Sign convention is the LAMMPS-AWSEM / Ferreiro convention::

        FI > 0  →  minimally frustrated (native better than decoys)
        FI < 0  →  highly frustrated    (native worse than decoys)
    """
    if eps > 0:
        safe_std = torch.clamp(decoy_std, min=eps)
        return (decoy_mean - e_native) / safe_std
    if degenerate_threshold > 0 and decoy_std.numel() > 0:
        degenerate_mask = decoy_std < degenerate_threshold
        if bool(degenerate_mask.any()):
            import warnings as _w
            _n_degenerate = int(degenerate_mask.sum().item())
            _w.warn(
                f"{_n_degenerate} of {decoy_std.numel()} positions had "
                f"decoy_std < {degenerate_threshold:.0e} (decoy ensemble "
                f"is degenerate, typically because the chain is too "
                f"uniform in AA composition for frustration analysis); "
                f"FI clamped to 0 at those positions.",
                UserWarning,
                stacklevel=2,
            )
            safe_std = torch.where(
                degenerate_mask,
                torch.ones_like(decoy_std),
                decoy_std,
            )
            fi = (decoy_mean - e_native) / safe_std
            return torch.where(
                degenerate_mask, torch.zeros_like(fi), fi
            )
    return (decoy_mean - e_native) / decoy_std


def classify_frustration(
    fi: torch.Tensor,
    *,
    high_threshold: float = HIGHLY_FRUSTRATED_THRESHOLD,
    minimal_threshold: float = MINIMALLY_FRUSTRATED_THRESHOLD,
) -> torch.Tensor:
    """Ferreiro (2007) three-state classification.

    Boundary convention matches ``frustratometeR/inst/Scripts/RenumFiles.pl``:

    * ``FI <= -1.0``  → highly frustrated   (returns 0)
    * ``-1.0 < FI < 0.78`` → neutral        (returns 1)
    * ``FI >= 0.78`` → minimally frustrated (returns 2)

    Returns
    -------
    Integer tensor (``torch.long``) with the same shape as ``fi``.
    """
    cls = torch.ones_like(fi, dtype=torch.long)
    cls[fi <= high_threshold] = CLASS_HIGHLY
    cls[fi >= minimal_threshold] = CLASS_MINIMALLY
    return cls


def welltype_from_contact(
    rij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    *,
    r_short: float = WELLTYPE_R_SHORT_A,
    rho_water_cutoff: float = WELLTYPE_RHO_WATER,
) -> torch.Tensor:
    """Per-contact ``Welltype`` classification per ``RenumFiles.pl``.

    Returns an integer tensor of the same shape as ``rij``:

    * 0 = ``short``           (r_ij < 6.5)
    * 1 = ``water-mediated``  (r_ij >= 6.5 AND rho_i < 2.6 AND rho_j < 2.6)
    * 2 = ``long``            (r_ij >= 6.5 AND (rho_i >= 2.6 OR rho_j >= 2.6))
    """
    short_mask = rij < r_short
    water_mask = (rho_i < rho_water_cutoff) & (rho_j < rho_water_cutoff)
    well = torch.full_like(rij, WELL_LONG, dtype=torch.long)
    well[(~short_mask) & water_mask] = WELL_WATER_MEDIATED
    well[short_mask] = WELL_SHORT
    return well


# --- helpers for the file writers --------------------------------------------
def _aa_idx_to_letter(aa_idx: torch.Tensor) -> list[str]:
    """Vectorised int → one-letter mapping using the OpenAWSEM gamma column order."""
    return [_IDX_TO_ONE[int(a)] for a in aa_idx.tolist()]


def _chain_letters(chain_ids: list[str], i_indices: torch.Tensor) -> list[str]:
    """Pick chain letters at the supplied 0-indexed positions."""
    out = []
    idx_list = i_indices.tolist()
    for k in idx_list:
        out.append(chain_ids[int(k)])
    return out


def _author_resnum(residue_numbers: torch.Tensor, i_indices: torch.Tensor) -> list[int]:
    """Pick PDB author residue numbers at the supplied 0-indexed positions."""
    return [int(residue_numbers[int(k)].item()) for k in i_indices.tolist()]


def _chain_int_index(chain_ids: list[str], idx: torch.Tensor) -> list[int]:
    """Convert chain letters at given positions to the 1-indexed integer
    LAMMPS uses in the raw dump (order-of-first-appearance, +1)."""
    cid_map: dict[str, int] = {}
    for c in chain_ids:
        if c not in cid_map:
            cid_map[c] = len(cid_map) + 1
    return [cid_map[chain_ids[int(k)]] for k in idx.tolist()]


def _xb_coords(coords: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the LAMMPS dump coordinate per residue.

    LAMMPS-AWSEM (``fix_backbone.cpp:5089-5091, 5155-5156``) writes::

        xi = (se[i_resno] == 'G' ? xca[i] : xcb[i])

    i.e. CB for non-Gly residues, CA for Gly. We mirror that here.

    Non-Gly residues that are missing CB in the PDB (``cb_coords`` row is NaN)
    fall back to CA — LAMMPS would have read these from the data file, which
    OpenAWSEM / PDBFixer guarantees are populated; if the parser saw NaN we
    cannot do better without re-running PDBFixer. The fallback keeps the
    writer from emitting NaN strings, which would break downstream parsers.

    Returns
    -------
    (N, 3) tensor matching ``ca_coords``' device / dtype.
    """
    ca = coords["ca_coords"]
    cb = coords.get("cb_coords")
    is_gly = coords.get("is_gly")
    if cb is None or is_gly is None:
        # Defensive: a caller built ``coords`` by hand and only filled CA.
        # Match the legacy behaviour of always returning CA so tests that
        # construct minimal coord dicts (rare in this repo) still work.
        return ca
    # Per-row fallback: a residue picks CA if it's GLY OR if ANY axis of
    # its CB row is non-finite. Mirrors `_resolve_contact_coords` in
    # `_contact_common.py:135-165`. The earlier per-element mask
    # (`is_gly.unsqueeze(1) | torch.isnan(cb)`) could broadcast to (N, 3)
    # and produce a Frankenstein (CB.x, CA.y, CB.z) row when CB had NaN
    # in only one axis. QA-3 H-1 fix (2026-05-21).
    nan_row = ~torch.isfinite(cb).all(dim=-1)                       # (N,)
    row_mask = (is_gly | nan_row).unsqueeze(-1)                     # (N, 1)
    return torch.where(row_mask, ca, cb)


# --- dump writers -------------------------------------------------------------
def emit_tertiary_frustration_dat(
    *,
    mode: Literal["configurational", "mutational"],
    coords: dict[str, torch.Tensor],
    pair_i: torch.Tensor,                  # 0-indexed
    pair_j: torch.Tensor,                  # 0-indexed
    r_ij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,              # scalar (config) or (N_pair,) (mut)
    decoy_std: torch.Tensor,               # scalar (config) or (N_pair,) (mut)
    output_path: str | Path,
    fi: torch.Tensor | None = None,
    precision: int = 3,
) -> None:
    """Write the LAMMPS-AWSEM raw ``tertiary_frustration.dat`` format.

    Schema matches ``fix_backbone.cpp:5104``::

        i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j
        E_native <decoy_energies> std(decoy_energies) f_ij

    Parameters
    ----------
    mode : {'configurational', 'mutational'}
        Controls how ``decoy_mean`` / ``decoy_std`` are broadcast: scalars
        for configurational (one stat shared across all rows) or
        ``(N_pair,)`` tensors for mutational (per-pair).
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``,
        ``cb_coords``, ``is_gly`` (for the CB-or-CA coord pick per
        ``fix_backbone.cpp:5089-5091``), ``residue_types`` (for AA letters),
        ``chain_ids`` (list[str]) and ``residue_numbers``.
    pair_i, pair_j : (N_pair,) int64 tensors
        0-indexed native-pair positions.
    r_ij, rho_i, rho_j, e_native : (N_pair,) float tensors
        Per-pair native data.
    decoy_mean, decoy_std : tensors
        Configurational: 0-D (scalar) — broadcast to all rows.
        Mutational: (N_pair,) — per-pair stats.
    output_path : str or Path
    fi : (N_pair,) float, optional
        Pre-computed FI. If ``None``, computed from
        ``(decoy_mean - e_native) / decoy_std``.
    precision : int
        Decimal places for floats. Default 3 to match LAMMPS dump.
    """
    p = Path(output_path)
    n_pair = int(pair_i.numel())

    if fi is None:
        if decoy_mean.dim() == 0:
            # configurational: scalar broadcast
            fi = (decoy_mean - e_native) / decoy_std
        else:
            fi = (decoy_mean - e_native) / decoy_std

    # broadcast decoy_mean / decoy_std to per-pair tensors for printing
    if decoy_mean.dim() == 0:
        dm_arr = decoy_mean.expand(n_pair)
        ds_arr = decoy_std.expand(n_pair)
    else:
        dm_arr = decoy_mean
        ds_arr = decoy_std

    aa = coords["residue_types"]
    chain_ids = coords["chain_ids"]
    xb = _xb_coords(coords)  # CB for non-Gly, CA for Gly — matches LAMMPS

    i_chain_int = _chain_int_index(chain_ids, pair_i)
    j_chain_int = _chain_int_index(chain_ids, pair_j)
    aa_i_letters = _aa_idx_to_letter(aa[pair_i])
    aa_j_letters = _aa_idx_to_letter(aa[pair_j])

    pi_list = pair_i.tolist()
    pj_list = pair_j.tolist()
    rij_list = r_ij.tolist()
    rhoi_list = rho_i.tolist()
    rhoj_list = rho_j.tolist()
    enat_list = e_native.tolist()
    dm_list = dm_arr.tolist()
    ds_list = ds_arr.tolist()
    fi_list = fi.tolist()

    f = precision
    width = max(8, f + 5)
    fmt = (
        f"{{:5d}} {{:5d}} {{:3d}} {{:3d}} "
        f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
        f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
        f"{{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
        f"{{}} {{}} {{:{width}.{f}f}} {{:{width}.{f}f}} {{:{width}.{f}f}} "
        f"{{:{width}.{f}f}}"
    )

    lines: list[str] = [
        "# i j i_chain j_chain xi yi zi xj yj zj r_ij rho_i rho_j a_i a_j "
        "native_energy <decoy_energies> std(decoy_energies) f_ij",
        "# timestep: 0",
    ]
    for k in range(n_pair):
        i0 = pi_list[k]
        j0 = pj_list[k]
        xi = xb[i0].tolist()
        xj = xb[j0].tolist()
        lines.append(fmt.format(
            i0 + 1, j0 + 1, i_chain_int[k], j_chain_int[k],
            xi[0], xi[1], xi[2], xj[0], xj[1], xj[2],
            rij_list[k], rhoi_list[k], rhoj_list[k],
            aa_i_letters[k], aa_j_letters[k],
            enat_list[k], dm_list[k], ds_list[k], fi_list[k],
        ))

    p.write_text("\n".join(lines) + "\n")


def emit_singleresidue_dat(
    *,
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,
    decoy_std: torch.Tensor,
    output_path: str | Path,
    fi: torch.Tensor | None = None,
    raw: bool = True,
    precision: int = 3,
) -> None:
    """Write the singleresidue dump (per-residue native + decoy + FI).

    Two output flavours:

    * ``raw=True`` (default) — LAMMPS-AWSEM raw format per
      ``fix_backbone.cpp:5168``:
      ``i i_chain xi yi zi rho_i a_i E_native decoy_mean decoy_std FI``
    * ``raw=False`` — frustratometeR post-processed schema with header
      ``Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex``
      (author residue number + chain letter, no coords).

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``,
        ``cb_coords``, ``is_gly`` (for the CB-or-CA coord pick per
        ``fix_backbone.cpp:5155-5156``), ``residue_types``, ``chain_ids``,
        ``residue_numbers``.
    rho, e_native, decoy_mean, decoy_std : (N,) tensors
    output_path : str or Path
    fi : (N,) tensor, optional
        Pre-computed FI; defaults to ``(decoy_mean - e_native) / decoy_std``.
    raw : bool
        True → LAMMPS raw dump format; False → frustratometeR post-processed.
    precision : int
        Decimal places. Default 3.
    """
    p = Path(output_path)
    n = int(rho.numel())

    if fi is None:
        fi = (decoy_mean - e_native) / decoy_std

    aa = coords["residue_types"]
    chain_ids = coords["chain_ids"]
    resnums = coords["residue_numbers"]
    xb = _xb_coords(coords)  # CB for non-Gly, CA for Gly — matches LAMMPS

    aa_letters = _aa_idx_to_letter(aa)
    all_idx = torch.arange(n, dtype=torch.int64)
    i_chain_int = _chain_int_index(chain_ids, all_idx)

    rho_list = rho.tolist()
    e_list = e_native.tolist()
    dm_list = decoy_mean.tolist()
    ds_list = decoy_std.tolist()
    fi_list = fi.tolist()

    f = precision
    width = max(8, f + 5)
    lines: list[str] = []
    if raw:
        lines.append(
            "# i i_chain xi yi zi rho_i a_i native_energy <decoy_energies> "
            "std(decoy_energies) f_i"
        )
        for k in range(n):
            xi = xb[k].tolist()
            lines.append(
                f"{k+1:5d} {i_chain_int[k]:5d} "
                f"{xi[0]:{width}.{f}f} {xi[1]:{width}.{f}f} {xi[2]:{width}.{f}f} "
                f"{rho_list[k]:{width}.{f}f} {aa_letters[k]} "
                f"{e_list[k]:{width}.{f}f} {dm_list[k]:{width}.{f}f} "
                f"{ds_list[k]:{width}.{f}f} {fi_list[k]:{width}.{f}f}"
            )
    else:
        lines.append("Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex")
        for k in range(n):
            lines.append(
                f"{int(resnums[k].item())} {chain_ids[k]} "
                f"{rho_list[k]:.{f}f} {aa_letters[k]} "
                f"{e_list[k]:.{f}f} {dm_list[k]:.{f}f} "
                f"{ds_list[k]:.{f}f} {fi_list[k]:.{f}f}"
            )

    p.write_text("\n".join(lines) + "\n")


def emit_postprocessed_pair_dat(
    *,
    coords: dict[str, torch.Tensor],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    r_ij: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    e_native: torch.Tensor,
    decoy_mean: torch.Tensor,
    decoy_std: torch.Tensor,
    output_path: str | Path,
    fi: torch.Tensor | None = None,
    welltype: torch.Tensor | None = None,
    frst_state: torch.Tensor | None = None,
    precision: int = 3,
) -> None:
    """Write the frustratometeR-style post-processed contact dat.

    Schema (matches both ``<PDB>_configurational.dat`` and
    ``<PDB>_mutational.dat`` produced by ``RenumFiles.pl``)::

        Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2
        NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState

    * ``Res1``, ``Res2`` are author residue numbers (not 1-indexed positions).
    * ``ChainRes1``, ``ChainRes2`` are chain letters.
    * ``Welltype`` ∈ {short, water-mediated, long} per ``r_ij`` and rho's.
    * ``FrstState`` ∈ {highly, neutral, minimally} per Ferreiro thresholds.
    """
    p = Path(output_path)
    n_pair = int(pair_i.numel())

    if fi is None:
        if decoy_mean.dim() == 0:
            fi = (decoy_mean - e_native) / decoy_std
        else:
            fi = (decoy_mean - e_native) / decoy_std

    if decoy_mean.dim() == 0:
        dm_arr = decoy_mean.expand(n_pair)
        ds_arr = decoy_std.expand(n_pair)
    else:
        dm_arr = decoy_mean
        ds_arr = decoy_std

    if welltype is None:
        welltype = welltype_from_contact(r_ij, rho_i, rho_j)
    if frst_state is None:
        frst_state = classify_frustration(fi)

    aa = coords["residue_types"]
    chain_ids = coords["chain_ids"]
    resnums = coords["residue_numbers"]

    aa_i_letters = _aa_idx_to_letter(aa[pair_i])
    aa_j_letters = _aa_idx_to_letter(aa[pair_j])

    pi_list = pair_i.tolist()
    pj_list = pair_j.tolist()
    rhoi_list = rho_i.tolist()
    rhoj_list = rho_j.tolist()
    enat_list = e_native.tolist()
    dm_list = dm_arr.tolist()
    ds_list = ds_arr.tolist()
    fi_list = fi.tolist()
    well_list = welltype.tolist()
    fs_list = frst_state.tolist()

    f = precision
    lines: list[str] = [
        "Res1 Res2 ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2 "
        "NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype FrstState"
    ]
    for k in range(n_pair):
        i0 = pi_list[k]
        j0 = pj_list[k]
        lines.append(
            f"{int(resnums[i0].item())} {int(resnums[j0].item())} "
            f"{chain_ids[i0]} {chain_ids[j0]} "
            f"{rhoi_list[k]:.{f}f} {rhoj_list[k]:.{f}f} "
            f"{aa_i_letters[k]} {aa_j_letters[k]} "
            f"{enat_list[k]:.{f}f} {dm_list[k]:.{f}f} "
            f"{ds_list[k]:.{f}f} {fi_list[k]:.{f}f} "
            f"{_WELL_NAMES[int(well_list[k])]} {_CLASS_NAMES[int(fs_list[k])]}"
        )

    p.write_text("\n".join(lines) + "\n")


__all__ = [
    "HIGHLY_FRUSTRATED_THRESHOLD",
    "MINIMALLY_FRUSTRATED_THRESHOLD",
    "WELLTYPE_R_SHORT_A",
    "WELLTYPE_RHO_WATER",
    "CLASS_HIGHLY",
    "CLASS_NEUTRAL",
    "CLASS_MINIMALLY",
    "WELL_SHORT",
    "WELL_WATER_MEDIATED",
    "WELL_LONG",
    "compute_frustration_index",
    "classify_frustration",
    "welltype_from_contact",
    "emit_tertiary_frustration_dat",
    "emit_singleresidue_dat",
    "emit_postprocessed_pair_dat",
]
