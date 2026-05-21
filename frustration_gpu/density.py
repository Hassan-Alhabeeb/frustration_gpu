"""Per-residue density aggregation (Phase 4 — ``<PDB>_5adens.dat`` schema).

Reproduces frustratometeR's ``XAdens`` (``R/functions.R:50-97``) in PyTorch:

    For each residue i, count the number of native-pair midpoints (between
    columns 5-7 and 8-10 of ``tertiary_frustration.dat``, i.e. the CB-or-CA
    coords of i and j) whose distance to CA[i] is < ``ratio`` (default 5 Å).
    Within that sphere, count how many midpoints carry an ``FI <= -1.0``
    (highly), ``-1.0 < FI < 0.78`` (neutral), ``FI >= 0.78`` (minimally).

Output schema (matches ``<PDB>_5adens.dat`` exactly)::

    Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst \
        relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated

* ``Res`` is the author residue number (from the PDB file, NOT 1-indexed).
* ``ChainRes`` is the chain letter.
* ``Total`` is the in-sphere count of native-pair midpoints.
* ``rel*`` ratios are zero when ``Total == 0`` (matches the R source's
  ``if(total_density > 0)`` guard).

Hand-checked algorithm
----------------------
For 5AON residue 23 (chain A) the reference dump reports
``Total=7 nHighlyFrst=0 nNeutrallyFrst=6 nMinimallyFrst=1``. The post-
processed ``5AON_configurational.dat`` file shows only 5 pairs starting
with res 23 — the missing 2 come from pairs that *don't* start with 23
but whose midpoint sits within 5 Å of CA[23]. This confirms the
midpoint-sphere semantics (NOT a pair-list filter).

vps midpoint convention
~~~~~~~~~~~~~~~~~~~~~~~
The R source builds ``vps`` from columns ``(1, 2, 5, 6, 7, 8, 9, 10, 19)``
of ``tertiary_frustration.dat``::

    vps_x = (xj + xi) / 2     (cols 8 and 5)
    vps_y = (yj + yi) / 2     (cols 9 and 6)
    vps_z = (zj + zi) / 2     (cols 10 and 7)
    vps_fi = col 19           (f_ij)

Columns 5-7 and 8-10 of the LAMMPS dump are the CB-of-non-Gly / CA-of-Gly
coords (per ``fix_backbone.cpp:5089-5091``). We mirror that pick via
:func:`src.frustration._xb_coords`, then take the midpoint per native pair.

CA-of-i used for the sphere center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The R source's ``CA_xyz`` always pulls the CA (line 53-61 of functions.R)
— even for Gly. We honour that: the centre is ``coords["ca_coords"][i]``,
not the effective-CB.

LOC budget: ~150-250 lines + this docstring.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .frustration import (
    HIGHLY_FRUSTRATED_THRESHOLD,
    MINIMALLY_FRUSTRATED_THRESHOLD,
    _xb_coords,
)

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_DENSITY_RATIO_A: float = 5.0
"""Sphere radius in Å — matches frustratometeR's ``XAdens(Pdb, ratio = 5)``."""


def compute_residue_density(
    *,
    coords: dict[str, torch.Tensor],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    fi: torch.Tensor,
    ratio: float = DEFAULT_DENSITY_RATIO_A,
    classification_thresholds: tuple = (
        HIGHLY_FRUSTRATED_THRESHOLD,
        MINIMALLY_FRUSTRATED_THRESHOLD,
    ),
) -> dict[str, torch.Tensor]:
    """Aggregate native-pair FI into per-residue density counts.

    For each residue ``i``, count how many native-pair midpoints lie
    within ``ratio`` of ``CA[i]``. Within that count, split by the
    Ferreiro three-state classification.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain
        ``ca_coords``, ``cb_coords``, ``is_gly``, ``residue_numbers``,
        ``chain_ids``.
    pair_i, pair_j : (N_pair,) int64
        0-indexed native-pair positions (same convention used by the
        Phase 3 writers).
    fi : (N_pair,) float
        Per-pair frustration index.
    ratio : float
        Sphere radius in Å. Default 5.0 (matches XAdens default).
    classification_thresholds : (high, minimal) floats
        Frustration-index thresholds. Defaults ``(-1.0, 0.78)`` per
        Ferreiro 2007 / frustratometeR. ``FI <= high`` → highly,
        ``high < FI < minimal`` → neutral, ``FI >= minimal`` → minimally.
        These bounds *exactly* match the R script's
        ``vps[, 6] <= (-1)`` / ``(> (-1) & < (0.78))`` / ``>= 0.78``
        comparisons — note the asymmetric inclusive/exclusive convention.

    Returns
    -------
    dict with these keys (all (N,) tensors except ``residue_numbers`` /
    ``chain_ids`` for direct DataFrame construction):
        ``residue_numbers`` (N,) int64 — author residue numbers
        ``chain_ids``       list[str]  — chain letter per residue
        ``Total``           (N,) int64 — in-sphere midpoint count
        ``nHighlyFrst``     (N,) int64
        ``nNeutrallyFrst``  (N,) int64
        ``nMinimallyFrst``  (N,) int64
        ``relHighlyFrustrated``    (N,) float — ratio (0 when Total==0)
        ``relNeutralFrustrated``   (N,) float
        ``relMinimallyFrustrated`` (N,) float

    Notes
    -----
    * Memory complexity O(N × N_pair) — the per-residue sphere check
      builds a single (N, N_pair) bool tensor. For 11BG (N=248,
      N_pair=1517) this is ~376 K bool = 376 KB. Fine for any reasonable
      protein; for 5000+ residues consider chunking the residue axis.
    * The midpoint is taken between the **effective-CB** coords (CB for
      non-Gly, CA for Gly) — same convention LAMMPS-AWSEM dumps.
      The CENTRE of the sphere is the CA, not the effective-CB —
      this matches the R reference ``CA_xyz`` line.
    """
    high_thr, minimal_thr = classification_thresholds

    ca = coords["ca_coords"]
    device = ca.device
    dtype = ca.dtype if ca.dtype.is_floating_point else torch.float64
    n_res = int(ca.shape[0])

    # QA-MISC #27 & #52: validate input shapes and content before any work.
    # Previously this function silently broadcast a too-short ``fi`` across
    # all pair midpoints (or zip-truncated against ``pair_i``), and negative
    # ``pair_i`` / ``pair_j`` would wrap silently via advanced indexing into
    # the coord tensor — producing nonsense distances with no warning.
    n_pair_i = int(pair_i.numel())
    n_pair_j = int(pair_j.numel())
    n_fi = int(fi.numel())
    if n_pair_i != n_pair_j or n_pair_i != n_fi:
        raise ValueError(
            "compute_residue_density: pair_i, pair_j, fi must all have the "
            f"same length; got pair_i={n_pair_i}, pair_j={n_pair_j}, fi={n_fi}."
        )
    if n_pair_i > 0:
        pi_min = int(pair_i.min().item())
        pi_max = int(pair_i.max().item())
        pj_min = int(pair_j.min().item())
        pj_max = int(pair_j.max().item())
        if pi_min < 0 or pi_max >= n_res or pj_min < 0 or pj_max >= n_res:
            raise ValueError(
                f"compute_residue_density: pair indices out of range [0, {n_res}); "
                f"pair_i range [{pi_min}, {pi_max}], pair_j range [{pj_min}, {pj_max}]."
            )
    if n_fi > 0 and not bool(torch.isfinite(fi).all()):
        n_bad = int((~torch.isfinite(fi)).sum().item())
        raise ValueError(
            f"compute_residue_density: {n_bad} of {n_fi} frustration indices "
            "are non-finite (NaN or inf). Classifying them as 'neutral' "
            "silently would bias the per-residue counts."
        )

    # Match the LAMMPS dump's coordinate choice: CB for non-Gly, CA for Gly.
    xb = _xb_coords(coords).to(dtype=dtype)                         # (N, 3)

    pair_i = pair_i.to(device=device, dtype=torch.int64)
    pair_j = pair_j.to(device=device, dtype=torch.int64)
    fi = fi.to(device=device, dtype=dtype)

    # Per-pair midpoint (between the dump's xi/yi/zi and xj/yj/zj).
    midpoints = 0.5 * (xb[pair_i] + xb[pair_j])                     # (N_pair, 3)

    # CA-to-midpoint distance — (N, N_pair).
    ca_dtype = ca.to(dtype=dtype)
    # diff[i, p, :] = midpoint[p] - CA[i]
    diff = midpoints.unsqueeze(0) - ca_dtype.unsqueeze(1)           # (N, N_pair, 3)
    dist = torch.linalg.vector_norm(diff, dim=-1)                   # (N, N_pair)
    # NaN-safe: any pair whose midpoint involves NaN coords is excluded.
    finite = torch.isfinite(dist)
    in_sphere = finite & (dist < ratio)                              # (N, N_pair)

    # Per-pair classification (we re-classify here rather than depend on
    # the caller passing a classification tensor — the R script's classes
    # are computed from `vps[, 6]` directly, NOT from any external label).
    highly = fi <= high_thr                                          # (N_pair,)
    minimally = fi >= minimal_thr
    neutral = (~highly) & (~minimally)

    # Aggregate counts per residue: bool & per-class → int sum along the
    # pair axis.
    total = in_sphere.sum(dim=1).to(torch.int64)                     # (N,)
    n_high = (in_sphere & highly.unsqueeze(0)).sum(dim=1).to(torch.int64)
    n_neut = (in_sphere & neutral.unsqueeze(0)).sum(dim=1).to(torch.int64)
    n_min = (in_sphere & minimally.unsqueeze(0)).sum(dim=1).to(torch.int64)

    # Ratios; guard zero-Total with 0.0 (matches the R `if(total_density > 0)`).
    total_f = total.to(dtype=dtype)
    rel_high = torch.where(
        total > 0, n_high.to(dtype=dtype) / torch.clamp(total_f, min=1.0),
        torch.zeros_like(total_f),
    )
    rel_neut = torch.where(
        total > 0, n_neut.to(dtype=dtype) / torch.clamp(total_f, min=1.0),
        torch.zeros_like(total_f),
    )
    rel_min = torch.where(
        total > 0, n_min.to(dtype=dtype) / torch.clamp(total_f, min=1.0),
        torch.zeros_like(total_f),
    )

    return {
        "residue_numbers": coords["residue_numbers"].to(device=device, dtype=torch.int64),
        "chain_ids": list(coords["chain_ids"]),
        "Total": total,
        "nHighlyFrst": n_high,
        "nNeutrallyFrst": n_neut,
        "nMinimallyFrst": n_min,
        "relHighlyFrustrated": rel_high,
        "relNeutralFrustrated": rel_neut,
        "relMinimallyFrustrated": rel_min,
    }


def density_to_dataframe(density: dict[str, torch.Tensor]) -> pd.DataFrame:
    """Convert the dict returned by :func:`compute_residue_density` to a
    pandas DataFrame matching the ``<PDB>_5adens.dat`` column order.

    pandas is imported lazily so the rest of the module stays pandas-free.
    """
    import pandas as pd

    df = pd.DataFrame({
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
    return df


def emit_5adens_dat(
    *,
    density: dict[str, torch.Tensor],
    output_path: str | Path,
) -> None:
    """Write the dict from :func:`compute_residue_density` to a
    ``<PDB>_5adens.dat`` text file matching the frustratometeR schema.

    The format is space-separated, header on line 1::

        Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst \
            relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated

    Ratios are written as full-precision Python floats (matches the R
    script's ``write()`` behaviour with default ``digits = 15``).
    """
    from pathlib import Path
    p = Path(output_path)
    res = density["residue_numbers"].detach().cpu().tolist()
    chains = density["chain_ids"]
    total = density["Total"].detach().cpu().tolist()
    n_h = density["nHighlyFrst"].detach().cpu().tolist()
    n_n = density["nNeutrallyFrst"].detach().cpu().tolist()
    n_m = density["nMinimallyFrst"].detach().cpu().tolist()
    r_h = density["relHighlyFrustrated"].detach().cpu().tolist()
    r_n = density["relNeutralFrustrated"].detach().cpu().tolist()
    r_m = density["relMinimallyFrustrated"].detach().cpu().tolist()

    header = (
        "Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst "
        "relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated"
    )
    lines: list[str] = [header]
    n = len(res)
    for k in range(n):
        lines.append(
            f"{res[k]} {chains[k]} {total[k]} {n_h[k]} {n_n[k]} {n_m[k]} "
            f"{r_h[k]} {r_n[k]} {r_m[k]}"
        )
    p.write_text("\n".join(lines) + "\n")


__all__ = [
    "DEFAULT_DENSITY_RATIO_A",
    "compute_residue_density",
    "density_to_dataframe",
    "emit_5adens_dat",
]
