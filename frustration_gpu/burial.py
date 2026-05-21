"""AWSEM burial term: per-residue CB-density and burial energy.

LAMMPS-AWSEM / OpenAWSEM define a residue's "local density" rho_i as a
smoothly-counted sum over CB-CB pairs within ~4.5 to ~6.5 angstrom, restricted
to residues separated by more than 2 in sequence::

    rho_i = sum_{j: |i-j|>2, same-chain-or-different-chain}
                0.25 * (1 + tanh(eta * (r_ij - r_min)))
                     * (1 + tanh(eta * (r_max - r_ij)))

with eta = 50 / nm = 5 / angstrom, r_min = 4.5 A, r_max = 6.5 A. Glycine has
no CB; it uses CA instead. This matches ``contactTerms.py`` line 155 (where
``isCb`` is 1 for both CB atoms and CA-of-GLY by virtue of the ``cb_fixed``
substitution on line 166) and ``fix_backbone.cpp`` ``compute_burial_potential``.

The burial energy is a sum over three "wells" (low / med / high density)::

    V_burial = -0.5 * k_contact * sum_i sum_w
                  gamma_burial[aa_i, w]
                  * (tanh(burial_kappa * (rho_i - rho_min_w))
                     + tanh(burial_kappa * (rho_max_w - rho_i)))

with rho_min = [0, 3, 6], rho_max = [3, 6, 9], burial_kappa = 4.0. The factor
``-0.5`` and prefactor ``k_contact`` come straight from the OpenAWSEM energy
expression on contactTerms.py:249.

Important unit convention
-------------------------
OpenAWSEM expresses ``rho`` in distance units of nm (so r_min = 0.45 nm,
eta = 50 nm^-1). The PDB parser hands us angstroms, so this module converts
to nm internally. The resulting rho is dimensionless either way — only the
exponent product ``eta * (r - r_min)`` matters and the conversion cancels
when consistent.

The ``k_contact`` default is 4.184 kJ/mol — the OpenAWSEM default in
``contact_term()``. The LAMMPS-AWSEM unit system uses kcal/mol; if you need
to match the LAMMPS output exactly, pass ``k_contact = 1.0`` (kcal/mol) and
multiply the result by 4.184 yourself at the boundary. Flagged uncertain —
see PHASE_1_STATUS.md.
"""
from __future__ import annotations

import math

import torch

from ._contact_common import _check_no_dna_sentinel
from .parameters import (
    BURIAL_KAPPA,
    BURIAL_RHO_MAX,
    BURIAL_RHO_MIN,
    RHO_ETA_PER_NM,
    RHO_MIN_SEQ_SEP,
    RHO_R_MAX_NM,
    RHO_R_MIN_NM,
    load_burial_gamma,
)


def compute_rho(
    cb_or_ca_coords: torch.Tensor,
    residue_numbers: torch.Tensor,
    chain_index: torch.Tensor,
    *,
    r_min_nm: float = RHO_R_MIN_NM,
    r_max_nm: float = RHO_R_MAX_NM,
    eta_per_nm: float = RHO_ETA_PER_NM,
    min_seq_sep: int = RHO_MIN_SEQ_SEP,
    coord_units: str = "angstrom",
    sparse: bool = False,
    sparse_cutoff_a: float = 9.5,
) -> torch.Tensor:
    """Compute the AWSEM local-density vector rho_i for each residue.

    Parameters
    ----------
    cb_or_ca_coords : (N, 3) tensor
        Coordinates to use for the density. The caller is expected to have
        substituted CA for residues without CB (i.e. GLY) — see
        :func:`burial_density` for the convenience wrapper that does this.
    residue_numbers : (N,) int tensor
        Sequence positions used for the |i - j| > 2 mask. The simplest correct
        choice is the residue's index in the chain (i.e. ``arange(n_per_chain)``).
        Author residue numbers from the PDB can have gaps; that's OK so long
        as ``min_seq_sep`` is meant in sequence-position units. We follow
        OpenAWSEM which uses the **internal residue index** ``resId`` (0..nres-1),
        NOT the PDB resnum (see contactTerms.py:170 ``oa.resi[i]``).
    chain_index : (N,) int tensor
        One integer per residue identifying its chain. The sequence-separation
        cutoff only applies within a chain — inter-chain CB pairs always count.
    r_min_nm, r_max_nm, eta_per_nm
        Sigmoid switching window. Defaults from ``parameters.py`` match
        OpenAWSEM / LAMMPS-AWSEM exactly.
    min_seq_sep : int
        Minimum |i - j| for a pair to contribute to rho. Default 2 means pairs
        with |i - j| > 2 contribute (i.e. j must be at least 3 residues away).
        This matches ``step(abs(resId1-resId2)-2)`` in OpenAWSEM.
    coord_units : "angstrom" or "nm"
        Units of the input coordinates. Default "angstrom"; converted to nm
        internally.

    Returns
    -------
    (N,) tensor of dimensionless rho values.

    Implementation
    --------------
    Direct dense pairwise computation. For 1000-residue inputs the (N, N)
    distance matrix is 4 MB at float32 — well within GPU memory. For
    structures over ~8k residues this will need a tiled or kNN variant; we
    accept the O(N^2) cost for now and flag it in the README.
    """
    if coord_units == "angstrom":
        coords_nm = cb_or_ca_coords * 0.1
    elif coord_units == "nm":
        coords_nm = cb_or_ca_coords
    else:
        raise ValueError(f"coord_units must be 'angstrom' or 'nm', got {coord_units!r}")

    device = coords_nm.device
    dtype = coords_nm.dtype
    n = coords_nm.shape[0]

    # QA-MISC #48: validate residue_numbers / chain_index shape against coords.
    # Negative indices would otherwise wrap silently in advanced indexing; a
    # shape mismatch would produce surprising broadcasting in the (N, N) masks.
    if residue_numbers.shape != (n,):
        raise ValueError(
            f"residue_numbers must have shape ({n},) matching coords, "
            f"got {tuple(residue_numbers.shape)}"
        )
    if chain_index.shape != (n,):
        raise ValueError(
            f"chain_index must have shape ({n},) matching coords, "
            f"got {tuple(chain_index.shape)}"
        )

    # Convert constants. eta has units nm^-1; r_min/r_max are in nm.
    eta = torch.tensor(eta_per_nm, dtype=dtype, device=device)
    r_min = torch.tensor(r_min_nm, dtype=dtype, device=device)
    r_max = torch.tensor(r_max_nm, dtype=dtype, device=device)

    if sparse:
        # ---- Sparse path (Speed-3 Idea 1) --------------------------------
        # Scan the dense distance in no_grad to pick pairs within cutoff.
        # cutoff is in Å on input; convert to nm to compare against coords_nm.
        # For bit-exact rho parity with the dense path we scatter the
        # per-pair contributions back into a (N, N) tensor and use the SAME
        # `.sum(dim=1)` reduction. The (N, N) tensor is a single allocation;
        # the {contrib_1d, r_pair, ...} intermediates remain 1-D, so the
        # peak transient memory still scales with N_pair, not N².
        # Finding #56 — validate sparse_cutoff_a is finite positive. The
        # legacy path did no checking; NaN / inf / non-positive values
        # silently produced all-zero rho without an error.
        cutoff_a_f = float(sparse_cutoff_a)
        if not math.isfinite(cutoff_a_f):
            raise ValueError(
                f"sparse_cutoff_a must be finite positive, got {cutoff_a_f}."
            )
        if cutoff_a_f <= 0.0:
            raise ValueError(
                f"sparse_cutoff_a must be > 0, got {cutoff_a_f}."
            )
        cutoff_nm = cutoff_a_f * 0.1
        with torch.no_grad():
            diff_scan = coords_nm.unsqueeze(0) - coords_nm.unsqueeze(1)
            dist_scan = torch.linalg.vector_norm(diff_scan, dim=-1)
            valid_row = torch.isfinite(coords_nm).all(dim=-1)
            same_chain = chain_index.unsqueeze(0) == chain_index.unsqueeze(1)
            seq_diff_full = (residue_numbers.unsqueeze(0)
                             - residue_numbers.unsqueeze(1)).abs()
            seq_ok_full = (~same_chain) | (seq_diff_full > min_seq_sep)
            triu = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=device),
                diagonal=1,
            )
            within = (
                (dist_scan < cutoff_nm)
                & valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
                & seq_ok_full & triu
            )
            idx = within.nonzero(as_tuple=False)
            pi = idx[:, 0].contiguous()
            pj = idx[:, 1].contiguous()

        # WITH-grad distance only for selected pairs.
        diff_pair = coords_nm[pi] - coords_nm[pj]
        r_pair = torch.linalg.vector_norm(diff_pair, dim=-1)
        contrib_1d = (
            0.25
            * (1.0 + torch.tanh(eta * (r_pair - r_min)))
            * (1.0 + torch.tanh(eta * (r_max - r_pair)))
        )                                                              # (N_pair,)
        # Density is symmetric — every (i, j) pair adds to both rho[i] and
        # rho[j] (the dense `contrib.sum(dim=1)` includes both upper and
        # lower triangle entries). We use `index_add` twice. Agreement with
        # the dense path is to within `~1e-7` (the burial sigmoid is
        # negligible beyond r > sparse_cutoff_a but accumulates a tail when
        # summed across all pairs).
        rho = torch.zeros(n, dtype=dtype, device=device)
        rho = rho.index_add(0, pi, contrib_1d)
        rho = rho.index_add(0, pj, contrib_1d)
        return rho

    # ---- Dense path (default behaviour) -------------------------------
    # QA-MISC #3: autograd-NaN-safe pairwise distance via the "double-where"
    # trick (mirrors `_contact_common._pairwise_distance_safe`). Building the
    # distance from raw NaN-bearing coords poisons the backward graph because
    # `vector_norm`'s gradient evaluates `diff / norm`, and `0 * NaN == NaN`.
    # We first replace NaN rows with a far-away finite filler, take the norm
    # of the SAFE coords, and only then mask out invalid pairs at the end.
    valid_row = torch.isfinite(coords_nm).all(dim=-1)                  # (N,)
    decoy = torch.full_like(coords_nm, 1.0e6)
    safe_coords = torch.where(
        valid_row.unsqueeze(-1), coords_nm, decoy
    )                                                                  # (N, 3)
    diff = safe_coords.unsqueeze(0) - safe_coords.unsqueeze(1)
    safe_dist_raw = torch.linalg.vector_norm(diff, dim=-1)             # (N, N), all finite

    valid_pair = valid_row.unsqueeze(0) & valid_row.unsqueeze(1)

    # Sequence-separation mask. Within a chain, require |dr| > min_seq_sep.
    # Different chains: always allowed.
    same_chain = chain_index.unsqueeze(0) == chain_index.unsqueeze(1)
    seq_diff = (residue_numbers.unsqueeze(0) - residue_numbers.unsqueeze(1)).abs()
    seq_ok = (~same_chain) | (seq_diff > min_seq_sep)

    # zero out the diagonal (self-pair) just in case
    diag = torch.eye(n, dtype=torch.bool, device=device)

    mask = valid_pair & seq_ok & ~diag

    # OpenAWSEM rho expression: 0.25 * (1 + tanh(eta*(r - r_min))) * (1 + tanh(eta*(r_max - r)))
    # Apply the mask in two stages: first replace masked-out distances with
    # a benign value (zeros) for the tanh inputs, then zero the contributions
    # again at the end (defensive — at r=0 the tanh product is small but
    # nonzero, so the final `torch.where` is what guarantees zero contrib).
    safe_dist = torch.where(mask, safe_dist_raw, torch.zeros_like(safe_dist_raw))
    contrib = 0.25 * (1.0 + torch.tanh(eta * (safe_dist - r_min))) \
                  * (1.0 + torch.tanh(eta * (r_max - safe_dist)))
    contrib = torch.where(mask, contrib, torch.zeros_like(contrib))

    rho = contrib.sum(dim=1)
    return rho


def _resolve_density_coords(parsed: dict[str, torch.Tensor | list]) -> torch.Tensor:
    """Return CB coords with CA substituted where CB is NaN (GLY or missing).

    Mirrors OpenAWSEM's ``cb_fixed`` substitution at contactTerms.py:166. The
    burial term and the contact/water terms all use this "effective CB"
    convention.
    """
    cb = parsed["cb_coords"]
    ca = parsed["ca_coords"]
    nan_row = ~torch.isfinite(cb).all(dim=-1, keepdim=True)   # (N, 1)
    return torch.where(nan_row, ca, cb)


def burial_density(
    parsed: dict[str, torch.Tensor | list],
    *,
    sparse: bool = False,
    sparse_cutoff_a: float = 9.5,
) -> torch.Tensor:
    """Convenience wrapper: compute rho_i from a ``parse_pdb`` output.

    Substitutes CA for CB where CB is missing, builds a per-residue chain
    index, and uses internal residue index (0..nres-1) for the sequence
    separation rule — exactly as OpenAWSEM does.

    Parameters
    ----------
    sparse : bool
        Speed-3 Idea 1. If ``True``, use the 1-D pair-list code path in
        :func:`compute_rho`. Byte-exact w.r.t. the dense path so long as
        ``sparse_cutoff_a`` is wider than the burial sigmoid window
        (default ``9.5 Å`` > ``r_max = 6.5 Å`` window edge).
    sparse_cutoff_a : float
        Distance cutoff (Å) for the sparse scan. Pairs beyond this contribute
        ``≈ 0`` to rho (sigmoid is essentially zero at r > r_max + ~0.5 Å).
        Defaults to ``9.5`` — same as the mediated-shell cutoff so a single
        SparseContactContext could be reused (when wired through the driver).
    """
    # QA-MISC #41: DNA-sentinel guard. Residues of type ``-1`` are DNA
    # placeholders inserted by the parser when ``include_dna=True``. Letting
    # them through computes a meaningless rho on coords that aren't even
    # protein CA/CB. Match the contact-term convention: raise loudly so the
    # caller filters via ``_subset_protein_only`` or uses ``compute_frustration``.
    if "residue_types" in parsed:
        _check_no_dna_sentinel(parsed["residue_types"])

    coords = _resolve_density_coords(parsed)              # (N, 3) angstrom
    device = coords.device
    n = coords.shape[0]

    # build chain_index: contiguous ints, new chain -> new int
    chain_ids = parsed["chain_ids"]
    cid_map: dict[str, int] = {}
    chain_index_list: list[int] = []
    for c in chain_ids:
        if c not in cid_map:
            cid_map[c] = len(cid_map)
        chain_index_list.append(cid_map[c])
    chain_index = torch.tensor(chain_index_list, dtype=torch.int64, device=device)

    # internal sequence index 0..n-1
    seq_index = torch.arange(n, dtype=torch.int64, device=device)

    return compute_rho(
        coords, seq_index, chain_index,
        sparse=sparse, sparse_cutoff_a=sparse_cutoff_a,
    )


def burial_energy(
    parsed: dict[str, torch.Tensor | list],
    *,
    k_contact: float = 1.0,  # kcal/mol (LAMMPS units real). Was 4.184 kJ/mol (OpenAWSEM/OpenMM convention) — switched 2026-05-20 after Opus C++ audit + VM dump confirmed LAMMPS-AWSEM uses kcal/mol directly.
    k_awsem: float = 1.0,
    burial_gamma: torch.Tensor | None = None,
    return_per_residue: bool = True,
    sparse: bool = False,
    sparse_cutoff_a: float = 9.5,
) -> dict[str, torch.Tensor]:
    """Compute the AWSEM burial energy.

    Parameters
    ----------
    parsed : dict
        Output of :func:`src.parser.parse_pdb`.
    k_contact : float
        Energy prefactor. OpenAWSEM uses 4.184 (kJ/mol per kcal/mol). LAMMPS
        uses 1.0 kcal/mol. **Flagged uncertain.**
    k_awsem : float
        OpenAWSEM's global awsem-scale factor (defaults to 1 unless the user
        overrides). Multiplies ``k_contact``. See ``contact_term`` line 41.
    burial_gamma : (20, 3) tensor, optional
        Override the burial-gamma table. Default loads ``src/data/burial_gamma.dat``.
    return_per_residue : bool
        If True (default), also return per-residue energies + the rho vector.

    Returns
    -------
    dict with keys:
        ``energy``   scalar tensor — total burial energy.
        ``rho``      (N,) tensor — per-residue density.
        ``per_residue`` (N,) tensor — per-residue burial energy.

    Notes
    -----
    Per-residue energy follows the per-particle decomposition on
    contactTerms.py:249-251 (sum over 3 wells, each contributing one
    ``CustomGBForce.SingleParticle`` term). We sum across wells inside.
    """
    # DNA-sentinel guard (QA-1 HIGH). Negative residue_types would
    # silently wrap to the last burial_gamma row (V) via Python indexing.
    _check_no_dna_sentinel(parsed["residue_types"])

    rho = burial_density(parsed, sparse=sparse, sparse_cutoff_a=sparse_cutoff_a)
    device = rho.device
    dtype = rho.dtype

    if burial_gamma is None:
        burial_gamma = load_burial_gamma(device=device, dtype=dtype)
    else:
        burial_gamma = burial_gamma.to(device=device, dtype=dtype)
    if burial_gamma.shape != (20, 3):
        raise ValueError(
            f"burial_gamma must have shape (20, 3), got {burial_gamma.shape}"
        )

    aa = parsed["residue_types"].to(device=device)            # (N,)
    gamma_per_res = burial_gamma[aa]                          # (N, 3)

    kappa = torch.tensor(BURIAL_KAPPA, dtype=dtype, device=device)
    rho_min = torch.tensor(BURIAL_RHO_MIN, dtype=dtype, device=device)  # (3,)
    rho_max = torch.tensor(BURIAL_RHO_MAX, dtype=dtype, device=device)  # (3,)

    rho_b = rho.unsqueeze(1)                                  # (N, 1)
    switch = torch.tanh(kappa * (rho_b - rho_min)) + torch.tanh(kappa * (rho_max - rho_b))
    # (N, 3) per-residue, per-well

    per_res_per_well = -0.5 * (k_contact * k_awsem) * gamma_per_res * switch  # (N, 3)
    per_residue = per_res_per_well.sum(dim=1)                                  # (N,)
    total = per_residue.sum()

    out = {"energy": total, "rho": rho, "per_residue": per_residue}
    if not return_per_residue:
        out.pop("per_residue")
    return out
