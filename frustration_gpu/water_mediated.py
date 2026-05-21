"""AWSEM water-mediated contact energy term (Phase 2b).

Implements the *mediated* half of ``V_Water`` from the AWSEM Hamiltonian —
the outer sigmoid shell at 6.5-9.5 Å between effective-CB atoms. Combined
with :mod:`src.direct_contact`, this gives the full ``V_Water`` term that
appears in LAMMPS-AWSEM's ``energy.log``.

Formula
-------
For every pair of residues (i, j) within the same chain at sequence
separation ``|i - j| >= contact_min_seq_sep`` (default 2 from
``fix_backbone_coeff.data`` ``[Water]`` line ``2 2``), or between any two
residues in different chains::

    θ_mediated(r_ij) = (1/4) × (1 + tanh(η × (r_ij - r_min2))) × (1 + tanh(η × (r_max2 - r_ij)))

with ``r_min2 = 6.5 Å``, ``r_max2 = 9.5 Å``, ``η = 5.0 Å⁻¹``.

The mediated gamma is a burial-density-dependent blend of two tables, one
"protein-mediated" and one "water-mediated"::

    σ_water(ρ) = (1/2) × (1 - tanh(η_σ × (ρ - ρ_0)))
    σ_water(ρ_i, ρ_j)  = σ_water(ρ_i) × σ_water(ρ_j)
    σ_protein(ρ_i, ρ_j) = 1 - σ_water(ρ_i, ρ_j)

So the pair contribution is::

    V_mediated(i, j) = -k_water × θ_mediated(r_ij)
                       × (σ_protein × γ_med_protein[aa_i, aa_j]
                          + σ_water × γ_med_water[aa_i, aa_j])

with ``η_σ = 7.0`` (dimensionless, ``well->par.kappa_sigma``) and
``ρ_0 = 2.6`` (dimensionless burial threshold, ``well->par.treshold``).

This matches ``fix_backbone.cpp:5459-5473`` exactly:

* The C++ writes ``sigma_wat = 0.25 * (1 - tanh(kσ(ρ_i - ρ_0))) * (1 - tanh(kσ(ρ_j - ρ_0)))``.
  Note the C++ uses ``0.25`` (a single product) rather than splitting it as
  ``0.5 * 0.5`` per residue. Numerically identical.
* The C++ writes ``sigma_gamma_mediated = sigma_prot*γ_med_prot + sigma_wat*γ_med_wat``
  (no further prefactor); the ``k_water`` prefactor enters through gamma
  pre-multiplication at file load (``:629-633``). We keep ``k_water`` as a
  separate runtime knob to match :func:`src.direct_contact.direct_contact_energy`.
* Sign convention: ``V_mediated = -k_water × sigma_gamma_mediated × θ_mediated``
  (matches ``:5473``).

Parameters from ``fix_backbone_coeff.data`` ``[Water]`` block
--------------------------------------------------------------
* ``η = 5.0`` Å⁻¹       (== direct-shell η; ``well->par.kappa``)
* ``η_σ = 7.0``         (``well->par.kappa_sigma``)
* ``ρ_0 = 2.6``         (``well->par.treshold``)
* ``r_min2 = 6.5`` Å    (``well->par.well_r_min[1]``)
* ``r_max2 = 9.5`` Å    (``well->par.well_r_max[1]``)

The two mediated gamma tables live in rows 210-419 of ``gamma.dat`` — TWO
columns of values (column 0 = protein-mediated, column 1 = water-mediated).
:func:`src.contact_gamma.load_mediated_gamma` returns both as (20, 20)
tensors.

Burial density ρ
----------------
The ρ argument must be the AWSEM-style local CB density from
:func:`src.burial.compute_rho` (or equivalently the ``"rho"`` field returned
by :func:`src.burial.burial_energy`). Units: dimensionless count of nearby
effective-CB atoms after sigmoid weighting.

Units
-----
``k_water = 1.0 kcal/mol`` by default (LAMMPS ``units real``). The returned
energy is in kcal/mol. Multiply by 4.184 to convert to kJ/mol.

Return-dict conventions
-----------------------
When ``return_pair_matrix=True`` the dict mirrors
:func:`src.direct_contact.direct_contact_energy`:

* ``pair_energy`` and ``pair_mask`` are upper-triangular (``i < j``);
* ``distances`` is the full symmetric matrix;
* ``theta`` is the full mediated-sigmoid matrix (also full-symmetric);
* ``sigma_wat`` and ``sigma_prot`` are full-symmetric.

NaN safety + memory
-------------------
Same dense O(N²) construction as :mod:`src.direct_contact`, sharing the
NaN-safe pairwise distance helper in :mod:`src._contact_common`. At N=8689
(4PKN) the transient memory footprint is comparable (~3-5 GB float32). For
large inputs, future work will tile or sparsify.

LOC budget: ~250-400 lines of code + this docstring.
"""
from __future__ import annotations

import warnings

import torch

from ._contact_common import (
    MEDIATED_SPARSE_MIN_SAFE_A,
    ContactContext,
    SparseContactContext,
    _build_chain_index,
    _check_no_dna_sentinel,
    _check_residue_types_in_range,
    _pair_mask,
    _pairwise_distance_safe,
    _resolve_contact_coords,
    _validate_context_device,
    _validate_context_fingerprint,
    _warn_sparse_cutoff,
)
from .contact_gamma import load_mediated_gamma

# --- numerical constants (Å units except where noted) -------------------------
# Mirror the C++ ``[Water]`` block (``fix_backbone.cpp:257-266``) and
# ``compute_water_potential`` (``:5459-5473``).
MEDIATED_R_MIN_A: float = 6.5     # well_r_min[1]
MEDIATED_R_MAX_A: float = 9.5     # well_r_max[1]
MEDIATED_ETA_PER_A: float = 5.0   # well->par.kappa  (== direct-shell η)
MEDIATED_ETA_SIGMA: float = 7.0   # well->par.kappa_sigma  (dimensionless)
MEDIATED_RHO_0: float = 2.6       # well->par.treshold (dimensionless ρ_0)
CONTACT_MIN_SEQ_SEP: int = 2      # ``|i - j| >= 2`` from [Water]'s "2 2" line


# --- public API ---------------------------------------------------------------
def water_mediated_energy(
    coords: dict[str, torch.Tensor],
    *,
    rho: torch.Tensor,
    gamma_mediated_protein: torch.Tensor | None = None,
    gamma_mediated_water: torch.Tensor | None = None,
    k_water: float = 1.0,
    r_min: float = MEDIATED_R_MIN_A,
    r_max: float = MEDIATED_R_MAX_A,
    eta: float = MEDIATED_ETA_PER_A,
    eta_sigma: float = MEDIATED_ETA_SIGMA,
    rho_0: float = MEDIATED_RHO_0,
    contact_min_seq_sep: int = CONTACT_MIN_SEQ_SEP,
    device: torch.device | None = None,
    return_pair_matrix: bool = False,
    _context: ContactContext | SparseContactContext | None = None,
    sparse: bool = False,
    use_cdist: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Compute the AWSEM water-mediated contact energy ``V_mediated``.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain
        ``ca_coords``, ``cb_coords``, ``residue_types``, ``chain_ids``.
    rho : (N,) tensor
        Per-residue AWSEM burial density from :func:`src.burial.compute_rho`
        (or equivalently ``burial_energy(parsed)["rho"]``). Dimensionless.
    gamma_mediated_protein, gamma_mediated_water : (20, 20) tensors, optional
        Mediated gamma tables. If ``None`` (default) both are loaded from
        ``src/data/gamma.dat`` via
        :func:`src.contact_gamma.load_mediated_gamma`. Must be symmetric.
    k_water : float
        Energy prefactor. Default ``1.0`` kcal/mol (LAMMPS ``units real``).

        **Do not pass custom gamma tables that are already pre-multiplied by
        ``k_water`` together with ``k_water != 1.0`` — you'll double-fold
        the prefactor. A runtime warning is issued in that combination.**
    r_min, r_max : float
        Mediated sigmoid window edges, in Å. Defaults ``6.5`` and ``9.5``
        from ``[Water]`` well 1.
    eta : float
        Distance-sigmoid sharpness, in 1/Å. Default ``5.0``.
    eta_sigma : float
        Burial-sigmoid sharpness (dimensionless). Default ``7.0``.
    rho_0 : float
        Burial threshold (dimensionless). Default ``2.6``.
    contact_min_seq_sep : int
        Minimum same-chain sequence separation. Default ``2``.
    device : torch.device, optional
        Destination device. Defaults to the device of ``coords["ca_coords"]``.
    return_pair_matrix : bool
        If ``True``, return a dict with the scalar ``energy`` AND the
        per-pair matrices (upper-triangular pair_energy & pair_mask; full
        symmetric distances/theta/sigmas).
    sparse : bool
        Speed-sprint #3 Idea 1. If ``True``, expect a
        :class:`SparseContactContext` in ``_context`` and run the 1-D
        pair-list code path. Byte-exact w.r.t. dense. Default ``False``.

        ``return_pair_matrix=True`` is supported alongside ``sparse=True``:
        the 1-D pair_energy is scattered back into an upper-triangular
        ``(N, N)`` tensor at no precision cost. The auxiliary diagnostic
        matrices (theta/sigma_wat/sigma_prot/gamma_pair) are NOT re-densified
        in the sparse + return_pair_matrix mode — they are returned as 1-D
        ``(N_pair,)`` tensors keyed by the same ``pair_i / pair_j``. Callers
        relying on full-matrix diagnostics should stick with ``sparse=False``.
    use_cdist : bool
        Speed-sprint #3 Idea 2. If ``True`` use :func:`torch.cdist` in the
        distance build. Default ``False`` for byte-exact parity.

    Returns
    -------
    torch.Tensor (scalar) when ``return_pair_matrix is False`` (the default),
    a kcal/mol scalar.
    dict otherwise, with keys::

        energy       (scalar) total V_mediated
        pair_energy  (N, N) upper-triangular pair-energy matrix
        pair_mask    (N, N) bool — True where a pair contributed
        distances    (N, N) effective-CB pairwise distance, Å
        theta        (N, N) mediated sigmoid switch
        sigma_wat    (N, N) water-mediated blend coefficient
        sigma_prot   (N, N) protein-mediated blend coefficient
        gamma_pair   (N, N) blended mediated gamma

    Notes
    -----
    * Cross-chain pairs ALWAYS included; same-chain pairs require
      ``|i - j| >= contact_min_seq_sep``.
    * Self-pairs and pairs with NaN effective-CB are excluded.
    * For n=1 / n=0 we return ``tensor(0.0)``.
    * The burial density ``rho`` is *not* differentiable w.r.t. the
      caller's coords here — if you need joint backprop, pass a ρ tensor
      that itself has a graph back to coords (e.g. from
      :func:`src.burial.burial_density`).
    """
    # --- DNA-sentinel guard (QA-1 HIGH) -----------------------------------
    _check_no_dna_sentinel(coords["residue_types"])
    # Finding #39 — also reject out-of-range positives.
    _check_residue_types_in_range(coords["residue_types"])

    # --- resolve device + coords ------------------------------------------
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Opt sprint Idea 3 + Speed-3 Idea 1: dense or sparse context support.
    # Finding #31 — reject cross-type _context with a clear ValueError.
    is_sparse_ctx = isinstance(_context, SparseContactContext)
    if sparse and not is_sparse_ctx:
        raise ValueError(
            "sparse=True requires a SparseContactContext via _context= "
            "(build via build_contact_context(coords, sparse_cutoff=...))."
        )
    if is_sparse_ctx and not sparse:
        raise ValueError(
            "_context is a SparseContactContext but sparse=False."
        )

    if _context is not None:
        # Finding #11 — device-mismatch validation.
        _validate_context_device(_context, device)
        # Finding #30 — stale-context guard.
        _validate_context_fingerprint(_context, coords)
        # Finding #29 — warn when sparse_cutoff is below the water-mediated
        # safe minimum (14 Å). Dropped tail is material here.
        if is_sparse_ctx:
            _warn_sparse_cutoff(
                "water_mediated_energy",
                _context,
                MEDIATED_SPARSE_MIN_SAFE_A,
            )
        cb_or_ca = _context.cb_or_ca
        dtype = _context.dtype
        n = _context.n
    else:
        cb_or_ca = _resolve_contact_coords(coords, device=device)
        dtype = cb_or_ca.dtype
        n = cb_or_ca.shape[0]

    # --- n=0 / n=1 short-circuit ------------------------------------------
    if n < 2:
        zero = torch.zeros((), dtype=dtype, device=device)
        if not return_pair_matrix:
            return zero
        empty_mat = torch.zeros((n, n), dtype=dtype, device=device)
        empty_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
        return {
            "energy": zero,
            "pair_energy": empty_mat,
            "pair_mask": empty_mask,
            "distances": empty_mat,
            "theta": empty_mat,
            "sigma_wat": empty_mat,
            "sigma_prot": empty_mat,
            "gamma_pair": empty_mat,
        }

    aa = coords["residue_types"].to(device=device)                  # (N,)
    if _context is not None:
        chain_idx = _context.chain_idx
    else:
        chain_ids = coords["chain_ids"]
        chain_idx = _build_chain_index(chain_ids, device=device)        # (N,)
    rho = rho.to(device=device, dtype=dtype)                        # (N,)
    if rho.shape != (n,):
        raise ValueError(f"rho must be (N,) with N={n}, got {tuple(rho.shape)}")

    # --- load mediated gamma tables --------------------------------------
    user_gamma = (gamma_mediated_protein is not None) or (gamma_mediated_water is not None)
    if gamma_mediated_protein is None or gamma_mediated_water is None:
        g_p_default, g_w_default = load_mediated_gamma(device=device, dtype=dtype)
        if gamma_mediated_protein is None:
            gamma_mediated_protein = g_p_default
        if gamma_mediated_water is None:
            gamma_mediated_water = g_w_default
    gamma_mediated_protein = gamma_mediated_protein.to(device=device, dtype=dtype)
    gamma_mediated_water = gamma_mediated_water.to(device=device, dtype=dtype)
    if gamma_mediated_protein.shape != (20, 20):
        raise ValueError(
            f"gamma_mediated_protein must be (20, 20), got {tuple(gamma_mediated_protein.shape)}"
        )
    if gamma_mediated_water.shape != (20, 20):
        raise ValueError(
            f"gamma_mediated_water must be (20, 20), got {tuple(gamma_mediated_water.shape)}"
        )

    if user_gamma and k_water != 1.0:
        warnings.warn(
            "water_mediated_energy: passing a custom mediated gamma table "
            "AND ``k_water != 1.0`` is ambiguous — the C++ reference folds "
            "k_water into the loaded gamma at load time, so if your custom "
            "tables are already premultiplied you will double-fold. The "
            "PyTorch convention here treats the gamma tables as RAW.",
            stacklevel=2,
        )

    # --- shared scalar tensors -------------------------------------------
    eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
    r_min_t = torch.as_tensor(r_min, dtype=dtype, device=device)
    r_max_t = torch.as_tensor(r_max, dtype=dtype, device=device)
    eta_sigma_t = torch.as_tensor(eta_sigma, dtype=dtype, device=device)
    rho_0_t = torch.as_tensor(rho_0, dtype=dtype, device=device)
    k_water_t = torch.as_tensor(k_water, dtype=dtype, device=device)

    # Per-residue σ_water — same for both code paths (N,).
    sigma_water_per_res = 0.5 * (1.0 - torch.tanh(eta_sigma_t * (rho - rho_0_t)))

    if sparse:
        # ---- Sparse path (Speed-3 Idea 1) --------------------------------
        # 1-D intermediates; dense total agrees to within ~1e-12 relative
        # (tail of theta beyond sparse_cutoff is ~1e-22 per pair).
        ctx = _context  # SparseContactContext
        pair_i = ctx.pair_i
        pair_j = ctx.pair_j
        r_pair = ctx.r_ij                                              # (N_pair,)
        if contact_min_seq_sep in ctx.pair_mask_min_sep:
            pair_mask_sep = ctx.pair_mask_min_sep[contact_min_seq_sep]
        else:
            pair_mask_sep = (~ctx.same_chain) | (
                ctx.seq_diff >= contact_min_seq_sep
            )

        # Per-pair σ_water = σ_i × σ_j → blended gamma.
        sigma_wat_1d = sigma_water_per_res[pair_i] * sigma_water_per_res[pair_j]
        sigma_prot_1d = 1.0 - sigma_wat_1d
        g_prot_pair_1d = gamma_mediated_protein[aa[pair_i], aa[pair_j]]
        g_wat_pair_1d = gamma_mediated_water[aa[pair_i], aa[pair_j]]
        gamma_pair_1d = sigma_prot_1d * g_prot_pair_1d + sigma_wat_1d * g_wat_pair_1d

        # Fused energy expression (Idea 3).
        pair_energy_1d_full = (
            -k_water_t * gamma_pair_1d * 0.25
            * (1.0 + torch.tanh(eta_t * (r_pair - r_min_t)))
            * (1.0 + torch.tanh(eta_t * (r_max_t - r_pair)))
        )
        pair_energy_1d = torch.where(
            pair_mask_sep,
            pair_energy_1d_full,
            torch.zeros_like(pair_energy_1d_full),
        )
        total = pair_energy_1d.sum()

        if not return_pair_matrix:
            return total

        # Re-densify pair_energy and pair_mask for the dict API.
        pair_energy_upper = torch.zeros((n, n), dtype=dtype, device=device)
        pair_energy_upper[pair_i, pair_j] = pair_energy_1d
        pair_mask_full = torch.zeros((n, n), dtype=torch.bool, device=device)
        pair_mask_full[pair_i, pair_j] = pair_mask_sep
        # Finding #6 — mask theta on excluded pairs so the diagnostic
        # doesn't show nonzero values for pairs that did not contribute.
        # On the sparse path the theta is already only computed over
        # in-cutoff pairs; we additionally zero out seq-sep-excluded ones.
        theta_1d_raw = 0.25 * (
            (1.0 + torch.tanh(eta_t * (r_pair - r_min_t)))
            * (1.0 + torch.tanh(eta_t * (r_max_t - r_pair)))
        )
        theta_1d = torch.where(
            pair_mask_sep, theta_1d_raw, torch.zeros_like(theta_1d_raw)
        )
        # Finding #42 — sparse context no longer caches dense distances.
        # Fall back to the 1-D per-pair distances; callers wanting the full
        # symmetric matrix should use the dense path.
        return {
            "energy": total,
            "pair_energy": pair_energy_upper,
            "pair_mask": pair_mask_full,
            "distances": ctx.dist if ctx.dist is not None else r_pair,
            "theta": theta_1d,
            "sigma_wat": sigma_wat_1d,
            "sigma_prot": sigma_prot_1d,
            "gamma_pair": gamma_pair_1d,
        }

    # ---- Dense path -----------------------------------------------------
    if _context is not None and contact_min_seq_sep in _context.geom_mask_min_sep:
        mask = _context.geom_mask_min_sep[contact_min_seq_sep]      # (N, N)
    else:
        mask = _pair_mask(cb_or_ca, chain_idx, contact_min_seq_sep)     # (N, N)

    # --- pairwise distance (NaN-safe via double-where) --------------------
    fill = 0.5 * (r_min + r_max)
    dist, safe_dist = _pairwise_distance_safe(
        cb_or_ca, mask, fill_value=fill, use_cdist=use_cdist,
    )

    # σ_water / σ_protein on the (N, N) grid.
    sigma_wat = sigma_water_per_res.unsqueeze(1) * sigma_water_per_res.unsqueeze(0)
    sigma_prot = 1.0 - sigma_wat

    # Per-pair gamma blend (kept for return-dict diagnostics).
    g_prot_pair = gamma_mediated_protein[aa.unsqueeze(1), aa.unsqueeze(0)]
    g_wat_pair = gamma_mediated_water[aa.unsqueeze(1), aa.unsqueeze(0)]
    gamma_pair = sigma_prot * g_prot_pair + sigma_wat * g_wat_pair

    # Fused energy expression (Idea 3) — V_mediated = -k_water × γ × θ.
    full_pair_energy = (
        -k_water_t * gamma_pair * 0.25
        * (1.0 + torch.tanh(eta_t * (safe_dist - r_min_t)))
        * (1.0 + torch.tanh(eta_t * (r_max_t - safe_dist)))
    )                                                                  # (N, N)
    pair_energy = torch.where(mask, full_pair_energy, torch.zeros_like(full_pair_energy))

    # --- total energy: sum over UNORDERED pairs (upper triangle) ---------
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    pair_energy_upper = torch.where(upper, pair_energy, torch.zeros_like(pair_energy))

    total = pair_energy_upper.sum()

    if not return_pair_matrix:
        return total
    # Finding #6 — the diagnostic ``theta`` is reconstructed from
    # ``safe_dist``, which has masked-out / NaN pair positions filled with
    # the mid-window value. That filled distance evaluates to ``theta ≈ 1``
    # (the peak of the sigmoid window), so callers inspecting ``theta``
    # would see misleadingly large entries for cross-chain / seq-skipped /
    # NaN-row pairs that never contributed to the energy. Mask the
    # diagnostic with the same ``mask`` used for the energy reduction so
    # the user sees a faithful picture: theta == 0 on excluded pairs.
    theta_raw = 0.25 * (
        (1.0 + torch.tanh(eta_t * (safe_dist - r_min_t)))
        * (1.0 + torch.tanh(eta_t * (r_max_t - safe_dist)))
    )
    theta = torch.where(mask, theta_raw, torch.zeros_like(theta_raw))
    return {
        "energy": total,
        "pair_energy": pair_energy_upper,
        "pair_mask": mask & upper,
        "distances": dist,
        "theta": theta,
        "sigma_wat": sigma_wat,
        "sigma_prot": sigma_prot,
        "gamma_pair": gamma_pair,
    }


def water_mediated_pair_energy(
    r_ij: torch.Tensor | float,
    aa_i: int,
    aa_j: int,
    rho_i: float,
    rho_j: float,
    *,
    gamma_mediated_protein: torch.Tensor | None = None,
    gamma_mediated_water: torch.Tensor | None = None,
    k_water: float = 1.0,
    r_min: float = MEDIATED_R_MIN_A,
    r_max: float = MEDIATED_R_MAX_A,
    eta: float = MEDIATED_ETA_PER_A,
    eta_sigma: float = MEDIATED_ETA_SIGMA,
    rho_0: float = MEDIATED_RHO_0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Scalar V_mediated for a single (r, aa_i, aa_j, ρ_i, ρ_j) tuple.

    Used by tests for the per-pair hand-check. Mirrors
    :func:`src.direct_contact.direct_pair_energy`. No sequence-separation logic.

    Findings #39 / #57 — validate inputs:
    * ``aa_i, aa_j`` must lie in ``[0, 20)``.
    * Custom ``gamma_mediated_protein`` / ``gamma_mediated_water`` (if
      supplied) must have shape ``(20, 20)``.
    * ``r_ij`` must be finite and non-negative.

    Returns a 0-d tensor in kcal/mol.
    """
    if not (0 <= int(aa_i) < 20):
        raise ValueError(
            f"aa_i must lie in [0, 20), got {int(aa_i)}."
        )
    if not (0 <= int(aa_j) < 20):
        raise ValueError(
            f"aa_j must lie in [0, 20), got {int(aa_j)}."
        )
    if gamma_mediated_protein is None or gamma_mediated_water is None:
        g_p_default, g_w_default = load_mediated_gamma(device=device, dtype=dtype)
        if gamma_mediated_protein is None:
            gamma_mediated_protein = g_p_default
        if gamma_mediated_water is None:
            gamma_mediated_water = g_w_default
    # Finding #57 — validate shape on any custom table BEFORE indexing.
    if tuple(gamma_mediated_protein.shape) != (20, 20):
        raise ValueError(
            f"gamma_mediated_protein must have shape (20, 20), got "
            f"{tuple(gamma_mediated_protein.shape)}."
        )
    if tuple(gamma_mediated_water.shape) != (20, 20):
        raise ValueError(
            f"gamma_mediated_water must have shape (20, 20), got "
            f"{tuple(gamma_mediated_water.shape)}."
        )
    # Finding #57 (audit-fix follow-up) — also reject NaN/inf values.
    # A NaN-filled gamma table previously passed shape validation and
    # silently propagated NaN through every downstream energy.
    if not bool(torch.isfinite(gamma_mediated_protein).all()):
        raise ValueError(
            "gamma_mediated_protein contains non-finite values (NaN or inf). "
            "Silent NaN propagation would poison every dependent energy."
        )
    if not bool(torch.isfinite(gamma_mediated_water).all()):
        raise ValueError(
            "gamma_mediated_water contains non-finite values (NaN or inf). "
            "Silent NaN propagation would poison every dependent energy."
        )
    gamma_mediated_protein = gamma_mediated_protein.to(dtype=dtype)
    gamma_mediated_water = gamma_mediated_water.to(dtype=dtype)
    if device is not None:
        gamma_mediated_protein = gamma_mediated_protein.to(device=device)
        gamma_mediated_water = gamma_mediated_water.to(device=device)
    dev = gamma_mediated_protein.device

    r = torch.as_tensor(r_ij, dtype=dtype, device=dev)
    # Finding #57/#58 — validate r_ij is finite and non-negative.
    if not bool(torch.isfinite(r).all()):
        raise ValueError(f"r_ij must be finite, got {float(r):g}")
    if float(r) < 0.0:
        raise ValueError(f"r_ij must be non-negative, got {float(r):g}")
    eta_t = torch.as_tensor(eta, dtype=dtype, device=dev)
    r_min_t = torch.as_tensor(r_min, dtype=dtype, device=dev)
    r_max_t = torch.as_tensor(r_max, dtype=dtype, device=dev)
    eta_sigma_t = torch.as_tensor(eta_sigma, dtype=dtype, device=dev)
    rho_0_t = torch.as_tensor(rho_0, dtype=dtype, device=dev)
    rho_i_t = torch.as_tensor(rho_i, dtype=dtype, device=dev)
    rho_j_t = torch.as_tensor(rho_j, dtype=dtype, device=dev)

    t_min = torch.tanh(eta_t * (r - r_min_t))
    t_max = torch.tanh(eta_t * (r_max_t - r))
    theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)

    sigma_wat_i = 0.5 * (1.0 - torch.tanh(eta_sigma_t * (rho_i_t - rho_0_t)))
    sigma_wat_j = 0.5 * (1.0 - torch.tanh(eta_sigma_t * (rho_j_t - rho_0_t)))
    sigma_wat = sigma_wat_i * sigma_wat_j
    sigma_prot = 1.0 - sigma_wat

    g_p = gamma_mediated_protein[aa_i, aa_j]
    g_w = gamma_mediated_water[aa_i, aa_j]
    gamma_blended = sigma_prot * g_p + sigma_wat * g_w
    return -k_water * gamma_blended * theta
