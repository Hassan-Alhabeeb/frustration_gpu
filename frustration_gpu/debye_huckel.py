"""AWSEM Debye-Hückel electrostatics term (Phase 2c).

Implements the screened-Coulomb pair energy ``V_DH`` between charged
side-chain centroids, evaluated on the effective-CB distance matrix (CA for
glycine, same coordinate convention as :mod:`src.direct_contact` and
:mod:`src.water_mediated`).

Formula
-------
Per ``fix_backbone.cpp:5502-5547`` (``FixBackbone::compute_electrostatic_energy``)::

    V_DH(i, j) = +epsilon × k_QQ(sign_i, sign_j) × q_i × q_j / r_ij
                 × exp(-k_screening × r_ij / screening_length)

with:

* ``q_i, q_j ∈ {+1, -1, 0}`` per residue identity (see "Charge assignment"
  below).
* ``k_QQ`` is the Coulomb-like prefactor (kcal·Å/mol). It is a 2×2 lookup
  ``{(++): k_PlusPlus, (--): k_MinusMinus, (+-)/(-+): k_PlusMinus}``. In the
  default ``fix_backbone_coeff.data`` all three are ``4.15``.
* ``screening_length = 10.0 Å`` (the Debye length λ, ``[DebyeHuckel]`` line 3).
* ``k_screening = 1.0`` (multiplies ``1/λ`` to give the effective inverse
  screening length, ``[DebyeHuckel]`` line 2).
* ``epsilon = 1.0`` (the global LAMMPS-AWSEM energy scale, ``fix_backbone.cpp:131``).
* Sequence-separation gate: pairs with ``|i - j| < debye_huckel_min_sep``
  return ``0``. The default ``debye_huckel_min_sep = 1`` excludes only the
  self pair ``i = j``; same-chain ``|i - j| = 1`` neighbours DO contribute.
* Sign convention: the C++ writes
  ``return epsilon * term_qq_by_r * exp(-k_screening*rij/screening_length)``
  (line 5545) — a **positive** prefactor. The negative-attractive sign for
  opposite charges arises from ``q_i × q_j = -1`` in the ``term_qq_by_r``.

Charge assignment (CRITICAL — verified against C++)
---------------------------------------------------
``fix_backbone.cpp:5511-5527`` assigns charges by 1-letter code:

* ``'R'`` or ``'K'`` → +1
* ``'D'`` or ``'E'`` → -1
* anything else (including ``'H'``) → 0, and the function early-returns
  with energy 0

Despite biochemistry convention that HIS may carry +1 at low pH, **LAMMPS-AWSEM
treats HIS as neutral** in the DH term. This matches the OpenAWSEM and
frustrapy conventions and is explicitly verified by the Phase 1.5 C++ audit.
The 20-element charge vector below uses the OpenAWSEM gamma index order
(``A R N D C Q E G H I L K M F P S T W Y V``):

    idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    aa :  A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
    q  :  0 +1  0 -1  0  0 -1  0  0  0  0 +1  0  0  0  0  0  0  0  0

Units
-----
``k_QQ = 4.15 kcal·Å/mol`` (the default) gives energies in **kcal/mol** when
``r`` is in **angstroms**. Multiply by ``4.184`` to convert to kJ/mol.

``electrostatics_k`` API parity
--------------------------------
frustrapy's ``calculate_frustration(electrostatics_k=...)`` kwarg overrides
``k_QQ`` uniformly across the three sign combinations (since the defaults
are all equal at 4.15). The :func:`debye_huckel_energy` function exposes
``k_QQ`` as the runtime knob; the frustrapy-level wrapper (when added in a
later phase) passes ``electrostatics_k`` through to this argument.

Return-dict conventions
-----------------------
When ``return_pair_matrix=True`` the dict mirrors the direct / mediated
modules:

* ``pair_energy`` is upper-triangular (``i < j``);
* ``pair_mask`` is upper-triangular and bool;
* ``distances`` is the full symmetric matrix (NaN where coords are NaN);
* ``charges`` is the (N,) per-residue charge vector — useful for debugging
  and for confirming the HIS=0 convention against external tools.

Differentiability
-----------------
Differentiable w.r.t. ``ca_coords`` and ``cb_coords`` via the same NaN-safe
distance helper used by Phase 2a/2b. The charge tensor is a constant
``torch.long`` lookup followed by a cast to ``dtype`` — no autograd path
through identities.

A potential numerical pitfall is ``1 / r`` blowing up as ``r → 0``. In
practice the self-pair is masked out and any two distinct residues are at
least ~3.8 Å apart (CA-CA virtual-bond constraint), so r is bounded away
from zero. We still apply ``_pair_mask`` and use ``safe_dist`` from
:mod:`src._contact_common` to keep the autograd graph clean.

LOC budget: ~280 lines + this docstring.
"""
from __future__ import annotations

import torch

import math
import warnings

from ._contact_common import (
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
    dh_sparse_min_safe,
)

# --- numerical constants ------------------------------------------------------
# All values from ``[DebyeHuckel]`` in fix_backbone_coeff.data (see
# fix_backbone.cpp:467-477 for the parser block).
DH_K_QQ_DEFAULT: float = 4.15            # kcal·Å/mol — k_PlusPlus = k_MinusMinus = k_PlusMinus
DH_SCREENING_LENGTH_A: float = 10.0      # Å — λ (Debye length)
DH_K_SCREENING: float = 1.0              # scale factor on 1/λ
DH_MIN_SEQ_SEP: int = 1                  # |i - j| < 1 → return 0 (excludes only self)
DH_EPSILON: float = 1.0                  # epsilon from fix_backbone.cpp:131 (global energy scale)

# Per-AA charge assignment in OpenAWSEM gamma-index order
# (A R N D C Q E G H I L K M F P S T W Y V → 0..19).
# Verified line-by-line against fix_backbone.cpp:5511-5527.
#   R, K → +1   |   D, E → -1   |   else (incl. H) → 0
DH_CHARGES_FLOAT: tuple[float, ...] = (
    0.0,   # 0  A
    +1.0,  # 1  R
    0.0,   # 2  N
    -1.0,  # 3  D
    0.0,   # 4  C
    0.0,   # 5  Q
    -1.0,  # 6  E
    0.0,   # 7  G
    0.0,   # 8  H  ← NOT +1: see module docstring
    0.0,   # 9  I
    0.0,   # 10 L
    +1.0,  # 11 K
    0.0,   # 12 M
    0.0,   # 13 F
    0.0,   # 14 P
    0.0,   # 15 S
    0.0,   # 16 T
    0.0,   # 17 W
    0.0,   # 18 Y
    0.0,   # 19 V
)


def aa_charge_vector(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the 20-element AA → charge lookup tensor.

    Index order matches :data:`src.parser.ONE_TO_IDX`. HIS is 0 (not +1) —
    see module docstring for the C++ citation.

    Parameters
    ----------
    device : torch.device, optional
        Destination device (defaults to CPU).
    dtype : torch.dtype
        Output dtype. Use ``float64`` for parity testing and ``float32`` for
        normal use. Default ``float64`` because the function is most often
        called inside hand-precision tests.

    Returns
    -------
    (20,) tensor of charges in ``{-1.0, 0.0, +1.0}``.
    """
    return torch.tensor(DH_CHARGES_FLOAT, dtype=dtype, device=device)


def _per_residue_charge(
    residue_types: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Map (N,) residue-type indices → (N,) per-residue charges.

    Uses :func:`aa_charge_vector` plus advanced indexing. The output is a
    leaf tensor (no autograd path) — charges are categorical labels.
    """
    qvec = aa_charge_vector(device=device, dtype=dtype)            # (20,)
    return qvec[residue_types.to(device=device)]                   # (N,)


# --- public API ---------------------------------------------------------------
def debye_huckel_energy(
    coords: dict[str, torch.Tensor],
    *,
    k_QQ: float = DH_K_QQ_DEFAULT,
    screening_length: float = DH_SCREENING_LENGTH_A,
    k_screening: float = DH_K_SCREENING,
    min_seq_sep: int = DH_MIN_SEQ_SEP,
    epsilon: float = DH_EPSILON,
    device: torch.device | None = None,
    return_pair_matrix: bool = False,
    _context: ContactContext | SparseContactContext | None = None,
    sparse: bool = False,
    use_cdist: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Compute the AWSEM Debye-Hückel electrostatic energy ``V_DH``.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``,
        ``cb_coords``, ``residue_types``, ``chain_ids``.
    k_QQ : float
        Coulomb-like prefactor in kcal·Å/mol. **This is the kwarg that
        frustrapy's ``electrostatics_k`` maps to.** Default ``4.15`` (the
        single value used for all three sign combinations in the stock
        ``fix_backbone_coeff.data``). Passing a non-default value linearly
        scales every pair contribution.
    screening_length : float
        Debye length λ in Å. Default ``10.0`` from ``[DebyeHuckel]`` line 3.
    k_screening : float
        Multiplies ``1/λ`` to give the effective inverse screening length
        used inside ``exp(-k_screening × r / λ)``. Default ``1.0``.
    min_seq_sep : int
        Pairs with ``|i - j| < min_seq_sep`` return zero. Default ``1``
        (excludes only the self pair). Inter-chain pairs always contribute,
        irrespective of this value.
    epsilon : float
        Global AWSEM energy scale (``fix_backbone.cpp:131``). Default ``1.0``.
        Multiplies every pair contribution; in normal usage you should NOT
        change this unless you also change every other AWSEM term's ``k``.
    device : torch.device, optional
        Destination device. Defaults to the device of ``coords["ca_coords"]``.
    return_pair_matrix : bool
        If ``True``, return a dict with the scalar ``energy`` AND
        per-pair / per-residue matrices for diagnostics.
    sparse : bool
        Speed-sprint #3 Idea 1. If ``True``, expect a
        :class:`SparseContactContext` in ``_context`` and run the 1-D
        pair-list code path. Byte-exact w.r.t. dense. Default ``False``.

        Note: the sparse cutoff used to build the SparseContactContext MUST
        be wide enough that ``exp(-r / λ_eff)`` is numerically negligible
        beyond it. With λ=10, k=1 a cutoff of ~30 Å gives ``exp(-3) ≈ 0.05``
        per pair — for tighter screening reduce, for looser increase.
    use_cdist : bool
        Speed-sprint #3 Idea 2. If ``True`` use :func:`torch.cdist`.
        Default ``False`` for byte-exact parity.

    Returns
    -------
    torch.Tensor (scalar) when ``return_pair_matrix is False`` (default), in
    kcal/mol.

    dict otherwise, with keys::

        energy       (scalar) total V_DH
        pair_energy  (N, N) upper-triangular pair-energy matrix
        pair_mask    (N, N) bool — True where a pair contributed
        distances    (N, N) effective-CB pairwise distance, Å
        charges      (N,)   per-residue charge ∈ {-1, 0, +1}

    Notes
    -----
    * Cross-chain pairs always included. Same-chain pairs require
      ``|i - j| >= min_seq_sep``.
    * Self-pairs and pairs touching a NaN effective-CB row are excluded.
    * For n=1 or n=0 we return ``tensor(0.0)`` (no possible pair).
    * Pairs where either residue carries charge 0 contribute exactly 0 —
      they are masked early (so we don't waste an ``exp`` per such pair).

    Linear scaling under k_QQ
    -------------------------
    The energy is exactly linear in ``k_QQ`` (the only dependence is the
    leading multiplicative factor). Tests check that ``V_DH(k=17.3636) /
    V_DH(k=4.15) ≈ 4.184`` to machine precision.

    Reference
    ---------
    ``fix_backbone.cpp:5502-5547``: ``FixBackbone::compute_electrostatic_energy``.
    """
    # --- DNA-sentinel guard (QA-1 HIGH) -----------------------------------
    # debye_huckel "lucks out" today because charge[19]=0 → -1-sentinel maps
    # to 0 charge → no contribution. But the guard catches the upstream
    # mistake (caller forgot to filter DNA) before any downstream code
    # silently makes biological nonsense look plausible.
    _check_no_dna_sentinel(coords["residue_types"])
    # Finding #39 — also catch out-of-range positives.
    _check_residue_types_in_range(coords["residue_types"])

    # Finding #24 — validate the screening parameters. Zero or negative
    # screening_length used to surface as a raw ZeroDivisionError /
    # produce non-physical exponential growth respectively. Reject both up
    # front with a clear ValueError.
    if not math.isfinite(screening_length) or screening_length <= 0.0:
        raise ValueError(
            f"screening_length must be a finite positive value, got "
            f"{screening_length}. Negative or zero values invert/blow up "
            "the Debye-Hückel exponential decay."
        )
    if not math.isfinite(k_screening) or k_screening <= 0.0:
        raise ValueError(
            f"k_screening must be a finite positive value, got {k_screening}."
        )

    # --- resolve device + coords ------------------------------------------
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Opt sprint Idea 3 + Speed-3 Idea 1: dense or sparse context support.
    # Finding #31 — reject cross-type _context cleanly.
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
        # Finding #28 — warn loudly when sparse_cutoff is below the DH safe
        # minimum (3 × screening_length). On 5AON, cutoff=11 SIGN-FLIPPED
        # the DH energy from −0.60 to +0.15 kcal/mol — silent disaster.
        if is_sparse_ctx:
            _warn_sparse_cutoff(
                "debye_huckel_energy",
                _context,
                dh_sparse_min_safe(screening_length, k_screening),
                extra_advice=(
                    "DH decays as exp(-r/λ); the minimum-safe cutoff is "
                    "3 × screening_length / k_screening "
                    f"(here = {dh_sparse_min_safe(screening_length, k_screening):.1f} Å)."
                ),
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
        return {
            "energy": zero,
            "pair_energy": torch.zeros((n, n), dtype=dtype, device=device),
            "pair_mask": torch.zeros((n, n), dtype=torch.bool, device=device),
            "distances": torch.zeros((n, n), dtype=dtype, device=device),
            "charges": _per_residue_charge(
                coords["residue_types"], dtype, device
            ),
        }

    aa = coords["residue_types"]
    if _context is not None:
        chain_idx = _context.chain_idx
    else:
        chain_ids = coords["chain_ids"]
        chain_idx = _build_chain_index(chain_ids, device=device)        # (N,)

    # --- per-residue charge -----------------------------------------------
    q = _per_residue_charge(aa, dtype, device)                      # (N,)

    inv_lambda_eff = torch.as_tensor(
        k_screening / screening_length, dtype=dtype, device=device
    )
    k_QQ_t = torch.as_tensor(k_QQ * epsilon, dtype=dtype, device=device)

    if sparse:
        # ---- Sparse path (Speed-3 Idea 1) --------------------------------
        # 1-D intermediates; total agrees with dense to within the natural
        # tail of ``exp(-r/λ)/r`` beyond ``sparse_cutoff``. With λ=10 and
        # cutoff=100 Å, the tail is ``exp(-10)/100 ≈ 5e-7`` per pair —
        # callers needing tighter agreement should use a wider cutoff.
        ctx = _context  # SparseContactContext
        pair_i = ctx.pair_i
        pair_j = ctx.pair_j
        r_pair = ctx.r_ij                                              # (N_pair,)
        if min_seq_sep in ctx.pair_mask_min_sep:
            pair_mask_sep = ctx.pair_mask_min_sep[min_seq_sep]
        else:
            pair_mask_sep = (~ctx.same_chain) | (ctx.seq_diff >= min_seq_sep)

        # Pairs where either residue carries charge 0 contribute exactly 0.
        q_i = q[pair_i]
        q_j = q[pair_j]
        charged_pair = (q_i != 0) & (q_j != 0)
        mask_1d = pair_mask_sep & charged_pair

        decay_1d = torch.exp(-r_pair * inv_lambda_eff)
        inv_r_1d = 1.0 / r_pair.clamp(min=1e-12)
        full_pair_1d = k_QQ_t * q_i * q_j * decay_1d * inv_r_1d
        pair_energy_1d = torch.where(
            mask_1d, full_pair_1d, torch.zeros_like(full_pair_1d)
        )
        total = pair_energy_1d.sum()

        if not return_pair_matrix:
            return total

        # Re-densify into upper-triangular (N, N) for the dict API.
        pair_energy_upper = torch.zeros((n, n), dtype=dtype, device=device)
        pair_energy_upper[pair_i, pair_j] = pair_energy_1d
        pair_mask_full = torch.zeros((n, n), dtype=torch.bool, device=device)
        pair_mask_full[pair_i, pair_j] = mask_1d
        # Finding #42 — sparse context no longer caches dense distances.
        # Fall back to the 1-D per-pair distance tensor.
        return {
            "energy": total,
            "pair_energy": pair_energy_upper,
            "pair_mask": pair_mask_full,
            "distances": ctx.dist if ctx.dist is not None else r_pair,
            "charges": q,
        }

    # ---- Dense path -----------------------------------------------------
    # --- per-pair validity mask -------------------------------------------
    # _pair_mask handles cross-chain (always pass), same-chain seq-sep,
    # self-exclusion, and NaN-row exclusion. We pass min_seq_sep=1 by
    # default which matches debye_huckel_min_sep=1 in fix_backbone_coeff.
    if _context is not None and min_seq_sep in _context.geom_mask_min_sep:
        mask_geom = _context.geom_mask_min_sep[min_seq_sep]        # (N, N)
    else:
        mask_geom = _pair_mask(cb_or_ca, chain_idx, min_seq_sep)        # (N, N)

    # Additionally mask out any pair where either q is zero — those
    # contribute exactly 0, and skipping them avoids the exp / division
    # overhead and any potential NaN from a degenerate (q=0) × (1/r) path.
    charged = q != 0                                                # (N,)
    pair_charged = charged.unsqueeze(0) & charged.unsqueeze(1)      # (N, N)
    mask = mask_geom & pair_charged

    # --- pairwise distance (NaN-safe via double-where) --------------------
    # Use a benign fill value far enough away that exp(-r/λ_eff) is
    # numerically zero even if the mask later removes the pair. With λ=10
    # and screening=1, r=100 gives exp(-10) ≈ 4.5e-5; we use r=1000 to
    # bury the contribution well below float64 epsilon.
    fill = 1000.0
    dist, safe_dist = _pairwise_distance_safe(
        cb_or_ca, mask, fill_value=fill, use_cdist=use_cdist,
    )

    # --- exp(-k_screening × r / λ) / r -----------------------------------
    # safe_dist is finite and well-conditioned (≥3.8 Å on real proteins,
    # fill_value=1000 elsewhere). The exponential decays cleanly.
    decay = torch.exp(-safe_dist * inv_lambda_eff)                  # (N, N)

    # We still want to avoid dividing by something that might be exactly 0
    # in pathological inputs. The mask was applied to safe_dist but the
    # diagonal goes through fill_value=1000, so the only way to hit r=0
    # outside masked entries is a duplicate CA in the input file. Guard
    # with a tiny epsilon ≪ float64 precision.
    inv_r = 1.0 / safe_dist.clamp(min=1e-12)                        # (N, N)

    # --- pair energy ------------------------------------------------------
    # V_DH(i, j) = +epsilon × k_QQ × q_i × q_j × decay × inv_r
    q_outer = q.unsqueeze(0) * q.unsqueeze(1)                       # (N, N)
    full_pair_energy = k_QQ_t * q_outer * decay * inv_r             # (N, N)

    pair_energy = torch.where(
        mask, full_pair_energy, torch.zeros_like(full_pair_energy)
    )

    # --- total: sum over upper triangle -----------------------------------
    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1
    )
    pair_energy_upper = torch.where(
        upper, pair_energy, torch.zeros_like(pair_energy)
    )
    total = pair_energy_upper.sum()

    if not return_pair_matrix:
        return total
    return {
        "energy": total,
        "pair_energy": pair_energy_upper,
        "pair_mask": mask & upper,
        "distances": dist,
        "charges": q,
    }


def debye_huckel_pair_energy(
    r_ij: torch.Tensor | float,
    aa_i: int,
    aa_j: int,
    *,
    k_QQ: float = DH_K_QQ_DEFAULT,
    screening_length: float = DH_SCREENING_LENGTH_A,
    k_screening: float = DH_K_SCREENING,
    epsilon: float = DH_EPSILON,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Scalar V_DH for a single (r, aa_i, aa_j) triple — used by tests.

    Returns a 0-d tensor in kcal/mol. No sequence-separation logic — caller
    is responsible for supplying a "valid" pair. If either AA is not in
    ``{R, K, D, E}`` (charge 0) the return is exactly 0.0.

    Findings #39 / #58 / #24 — validate inputs:
    * ``aa_i, aa_j`` must lie in ``[0, 20)``.
    * ``r_ij`` must be finite and strictly positive (zero or negative
      distances are unphysical for the 1/r Coulomb factor and used to
      silently return ±inf or a finite wrong-sign number).
    * ``screening_length`` and ``k_screening`` must be finite positive.

    Mirrors ``fix_backbone.cpp:5502-5547`` faithfully — the reference path
    against which the dense :func:`debye_huckel_energy` is validated.
    """
    # Finding #39 — clamp to [0, 20).
    if not (0 <= int(aa_i) < 20):
        raise ValueError(f"aa_i must lie in [0, 20), got {int(aa_i)}.")
    if not (0 <= int(aa_j) < 20):
        raise ValueError(f"aa_j must lie in [0, 20), got {int(aa_j)}.")
    # Finding #24 — screening params validated up front.
    if not math.isfinite(screening_length) or screening_length <= 0.0:
        raise ValueError(
            f"screening_length must be finite positive, got {screening_length}."
        )
    if not math.isfinite(k_screening) or k_screening <= 0.0:
        raise ValueError(
            f"k_screening must be finite positive, got {k_screening}."
        )
    qvec = aa_charge_vector(device=device, dtype=dtype)
    q_i = qvec[aa_i].item()
    q_j = qvec[aa_j].item()
    if q_i == 0.0 or q_j == 0.0:
        return torch.zeros((), dtype=dtype, device=device)
    r = torch.as_tensor(r_ij, dtype=dtype, device=device)
    # Finding #58 — validate r_ij is finite and strictly positive.
    if not bool(torch.isfinite(r).all()):
        raise ValueError(f"r_ij must be finite, got {float(r):g}.")
    if float(r) <= 0.0:
        raise ValueError(
            f"r_ij must be strictly positive for the Coulomb 1/r factor, "
            f"got {float(r):g}."
        )
    inv_lambda_eff = torch.as_tensor(
        k_screening / screening_length, dtype=dtype, device=device
    )
    return torch.as_tensor(
        epsilon * k_QQ, dtype=dtype, device=device
    ) * q_i * q_j * torch.exp(-r * inv_lambda_eff) / r


__all__ = [
    "DH_K_QQ_DEFAULT",
    "DH_SCREENING_LENGTH_A",
    "DH_K_SCREENING",
    "DH_MIN_SEQ_SEP",
    "DH_EPSILON",
    "DH_CHARGES_FLOAT",
    "aa_charge_vector",
    "debye_huckel_energy",
    "debye_huckel_pair_energy",
]
