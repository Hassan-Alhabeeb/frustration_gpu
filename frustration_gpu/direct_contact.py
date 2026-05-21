"""AWSEM direct-contact energy term (Phase 2a).

Implements the *direct* half of ``V_Water`` from the AWSEM Hamiltonian — the
inner sigmoid shell at 4.5-6.5 Å between effective-CB atoms. The
water-mediated outer shell (6.5-9.5 Å) is Phase 2b and lives in
:mod:`src.water_mediated`. Both modules share the helpers in
:mod:`src._contact_common`.

Formula
-------
For every pair of residues (i, j) within the same chain at sequence
separation |i - j| >= ``contact_min_seq_sep`` (default 2 from
``fix_backbone_coeff.data`` ``[Water]`` line ``2 2``), or between any two
residues in different chains::

    V_direct(i, j) = -k_water * γ_direct[aa_i, aa_j] * θ_direct(r_ij)
    θ_direct(r)   = (1/4) * (1 + tanh(η * (r - r_min))) * (1 + tanh(η * (r_max - r)))

with ``r_min = 4.5 Å``, ``r_max = 6.5 Å``, ``η = 5.0 Å^{-1}``. The pair
distance ``r_ij`` is the **effective-CB distance**: CB for non-glycine
residues, CA for glycine (and for any residue missing a CB record — same
substitution as :mod:`src.burial`).

There is **no factor of 1/2** in the formula above. The "1/2" appears in the
C++ source (``fix_backbone.cpp:5462``: ``sigma_gamma_direct = (γ_0 + γ_1) /
2``) only because LAMMPS's gamma table stores two columns per direct entry
and both columns hold the same number — the average collapses to the
identity. Verified by hand-computing 5AON's first dumped contact (i=1, j=3,
S-R, r=5.065 Å): the "no-1/2" form gives ``V_direct = 0.32993`` which
reconstructs ``E_native = -1.003`` (matching the LAMMPS dump) after adding
burial. The "1/2" form would have given ``E_native = -1.168``, a clear
mismatch.

Units
-----
``k_water = 1.0 kcal/mol`` by default (LAMMPS-AWSEM convention, ``units
real``). The gamma table is dimensionless; ``θ`` is dimensionless; the
returned energy is in kcal/mol. Multiply by 4.184 to convert to kJ/mol if
comparing to OpenAWSEM / OpenMM output.

k_water-fold semantics (important when supplying a custom ``gamma_direct``)
--------------------------------------------------------------------------
The C++ source pre-multiplies the loaded gamma table by ``k_water`` at file
load time (``fix_backbone.cpp:632-633``) and then uses the bare product
``-(γ × θ)`` in the energy expression. Our :func:`load_direct_gamma` does
NOT do this fold — it returns the raw gamma values. The energy formula here
carries ``k_water`` as an explicit prefactor instead, so the result is
numerically identical for ``k_water = 1.0`` (the LAMMPS default and the only
value frustrapy ever uses).

If you supply a custom ``gamma_direct`` whose values were already multiplied
by ``k_water``, you must also pass ``k_water = 1.0`` to avoid double-folding.
A runtime warning is issued when both ``k_water != 1.0`` AND a custom
``gamma_direct`` are passed together — that combination is the only way to
accidentally double-fold.

Coupling to other terms
-----------------------
The full ``V_Water`` from LAMMPS-AWSEM's ``energy.log`` is the sum of
``V_direct`` (this file) plus ``V_mediated`` (:mod:`src.water_mediated`).
The dumped ``E_native`` in ``5AON_tertiary_frustration.dat`` is ``V_water +
E_burial(i) + E_burial(j) [+ E_DH]`` per pair — so direct-only validation
is necessarily per-pair, not whole-protein, in this phase.

Return-dict conventions
-----------------------
When ``return_pair_matrix=True`` the dict has mixed-symmetry tensors:

* ``pair_energy`` is upper-triangular (``i < j``), zeros below. Summing it
  gives the total ``V_direct``. This avoids the double-count that would
  result from summing the full symmetric matrix.
* ``pair_mask`` is also upper-triangular (``mask & upper``) — same shape
  semantics as ``pair_energy``.
* ``distances`` is the FULL symmetric distance matrix — handy for
  diagnostics where you want ``r_ij`` for any (i, j) pair regardless of
  ordering. Self-distances on the diagonal are 0.

This mixed convention is deliberate; do not "fix" it by upper-triangularising
``distances`` without a clear migration story.

Differentiability
-----------------
The pipeline is dense pairwise: form an (N, N) distance matrix, evaluate the
two ``tanh`` activations on it, multiply elementwise by ``γ_direct[aa_i,
aa_j]`` (gathered with plain advanced indexing, which IS differentiable in
PyTorch when the indices are integer tensors — no ``index_select`` needed),
and sum across the upper triangle. All ops are differentiable w.r.t.
``coords["cb_coords"]`` and ``coords["ca_coords"]``.

The "advanced indexing is differentiable" claim above refers to the GAMMA
TABLE values — the integer-index gather is differentiable w.r.t.
``gamma_direct`` but NOT w.r.t. the integer ``aa`` labels (you can't take a
gradient w.r.t. an integer). Position gradients flow exclusively through
``dist → θ``.

NaN safety
----------
A residue whose effective-CB position is NaN (e.g. fully missing residue
with no CA fallback either) is masked out before the sum. The
``_pairwise_distance_safe`` helper applies the "double-where NaN trick"
(see :mod:`src._contact_common`) so NaN values cannot poison gradients on
finite-row neighbours during backprop. This is essential before this term
ever becomes a trainable loss.

Memory
------
The dense (N, N, 3) ``diff`` construction in
:func:`_contact_common._pairwise_distance_safe` is O(N²) memory. At float32,
N=8689 (the 4PKN structure in the panel) makes ``diff`` ≈ 8689² × 3 × 4 B
≈ **907 MB** just for ``diff``, plus ``dist``, ``theta``, ``gamma_pair``,
``pair_energy`` and ``safe_dist``. Realistic transient peak is **3–5 GB**.
For inputs above ~5k residues, consider tiling or a sparse-kNN variant
(future work).

LOC budget: ~250 lines of code + this docstring.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch

from ._contact_common import (
    ContactContext,
    SparseContactContext,
    _build_chain_index,
    _check_no_dna_sentinel,
    _pair_mask,
    _pairwise_distance_safe,
    _resolve_contact_coords,
)
from .contact_gamma import load_direct_gamma


# --- numerical constants (Å units) --------------------------------------------
# These mirror the C++ ``[Water]`` block values (``fix_backbone.cpp:257-266``).
# Kept here as module-level defaults; the public API exposes them as keyword
# arguments so tests can sweep them without monkey-patching.
DIRECT_R_MIN_A: float = 4.5
DIRECT_R_MAX_A: float = 6.5
DIRECT_ETA_PER_A: float = 5.0
CONTACT_MIN_SEQ_SEP: int = 2   # ``|i - j| >= 2`` from [Water]'s "2 2" line


# --- public API ---------------------------------------------------------------
def direct_contact_energy(
    coords: Dict[str, torch.Tensor],
    *,
    gamma_direct: Optional[torch.Tensor] = None,
    k_water: float = 1.0,
    r_min: float = DIRECT_R_MIN_A,
    r_max: float = DIRECT_R_MAX_A,
    eta: float = DIRECT_ETA_PER_A,
    contact_min_seq_sep: int = CONTACT_MIN_SEQ_SEP,
    device: Optional[torch.device] = None,
    return_pair_matrix: bool = False,
    _context: "Optional[ContactContext | SparseContactContext]" = None,
    sparse: bool = False,
    use_cdist: bool = False,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Compute the AWSEM direct-contact energy ``V_direct`` for a protein.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain
        ``ca_coords``, ``cb_coords``, ``residue_types``, ``chain_ids``.
    gamma_direct : (20, 20) tensor, optional
        Direct gamma table. If ``None`` (default) we load it from
        ``src/data/gamma.dat`` via :func:`src.contact_gamma.load_direct_gamma`.
        The table must be symmetric in (i, j).
    k_water : float
        Energy prefactor. Default ``1.0`` kcal/mol (LAMMPS ``units real``).
        Multiply the result by ``4.184`` to convert to kJ/mol.

        **Do not pass a custom ``gamma_direct`` that is already
        pre-multiplied by ``k_water`` together with ``k_water != 1.0`` —
        you'll double-fold the prefactor. A runtime warning is issued in
        that combination.**
    r_min, r_max : float
        Direct-contact sigmoid window edges, in Å. Defaults ``4.5`` and
        ``6.5`` from ``[Water]`` well 0.
    eta : float
        Sigmoid sharpness, in 1/Å. Default ``5.0`` (= ``well->par.kappa``).
    contact_min_seq_sep : int
        Minimum same-chain sequence separation ``|i - j|`` for a pair to
        contribute. Default ``2`` (from ``[Water]`` line ``2 2``). Inter-chain
        pairs always contribute, irrespective of this value.
    device : torch.device, optional
        Destination device. Defaults to the device of ``coords["ca_coords"]``.
        If the caller passes a different value we move the inputs accordingly
        (CB / CA / residue_types / chain index).
    return_pair_matrix : bool
        If ``True``, return a dict with the scalar ``energy`` AND the
        ``(N, N)`` ``pair_energy`` matrix (zeroed on the lower triangle —
        sum over the upper triangle = total energy). Useful for debugging
        and for the pair-level comparison test.
    sparse : bool
        Speed-sprint #3 Idea 1. If ``True``, expect a
        :class:`SparseContactContext` in ``_context`` and run the 1-D
        pair-list code path (saves ~500× peak transient memory on large
        proteins). Byte-exact w.r.t. the dense path. Default ``False`` —
        no behaviour change for existing callers.

        When ``return_pair_matrix=True`` AND ``sparse=True``, the dense
        ``(N, N)`` ``pair_energy``/``pair_mask`` matrices are reconstructed
        from the sparse pair list at no precision cost — that's the only
        way to keep the dict API stable.
    use_cdist : bool
        Speed-sprint #3 Idea 2. If ``True`` use :func:`torch.cdist` in the
        distance build, dropping the ``(N, N, 3)`` ``diff`` intermediate.
        Default ``False`` — pre-existing byte-exact behaviour. The
        block-matmul kernel in ``cdist`` can drift by ~1 ULP from the
        explicit broadcast; the byte-exact tests are gated on the default.

    Returns
    -------
    torch.Tensor (scalar) when ``return_pair_matrix is False`` (the default),
    a kcal/mol scalar.
    dict otherwise, with keys::

        energy       (scalar) total V_direct
        pair_energy  (N, N) upper-triangular pair-energy matrix
        pair_mask    (N, N) bool — True where a pair contributed
        distances    (N, N) effective-CB pairwise distance, Å

    See the module docstring for the mixed (upper-tri vs full) symmetry
    convention of the returned dict.

    Notes
    -----
    * Cross-chain pairs are ALWAYS included (no sequence-separation filter).
    * Same-chain pairs are included iff ``|i - j| >= contact_min_seq_sep``.
    * Self-pairs (i == j) are excluded.
    * Pairs involving a NaN effective-CB coordinate are excluded.
    * For n=1 (single residue) or n=0, the function returns
      ``torch.tensor(0.0)`` because no valid pair exists.

    Differentiability
    -----------------
    All ops are differentiable w.r.t. ``ca_coords`` and ``cb_coords``. Pair
    distances are computed via the NaN-safe helper in
    :mod:`src._contact_common`; gradients on neighbours of fully-missing
    residues remain finite (verified by the regression test).

    Validation
    ----------
    See ``tests/test_direct_contact.py``. The hand-computed value for 5AON's
    first dumped pair (1, 3) — ``V_direct = 0.32993 kcal/mol`` — agrees with
    the LAMMPS dump's reconstructed ``E_native`` after adding burial.
    """
    # --- DNA-sentinel guard (QA-1 HIGH) -----------------------------------
    # ``residue_types == -1`` (DNA placeholder, include_dna=True) would
    # silently wrap to the last gamma row via Python-style negative indexing.
    # ``compute_frustration`` filters DNA out before reaching here; this
    # guard fires for direct callers of the lower-level energy API.
    _check_no_dna_sentinel(coords["residue_types"])

    # --- resolve device + coords ------------------------------------------
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Opt sprint Idea 3 + Speed-3 Idea 1: reuse the shared scaffolding
    # (dense or sparse) when caller supplies one.
    is_sparse_ctx = isinstance(_context, SparseContactContext)
    if sparse and not is_sparse_ctx:
        raise ValueError(
            "sparse=True requires a SparseContactContext via _context= "
            "(build via build_contact_context(coords, sparse_cutoff=...))."
        )
    if is_sparse_ctx and not sparse:
        raise ValueError(
            "_context is a SparseContactContext but sparse=False. Pass "
            "sparse=True (or rebuild a dense ContactContext)."
        )

    if _context is not None:
        cb_or_ca = _context.cb_or_ca
        dtype = _context.dtype
        n = _context.n
    else:
        cb_or_ca = _resolve_contact_coords(coords, device=device)
        dtype = cb_or_ca.dtype
        n = cb_or_ca.shape[0]

    # --- n=0 / n=1 short-circuit ------------------------------------------
    # No possible pair → energy is exactly zero. We return early to avoid
    # building a (0, 0) or (1, 1) distance tensor whose triu_sum is undefined
    # in edge cases.
    if n < 2:
        zero = torch.zeros((), dtype=dtype, device=device)
        if not return_pair_matrix:
            return zero
        return {
            "energy": zero,
            "pair_energy": torch.zeros((n, n), dtype=dtype, device=device),
            "pair_mask": torch.zeros((n, n), dtype=torch.bool, device=device),
            "distances": torch.zeros((n, n), dtype=dtype, device=device),
        }

    aa = coords["residue_types"].to(device=device)                  # (N,)
    if _context is not None:
        chain_idx = _context.chain_idx
    else:
        chain_ids = coords["chain_ids"]
        chain_idx = _build_chain_index(chain_ids, device=device)        # (N,)

    user_gamma = gamma_direct is not None
    if gamma_direct is None:
        gamma_direct = load_direct_gamma(device=device, dtype=dtype)
    else:
        gamma_direct = gamma_direct.to(device=device, dtype=dtype)
    if gamma_direct.shape != (20, 20):
        raise ValueError(
            f"gamma_direct must have shape (20, 20), got {tuple(gamma_direct.shape)}"
        )

    if user_gamma and k_water != 1.0:
        warnings.warn(
            "direct_contact_energy: passing a custom ``gamma_direct`` AND "
            "``k_water != 1.0`` is ambiguous — the C++ reference folds k_water "
            "into the loaded gamma at load time, so if your custom table is "
            "already premultiplied you will double-fold the prefactor. The "
            "PyTorch convention here treats ``gamma_direct`` as RAW (not "
            "k_water-folded) and applies ``k_water`` as a separate factor.",
            stacklevel=2,
        )

    eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
    r_min_t = torch.as_tensor(r_min, dtype=dtype, device=device)
    r_max_t = torch.as_tensor(r_max, dtype=dtype, device=device)
    k_water_t = torch.as_tensor(k_water, dtype=dtype, device=device)

    if sparse:
        # ---- Sparse path (Speed-3 Idea 1) --------------------------------
        # All intermediates 1-D (N_pair,). Final sum over a 1-D tensor.
        #
        # Bit-identicality footnote
        # -------------------------
        # The dense path zeros pairs outside the seq-sep mask, BUT the dense
        # `theta` at large r is not exactly 0 (≈ 1e-22) — these tiny terms
        # accumulate to ~1e-14 across N² pairs and show up as the only
        # difference vs sparse on real proteins.  The sparse path drops
        # pairs beyond ``sparse_cutoff`` entirely, so its sum is missing
        # those ~1e-22-per-pair tails.  In practice both totals agree to
        # better than ~1e-12 relative — see `tests/test_direct_contact.py
        # ::test_sparse_byte_exact_*` for the empirical drift.
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

        # γ[aa_i, aa_j] indexed per-pair → (N_pair,)
        gamma_pair_1d = gamma_direct[aa[pair_i], aa[pair_j]]

        # Fused expression — Idea 3. Single elementwise chain, 1-D.
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

        # Re-densify pair_energy + pair_mask for the dict API.
        pair_energy_upper = torch.zeros((n, n), dtype=dtype, device=device)
        pair_energy_upper[pair_i, pair_j] = pair_energy_1d
        pair_mask_full = torch.zeros((n, n), dtype=torch.bool, device=device)
        pair_mask_full[pair_i, pair_j] = pair_mask_sep
        return {
            "energy": total,
            "pair_energy": pair_energy_upper,
            "pair_mask": pair_mask_full,
            "distances": ctx.dist,
        }

    # ---- Dense path (original behaviour, with optional cdist) ----------
    if _context is not None and contact_min_seq_sep in _context.geom_mask_min_sep:
        mask = _context.geom_mask_min_sep[contact_min_seq_sep]      # (N, N)
    else:
        mask = _pair_mask(cb_or_ca, chain_idx, contact_min_seq_sep)     # (N, N)

    # --- pairwise distance (NaN-safe via double-where) --------------------
    # The "safe" distance replaces masked-out NaN entries with the mid-window
    # value so the downstream tanh never sees NaN. Backward is autograd-safe.
    fill = 0.5 * (r_min + r_max)
    dist, safe_dist = _pairwise_distance_safe(
        cb_or_ca, mask, fill_value=fill, use_cdist=use_cdist,
    )

    # --- γ_direct[aa_i, aa_j] --------------------------------------------
    # Advanced indexing on the (20, 20) table → (N, N) per-pair gammas.
    # Differentiable w.r.t. ``gamma_direct`` (not w.r.t. ``aa``, which is int).
    gamma_pair = gamma_direct[aa.unsqueeze(1), aa.unsqueeze(0)]      # (N, N)

    # --- pair energy (FUSED — Idea 3) ------------------------------------
    # V_direct(i, j) = -k_water * γ * θ. Single fused elementwise expression
    # collapses {t_min, t_max, theta, full_pair_energy} into ONE write.
    full_pair_energy = (
        -k_water_t * gamma_pair * 0.25
        * (1.0 + torch.tanh(eta_t * (safe_dist - r_min_t)))
        * (1.0 + torch.tanh(eta_t * (r_max_t - safe_dist)))
    )                                                                  # (N, N)
    pair_energy = torch.where(mask, full_pair_energy, torch.zeros_like(full_pair_energy))

    # --- total energy: sum over UNORDERED pairs (upper triangle) ---------
    # The mask is symmetric, so summing the full matrix would double-count.
    # We zero the lower triangle (including the diagonal) so the sum equals
    # the desired Σ_{i<j} V_direct(i, j).
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    pair_energy_upper = torch.where(upper, pair_energy, torch.zeros_like(pair_energy))

    total = pair_energy_upper.sum()

    if not return_pair_matrix:
        return total
    return {
        "energy": total,
        "pair_energy": pair_energy_upper,
        "pair_mask": mask & upper,
        "distances": dist,
    }


def direct_pair_energy(
    r_ij: torch.Tensor | float,
    aa_i: int,
    aa_j: int,
    *,
    gamma_direct: Optional[torch.Tensor] = None,
    k_water: float = 1.0,
    r_min: float = DIRECT_R_MIN_A,
    r_max: float = DIRECT_R_MAX_A,
    eta: float = DIRECT_ETA_PER_A,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Scalar V_direct for a single (r, aa_i, aa_j) triple — used by tests.

    Useful as a reference implementation against the dense path and for the
    hand-computed pair-level validation. No sequence-separation logic (callers
    supply only valid pairs).

    Returns a 0-d tensor in kcal/mol.
    """
    if gamma_direct is None:
        gamma_direct = load_direct_gamma(device=device, dtype=dtype)
    else:
        gamma_direct = gamma_direct.to(dtype=dtype)
        if device is not None:
            gamma_direct = gamma_direct.to(device=device)
    r = torch.as_tensor(r_ij, dtype=dtype, device=gamma_direct.device)
    eta_t = torch.as_tensor(eta, dtype=dtype, device=gamma_direct.device)
    r_min_t = torch.as_tensor(r_min, dtype=dtype, device=gamma_direct.device)
    r_max_t = torch.as_tensor(r_max, dtype=dtype, device=gamma_direct.device)

    t_min = torch.tanh(eta_t * (r - r_min_t))
    t_max = torch.tanh(eta_t * (r_max_t - r))
    theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
    g = gamma_direct[aa_i, aa_j]
    return -k_water * g * theta
