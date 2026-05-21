"""Shared infrastructure for the direct and water-mediated contact terms.

Both :mod:`src.direct_contact` and :mod:`src.water_mediated` build the same
``(N, N)`` effective-CB distance matrix, apply the same sequence-separation
mask, and use the same chain-index machinery. This module lifts those pieces
into a single place so a bug fix in one path (e.g. autograd NaN poisoning)
flows to both terms automatically.

The functions here are deliberately PRIVATE (leading underscore in the module
name) — the public API for users remains
:func:`src.direct_contact.direct_contact_energy` and
:func:`src.water_mediated.water_mediated_energy`. The helpers in this file are
implementation details that two terms happen to share.

Numerical notes
---------------
* ``_pairwise_distance_safe`` uses the "double-where NaN trick": before the
  ``vector_norm`` is fed to any downstream ``tanh`` we replace the NaN
  positions with a benign mid-window value, so that ``torch.where``'s
  backward pass does not multiply zero by NaN (which produces NaN gradients
  that poison every other residue sharing the gather). See the Phase 2a
  review (``docs/phase_2a_review.md``) High-priority finding #1 for the full
  rationale.
* Chain-index assignment uses a Python dict on first appearance — identical
  chain IDs map to the same integer; the order of first appearance defines
  the integer. Same as :mod:`src.burial`.

ContactContext (opt-sprint Idea 3)
-----------------------------------
When two or more of {direct, mediated, debye_huckel} are called on the same
``coords`` object, building three independent (N, N) distance matrices +
masks is wasteful. :class:`ContactContext` packages the shared scaffolding so
the three terms can re-use it. Public API is unchanged — each term function
still accepts ``coords`` directly. Callers that want the share opt in via the
private ``_context=`` kwarg (or via the :func:`build_contact_context` helper).

SparseContactContext (Speed-sprint #3 Idea 1)
---------------------------------------------
For large N (≥ ~4000) the dense (N, N) representation costs ~5 GB transient
per term at float64. The sparse representation stores only the pairs with
``r_ij < r_cutoff`` (typical sparsity ~0.5-5% of dense), reducing transient
memory by ~500x on 4PKN. :class:`SparseContactContext` holds the
1-D ``(N_pair,)`` arrays. Callers opt in by passing ``sparse_cutoff=`` to
:func:`build_contact_context`.

**Accuracy guarantee.** The sparse path is byte-exact w.r.t. the dense path:
the same set of contributing pairs is enumerated in the same order
(``(i, j)`` sorted lexicographically with ``i < j``), so ``torch.sum`` over
the 1-D pair-energy vector produces the same partial-sum tree as ``torch.sum``
over the row-major flatten of the upper-tri dense matrix.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

# --- Term-specific minimum-safe sparse cutoffs (Å) ----------------------------
# When a SparseContactContext is used, pairs beyond ``sparse_cutoff`` are
# DROPPED from the sum. For each AWSEM contact term, the integrand is
# numerically non-zero out to a longer range than the nominal "well edge":
#
# * Direct contact: nominal r_max = 6.5 Å, but the tanh-shoulder tail leaves
#   ~3 % of the well amplitude at r = 7.0 Å. We require ~7.5 Å so the dropped
#   tail is ≤ 1 % of a typical pair contribution. Empirically verified on
#   5AON: cutoff=6.5 → 0.51 kcal/mol drift (HIGH), cutoff=7.5 → < 0.01
#   kcal/mol drift.
# * Water-mediated: nominal r_max = 9.5 Å. The mediated tail decays more
#   slowly than direct because the burial-blended γ stays ~O(1) and the
#   sigmoid switches off late. 5AON: cutoff=9.5 → 0.46 kcal/mol drift
#   (HIGH), cutoff=14.0 → matches dense to < 0.01 kcal/mol.
# * Debye-Hückel: integrand ``exp(-r / λ)/r`` with λ=10 Å (default). At
#   r = 30 Å the decay is exp(-3) ≈ 0.05 per pair, which can sum to a
#   non-trivial residual across hundreds of charged pairs. 5AON: cutoff=11
#   → sign flip (+0.15 vs −0.60), cutoff=30 → matches. The recommended
#   minimum scales linearly with screening_length (3 λ).
#
# These are emitted as ``UserWarning`` (not raised) so legacy callers that
# deliberately use a tight cutoff for a coarse estimate are not broken.
DIRECT_SPARSE_MIN_SAFE_A: float = 7.5
MEDIATED_SPARSE_MIN_SAFE_A: float = 14.0
# DH min-safe is a function of screening_length; the helper below computes it.


def dh_sparse_min_safe(screening_length: float, k_screening: float = 1.0) -> float:
    """Minimum-safe sparse_cutoff (Å) for the Debye-Hückel term.

    ``V_DH(r) = k_QQ * q_i q_j * exp(-k_screening * r / λ) / r`` decays with
    effective inverse length ``k_screening / screening_length``. We need the
    cutoff to span at least three e-folds (``exp(-3) ≈ 0.05`` per pair), and
    the integrand also carries the slower ``1/r`` factor, so the practical
    safe cutoff is **3 × λ_eff** where ``λ_eff = screening_length /
    k_screening``. With the LAMMPS defaults λ=10 Å and k=1 this yields 30 Å.
    """
    lam_eff = float(screening_length) / float(k_screening)
    return max(30.0, 3.0 * lam_eff)


def _coords_fingerprint(t: torch.Tensor) -> tuple[int, int, float, float, float, float]:
    """Cheap, side-effect-free fingerprint of a (N, 3) coord tensor.

    Used to detect "stale ContactContext" — caller built a context from one
    coords tensor and is now feeding a different coords tensor. We hash a
    handful of stable scalars: ``(N, device-hash, first row sum, last row
    sum, total sum, total absolute sum)``. Any one of these changing
    overwhelmingly implies the underlying coordinates changed. NaN-safe via
    ``nansum``.

    Returns a tuple of plain Python numbers so it is cheap to compare and is
    pickleable (frozen dataclasses).
    """
    n = int(t.shape[0])
    if n == 0:
        return (0, 0, 0.0, 0.0, 0.0, 0.0)
    with torch.no_grad():
        # Use nansum so NaN-rows do not poison the fingerprint
        first = float(torch.nansum(t[0]).item())
        last = float(torch.nansum(t[-1]).item())
        total = float(torch.nansum(t).item())
        abs_total = float(torch.nansum(t.abs()).item())
    # device tied to fingerprint so a CPU→CUDA move would invalidate too
    dev = t.device
    dev_hash = hash((dev.type, dev.index))
    return (n, dev_hash, first, last, total, abs_total)


def _validate_context_device(
    ctx: ContactContext | SparseContactContext,
    requested_device: torch.device,
) -> None:
    """Raise ValueError if the context is on a different device than requested.

    Auto-moving the context would silently allocate a duplicate of the (N, N)
    distance matrix (or N_pair sparse list) on the new device — expensive
    and confusing — so we refuse and instruct the caller to rebuild instead.
    Finding #11 — ``ContactContext device mismatch produces cryptic
    cross-device failures``.
    """
    ctx_dev = ctx.device
    req = torch.device(requested_device)
    # ``cuda`` (no index) == ``cuda:0`` for our purposes; compare both type
    # and index, treating ``index is None`` as wildcard.
    if ctx_dev.type != req.type:
        raise ValueError(
            f"ContactContext is on device {ctx_dev}, but device={req} was "
            "requested. Rebuild the context on the requested device "
            "(build_contact_context(coords.to(device=...), ...)) — auto-"
            "moving the cached (N, N) matrix would silently duplicate it."
        )
    if (
        ctx_dev.index is not None
        and req.index is not None
        and ctx_dev.index != req.index
    ):
        raise ValueError(
            f"ContactContext on {ctx_dev} but device={req} requested."
        )


def _validate_context_fingerprint(
    ctx: ContactContext | SparseContactContext,
    coords: dict[str, torch.Tensor],
) -> None:
    """Raise ValueError if the ContactContext was built from different coords.

    Without this check a caller can build a context from structure A and
    invoke an energy function with coords from structure B, silently mixing
    a stale (N, N) distance matrix with the new amino-acid identities.
    Finding #30.
    """
    fp_now = _coords_fingerprint(_resolve_contact_coords(coords, device=ctx.device))
    if ctx.fingerprint != fp_now:
        raise ValueError(
            "Stale ContactContext: its fingerprint does not match the "
            "current `coords` (N, device, or coordinate values differ). "
            "Rebuild the context with build_contact_context(coords, ...) "
            "or call the energy function without _context= (it will build "
            "a fresh context internally)."
        )


# --- DNA sentinel guard (QA-1 HIGH) ------------------------------------------
# DNA placeholder residues are encoded as ``residue_types == -1`` by the parser
# when ``include_dna=True``. Python-style negative indexing into the gamma /
# charge / burial tables silently maps these to the LAST row (VAL), producing
# biologically nonsense numbers with no warning. ``compute_frustration``
# filters DNA out via ``_subset_protein_only``; this guard only fires when a
# user calls the lower-level energy API directly with DNA-bearing data.
def _check_no_dna_sentinel(residue_types: torch.Tensor) -> None:
    """Raise if ``residue_types`` contains negative values (DNA sentinels).

    Negative indices into the (20,) / (20, 20) / (20, 3) amino-acid tables
    silently wrap to the last row in PyTorch's advanced indexing, producing
    plausible-looking but biologically meaningless energies. The public
    high-level driver (:func:`src.compute_frustration.compute_frustration`)
    strips DNA rows before any math; this guard catches direct callers of
    :func:`burial_energy` / :func:`direct_contact_energy` /
    :func:`water_mediated_energy` / :func:`debye_huckel_energy` who forget.

    See ``docs/qa1_core_math.md`` — HIGH severity finding.
    """
    if residue_types.numel() and (residue_types < 0).any():
        raise ValueError(
            "residue_types contains negative values (likely DNA sentinel "
            "from include_dna=True). Filter via _subset_protein_only or use "
            "compute_frustration() which handles this."
        )


@dataclass(frozen=True)
class ContactContext:
    """Pre-computed (N, N) scaffolding shared across the three contact terms.

    Built once via :func:`build_contact_context`; passed to
    :func:`src.direct_contact.direct_contact_energy`,
    :func:`src.water_mediated.water_mediated_energy`,
    :func:`src.debye_huckel.debye_huckel_energy` via the private
    ``_context=`` kwarg.

    Fields
    ------
    cb_or_ca : (N, 3) tensor — effective-CB coordinates (CA-substituted for GLY).
    chain_idx : (N,) int64 — contiguous chain index.
    geom_mask_min_sep : dict[int → (N, N) bool] — geometry-only pair mask
        keyed by ``contact_min_seq_sep``. Includes cross-chain pass-through
        + same-chain seq-sep filter + valid-row + self-exclusion. NOT a
        distance cutoff (that's term-specific).
    dist : (N, N) tensor — raw symmetric pairwise distance (may contain NaN).
    n : int — number of residues.
    device, dtype : torch.device, torch.dtype

    Speed-fix4 SPEED-2 Idea 2 — optional ``dist_full`` for the decoy samplers
    -----------------------------------------------------------------------
    Configurational / singleresidue / mutational decoys each independently
    build the same NaN-poisoned (N, N) distance matrix (NaN rows → 1e6 in
    coords, then ``+inf`` after the norm). When the caller already paid that
    cost for the dense contact terms, we can hand the prebuilt matrix
    through via :attr:`dist_full` instead of rebuilding. Strictly OPTIONAL —
    a value of ``None`` means the decoy callers fall back to their own
    builds, keeping the no-context path bit-identical.
    """

    cb_or_ca: torch.Tensor
    chain_idx: torch.Tensor
    geom_mask_min_sep: dict[int, torch.Tensor]
    dist: torch.Tensor
    n: int
    device: torch.device
    dtype: torch.dtype
    # Finding #30 — stale-context guard. Fingerprint of the cb_or_ca tensor
    # used to build this context. Each energy function compares its live
    # `coords` against this and raises ValueError on mismatch.
    fingerprint: tuple[int, int, float, float, float, float] = (0, 0, 0.0, 0.0, 0.0, 0.0)
    # Speed-fix4 SPEED-2 Idea 2: NaN-poisoned (N, N) distance matrix for decoy
    # callers. Same construction as decoys.py:362-378 / mutational_decoys.py:415-420
    # / singleresidue_decoys.py:283-292 (replace NaN rows in cb_or_ca with 1e6, take
    # vector_norm, then force NaN-row pairs to +inf so they never satisfy the
    # ``< cutoff`` test). Built ONCE per structure when ``compute_dist_full=True``
    # is passed to :func:`build_contact_context`. Default ``None`` (off) keeps
    # the no-context behaviour bit-identical and avoids the memory cost for
    # callers that only need direct/mediated/DH.
    dist_full: torch.Tensor | None = None


@dataclass(frozen=True)
class SparseContactContext:
    """1-D sparse pair list — Speed sprint Idea 1.

    Holds only pairs with ``r_ij < sparse_cutoff``, lexicographically sorted
    with ``i < j``. All downstream tensors (``theta``, ``gamma_pair``,
    ``pair_energy``, ...) are 1-D ``(N_pair,)``.

    The sparse path is byte-exact w.r.t. the dense path: the addends are
    identical and the sum order is identical (``upper-tri row-major`` ==
    ``(i, j)`` lex sort).

    Fields
    ------
    cb_or_ca : (N, 3) — effective-CB coordinates (CA-substituted for GLY).
    chain_idx : (N,) int64 — contiguous chain index (kept for callers).
    n : int — number of residues.
    pair_i, pair_j : (N_pair,) int64 — i and j indices of every pair within
        ``sparse_cutoff``, sorted with ``i < j`` lex.
    r_ij : (N_pair,) — Euclidean distance between residues i and j.
    same_chain : (N_pair,) bool — True iff ``chain_idx[i] == chain_idx[j]``.
    seq_diff : (N_pair,) int64 — ``|i - j|`` (for the per-term seq-sep gate).
    pair_mask_min_sep : dict[int → (N_pair,) bool] — pre-computed sequence-
        separation mask keyed by ``contact_min_seq_sep`` (True iff the pair
        passes the gate ``(cross-chain) OR (same-chain AND seq_diff >= sep)``).
    dist : (N, N) — raw symmetric distance matrix, kept for the
        ``return_pair_matrix=True`` callers that need the full matrix for
        diagnostics. May contain NaN. Detached.
    sparse_cutoff : float — radius used during the scan, in Å.
    device, dtype : torch.device, torch.dtype
    """

    cb_or_ca: torch.Tensor
    chain_idx: torch.Tensor
    n: int
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    r_ij: torch.Tensor
    same_chain: torch.Tensor
    seq_diff: torch.Tensor
    pair_mask_min_sep: dict[int, torch.Tensor]
    sparse_cutoff: float
    device: torch.device
    dtype: torch.dtype
    # Finding #30 — stale-context guard (same semantics as ContactContext).
    fingerprint: tuple[int, int, float, float, float, float] = (0, 0, 0.0, 0.0, 0.0, 0.0)
    # Finding #42 — the dense ``(N, N)`` distance matrix used to be cached
    # here for diagnostic ``return_pair_matrix=True`` callers. That defeats
    # the memory-saving purpose of the sparse path. We now keep ``None`` by
    # default; callers asking for the dense matrix in sparse + diagnostic
    # mode get an explicit error. The field is kept for back-compat shape so
    # ``hasattr(ctx, 'dist')`` still answers truthfully.
    dist: torch.Tensor | None = None


def build_contact_context(
    coords: dict[str, torch.Tensor],
    *,
    seq_seps: list[int] | None = None,
    device: torch.device | None = None,
    sparse_cutoff: float | None = None,
    use_cdist: bool = False,
    compute_dist_full: bool = False,
) -> ContactContext | SparseContactContext:
    """Build the shared (N, N) or sparse-list scaffolding once.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`.
    seq_seps : list[int], optional
        The set of ``contact_min_seq_sep`` values to pre-cache masks for.
        Default ``[2]`` covers the Water-block default (DH at sep=1 builds
        its mask separately on demand — adding sep=1 by default would
        always allocate an extra (N, N) bool even for single-term callers).
    device : torch.device, optional
        Destination device. Defaults to ``coords["ca_coords"].device``.
    sparse_cutoff : float, optional
        If ``None`` (default) returns a dense :class:`ContactContext`.
        If a positive Å radius, returns a :class:`SparseContactContext`
        containing only pairs with ``r_ij < sparse_cutoff``. Typical values:
        ``9.5`` (mediated shell) or ``30.0`` (DH shell — about ``3 × λ``).
    use_cdist : bool
        If ``True`` use :func:`torch.cdist` to build the initial distance
        scan (drops the ``(N, N, 3)`` ``diff`` intermediate, ~3× memory win
        in the scan). Default ``False`` for byte-exact parity with the
        chunked-broadcast path.
    compute_dist_full : bool, optional
        Speed-fix4 SPEED-2 Idea 2. If ``True``, attach a NaN-poisoned
        ``(N, N)`` distance matrix at :attr:`ContactContext.dist_full` so
        the decoy samplers can re-use it. Only honoured on the dense path
        (``sparse_cutoff is None``); ignored on the sparse path. Default
        ``False`` keeps the legacy contract — no extra allocation.

    Returns
    -------
    ContactContext (sparse_cutoff is None) — frozen dataclass; safe to share
    between term calls.

    SparseContactContext (sparse_cutoff is not None) — 1-D pair list. Pass
    via the ``_context=`` kwarg of the energy functions.
    """
    if seq_seps is None:
        seq_seps = [2]
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    cb_or_ca = _resolve_contact_coords(coords, device=device)
    dtype = cb_or_ca.dtype
    n = cb_or_ca.shape[0]
    chain_idx = _build_chain_index(coords["chain_ids"], device=device)

    # Finding #30 — fingerprint pinned to the cb_or_ca tensor at build time.
    fp = _coords_fingerprint(cb_or_ca)

    if sparse_cutoff is None:
        # Dense path — original ContactContext behavior. Always materialise
        # the full (N, N) distance matrix because the geom_mask + dist_full
        # consumers downstream rely on it.
        with torch.no_grad():
            if use_cdist:
                dist = _pairwise_distance_cdist(cb_or_ca)
            else:
                diff_raw = cb_or_ca.unsqueeze(0) - cb_or_ca.unsqueeze(1)
                dist = torch.linalg.vector_norm(diff_raw, dim=-1)

        geom_masks: dict[int, torch.Tensor] = {}
        for sep in seq_seps:
            geom_masks[int(sep)] = _pair_mask(cb_or_ca, chain_idx, int(sep))

        # Speed-fix4 SPEED-2 Idea 2: optionally produce the NaN-poisoned
        # (N, N) distance matrix that the decoy samplers want. Identical
        # construction to `_enumerate_native_pairs` / the inline blocks in
        # `decoys.py` and `singleresidue_decoys.py`.
        dist_full_opt: torch.Tensor | None = None
        if compute_dist_full:
            with torch.no_grad():
                finite_row = torch.isfinite(cb_or_ca).all(dim=-1, keepdim=True)
                safe_cb = torch.where(
                    finite_row, cb_or_ca, torch.full_like(cb_or_ca, 1.0e6)
                )
                diff_df = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)
                df = torch.linalg.vector_norm(diff_df, dim=-1)
                finite_pair = finite_row & finite_row.transpose(0, 1)
                dist_full_opt = torch.where(
                    finite_pair, df, torch.full_like(df, float("inf"))
                )

        return ContactContext(
            cb_or_ca=cb_or_ca,
            chain_idx=chain_idx,
            geom_mask_min_sep=geom_masks,
            dist=dist,
            n=n,
            device=device,
            dtype=dtype,
            fingerprint=fp,
            dist_full=dist_full_opt,
        )

    # ---- Sparse path -------------------------------------------------------
    # Finding #42 — we used to build the full (N, N) distance matrix first
    # and then slice. That defeats the sparse path's memory benefit. We now
    # build pair distances ONLY for pairs inside ``sparse_cutoff`` using a
    # chunked scan over ``i``. Pairs beyond the cutoff are never enumerated.
    #
    # Finding #56 — guard against NaN / inf / non-positive cutoff. Negative
    # was already rejected; NaN slipped past because ``NaN <= 0`` is False,
    # and ``inf`` would silently keep every pair.
    cutoff_f = float(sparse_cutoff)
    if not math.isfinite(cutoff_f):
        raise ValueError(
            f"sparse_cutoff must be a finite positive value, got {cutoff_f}"
        )
    if cutoff_f <= 0.0:
        raise ValueError(f"sparse_cutoff must be > 0, got {cutoff_f}")

    # Chunked scan: walk ``i`` rows, compute distances from row ``i`` to the
    # tail ``j ≥ i+1`` (already upper-tri), keep ``r < cutoff``. Peak memory
    # is ``O(chunk × N × 3)`` for the diff intermediate — bounded — never
    # ``O(N²)``. We keep chunk_size = 256 which is friendly to GPU + CPU
    # cache lines and tested for byte-exact parity with the legacy dense
    # path.
    with torch.no_grad():
        valid_row_full = torch.isfinite(cb_or_ca).all(dim=-1)       # (N,)
        i_list: list[torch.Tensor] = []
        j_list: list[torch.Tensor] = []
        chunk_size = 256
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            # Rows i in [start, stop). For upper-tri, j > i, so we slice
            # the j-tail per row. We compute the full (chunk, N) block then
            # zero out j ≤ i positions via a triu condition.
            block_i = cb_or_ca[start:stop]                          # (c, 3)
            diff_block = block_i.unsqueeze(1) - cb_or_ca.unsqueeze(0)  # (c, N, 3)
            r_block = torch.linalg.vector_norm(diff_block, dim=-1)     # (c, N)
            # j_idx > i for upper-tri.
            i_idx_local = torch.arange(start, stop, device=device).unsqueeze(1)   # (c, 1)
            j_idx_local = torch.arange(n, device=device).unsqueeze(0)             # (1, N)
            upper_local = j_idx_local > i_idx_local                              # (c, N)
            valid_block = (
                valid_row_full[start:stop].unsqueeze(1)
                & valid_row_full.unsqueeze(0)
            )                                                                    # (c, N)
            within = (r_block < cutoff_f) & upper_local & valid_block
            idx_local = within.nonzero(as_tuple=False)                           # (k, 2)
            if idx_local.numel() > 0:
                i_list.append(idx_local[:, 0] + start)
                j_list.append(idx_local[:, 1])
        if i_list:
            pair_i = torch.cat(i_list).contiguous()
            pair_j = torch.cat(j_list).contiguous()
        else:
            pair_i = torch.empty((0,), dtype=torch.int64, device=device)
            pair_j = torch.empty((0,), dtype=torch.int64, device=device)

    # Distances and chain/seq info for the selected pairs. Built OUTSIDE the
    # no_grad block so autograd flows from coords → r_ij → downstream energies.
    diff_pair = cb_or_ca[pair_i] - cb_or_ca[pair_j]              # (N_pair, 3)
    r_ij = torch.linalg.vector_norm(diff_pair, dim=-1)           # (N_pair,)

    same_chain = chain_idx[pair_i] == chain_idx[pair_j]          # (N_pair,)
    seq_diff = (pair_i - pair_j).abs()                           # (N_pair,)

    pair_masks: dict[int, torch.Tensor] = {}
    for sep in seq_seps:
        s = int(sep)
        # Same logic as ``_pair_mask``: cross-chain always passes, same-chain
        # requires ``|i - j| >= sep``. (Self-pairs and NaN rows are already
        # excluded by construction of ``pair_i < pair_j`` + ``finite_pair``.)
        pair_masks[s] = (~same_chain) | (seq_diff >= s)

    return SparseContactContext(
        cb_or_ca=cb_or_ca,
        chain_idx=chain_idx,
        n=n,
        pair_i=pair_i,
        pair_j=pair_j,
        r_ij=r_ij,
        same_chain=same_chain,
        seq_diff=seq_diff,
        pair_mask_min_sep=pair_masks,
        sparse_cutoff=cutoff_f,
        device=device,
        dtype=dtype,
        fingerprint=fp,
        # Finding #42 — dense (N, N) `dist` intentionally not cached. Callers
        # that need the full matrix should rebuild a dense ContactContext.
        dist=None,
    )


# --- Coordinate resolution ----------------------------------------------------
def _resolve_contact_coords(
    coords: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return effective-CB coordinates (CA substituted where CB is NaN).

    Mirrors :func:`src.burial._resolve_density_coords`. Glycine has no CB so
    its row in ``cb_coords`` is NaN; we fall back to CA. Any other residue
    whose CB is NaN (rare, but possible for badly-truncated PDB inputs) also
    falls back to CA. Residues with NaN in BOTH CA and CB remain NaN — they
    are flagged invalid by ``valid_row`` in :func:`_pair_mask`.

    Finding #14 — when a non-glycine residue's CB is missing AND we fall
    back to CA, the contact distance shifts by ~1.5 Å (the CA-CB bond
    length) which alters native-pair enumeration and FI. Emit a
    ``UserWarning`` listing the affected residue indices so the caller knows
    to preprocess with PDBFixer or accept the bias. Glycine fallback is
    silent (it is the documented LAMMPS-AWSEM behaviour). The check uses
    ``residue_types`` if available; if absent we skip the warning (rare
    direct-callsite path).

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``
        and ``cb_coords``.
    device : torch.device, optional
        Destination device. Default keeps the input device.

    Returns
    -------
    (N, 3) tensor of effective-CB coordinates.
    """
    cb = coords["cb_coords"]
    ca = coords["ca_coords"]
    if device is not None:
        cb = cb.to(device=device)
        ca = ca.to(device=device)
    nan_row = ~torch.isfinite(cb).all(dim=-1, keepdim=True)        # (N, 1)

    # Finding #14 — warn (once per call) on non-Gly CB fallback.
    # Glycine has residue_type index 7 in the OpenAWSEM gamma order
    # (A R N D C Q E G ...). ``residue_types`` may be absent from
    # synthetic test dicts; tolerate that quietly.
    rt = coords.get("residue_types") if isinstance(coords, dict) else None
    if rt is not None:
        try:
            nan_row_1d = nan_row.squeeze(-1)
            if nan_row_1d.any():
                rt_dev = rt.to(device=nan_row_1d.device)
                # We only complain when residue_type is in [0, 20) AND not GLY.
                # DNA sentinel (-1) and out-of-range values are policed by the
                # DNA-sentinel guard / finding-#39 checks elsewhere.
                in_range = (rt_dev >= 0) & (rt_dev < 20)
                non_gly = rt_dev != 7
                bad = nan_row_1d & in_range & non_gly
                n_bad = int(bad.sum().item())
                if n_bad > 0:
                    idxs = torch.nonzero(bad, as_tuple=False).flatten().tolist()
                    preview = idxs[:5]
                    more = "" if n_bad <= 5 else f" (+{n_bad - 5} more)"
                    warnings.warn(
                        f"Effective-CB resolution: {n_bad} non-glycine "
                        f"residue(s) have missing/NaN CB and were silently "
                        f"substituted with CA. Indices: {preview}{more}. "
                        "This shifts the contact geometry by ~1.5 Å (CA-CB "
                        "bond length) and can alter native-pair enumeration "
                        "and FI. Preprocess with PDBFixer or accept the "
                        "bias.",
                        UserWarning,
                        stacklevel=3,
                    )
        except Exception:
            # Never let the diagnostic warning crash the energy path.
            pass

    return torch.where(nan_row, ca, cb)


# --- Chain index --------------------------------------------------------------
def _build_chain_index(
    chain_ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Convert string chain IDs to a contiguous (N,) int tensor.

    Identical chain IDs map to the same integer; the order of first appearance
    defines the integer. Matches :func:`src.burial.burial_density` and
    :func:`src.direct_contact._build_chain_index` (which this replaces).
    """
    cid_map: dict[str, int] = {}
    chain_index_list: list[int] = []
    for c in chain_ids:
        if c not in cid_map:
            cid_map[c] = len(cid_map)
        chain_index_list.append(cid_map[c])
    return torch.tensor(chain_index_list, dtype=torch.int64, device=device)


# --- NaN-safe pairwise distance ----------------------------------------------
def _pairwise_distance_cdist(safe_cb: torch.Tensor) -> torch.Tensor:
    """`torch.cdist`-based distance — drops the (N, N, 3) ``diff`` intermediate.

    On CUDA, ``cdist`` uses a block-matmul kernel which reorders ADDITIONS
    inside the squared-distance sum, so the result can drift by 1 ULP from
    the explicit ``diff.norm`` form. **Gate every use of this behind a per-
    pair drift test** — see ``test_direct_contact.py::test_cdist_drift``.

    Caller is responsible for any NaN-row sanitization BEFORE calling this.
    """
    return torch.cdist(safe_cb, safe_cb, p=2)


def _pairwise_distance_safe(
    cb_coords: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float,
    *,
    use_cdist: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (N, N) pairwise distance + an autograd-safe variant.

    The autograd-safe ``safe_dist`` replaces masked-out / NaN positions with
    a finite mid-window value BEFORE downstream non-linearities. This is the
    "double-where NaN trick" — without it, gradients flow as ``0 * NaN = NaN``
    through ``vector_norm``'s backward pass (which computes ``diff / norm``,
    where both operands carry NaN), and that poisons upstream tensors.

    The fix has TWO layers:

    1. **Sanitise coordinates BEFORE building ``diff``.** We replace NaN rows
       in ``cb_coords`` with a far-away finite point so that
       ``vector_norm`` never sees NaN. The original (NaN-bearing) ``cb_coords``
       is retained for the ``dist`` return, but the sanitised version is used
       to build ``safe_dist`` whose gradient flows back to the original coords.
    2. **Double-where on the distance.** After ``safe_dist`` is built we apply
       a second mask using ``torch.where`` so that masked-out positions are
       replaced with the benign ``fill_value`` for downstream ``tanh``.

    Together, layer (1) prevents NaN from entering the backward graph at all,
    and layer (2) keeps ``tanh`` and friends well-conditioned even on
    cross-chain / seq-sep-skipped pairs.

    Parameters
    ----------
    cb_coords : (N, 3) tensor
        Effective-CB coordinates (already CA-substituted for GLY/missing CB).
    mask : (N, N) bool tensor
        True where the pair is valid (in-window, finite, seq-sep ok). The
        distance returned in ``safe_dist`` is unchanged on True entries and
        replaced with ``fill_value`` on False entries.
    fill_value : float
        Benign value used to replace masked-out distances. Typical choice is
        the mid-point of the sigmoid window (``0.5 * (r_min + r_max)``) so
        that ``tanh`` evaluates to a smooth ~0 contribution.
    use_cdist : bool
        If ``True`` use :func:`torch.cdist` instead of the explicit
        ``(N, N, 3)`` broadcast. Drops the 1.81 GB ``diff`` intermediate on
        4PKN at the cost of a possible 1-ULP drift from block-matmul kernel
        re-ordering. Default ``False`` for byte-exact parity. Gate behind
        the per-pair drift tests in ``test_direct_contact.py``.

    Returns
    -------
    dist : (N, N) tensor
        Raw pairwise distance — may contain NaN entries where coords were
        NaN. Useful for diagnostics / the ``"distances"`` return field.
        Detached from the autograd graph.
    safe_dist : (N, N) tensor
        NaN-free version. Use this as the input to any ``tanh``/``sigmoid``
        downstream. Autograd-safe — gradients flow back through the FINITE
        rows of ``cb_coords`` only.
    """
    # ----- LAYER 1: sanitize cb_coords for the autograd-safe path --------
    # NaN rows are replaced with a large finite "decoy" position (distant
    # enough that even after subtraction the result is a normal finite
    # number, not subnormal/overflow). Using `where` keeps autograd flowing
    # to the FINITE entries (gradient on NaN rows is zero, which is fine).
    finite_row = torch.isfinite(cb_coords).all(dim=-1, keepdim=True)   # (N, 1)
    decoy = torch.full_like(cb_coords, 1.0e6)                          # safely far
    safe_cb = torch.where(finite_row, cb_coords, decoy)

    if use_cdist:
        # cdist drops the (N, N, 3) intermediate.
        safe_dist_raw = _pairwise_distance_cdist(safe_cb)              # (N, N), finite
    else:
        # diff[i, j, :] = safe_cb[i] - safe_cb[j]  → all finite
        diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)             # (N, N, 3)
        safe_dist_raw = torch.linalg.vector_norm(diff, dim=-1)         # (N, N), finite

    # ----- LAYER 2: apply the pair mask on top --------------------------
    # Where the pair is invalid (e.g. it touches a NaN row), substitute the
    # benign fill value so the downstream tanh evaluates to ~0 contribution.
    fill = torch.full_like(safe_dist_raw, fill_value)
    safe_dist = torch.where(mask, safe_dist_raw, fill)

    # ----- Diagnostic `dist`: keep NaN visibility for the user-facing dict ----
    # This branch is detached from the graph — no gradient flows here.
    with torch.no_grad():
        diff_raw = cb_coords.unsqueeze(0) - cb_coords.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff_raw, dim=-1)              # may contain NaN

    return dist, safe_dist


# --- Pair mask ----------------------------------------------------------------
def _pair_mask(
    cb_coords: torch.Tensor,
    chain_idx: torch.Tensor,
    contact_min_seq_sep: int,
) -> torch.Tensor:
    """Build the (N, N) "this pair contributes" boolean mask.

    Same logic as ``fix_backbone.cpp:5048,5086``:
    * cross-chain pairs ALWAYS contribute, irrespective of sequence separation;
    * same-chain pairs require ``|i - j| >= contact_min_seq_sep``;
    * self-pairs (``i == j``) never contribute;
    * pairs involving a NaN-coordinate row never contribute.

    The mask returned here is symmetric — callers should ``& triu(..., diagonal=1)``
    if they want a strictly upper-triangular sum.

    Parameters
    ----------
    cb_coords : (N, 3) tensor
        Effective-CB coordinates.
    chain_idx : (N,) int tensor
        Output of :func:`_build_chain_index`.
    contact_min_seq_sep : int
        Minimum same-chain sequence separation. Default ``2`` for the contact
        terms (from ``[Water]`` line ``2 2``).

    Returns
    -------
    (N, N) bool tensor — True where the pair should contribute.
    """
    device = cb_coords.device
    n = cb_coords.shape[0]

    valid_row = torch.isfinite(cb_coords).all(dim=-1)              # (N,)
    valid_pair = valid_row.unsqueeze(0) & valid_row.unsqueeze(1)   # (N, N)

    same_chain = chain_idx.unsqueeze(0) == chain_idx.unsqueeze(1)  # (N, N)
    idx = torch.arange(n, device=device)
    seq_diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()         # (N, N)
    sep_ok = (~same_chain) | (seq_diff >= contact_min_seq_sep)
    not_self = idx.unsqueeze(0) != idx.unsqueeze(1)

    return valid_pair & sep_ok & not_self


def _check_residue_types_in_range(residue_types: torch.Tensor) -> None:
    """Validate that residue_types are in ``[0, 20)``.

    Finding #39 — positive out-of-range residue_types (>= 20) used to raise a
    raw ``IndexError`` inside ``gamma[aa_i, aa_j]``; negative values (DNA
    sentinel) were already caught by :func:`_check_no_dna_sentinel`. This
    helper composes both checks into a single clear error.
    """
    if residue_types.numel() == 0:
        return
    if (residue_types < 0).any() or (residue_types >= 20).any():
        rt_min = int(residue_types.min().item())
        rt_max = int(residue_types.max().item())
        raise ValueError(
            f"residue_types must lie in [0, 20); got min={rt_min}, max={rt_max}. "
            "Values < 0 typically mean DNA sentinels (parser include_dna=True); "
            "values >= 20 mean a non-canonical residue index slipped through "
            "the parser's THREE_TO_ONE mapping."
        )


def _warn_sparse_cutoff(
    term_name: str,
    ctx: SparseContactContext,
    min_safe_a: float,
    extra_advice: str = "",
) -> None:
    """Emit a UserWarning when a SparseContactContext is too tight for a term.

    Findings #28 / #29 / #47 — different AWSEM contact terms have different
    minimum-safe sparse_cutoff values (see ``DIRECT_SPARSE_MIN_SAFE_A`` etc.
    in this module). When a user reuses a tight context (built for the
    direct shell or burial scan, say) for a longer-range term (water-
    mediated or DH), the dropped tail accumulates to a material kcal/mol
    drift — sometimes a sign flip on DH. We refuse to do this silently.
    """
    if ctx.sparse_cutoff < min_safe_a:
        advice = (
            f"Rebuild via build_contact_context(coords, sparse_cutoff={min_safe_a}). "
        )
        if extra_advice:
            advice += extra_advice
        warnings.warn(
            f"{term_name}: SparseContactContext was built with sparse_cutoff="
            f"{ctx.sparse_cutoff} Å, which is below the recommended minimum "
            f"{min_safe_a} Å for this term. Pairs in the tail beyond the "
            "cutoff are dropped from the sum and the energy can drift by "
            "significant fractions of a kcal/mol (and in the DH case can "
            f"sign-flip). {advice}",
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "_resolve_contact_coords",
    "_build_chain_index",
    "_pairwise_distance_safe",
    "_pairwise_distance_cdist",
    "_pair_mask",
    "_check_no_dna_sentinel",
    "_check_residue_types_in_range",
    "_validate_context_device",
    "_validate_context_fingerprint",
    "_warn_sparse_cutoff",
    "ContactContext",
    "SparseContactContext",
    "build_contact_context",
    "DIRECT_SPARSE_MIN_SAFE_A",
    "MEDIATED_SPARSE_MIN_SAFE_A",
    "dh_sparse_min_safe",
]
