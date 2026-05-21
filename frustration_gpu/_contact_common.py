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

from dataclasses import dataclass

import torch


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
    dist: torch.Tensor
    sparse_cutoff: float
    device: torch.device
    dtype: torch.dtype


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

    # Raw symmetric distance — single allocation, shared.
    with torch.no_grad():
        if use_cdist:
            dist = _pairwise_distance_cdist(cb_or_ca)
        else:
            diff_raw = cb_or_ca.unsqueeze(0) - cb_or_ca.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff_raw, dim=-1)

    if sparse_cutoff is None:
        # Dense path — original ContactContext behavior.
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
            dist_full=dist_full_opt,
        )

    # ---- Sparse path -------------------------------------------------------
    # One dense scan (in no_grad to free as quickly as possible) extracts the
    # pairs within the cutoff. NaN-row sanitization is applied so the scan
    # itself doesn't trip on missing CB coords.
    cutoff_f = float(sparse_cutoff)
    if cutoff_f <= 0.0:
        raise ValueError(f"sparse_cutoff must be > 0, got {cutoff_f}")

    with torch.no_grad():
        valid_row = torch.isfinite(cb_or_ca).all(dim=-1)        # (N,)
        # Upper-tri ``i < j`` selection, distance below cutoff, both rows finite.
        triu_mask = torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=device),
            diagonal=1,
        )
        finite_pair = valid_row.unsqueeze(0) & valid_row.unsqueeze(1)
        within_cutoff = (dist < cutoff_f) & finite_pair & triu_mask
        # ``nonzero`` gives row-major (= lex) order, which is exactly the
        # ``i < j`` upper-tri row-major order we need for sum-ordering parity.
        idx = within_cutoff.nonzero(as_tuple=False)             # (N_pair, 2)
        pair_i = idx[:, 0].contiguous()
        pair_j = idx[:, 1].contiguous()

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
        dist=dist,
        sparse_cutoff=cutoff_f,
        device=device,
        dtype=dtype,
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


__all__ = [
    "_resolve_contact_coords",
    "_build_chain_index",
    "_pairwise_distance_safe",
    "_pairwise_distance_cdist",
    "_pair_mask",
    "_check_no_dna_sentinel",
    "ContactContext",
    "SparseContactContext",
    "build_contact_context",
]
