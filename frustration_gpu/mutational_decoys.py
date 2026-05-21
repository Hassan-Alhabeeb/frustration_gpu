"""AWSEM decoy machinery — mutational mode (Phase 3b).

Reproduces ``FixBackbone::compute_decoy_ixns`` in mutational mode
(``fix_backbone.cpp:5249-5328``) and ``compute_native_ixn`` in mutational
mode (``:5215-5245``) in pure PyTorch.

Mutational mode
---------------
For each native pair (i, j) the frustration index is computed against an
ensemble of 1000 decoys that:

* **Hold** (r_ij, rho_i, rho_j) at the NATIVE values.
* **Scramble** only the amino-acid identities (aa_i, aa_j).

The native AND decoy per-pair energies BOTH include cross-term water
contributions with every third residue k that is within
``tert_frust_cutoff = 9.5 Å`` of i or j (the (i, k) and (j, k) shells).
The decoy holds aa_k at its NATIVE identity while scrambling aa_i and
aa_j — effectively asking "if (i, j) mutates, what is the energy
change including its full contact environment?"

Cross-term mask convention (verified against C++ lines 5300-5327)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The (i, k) and (j, k) cross-term loops use **only the spatial cutoff**
``r < tert_frust_cutoff = 9.5 Å``. There is no sequence-separation
filter on these third-residue contributions (contrast with the (i, j)
pair itself, which uses the standard ``|i - j| >= 2 OR cross-chain``
contact-mask from the outer iteration at ``fix_backbone.cpp:5086``).

The self/double-counting protection is just ``k != i`` and ``k != j``
(line 5302). This means the cross-term water contribution for (i, k=i±1)
IS included — the C++ does not skip nearest neighbours here.

Vectorisation strategy (hot path, ~5.5M burial evaluations on 11BG)
-------------------------------------------------------------------
A naive per-(pair, decoy, k) Python loop would be O(N_pair × N_decoys × N)
≈ 1517 × 1000 × 248 ≈ 4×10^8 ops for 11BG. We achieve the same result
in O(N² × 20) precompute + O(N_pair × N_decoys) postprocess as follows:

1. Build **per-residue per-alphabet "incoming water sum"**::

        T[i, α] = Σ_{k, r_ik<9.5, k!=i} water_pair(r_ik, α, aa_k_native,
                                                   rho_i, rho_k)

   Shape (N, 20). Computed by sweeping α ∈ {0..19} and doing a single
   (N, N) sum per α. Memory: O(N × 20).

2. Build **per-pair per-alphabet "contact contribution for k=j"**::

        U[i, j, α] = water_pair(r_ij, α, aa_j_native, rho_i, rho_j) *
                     1{r_ij < 9.5 AND i != j}

   We compute this on-the-fly per pair to avoid the (N, N, 20) memory
   cost (~10 MB at float64 for 11BG — fine, but unnecessary).

3. For each native pair (i, j), the decoy energy is::

        cross_i[d] = T[i, α_i[d]] - U[i, j, α_i[d]]
        cross_j[d] = T[j, α_j[d]] - U[j, i, α_j[d]]
        pair[d]    = water_pair(r_ij, α_i[d], α_j[d], rho_i, rho_j)
        burial[d]  = burial_pair(α_i[d], rho_i) + burial_pair(α_j[d], rho_j)
        E_decoy(i, j, d) = pair[d] + cross_i[d] + cross_j[d] + burial[d]

   The native energy uses the same identity but with α set to the native
   aa indices — equivalent to::

        E_native(i, j) = S_i_native + S_j_native - W_full_native[i, j]
                       + burial(aa_i, rho_i) + burial(aa_j, rho_j)

   where ``S_i_native = Σ_{k, r_ik<9.5} water_pair(r_ik, aa_i, aa_k, rho_i, rho_k)``.

Burial cross-term subtlety
~~~~~~~~~~~~~~~~~~~~~~~~~~
The C++ ``compute_native_ixn`` for mutational mode (line 5215-5245) adds
``burial_energy_i`` (= burial(aa_i, rho_i)) and ``burial_energy_j``
exactly ONCE each — the burial energies of the third residues k are
NOT added (the cross loop only updates ``water_energy``). We follow this
exactly.

Per-pair (decoy_mean, decoy_std) → not cached
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because the cross-term sums depend on which residue is the "anchor" (i
or j) — i.e. ``T[i, α]`` differs from ``T[j, α]`` since the neighbour
sets and rho values differ — every (i, j) pair gets a fresh
``(decoy_mean, decoy_std)``. This is the headline difference from
configurational mode and matches the LAMMPS dump
``mutational/<PDB>_tertiary_frustration.dat`` schema.

RNG seeding
~~~~~~~~~~~
Single CPU ``torch.Generator`` seeded at start; same PRNG sequence as
configurational mode for deterministic cross-mode comparisons. We DO
NOT match the C++ libc ``rand()`` sequence — the per-pair noise floor
between our stats and LAMMPS' is ~5% (compounds the ~3% configurational
floor with the per-pair variance amplification).

LOC budget: ~400-500 lines + this docstring.
"""
from __future__ import annotations

import torch

import warnings

from ._contact_common import (
    _build_chain_index,
    _check_no_dna_sentinel,
    _resolve_contact_coords,
)
from .decoys import (
    DEFAULT_CONTACT_CUTOFF_A,
    DEFAULT_N_DECOYS,
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
    lammps_dump_rho,
    water_theta,
)
from .parameters import BURIAL_KAPPA, BURIAL_RHO_MAX, BURIAL_RHO_MIN

# Same as the contact-term default (``[Water]`` line "2 2" → |i - j| >= 2).
PAIR_MIN_SEQ_SEP: int = 2


# ---------------------------------------------------------------------------
# Helpers — vectorised water-pair energy + burial-residue energy
# ---------------------------------------------------------------------------

def _water_pair_full(
    r: torch.Tensor,
    aa_i: torch.Tensor,
    aa_j: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    gamma_direct: torch.Tensor,
    gamma_med_prot: torch.Tensor,
    gamma_med_wat: torch.Tensor,
    *,
    direct_r_min: float,
    direct_r_max: float,
    mediated_r_min: float,
    mediated_r_max: float,
    eta: torch.Tensor,
    eta_sigma: torch.Tensor,
    rho_0: torch.Tensor,
    k_water: torch.Tensor,
) -> torch.Tensor:
    """Per-element water-pair energy ``-k_water·(γ_dir·θ_dir + γ_med·θ_med)``.

    All inputs are broadcast-compatible. The shapes used in this module:

    * ``r``, ``rho_i``, ``rho_j``: typically (N, N) or (n_decoys,)
    * ``aa_i``, ``aa_j``: int64 tensors broadcastable to the same shape

    Returns a tensor of the same broadcast shape as the inputs.

    The formula matches :func:`src.water_mediated.water_mediated_pair_energy`
    + :func:`src.direct_contact.direct_pair_energy` collapsed into one
    expression (single sigmoid per shell, no cutoff masks applied here —
    the caller masks downstream).
    """
    g_dir = gamma_direct[aa_i, aa_j]
    g_mp = gamma_med_prot[aa_i, aa_j]
    g_mw = gamma_med_wat[aa_i, aa_j]

    theta_direct = water_theta(r, direct_r_min, direct_r_max, eta)
    theta_med = water_theta(r, mediated_r_min, mediated_r_max, eta)

    sigma_wat = (
        0.25
        * (1.0 - torch.tanh(eta_sigma * (rho_i - rho_0)))
        * (1.0 - torch.tanh(eta_sigma * (rho_j - rho_0)))
    )
    sigma_prot = 1.0 - sigma_wat
    sigma_gamma_direct = g_dir
    sigma_gamma_med = sigma_prot * g_mp + sigma_wat * g_mw

    return -k_water * (sigma_gamma_direct * theta_direct + sigma_gamma_med * theta_med)


# ---------------------------------------------------------------------------
# Identity-independent (r, rho)-only kernel — Opt Sprint Idea 1
# ---------------------------------------------------------------------------

def _water_rho_terms(
    r: torch.Tensor,
    rho_i: torch.Tensor,
    rho_j: torch.Tensor,
    *,
    direct_r_min: float,
    direct_r_max: float,
    mediated_r_min: float,
    mediated_r_max: float,
    eta: torch.Tensor,
    eta_sigma: torch.Tensor,
    rho_0: torch.Tensor,
):
    """Build the identity-independent (r, ρ)-only ingredients of the water pair.

    Returns
    -------
    theta_direct, theta_med, sigma_wat, sigma_prot — each broadcast-compatible
    with the inputs (typically (N, N)).

    These four tensors do NOT depend on AA identity α and so can be hoisted
    out of any sweep over α — see :func:`_water_per_alpha_fused`. Pure
    algebraic refactor of the prefix of :func:`_water_pair_full`.
    """
    theta_direct = water_theta(r, direct_r_min, direct_r_max, eta)
    theta_med = water_theta(r, mediated_r_min, mediated_r_max, eta)
    sigma_wat = (
        0.25
        * (1.0 - torch.tanh(eta_sigma * (rho_i - rho_0)))
        * (1.0 - torch.tanh(eta_sigma * (rho_j - rho_0)))
    )
    sigma_prot = 1.0 - sigma_wat
    return theta_direct, theta_med, sigma_wat, sigma_prot


def _water_per_alpha_fused(
    theta_direct: torch.Tensor,
    theta_med: torch.Tensor,
    sigma_wat: torch.Tensor,
    sigma_prot: torch.Tensor,
    aa_col: torch.Tensor,
    gamma_direct: torch.Tensor,
    gamma_med_prot: torch.Tensor,
    gamma_med_wat: torch.Tensor,
    k_water: torch.Tensor,
    *,
    alpha_chunk_size: int = 0,
) -> torch.Tensor:
    """Compute the per-α water-pair tensor of shape ``(20, N, N)``.

    Mathematically identical to looping ``α = 0..19`` and calling
    :func:`_water_pair_full` with ``aa_i = α``. By stacking the result we
    amortise PyTorch's per-op kernel-launch overhead — critical on GPU.

    The (r, ρ)-only ingredients are passed in (already broadcast to
    ``(N, N)``); aa-dependent work is just gamma-table gathers and a
    weighted sum.

    Parameters
    ----------
    theta_direct, theta_med, sigma_wat, sigma_prot : (N, N) tensors
        From :func:`_water_rho_terms`.
    aa_col : (N,) or (N, N) int64
        Column-index amino-acid identity (the "k" / "j" partner).
        If 1-D it is unsqueezed to (1, N).
    gamma_direct, gamma_med_prot, gamma_med_wat : (20, 20) tensors
    k_water : scalar tensor
    alpha_chunk_size : int, optional
        If > 0, process α in chunks of this size to bound peak memory.
        At ``0`` (default) we build the full ``(20, N, N)`` tensor in
        one shot — fine for N ≲ 1800 in float64 on a 12 GB GPU.
        Choose ``alpha_chunk_size`` such that
        ``alpha_chunk_size · N² · 8 B`` fits comfortably.

    Returns
    -------
    (20, N, N) tensor — ``out[α, i, j] = water_pair(r_ij, α, aa_col_j,
    ρ_i, ρ_j) / mask_off-masked-out``. Caller is responsible for applying
    the cross/contact mask via a single ``masked_fill`` after this call.
    """
    # aa_col shape handling
    if aa_col.dim() == 1:
        aa_col_row = aa_col.unsqueeze(0)  # (1, N)
        n = aa_col.shape[0]
    else:
        # already (N, N) — collapse to (1, N) since the value is identical along rows
        aa_col_row = aa_col[:1]  # (1, N)
        n = aa_col.shape[1]

    # Pre-multiply theta_direct by the always-1 sigma_gamma_direct factor — wait,
    # that's literally γ_dir per α. We compute as:
    #   -k_water * (γ_dir[α, aa_col] * θ_dir + (σ_prot * γ_mp[α, aa_col] + σ_wat * γ_mw[α, aa_col]) * θ_med)
    # Stack of gamma rows: gamma_direct[: , aa_col] has shape (20, n) (since we index
    # the second axis with a (n,) tensor and keep the first). We need
    # (20, 1, n) so it broadcasts against the (N, N) tensors below.
    aa_col_flat = aa_col_row.squeeze(0)  # (N,)

    # Sub-select rows: (20, N) per gamma table — broadcast to (20, 1, N).
    # Note: gamma_xxx[:, aa_col_flat] picks ALL 20 alpha rows, ALL N columns.
    g_dir_all = gamma_direct[:, aa_col_flat].unsqueeze(1)  # (20, 1, N)
    g_mp_all = gamma_med_prot[:, aa_col_flat].unsqueeze(1)  # (20, 1, N)
    g_mw_all = gamma_med_wat[:, aa_col_flat].unsqueeze(1)  # (20, 1, N)

    # Broadcast templates: (1, N, N)
    th_dir = theta_direct.unsqueeze(0)
    th_med = theta_med.unsqueeze(0)
    sw = sigma_wat.unsqueeze(0)
    sp = sigma_prot.unsqueeze(0)

    if alpha_chunk_size <= 0 or alpha_chunk_size >= 20:
        # One-shot
        sigma_gamma_med = sp * g_mp_all + sw * g_mw_all  # (20, N, N)
        out = -k_water * (g_dir_all * th_dir + sigma_gamma_med * th_med)
        return out

    # Chunked. We do NOT materialise the (20, N, N) output (would be
    # 12 GB at N=8689 float64 and OOM a 12 GB card). Instead we keep the
    # (chunk, N, N) intermediate alive only for one chunk at a time and
    # sum-reduce on the fly. Callers that want the (N, 20) column-sum
    # should request it via `return_sum_dim=2`. By default we still
    # return the full (20, N, N) — but only when chunk size covers all
    # 20 alphas in one shot (the one-shot branch above), so this branch
    # is unreachable for the "want full cube" use case.
    out = torch.empty(
        (20, n, n), dtype=theta_direct.dtype, device=theta_direct.device
    )
    for s in range(0, 20, alpha_chunk_size):
        e = min(s + alpha_chunk_size, 20)
        g_dir_c = g_dir_all[s:e]
        g_mp_c = g_mp_all[s:e]
        g_mw_c = g_mw_all[s:e]
        sigma_gamma_med_c = sp * g_mp_c + sw * g_mw_c
        out[s:e] = -k_water * (g_dir_c * th_dir + sigma_gamma_med_c * th_med)
    return out


def _water_per_alpha_fused_sum(
    theta_direct: torch.Tensor,
    theta_med: torch.Tensor,
    sigma_wat: torch.Tensor,
    sigma_prot: torch.Tensor,
    aa_col_row: torch.Tensor,
    gamma_direct: torch.Tensor,
    gamma_med_prot: torch.Tensor,
    gamma_med_wat: torch.Tensor,
    k_water: torch.Tensor,
    *,
    alpha_chunk_size: int = 0,
) -> torch.Tensor:
    """Chunked variant that returns ``(20, N)`` (sum over the last axis).

    Same math as :func:`_water_per_alpha_fused` followed by
    ``out.sum(dim=2)``, but never materialises the full ``(20, N, N)``
    tensor — peak working memory is bounded by
    ``(alpha_chunk_size, N, N)`` plus the ``(20, N)`` accumulator. This
    is what makes 4PKN-scale (N=8689, float64 ≈ 12 GB full-cube) fit on
    a 12 GB card.

    For comfortable-VRAM cases callers can set ``alpha_chunk_size=0``
    and we fall through to the one-shot path (identical math, slightly
    faster because PyTorch can fuse more aggressively).
    """
    n = theta_direct.shape[-1]
    dtype = theta_direct.dtype
    device = theta_direct.device
    aa_col_flat = aa_col_row.squeeze(0)

    g_dir_all = gamma_direct[:, aa_col_flat].unsqueeze(1)
    g_mp_all = gamma_med_prot[:, aa_col_flat].unsqueeze(1)
    g_mw_all = gamma_med_wat[:, aa_col_flat].unsqueeze(1)

    th_dir = theta_direct.unsqueeze(0)
    th_med = theta_med.unsqueeze(0)
    sw = sigma_wat.unsqueeze(0)
    sp = sigma_prot.unsqueeze(0)

    if alpha_chunk_size <= 0 or alpha_chunk_size >= 20:
        sigma_gamma_med = sp * g_mp_all + sw * g_mw_all  # (20, N, N)
        cube = -k_water * (g_dir_all * th_dir + sigma_gamma_med * th_med)
        return cube.sum(dim=2)  # (20, N)

    T_alpha_t = torch.empty((20, n), dtype=dtype, device=device)
    for s in range(0, 20, alpha_chunk_size):
        e = min(s + alpha_chunk_size, 20)
        g_dir_c = g_dir_all[s:e]
        g_mp_c = g_mp_all[s:e]
        g_mw_c = g_mw_all[s:e]
        sigma_gamma_med_c = sp * g_mp_c + sw * g_mw_c
        chunk_cube = -k_water * (g_dir_c * th_dir + sigma_gamma_med_c * th_med)
        T_alpha_t[s:e] = chunk_cube.sum(dim=2)
        del chunk_cube, sigma_gamma_med_c
    return T_alpha_t


def _choose_alpha_chunk(n: int, device: torch.device, dtype: torch.dtype) -> int:
    """Pick an alpha-chunk size that keeps the (chunk, N, N) tensor under
    ~0.5×available VRAM for CUDA devices.

    Returns ``0`` (= no chunking) if memory is comfortable.
    """
    if device.type != "cuda":
        return 0
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    full_bytes = 20 * n * n * elem_bytes
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        return 0
    # We need a few intermediates alongside the (20, N, N) tensor, so use
    # 0.25 * free as the headroom budget.
    budget = max(1, int(free * 0.25))
    if full_bytes <= budget:
        return 0
    per_alpha = elem_bytes * n * n
    if per_alpha == 0:
        return 0
    chunk = max(1, budget // per_alpha)
    return min(chunk, 20)


def _burial_residue_energy(
    aa: torch.Tensor,
    rho: torch.Tensor,
    burial_gamma: torch.Tensor,
    *,
    burial_kappa: torch.Tensor,
    burial_rho_min: tuple,
    burial_rho_max: tuple,
    k_burial: torch.Tensor,
) -> torch.Tensor:
    """Per-element burial energy ``Σ_w -0.5·k_burial·γ[aa,w]·(t_min+t_max)``.

    ``aa`` and ``rho`` must be the same shape (or broadcast to it). Returns
    a tensor of that shape. Matches ``fix_backbone.cpp:5478-5500``.

    Speed-fix4 QA-2 M-1: vectorised across the 3 burial wells (was a Python
    ``for w_idx in range(3)`` loop + ``torch.stack``). Same trick as
    ``decoys.py:_burial_total``: ``rho.unsqueeze(-1)`` broadcasts against
    the (3,) ``rho_min_t`` / ``rho_max_t`` tensors so the three wells are
    evaluated in a single fused tanh + add. Bit-identical output (the
    ``stack(wells, dim=-1)`` previously produced an identical (..., 3)
    tensor; we now build that tensor with broadcasts instead of a Python
    loop).
    """
    bg = burial_gamma[aa]                           # (..., 3)
    # Materialise the (3,) ρ-window vectors on the same device/dtype as ρ.
    rho_min_t = torch.as_tensor(
        burial_rho_min, dtype=rho.dtype, device=rho.device
    )                                               # (3,)
    rho_max_t = torch.as_tensor(
        burial_rho_max, dtype=rho.dtype, device=rho.device
    )                                               # (3,)
    rho_e = rho.unsqueeze(-1)                       # (..., 1)
    t_min = torch.tanh(burial_kappa * (rho_e - rho_min_t))    # (..., 3)
    t_max = torch.tanh(burial_kappa * (rho_max_t - rho_e))    # (..., 3)
    switch = t_min + t_max                          # (..., 3)
    return -0.5 * k_burial * (bg * switch).sum(dim=-1)


# ---------------------------------------------------------------------------
# Native pair enumeration
# ---------------------------------------------------------------------------

def _enumerate_native_pairs(
    coords: dict[str, torch.Tensor],
    *,
    contact_cutoff: float,
    pair_min_seq_sep: int,
    device: torch.device,
):
    """Find all (i, j) native pairs that pass the outer-loop filter.

    Mirrors ``fix_backbone.cpp:5076-5086``:

    * i < j
    * same-chain ⇒ ``|i - j| >= pair_min_seq_sep``
    * cross-chain ⇒ always
    * ``r_ij < contact_cutoff`` (effective-CB distance)

    Returns
    -------
    pair_i, pair_j : (N_pair,) int64
        Upper-triangular pair indices.
    r_ij_pair : (N_pair,) float
        Effective-CB distance at each pair.
    cb_or_ca : (N, 3) float
        Effective-CB coordinates (returned for downstream use).
    dist_full : (N, N) float
        Effective-CB pairwise distance matrix (NaN-safe).
    same_chain : (N, N) bool
        Same-chain mask (returned for cross-term usage).
    """
    cb_or_ca = _resolve_contact_coords(coords, device=device)
    n = cb_or_ca.shape[0]
    chain_idx = _build_chain_index(coords["chain_ids"], device=device)

    # NaN-safe pairwise distance
    finite_row = torch.isfinite(cb_or_ca).all(dim=-1, keepdim=True)
    safe_cb = torch.where(finite_row, cb_or_ca, torch.full_like(cb_or_ca, 1.0e6))
    diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)
    dist_full = torch.linalg.vector_norm(diff, dim=-1)
    finite_pair = finite_row & finite_row.transpose(0, 1)
    dist_full = torch.where(finite_pair, dist_full, torch.full_like(dist_full, float("inf")))

    same_chain = chain_idx.unsqueeze(0) == chain_idx.unsqueeze(1)
    idx = torch.arange(n, device=device)
    seq_diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    pair_mask = (
        (dist_full < contact_cutoff)
        & ((~same_chain) | (seq_diff >= pair_min_seq_sep))
        # Upper-triangular pair enumeration (i < j). Note:
        #   idx.unsqueeze(1)  → (N, 1) rows
        #   idx.unsqueeze(0)  → (1, N) cols
        # So `rows < cols` selects upper-tri (row_i < col_j); torch.nonzero
        # returns (pair_i = row, pair_j = col) with pair_i < pair_j.
        # Phase 5 P1 fix (2026-05-20): was `idx.unsqueeze(0) < idx.unsqueeze(1)`
        # which is `col < row` → lower-tri → pair_i > pair_j; that broke
        # byte-exact comparability vs LAMMPS mutational dumps which emit i<j.
        & (idx.unsqueeze(1) < idx.unsqueeze(0))
        & finite_pair
    )
    pair_i, pair_j = torch.nonzero(pair_mask, as_tuple=True)
    r_ij_pair = dist_full[pair_i, pair_j]
    return pair_i, pair_j, r_ij_pair, cb_or_ca, dist_full, same_chain


# ---------------------------------------------------------------------------
# AA index sampling (uniform residue-index → AA-composition draws)
# ---------------------------------------------------------------------------

def _sample_aa_pair_indices(
    aa: torch.Tensor,
    n_pair: int,
    n_decoys: int,
    *,
    seed: int,
    device: torch.device,
):
    """Sample (n_pair, n_decoys) pairs of decoy aa indices.

    Each entry is drawn by picking a uniform random residue index in
    [0, N) and reading off its AA — so the empirical distribution
    follows the protein's composition (NOT 1/20 uniform). Same
    convention as :func:`src.decoys.sample_configurational_decoys`.

    Returns
    -------
    aa_i_dec : (n_pair, n_decoys) int64
    aa_j_dec : (n_pair, n_decoys) int64
    """
    n = aa.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # Two independent draws for i and j slot. Sample on CPU to preserve the
    # PRNG sequence (cross-device reproducibility convention).
    idx_i = torch.randint(0, n, (n_pair, n_decoys), generator=gen)
    idx_j = torch.randint(0, n, (n_pair, n_decoys), generator=gen)
    # Speed-fix4 SPEED-1 Idea 3: skip the 1.5M-element CPU gather + the
    # ``aa.cpu()`` sync. Move the small ``aa`` tensor (N int64) once to
    # device and run the gather on-device. Integer gather → bit-identical
    # output regardless of which device the indexing runs on.
    aa_dev = aa.to(device=device, dtype=torch.int64)
    idx_i_dev = idx_i.to(device=device, non_blocking=True)
    idx_j_dev = idx_j.to(device=device, non_blocking=True)
    aa_i_dec = aa_dev[idx_i_dev]
    aa_j_dec = aa_dev[idx_j_dev]
    return aa_i_dec, aa_j_dec


# ---------------------------------------------------------------------------
# Core: vectorised native + decoy per-pair energies
# ---------------------------------------------------------------------------

def _precompute_T_alpha(
    dist_full: torch.Tensor,
    aa_native: torch.Tensor,
    rho: torch.Tensor,
    gamma_direct: torch.Tensor,
    gamma_med_prot: torch.Tensor,
    gamma_med_wat: torch.Tensor,
    *,
    contact_cutoff: float,
    direct_r_min: float,
    direct_r_max: float,
    mediated_r_min: float,
    mediated_r_max: float,
    eta: torch.Tensor,
    eta_sigma: torch.Tensor,
    rho_0: torch.Tensor,
    k_water: torch.Tensor,
) -> torch.Tensor:
    """Build the (N, 20) "incoming water-sum at residue i with decoy aa α".

    For each anchor residue ``i`` and each alphabet ``α ∈ {0..19}``::

        T[i, α] = Σ_{k != i, r_ik < cutoff} water_pair(r_ik, α, aa_k_native,
                                                       rho_i, rho_k_native)

    Computed by iterating α (only 20 iterations) — vastly cheaper than
    iterating decoys or pairs.

    Memory: O(N × 20). On 11BG (N=248) at float64: 39 KB.
    """
    n = dist_full.shape[0]
    device = dist_full.device
    dtype = dist_full.dtype

    cross_mask_neighbor = (dist_full < contact_cutoff)
    diag = torch.eye(n, dtype=torch.bool, device=device)
    cross_mask = cross_mask_neighbor & (~diag)              # (N, N) — k != i, r<cutoff

    # Opt sprint #1: hoist identity-independent (r, ρ) terms out of α-loop.
    # Broadcast rho: rho_i is the anchor (rows), rho_k is the neighbor (cols).
    # No `.contiguous()` — these tensors are consumed elementwise downstream
    # and a view is sufficient; this halves alloc for big N.
    rho_row = rho.unsqueeze(1)                              # (N, 1) view
    rho_col = rho.unsqueeze(0)                              # (1, N) view
    theta_direct, theta_med, sigma_wat, sigma_prot = _water_rho_terms(
        dist_full, rho_row, rho_col,
        direct_r_min=direct_r_min, direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min, mediated_r_max=mediated_r_max,
        eta=eta, eta_sigma=eta_sigma, rho_0=rho_0,
    )
    # Each of the four is (N, N) (broadcast result of (N, N) * (N, 1) etc.)
    theta_direct = theta_direct.expand(n, n).contiguous()
    theta_med = theta_med.expand(n, n).contiguous()
    sigma_wat = sigma_wat.expand(n, n).contiguous()
    sigma_prot = sigma_prot.expand(n, n).contiguous()

    # Pre-zero masked positions in the (r, ρ) ingredients to absorb the
    # cross-mask once, avoiding the per-α torch.where.
    zero = torch.zeros_like(theta_direct)
    theta_direct_m = torch.where(cross_mask, theta_direct, zero)
    theta_med_m = torch.where(cross_mask, theta_med, zero)

    if device.type == "cuda":
        # Opt sprint #2 (GPU): build the (20, N, N) per-α water tensor in one
        # fused op (with α-chunking when the full tensor would exceed VRAM).
        # Justified by ~30 µs/launch * O(8 ops) * 20 alpha = ms-scale overhead
        # that is amortised away.
        #
        # For N where the full (20, N, N) tensor would exceed half-VRAM we
        # use the sum-variant which never materialises the cube — peak
        # working memory becomes (chunk_size, N, N) + (20, N), bounding
        # the 4PKN-scale (N=8689) path to ~2.4 GB intermediates instead
        # of the 12 GB full cube.
        alpha_chunk = _choose_alpha_chunk(n, device, dtype)
        if alpha_chunk > 0 and alpha_chunk < 20:
            T_alpha_t = _water_per_alpha_fused_sum(
                theta_direct_m, theta_med_m, sigma_wat, sigma_prot,
                aa_native,
                gamma_direct, gamma_med_prot, gamma_med_wat,
                k_water,
                alpha_chunk_size=alpha_chunk,
            )  # (20, N)
            T_alpha = T_alpha_t.transpose(0, 1).contiguous()  # (N, 20)
            return T_alpha

        w_all = _water_per_alpha_fused(
            theta_direct_m, theta_med_m, sigma_wat, sigma_prot,
            aa_native,
            gamma_direct, gamma_med_prot, gamma_med_wat,
            k_water,
            alpha_chunk_size=alpha_chunk,
        )                                                   # (20, N, N)
        T_alpha = w_all.sum(dim=2).transpose(0, 1).contiguous()  # (N, 20)
        return T_alpha

    # CPU: no kernel-launch overhead. Loop α and accumulate one (N,) column
    # at a time. Memory peak stays at O(N²) (one slice). Still benefits from
    # Idea 1's (r, ρ)-only hoisting — that work happened above.
    aa_col_flat = aa_native  # (N,)
    g_dir_rows = gamma_direct[:, aa_col_flat]    # (20, N)
    g_mp_rows = gamma_med_prot[:, aa_col_flat]   # (20, N)
    g_mw_rows = gamma_med_wat[:, aa_col_flat]    # (20, N)

    T_alpha = torch.empty((n, 20), dtype=dtype, device=device)
    for alpha in range(20):
        g_dir = g_dir_rows[alpha].unsqueeze(0)  # (1, N) broadcasts to (N, N)
        g_mp = g_mp_rows[alpha].unsqueeze(0)
        g_mw = g_mw_rows[alpha].unsqueeze(0)
        sigma_gamma_med = sigma_prot * g_mp + sigma_wat * g_mw
        w = -k_water * (g_dir * theta_direct_m + sigma_gamma_med * theta_med_m)
        T_alpha[:, alpha] = w.sum(dim=1)
    return T_alpha


def _per_pair_U(
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    r_ij_pair: torch.Tensor,
    aa_i_dec: torch.Tensor,
    aa_j_dec: torch.Tensor,
    aa_native: torch.Tensor,
    rho: torch.Tensor,
    gamma_direct: torch.Tensor,
    gamma_med_prot: torch.Tensor,
    gamma_med_wat: torch.Tensor,
    *,
    contact_cutoff: float,
    direct_r_min: float,
    direct_r_max: float,
    mediated_r_min: float,
    mediated_r_max: float,
    eta: torch.Tensor,
    eta_sigma: torch.Tensor,
    rho_0: torch.Tensor,
    k_water: torch.Tensor,
) -> tuple:
    """Compute per-pair-per-decoy "(i,k=j)" and "(j,k=i)" contributions.

    These are subtracted out of ``T[i, α_i[d]]`` and ``T[j, α_j[d]]``
    respectively to honour ``k != j`` and ``k != i`` in the C++ inner
    loop (line 5302).

    Shapes
    ------
    pair_i, pair_j : (N_pair,)
    r_ij_pair      : (N_pair,)
    aa_i_dec       : (N_pair, n_decoys)  decoy α at slot i
    aa_j_dec       : (N_pair, n_decoys)  decoy α at slot j

    Returns
    -------
    U_iSlot_kj : (N_pair, n_decoys) — value of water_pair(r_ij, α_i[d],
                                       aa_j_native, rho_i, rho_j). The
                                       ``r_ij < contact_cutoff`` mask is
                                       guaranteed True for every native pair
                                       (the outer enumeration only emits
                                       pairs that already satisfy it; see
                                       ``_enumerate_native_pairs``).
    U_jSlot_ki : (N_pair, n_decoys) — value of water_pair(r_ij, α_j[d],
                                       aa_i_native, rho_j, rho_i). Same
                                       cutoff-by-construction.
    pair_term  : (N_pair, n_decoys) — water_pair(r_ij, α_i[d], α_j[d],
                                       rho_i, rho_j). The pair's own
                                       contribution.

    Speed-fix4 SPEED-1 Idea 2 + QA-2 M-2 + QA-2 L-4
    -----------------------------------------------
    The three calls below share IDENTICAL ``(r_ij, rho_i, rho_j)`` inputs
    (sigma_wat is symmetric in its two ρ arguments, and θ depends only on
    r). We compute ``θ_dir, θ_med, σ_wat, σ_prot`` ONCE via
    :func:`_water_rho_terms`, then do three γ-gathers + three weighted
    sums. Previously each call rebuilt the (N_pair, n_decoys) θ/σ tensors,
    paying 12 redundant tanh ops per element (≈ 4.5M extra ops on 11BG).

    Also drops the five ``.contiguous()`` broadcasts (L-4): the
    ``_water_pair_full`` body only does elementwise multiplies and
    `gamma[...]` advanced indexing, both of which accept non-contiguous
    views. At 11BG this saves ~60 MB of scratch; at 4PKN-scale projects
    to ~6 GB savings.

    The dead ``in_cutoff`` mask (L-2) is gone — the enumeration guarantees
    the cutoff already.
    """
    n_pair, n_decoys = aa_i_dec.shape

    rho_i_pair = rho[pair_i].unsqueeze(1)                   # (N_pair, 1)
    rho_j_pair = rho[pair_j].unsqueeze(1)
    aa_i_nat_pair = aa_native[pair_i].unsqueeze(1)          # (N_pair, 1)
    aa_j_nat_pair = aa_native[pair_j].unsqueeze(1)
    r_ij_b = r_ij_pair.unsqueeze(1)                         # (N_pair, 1)

    # Broadcast views — NO contiguous, NO new allocation. Each view is
    # consumed via elementwise multiply / fancy indexing, both of which
    # PyTorch dispatches happily on non-contiguous strides.
    aa_j_nat_b = aa_j_nat_pair.expand(n_pair, n_decoys)
    aa_i_nat_b = aa_i_nat_pair.expand(n_pair, n_decoys)
    rho_i_b = rho_i_pair.expand(n_pair, n_decoys)
    rho_j_b = rho_j_pair.expand(n_pair, n_decoys)
    r_ij_full = r_ij_b.expand(n_pair, n_decoys)

    # --- Hoist identity-independent (r, ρ) ingredients --------------------
    # ``sigma_wat`` is symmetric in (ρ_i, ρ_j); ``θ`` depends only on r;
    # ``σ_prot = 1 - σ_wat``. So all three downstream water_pair calls share
    # IDENTICAL θ/σ tensors. Build them ONCE.
    theta_direct, theta_med, sigma_wat, sigma_prot = _water_rho_terms(
        r_ij_full, rho_i_b, rho_j_b,
        direct_r_min=direct_r_min, direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min, mediated_r_max=mediated_r_max,
        eta=eta, eta_sigma=eta_sigma, rho_0=rho_0,
    )

    # --- Three γ-gathers, one per (α_row, α_col) endpoint pair ------------
    # Each gather produces a (N_pair, n_decoys) tensor; the weighted sum
    # ``-k * (γ_dir·θ_dir + (σ_prot·γ_mp + σ_wat·γ_mw)·θ_med)`` is then
    # computed THREE times — but each evaluation just does
    # elementwise mul/add on the shared θ/σ tensors. Bit-identical to the
    # pre-fix `_water_pair_full` calls because the same arithmetic is
    # evaluated on the same inputs at the same precision.

    # (i-slot, k=j): γ at (α_i_dec, aa_j_native)
    g_dir_i = gamma_direct[aa_i_dec, aa_j_nat_b]
    g_mp_i = gamma_med_prot[aa_i_dec, aa_j_nat_b]
    g_mw_i = gamma_med_wat[aa_i_dec, aa_j_nat_b]
    U_iSlot_kj = -k_water * (
        g_dir_i * theta_direct
        + (sigma_prot * g_mp_i + sigma_wat * g_mw_i) * theta_med
    )

    # (j-slot, k=i): γ at (α_j_dec, aa_i_native), with ρ swapped — but
    # σ_wat(ρ_i, ρ_j) is symmetric and θ depends only on r, so the same
    # θ/σ tensors apply.
    g_dir_j = gamma_direct[aa_j_dec, aa_i_nat_b]
    g_mp_j = gamma_med_prot[aa_j_dec, aa_i_nat_b]
    g_mw_j = gamma_med_wat[aa_j_dec, aa_i_nat_b]
    U_jSlot_ki = -k_water * (
        g_dir_j * theta_direct
        + (sigma_prot * g_mp_j + sigma_wat * g_mw_j) * theta_med
    )

    # Pair's own contribution: γ at (α_i_dec, α_j_dec)
    g_dir_p = gamma_direct[aa_i_dec, aa_j_dec]
    g_mp_p = gamma_med_prot[aa_i_dec, aa_j_dec]
    g_mw_p = gamma_med_wat[aa_i_dec, aa_j_dec]
    pair_term = -k_water * (
        g_dir_p * theta_direct
        + (sigma_prot * g_mp_p + sigma_wat * g_mw_p) * theta_med
    )

    return U_iSlot_kj, U_jSlot_ki, pair_term


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def mutational_decoy_stats(
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor | None = None,
    *,
    n_decoys: int = DEFAULT_N_DECOYS,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF_A,
    pair_min_seq_sep: int = PAIR_MIN_SEQ_SEP,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    k_water: float = 1.0,
    k_burial: float = 1.0,
    eta: float = WATER_ETA_PER_A,
    eta_sigma: float = WATER_ETA_SIGMA,
    rho_0: float = WATER_RHO_0,
    direct_r_min: float = DIRECT_R_MIN_A,
    direct_r_max: float = DIRECT_R_MAX_A,
    mediated_r_min: float = MEDIATED_R_MIN_A,
    mediated_r_max: float = MEDIATED_R_MAX_A,
    burial_kappa: float = BURIAL_KAPPA,
    burial_rho_min: tuple = BURIAL_RHO_MIN,
    burial_rho_max: tuple = BURIAL_RHO_MAX,
) -> dict[str, torch.Tensor]:
    """Compute per-pair (E_native, decoy_mean, decoy_std) for mutational mode.

    Output schema matches the LAMMPS ``mutational/<PDB>_tertiary_frustration.dat``
    columns: one row per native (i, j) pair with **different**
    ``(decoy_mean, decoy_std)`` per row (unlike configurational which
    caches a single pair).

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`.
    rho : (N,) tensor, optional
        LAMMPS-dump-compatible rho (``min_seq_sep = 12``). If ``None``,
        computed via :func:`src.decoys.lammps_dump_rho`.
    n_decoys : int
        Number of decoys per pair. Default 1000.
    contact_cutoff : float
        Spatial cutoff in Å. Default 9.5.
    pair_min_seq_sep : int
        Outer-loop sequence-separation requirement ``|i - j|``. Default 2.
    seed : int
        Master seed for the AA-pair index sampler.
    device : torch.device, optional
        Destination device. Defaults to ``coords["ca_coords"]``.
    dtype : torch.dtype
        Working precision. Default ``float64``.

    Returns
    -------
    dict with keys:
        ``pair_i``, ``pair_j`` (N_pair,) int64 — 0-indexed native pair indices
        ``r_ij``               (N_pair,) float — pair distances in Å
        ``rho_i``, ``rho_j``   (N_pair,) float — dump rho at i and j
        ``E_native``           (N_pair,) float — per-pair native energy
                              (includes (i,j) + cross + burial_i + burial_j)
        ``decoy_mean``         (N_pair,) float — mean(E_decoy) per pair
        ``decoy_std``          (N_pair,) float — std(E_decoy) per pair
                              (population, ddof=0)
        ``aa_i_dec``           (N_pair, n_decoys) int64 — sampled decoy α at i
        ``aa_j_dec``           (N_pair, n_decoys) int64 — sampled decoy α at j

    Notes
    -----
    * Both native AND decoy include (i,j) pair-term + cross-terms +
      burial_i + burial_j. The frustration index is then
      ``(decoy_mean - E_native) / decoy_std`` (computed in Phase 3c).
    * Cross-term mask is **only spatial** (``r_ik < 9.5``) — no seq-sep.
    * Self-pair exclusions ``k != i`` and ``k != j`` are honoured via the
      ``-W_full[i,j]`` subtraction in the native formula and the per-pair
      U subtraction in the decoy formula.
    """
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Finding #22 — std requires n_decoys >= 2.
    if int(n_decoys) < 2:
        raise ValueError(
            f"mutational_decoy_stats requires n_decoys >= 2 (got {n_decoys}); "
            "std is undefined with a single decoy."
        )

    # Finding #10 — DNA sentinel guard at the public-API entry.
    aa_native_raw = coords["residue_types"]
    _check_no_dna_sentinel(aa_native_raw)

    if rho is None:
        rho = lammps_dump_rho(coords, device=device)
    rho = rho.to(device=device, dtype=dtype)
    aa_native = aa_native_raw.to(device=device, dtype=torch.int64)
    n_residues = aa_native.shape[0]

    # Finding #54 — rho shape must match N.
    if rho.shape != (n_residues,):
        raise ValueError(
            f"rho shape {tuple(rho.shape)} does not match (N,) where "
            f"N=len(coords['ca_coords'])={n_residues}."
        )

    # --- enumerate native pairs --------------------------------------------
    pair_i, pair_j, r_ij_pair, _, dist_full, _same_chain = _enumerate_native_pairs(
        coords,
        contact_cutoff=contact_cutoff,
        pair_min_seq_sep=pair_min_seq_sep,
        device=device,
    )
    dist_full = dist_full.to(dtype=dtype)
    r_ij_pair = r_ij_pair.to(dtype=dtype)
    n_pair = int(pair_i.numel())

    # Finding #34 — bail out BEFORE expensive O(N²) precompute when there
    # are zero native pairs. Earlier versions built T_alpha + native
    # energies first and only short-circuited at the decoy step.
    if n_pair == 0:
        zero_pair = torch.zeros(0, dtype=dtype, device=device)
        return {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "r_ij": r_ij_pair,
            "rho_i": zero_pair,
            "rho_j": zero_pair,
            "E_native": zero_pair,
            "decoy_mean": zero_pair,
            "decoy_std": zero_pair,
            "aa_i_dec": torch.zeros((0, n_decoys), dtype=torch.int64, device=device),
            "aa_j_dec": torch.zeros((0, n_decoys), dtype=torch.int64, device=device),
        }

    # --- gamma tables (cached) ---------------------------------------------
    device_str = str(device)
    dtype_str = _dtype_to_str(dtype)
    gamma_direct = _cached_load_direct_gamma(device_str, dtype_str)
    gamma_med_prot, gamma_med_wat = _cached_load_mediated_gamma(device_str, dtype_str)
    burial_gamma = _cached_load_burial_gamma(device_str, dtype_str)

    # --- scalar tensors for the formula ------------------------------------
    eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
    eta_sigma_t = torch.as_tensor(eta_sigma, dtype=dtype, device=device)
    rho_0_t = torch.as_tensor(rho_0, dtype=dtype, device=device)
    k_water_t = torch.as_tensor(k_water, dtype=dtype, device=device)
    k_burial_t = torch.as_tensor(k_burial, dtype=dtype, device=device)
    burial_kappa_t = torch.as_tensor(burial_kappa, dtype=dtype, device=device)

    # --- precompute T[i, α] = Σ_k water_pair(r_ik, α, aa_k_native, rho_i, rho_k) ----
    T_alpha = _precompute_T_alpha(
        dist_full,
        aa_native,
        rho,
        gamma_direct,
        gamma_med_prot,
        gamma_med_wat,
        contact_cutoff=contact_cutoff,
        direct_r_min=direct_r_min,
        direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min,
        mediated_r_max=mediated_r_max,
        eta=eta_t,
        eta_sigma=eta_sigma_t,
        rho_0=rho_0_t,
        k_water=k_water_t,
    )                                                               # (N, 20)

    # --- native energy per pair --------------------------------------------
    # S_i_native = T[i, aa_i_native]
    S_native = T_alpha.gather(1, aa_native.unsqueeze(1)).squeeze(1)  # (N,)
    S_i = S_native[pair_i]
    S_j = S_native[pair_j]

    # W_full_native[i,j] = water_pair(r_ij, aa_i_native, aa_j_native, rho_i, rho_j)
    rho_i_p = rho[pair_i]
    rho_j_p = rho[pair_j]
    aa_i_p = aa_native[pair_i]
    aa_j_p = aa_native[pair_j]
    W_native_pair = _water_pair_full(
        r_ij_pair,
        aa_i_p,
        aa_j_p,
        rho_i_p,
        rho_j_p,
        gamma_direct,
        gamma_med_prot,
        gamma_med_wat,
        direct_r_min=direct_r_min,
        direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min,
        mediated_r_max=mediated_r_max,
        eta=eta_t,
        eta_sigma=eta_sigma_t,
        rho_0=rho_0_t,
        k_water=k_water_t,
    )                                                                # (N_pair,)

    burial_i_native = _burial_residue_energy(
        aa_i_p, rho_i_p, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=burial_rho_min,
        burial_rho_max=burial_rho_max,
        k_burial=k_burial_t,
    )
    burial_j_native = _burial_residue_energy(
        aa_j_p, rho_j_p, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=burial_rho_min,
        burial_rho_max=burial_rho_max,
        k_burial=k_burial_t,
    )

    # Native pair energy:
    #   pair_term = W_native_pair
    #   cross_i   = S_i - W_native_pair   (the k=j contribution to T[i, aa_i] is W_native_pair)
    #   cross_j   = S_j - W_native_pair   (the k=i contribution to T[j, aa_j] is W_native_pair)
    # => E_native = W_native_pair + (S_i - W_native_pair) + (S_j - W_native_pair) + B_i + B_j
    #             = S_i + S_j - W_native_pair + B_i + B_j
    E_native = S_i + S_j - W_native_pair + burial_i_native + burial_j_native

    # --- decoy sampling ----------------------------------------------------
    if n_pair == 0:
        return {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "r_ij": r_ij_pair,
            "rho_i": rho_i_p,
            "rho_j": rho_j_p,
            "E_native": E_native,
            "decoy_mean": torch.zeros(0, dtype=dtype, device=device),
            "decoy_std": torch.zeros(0, dtype=dtype, device=device),
            "aa_i_dec": torch.zeros((0, n_decoys), dtype=torch.int64, device=device),
            "aa_j_dec": torch.zeros((0, n_decoys), dtype=torch.int64, device=device),
        }

    aa_i_dec, aa_j_dec = _sample_aa_pair_indices(
        aa_native, n_pair, n_decoys, seed=seed, device=device,
    )                                                                # (N_pair, n_decoys)

    # --- decoy per-pair U + pair_term --------------------------------------
    U_iSlot_kj, U_jSlot_ki, pair_term = _per_pair_U(
        pair_i, pair_j, r_ij_pair,
        aa_i_dec, aa_j_dec,
        aa_native, rho,
        gamma_direct, gamma_med_prot, gamma_med_wat,
        contact_cutoff=contact_cutoff,
        direct_r_min=direct_r_min,
        direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min,
        mediated_r_max=mediated_r_max,
        eta=eta_t,
        eta_sigma=eta_sigma_t,
        rho_0=rho_0_t,
        k_water=k_water_t,
    )                                                                # all (N_pair, n_decoys)

    # T-gather per pair: T_pair_i[pair, d] = T_alpha[pair_i[pair], aa_i_dec[pair, d]]
    T_alpha_at_i = T_alpha[pair_i]                                   # (N_pair, 20)
    T_alpha_at_j = T_alpha[pair_j]                                   # (N_pair, 20)
    T_i_dec = T_alpha_at_i.gather(1, aa_i_dec)                       # (N_pair, n_decoys)
    T_j_dec = T_alpha_at_j.gather(1, aa_j_dec)

    cross_i = T_i_dec - U_iSlot_kj
    cross_j = T_j_dec - U_jSlot_ki

    # Burial per decoy — QA-2 L-4: drop .contiguous() (downstream
    # `_burial_residue_energy` only does elementwise tanh/mul/add and
    # gamma[aa] indexing, none of which require contiguous strides).
    rho_i_b = rho_i_p.unsqueeze(1).expand(n_pair, n_decoys)
    rho_j_b = rho_j_p.unsqueeze(1).expand(n_pair, n_decoys)
    burial_i_dec = _burial_residue_energy(
        aa_i_dec, rho_i_b, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=burial_rho_min,
        burial_rho_max=burial_rho_max,
        k_burial=k_burial_t,
    )
    burial_j_dec = _burial_residue_energy(
        aa_j_dec, rho_j_b, burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=burial_rho_min,
        burial_rho_max=burial_rho_max,
        k_burial=k_burial_t,
    )

    E_decoy = pair_term + cross_i + cross_j + burial_i_dec + burial_j_dec  # (N_pair, n_decoys)

    decoy_mean = E_decoy.mean(dim=1)
    decoy_std = E_decoy.std(dim=1, unbiased=False)

    # Finding #22 — warn when any per-pair decoy std collapses to zero.
    if bool((decoy_std == 0).any().item()):
        n_collapsed = int((decoy_std == 0).sum().item())
        warnings.warn(
            f"mutational_decoy_stats: decoy_std == 0 for "
            f"{n_collapsed}/{n_pair} pair(s); FI would be undefined there.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "r_ij": r_ij_pair,
        "rho_i": rho_i_p,
        "rho_j": rho_j_p,
        "E_native": E_native,
        "decoy_mean": decoy_mean,
        "decoy_std": decoy_std,
        "aa_i_dec": aa_i_dec,
        "aa_j_dec": aa_j_dec,
    }


__all__ = [
    "PAIR_MIN_SEQ_SEP",
    "mutational_decoy_stats",
]
