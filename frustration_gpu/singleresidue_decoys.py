"""AWSEM decoy machinery — singleresidue mode (Phase 3b).

Reproduces ``FixBackbone::compute_tert_frust_singleresidue`` and
``compute_singleresidue_decoy_ixns`` (``fix_backbone.cpp:5136-5189`` and
``:5395-5411``) in pure PyTorch.

Singleresidue mode
------------------
For each residue ``i`` independently:

* **Native energy** ``E_native(i) = burial(aa_i, rho_i) +
  Σ_{j != i, contact_mask} water_pair(r_ij, aa_i, aa_j, rho_i, rho_j)``
  where the ``contact_mask`` is the standard AWSEM mask
  ``(r_ij < 9.5) AND (|i - j| >= 2 OR cross-chain)``.

* **Decoys** scramble only ``aa_i`` (1000 random draws by uniform
  residue-index → AA composition follows protein). All other residues
  hold their native AA; ``rho_i, r_ij, rho_j, aa_j`` all unchanged.

* **Frustration index** ``FI(i) = (mean(E_decoy) - E_native) /
  std(E_decoy)`` — a per-residue scalar (NOT a per-pair matrix).

Output schema matches the LAMMPS
``singleresidue/<PDB>_singleresidue.dat`` file::

    Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex

No ``tertiary_frustration.dat`` and no ``_5adens.dat`` are produced — the
spec for this mode is per-residue only.

Vectorisation
-------------
The hot path is a (N, 20) "per-residue per-alphabet" precompute::

    W_sr[i, α] = Σ_{j != i, r_ij < 9.5, (|i-j| >= 2 OR cross-chain)}
                 water_pair(r_ij, α, aa_j_native, rho_i, rho_j_native)

Computed by sweeping α (20 passes), each pass being a single (N, N)
matrix evaluation. No loops over decoys or pairs.

After the precompute::

    E_native(i)    = burial(aa_i_native, rho_i) + W_sr[i, aa_i_native]
    E_decoy(i, d)  = burial(α_d, rho_i)         + W_sr[i, α_d]

So the (n_decoys,) ensemble per residue is just a gather + add — cheap.

Sequence-separation convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike mutational mode (where cross-terms (i, k) use ONLY spatial
cutoff), the singleresidue native loop at C++ line 5383 enforces::

    if (rij < cutoff && (abs(i_resno-j_resno) >= contact_cutoff
                         || i_chno != j_chno))

— so the standard AWSEM seq-sep filter applies. We use
``pair_min_seq_sep = 2`` (the same value as direct/water-mediated).

LOC budget: ~300-400 lines + this docstring.
"""
from __future__ import annotations

import torch

from ._contact_common import (
    ContactContext,
    _build_chain_index,
    _resolve_contact_coords,
    _validate_context_device,
    _validate_context_fingerprint,
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
)
from ._contact_common import _check_no_dna_sentinel
from .mutational_decoys import (
    PAIR_MIN_SEQ_SEP,
    _burial_residue_energy,
    _choose_alpha_chunk,
    _water_per_alpha_fused,
    _water_per_alpha_fused_sum,
    _water_rho_terms,
)
from .parameters import BURIAL_KAPPA, BURIAL_RHO_MAX, BURIAL_RHO_MIN

import warnings

# ---------------------------------------------------------------------------
# Precompute W_sr[i, α]: per-anchor per-alphabet contact-energy sum
# ---------------------------------------------------------------------------

def _precompute_W_sr(
    dist_full: torch.Tensor,
    pair_mask: torch.Tensor,
    aa_native: torch.Tensor,
    rho: torch.Tensor,
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
    """Per-residue per-alphabet contact-energy sum.

    Returns
    -------
    (N, 20) tensor where::

        W_sr[i, α] = Σ_{j} pair_mask[i, j] *
                          water_pair(r_ij, α, aa_j_native, rho_i, rho_j_native)

    Memory: O(N × 20). On 11BG (N=248) float64: 39 KB.
    """
    n = dist_full.shape[0]
    device = dist_full.device
    dtype = dist_full.dtype

    # Opt sprint #1: hoist (r, ρ)-only terms out of α-loop (algebraic refactor).
    rho_row = rho.unsqueeze(1)  # (N, 1) view
    rho_col = rho.unsqueeze(0)  # (1, N) view
    theta_direct, theta_med, sigma_wat, sigma_prot = _water_rho_terms(
        dist_full, rho_row, rho_col,
        direct_r_min=direct_r_min, direct_r_max=direct_r_max,
        mediated_r_min=mediated_r_min, mediated_r_max=mediated_r_max,
        eta=eta, eta_sigma=eta_sigma, rho_0=rho_0,
    )
    theta_direct = theta_direct.expand(n, n).contiguous()
    theta_med = theta_med.expand(n, n).contiguous()
    sigma_wat = sigma_wat.expand(n, n).contiguous()
    sigma_prot = sigma_prot.expand(n, n).contiguous()

    # Pre-mask the (r, ρ)-only θ tensors once.
    zero = torch.zeros_like(theta_direct)
    theta_direct_m = torch.where(pair_mask, theta_direct, zero)
    theta_med_m = torch.where(pair_mask, theta_med, zero)

    if device.type == "cuda":
        # Opt sprint #2 (GPU): fused (20, N, N) tensor in one shot
        # (α-chunked when memory tight).
        alpha_chunk = _choose_alpha_chunk(n, device, dtype)
        # Bug #1 fix (OOM on large N): when chunking is required, route
        # through the sum-variant which never materialises the (20, N, N)
        # cube. Peak working memory drops to (chunk, N, N) + (20, N).
        # Mirrors mutational mode's `_precompute_T_alpha` strategy.
        if alpha_chunk > 0 and alpha_chunk < 20:
            T_t = _water_per_alpha_fused_sum(
                theta_direct_m, theta_med_m, sigma_wat, sigma_prot,
                aa_native,
                gamma_direct, gamma_med_prot, gamma_med_wat,
                k_water,
                alpha_chunk_size=alpha_chunk,
            )  # (20, N)
            W_sr = T_t.transpose(0, 1).contiguous()  # (N, 20)
            return W_sr

        w_all = _water_per_alpha_fused(
            theta_direct_m, theta_med_m, sigma_wat, sigma_prot,
            aa_native,
            gamma_direct, gamma_med_prot, gamma_med_wat,
            k_water,
            alpha_chunk_size=alpha_chunk,
        )                                                   # (20, N, N)
        W_sr = w_all.sum(dim=2).transpose(0, 1).contiguous()    # (N, 20)
        return W_sr

    # CPU: keep the per-α loop but with Idea 1's hoisting already applied.
    g_dir_rows = gamma_direct[:, aa_native]    # (20, N)
    g_mp_rows = gamma_med_prot[:, aa_native]   # (20, N)
    g_mw_rows = gamma_med_wat[:, aa_native]    # (20, N)

    W_sr = torch.empty((n, 20), dtype=dtype, device=device)
    for alpha in range(20):
        g_dir = g_dir_rows[alpha].unsqueeze(0)
        g_mp = g_mp_rows[alpha].unsqueeze(0)
        g_mw = g_mw_rows[alpha].unsqueeze(0)
        sigma_gamma_med = sigma_prot * g_mp + sigma_wat * g_mw
        w = -k_water * (g_dir * theta_direct_m + sigma_gamma_med * theta_med_m)
        W_sr[:, alpha] = w.sum(dim=1)
    return W_sr


# ---------------------------------------------------------------------------
# Sample decoy aa per residue
# ---------------------------------------------------------------------------

def _sample_aa_per_residue(
    aa: torch.Tensor,
    n_decoys: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``(N, n_decoys)`` decoy aa indices via uniform residue draws.

    Returns
    -------
    (N, n_decoys) int64 — decoy α at slot i for each decoy d.

    Matches :func:`src.decoys.sample_configurational_decoys` for AA draws
    (uniform residue-index → AA composition follows protein).

    Speed-fix4 SPEED-2 Idea 3: We keep the CPU ``torch.Generator(seed)`` so
    the PRNG sequence is bit-identical to the previous implementation, but
    we now move ``idx`` to the device and gather on-device. This drops the
    full-CPU gather + the ``aa.cpu()`` sync (two H2D round-trips become
    one). Integer gather → bit-identical output independent of device.
    """
    n = aa.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randint(0, n, (n, n_decoys), generator=gen)
    # On-device gather: ``aa`` is already on ``device``; ``idx`` is the small
    # one (N · n_decoys int64). Single H2D copy of ``idx``, then on-device
    # advanced indexing of ``aa``.
    idx_dev = idx.to(device=device, non_blocking=True)
    aa_dev = aa.to(device=device, dtype=torch.int64)
    return aa_dev[idx_dev]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def singleresidue_decoy_stats(
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
    _context: ContactContext | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-residue (E_native, decoy_mean, decoy_std, FI) for singleresidue.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`.
    rho : (N,) tensor, optional
        LAMMPS-dump-compatible rho (``min_seq_sep = 12``). If ``None``,
        computed via :func:`src.decoys.lammps_dump_rho`.
    n_decoys : int
        Number of decoys per residue. Default 1000.
    contact_cutoff : float
        Spatial cutoff in Å. Default 9.5.
    pair_min_seq_sep : int
        Sequence-separation requirement ``|i - j|``. Default 2.
        Inter-chain pairs always contribute.
    seed : int
        Master seed for the per-residue AA sampler.
    device, dtype : torch.device / dtype
        Computation precision and target device.

    Returns
    -------
    dict with keys (all (N,)):
        ``rho``          per-residue dump rho
        ``aa_native``    int64 native aa indices
        ``E_native``     per-residue native energy contribution
        ``decoy_mean``   mean(E_decoy) per residue
        ``decoy_std``    std(E_decoy) per residue (population, ddof=0)
        ``FI``           (mean - native) / std   — the frustration index
    """
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Finding #22 — singleresidue needs n_decoys >= 2 to compute std.
    if int(n_decoys) < 2:
        raise ValueError(
            f"singleresidue_decoy_stats requires n_decoys >= 2 (got {n_decoys}); "
            "std is undefined with a single decoy and FI would be silently zero."
        )

    # Finding #10 — DNA sentinel guard at the public-API entry. Lower-level
    # energy APIs already check this; decoys must too because negative
    # indices into the (20, 20) γ / (20, 3) burial tables silently wrap.
    aa_native_raw = coords["residue_types"]
    _check_no_dna_sentinel(aa_native_raw)

    if rho is None:
        rho = lammps_dump_rho(coords, device=device)
    rho = rho.to(device=device, dtype=dtype)
    aa_native = aa_native_raw.to(device=device, dtype=torch.int64)
    n = aa_native.shape[0]

    # Finding #54 — validate rho shape against the residue count derived
    # from coords; a mismatched rho silently broadcasts.
    if rho.shape != (n,):
        raise ValueError(
            f"rho shape {tuple(rho.shape)} does not match (N,) where "
            f"N=len(coords['ca_coords'])={n}."
        )

    # --- build (N, N) distance matrix + contact mask -----------------------
    # Speed-fix4 SPEED-2 Idea 2: re-use ``_context.dist_full`` when the caller
    # has already built it. Bit-identical construction (same 1e6 sentinel,
    # same vector_norm, same +inf fill on NaN-row pairs).
    cb_or_ca = _resolve_contact_coords(coords, device=device)
    chain_idx = _build_chain_index(coords["chain_ids"], device=device)
    finite_row = torch.isfinite(cb_or_ca).all(dim=-1, keepdim=True)
    finite_pair_2d = finite_row.expand(n, n) & finite_row.transpose(0, 1).expand(n, n)

    # Finding #30 — fingerprint guard. Without this check, a caller can build
    # a context from structure A and feed coords from structure B; the cached
    # `_context.dist_full` would silently mix with the new coords. Mirror the
    # direct/water/DH energy modules' pattern: validate device first, then
    # fingerprint, before consuming any cached tensor.
    if _context is not None:
        _validate_context_device(_context, device)
        _validate_context_fingerprint(_context, coords)

    if _context is not None and _context.dist_full is not None:
        dist_full = _context.dist_full.to(dtype=dtype)
    else:
        safe_cb = torch.where(finite_row, cb_or_ca, torch.full_like(cb_or_ca, 1.0e6))
        diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)
        dist_full = torch.linalg.vector_norm(diff, dim=-1).to(dtype=dtype)
        dist_full = torch.where(
            finite_pair_2d, dist_full, torch.full_like(dist_full, float("inf"))
        )

    # QA-2 L-1 cleanup: removed dead ``if False`` branch that previously
    # computed ``finite_pair`` then immediately overwrote it. ``finite_pair_2d``
    # above is the only mask consumed below.
    same_chain = chain_idx.unsqueeze(0) == chain_idx.unsqueeze(1)
    idx = torch.arange(n, device=device)
    seq_diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    contact_pair_mask = (
        (dist_full < contact_cutoff)
        & ((~same_chain) | (seq_diff >= pair_min_seq_sep))
        & (idx.unsqueeze(0) != idx.unsqueeze(1))
        & finite_pair_2d
    )

    # --- gamma tables (cached) ---------------------------------------------
    device_str = str(device)
    dtype_str = _dtype_to_str(dtype)
    gamma_direct = _cached_load_direct_gamma(device_str, dtype_str)
    gamma_med_prot, gamma_med_wat = _cached_load_mediated_gamma(device_str, dtype_str)
    burial_gamma = _cached_load_burial_gamma(device_str, dtype_str)

    # --- scalar params -----------------------------------------------------
    eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
    eta_sigma_t = torch.as_tensor(eta_sigma, dtype=dtype, device=device)
    rho_0_t = torch.as_tensor(rho_0, dtype=dtype, device=device)
    k_water_t = torch.as_tensor(k_water, dtype=dtype, device=device)
    k_burial_t = torch.as_tensor(k_burial, dtype=dtype, device=device)
    burial_kappa_t = torch.as_tensor(burial_kappa, dtype=dtype, device=device)

    # --- precompute (N, 20) ------------------------------------------------
    W_sr = _precompute_W_sr(
        dist_full,
        contact_pair_mask,
        aa_native,
        rho,
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
    )

    # --- per-residue burial table (N, 20) ----------------------------------
    # B[i, α] = burial(α, rho_i)
    # Vectorise: rho repeats across α; α repeats across i.
    rho_expand = rho.unsqueeze(1).expand(n, 20)                      # (N, 20)
    alpha_expand = torch.arange(20, device=device).unsqueeze(0).expand(n, 20)
    B_table = _burial_residue_energy(
        alpha_expand.contiguous(),
        rho_expand.contiguous(),
        burial_gamma,
        burial_kappa=burial_kappa_t,
        burial_rho_min=burial_rho_min,
        burial_rho_max=burial_rho_max,
        k_burial=k_burial_t,
    )                                                                # (N, 20)

    # --- native energies ---------------------------------------------------
    E_native_per_res = W_sr.gather(1, aa_native.unsqueeze(1)).squeeze(1) + \
                       B_table.gather(1, aa_native.unsqueeze(1)).squeeze(1)

    if n == 0:
        return {
            "rho": rho,
            "aa_native": aa_native,
            "E_native": E_native_per_res,
            "decoy_mean": torch.zeros(0, dtype=dtype, device=device),
            "decoy_std": torch.zeros(0, dtype=dtype, device=device),
            "FI": torch.zeros(0, dtype=dtype, device=device),
            "aa_dec": torch.zeros((0, n_decoys), dtype=torch.int64, device=device),
        }

    # --- decoy sampling + energies ----------------------------------------
    aa_dec = _sample_aa_per_residue(aa_native, n_decoys, seed=seed, device=device)
    # (N, n_decoys)

    # E_decoy[i, d] = W_sr[i, aa_dec[i, d]] + B_table[i, aa_dec[i, d]]
    W_dec = W_sr.gather(1, aa_dec)                                   # (N, n_decoys)
    B_dec = B_table.gather(1, aa_dec)                                # (N, n_decoys)
    E_decoy = W_dec + B_dec                                          # (N, n_decoys)

    decoy_mean = E_decoy.mean(dim=1)
    decoy_std = E_decoy.std(dim=1, unbiased=False)

    # Finding #22 — emit a warning when the per-residue decoy std collapses
    # for ANY residue (FI is zero by construction at those slots).
    if bool((decoy_std == 0).any().item()):
        n_collapsed = int((decoy_std == 0).sum().item())
        warnings.warn(
            f"singleresidue_decoy_stats: decoy_std == 0 for {n_collapsed}/{n} "
            "residue(s); FI forced to 0 there. Likely n_decoys too small or "
            "the residue has no contacts.",
            RuntimeWarning,
            stacklevel=2,
        )

    # FI = (mean - native) / std; guard division by zero
    safe_std = torch.where(decoy_std > 0, decoy_std, torch.ones_like(decoy_std))
    FI = (decoy_mean - E_native_per_res) / safe_std
    FI = torch.where(decoy_std > 0, FI, torch.zeros_like(FI))

    return {
        "rho": rho,
        "aa_native": aa_native,
        "E_native": E_native_per_res,
        "decoy_mean": decoy_mean,
        "decoy_std": decoy_std,
        "FI": FI,
        "aa_dec": aa_dec,
    }


__all__ = [
    "singleresidue_decoy_stats",
]
