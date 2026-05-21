"""AWSEM decoy machinery (Phase 3a — configurational mode).

Reproduces ``FixBackbone::compute_decoy_ixns`` (``fix_backbone.cpp:5249-5344``)
in pure PyTorch. The decoy ensemble drives the *frustration index*::

    FI(i, j) = (mean(E_decoy) - E_native) / std(E_decoy)

CRITICAL: the rho used by the decoy formula is NOT the same rho used for
burial energy — *because frustratometeR ships a patched LAMMPS binary*
---------------------------------------------------------------------------
The validation dumps (`tertiary_frustration.dat`) are produced by the
frustratometeR R package, which **ships TWO precompiled LAMMPS binaries**
where the rho sequence-separation cutoff (``SeqDist``) is baked into the
binary at compile time:

* ``lmp_serial_3_Linux``  → rho computed with ``min_seq_sep = 3``
* ``lmp_serial_12_Linux`` → rho computed with ``min_seq_sep = 12``

frustratometeR's default is ``SeqDist = 12``, so the panel dumps in
``benchmark/cpu_baseline/`` were generated with the 12-binary. See
``frustratometeR/R/functions.R:276,285,302,427-432``. The upstream
``adavtyan/awsemmd`` C++ in our reference mirror uses ``min_seq_sep = 1``
for both burial AND decoy rho — that's NOT a bug, it's the same code
*before* frustratometeR's binary patch.

For burial energy, the upstream ``min_seq_sep = 1`` is what we (and the
dump's V_burial column) want — verified at machine precision (V_burial =
-41.799 for 5AON).

For the decoy formula and the displayed ``rho_i, rho_j`` columns, we need
to use the SAME value frustratometeR's bundled binary uses (default 12).
The :func:`lammps_dump_rho` helper accepts ``min_seq_sep`` as a parameter
so callers can validate against ``SeqDist = 3`` binaries or against
OpenAWSEM (``min_sequence_separation_rho = 2``).

Configurational vs mutational vs singleresidue
----------------------------------------------
LAMMPS-AWSEM's ``tert_frust_mode`` selects how decoys are sampled:

* **configurational** (this module, Phase 3a)
    Sample 1000 decoys ONCE per structure, then reuse the same
    ``(decoy_mean, decoy_std)`` for every native pair. Per decoy ``k``,
    all five inputs are randomised independently:

        - ``aa_i_decoy[k]`` ∈ {0..19}: pick a random residue index, read
          off its identity. The empirical AA-distribution follows the
          protein's own composition (NOT 1/20 uniform).
        - ``aa_j_decoy[k]`` ∈ {0..19}: same, second draw.
        - ``r_ij_decoy[k]``: pick a random pair of distinct residue
          indices ``(p, q)``; take the effective-CB distance ``r_pq``.
          Reject and resample until ``r_pq < contact_cutoff`` (9.5 Å)
          AND ``p != q`` (line 5262 of the C++ uses ``||``).
        - ``rho_i_decoy[k]``, ``rho_j_decoy[k]``: pick a fresh INDEPENDENT
          pair of residue indices (no contact constraint); read off
          ``rho`` at each index.

    Per decoy ``k`` the energy is::

        E_decoy[k] = V_water_pair(r_ij_decoy[k], aa_i_decoy[k], aa_j_decoy[k],
                                  rho_i_decoy[k], rho_j_decoy[k])
                   + V_burial_residue(aa_i_decoy[k], rho_i_decoy[k])
                   + V_burial_residue(aa_j_decoy[k], rho_j_decoy[k])

    Critically, **DH is NOT included** because Frustrapy's ``awsem.in``
    sets ``huckel_flag = false`` (the C++ check at line 5290 short-circuits
    to ``electrostatic_energy = 0``). DH stays as an auxiliary feature, not
    in the per-pair native sum.

* **mutational** (Phase 3b)
    For each native pair (i, j), regenerate 1000 decoys, holding
    ``(rij, rho_i, rho_j)`` at the NATIVE values and only scrambling the
    ``aa_i``, ``aa_j`` identities (plus including all (i, k) and (j, k)
    surrounding contacts).

* **singleresidue** (Phase 3b)
    Per-residue decoys; scrambles one identity, integrates over all
    contacts; output is per-residue (not per-pair).

Why "configurational" is the cached-once mode
---------------------------------------------
The cache is justified by symmetry: in configurational mode the decoy
distribution does NOT depend on the native pair (i, j) at all — every
input is drawn from the global pool. So ``decoy_mean`` and ``decoy_std``
are scalars shared across all rows of ``tertiary_frustration.dat``. The
C++ enforces this by gating ``already_computed_configurational_decoys = 1``
after the first call (line 5341).

RNG noise floor
---------------
LAMMPS-AWSEM uses ``rand()`` from libc, which is not portable and not
reproducible across compilers. Our PyTorch path uses ``torch.Generator``
with a fixed seed. We can NOT match the C++ decoy stats exactly —
the noise floor is ~3% relative (the spread between independent
``n_decoys=1000`` runs on the same protein). The validation tolerance
in :mod:`tests.test_decoys` is therefore 3%, which is the same floor that
gates Phase 3c's FI-Pearson comparison.

Contract on inputs
------------------
``aa`` is the only identity input — coords and ``rho`` are
identity-independent in both decoy modes. The gamma-indexing pattern
``gamma[aa_i.unsqueeze(1), aa_j.unsqueeze(0)]`` returns a per-decoy
vector ``(n_decoys,)`` when ``aa_i`` and ``aa_j`` are 1-D tensors of
length ``n_decoys`` (we use 0-D broadcast inside the elementwise
formulas, not the 2-D outer product). The 2-D outer-product pattern is
correct for the *native* sum over all (i, j) pairs but would be
quadratically wrong here.

Module-level cache for the mediated gamma
-----------------------------------------
:func:`src.contact_gamma.load_mediated_gamma` does a file-IO + parse on
every call. Phase 2b reviewer flagged this would be hot in future
mutational mode (which re-creates 1000 decoys per pair). We wrap the
loader with ``functools.lru_cache`` on a key of
``(device_str, dtype_str)`` so a re-call inside this module is O(1).

LOC budget: ~400 lines + this docstring.
"""
from __future__ import annotations

import functools
import warnings

import torch

from ._contact_common import (
    ContactContext,
    _build_chain_index,
    _check_no_dna_sentinel,
    _resolve_contact_coords,
    _validate_context_device,
    _validate_context_fingerprint,
)
from .burial import compute_rho
from .contact_gamma import load_direct_gamma, load_mediated_gamma
from .parameters import (
    BURIAL_KAPPA,
    BURIAL_RHO_MAX,
    BURIAL_RHO_MIN,
    load_burial_gamma,
)

# --- LAMMPS-dump-compatible rho -----------------------------------------------
# The LAMMPS-AWSEM tertiary_frustration.dat dump uses a rho value that is
# computed with the same smooth sigmoid as src.burial.compute_rho but with
# ``min_seq_sep = 12`` (verified across 6 PDBs spanning n=49 to n=830). This
# is DIFFERENT from the rho used to compute V_burial (which uses min_seq_sep=1
# and matches LAMMPS's energy.log Burial column to machine precision). See the
# module docstring for the full forensic.
LAMMPS_DUMP_RHO_MIN_SEQ_SEP: int = 12


# Defaults that mirror the C++ ``[Water]`` and ``[Burial]`` blocks
# (`fix_backbone.cpp:257-266` and `:5444-5500`).
DEFAULT_CONTACT_CUTOFF_A: float = 9.5    # tert_frust_cutoff (== mediated r_max)
DEFAULT_N_DECOYS: int = 1000

DIRECT_R_MIN_A: float = 4.5
DIRECT_R_MAX_A: float = 6.5
MEDIATED_R_MIN_A: float = 6.5
MEDIATED_R_MAX_A: float = 9.5
WATER_ETA_PER_A: float = 5.0
WATER_ETA_SIGMA: float = 7.0
WATER_RHO_0: float = 2.6


# --- module-level cache for gamma loaders -------------------------------------
# Wrapping the loader avoids repeated file IO when the decoy driver is invoked
# many times (e.g. one call per residue in a future mutational driver). The
# cache key includes device + dtype so different precision/devices co-exist.
@functools.lru_cache(maxsize=8)
def _cached_load_mediated_gamma(device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    return load_mediated_gamma(device=device, dtype=dtype)


@functools.lru_cache(maxsize=8)
def _cached_load_direct_gamma(device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    return load_direct_gamma(device=device, dtype=dtype)


@functools.lru_cache(maxsize=8)
def _cached_load_burial_gamma(device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    return load_burial_gamma(device=device, dtype=dtype)


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Return the bare dtype name for cache keying (e.g. ``'float32'``)."""
    return str(dtype).removeprefix("torch.")


# --- LAMMPS-dump-compatible rho helper ---------------------------------------
def lammps_dump_rho(
    coords: dict[str, torch.Tensor],
    *,
    min_seq_sep: int = LAMMPS_DUMP_RHO_MIN_SEQ_SEP,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the per-residue rho that LAMMPS-AWSEM writes to
    ``tertiary_frustration.dat`` and uses internally in
    ``compute_decoy_ixns``.

    This rho is *empirically* the same smooth sigmoid as
    :func:`src.burial.compute_rho` but with ``min_seq_sep = 12`` instead of
    the documented ``1``. The difference reflects an undocumented filter in
    the LAMMPS-AWSEM C++ which we have not been able to reverse-engineer
    from the source — but the resulting rho values match the dump to
    1e-4 across the validation panel.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`.
    min_seq_sep : int
        Sequence-separation cutoff. Default 12 (matches LAMMPS dump).
        ``compute_rho`` returns Σ_j θ(r_ij) over j with |i-j| > min_seq_sep
        (within-chain) or ALL j (cross-chain).
    device : torch.device, optional
        Destination device. Defaults to the device of ``coords["ca_coords"]``.

    Returns
    -------
    (N,) tensor of dimensionless rho values.

    Notes
    -----
    The cross-chain handling matches :func:`src.burial.compute_rho`:
    inter-chain pairs always contribute (no seq-sep filter). This was
    verified by inspecting :func:`src.burial.compute_rho` line 134
    (``(~same_chain) | (seq_diff > min_seq_sep)``).
    """
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    cb_or_ca = _resolve_contact_coords(coords, device=device)
    n = cb_or_ca.shape[0]
    chain_idx = _build_chain_index(coords["chain_ids"], device=device)
    seq_index = torch.arange(n, dtype=torch.int64, device=device)
    return compute_rho(
        cb_or_ca,
        seq_index,
        chain_idx,
        min_seq_sep=min_seq_sep,
    )


# --- exposed switching functions (also used by the dense terms) --------------
def water_theta(
    r: torch.Tensor,
    r_min: float,
    r_max: float,
    eta: float,
) -> torch.Tensor:
    """θ(r) = ¼ × (1 + tanh(η(r - r_min))) × (1 + tanh(η(r_max - r))).

    Same sigmoid window used by :mod:`src.direct_contact` and
    :mod:`src.water_mediated`. Exposed here so the decoy formula can
    re-use it without circular imports.
    """
    t_min = torch.tanh(eta * (r - r_min))
    t_max = torch.tanh(eta * (r_max - r))
    return 0.25 * (1.0 + t_min) * (1.0 + t_max)


def burial_switch(
    rho: torch.Tensor,
    rho_min_w: float,
    rho_max_w: float,
    kappa: float,
) -> torch.Tensor:
    """tanh(κ(ρ - ρ_min)) + tanh(κ(ρ_max - ρ)) for a single burial well.

    Used by V_burial. Per ``fix_backbone.cpp:5483-5497`` the per-well
    contribution is ``-0.5 × k_burial × γ_burial[aa, w] × (t_min + t_max)``.
    """
    return torch.tanh(kappa * (rho - rho_min_w)) + torch.tanh(kappa * (rho_max_w - rho))


# --- main sampler -------------------------------------------------------------
def sample_configurational_decoys(
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor,
    *,
    n_decoys: int = DEFAULT_N_DECOYS,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF_A,
    seed: int = 0,
    device: torch.device | None = None,
    max_resample_iter: int = 64,
    _context: ContactContext | None = None,
) -> dict[str, torch.Tensor]:
    """Sample ``n_decoys`` configurational decoys for the protein.

    The five outputs are independent draws from the structure's residue
    pool. Configurational caching: this sample is meant to be reused for
    every native (i, j) pair — see :mod:`src.decoys` module docstring.

    Parameters
    ----------
    coords : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``,
        ``cb_coords``, ``residue_types``.
    rho : (N,) tensor
        Per-residue burial density (from :func:`src.burial.compute_rho`).
    n_decoys : int
        Number of decoy samples. Default 1000 (matches LAMMPS-AWSEM and
        frustrapy).
    contact_cutoff : float
        Maximum allowed decoy ``r_ij`` in Å. Default 9.5 (= the C++
        ``tert_frust_cutoff``, same as mediated ``r_max``).
    seed : int
        Seed for the PyTorch generator used to draw residue indices.
    device : torch.device, optional
        Destination device. Defaults to the device of ``coords["ca_coords"]``.
    max_resample_iter : int
        Safety cap on the number of resampling iterations for the
        rejection sampler (drawing ``r_ij`` from in-contact pairs).
        Default 64 — empirically enough for any folded protein (~30%
        in-contact rate gives mean iteration count ~3).

    Returns
    -------
    dict with keys:
        ``aa_i_decoy``  (n_decoys,) int64 — decoy aa-index for slot i
        ``aa_j_decoy``  (n_decoys,) int64 — decoy aa-index for slot j
        ``rij_decoy``   (n_decoys,) float — decoy CB-CB distance (Å)
        ``rho_i_decoy`` (n_decoys,) float — decoy burial density for slot i
        ``rho_j_decoy`` (n_decoys,) float — decoy burial density for slot j

    Notes
    -----
    * Per-residue indices are drawn via :meth:`torch.randint` with a CPU
      generator that is seeded explicitly. The CPU draw is then moved to
      the destination device — this avoids a CUDA generator dependency
      and keeps test reproducibility cross-device.
    * The rejection sampler for ``r_ij`` may need a few iterations on
      sparse/extended structures. We do not match the C++ libc ``rand()``
      sequence exactly; the noise floor between independent samples is
      ~3% relative (see module docstring).
    """
    ca = coords["ca_coords"]
    if device is None:
        device = ca.device
    else:
        device = torch.device(device)

    # Finding #22 — n_decoys >= 2 (std otherwise undefined / FI silently 0).
    if int(n_decoys) < 2:
        raise ValueError(
            f"sample_configurational_decoys requires n_decoys >= 2 (got {n_decoys})."
        )

    # Finding #10 — DNA sentinel guard.
    _check_no_dna_sentinel(coords["residue_types"])

    cb_or_ca = _resolve_contact_coords(coords, device=device)        # (N, 3)
    aa = coords["residue_types"].to(device=device, dtype=torch.int64)  # (N,)
    n = cb_or_ca.shape[0]

    # Finding #54 — rho must be a length-N 1-D tensor.
    if rho.shape != (n,):
        raise ValueError(
            f"rho shape {tuple(rho.shape)} does not match (N,) where "
            f"N=len(coords['ca_coords'])={n}."
        )
    rho = rho.to(device=device, dtype=cb_or_ca.dtype)                  # (N,)

    if n < 2:
        raise ValueError(
            f"sample_configurational_decoys requires N >= 2 (got N={n}); "
            "decoy sampling is undefined for monomeric inputs."
        )

    # --- precompute the (N,N) effective-CB distance matrix ------------------
    # NaN-safe: rows with NaN coords (e.g. fully missing residue) produce NaN
    # distances. We replace those with +inf to ensure they never satisfy the
    # ``< contact_cutoff`` rejection criterion. The diagonal is zero, which is
    # ALSO masked out by the ``i == j`` rejection on line 5262 of the C++.
    #
    # Speed-fix4 SPEED-2 Idea 2: re-use ``_context.dist_full`` when the
    # caller has already built it. Construction is bit-identical to the
    # inline path here.
    #
    # Finding #30 — fingerprint guard. Without this check, a caller can build
    # a context from structure A and feed coords from structure B; the cached
    # `_context.dist_full` would silently mix with the new coords. Mirror the
    # direct/water/DH energy modules' pattern: validate device first, then
    # fingerprint, before consuming any cached tensor.
    if _context is not None:
        _validate_context_device(_context, device)
        _validate_context_fingerprint(_context, coords)
    if _context is not None and _context.dist_full is not None:
        dist_full = _context.dist_full.to(device=device, dtype=cb_or_ca.dtype)
    else:
        finite_row = torch.isfinite(cb_or_ca).all(dim=-1, keepdim=True)    # (N, 1)
        safe_cb = torch.where(
            finite_row,
            cb_or_ca,
            torch.full_like(cb_or_ca, 1.0e6),
        )
        diff = safe_cb.unsqueeze(0) - safe_cb.unsqueeze(1)                 # (N, N, 3)
        dist_full = torch.linalg.vector_norm(diff, dim=-1)                 # (N, N)
        # NaN-poisoned (originally NaN) rows: force to +inf so the rejection
        # criterion ``< cutoff`` is always False for those pairs.
        finite_pair = finite_row & finite_row.transpose(0, 1)              # (N, N)
        dist_full = torch.where(
            finite_pair,
            dist_full,
            torch.full_like(dist_full, float("inf")),
        )

    # --- CPU generator (so seed is portable across devices) ----------------
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # --- AA draws: uniform residue-index → AA composition follows protein -
    aa_i_idx = torch.randint(0, n, (n_decoys,), generator=gen)             # CPU int64
    aa_j_idx = torch.randint(0, n, (n_decoys,), generator=gen)             # CPU int64
    aa_i_decoy = aa[aa_i_idx.to(device=device)]                            # (n_decoys,)
    aa_j_decoy = aa[aa_j_idx.to(device=device)]                            # (n_decoys,)

    # --- rho draws: ANOTHER independent pair (no contact constraint) ------
    rho_i_idx = torch.randint(0, n, (n_decoys,), generator=gen)
    rho_j_idx = torch.randint(0, n, (n_decoys,), generator=gen)
    rho_i_decoy = rho[rho_i_idx.to(device=device)]                         # (n_decoys,)
    rho_j_decoy = rho[rho_j_idx.to(device=device)]                         # (n_decoys,)

    # --- rij draws: direct inverse-CDF sampling over the in-contact set ----
    # Speed-fix4 SPEED-2 Idea 1: replace the rejection loop with analytic
    # uniform-over-S sampling, where S = {(p, q) : p != q, r_pq < cutoff}.
    #
    # The C++ does (line 5262):
    #     do { p = rand() % N; q = rand() % N; r_pq = ...; }
    #     while (r_pq >= cutoff || p == q);
    # — which is uniform sampling from S. Inverse-CDF sampling enumerates S
    # once, then draws ``n_decoys`` uniform indices into the enumeration:
    # mathematically the same distribution, but:
    #   (a) no per-iteration .item() sync (the rejection loop did 3-4 stalls
    #       on average; on sparse PDBs it hit `max_resample_iter` and fell
    #       into a documented biased fallback);
    #   (b) no biased fallback — every sample is uniform-over-S by
    #       construction (or RuntimeError if S is empty);
    #   (c) one randint call instead of 1-4.
    #
    # NOTE: the PRNG sequence (1 randint vs 1-4) differs from the old loop,
    # so per-PDB ``decoy_mean`` / ``decoy_std`` shift at the ~3% RNG floor.
    # The Spearman gate (which orders FI) is preserved. This is documented
    # in docs/speed_fix4_results.md as a free correctness win + a known
    # numerical shift; the old biased fallback also shifted outputs in an
    # uncontrolled way on sparse PDBs.
    eye_n = torch.eye(n, dtype=torch.bool, device=device)
    finite_off_diag = torch.where(
        eye_n,
        torch.full_like(dist_full, float("inf")),
        dist_full,
    )
    flat = finite_off_diag.flatten()
    in_contact_dists = flat[flat < contact_cutoff]
    if in_contact_dists.numel() == 0:
        raise RuntimeError(
            f"sample_configurational_decoys: no in-contact pairs found "
            f"(N={n}, cutoff={contact_cutoff} Å). The structure may be "
            "fragmented or in non-physical coordinates."
        )
    n_in_contact = int(in_contact_dists.numel())
    # Single randint call, CPU generator for cross-device reproducibility.
    rij_idx_cpu = torch.randint(0, n_in_contact, (n_decoys,), generator=gen)
    rij_idx = rij_idx_cpu.to(device=device, non_blocking=True)
    rij_decoy = in_contact_dists[rij_idx].to(dtype=cb_or_ca.dtype)
    # ``max_resample_iter`` is now a no-op kwarg kept for API compatibility.
    _ = max_resample_iter

    return {
        "aa_i_decoy": aa_i_decoy,
        "aa_j_decoy": aa_j_decoy,
        "rij_decoy": rij_decoy,
        "rho_i_decoy": rho_i_decoy,
        "rho_j_decoy": rho_j_decoy,
    }


# --- decoy energy computation -------------------------------------------------
def compute_configurational_decoy_energy(
    decoys: dict[str, torch.Tensor],
    *,
    gamma_direct: torch.Tensor | None = None,
    gamma_mediated_protein: torch.Tensor | None = None,
    gamma_mediated_water: torch.Tensor | None = None,
    burial_gamma: torch.Tensor | None = None,
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
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Compute the per-decoy energy ``E_decoy = V_water + V_burial_i + V_burial_j``.

    Matches ``FixBackbone::compute_decoy_ixns`` line 5331::

        tert_frust_decoy_energies[k] =
            water_energy(rij, aa_i, aa_j, rho_i, rho_j)
            + burial_energy(aa_i, rho_i)
            + burial_energy(aa_j, rho_j)

    with ``electrostatic_energy = 0`` (Frustrapy gates ``huckel_flag = false``,
    see C++ line 5293).

    Parameters
    ----------
    decoys : dict
        Output of :func:`sample_configurational_decoys`.
    gamma_direct, gamma_mediated_protein, gamma_mediated_water : (20, 20) tensors, optional
        Contact gamma tables. Loaded from ``src/data/gamma.dat`` via the
        cached loaders if not provided.
    burial_gamma : (20, 3) tensor, optional
        Burial gamma table. Loaded from ``src/data/burial_gamma.dat`` via
        the cached loader if not provided.
    k_water, k_burial : float
        Energy prefactors. Defaults 1.0 (LAMMPS ``units real`` convention).
    eta, eta_sigma, rho_0 : float
        Water-mediated sigmoid sharpness and burial-threshold constants.
        Defaults match the C++ ``[Water]`` block.
    direct_r_min/max, mediated_r_min/max : float
        Sigmoid window edges in Å. Defaults match the C++ ``[Water]`` block.
    burial_kappa, burial_rho_min, burial_rho_max : floats / 3-tuples
        Burial sigmoid params. Defaults match the C++ ``[Burial]`` block.
    device : torch.device, optional
        Destination device. Defaults to the device of the decoy tensors.
    dtype : torch.dtype
        Working precision. Default ``float64`` — recommend keeping this so
        the (mean, std) reductions don't lose precision.

    Returns
    -------
    dict with keys:
        ``decoy_energies`` (n_decoys,) float — per-decoy E
        ``decoy_mean``     scalar float       — mean(E_decoy)
        ``decoy_std``      scalar float       — std(E_decoy) (population, ddof=0
                                                — matches the C++ compute_array_std)

    Notes
    -----
    * Population std (``ddof = 0``) is used to match LAMMPS-AWSEM's
      ``compute_array_std`` (which divides by ``N``, not ``N - 1``).
    * The gamma-indexing pattern ``g[aa_i, aa_j]`` here treats ``aa_i`` and
      ``aa_j`` as 1-D vectors of length ``n_decoys`` — this returns a 1-D
      vector by elementwise gather, NOT a 2-D outer product. See module
      docstring "Contract on inputs".
    """
    aa_i = decoys["aa_i_decoy"]
    aa_j = decoys["aa_j_decoy"]
    rij = decoys["rij_decoy"]
    rho_i = decoys["rho_i_decoy"]
    rho_j = decoys["rho_j_decoy"]

    # Finding #55 — every decoy field must share the same 1-D length.
    field_shapes = {
        "aa_i_decoy": tuple(aa_i.shape),
        "aa_j_decoy": tuple(aa_j.shape),
        "rij_decoy": tuple(rij.shape),
        "rho_i_decoy": tuple(rho_i.shape),
        "rho_j_decoy": tuple(rho_j.shape),
    }
    distinct = set(field_shapes.values())
    if len(distinct) != 1:
        raise ValueError(
            f"compute_configurational_decoy_energy: decoy field shapes "
            f"disagree (would silently broadcast). Got {field_shapes}."
        )
    only_shape = next(iter(distinct))
    if len(only_shape) != 1:
        raise ValueError(
            f"compute_configurational_decoy_energy: decoy fields must be "
            f"1-D, got shape {only_shape}."
        )
    n_decoys = only_shape[0]
    # Finding #22 — std needs >= 2.
    if n_decoys < 2:
        raise ValueError(
            f"compute_configurational_decoy_energy: n_decoys >= 2 required "
            f"(got {n_decoys}); std is undefined."
        )

    # Finding #10 — guard against DNA sentinels in the decoy aa indices.
    _check_no_dna_sentinel(aa_i)
    _check_no_dna_sentinel(aa_j)

    if device is None:
        device = rij.device
    else:
        device = torch.device(device)

    aa_i = aa_i.to(device=device, dtype=torch.int64)
    aa_j = aa_j.to(device=device, dtype=torch.int64)
    rij = rij.to(device=device, dtype=dtype)
    rho_i = rho_i.to(device=device, dtype=dtype)
    rho_j = rho_j.to(device=device, dtype=dtype)

    # --- load gamma tables (cached) ----------------------------------------
    device_str = str(device)
    dtype_str = _dtype_to_str(dtype)

    if gamma_direct is None:
        gamma_direct = _cached_load_direct_gamma(device_str, dtype_str)
    else:
        gamma_direct = gamma_direct.to(device=device, dtype=dtype)
    if gamma_mediated_protein is None or gamma_mediated_water is None:
        gp_def, gw_def = _cached_load_mediated_gamma(device_str, dtype_str)
        if gamma_mediated_protein is None:
            gamma_mediated_protein = gp_def
        if gamma_mediated_water is None:
            gamma_mediated_water = gw_def
    gamma_mediated_protein = gamma_mediated_protein.to(device=device, dtype=dtype)
    gamma_mediated_water = gamma_mediated_water.to(device=device, dtype=dtype)
    if burial_gamma is None:
        burial_gamma = _cached_load_burial_gamma(device_str, dtype_str)
    else:
        burial_gamma = burial_gamma.to(device=device, dtype=dtype)

    # Finding #43 — validate gamma table shapes. Index-wrap on (21, 21)
    # tables would silently produce off-by-one γ values.
    if tuple(gamma_direct.shape) != (20, 20):
        raise ValueError(
            f"gamma_direct must have shape (20, 20); got {tuple(gamma_direct.shape)}."
        )
    if tuple(gamma_mediated_protein.shape) != (20, 20):
        raise ValueError(
            f"gamma_mediated_protein must have shape (20, 20); got "
            f"{tuple(gamma_mediated_protein.shape)}."
        )
    if tuple(gamma_mediated_water.shape) != (20, 20):
        raise ValueError(
            f"gamma_mediated_water must have shape (20, 20); got "
            f"{tuple(gamma_mediated_water.shape)}."
        )
    if tuple(burial_gamma.shape) != (20, 3):
        raise ValueError(
            f"burial_gamma must have shape (20, 3); got {tuple(burial_gamma.shape)}."
        )

    # --- per-decoy gamma gathers (ELEMENTWISE, length n_decoys) ------------
    # Pattern: ``g[aa_i, aa_j]`` with ``aa_i`` and ``aa_j`` both 1-D returns a
    # 1-D result. The outer-product pattern is reserved for the native
    # ``(N, N)`` sum elsewhere.
    g_dir = gamma_direct[aa_i, aa_j]                                      # (n_decoys,)
    g_mp = gamma_mediated_protein[aa_i, aa_j]                             # (n_decoys,)
    g_mw = gamma_mediated_water[aa_i, aa_j]                               # (n_decoys,)

    # --- sigmoid switches ---------------------------------------------------
    eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
    eta_sigma_t = torch.as_tensor(eta_sigma, dtype=dtype, device=device)
    rho_0_t = torch.as_tensor(rho_0, dtype=dtype, device=device)

    theta_direct = water_theta(rij, direct_r_min, direct_r_max, eta_t)    # (n_decoys,)
    theta_mediated = water_theta(rij, mediated_r_min, mediated_r_max, eta_t)

    # σ_wat(ρ_i, ρ_j) = ¼(1 - tanh(κ_σ(ρ_i - ρ_0)))(1 - tanh(κ_σ(ρ_j - ρ_0)))
    # — single 0.25 product, matches `fix_backbone.cpp:5459`.
    sigma_wat = (
        0.25
        * (1.0 - torch.tanh(eta_sigma_t * (rho_i - rho_0_t)))
        * (1.0 - torch.tanh(eta_sigma_t * (rho_j - rho_0_t)))
    )
    sigma_prot = 1.0 - sigma_wat

    # --- V_water per decoy --------------------------------------------------
    # σ_γ_direct = (γ0 + γ1) / 2 in the C++ — both columns are identical for
    # the direct block, so we just use ``g_dir``. Matches our Phase 2a
    # convention.
    sigma_gamma_direct = g_dir
    sigma_gamma_mediated = sigma_prot * g_mp + sigma_wat * g_mw

    k_water_t = torch.as_tensor(k_water, dtype=dtype, device=device)
    v_water = -k_water_t * (
        sigma_gamma_direct * theta_direct + sigma_gamma_mediated * theta_mediated
    )                                                                      # (n_decoys,)

    # --- V_burial per decoy (sum over the 3 wells, for both i and j) -------
    kappa_t = torch.as_tensor(burial_kappa, dtype=dtype, device=device)
    k_burial_t = torch.as_tensor(k_burial, dtype=dtype, device=device)

    # burial gamma table is (20, 3) — gather (n_decoys, 3) rows by aa index.
    bg_i = burial_gamma[aa_i]                                             # (n_decoys, 3)
    bg_j = burial_gamma[aa_j]                                             # (n_decoys, 3)

    # Per-well switch values: (n_decoys, 3). VECTORISED — no Python loop over wells.
    # Phase 3a review M-3 fix: was a Python `for w_idx in range(3)` loop; replaced with
    # a single broadcast op so Phase 3b's mutational mode (N²/2 pairs) doesn't pay
    # the per-pair Python overhead.
    rho_min_t = torch.as_tensor(burial_rho_min, dtype=dtype, device=device)   # (3,)
    rho_max_t = torch.as_tensor(burial_rho_max, dtype=dtype, device=device)   # (3,)

    def _burial_total(rho_vec: torch.Tensor, bg_per_decoy: torch.Tensor) -> torch.Tensor:
        # rho_vec: (n_decoys,); bg_per_decoy: (n_decoys, 3)
        # Broadcast: rho_vec.unsqueeze(-1) is (n_decoys, 1); rho_min_t is (3,) → (n_decoys, 3)
        t_min = torch.tanh(kappa_t * (rho_vec.unsqueeze(-1) - rho_min_t))     # (n_decoys, 3)
        t_max = torch.tanh(kappa_t * (rho_max_t - rho_vec.unsqueeze(-1)))     # (n_decoys, 3)
        # ``-0.5 × k_burial × γ × (t_min + t_max)`` per well, summed (C++ :5495-5497)
        return -0.5 * k_burial_t * (bg_per_decoy * (t_min + t_max)).sum(dim=-1)

    v_burial_i = _burial_total(rho_i, bg_i)
    v_burial_j = _burial_total(rho_j, bg_j)

    decoy_energies = v_water + v_burial_i + v_burial_j                    # (n_decoys,)

    decoy_mean = decoy_energies.mean()
    # population std (ddof=0) matches the C++ compute_array_std
    decoy_std = decoy_energies.std(unbiased=False)

    # Finding #22 — emit a warning if the entire decoy std collapsed.
    if bool((decoy_std == 0).item()):
        warnings.warn(
            "compute_configurational_decoy_energy: decoy_std == 0; FI would "
            "be undefined for every native pair using this configurational "
            "cache.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "decoy_energies": decoy_energies,
        "decoy_mean": decoy_mean,
        "decoy_std": decoy_std,
        # also forward the intermediate per-decoy breakdown for debugging
        "v_water": v_water,
        "v_burial_i": v_burial_i,
        "v_burial_j": v_burial_j,
    }


# --- thin convenience wrapper -------------------------------------------------
def configurational_decoy_stats(
    coords: dict[str, torch.Tensor],
    rho: torch.Tensor | None = None,
    *,
    n_decoys: int = DEFAULT_N_DECOYS,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF_A,
    min_seq_sep_rho: int = LAMMPS_DUMP_RHO_MIN_SEQ_SEP,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """One-stop: sample + compute. Returns the stats + the raw decoys.

    Parameters
    ----------
    min_seq_sep_rho : int
        Sequence-separation cutoff used for the internal ``lammps_dump_rho``
        computation when ``rho`` is None. Default 12 matches frustratometeR's
        ``lmp_serial_12_Linux`` binary (default SeqDist=12). Pass 3 to match
        ``lmp_serial_3_Linux`` (SeqDist=3 binary), or 2 to match OpenAWSEM's
        ``min_sequence_separation_rho``.

    If ``rho`` is None, it is computed from :func:`lammps_dump_rho` (which
    matches the LAMMPS-AWSEM dump's rho values used in the decoy formula).
    Caller may pass ``rho=burial_energy(coords)["rho"]`` to compare against
    the "physically correct" rho — the resulting decoy stats will NOT
    match the LAMMPS dump (off by ~30%), but they will reflect a self-
    consistent calculation using the same rho convention as V_burial.

    The returned dict has the union of keys from
    :func:`sample_configurational_decoys` and
    :func:`compute_configurational_decoy_energy`.
    """
    if rho is None:
        rho = lammps_dump_rho(coords, min_seq_sep=min_seq_sep_rho, device=device)
    decoys = sample_configurational_decoys(
        coords=coords,
        rho=rho,
        n_decoys=n_decoys,
        contact_cutoff=contact_cutoff,
        seed=seed,
        device=device,
    )
    energy_dict = compute_configurational_decoy_energy(
        decoys=decoys,
        device=device,
        dtype=dtype,
    )
    # merge (decoy_energies/mean/std + breakdown wins)
    out = {**decoys, **energy_dict}
    return out


__all__ = [
    "DEFAULT_CONTACT_CUTOFF_A",
    "DEFAULT_N_DECOYS",
    "LAMMPS_DUMP_RHO_MIN_SEQ_SEP",
    "burial_switch",
    "compute_configurational_decoy_energy",
    "configurational_decoy_stats",
    "lammps_dump_rho",
    "sample_configurational_decoys",
    "water_theta",
]
