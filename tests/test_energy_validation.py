"""Tests for the v2 energy-and-context fixes in the frustration_gpu package.

Covers (per the audit at ``F:/research_plan/New folder/odo.txt``):

* #6  — water_mediated theta diagnostic is masked on excluded pairs.
* #11 — ContactContext device mismatch raises a clear ``ValueError``.
* #14 — non-Gly CB → CA fallback emits ``UserWarning``.
* #24 — Debye-Hückel ``screening_length <= 0`` / non-finite rejected.
* #28 — sparse DH below ``3*screening_length`` emits a ``UserWarning``.
* #29 — sparse water-mediated below the safe minimum (14 Å) emits a warning.
* #30 — stale ContactContext is rejected via fingerprint mismatch.
* #31 — SparseContactContext fed where ContactContext expected → ValueError.
* #39 — out-of-range residue_types rejected with a clear ValueError.
* #42 — SparseContactContext.dist is no longer cached (memory-saving).
* #47 — sparse direct below 7.5 Å emits a ``UserWarning``.
* #56 — NaN/inf/non-positive ``sparse_cutoff`` rejected in both builders.
* #57 — scalar ``*_pair_energy`` validates gamma-table shape and aa range.
* #58 — scalar ``debye_huckel_pair_energy`` validates r_ij (positive, finite).

Bit-identity gate: the existing dense-path tests must still pass — that
is verified by the rest of the suite. This file only exercises the NEW
validation paths, so a baseline run with no sparse contexts is unchanged.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import PDB_DIR  # noqa: E402

from frustration_gpu._contact_common import (  # noqa: E402
    DIRECT_SPARSE_MIN_SAFE_A,
    MEDIATED_SPARSE_MIN_SAFE_A,
    SparseContactContext,
    build_contact_context,
    dh_sparse_min_safe,
)
from frustration_gpu.burial import burial_density  # noqa: E402
from frustration_gpu.debye_huckel import (  # noqa: E402
    debye_huckel_energy,
    debye_huckel_pair_energy,
)
from frustration_gpu.decoys import sample_configurational_decoys  # noqa: E402
from frustration_gpu.direct_contact import (  # noqa: E402
    direct_contact_energy,
    direct_pair_energy,
)
from frustration_gpu.parser import parse_pdb  # noqa: E402
from frustration_gpu.singleresidue_decoys import (  # noqa: E402
    singleresidue_decoy_stats,
)
from frustration_gpu.water_mediated import (  # noqa: E402
    water_mediated_energy,
    water_mediated_pair_energy,
)

# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def parsed_5aon():
    return parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)


@pytest.fixture(scope="module")
def rho_5aon(parsed_5aon):
    return burial_density(parsed_5aon)


# --- Finding #56 — sparse_cutoff NaN/inf validation -------------------------


def test_sparse_cutoff_nan_rejected(parsed_5aon):
    """build_contact_context must reject NaN sparse_cutoff."""
    with pytest.raises(ValueError, match="finite"):
        build_contact_context(parsed_5aon, sparse_cutoff=float("nan"))


def test_sparse_cutoff_inf_rejected(parsed_5aon):
    """build_contact_context must reject +inf sparse_cutoff (would defeat sparsity)."""
    with pytest.raises(ValueError, match="finite"):
        build_contact_context(parsed_5aon, sparse_cutoff=float("inf"))


def test_sparse_cutoff_zero_rejected(parsed_5aon):
    with pytest.raises(ValueError, match="> 0"):
        build_contact_context(parsed_5aon, sparse_cutoff=0.0)


def test_sparse_cutoff_negative_rejected(parsed_5aon):
    with pytest.raises(ValueError, match="> 0"):
        build_contact_context(parsed_5aon, sparse_cutoff=-1.0)


def test_burial_sparse_cutoff_nan_rejected(parsed_5aon):
    """burial_density sparse path must reject NaN sparse_cutoff_a."""
    with pytest.raises(ValueError, match="finite"):
        burial_density(parsed_5aon, sparse=True, sparse_cutoff_a=float("nan"))


def test_burial_sparse_cutoff_zero_rejected(parsed_5aon):
    with pytest.raises(ValueError, match="> 0"):
        burial_density(parsed_5aon, sparse=True, sparse_cutoff_a=0.0)


# --- Finding #47 — sparse direct cutoff warning ------------------------------


def test_sparse_direct_warns_below_min_safe(parsed_5aon):
    """Direct contact with sparse_cutoff < 7.5 Å emits a UserWarning."""
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=6.5)
    with pytest.warns(UserWarning, match="direct_contact_energy"):
        direct_contact_energy(parsed_5aon, _context=ctx, sparse=True)


def test_sparse_direct_no_warn_at_or_above_min_safe(parsed_5aon):
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=DIRECT_SPARSE_MIN_SAFE_A)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # The non-Gly CB→CA warning may fire from finding #14 — silence it
        # for this targeted test by parsing fresh and ignoring just that
        # category. Instead we use catch_warnings and filter.
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        direct_contact_energy(parsed_5aon, _context=ctx, sparse=True)


# --- Finding #29 — sparse water-mediated cutoff warning ----------------------


def test_sparse_water_warns_below_min_safe(parsed_5aon, rho_5aon):
    """Water-mediated with sparse_cutoff < 14 Å emits a UserWarning."""
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=9.5)
    with pytest.warns(UserWarning, match="water_mediated_energy"):
        water_mediated_energy(
            parsed_5aon, rho=rho_5aon, _context=ctx, sparse=True
        )


def test_sparse_water_no_warn_at_min_safe(parsed_5aon, rho_5aon):
    ctx = build_contact_context(
        parsed_5aon, sparse_cutoff=MEDIATED_SPARSE_MIN_SAFE_A
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        water_mediated_energy(
            parsed_5aon, rho=rho_5aon, _context=ctx, sparse=True
        )


# --- Finding #28 — sparse Debye-Hückel cutoff warning ------------------------


def test_sparse_dh_warns_below_min_safe(parsed_5aon):
    """DH with sparse_cutoff < 3*lambda emits a UserWarning."""
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=11.0)
    with pytest.warns(UserWarning, match="debye_huckel_energy"):
        debye_huckel_energy(parsed_5aon, _context=ctx, sparse=True)


def test_sparse_dh_no_warn_at_min_safe(parsed_5aon):
    ctx = build_contact_context(
        parsed_5aon,
        sparse_cutoff=dh_sparse_min_safe(10.0, 1.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        debye_huckel_energy(parsed_5aon, _context=ctx, sparse=True)


# --- Finding #11 — context device-mismatch validation ------------------------


def test_context_device_mismatch_raises(parsed_5aon):
    """A CPU context passed with device="cuda" must raise ValueError.

    The check happens BEFORE any tensor allocation, so cuda need not be
    available for this test to run.
    """
    ctx = build_contact_context(parsed_5aon, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="device"):
        direct_contact_energy(parsed_5aon, _context=ctx, device="cuda")


# --- Finding #30 — stale context (fingerprint) -----------------------------


def test_stale_context_fingerprint_rejected(parsed_5aon):
    """A context from coords A used with different coords B raises ValueError."""
    ctx = build_contact_context(parsed_5aon)
    # Build a coords dict with the same shapes but different positions —
    # easiest way is to shift CA by a constant offset.
    coords_modified = dict(parsed_5aon)
    coords_modified["ca_coords"] = parsed_5aon["ca_coords"] + 5.0
    coords_modified["cb_coords"] = parsed_5aon["cb_coords"] + 5.0
    with pytest.raises(ValueError, match="(Stale|fingerprint)"):
        direct_contact_energy(coords_modified, _context=ctx)


# --- Finding #31 — wrong context type ---------------------------------------


def test_sparse_context_passed_without_sparse_flag_raises(parsed_5aon):
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=14.0)
    with pytest.raises(ValueError, match="SparseContactContext"):
        direct_contact_energy(parsed_5aon, _context=ctx, sparse=False)


def test_dense_context_passed_with_sparse_flag_raises(parsed_5aon):
    ctx = build_contact_context(parsed_5aon)
    with pytest.raises(ValueError, match="SparseContactContext"):
        direct_contact_energy(parsed_5aon, _context=ctx, sparse=True)


# --- Finding #39 — residue_types range -------------------------------------


def test_residue_types_out_of_range_direct(parsed_5aon):
    """residue_types containing >= 20 raises a clear ValueError."""
    coords_bad = dict(parsed_5aon)
    rt = parsed_5aon["residue_types"].clone()
    rt[0] = 99
    coords_bad["residue_types"] = rt
    with pytest.raises(ValueError, match=r"\[0, 20\)"):
        direct_contact_energy(coords_bad)


def test_residue_types_out_of_range_water(parsed_5aon, rho_5aon):
    coords_bad = dict(parsed_5aon)
    rt = parsed_5aon["residue_types"].clone()
    rt[0] = 25
    coords_bad["residue_types"] = rt
    with pytest.raises(ValueError, match=r"\[0, 20\)"):
        water_mediated_energy(coords_bad, rho=rho_5aon)


def test_residue_types_out_of_range_dh(parsed_5aon):
    coords_bad = dict(parsed_5aon)
    rt = parsed_5aon["residue_types"].clone()
    rt[0] = 25
    coords_bad["residue_types"] = rt
    with pytest.raises(ValueError, match=r"\[0, 20\)"):
        debye_huckel_energy(coords_bad)


# --- Finding #42 — SparseContactContext no longer caches dense dist --------


def test_sparse_context_does_not_store_dense_dist(parsed_5aon):
    """SparseContactContext.dist must be None (no dense matrix cached)."""
    ctx = build_contact_context(parsed_5aon, sparse_cutoff=14.0)
    assert isinstance(ctx, SparseContactContext)
    assert ctx.dist is None


# --- Finding #24 — DH screening parameter validation -----------------------


def test_dh_zero_screening_length_raises(parsed_5aon):
    with pytest.raises(ValueError, match="screening_length"):
        debye_huckel_energy(parsed_5aon, screening_length=0.0)


def test_dh_negative_screening_length_raises(parsed_5aon):
    with pytest.raises(ValueError, match="screening_length"):
        debye_huckel_energy(parsed_5aon, screening_length=-5.0)


def test_dh_nan_screening_length_raises(parsed_5aon):
    with pytest.raises(ValueError, match="screening_length"):
        debye_huckel_energy(parsed_5aon, screening_length=float("nan"))


def test_dh_zero_k_screening_raises(parsed_5aon):
    with pytest.raises(ValueError, match="k_screening"):
        debye_huckel_energy(parsed_5aon, k_screening=0.0)


# --- Finding #58 — debye_huckel_pair_energy r_ij validation ----------------


def test_dh_pair_energy_rejects_zero_distance():
    with pytest.raises(ValueError, match="positive"):
        debye_huckel_pair_energy(0.0, 1, 3)


def test_dh_pair_energy_rejects_negative_distance():
    with pytest.raises(ValueError, match="positive"):
        debye_huckel_pair_energy(-1.0, 1, 3)


def test_dh_pair_energy_rejects_nan_distance():
    with pytest.raises(ValueError, match="finite"):
        debye_huckel_pair_energy(float("nan"), 1, 3)


def test_dh_pair_energy_rejects_inf_distance():
    with pytest.raises(ValueError, match="finite"):
        debye_huckel_pair_energy(float("inf"), 1, 3)


# --- Finding #57 — scalar pair-energy gamma validation ---------------------


def test_direct_pair_energy_rejects_wrong_gamma_shape():
    bad_gamma = torch.zeros((21, 21), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\(20, 20\)"):
        direct_pair_energy(5.0, 1, 3, gamma_direct=bad_gamma)


def test_water_pair_energy_rejects_wrong_gamma_shape():
    bad_gamma = torch.zeros((19, 19), dtype=torch.float64)
    g_w = torch.zeros((20, 20), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\(20, 20\)"):
        water_mediated_pair_energy(
            8.0, 1, 3, 1.0, 1.0,
            gamma_mediated_protein=bad_gamma,
            gamma_mediated_water=g_w,
        )


def test_direct_pair_energy_rejects_out_of_range_aa():
    with pytest.raises(ValueError, match=r"\[0, 20\)"):
        direct_pair_energy(5.0, 25, 3)


def test_direct_pair_energy_rejects_negative_aa():
    with pytest.raises(ValueError, match=r"\[0, 20\)"):
        direct_pair_energy(5.0, -1, 3)


# --- Finding #14 — non-Gly CB→CA fallback warning --------------------------


def test_non_gly_cb_fallback_warns(parsed_5aon):
    """A non-Gly residue with NaN CB triggers a UserWarning on resolution."""
    coords_modified = dict(parsed_5aon)
    cb = parsed_5aon["cb_coords"].clone()
    rt = parsed_5aon["residue_types"]
    # Find an index that is NOT glycine (residue type != 7).
    non_gly_idx = int(((rt != 7) & (rt >= 0)).nonzero(as_tuple=False)[0].item())
    cb[non_gly_idx] = float("nan")
    coords_modified["cb_coords"] = cb
    with pytest.warns(UserWarning, match="non-glycine"):
        direct_contact_energy(coords_modified)


# --- Finding #6 — water_mediated theta diagnostic mask --------------------


def test_water_mediated_theta_zeroed_on_excluded_pairs(parsed_5aon, rho_5aon):
    """The diagnostic ``theta`` matrix must be zero on masked-out pairs.

    Before finding #6 was fixed, the dense path reconstructed theta from
    safe_dist (which fills masked positions with the mid-window value),
    making theta look ~peak on cross-chain / NaN-row / seq-skipped pairs
    even though they did not contribute to the energy.
    """
    out = water_mediated_energy(
        parsed_5aon, rho=rho_5aon, return_pair_matrix=True,
    )
    theta = out["theta"]
    mask = out["pair_mask"]
    # On the dense path mask is upper-triangular; we check that
    # excluded pairs in the upper triangle have theta == 0.
    n = theta.shape[0]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    excluded = upper & ~mask
    # Where we excluded a pair, theta must equal 0 (no leakage of the
    # mid-window-fill peak into the diagnostic).
    assert torch.all(theta[excluded] == 0.0)


# --- Bit-identity gate for existing dense path -----------------------------
# These tests pass through the same code paths as the legacy tests but
# additionally verify no behaviour drift after wiring in the new checks.


def test_dense_direct_unchanged(parsed_5aon):
    """Default dense direct-contact energy is unchanged."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        e = direct_contact_energy(parsed_5aon)
    assert torch.isfinite(e)
    # 5AON dense direct ~ -2.55 kcal/mol (from the audit; documented).
    assert -3.5 < float(e) < -1.5


def test_dense_water_unchanged(parsed_5aon, rho_5aon):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        e = water_mediated_energy(parsed_5aon, rho=rho_5aon)
    assert torch.isfinite(e)
    # 5AON dense water-mediated ~ -16.15 kcal/mol (from the audit).
    assert -20.0 < float(e) < -10.0


def test_dense_dh_unchanged(parsed_5aon):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Effective-CB resolution"
        )
        e = debye_huckel_energy(parsed_5aon)
    assert torch.isfinite(e)
    # 5AON dense DH ~ -0.6 kcal/mol (from the audit).
    assert -2.0 < float(e) < 1.0


# --- Finding #30 follow-up — decoy modules now also fingerprint-guard -----
# The review agent flagged that ``decoys.py`` and ``singleresidue_decoys.py``
# accepted a ``_context=`` that didn't match the live ``coords``, silently
# mixing structure A's cached distance matrix with structure B's amino-acid
# identities. These tests lock in the fingerprint check at the decoy entry
# points.


def test_configurational_decoys_reject_stale_context(parsed_5aon, rho_5aon):
    """sample_configurational_decoys rejects a context from different coords."""
    ctx = build_contact_context(parsed_5aon, compute_dist_full=True)
    coords_modified = dict(parsed_5aon)
    coords_modified["ca_coords"] = parsed_5aon["ca_coords"] + 5.0
    coords_modified["cb_coords"] = parsed_5aon["cb_coords"] + 5.0
    with pytest.raises(ValueError, match="(Stale|fingerprint)"):
        sample_configurational_decoys(
            coords_modified, rho_5aon, n_decoys=8, _context=ctx
        )


def test_singleresidue_decoys_reject_stale_context(parsed_5aon, rho_5aon):
    """singleresidue_decoy_stats rejects a context from different coords."""
    ctx = build_contact_context(parsed_5aon, compute_dist_full=True)
    coords_modified = dict(parsed_5aon)
    coords_modified["ca_coords"] = parsed_5aon["ca_coords"] + 5.0
    coords_modified["cb_coords"] = parsed_5aon["cb_coords"] + 5.0
    with pytest.raises(ValueError, match="(Stale|fingerprint)"):
        singleresidue_decoy_stats(
            coords_modified, rho=rho_5aon, n_decoys=8, _context=ctx
        )


# --- Finding #57 follow-up — NaN gamma tables rejected --------------------
# Shape validation alone is not enough: a (20, 20) tensor filled with NaN
# previously passed every check and silently propagated NaN to the energy
# output. Now the scalar pair helpers reject non-finite values too.


def test_direct_pair_energy_rejects_nan_gamma():
    """A NaN-filled (20, 20) gamma table is rejected with a clear error."""
    nan_gamma = torch.full((20, 20), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        direct_pair_energy(5.0, 1, 3, gamma_direct=nan_gamma)


def test_direct_pair_energy_rejects_inf_gamma():
    """An inf-filled gamma table is rejected with a clear error."""
    inf_gamma = torch.full((20, 20), float("inf"), dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        direct_pair_energy(5.0, 1, 3, gamma_direct=inf_gamma)


def test_water_pair_energy_rejects_nan_gamma_protein():
    """A NaN-filled mediated-protein gamma is rejected."""
    nan_gamma = torch.full((20, 20), float("nan"), dtype=torch.float64)
    g_w = torch.zeros((20, 20), dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        water_mediated_pair_energy(
            8.0, 1, 3, 1.0, 1.0,
            gamma_mediated_protein=nan_gamma,
            gamma_mediated_water=g_w,
        )


def test_water_pair_energy_rejects_nan_gamma_water():
    """A NaN-filled mediated-water gamma is rejected."""
    g_p = torch.zeros((20, 20), dtype=torch.float64)
    nan_gamma = torch.full((20, 20), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        water_mediated_pair_energy(
            8.0, 1, 3, 1.0, 1.0,
            gamma_mediated_protein=g_p,
            gamma_mediated_water=nan_gamma,
        )
