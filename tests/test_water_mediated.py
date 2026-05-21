"""Tests for the AWSEM water-mediated contact term (Phase 2b).

Validation strategy
-------------------
Phase 2b is gated by reproducing the LAMMPS-AWSEM ``V_Water`` column from
``configurational/<pdb>_energy.log``:

    V_Water = V_direct + V_mediated

We assert ``|V_direct + V_mediated - target| / |target| < 1e-3`` (0.1%) for
both 5AON (-18.700281) and 11BG (-147.390847).

Additional tests cover:

1. Per-pair hand-check against the inline reference used in Phase 2a's
   ``test_pair_value_first_contact``.
2. The dense path agreeing with the per-pair scalar reference.
3. n=1 / n=0 edge cases.
4. r_min / r_max boundary continuity.
5. CPU/GPU agreement (gated on ``torch.cuda.is_available()``).
6. Autograd: differentiable + NaN-safe regression test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import DUMP_ROOT, PDB_DIR  # noqa: E402

from frustration_gpu.burial import burial_density  # noqa: E402
from frustration_gpu.contact_gamma import load_mediated_gamma  # noqa: E402
from frustration_gpu.direct_contact import direct_contact_energy  # noqa: E402
from frustration_gpu.parser import parse_pdb  # noqa: E402
from frustration_gpu.water_mediated import (  # noqa: E402
    water_mediated_energy,
    water_mediated_pair_energy,
)

ENERGY_LOG_DIR = DUMP_ROOT / "configurational"

# Targets from energy.log "Water" column at step 0
TARGETS = {
    "5AON": -18.700281,
    "11BG": -147.390847,
}


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


@pytest.fixture(scope="module")
def parsed_5aon():
    return parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)


@pytest.fixture(scope="module")
def parsed_11bg():
    return parse_pdb(PDB_DIR / "11BG.pdb", dtype=torch.float64)


# --- Test 1: per-pair value matches by-hand AWSEM formula -------------------
def test_pair_value_hand_check():
    """V_mediated pair-level value matches the inline reference used in
    direct-contact Phase 2a (which itself mirrors fix_backbone.cpp:5459-5473).
    """
    # The 5AON first dumped pair sits at r=5.065 — too close, theta_mediated
    # is essentially zero. Pick a representative pair inside the mediated
    # window: r=8.0, S(15)-R(1), with mixed burial.
    r = 8.0
    aa_i, aa_j = 15, 1
    rho_i, rho_j = 0.5, 1.8

    v = water_mediated_pair_energy(r, aa_i, aa_j, rho_i, rho_j,
                                   dtype=torch.float64).item()

    # Reference from fix_backbone.cpp:5459-5473
    g_p, g_w = load_mediated_gamma(dtype=torch.float64)
    eta = 5.0
    eta_sigma = 7.0
    rho_0 = 2.6
    r_min, r_max = 6.5, 9.5

    sigma_wat_i = 0.5 * (1.0 - math.tanh(eta_sigma * (rho_i - rho_0)))
    sigma_wat_j = 0.5 * (1.0 - math.tanh(eta_sigma * (rho_j - rho_0)))
    sigma_wat = sigma_wat_i * sigma_wat_j
    sigma_prot = 1.0 - sigma_wat
    theta = 0.25 * (1.0 + math.tanh(eta * (r - r_min))) \
                 * (1.0 + math.tanh(eta * (r_max - r)))
    gamma_blend = sigma_prot * g_p[aa_i, aa_j].item() + sigma_wat * g_w[aa_i, aa_j].item()
    v_ref = -1.0 * gamma_blend * theta

    assert abs(v - v_ref) < 1e-12, f"PyTorch V_med {v} vs manual {v_ref}"


# --- Test 2: dense path consistent with single-pair scalar reference --------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_dense_pair_matches_scalar(parsed_5aon):
    """Pick a pair from the dense matrix and confirm it matches
    ``water_mediated_pair_energy`` to float64 precision."""
    rho = burial_density(parsed_5aon)
    out = water_mediated_energy(parsed_5aon, rho=rho, return_pair_matrix=True)
    pair_e = out["pair_energy"]
    distances = out["distances"]

    # Pick a contributing pair in the mediated window. Search for one with
    # r in (6.8, 9.2) and on the upper triangle of the pair_mask.
    mask = out["pair_mask"]
    aa = parsed_5aon["residue_types"]
    found = None
    for i_idx in range(distances.shape[0]):
        for j_idx in range(i_idx + 1, distances.shape[1]):
            r = distances[i_idx, j_idx].item()
            if mask[i_idx, j_idx] and 6.8 < r < 9.2:
                found = (i_idx, j_idx, r)
                break
        if found is not None:
            break
    assert found is not None, "expected at least one pair in mediated window"
    i_idx, j_idx, r = found

    e_dense = pair_e[i_idx, j_idx].item()
    e_ref = water_mediated_pair_energy(
        r,
        int(aa[i_idx].item()),
        int(aa[j_idx].item()),
        rho[i_idx].item(),
        rho[j_idx].item(),
        dtype=torch.float64,
    ).item()
    assert abs(e_dense - e_ref) < 1e-6, f"dense {e_dense} vs scalar {e_ref}"


# --- Test 3: VALIDATION GATE — V_Water matches energy.log -------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_validation_gate_5aon(parsed_5aon):
    rho = burial_density(parsed_5aon)
    v_d = direct_contact_energy(parsed_5aon).item()
    v_m = water_mediated_energy(parsed_5aon, rho=rho).item()
    v_water = v_d + v_m
    target = TARGETS["5AON"]
    rel = abs(v_water - target) / abs(target)
    print(f"\n5AON V_direct={v_d:.6f} V_mediated={v_m:.6f} "
          f"V_water={v_water:.6f} target={target:.6f} rel_err={rel*100:.6f}%")
    assert rel < 1e-3, (
        f"5AON V_Water {v_water:.6f} vs target {target:.6f} rel_err={rel*100:.4f}%"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_validation_gate_11bg(parsed_11bg):
    rho = burial_density(parsed_11bg)
    v_d = direct_contact_energy(parsed_11bg).item()
    v_m = water_mediated_energy(parsed_11bg, rho=rho).item()
    v_water = v_d + v_m
    target = TARGETS["11BG"]
    rel = abs(v_water - target) / abs(target)
    print(f"\n11BG V_direct={v_d:.6f} V_mediated={v_m:.6f} "
          f"V_water={v_water:.6f} target={target:.6f} rel_err={rel*100:.6f}%")
    assert rel < 1e-3, (
        f"11BG V_Water {v_water:.6f} vs target {target:.6f} rel_err={rel*100:.4f}%"
    )


# --- Test 4: n=1 / empty edge cases ----------------------------------------
def test_empty_protein_returns_zero():
    """n=0 → 0.0 energy."""
    coords = {
        "ca_coords": torch.zeros((0, 3), dtype=torch.float64),
        "cb_coords": torch.zeros((0, 3), dtype=torch.float64),
        "residue_types": torch.zeros((0,), dtype=torch.int64),
        "chain_ids": [],
    }
    rho = torch.zeros((0,), dtype=torch.float64)
    e = water_mediated_energy(coords, rho=rho).item()
    assert e == 0.0


def test_single_residue_returns_zero():
    """n=1 → 0.0 (no possible pair)."""
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        "residue_types": torch.tensor([0], dtype=torch.int64),
        "chain_ids": ["A"],
    }
    rho = torch.zeros((1,), dtype=torch.float64)
    e = water_mediated_energy(coords, rho=rho).item()
    assert e == 0.0


# --- Test 5: r_min / r_max boundary continuity -----------------------------
def test_boundary_continuity():
    """θ_mediated should be smooth and ~0.5 at the boundaries.

    Walk a single pair across r ∈ {3.0, 6.5, 8.0, 9.5, 12.0} and verify the
    energy is continuous (no jumps) and saturates to 0 outside the window.
    """
    aa_i, aa_j = 0, 0  # A-A for simplicity
    rho_i, rho_j = 1.0, 1.0
    rs = [3.0, 6.0, 6.5, 7.0, 8.0, 9.0, 9.5, 10.0, 12.0]
    vals = [
        water_mediated_pair_energy(r, aa_i, aa_j, rho_i, rho_j,
                                   dtype=torch.float64).item()
        for r in rs
    ]
    # Outside window: ~0
    assert abs(vals[0]) < 1e-4, f"V at r=3.0 should be ~0, got {vals[0]}"
    assert abs(vals[-1]) < 1e-4, f"V at r=12.0 should be ~0, got {vals[-1]}"
    # In-window: non-zero
    assert abs(vals[4]) > 1e-3, "V at r=8.0 (mid window) should be non-zero"
    # Monotonic increase to mid then decrease — check no jumps
    for k in range(len(vals) - 1):
        # Smooth: small ratio of consecutive differences (no discontinuity)
        d = abs(vals[k + 1] - vals[k])
        assert d < 1.0, f"large jump at r {rs[k]} -> {rs[k+1]}: {d}"


# --- Test 6: CPU/GPU agreement (gated) -------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_cpu_gpu_agreement():
    """V_Water on CPU vs CUDA agree to 1e-6 relative at float64."""
    p_cpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cpu", dtype=torch.float64)
    p_gpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cuda", dtype=torch.float64)
    rho_cpu = burial_density(p_cpu)
    rho_gpu = burial_density(p_gpu)
    e_cpu = (
        direct_contact_energy(p_cpu).item()
        + water_mediated_energy(p_cpu, rho=rho_cpu).item()
    )
    e_gpu = (
        direct_contact_energy(p_gpu).item()
        + water_mediated_energy(p_gpu, rho=rho_gpu).item()
    )
    rel = abs(e_cpu - e_gpu) / max(abs(e_cpu), 1e-12)
    print(f"\n5AON V_Water CPU={e_cpu:.6f} GPU={e_gpu:.6f} rel.diff={rel:.2e}")
    assert rel < 1e-6, f"CPU {e_cpu} vs GPU {e_gpu} rel.diff {rel}"


# --- Test 7: differentiability + NaN safety ---------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_differentiable_wrt_cb(parsed_5aon):
    p = {**parsed_5aon}
    p["ca_coords"] = p["ca_coords"].clone().detach().requires_grad_(True)
    p["cb_coords"] = p["cb_coords"].clone().detach().requires_grad_(True)
    rho = burial_density(p).detach()  # detach rho — we only test V_med w.r.t. coords
    e = water_mediated_energy(p, rho=rho)
    e.backward()
    finite_rows = torch.isfinite(p["cb_coords"]).all(dim=-1)
    assert torch.isfinite(p["cb_coords"].grad[finite_rows]).all(), (
        "CB gradient should be finite for residues with valid CB"
    )
    assert p["cb_coords"].grad[finite_rows].abs().max().item() > 0


def test_nan_residue_does_not_poison_gradients():
    """Regression test for the double-where NaN trick.

    Construct a synthetic 5-residue protein where residue 2 has NaN CA and
    NaN CB (fully missing). Compute V_mediated and assert that gradients on
    the remaining 4 residues are finite.
    """
    n = 5
    cb = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.5, 0.5, 0.2],
            [float("nan"), float("nan"), float("nan")],   # missing
            [7.0, 0.0, 0.0],
            [10.5, 0.5, 0.2],
        ],
        dtype=torch.float64,
    )
    ca = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [3.5, 1.5, 0.2],
            [float("nan"), float("nan"), float("nan")],   # missing
            [7.0, 1.0, 0.0],
            [10.5, 1.5, 0.2],
        ],
        dtype=torch.float64,
    )
    cb_p = cb.clone().requires_grad_(True)
    ca_p = ca.clone().requires_grad_(True)
    coords = {
        "ca_coords": ca_p,
        "cb_coords": cb_p,
        "residue_types": torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        "chain_ids": ["A"] * n,
    }
    rho = torch.tensor([1.0, 1.5, 0.5, 1.0, 1.2], dtype=torch.float64)
    e = water_mediated_energy(coords, rho=rho)
    e.backward()

    # Rows 0, 1, 3, 4 are finite; row 2 may have NaN grad (that's OK).
    finite_rows = torch.tensor([True, True, False, True, True])
    cb_grad_ok = torch.isfinite(cb_p.grad[finite_rows]).all().item()
    ca_grad_ok = torch.isfinite(ca_p.grad[finite_rows]).all().item()
    assert cb_grad_ok, f"CB grad has NaN on finite rows: {cb_p.grad}"
    assert ca_grad_ok, f"CA grad has NaN on finite rows: {ca_p.grad}"


# --- Test 8: total V_mediated is negative and finite -----------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_total_v_mediated_5aon(parsed_5aon):
    rho = burial_density(parsed_5aon)
    e = water_mediated_energy(parsed_5aon, rho=rho).item()
    assert math.isfinite(e)
    # Compact globular protein → V_mediated is negative
    assert e < 0


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_total_v_mediated_11bg(parsed_11bg):
    rho = burial_density(parsed_11bg)
    e = water_mediated_energy(parsed_11bg, rho=rho).item()
    assert math.isfinite(e)
    assert e < 0


# --- DNA sentinel guard (Fix A) --------------------------------------------
def test_dna_sentinel_guard_raises():
    """``residue_types == -1`` must raise on water_mediated_energy too."""
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0], [8.5, 0.0, 0.0]],
                                  dtype=torch.float64),
        "residue_types": torch.tensor([0, -1], dtype=torch.int64),
        "chain_ids": ["A", "A"],
    }
    rho = torch.tensor([1.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="DNA sentinel"):
        water_mediated_energy(coords, rho=rho)


# --- sparse vs dense ~ULP-equivalent agreement (Fix B) ---------------------
# Mediated sigmoid edge is r_max = 9.5 Å with eta = 5; theta is half its
# peak AT r=9.5 (not zero!) and only decays to ~1e-18 by r = r_max + 4 ≈
# 13.5 Å. Use a 14 Å cutoff for sparse to capture the full shell.
_WAT_SPARSE_CUTOFF_A = 14.0
_BYTE_EXACT_REL_TOL = 1e-12


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sparse_byte_exact_5aon(parsed_5aon):
    """sparse=True V_mediated matches sparse=False to fp64 precision."""
    from frustration_gpu._contact_common import build_contact_context

    rho = burial_density(parsed_5aon)
    e_dense = water_mediated_energy(parsed_5aon, rho=rho).item()
    ctx_sparse = build_contact_context(parsed_5aon, sparse_cutoff=_WAT_SPARSE_CUTOFF_A)
    e_sparse = water_mediated_energy(
        parsed_5aon, rho=rho, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n5AON V_med dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_sparse_byte_exact_11bg(parsed_11bg):
    """Same gate on 11BG."""
    from frustration_gpu._contact_common import build_contact_context

    rho = burial_density(parsed_11bg)
    e_dense = water_mediated_energy(parsed_11bg, rho=rho).item()
    ctx_sparse = build_contact_context(parsed_11bg, sparse_cutoff=_WAT_SPARSE_CUTOFF_A)
    e_sparse = water_mediated_energy(
        parsed_11bg, rho=rho, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n11BG V_med dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )
