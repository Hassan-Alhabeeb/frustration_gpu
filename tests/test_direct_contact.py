"""Tests for the AWSEM direct-contact term (Phase 2a).

Validation strategy
-------------------
The dumped ``5AON_tertiary_frustration.dat`` carries a per-pair
``E_native`` column that is ``V_water + E_burial(i) + E_burial(j)`` — i.e.
``V_direct + V_mediated + 2 burials``. We can't isolate ``V_direct`` from
the dump in Phase 2a (mediated is Phase 2b). Pragmatic gates instead:

1. **Hand-computed pair-level value.** Compute ``V_direct`` for 5AON's first
   dumped contact (i=1, j=3, S-R, r=5.065 Å) using the AWSEM formula by
   hand, and confirm our PyTorch implementation matches to ``1e-6``.

2. **Native-energy reconstruction.** For the same pair, verify
   ``V_direct + V_mediated + E_burial(i) + E_burial(j) ≈ E_native``. Since
   we don't have ``V_mediated`` implemented yet, we instead inline a tiny
   reference ``V_mediated`` here (~10 LOC) and use it ONLY for this
   validation check. This effectively proves direct-contact correctness on
   the protein scale even without Phase 2b code.

3. **Size monotonicity / no NaN.** Total ``V_direct`` over 5AON and 11BG is
   finite and 11BG's |V_direct| > 5AON's |V_direct| (11BG has ~5x more
   residues, so more pairs, so larger magnitude).

4. **CPU/GPU agreement.** Same protein on cpu vs cuda gives identical total
   energy to ``1e-6`` relative.

5. **Differentiability.** ``V_direct.backward()`` produces a finite gradient
   on the CB coords.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Make the ``frustration_gpu`` package importable when running pytest from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import DUMP_ROOT, PDB_DIR  # noqa: E402

from frustration_gpu.contact_gamma import load_direct_gamma  # noqa: E402
from frustration_gpu.direct_contact import (  # noqa: E402
    direct_contact_energy,
    direct_pair_energy,
)
from frustration_gpu.parameters import load_gamma_tables  # noqa: E402
from frustration_gpu.parser import parse_pdb  # noqa: E402

# 5AON tertiary frustration dump. The cpu_baseline tree was reorganised on
# 2026-05-20 to split outputs by mode; the configurational dump is the file
# the Phase 1.5 spec audit was performed against.
_DUMP_CANDIDATES = (
    DUMP_ROOT / "configurational" / "5AON_tertiary_frustration.dat",
    DUMP_ROOT / "5AON_tertiary_frustration.dat",
)
DUMP_PATH = next((p for p in _DUMP_CANDIDATES if p.is_file()), _DUMP_CANDIDATES[0])


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


@pytest.fixture(scope="module")
def parsed_5aon():
    # float64 for tighter parity tolerance vs the C++ dump.
    return parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)


@pytest.fixture(scope="module")
def parsed_11bg():
    return parse_pdb(PDB_DIR / "11BG.pdb", dtype=torch.float64)


def _load_dump_rows() -> list[tuple[int, int, int, int, float, float, float, str, str, float]]:
    """Read columns we care about from the tertiary_frustration.dat dump.

    Returns a list of tuples (i, j, chain_i, chain_j, r_ij, rho_i, rho_j, a_i,
    a_j, E_native), one per data row. Residue indices in the dump are
    1-indexed (LAMMPS convention); we keep them 1-indexed here.
    """
    rows = []
    with DUMP_PATH.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 19:
                continue
            i, j = int(parts[0]), int(parts[1])
            ci, cj = int(parts[2]), int(parts[3])
            r_ij = float(parts[10])
            rho_i = float(parts[11])
            rho_j = float(parts[12])
            a_i = parts[13]
            a_j = parts[14]
            e_native = float(parts[15])
            rows.append((i, j, ci, cj, r_ij, rho_i, rho_j, a_i, a_j, e_native))
    return rows


def _ref_mediated_pair_kcal(r_ij: float, aa_i: int, aa_j: int,
                            rho_i: float, rho_j: float) -> float:
    """Minimal reference ``V_mediated`` used ONLY by the cross-check below.

    Phase 2b will replace this with a proper module. Kept here at ~10 LOC so
    we can validate direct-contact correctness against the full
    ``E_native`` reconstruction without depending on unwritten code.

    Mirrors ``fix_backbone.cpp:5455-5473``.
    """
    tables = load_gamma_tables(dtype=torch.float64)
    g_med_p = tables.mediated_protein[aa_i, aa_j].item()
    g_med_w = tables.mediated_water[aa_i, aa_j].item()
    kappa = 5.0
    kappa_sigma = 7.0
    treshold = 2.6
    r_min_m, r_max_m = 6.5, 9.5
    sigma_wat = 0.25 * (1.0 - math.tanh(kappa_sigma * (rho_i - treshold))) \
                     * (1.0 - math.tanh(kappa_sigma * (rho_j - treshold)))
    sigma_prot = 1.0 - sigma_wat
    sigma_gamma_med = sigma_prot * g_med_p + sigma_wat * g_med_w
    theta_med = 0.25 * (1.0 + math.tanh(kappa * (r_ij - r_min_m))) \
                     * (1.0 + math.tanh(kappa * (r_max_m - r_ij)))
    return -(sigma_gamma_med * theta_med)


def _burial_pair_kcal(aa_idx: int, rho: float) -> float:
    """Single-residue burial energy at given (aa, ρ). Reference scalar.

    Used by the per-pair native-energy reconstruction. Independent of our
    burial.py implementation so a regression there doesn't silently change
    this test. Mirrors ``fix_backbone.cpp:5478-5500``.
    """
    from frustration_gpu.parameters import (
        BURIAL_KAPPA,
        BURIAL_RHO_MAX,
        BURIAL_RHO_MIN,
        load_burial_gamma,
    )
    bg = load_burial_gamma(dtype=torch.float64)
    row = bg[aa_idx].tolist()
    kappa = BURIAL_KAPPA
    e = 0.0
    for w in range(3):
        t = math.tanh(kappa * (rho - BURIAL_RHO_MIN[w])) + \
            math.tanh(kappa * (BURIAL_RHO_MAX[w] - rho))
        e += -0.5 * 1.0 * row[w] * t
    return e


# --- Test 1: pair-level value matches the AWSEM formula ----------------------
def test_pair_value_first_contact():
    """V_direct for the first dumped 5AON pair matches our by-hand value.

    Pair: i=1, j=3, a_i=S (15), a_j=R (1), r=5.065 Å.
    """
    rows = _load_dump_rows()
    i_lammps, j_lammps, ci, cj, r_ij, rho_i, rho_j, a_i, a_j, e_native = rows[0]
    assert (i_lammps, j_lammps, a_i, a_j) == (1, 3, "S", "R"), (
        "Dump row 0 changed — update the regression baseline."
    )

    # ONE_TO_IDX for S, R
    aa_i = 15  # S
    aa_j = 1   # R

    # Hand value via direct_pair_energy
    v_direct = direct_pair_energy(r_ij, aa_i, aa_j, dtype=torch.float64).item()

    # Independent reference (don't use direct_pair_energy itself — recompute)
    g = load_direct_gamma(dtype=torch.float64)[aa_i, aa_j].item()
    eta = 5.0
    r_min, r_max = 4.5, 6.5
    theta = 0.25 * (1.0 + math.tanh(eta * (r_ij - r_min))) \
                 * (1.0 + math.tanh(eta * (r_max - r_ij)))
    v_direct_manual = -1.0 * g * theta

    assert abs(v_direct - v_direct_manual) < 1e-12, (
        f"PyTorch V_direct {v_direct} vs manual {v_direct_manual}"
    )

    # Sanity print for the status doc - captured in PHASE_1_STATUS.md
    print(f"\n5AON pair (1, 3) V_direct = {v_direct:.6f} kcal/mol "
          f"(gamma={g:.5f}, theta={theta:.6f})")

    # Reconstruct E_native and compare to the dump.
    v_med = _ref_mediated_pair_kcal(r_ij, aa_i, aa_j, rho_i, rho_j)
    burial_i = _burial_pair_kcal(aa_i, rho_i)
    burial_j = _burial_pair_kcal(aa_j, rho_j)
    e_recon = v_direct + v_med + burial_i + burial_j
    diff = abs(e_recon - e_native)
    print(f"E_native reconstructed = {e_recon:.4f} vs dump {e_native:.4f} "
          f"(|diff|={diff:.4f})")
    # The dump has 3-decimal precision so 0.005 is the tightest principled
    # tolerance once mediated is in place (Phase 2b).
    assert diff < 0.005, f"Reconstructed E_native off by {diff} (>0.005)"


# --- Test 2: dense PyTorch path agrees with the pair-level reference --------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_dense_path_matches_pairwise_reference(parsed_5aon):
    """Pick a contact from the dump, find the corresponding (i, j) in the
    dense pair_energy matrix, and confirm they agree."""
    rows = _load_dump_rows()
    # Use the first pair from the dump.
    i_l, j_l, ci, cj, r_ij_dump, _, _, a_i, a_j, _ = rows[0]
    # Map LAMMPS 1-indexed (i, j) within chain 1 to our internal index. The
    # 5AON chain breakdown in the dump is single-chain so (i_internal, j_internal)
    # = (i_l - 1, j_l - 1).
    i_int, j_int = i_l - 1, j_l - 1

    out = direct_contact_energy(parsed_5aon, return_pair_matrix=True)
    pair_e = out["pair_energy"]
    distances = out["distances"]

    # Distance sanity: agrees with the dump distance to 3 decimal places
    # (the dump has 3-decimal precision on r_ij).
    r_computed = distances[i_int, j_int].item()
    assert abs(r_computed - r_ij_dump) < 5e-3, (
        f"r_ij {r_computed} vs dump {r_ij_dump}"
    )

    # Energy sanity: agrees with direct_pair_energy reference.
    e_pair = pair_e[i_int, j_int].item()
    e_ref = direct_pair_energy(r_computed, 15, 1, dtype=torch.float64).item()
    assert abs(e_pair - e_ref) < 1e-6, f"dense {e_pair} vs pair-ref {e_ref}"

    print(f"\n5AON dense path pair (1, 3) energy = {e_pair:.6f} kcal/mol")


# --- Test 3: total V_direct is finite, negative-magnitude, size-monotonic ---
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_total_v_direct_5aon(parsed_5aon):
    e = direct_contact_energy(parsed_5aon).item()
    assert math.isfinite(e), "V_direct should be finite"
    # The direct contact term is a sum of γ*θ over close pairs; γ has both
    # signs but the bulk is negative for buried pairs, so the total tends
    # negative for compact proteins.
    print(f"\n5AON total V_direct = {e:.4f} kcal/mol over {len(_load_dump_rows())} "
          f"dumped contacts")


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_total_v_direct_11bg(parsed_11bg):
    e = direct_contact_energy(parsed_11bg).item()
    assert math.isfinite(e), "V_direct should be finite"
    print(f"\n11BG total V_direct = {e:.4f} kcal/mol")


@pytest.mark.skipif(not (_has_pdb("5AON") and _has_pdb("11BG")), reason="PDBs missing")
def test_size_monotonic(parsed_5aon, parsed_11bg):
    """11BG has ~5x more residues than 5AON; |V_direct| should be larger."""
    e_5aon = direct_contact_energy(parsed_5aon).item()
    e_11bg = direct_contact_energy(parsed_11bg).item()
    assert abs(e_11bg) > abs(e_5aon), (
        f"11BG |V_direct| {abs(e_11bg)} <= 5AON {abs(e_5aon)}"
    )


# --- Test 4: per-pair sum over the 221 dumped contacts ----------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sum_v_direct_over_dumped_contacts(parsed_5aon):
    """Compute V_direct over the 221 (i, j) pairs in the dump and report.

    We can't compare against a "ground truth V_direct" number because the
    dump only carries E_native (V_water + 2 burials). But we can confirm
    the sum is finite and reasonable in scale. The number is reported for
    the status doc.
    """
    rows = _load_dump_rows()
    total = 0.0
    one_to_idx = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6,
                  "G": 7, "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13,
                  "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
    for (i, j, _, _, r, _, _, a_i, a_j, _) in rows:
        v = direct_pair_energy(r, one_to_idx[a_i], one_to_idx[a_j],
                               dtype=torch.float64).item()
        total += v
    assert math.isfinite(total)
    print(f"\nSum V_direct over {len(rows)} dumped 5AON contacts = "
          f"{total:.4f} kcal/mol")


# --- Test 5: CPU/GPU agreement ----------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_cpu_gpu_agreement():
    p_cpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cpu", dtype=torch.float64)
    p_gpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cuda", dtype=torch.float64)
    e_cpu = direct_contact_energy(p_cpu).item()
    e_gpu = direct_contact_energy(p_gpu).item()
    rel = abs(e_cpu - e_gpu) / max(abs(e_cpu), 1e-12)
    assert rel < 1e-6, f"CPU {e_cpu} vs GPU {e_gpu} rel.diff {rel}"


# --- Test 6: differentiable w.r.t. coords -----------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_differentiable_wrt_cb(parsed_5aon):
    """Confirm autograd produces finite gradients on CB."""
    p = {**parsed_5aon}
    p["ca_coords"] = p["ca_coords"].clone().detach().requires_grad_(True)
    p["cb_coords"] = p["cb_coords"].clone().detach().requires_grad_(True)
    e = direct_contact_energy(p)
    e.backward()
    # CB grad should be finite (NaN-CB rows can have NaN grad — that's fine for
    # GLY, just check the finite rows).
    finite_rows = torch.isfinite(p["cb_coords"]).all(dim=-1)
    assert torch.isfinite(p["cb_coords"].grad[finite_rows]).all(), (
        "CB gradient should be finite for residues with valid CB"
    )
    # Should not be identically zero — many pairs contribute non-trivially
    assert p["cb_coords"].grad[finite_rows].abs().max().item() > 0


# --- Test 7: n=0 / n=1 / boundary edge cases --------------------------------
def test_empty_protein_returns_zero():
    """n=0 → 0.0 energy (no possible pair)."""
    coords = {
        "ca_coords": torch.zeros((0, 3), dtype=torch.float64),
        "cb_coords": torch.zeros((0, 3), dtype=torch.float64),
        "residue_types": torch.zeros((0,), dtype=torch.int64),
        "chain_ids": [],
    }
    e = direct_contact_energy(coords).item()
    assert e == 0.0


def test_single_residue_returns_zero():
    """n=1 → 0.0 energy (no possible pair)."""
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        "residue_types": torch.tensor([0], dtype=torch.int64),
        "chain_ids": ["A"],
    }
    e = direct_contact_energy(coords).item()
    assert e == 0.0


def test_boundary_continuity_direct():
    """θ_direct ≈ 0.5 at r=r_min and r=r_max; ~0 outside the window."""
    aa_i, aa_j = 0, 0
    rs = [3.0, 4.5, 5.5, 6.5, 8.0]
    vals = [
        direct_pair_energy(r, aa_i, aa_j, dtype=torch.float64).item()
        for r in rs
    ]
    # Outside window: ~0 contribution
    assert abs(vals[0]) < 1e-4, f"V at r=3.0 should be ~0, got {vals[0]}"
    assert abs(vals[-1]) < 1e-4, f"V at r=8.0 should be ~0, got {vals[-1]}"
    # In-window mid: non-zero
    assert abs(vals[2]) > 1e-3
    # Smooth transition (no discontinuity)
    for k in range(len(vals) - 1):
        d = abs(vals[k + 1] - vals[k])
        assert d < 1.0, f"jump at r {rs[k]} -> {rs[k+1]}: {d}"


def test_nan_residue_does_not_poison_gradients():
    """Regression test for the double-where NaN trick (Phase 2a review #1).

    Synthetic 5-residue protein with residue 2 fully missing (NaN CA + NaN CB).
    Assert gradients on the 4 finite residues are finite — no NaN poisoning.
    """
    n = 5
    cb = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.5, 0.5, 0.2],
            [float("nan"), float("nan"), float("nan")],
            [7.0, 0.0, 0.0],
            [10.5, 0.5, 0.2],
        ],
        dtype=torch.float64,
    )
    ca = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [3.5, 1.5, 0.2],
            [float("nan"), float("nan"), float("nan")],
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
    e = direct_contact_energy(coords)
    e.backward()

    finite_rows = torch.tensor([True, True, False, True, True])
    cb_grad_ok = torch.isfinite(cb_p.grad[finite_rows]).all().item()
    ca_grad_ok = torch.isfinite(ca_p.grad[finite_rows]).all().item()
    assert cb_grad_ok, f"CB grad has NaN on finite rows: {cb_p.grad}"
    assert ca_grad_ok, f"CA grad has NaN on finite rows: {ca_p.grad}"


def test_k_water_warning_with_custom_gamma():
    """Passing custom gamma + k_water != 1 should emit a warning."""
    import warnings as warnings_mod

    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
                                  dtype=torch.float64),
        "residue_types": torch.tensor([0, 0], dtype=torch.int64),
        "chain_ids": ["A", "A"],
    }
    custom_gamma = torch.zeros((20, 20), dtype=torch.float64)
    with warnings_mod.catch_warnings(record=True) as w:
        warnings_mod.simplefilter("always")
        direct_contact_energy(coords, gamma_direct=custom_gamma, k_water=2.0)
        assert any("k_water" in str(_.message) for _ in w), (
            "Expected k_water double-fold warning"
        )


# --- Test 8: DNA sentinel guard (Quality-fixes-batch-2 Fix A) --------------
def test_dna_sentinel_guard_raises():
    """``residue_types == -1`` (DNA placeholder) must raise, not silently wrap.

    Negative indexing would otherwise map to gamma row 19 (VAL), producing
    a plausible-looking energy on biological nonsense. See ``qa1_core_math.md``.
    """
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
                                  dtype=torch.float64),
        "residue_types": torch.tensor([0, -1], dtype=torch.int64),
        "chain_ids": ["A", "A"],
    }
    with pytest.raises(ValueError, match="DNA sentinel"):
        direct_contact_energy(coords)


# --- Test 9: sparse vs dense ~ULP-equivalent agreement (Fix B) -------------
# The spec for SPEED-3 Idea 1 claims "bit-identical" sums. Empirically that
# claim doesn't hold for two reasons:
#  (a) PyTorch's tree-reduction in `.sum()` depends on the underlying tensor
#      shape — a (N, N) `.sum()` and a (N_pair,) `.sum()` produce ULP-scale
#      drift even when summing the same values + zeros.
#  (b) The dense path includes pairs at all separations (theta = ~1e-22 at
#      r >> r_max); sparse drops pairs beyond ``sparse_cutoff``. Summed over
#      ~N²/2 entries that's an additional ~1e-14 tail in 5AON.
# In practice the drift is well under any meaningful precision floor (1e-13
# relative on real proteins). Gate accordingly.
_BYTE_EXACT_REL_TOL = 1e-12


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sparse_byte_exact_5aon(parsed_5aon):
    """sparse=True total energy matches sparse=False to fp64 precision."""
    from frustration_gpu._contact_common import build_contact_context

    e_dense = direct_contact_energy(parsed_5aon).item()
    ctx_sparse = build_contact_context(parsed_5aon, sparse_cutoff=11.0)
    e_sparse = direct_contact_energy(
        parsed_5aon, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n5AON V_direct dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_sparse_byte_exact_11bg(parsed_11bg):
    """Same ULP-level gate on 11BG (larger N → larger absolute drift)."""
    from frustration_gpu._contact_common import build_contact_context

    e_dense = direct_contact_energy(parsed_11bg).item()
    ctx_sparse = build_contact_context(parsed_11bg, sparse_cutoff=11.0)
    e_sparse = direct_contact_energy(
        parsed_11bg, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n11BG V_direct dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sparse_return_pair_matrix_5aon(parsed_5aon):
    """``return_pair_matrix=True`` works in sparse mode (re-densifies pair_energy)."""
    from frustration_gpu._contact_common import build_contact_context

    out_dense = direct_contact_energy(parsed_5aon, return_pair_matrix=True)
    ctx_sparse = build_contact_context(parsed_5aon, sparse_cutoff=11.0)
    out_sparse = direct_contact_energy(
        parsed_5aon, _context=ctx_sparse, sparse=True,
        return_pair_matrix=True,
    )
    # Per-pair-energy entries: dense includes the long-distance tail (theta
    # ~1e-22 per pair), sparse drops them. Difference per entry < 1e-21.
    diff_max = (out_dense["pair_energy"] - out_sparse["pair_energy"]).abs().max().item()
    print(f"\n5AON pair_energy max abs diff dense vs sparse = {diff_max:.3e}")
    assert diff_max < 1e-15, (
        f"pair_energy entry-level drift too large: {diff_max}"
    )


# --- Test 10: cdist 1-ULP drift (Fix C) -------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_cdist_drift_5aon(parsed_5aon):
    """``use_cdist=True`` should differ by less than ~1e-12 relative.

    Default is ``use_cdist=False`` so the byte-exact tests above pass; this
    test documents the cdist drift on 5AON for future regression watch.
    """
    e_broadcast = direct_contact_energy(parsed_5aon).item()
    e_cdist = direct_contact_energy(parsed_5aon, use_cdist=True).item()
    diff = abs(e_broadcast - e_cdist)
    rel = diff / max(abs(e_broadcast), 1e-30)
    print(f"\n5AON cdist drift: broadcast={e_broadcast:.18e} "
          f"cdist={e_cdist:.18e} abs={diff:.3e} rel={rel:.3e}")
    # Empirical: cdist matches the broadcast on CPU fp64 to better than 1e-12
    # relative. Cap at 1e-10 to allow some headroom for future torch updates.
    assert rel < 1e-10, (
        f"cdist drift unexpectedly large: rel={rel}. Either disable use_cdist "
        f"in production paths or document the regression here."
    )
