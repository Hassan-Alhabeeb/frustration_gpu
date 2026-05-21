"""Tests for the AWSEM Debye-Hückel electrostatics term (Phase 2c).

Validation strategy
-------------------
DH is NOT activated in the default LAMMPS-AWSEM configurational dumps — the
``Electro.`` column in ``energy.log`` is ``0.000000`` for both the canonical
configurational runs and the ``electrostatics_k`` param-sweep runs in
``benchmark/cpu_baseline/param_sweep/`` (verified 2026-05-20). So we
**cannot** ground V_DH against an end-to-end ``Electro.`` number from the
LAMMPS dump.

What we CAN gate on with high confidence:

1. **Per-pair formula** matches ``fix_backbone.cpp:5502-5547`` exactly for
   hand-picked (r, aa_i, aa_j) cases — including the sign convention and
   the ``q_i × q_j`` lookup.
2. **Charge assignment** matches the C++: D/E → -1, R/K → +1, HIS → 0 (not
   +1, despite biochemistry). Polyalanine should give exactly zero energy.
3. **Linear scaling in k_QQ**: V_DH(k=17.3636) / V_DH(k=4.15) = 4.184 to
   machine precision (the formula has only one ``k_QQ`` factor).
4. **Sequence-separation gate**: ``min_seq_sep=1`` excludes only self-pairs
   (default); ``min_seq_sep=5`` excludes near-neighbours.
5. **Distance scaling**: a single D-K pair at known r gives the
   hand-computed Coulomb-screened value.
6. **CPU/GPU agreement** at machine precision.
7. **Differentiability** w.r.t. CB coords.

The "V_Total reconstruction" check (V_Water + V_Burial + V_DH ≈ -60.500
for 5AON) is trivially satisfied because the LAMMPS V_Total is computed
WITHOUT V_DH (DH gated off), so V_Water + V_Burial = -60.500 already
matches without DH. Our V_DH is non-zero (-0.60 kcal/mol on 5AON with
k=4.15) but this represents the contribution we WOULD add if DH were on
in the LAMMPS run — not a number to compare against the dump column.
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

from _paths import PDB_DIR  # noqa: E402

from frustration_gpu.debye_huckel import (  # noqa: E402
    DH_CHARGES_FLOAT,
    aa_charge_vector,
    debye_huckel_energy,
    debye_huckel_pair_energy,
)
from frustration_gpu.parser import ONE_TO_IDX, parse_pdb  # noqa: E402


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


@pytest.fixture(scope="module")
def parsed_5aon():
    return parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)


@pytest.fixture(scope="module")
def parsed_11bg():
    return parse_pdb(PDB_DIR / "11BG.pdb", dtype=torch.float64)


# --- Test 1: charge vector matches C++ verbatim -----------------------------
def test_charge_vector_against_cpp():
    """fix_backbone.cpp:5511-5527 sets q=+1 for R/K, q=-1 for D/E, q=0 else."""
    q = aa_charge_vector(dtype=torch.float64)
    assert q.shape == (20,)
    expected = {
        "R": +1.0,
        "K": +1.0,
        "D": -1.0,
        "E": -1.0,
    }
    # Anything else must be 0 — including HIS (the load-bearing audit point).
    for one_letter, idx in ONE_TO_IDX.items():
        target = expected.get(one_letter, 0.0)
        got = q[idx].item()
        assert got == target, f"AA {one_letter} idx={idx}: got {got}, expected {target}"

    # Explicit zero check on HIS — this is the C++ deviation from typical
    # biochemistry assumptions and is the most likely place we could have
    # silently regressed.
    assert q[ONE_TO_IDX["H"]].item() == 0.0, "HIS must be 0, not +1"

    # And the module-level constant tuple matches the tensor.
    assert tuple(q.tolist()) == DH_CHARGES_FLOAT


# --- Test 2: per-pair hand-check for a D-K pair -----------------------------
def test_pair_value_hand_check_D_K():
    """V_DH for D-K at r=10 Å:

    q_i × q_j = (-1)(+1) = -1
    V = +1.0 × 4.15 × (-1) × exp(-10/10) / 10
      = -4.15 × exp(-1) / 10
      = -4.15 × 0.36787944... / 10
      ≈ -0.152671
    """
    aa_D = ONE_TO_IDX["D"]
    aa_K = ONE_TO_IDX["K"]
    r = 10.0
    v = debye_huckel_pair_energy(r, aa_D, aa_K, dtype=torch.float64).item()
    expected = 1.0 * 4.15 * (-1.0) * math.exp(-1.0) / 10.0
    assert abs(v - expected) < 1e-12, f"DH(D-K, r=10) = {v}, expected {expected}"
    print(f"\nDH(D-K, r=10) = {v:.10f} kcal/mol (target {expected:.10f})")


# --- Test 3: per-pair hand-check for like-charge (E-D) ----------------------
def test_pair_value_hand_check_like_charge():
    """E-D at r=5 Å should be REPULSIVE (positive energy)."""
    aa_E = ONE_TO_IDX["E"]
    aa_D = ONE_TO_IDX["D"]
    r = 5.0
    v = debye_huckel_pair_energy(r, aa_E, aa_D, dtype=torch.float64).item()
    expected = 4.15 * (-1.0) * (-1.0) * math.exp(-0.5) / 5.0
    assert abs(v - expected) < 1e-12
    assert v > 0, f"like-charge pair should be repulsive, got {v}"
    print(f"\nDH(E-D, r=5)  = {v:.10f} kcal/mol  (repulsive, q×q=+1)")


# --- Test 4: per-pair returns zero for uncharged residue --------------------
def test_pair_zero_for_neutral():
    """Anything not in {D, E, R, K} contributes zero (incl. HIS)."""
    aa_A = ONE_TO_IDX["A"]
    aa_K = ONE_TO_IDX["K"]
    aa_H = ONE_TO_IDX["H"]

    # A-K: A is neutral → 0
    assert debye_huckel_pair_energy(5.0, aa_A, aa_K, dtype=torch.float64).item() == 0.0
    # H-K: H is neutral in AWSEM → 0
    assert debye_huckel_pair_energy(5.0, aa_H, aa_K, dtype=torch.float64).item() == 0.0
    # A-A: both neutral → 0
    assert debye_huckel_pair_energy(5.0, aa_A, aa_A, dtype=torch.float64).item() == 0.0


# --- Test 5: polyalanine protein has V_DH = 0 -------------------------------
def test_polyalanine_zero_energy():
    """A synthetic all-Ala protein should give exactly V_DH = 0."""
    n = 20
    rng = torch.Generator().manual_seed(0)
    ca = torch.randn((n, 3), generator=rng, dtype=torch.float64) * 5.0
    # offset CB by ~1.5 Å in a random direction so CB != CA
    cb = ca + torch.randn((n, 3), generator=rng, dtype=torch.float64) * 0.5
    coords = {
        "ca_coords": ca,
        "cb_coords": cb,
        "residue_types": torch.zeros(n, dtype=torch.int64),  # all A (idx 0)
        "chain_ids": ["A"] * n,
    }
    e = debye_huckel_energy(coords).item()
    assert e == 0.0, f"polyalanine V_DH must be 0, got {e}"


# --- Test 6: single D-K pair on a tiny synthetic protein --------------------
def test_synthetic_dk_pair_matches_pair_formula():
    """Two-residue D-K system placed at r=8 Å should give V_DH = single pair value."""
    coords = {
        "ca_coords": torch.tensor(
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=torch.float64
        ),
        # Place CB right on top of CA so the effective-CB distance is 8.0
        "cb_coords": torch.tensor(
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "residue_types": torch.tensor(
            [ONE_TO_IDX["D"], ONE_TO_IDX["K"]], dtype=torch.int64
        ),
        "chain_ids": ["A", "A"],
    }
    e_dense = debye_huckel_energy(coords).item()
    e_pair = debye_huckel_pair_energy(
        8.0, ONE_TO_IDX["D"], ONE_TO_IDX["K"], dtype=torch.float64
    ).item()
    assert abs(e_dense - e_pair) < 1e-12, (
        f"dense {e_dense} vs pair-ref {e_pair}"
    )
    print(f"\nsynthetic D-K (r=8): V_DH = {e_dense:.10f}")


# --- Test 7: linear scaling under k_QQ (electrostatics_k API) ---------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_linear_scaling_in_k_QQ_5aon(parsed_5aon):
    """V_DH(k=17.3636) / V_DH(k=4.15) must equal 17.3636 / 4.15 = 4.184."""
    e_default = debye_huckel_energy(parsed_5aon).item()
    e_4p15 = debye_huckel_energy(parsed_5aon, k_QQ=4.15).item()
    e_17 = debye_huckel_energy(parsed_5aon, k_QQ=17.3636).item()

    # default == 4.15
    assert abs(e_default - e_4p15) < 1e-12
    assert e_4p15 != 0.0, "5AON should have non-zero V_DH (≥2 charged residues)"

    ratio = e_17 / e_4p15
    expected = 17.3636 / 4.15
    assert abs(ratio - expected) < 1e-10, f"ratio {ratio} vs expected {expected}"
    print(f"\n5AON V_DH(k=4.15)    = {e_4p15:.6f} kcal/mol")
    print(f"5AON V_DH(k=17.3636) = {e_17:.6f} kcal/mol  ratio {ratio:.6f}")


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_linear_scaling_in_k_QQ_11bg(parsed_11bg):
    """Same linearity check on 11BG (different protein → independent witness)."""
    e_4p15 = debye_huckel_energy(parsed_11bg, k_QQ=4.15).item()
    e_17 = debye_huckel_energy(parsed_11bg, k_QQ=17.3636).item()
    if e_4p15 == 0.0:
        pytest.skip("11BG has no charge pairs at default min_seq_sep")
    ratio = e_17 / e_4p15
    expected = 17.3636 / 4.15
    assert abs(ratio - expected) < 1e-10
    print(f"\n11BG V_DH(k=4.15)    = {e_4p15:.6f} kcal/mol")
    print(f"11BG V_DH(k=17.3636) = {e_17:.6f} kcal/mol  ratio {ratio:.6f}")


# --- Test 8: total V_DH on 5AON is finite and reasonable --------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_total_v_dh_5aon(parsed_5aon):
    e = debye_huckel_energy(parsed_5aon).item()
    assert math.isfinite(e)
    # The DH term is small relative to V_Water/V_Burial on 5AON (which sum
    # to -60.500). Sanity check: |V_DH| < 5 kcal/mol — anything bigger means
    # the screening length or k_QQ has the wrong magnitude.
    assert abs(e) < 5.0, f"V_DH magnitude {abs(e)} suspiciously large"
    print(f"\n5AON total V_DH = {e:.6f} kcal/mol")


# --- Test 9: total V_DH on 11BG is finite ----------------------------------
@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_total_v_dh_11bg(parsed_11bg):
    e = debye_huckel_energy(parsed_11bg).item()
    assert math.isfinite(e)
    print(f"\n11BG total V_DH = {e:.6f} kcal/mol")


# --- Test 10: per-pair reconstruction of total ------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_dense_equals_per_pair_sum(parsed_5aon):
    """The dense V_DH must equal the sum over (i, j) of the scalar pair formula
    applied to every charged-charged pair at |i-j|>=1.
    """
    out = debye_huckel_energy(parsed_5aon, return_pair_matrix=True)
    dense_total = out["energy"].item()
    distances = out["distances"]
    aa = parsed_5aon["residue_types"]
    n = aa.shape[0]
    chains = parsed_5aon["chain_ids"]

    qvec = aa_charge_vector(dtype=torch.float64)
    total_ref = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            qi = qvec[aa[i]].item()
            qj = qvec[aa[j]].item()
            if qi == 0.0 or qj == 0.0:
                continue
            # min_seq_sep=1 → exclude only self (|i-j|=0). Since i<j here that's auto-OK
            r = distances[i, j].item()
            if not math.isfinite(r):
                continue
            total_ref += debye_huckel_pair_energy(
                r, int(aa[i]), int(aa[j]), dtype=torch.float64
            ).item()
    assert abs(dense_total - total_ref) < 1e-9, (
        f"dense {dense_total} vs per-pair sum {total_ref}"
    )
    print(f"\n5AON dense V_DH {dense_total:.10f}  vs  per-pair sum {total_ref:.10f}")


# --- Test 11: min_seq_sep gate behaves correctly ---------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_min_seq_sep_gate(parsed_5aon):
    """Increasing min_seq_sep monotonically reduces the per-pair-energy mass.

    The SIGNED total V_DH is NOT monotonic in min_seq_sep — V_DH carries both
    repulsive (positive) and attractive (negative) contributions, so dropping
    a repulsive close pair can INCREASE |V_DH| even though we removed a term.
    The right invariant is that the sum of absolute pair energies (the "DH
    mass") decreases monotonically as pairs are dropped, and that the active
    pair count decreases.
    """
    o_1 = debye_huckel_energy(parsed_5aon, min_seq_sep=1, return_pair_matrix=True)
    o_5 = debye_huckel_energy(parsed_5aon, min_seq_sep=5, return_pair_matrix=True)
    o_20 = debye_huckel_energy(parsed_5aon, min_seq_sep=20, return_pair_matrix=True)

    npairs_1 = o_1["pair_mask"].sum().item()
    npairs_5 = o_5["pair_mask"].sum().item()
    npairs_20 = o_20["pair_mask"].sum().item()
    assert npairs_1 >= npairs_5 >= npairs_20

    mass_1 = o_1["pair_energy"].abs().sum().item()
    mass_5 = o_5["pair_energy"].abs().sum().item()
    mass_20 = o_20["pair_energy"].abs().sum().item()
    assert mass_1 >= mass_5 >= mass_20, (
        f"DH mass not monotonic in min_seq_sep: "
        f"sep=1 {mass_1:.6f}, =5 {mass_5:.6f}, =20 {mass_20:.6f}"
    )

    # Also confirm sep=1 keeps strictly MORE pairs than sep=20 on a real
    # protein with many same-chain charge pairs.
    assert npairs_1 > npairs_20

    e_1 = o_1["energy"].item()
    e_5 = o_5["energy"].item()
    e_20 = o_20["energy"].item()
    print(
        f"\n5AON V_DH sep=1: {e_1:.6f} ({npairs_1:.0f} pairs)  "
        f"sep=5: {e_5:.6f} ({npairs_5:.0f})  "
        f"sep=20: {e_20:.6f} ({npairs_20:.0f})"
    )


# --- Test 12: CPU/GPU agreement --------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_cpu_gpu_agreement_5aon():
    p_cpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cpu", dtype=torch.float64)
    p_gpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cuda", dtype=torch.float64)
    e_cpu = debye_huckel_energy(p_cpu).item()
    e_gpu = debye_huckel_energy(p_gpu).item()
    rel = abs(e_cpu - e_gpu) / max(abs(e_cpu), 1e-12)
    assert rel < 1e-12, f"CPU {e_cpu} vs GPU {e_gpu} rel.diff {rel}"
    print(f"\n5AON V_DH CPU {e_cpu:.12f}  GPN {e_gpu:.12f}  rel-diff {rel:.2e}")


# --- Test 13: differentiable w.r.t. CB coords ------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_differentiable_wrt_cb(parsed_5aon):
    """V_DH.backward() produces a finite gradient on CB coords."""
    p = {**parsed_5aon}
    p["ca_coords"] = p["ca_coords"].clone().detach().requires_grad_(True)
    p["cb_coords"] = p["cb_coords"].clone().detach().requires_grad_(True)
    e = debye_huckel_energy(p)
    assert e.item() != 0.0, "5AON V_DH should be non-zero so we can backprop"
    e.backward()
    finite_rows = torch.isfinite(p["cb_coords"]).all(dim=-1)
    assert torch.isfinite(p["cb_coords"].grad[finite_rows]).all(), (
        "CB gradient must be finite for valid-CB residues"
    )
    # At least one charged residue should have nonzero gradient (CB on
    # D/E/R/K is what drives the DH term)
    assert p["cb_coords"].grad[finite_rows].abs().max().item() > 0


# --- Test 14: n=1 / n=0 short-circuit returns 0 ----------------------------
def test_edge_case_n1():
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        "residue_types": torch.tensor([ONE_TO_IDX["D"]], dtype=torch.int64),
        "chain_ids": ["A"],
    }
    e = debye_huckel_energy(coords).item()
    assert e == 0.0, "n=1 must produce V_DH = 0"


def test_edge_case_n0():
    coords = {
        "ca_coords": torch.zeros((0, 3), dtype=torch.float64),
        "cb_coords": torch.zeros((0, 3), dtype=torch.float64),
        "residue_types": torch.zeros((0,), dtype=torch.int64),
        "chain_ids": [],
    }
    e = debye_huckel_energy(coords).item()
    assert e == 0.0


# --- Test 15: return_pair_matrix has the expected dict keys ----------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_return_pair_matrix_shape(parsed_5aon):
    out = debye_huckel_energy(parsed_5aon, return_pair_matrix=True)
    n = parsed_5aon["residue_types"].shape[0]
    assert set(out.keys()) == {
        "energy", "pair_energy", "pair_mask", "distances", "charges"
    }
    assert out["energy"].shape == ()
    assert out["pair_energy"].shape == (n, n)
    assert out["pair_mask"].shape == (n, n)
    assert out["distances"].shape == (n, n)
    assert out["charges"].shape == (n,)
    # pair_energy is upper-triangular: lower triangle including diagonal is 0
    lower = torch.tril(out["pair_energy"], diagonal=0)
    assert lower.abs().max().item() == 0.0
    # sum of upper triangle == reported energy
    assert torch.isclose(
        out["pair_energy"].sum(), out["energy"], atol=1e-12
    ).item()


# --- Test 16: cross-chain pairs always contribute --------------------------
def test_cross_chain_no_seq_sep_filter():
    """Two single-residue chains: D in chain A, K in chain B at r=10.

    The seq-sep filter is INTRA-chain only — cross-chain pairs always contribute
    regardless of min_seq_sep. So even with a huge min_seq_sep, this pair
    contributes.
    """
    coords = {
        "ca_coords": torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "cb_coords": torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "residue_types": torch.tensor(
            [ONE_TO_IDX["D"], ONE_TO_IDX["K"]], dtype=torch.int64
        ),
        "chain_ids": ["A", "B"],
    }
    # Even with min_seq_sep=100 (way larger than the protein), cross-chain wins
    e = debye_huckel_energy(coords, min_seq_sep=100).item()
    expected = 4.15 * (-1.0) * (1.0) * math.exp(-1.0) / 10.0
    assert abs(e - expected) < 1e-12, f"cross-chain DH {e} vs {expected}"


# --- DNA sentinel guard (Fix A) --------------------------------------------
def test_dna_sentinel_guard_raises():
    """``residue_types == -1`` must raise on debye_huckel_energy.

    DH "lucks out" with q=0 for negative indices, but the guard catches the
    upstream mistake so future charge-table changes don't silently regress.
    """
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "cb_coords": torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "residue_types": torch.tensor([ONE_TO_IDX["D"], -1], dtype=torch.int64),
        "chain_ids": ["A", "A"],
    }
    with pytest.raises(ValueError, match="DNA sentinel"):
        debye_huckel_energy(coords)


# --- sparse vs dense ~ULP-equivalent agreement (Fix B) ---------------------
# DH is long-range (exp(-r/λ)/r decays slowly). With λ=10, a 100 Å cutoff
# leaves a tail of ``exp(-10)/100 ≈ 5e-7`` per pair, accumulating to ~1e-5
# absolute on small proteins. Use 150 Å (exp(-15)/150 ≈ 2e-9 per pair) for
# the byte-exact test. Real callers may want a tighter cutoff for speed.
_DH_SPARSE_CUTOFF_A = 150.0
_BYTE_EXACT_REL_TOL = 1e-10


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sparse_byte_exact_5aon(parsed_5aon):
    """sparse=True V_DH matches sparse=False to fp64 precision."""
    from frustration_gpu._contact_common import build_contact_context

    e_dense = debye_huckel_energy(parsed_5aon).item()
    # NOTE: DH default min_seq_sep=1, NOT 2. Pre-cache that sep in the ctx.
    ctx_sparse = build_contact_context(
        parsed_5aon, seq_seps=[1], sparse_cutoff=_DH_SPARSE_CUTOFF_A,
    )
    e_sparse = debye_huckel_energy(
        parsed_5aon, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n5AON V_DH dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_sparse_byte_exact_11bg(parsed_11bg):
    """Same gate on 11BG. 11BG is small enough that 150 Å covers everything."""
    from frustration_gpu._contact_common import build_contact_context

    e_dense = debye_huckel_energy(parsed_11bg).item()
    ctx_sparse = build_contact_context(
        parsed_11bg, seq_seps=[1], sparse_cutoff=_DH_SPARSE_CUTOFF_A,
    )
    e_sparse = debye_huckel_energy(
        parsed_11bg, _context=ctx_sparse, sparse=True,
    ).item()
    diff = abs(e_dense - e_sparse)
    rel = diff / max(abs(e_dense), 1e-30)
    print(f"\n11BG V_DH dense={e_dense:.18e} sparse={e_sparse:.18e} "
          f"abs={diff:.3e} rel={rel:.3e}")
    assert rel < _BYTE_EXACT_REL_TOL, (
        f"sparse vs dense drift exceeds {_BYTE_EXACT_REL_TOL}: rel={rel}"
    )
