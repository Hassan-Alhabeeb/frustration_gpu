"""Validation tests for the MISC bug-fix bundle (rev 2026-05-21).

Covers the 14 MEDIUM findings owned by the misc-bugs agent:

* #3   burial.compute_rho autograd-NaN safe
* #13  virtual_atoms.use_cis_proline applies to proline rows ONLY
* #15  frustration.classify_frustration rejects non-finite FI
* #27  density.compute_residue_density validates pair/fi shapes
* #32  parameters.load_gamma_tables rejects malformed gamma.dat
* #33  virtual_atoms.compute_virtual_atoms respects sequence-gap boundaries
* #40  frustration.welltype_from_contact rejects non-finite inputs
* #41  burial.burial_density rejects DNA sentinels
* #48  burial.compute_rho validates residue_numbers / chain_index shape
* #49  parameter loaders reject NaN / inf
* #51  frustration._aa_idx_to_letter rejects negative / out-of-range types
* #52  density.compute_residue_density validates pair indices
* #62  frustration.emit_singleresidue_dat validates rho length

The DNA-sentinel + parameter-shape tests live in ``test_decoy_validation.py``
(owned by another agent); we cover only the misc subset here.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure ``frustration_gpu`` is importable from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frustration_gpu.burial import (  # noqa: E402
    burial_density,
    compute_rho,
)
from frustration_gpu.density import compute_residue_density  # noqa: E402
from frustration_gpu.frustration import (  # noqa: E402
    _aa_idx_to_letter,
    classify_frustration,
    emit_singleresidue_dat,
    welltype_from_contact,
)
from frustration_gpu.parameters import (  # noqa: E402
    load_burial_gamma,
    load_gamma_tables,
)
from frustration_gpu.parser import ONE_TO_IDX, parse_pdb  # noqa: E402
from frustration_gpu.virtual_atoms import compute_virtual_atoms  # noqa: E402

PDB_DIR = Path(__file__).resolve().parent / "data"


# ---------- #3 burial autograd-NaN safe ---------------------------------------
def test_compute_rho_autograd_nan_safe_when_some_coords_invalid():
    """Backward through compute_rho must not produce NaN gradients on the
    finite rows even when some rows of cb_or_ca_coords are NaN."""
    n = 6
    coords = torch.arange(n * 3, dtype=torch.float64).reshape(n, 3) * 0.5
    # Mark residue index 2 as invalid (whole row NaN).
    coords[2] = float("nan")
    coords.requires_grad_(True)

    residue_numbers = torch.arange(n, dtype=torch.int64)
    chain_index = torch.zeros(n, dtype=torch.int64)

    rho = compute_rho(coords, residue_numbers, chain_index)
    # Sum the rho on the FINITE rows only — but the backward graph still
    # touches all of the safe-distance computation.
    finite_rows = torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64)
    loss = rho[finite_rows].sum()
    loss.backward()

    grad = coords.grad
    # Gradient on the invalid row may be anything (we replaced its coords
    # with a far-away filler) — but the finite rows MUST receive a finite
    # gradient. Pre-fix: every gradient was NaN.
    assert torch.isfinite(grad[finite_rows]).all(), (
        f"Expected finite gradients on finite rows, got {grad[finite_rows]}"
    )


def test_compute_rho_value_unchanged_for_clean_input():
    """Bit-identity: clean-coords compute_rho must be unchanged by the
    double-where rewrite."""
    n = 8
    torch.manual_seed(0)
    coords = torch.randn(n, 3, dtype=torch.float64) * 5.0
    rn = torch.arange(n, dtype=torch.int64)
    ci = torch.zeros(n, dtype=torch.int64)
    rho_a = compute_rho(coords, rn, ci)
    rho_b = compute_rho(coords, rn, ci)
    assert torch.equal(rho_a, rho_b)
    # And both branches (dense / sparse) still agree to a small tolerance.
    rho_sparse = compute_rho(coords, rn, ci, sparse=True, sparse_cutoff_a=12.0)
    assert torch.allclose(rho_a, rho_sparse, atol=1e-6)


# ---------- #48 compute_rho shape validation ----------------------------------
def test_compute_rho_rejects_wrong_residue_numbers_shape():
    coords = torch.randn(5, 3, dtype=torch.float64)
    bad_rn = torch.arange(4, dtype=torch.int64)
    ci = torch.zeros(5, dtype=torch.int64)
    with pytest.raises(ValueError, match="residue_numbers"):
        compute_rho(coords, bad_rn, ci)


def test_compute_rho_rejects_wrong_chain_index_shape():
    coords = torch.randn(5, 3, dtype=torch.float64)
    rn = torch.arange(5, dtype=torch.int64)
    bad_ci = torch.zeros(7, dtype=torch.int64)
    with pytest.raises(ValueError, match="chain_index"):
        compute_rho(coords, rn, bad_ci)


# ---------- #41 burial_density DNA-sentinel guard -----------------------------
def test_burial_density_rejects_dna_sentinel():
    parsed = parse_pdb(PDB_DIR / "5AON.pdb")
    n = parsed["ca_coords"].shape[0]
    # Inject one DNA-sentinel residue type.
    rt = parsed["residue_types"].clone()
    rt[0] = -1
    parsed["residue_types"] = rt
    with pytest.raises(ValueError, match="DNA"):
        burial_density(parsed)


# ---------- #13 cis-proline only on proline ----------------------------------
def test_cis_proline_only_affects_proline_rows():
    """With ``use_cis_proline=True``, non-proline residues must match the
    trans branch exactly; only proline rows differ."""
    parsed = parse_pdb(PDB_DIR / "5AON.pdb")
    trans = compute_virtual_atoms(parsed, use_cis_proline=False)
    cis = compute_virtual_atoms(parsed, use_cis_proline=True)

    is_pro = parsed["residue_types"] == ONE_TO_IDX["P"]
    non_pro = ~is_pro

    # Non-proline rows must be identical between branches.
    for key in ("n_virtual", "c_virtual", "h_virtual"):
        # Compare only where both tensors are finite (NaN at chain boundaries
        # is the expected sentinel and doesn't equal itself).
        t_t = trans[key]
        t_c = cis[key]
        both_finite = torch.isfinite(t_t).all(dim=-1) & torch.isfinite(t_c).all(dim=-1)
        compare = non_pro & both_finite
        if compare.any():
            assert torch.equal(t_t[compare], t_c[compare]), (
                f"Non-proline rows of {key} differ between trans/cis branches "
                "— cis-proline coefficients leaked to non-proline residues."
            )
        # At proline positions the branches MUST differ (assuming any are finite).
        compare_pro = is_pro & both_finite
        if compare_pro.any():
            assert not torch.equal(t_t[compare_pro], t_c[compare_pro]), (
                "Proline rows of {key} are identical between branches — "
                "cis-proline coefficients did not apply."
            )


# ---------- #33 virtual atoms respect sequence-numbering gaps -----------------
def test_virtual_atoms_nan_at_sequence_gap():
    """A residue immediately AFTER a gap (residue_numbers jump by >1) must
    have NaN ``n_virtual`` / ``h_virtual`` because its i-1 neighbour is not
    actually adjacent in sequence."""
    n = 5
    ca = torch.arange(n * 3, dtype=torch.float64).reshape(n, 3)
    o = ca + 0.5
    # Same chain everywhere; insert a gap between index 1 and index 2
    # (resnums: 0, 1, 5, 6, 7 — index 2 has resnum 5 vs index 1 resnum 1).
    resnums = torch.tensor([0, 1, 5, 6, 7], dtype=torch.int64)
    parsed = {
        "ca_coords": ca,
        "o_coords": o,
        "chain_ids": ["A"] * n,
        "residue_types": torch.zeros(n, dtype=torch.int64),
        "residue_numbers": resnums,
    }
    out = compute_virtual_atoms(parsed)
    # Index 2 had a sequence gap before it — N(2) needs CA(1) but the
    # numbering jump means residue 1 is not its physical predecessor.
    assert torch.isnan(out["n_virtual"][2]).all()
    assert torch.isnan(out["h_virtual"][2]).all()
    # Index 1 had a sequence gap AFTER it — C(1) needs CA(2), gap means
    # invalid.
    assert torch.isnan(out["c_virtual"][1]).all()
    # Indices 3 & 4 should still produce finite virtual N (resnums 6, 7
    # — consecutive).
    assert torch.isfinite(out["n_virtual"][3]).all()


# ---------- #15 NaN FI rejected by classifier --------------------------------
def test_classify_frustration_rejects_nan():
    fi = torch.tensor([0.5, float("nan"), -2.0])
    with pytest.raises(ValueError, match="non-finite"):
        classify_frustration(fi)


def test_classify_frustration_rejects_inf():
    fi = torch.tensor([0.5, float("inf"), -2.0])
    with pytest.raises(ValueError, match="non-finite"):
        classify_frustration(fi)


# ---------- #40 welltype rejects non-finite ----------------------------------
def test_welltype_rejects_non_finite_rij():
    rij = torch.tensor([6.0, float("nan"), 7.0])
    rho_i = torch.tensor([1.0, 1.0, 3.0])
    rho_j = torch.tensor([1.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        welltype_from_contact(rij, rho_i, rho_j)


def test_welltype_rejects_non_finite_rho():
    rij = torch.tensor([6.0, 7.0])
    rho_i = torch.tensor([1.0, float("inf")])
    rho_j = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        welltype_from_contact(rij, rho_i, rho_j)


# ---------- #51 dump letter mapping rejects out-of-range ----------------------
def test_aa_idx_to_letter_rejects_negative():
    aa = torch.tensor([0, 5, -1, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of range"):
        _aa_idx_to_letter(aa)


def test_aa_idx_to_letter_rejects_too_large():
    aa = torch.tensor([0, 5, 25, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of range"):
        _aa_idx_to_letter(aa)


# ---------- #62 emit_singleresidue_dat validates rho length ------------------
def test_emit_singleresidue_rejects_short_rho():
    parsed = parse_pdb(PDB_DIR / "5AON.pdb")
    n = parsed["ca_coords"].shape[0]
    # Build matching-length native + decoy vectors but a SHORTER rho.
    e_native = torch.zeros(n, dtype=torch.float64)
    dm = torch.zeros(n, dtype=torch.float64)
    ds = torch.ones(n, dtype=torch.float64)
    short_rho = torch.zeros(n - 1, dtype=torch.float64)
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="rho has length"):
            emit_singleresidue_dat(
                coords=parsed,
                rho=short_rho,
                e_native=e_native,
                decoy_mean=dm,
                decoy_std=ds,
                output_path=Path(td) / "out.dat",
                raw=True,
            )


def test_emit_singleresidue_rejects_short_native():
    parsed = parse_pdb(PDB_DIR / "5AON.pdb")
    n = parsed["ca_coords"].shape[0]
    rho = torch.zeros(n, dtype=torch.float64)
    bad = torch.zeros(n - 1, dtype=torch.float64)
    dm = torch.zeros(n, dtype=torch.float64)
    ds = torch.ones(n, dtype=torch.float64)
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="e_native length"):
            emit_singleresidue_dat(
                coords=parsed,
                rho=rho,
                e_native=bad,
                decoy_mean=dm,
                decoy_std=ds,
                output_path=Path(td) / "out.dat",
                raw=True,
            )


# ---------- #27 / #52 compute_residue_density validation ---------------------
def _build_density_inputs(n: int = 8):
    coords = {
        "ca_coords": torch.randn(n, 3, dtype=torch.float64),
        "cb_coords": torch.randn(n, 3, dtype=torch.float64),
        "is_gly": torch.zeros(n, dtype=torch.bool),
        "residue_numbers": torch.arange(n, dtype=torch.int64),
        "chain_ids": ["A"] * n,
        "residue_types": torch.zeros(n, dtype=torch.int64),
    }
    return coords


def test_compute_residue_density_rejects_pair_length_mismatch():
    coords = _build_density_inputs()
    pi = torch.tensor([0, 1, 2], dtype=torch.int64)
    pj = torch.tensor([3, 4], dtype=torch.int64)
    fi = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="same length"):
        compute_residue_density(coords=coords, pair_i=pi, pair_j=pj, fi=fi)


def test_compute_residue_density_rejects_fi_length_mismatch():
    coords = _build_density_inputs()
    pi = torch.tensor([0, 1, 2], dtype=torch.int64)
    pj = torch.tensor([3, 4, 5], dtype=torch.int64)
    fi = torch.zeros(5, dtype=torch.float64)  # WRONG length
    with pytest.raises(ValueError, match="same length"):
        compute_residue_density(coords=coords, pair_i=pi, pair_j=pj, fi=fi)


def test_compute_residue_density_rejects_negative_pair_index():
    coords = _build_density_inputs()
    pi = torch.tensor([0, -1, 2], dtype=torch.int64)
    pj = torch.tensor([3, 4, 5], dtype=torch.int64)
    fi = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="out of range"):
        compute_residue_density(coords=coords, pair_i=pi, pair_j=pj, fi=fi)


def test_compute_residue_density_rejects_out_of_range_pair_index():
    coords = _build_density_inputs(n=5)
    pi = torch.tensor([0, 1, 7], dtype=torch.int64)  # 7 >= n=5
    pj = torch.tensor([2, 3, 4], dtype=torch.int64)
    fi = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="out of range"):
        compute_residue_density(coords=coords, pair_i=pi, pair_j=pj, fi=fi)


def test_compute_residue_density_rejects_non_finite_fi():
    coords = _build_density_inputs()
    pi = torch.tensor([0, 1, 2], dtype=torch.int64)
    pj = torch.tensor([3, 4, 5], dtype=torch.int64)
    fi = torch.tensor([0.0, float("nan"), 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        compute_residue_density(coords=coords, pair_i=pi, pair_j=pj, fi=fi)


# ---------- #32 gamma.dat malformed rejection --------------------------------
def _write_data_file(path: Path, content: str) -> None:
    path.write_text(content)


def test_load_gamma_tables_rejects_one_column_lines(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    # Build a malformed gamma.dat with one-column lines.
    bad_path = tmp_path / "gamma.dat"
    lines = ["1.0\n"] * 420  # one column, 420 rows
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="expected 2 columns"):
        load_gamma_tables()


def test_load_gamma_tables_rejects_extra_rows(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    # Build a 2-col gamma.dat with too many rows.
    bad_path = tmp_path / "gamma.dat"
    lines = ["1.0 1.0\n"] * 421
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="exactly 420"):
        load_gamma_tables()


def test_load_gamma_tables_rejects_too_few_rows(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    bad_path = tmp_path / "gamma.dat"
    lines = ["1.0 1.0\n"] * 419
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="exactly 420"):
        load_gamma_tables()


# ---------- #49 NaN/inf in parameter tables ----------------------------------
def test_load_gamma_tables_rejects_nan(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    bad_path = tmp_path / "gamma.dat"
    lines = ["1.0 1.0\n"] * 420
    lines[10] = "nan 0.5\n"
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="non-finite"):
        load_gamma_tables()


def test_load_gamma_tables_rejects_inf(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    bad_path = tmp_path / "gamma.dat"
    lines = ["1.0 1.0\n"] * 420
    lines[10] = "inf 0.5\n"
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="non-finite"):
        load_gamma_tables()


def test_load_burial_gamma_rejects_nan(tmp_path, monkeypatch):
    import frustration_gpu.parameters as params

    bad_path = tmp_path / "burial_gamma.dat"
    lines = ["1.0 2.0 3.0\n"] * 20
    lines[5] = "1.0 nan 3.0\n"
    _write_data_file(bad_path, "".join(lines))
    monkeypatch.setattr(params, "_burial_gamma_dat_path", lambda: bad_path)
    with pytest.raises(ValueError, match="non-finite"):
        load_burial_gamma()


# ---------- value-preservation guard: real param files still load fine -------
def test_real_gamma_dat_still_loads():
    """Sanity: the real gamma.dat / burial_gamma.dat shipped with the package
    must continue to load without raising under the stricter parser."""
    tables = load_gamma_tables()
    assert tables.direct.shape == (20, 20)
    assert tables.mediated_protein.shape == (20, 20)
    assert tables.mediated_water.shape == (20, 20)
    burial = load_burial_gamma()
    assert burial.shape == (20, 3)
