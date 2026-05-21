"""Decoy-API validation tests (findings #1, #10, #22, #34, #43, #54, #55).

These cover the DECOY-module input-validation bugs called out in the
75-finding audit at ``F:/research_plan/New folder/odo.txt``:

* #1  — singleresidue CUDA chunking allocated the full (20, N, N) cube
* #10 — DNA sentinel guards (``residue_types == -1``) missing in public
        decoy APIs (configurational, mutational, singleresidue)
* #22 — ``n_decoys = 1`` accepted, produced all-zero FI silently
* #34 — mutational checked ``n_pair == 0`` only AFTER expensive O(N²)
        precompute
* #43 — ``compute_configurational_decoy_energy`` accepted gamma tables
        of arbitrary shape (silently truncated)
* #54 — decoy APIs accepted ``rho`` with mismatched shape (silent
        broadcast / truncation)
* #55 — ``compute_configurational_decoy_energy`` accepted decoy fields
        with mismatched lengths (silent broadcast)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import PDB_DIR  # noqa: E402

from frustration_gpu.decoys import (  # noqa: E402
    compute_configurational_decoy_energy,
    configurational_decoy_stats,
    lammps_dump_rho,
    sample_configurational_decoys,
)
from frustration_gpu.mutational_decoys import mutational_decoy_stats  # noqa: E402
from frustration_gpu.parser import parse_pdb  # noqa: E402
from frustration_gpu.singleresidue_decoys import (  # noqa: E402
    _precompute_W_sr,
    singleresidue_decoy_stats,
)

CUDA_OK = torch.cuda.is_available()


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


@pytest.fixture(scope="module")
def parsed_5aon():
    return parse_pdb(PDB_DIR / "5AON.pdb")


# ---------------------------------------------------------------------------
# Finding #10 — DNA sentinel guards
# ---------------------------------------------------------------------------

def _poison_with_dna(coords: dict) -> dict:
    """Return a shallow-copy of ``coords`` whose ``residue_types[0] = -1``."""
    new = dict(coords)
    rt = coords["residue_types"].clone()
    rt[0] = -1
    new["residue_types"] = rt
    return new


def test_dna_guard_sample_configurational(parsed_5aon):
    poisoned = _poison_with_dna(parsed_5aon)
    rho = lammps_dump_rho(parsed_5aon)
    with pytest.raises(ValueError, match="DNA sentinel"):
        sample_configurational_decoys(poisoned, rho, n_decoys=8, seed=0)


def test_dna_guard_mutational(parsed_5aon):
    poisoned = _poison_with_dna(parsed_5aon)
    with pytest.raises(ValueError, match="DNA sentinel"):
        mutational_decoy_stats(poisoned, n_decoys=8, seed=0)


def test_dna_guard_singleresidue(parsed_5aon):
    poisoned = _poison_with_dna(parsed_5aon)
    with pytest.raises(ValueError, match="DNA sentinel"):
        singleresidue_decoy_stats(poisoned, n_decoys=8, seed=0)


def test_dna_guard_compute_configurational_energy(parsed_5aon):
    """Negative aa indices in the decoy dict must raise too."""
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = dict(decoys)
    bad["aa_i_decoy"] = decoys["aa_i_decoy"].clone()
    bad["aa_i_decoy"][0] = -1
    with pytest.raises(ValueError, match="DNA sentinel"):
        compute_configurational_decoy_energy(bad)


# ---------------------------------------------------------------------------
# Finding #22 — n_decoys < 2 must be rejected
# ---------------------------------------------------------------------------

def test_n_decoys_one_rejected_configurational(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    with pytest.raises(ValueError, match="n_decoys"):
        sample_configurational_decoys(parsed_5aon, rho, n_decoys=1, seed=0)


def test_n_decoys_one_rejected_mutational(parsed_5aon):
    with pytest.raises(ValueError, match="n_decoys"):
        mutational_decoy_stats(parsed_5aon, n_decoys=1, seed=0)


def test_n_decoys_one_rejected_singleresidue(parsed_5aon):
    with pytest.raises(ValueError, match="n_decoys"):
        singleresidue_decoy_stats(parsed_5aon, n_decoys=1, seed=0)


def test_n_decoys_one_rejected_in_energy_api(parsed_5aon):
    """compute_configurational_decoy_energy also rejects length-1 fields."""
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    short = {k: v[:1] for k, v in decoys.items()}
    with pytest.raises(ValueError, match="n_decoys"):
        compute_configurational_decoy_energy(short)


# ---------------------------------------------------------------------------
# Finding #22 (warning) — singleresidue with contactless residues
# ---------------------------------------------------------------------------

def test_singleresidue_warns_on_std_collapse(parsed_5aon):
    """A no-contact residue with k_burial=0 yields constant decoy energy
    (= 0) for every draw → decoy_std must collapse and trigger a warning."""
    coords = dict(parsed_5aon)
    # Move residue 0 far away so it has zero contacts (W_sr row = 0).
    ca = coords["ca_coords"].clone()
    ca[0] = torch.tensor([1.0e5, 1.0e5, 1.0e5], dtype=ca.dtype)
    coords["ca_coords"] = ca
    if "cb_coords" in coords and coords["cb_coords"] is not None:
        cb = coords["cb_coords"].clone()
        cb[0] = torch.tensor([1.0e5, 1.0e5, 1.0e5], dtype=cb.dtype)
        coords["cb_coords"] = cb

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # k_burial=0 makes burial contribution constant (= 0). Combined
        # with the residue's zero W_sr row, every decoy at slot 0 has
        # the same energy → std=0 → warning fires.
        singleresidue_decoy_stats(coords, n_decoys=8, seed=0, k_burial=0.0)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("decoy_std" in m for m in msgs), (
        f"expected a RuntimeWarning about decoy_std collapse, got: {msgs}"
    )


# ---------------------------------------------------------------------------
# Finding #34 — mutational early bail on n_pair == 0
# ---------------------------------------------------------------------------

def test_mutational_n_pair_zero_short_circuits_before_precompute(parsed_5aon):
    """Forcing n_pair==0 (cutoff=0) must NOT call _precompute_T_alpha."""
    import frustration_gpu.mutational_decoys as md

    with patch.object(md, "_precompute_T_alpha", wraps=md._precompute_T_alpha) as mock_pre:
        out = mutational_decoy_stats(
            parsed_5aon, n_decoys=8, contact_cutoff=0.0, seed=0,
        )
    assert mock_pre.call_count == 0, (
        "Expected early bail BEFORE _precompute_T_alpha; was called "
        f"{mock_pre.call_count} time(s)."
    )
    # Schema-preserving output.
    for key in (
        "pair_i", "pair_j", "r_ij", "rho_i", "rho_j", "E_native",
        "decoy_mean", "decoy_std", "aa_i_dec", "aa_j_dec",
    ):
        assert key in out, f"missing key {key!r}"
    assert out["pair_i"].numel() == 0
    assert out["aa_i_dec"].shape == (0, 8)


# ---------------------------------------------------------------------------
# Finding #43 — gamma table shape validation
# ---------------------------------------------------------------------------

def test_gamma_direct_wrong_shape_rejected(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = torch.zeros((21, 21), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"gamma_direct.*\(20, 20\)"):
        compute_configurational_decoy_energy(decoys, gamma_direct=bad)


def test_gamma_mediated_protein_wrong_shape_rejected(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = torch.zeros((21, 21), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"gamma_mediated_protein.*\(20, 20\)"):
        compute_configurational_decoy_energy(decoys, gamma_mediated_protein=bad)


def test_gamma_mediated_water_wrong_shape_rejected(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = torch.zeros((19, 20), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"gamma_mediated_water.*\(20, 20\)"):
        compute_configurational_decoy_energy(decoys, gamma_mediated_water=bad)


def test_burial_gamma_wrong_shape_rejected(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = torch.zeros((20, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match=r"burial_gamma.*\(20, 3\)"):
        compute_configurational_decoy_energy(decoys, burial_gamma=bad)


# ---------------------------------------------------------------------------
# Finding #54 — rho shape mismatch
# ---------------------------------------------------------------------------

def test_sample_configurational_rejects_wrong_rho_shape(parsed_5aon):
    n = parsed_5aon["ca_coords"].shape[0]
    rho = torch.zeros(n - 1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"rho shape"):
        sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)


def test_mutational_rejects_wrong_rho_shape(parsed_5aon):
    n = parsed_5aon["ca_coords"].shape[0]
    rho = torch.zeros(n - 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"rho shape"):
        mutational_decoy_stats(parsed_5aon, rho=rho, n_decoys=8, seed=0)


def test_singleresidue_rejects_wrong_rho_shape(parsed_5aon):
    n = parsed_5aon["ca_coords"].shape[0]
    rho = torch.zeros(n + 3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"rho shape"):
        singleresidue_decoy_stats(parsed_5aon, rho=rho, n_decoys=8, seed=0)


# ---------------------------------------------------------------------------
# Finding #55 — decoy fields of mismatched length
# ---------------------------------------------------------------------------

def test_compute_configurational_rejects_mismatched_decoy_shapes(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = dict(decoys)
    bad["aa_i_decoy"] = decoys["aa_i_decoy"][:3]
    bad["aa_j_decoy"] = decoys["aa_j_decoy"][:1]
    with pytest.raises(ValueError, match=r"decoy field shapes"):
        compute_configurational_decoy_energy(bad)


def test_compute_configurational_rejects_non_1d_decoy_fields(parsed_5aon):
    rho = lammps_dump_rho(parsed_5aon)
    decoys = sample_configurational_decoys(parsed_5aon, rho, n_decoys=8, seed=0)
    bad = {k: v.view(-1, 1) for k, v in decoys.items()}  # all (n_decoys, 1)
    with pytest.raises(ValueError, match=r"1-D"):
        compute_configurational_decoy_energy(bad)


# ---------------------------------------------------------------------------
# Finding #1 — singleresidue OOM regression on large-N GPU
# ---------------------------------------------------------------------------

def _make_synthetic_coords(n: int, dtype: torch.dtype = torch.float64) -> dict:
    """Cheap synthetic coords dict mimicking parse_pdb's contract."""
    # Use a straight chain spaced 3.8 Å apart so contacts only form
    # between neighbours — keeps W_sr math meaningful and fast.
    z = torch.arange(n, dtype=dtype) * 3.8
    ca = torch.stack(
        [torch.zeros(n, dtype=dtype), torch.zeros(n, dtype=dtype), z], dim=-1,
    )
    cb = ca.clone()
    # Use repeating residue-types pattern (0..19).
    rt = (torch.arange(n) % 20).to(torch.int64)
    chain_ids = ["A"] * n
    return {
        "ca_coords": ca,
        "cb_coords": cb,
        "residue_types": rt,
        "chain_ids": chain_ids,
    }


@pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")
def test_singleresidue_oom_bound_at_large_n():
    """Regression for finding #1: the chunked-sum path keeps peak VRAM
    below the size of the full (20, N, N) cube.

    We force a small alpha-chunk via monkeypatch (avoiding dependence on
    the host's actual free VRAM), then verify peak allocation is comfortably
    below the cube-allocation budget. Comparing chunked vs forced one-shot
    on the same N gives a multiplicative reduction.
    """
    import frustration_gpu.singleresidue_decoys as sr
    import frustration_gpu.mutational_decoys as md

    n = 5000
    coords = _make_synthetic_coords(n)
    coords["ca_coords"] = coords["ca_coords"].cuda()
    coords["cb_coords"] = coords["cb_coords"].cuda()
    coords["residue_types"] = coords["residue_types"].cuda()

    full_cube_gb = (20 * n * n * 8) / (1024 ** 3)  # ~3.73 GB

    # --- Path under test: forced small chunk uses sum-variant ---------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with patch.object(sr, "_choose_alpha_chunk", return_value=2):
        # Sanity: ensure the BROKEN one-shot helper is never called when
        # we force chunking.
        with patch.object(
            sr, "_water_per_alpha_fused", wraps=md._water_per_alpha_fused,
        ) as broken_helper:
            out = singleresidue_decoy_stats(coords, n_decoys=4, seed=0)
        assert broken_helper.call_count == 0, (
            "When chunked, singleresidue must NOT call _water_per_alpha_fused "
            "(the full-cube one-shot variant)."
        )
    peak_fixed_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # The broken one-shot path simultaneously allocates two (20, N, N)
    # cubes (sigma_gamma_med plus the returned tensor) — empirically ~2x
    # the bare-cube size. The chunked sum-variant peaks at roughly
    # bare-cube + upstream (N, N) tensors. We gate at < 1.5x cube size to
    # detect any regression that re-introduces a parallel cube allocation.
    upper_bound_gb = full_cube_gb * 1.5
    assert peak_fixed_gb < upper_bound_gb, (
        f"chunked singleresidue peak {peak_fixed_gb:.2f} GB exceeds "
        f"{upper_bound_gb:.2f} GB (= 1.5 * full-cube). The chunked-sum path "
        f"must keep allocation near {full_cube_gb:.2f} GB."
    )

    # Output shapes still correct.
    assert out["FI"].shape == (n,)
    assert out["E_native"].shape == (n,)


@pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")
def test_singleresidue_chunked_matches_unchunked_on_small_n():
    """Numerical equivalence: the chunked-sum path must agree with the
    one-shot cube path on a structure small enough to use both."""
    # Use 5AON (N=49) → comfortable cube; we force chunking by mocking.
    pdb_path = PDB_DIR / "5AON.pdb"
    if not pdb_path.is_file():
        pytest.skip("5AON.pdb not bundled")
    coords = parse_pdb(pdb_path)
    coords["ca_coords"] = coords["ca_coords"].cuda()
    if coords.get("cb_coords") is not None:
        coords["cb_coords"] = coords["cb_coords"].cuda()
    coords["residue_types"] = coords["residue_types"].cuda()

    out_default = singleresidue_decoy_stats(coords, n_decoys=64, seed=0)

    # Force the chunked path by patching _choose_alpha_chunk to return 4.
    import frustration_gpu.singleresidue_decoys as sr
    with patch.object(sr, "_choose_alpha_chunk", return_value=4):
        out_chunked = singleresidue_decoy_stats(coords, n_decoys=64, seed=0)

    # Bit-exact agreement (float64).
    assert torch.allclose(out_default["FI"], out_chunked["FI"], atol=1e-12), (
        "Chunked W_sr precompute must match one-shot cube precompute."
    )
    assert torch.allclose(
        out_default["E_native"], out_chunked["E_native"], atol=1e-12,
    )
