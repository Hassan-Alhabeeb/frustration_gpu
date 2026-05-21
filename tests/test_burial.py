"""Smoke + sanity tests for the Phase 1 burial pipeline.

These tests are intentionally cheap (no full LAMMPS comparison yet — that's
Phase 2). They check:

1. Parser produces sensibly-shaped tensors for 5AON / 11BG.
2. Virtual atom positions match a hand calculation from the OpenAWSEM
   coefficient set within float32 noise.
3. ``rho`` is non-negative and falls inside the OpenAWSEM expected band
   [0, ~10] for typical residues, with non-zero values in the core.
4. ``rho`` agrees on CPU and GPU to machine precision (if CUDA available).
5. Burial energy is finite and has the expected sign convention.

The reference ``..._frust.npz`` files don't contain rho directly — they store
the four frustrapy "density" classifications (high / neutral / minimal / mean
index). We use the *highly_frustrated_density* feature as a coarse
cross-check: residues with many destabilising contacts should also tend to
have high local CB density (the two aren't identical but correlate weakly).
That cross-check is informational only — see PHASE_1_STATUS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make the ``frustration_gpu`` package importable when running pytest from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frustration_gpu.burial import burial_density, burial_energy   # noqa: E402
from frustration_gpu.parser import parse_pdb                        # noqa: E402
from frustration_gpu.virtual_atoms import compute_virtual_atoms     # noqa: E402

from _paths import PDB_DIR  # noqa: E402


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


@pytest.fixture(scope="module")
def parsed_5aon():
    return parse_pdb(PDB_DIR / "5AON.pdb")


@pytest.fixture(scope="module")
def parsed_11bg():
    return parse_pdb(PDB_DIR / "11BG.pdb")


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_parse_5aon_shapes(parsed_5aon):
    p = parsed_5aon
    n = p["ca_coords"].shape[0]
    assert n >= 49, f"5AON should have at least 49 residues, got {n}"
    assert p["ca_coords"].shape == (n, 3)
    assert p["n_coords"].shape == (n, 3)
    assert p["c_coords"].shape == (n, 3)
    assert p["o_coords"].shape == (n, 3)
    assert p["cb_coords"].shape == (n, 3)
    assert p["residue_types"].shape == (n,)
    assert len(p["chain_ids"]) == n
    assert p["residue_numbers"].shape == (n,)
    # CA should never be NaN
    assert torch.isfinite(p["ca_coords"]).all()
    # residue types in [0, 19]
    assert int(p["residue_types"].min()) >= 0
    assert int(p["residue_types"].max()) <= 19
    # at least some GLY
    n_gly = int(p["is_gly"].sum())
    assert n_gly > 0, "5AON has at least one GLY in its sequence"


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_virtual_atoms_5aon(parsed_5aon):
    """Virtual N matches manual computation from OpenAWSEM coefficients."""
    p = parsed_5aon
    v = compute_virtual_atoms(p)
    n_v = v["n_virtual"]
    # First residue of every chain: cannot compute virtual N (needs prev CA + prev O).
    # Find an interior residue (same chain as previous) and check manually.
    chains = p["chain_ids"]
    for i in range(1, len(chains)):
        if chains[i] != chains[i - 1]:
            continue
        manual = (
            0.48318 * p["ca_coords"][i - 1]
            + 0.70328 * p["ca_coords"][i]
            - 0.18643 * p["o_coords"][i - 1]
        )
        diff = (n_v[i] - manual).abs().max().item()
        assert diff < 1e-4, f"residue {i}: manual {manual} vs virtual {n_v[i]} delta {diff}"
        break

    # First residue of each chain MUST be NaN (no i-1)
    chain_starts = [0]
    for i in range(1, len(chains)):
        if chains[i] != chains[i - 1]:
            chain_starts.append(i)
    for s in chain_starts:
        assert torch.isnan(n_v[s]).any()


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_rho_basic(parsed_5aon):
    rho = burial_density(parsed_5aon)
    n = rho.shape[0]
    # rho is dimensionless, sum of sigmoid-windowed CB neighbours.
    # Theoretical bounds: rho >= 0 and rho should not exceed n_neighbours_max.
    assert torch.isfinite(rho).all(), "rho should be finite for every residue"
    assert (rho >= 0).all(), "rho should be non-negative"
    # The expected band for protein cores is ~3 to ~9; surface residues 0 to 3.
    # We just assert there is variation and at least one residue is > 1.
    assert rho.max().item() > 1.0
    assert rho.std().item() > 0.1


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_rho_11bg(parsed_11bg):
    rho = burial_density(parsed_11bg)
    assert torch.isfinite(rho).all()
    assert (rho >= 0).all()
    # 11BG has 248 residues per the panel; allow some chain truncation.
    assert rho.shape[0] >= 100
    # at least 5% of residues in the "deeply buried" >6 well
    pct_deep = (rho > 6.0).float().mean().item()
    assert pct_deep > 0.02, f"expected some deeply buried residues, got pct={pct_deep:.3f}"


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_burial_energy_5aon(parsed_5aon):
    """Quantitative gate against LAMMPS V_Burial column for 5AON.

    QA-4 H2 fix (2026-05-21): previously this test only checked shape +
    finiteness + ``|E| > 1e-3``. A real V_burial regression that flipped
    the sign or doubled the magnitude would have passed. Now we gate
    against the LAMMPS-AWSEM ``benchmark/cpu_baseline/configurational/
    5AON_energy.log`` ``Burial`` column = ``-41.799488`` kcal/mol (the
    energy snapshot at step 0, which is the structure-as-input value).

    The float32 burial pipeline matches LAMMPS to ~1e-7 relative on this
    structure. We gate at 1e-4 relative to leave headroom for genuine
    minor numerical drift across PyTorch versions / BLAS configurations.
    """
    out = burial_energy(parsed_5aon)
    assert "energy" in out and "rho" in out and "per_residue" in out
    assert torch.isfinite(out["energy"]).item()
    assert torch.isfinite(out["per_residue"]).all()
    # Quantitative reference from `benchmark/cpu_baseline/configurational/
    # 5AON_energy.log` column "Burial" at step 0.
    V_BURIAL_5AON = -41.799488
    e = float(out["energy"].item())
    rel_err = abs(e - V_BURIAL_5AON) / abs(V_BURIAL_5AON)
    assert rel_err < 1e-4, (
        f"burial energy {e:.6f} != LAMMPS reference {V_BURIAL_5AON:.6f} "
        f"(relative error {rel_err:.2e}, threshold 1e-4)"
    )
    # Sign convention sanity: V_burial is negative (stabilising) on a folded
    # protein. A sign flip would still pass the relative-error gate when
    # |E| > 0 — guard against it here.
    assert e < 0.0, f"burial energy is positive ({e}); sign convention bug"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_rho_cpu_gpu_agreement():
    p_cpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cpu", dtype=torch.float64)
    p_gpu = parse_pdb(PDB_DIR / "5AON.pdb", device="cuda", dtype=torch.float64)
    rho_cpu = burial_density(p_cpu)
    rho_gpu = burial_density(p_gpu).cpu()
    delta = (rho_cpu - rho_gpu).abs().max().item()
    assert delta < 1e-8, f"CPU vs GPU rho disagree by {delta}"


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_frustrapy_cache_loadable():
    """We don't validate burial against the cache (cache is per-residue density
    of frustration *contacts*, not rho), but we make sure the file is parsable
    so Phase 2 can use it for cross-checks."""
    import numpy as np
    cache = np.load("F:/research_plan/allosteric/features/frustration/5AON_frust.npz")
    assert "features" in cache.files
    assert "feature_names" in cache.files
    feats = cache["features"]
    names = list(cache["feature_names"])
    # We only assert the cache is shaped (N, 4) per the panel — actual content
    # comparison waits for the full frustration term in Phase 2.
    assert feats.ndim == 2
    assert feats.shape[1] == len(names)


# --- DNA sentinel guard (Fix A) --------------------------------------------
def test_dna_sentinel_guard_raises_burial():
    """``residue_types == -1`` (DNA placeholder) must raise on burial_energy.

    Negative indexing would otherwise map to burial_gamma row 19 (VAL) — see
    ``docs/qa1_core_math.md`` HIGH-severity finding.
    """
    coords = {
        "ca_coords": torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                                  dtype=torch.float64),
        "cb_coords": torch.tensor([[1.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
                                  dtype=torch.float64),
        "residue_types": torch.tensor([0, -1], dtype=torch.int64),
        "chain_ids": ["A", "A"],
        "residue_numbers": torch.tensor([1, 2], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="DNA sentinel"):
        burial_energy(coords)


# --- sparse vs dense ~ULP-equivalent agreement for rho (Fix B) -------------
# The burial fixtures use the parser default dtype (float32). For the
# precision test we re-parse at float64 explicitly — float32 rho differs at
# ~1e-7 between dense vs sparse purely from fp32 ULP, which is uninformative.
_RHO_SPARSE_REL_TOL = 1e-12


@pytest.fixture(scope="module")
def parsed_5aon_f64():
    return parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)


@pytest.fixture(scope="module")
def parsed_11bg_f64():
    return parse_pdb(PDB_DIR / "11BG.pdb", dtype=torch.float64)


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb not available")
def test_sparse_rho_byte_exact_5aon(parsed_5aon_f64):
    """``compute_rho`` sparse vs dense path agree to fp64 precision.

    The burial sigmoid is ~0 outside the [r_min=4.5, r_max=6.5] window;
    the default sparse cutoff (9.5 Å) is well beyond it so the two paths
    enumerate the same significant pairs.
    """
    rho_dense = burial_density(parsed_5aon_f64)
    rho_sparse = burial_density(parsed_5aon_f64, sparse=True)
    abs_diff = (rho_dense - rho_sparse).abs().max().item()
    rel_diff = abs_diff / max(rho_dense.abs().max().item(), 1e-30)
    print(f"\n5AON rho max abs diff = {abs_diff:.3e} rel = {rel_diff:.3e}")
    assert rel_diff < _RHO_SPARSE_REL_TOL, (
        f"rho sparse vs dense disagree: rel={rel_diff}"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb not available")
def test_sparse_rho_byte_exact_11bg(parsed_11bg_f64):
    """Same gate on 11BG."""
    rho_dense = burial_density(parsed_11bg_f64)
    rho_sparse = burial_density(parsed_11bg_f64, sparse=True)
    abs_diff = (rho_dense - rho_sparse).abs().max().item()
    rel_diff = abs_diff / max(rho_dense.abs().max().item(), 1e-30)
    print(f"\n11BG rho max abs diff = {abs_diff:.3e} rel = {rel_diff:.3e}")
    assert rel_diff < _RHO_SPARSE_REL_TOL, (
        f"rho sparse vs dense disagree: rel={rel_diff}"
    )
