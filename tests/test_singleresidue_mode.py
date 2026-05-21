"""Tests for AWSEM singleresidue-mode decoy machinery (Phase 3b).

Validation strategy
-------------------
For each panel PDB we compare against the LAMMPS-AWSEM dump
``benchmark/cpu_baseline/singleresidue/<PDB>_singleresidue.dat``:

* **Native energy** (column 5): deterministic per residue → exact match
  (mod 3-decimal print precision).
* **decoy_mean** (column 6) / **decoy_std** (column 7): RNG noise floor
  ~3% relative.
* **Frustration index FI** (column 8 = ``(decoy_mean - E_native) /
  decoy_std``): Spearman > 0.95 + rank-1/5/30 overlap > 80%.

Per the brief: absolute FI values may differ by 5-10%; what matters is
the rank ordering for frustration analysis. We use Spearman > 0.95
plus rank-30 overlap > 80% as the gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frustration_gpu.parser import parse_pdb                                          # noqa: E402
from frustration_gpu.singleresidue_decoys import singleresidue_decoy_stats            # noqa: E402

from _paths import DUMP_ROOT, PDB_DIR  # noqa: E402
DUMP_DIR = DUMP_ROOT / "singleresidue"
PANEL = ["5AON", "11BG", "1O3S", "3F9M"]


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


def _has_dump(pdb_id: str) -> bool:
    return (DUMP_DIR / f"{pdb_id}_singleresidue.dat").is_file()


def _parse_dump(pdb_id: str):
    """Parse the LAMMPS singleresidue dump → list of (nat, dm, ds, fi)."""
    rows = []
    fp = DUMP_DIR / f"{pdb_id}_singleresidue.dat"
    with fp.open() as fh:
        next(fh)  # header
        for line in fh:
            p = line.split()
            if len(p) < 8:
                continue
            # Res ChainRes DensityRes AA NativeEnergy DecoyEnergy SDEnergy FrstIndex
            rows.append((float(p[4]), float(p[5]), float(p[6]), float(p[7])))
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """No-scipy Spearman: rank-then-Pearson."""
    a_ranks = np.argsort(np.argsort(a)).astype(np.float64)
    b_ranks = np.argsort(np.argsort(b)).astype(np.float64)
    a_ranks -= a_ranks.mean()
    b_ranks -= b_ranks.mean()
    denom = np.linalg.norm(a_ranks) * np.linalg.norm(b_ranks)
    if denom == 0:
        return float("nan")
    return float(np.dot(a_ranks, b_ranks) / denom)


def _topk_overlap(ours: np.ndarray, theirs: np.ndarray, k: int) -> float:
    """Fraction of overlap between bottom-k of two arrays (most frustrated)."""
    kk = min(k, len(ours))
    o = set(np.argsort(ours)[:kk].tolist())
    t = set(np.argsort(theirs)[:kk].tolist())
    return len(o & t) / kk


# --- Gate 1: native energy matches dump --------------------------------------
@pytest.mark.parametrize("pdb_id", PANEL)
def test_singleresidue_native_matches_dump(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = singleresidue_decoy_stats(coords, seed=0)
    rows = _parse_dump(pdb_id)
    assert len(rows) == int(out["E_native"].numel()), \
        f"{pdb_id}: row mismatch ours={int(out['E_native'].numel())} dump={len(rows)}"
    theirs = np.array([r[0] for r in rows])
    ours = out["E_native"].cpu().numpy()
    max_diff = float(np.max(np.abs(ours - theirs)))
    assert max_diff < 5e-3, f"{pdb_id}: native max diff {max_diff:.4f}"


# --- Gate 2: FI Spearman > 0.95 ----------------------------------------------
@pytest.mark.parametrize("pdb_id", PANEL)
def test_singleresidue_FI_spearman(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = singleresidue_decoy_stats(coords, seed=0)
    rows = _parse_dump(pdb_id)
    theirs_fi = np.array([r[3] for r in rows])
    ours_fi = out["FI"].cpu().numpy()
    sp = _spearman(ours_fi, theirs_fi)
    assert sp > 0.95, f"{pdb_id}: FI Spearman {sp:.4f}"


# --- Gate 3: rank-1, rank-5, rank-30 overlap > 80% --------------------------
@pytest.mark.parametrize("pdb_id", PANEL)
def test_singleresidue_topk_overlap(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = singleresidue_decoy_stats(coords, seed=0)
    rows = _parse_dump(pdb_id)
    theirs_fi = np.array([r[3] for r in rows])
    ours_fi = out["FI"].cpu().numpy()
    for k in (1, 5, 30):
        kk = min(k, len(ours_fi))
        ov = _topk_overlap(ours_fi, theirs_fi, kk)
        # rank-1 should always be 100% on the panel; rank-5 / rank-30 require >= 80%
        assert ov >= 0.8, f"{pdb_id}: rank-{k} overlap {ov*100:.1f}% < 80%"


# --- Gate 4: decoy_mean / std Spearman --------------------------------------
@pytest.mark.parametrize("pdb_id", PANEL)
def test_singleresidue_decoy_stats_spearman(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = singleresidue_decoy_stats(coords, seed=0)
    rows = _parse_dump(pdb_id)
    theirs_dm = np.array([r[1] for r in rows])
    theirs_ds = np.array([r[2] for r in rows])
    ours_dm = out["decoy_mean"].cpu().numpy()
    ours_ds = out["decoy_std"].cpu().numpy()
    sp_dm = _spearman(ours_dm, theirs_dm)
    sp_ds = _spearman(ours_ds, theirs_ds)
    assert sp_dm > 0.95, f"{pdb_id}: decoy_mean Spearman {sp_dm:.4f}"
    assert sp_ds > 0.95, f"{pdb_id}: decoy_std  Spearman {sp_ds:.4f}"


# --- Gate 5: output shapes ---------------------------------------------------
def test_singleresidue_output_shapes():
    if not _has_pdb("5AON"):
        pytest.skip("5AON unavailable")
    coords = parse_pdb(PDB_DIR / "5AON.pdb")
    out = singleresidue_decoy_stats(coords, seed=0, n_decoys=128)
    n = int(out["E_native"].numel())
    assert out["decoy_mean"].shape == (n,)
    assert out["decoy_std"].shape == (n,)
    assert out["FI"].shape == (n,)
    assert out["aa_dec"].shape == (n, 128)
    assert torch.isfinite(out["E_native"]).all()
    assert torch.isfinite(out["decoy_mean"]).all()
    assert torch.isfinite(out["decoy_std"]).all()
    assert (out["decoy_std"] > 0).all()
    assert torch.isfinite(out["FI"]).all()


# --- Gate 6: reproducibility -------------------------------------------------
def test_singleresidue_seed_reproducibility():
    if not _has_pdb("5AON"):
        pytest.skip("5AON unavailable")
    coords = parse_pdb(PDB_DIR / "5AON.pdb")
    a = singleresidue_decoy_stats(coords, seed=42)
    b = singleresidue_decoy_stats(coords, seed=42)
    assert torch.allclose(a["FI"], b["FI"], atol=1e-9)
    c = singleresidue_decoy_stats(coords, seed=7)
    assert (a["FI"] - c["FI"]).abs().max() > 1e-3


# --- Gate 7: rank-1 most-frustrated residue is identical to LAMMPS pick ---
@pytest.mark.parametrize("pdb_id", PANEL)
def test_rank_one_most_frustrated(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = singleresidue_decoy_stats(coords, seed=0)
    rows = _parse_dump(pdb_id)
    theirs_fi = np.array([r[3] for r in rows])
    ours_fi = out["FI"].cpu().numpy()
    our_rank1 = int(np.argmin(ours_fi))
    their_rank1 = int(np.argmin(theirs_fi))
    assert our_rank1 == their_rank1, (
        f"{pdb_id}: rank-1 disagreement ours={our_rank1} theirs={their_rank1} "
        f"(ours_fi={ours_fi[our_rank1]:.3f}, theirs_fi={theirs_fi[their_rank1]:.3f})"
    )


# --- Gate 8: CPU vs GPU agreement --------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_singleresidue_cpu_gpu_agreement_5aon():
    coords_cpu = parse_pdb(PDB_DIR / "5AON.pdb")
    out_cpu = singleresidue_decoy_stats(coords_cpu, seed=0)
    coords_gpu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coords_cpu.items()}
    out_gpu = singleresidue_decoy_stats(coords_gpu, seed=0, device="cuda")
    diff_fi = (out_cpu["FI"] - out_gpu["FI"].cpu()).abs().max().item()
    assert diff_fi < 1e-6, f"CPU vs GPU FI max diff {diff_fi:.2e}"


# --- Gate 9: all 10 panel PDBs produce finite output ------------------------
def test_singleresidue_all_panel_no_nan():
    if not DUMP_DIR.is_dir():
        pytest.skip("dump directory unavailable")
    available = sorted(
        p.name.removesuffix("_singleresidue.dat")
        for p in DUMP_DIR.glob("*_singleresidue.dat")
        if _has_pdb(p.name.removesuffix("_singleresidue.dat"))
    )
    assert available, "no panel PDBs available"
    for pdb_id in available:
        coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
        out = singleresidue_decoy_stats(coords, seed=0)
        assert torch.isfinite(out["FI"]).all(), f"{pdb_id}: NaN FI"
        assert (out["decoy_std"] > 0).all(), f"{pdb_id}: decoy_std contains 0"
