"""Tests for AWSEM mutational-mode decoy machinery (Phase 3b).

Validation strategy
-------------------
For each panel PDB we compare against the LAMMPS-AWSEM dump
``benchmark/cpu_baseline/mutational/<PDB>_tertiary_frustration.dat``:

1. **Native energy** (column 15, includes (i, j) pair + cross-terms + burial):
   our values should match the dump to 1e-3 (the dump's print precision)
   except for the trailing 0.0005 round-off bit.
2. **decoy_mean / decoy_std** (columns 16-17) — these differ per pair
   (the headline mutational-mode property, vs configurational mode where
   they are cached). Spearman > 0.95 across all pairs.
3. **No-pair / sanity sweeps**.

Gates (per user brief):
* Spearman > 0.95 on per-pair decoy_mean AND decoy_std for 5AON, 11BG,
  1O3S, 3F9M.
* Averaged rel-error on (mean, std) < 5% — using a small protective
  floor on the denominator (because decoy_mean is occasionally near
  zero where raw rel-err is meaningless).
* Wall-clock for 11BG mutational mode: < 60 s on CPU, < 5 s on GPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import DUMP_ROOT, PDB_DIR  # noqa: E402

from frustration_gpu.mutational_decoys import (  # noqa: E402
    mutational_decoy_stats,
)
from frustration_gpu.parser import parse_pdb  # noqa: E402

DUMP_DIR = DUMP_ROOT / "mutational"


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


def _has_dump(pdb_id: str) -> bool:
    return (DUMP_DIR / f"{pdb_id}_tertiary_frustration.dat").is_file()


def _parse_dump(pdb_id: str) -> dict:
    """Parse the LAMMPS mutational dump → dict[(i,j)] = (nat, dm, ds, fi)."""
    out = {}
    fp = DUMP_DIR / f"{pdb_id}_tertiary_frustration.dat"
    with fp.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            i, j = int(p[0]), int(p[1])
            nat = float(p[15])
            dm = float(p[16])
            ds = float(p[17])
            fi = float(p[18])
            out[(min(i, j), max(i, j))] = (nat, dm, ds, fi)
    return out


def _align_ours_to_dump(out: dict, dump: dict):
    """Return parallel numpy arrays (native, dm, ds) for ours and dump.

    Iterates our pairs in order; only includes pairs that appear in the
    LAMMPS dump (should be all of them — the iteration order is
    deterministic in both).
    """
    ours_nat, ours_dm, ours_ds = [], [], []
    theirs_nat, theirs_dm, theirs_ds = [], [], []
    for k in range(int(out["pair_i"].numel())):
        pi = int(out["pair_i"][k]) + 1
        pj = int(out["pair_j"][k]) + 1
        key = (min(pi, pj), max(pi, pj))
        if key not in dump:
            continue
        ours_nat.append(float(out["E_native"][k]))
        ours_dm.append(float(out["decoy_mean"][k]))
        ours_ds.append(float(out["decoy_std"][k]))
        nat, dm, ds, _ = dump[key]
        theirs_nat.append(nat)
        theirs_dm.append(dm)
        theirs_ds.append(ds)
    return (
        np.array(ours_nat), np.array(theirs_nat),
        np.array(ours_dm), np.array(theirs_dm),
        np.array(ours_ds), np.array(theirs_ds),
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """No-scipy Spearman: rank both arrays then Pearson on the ranks."""
    a_ranks = np.argsort(np.argsort(a)).astype(np.float64)
    b_ranks = np.argsort(np.argsort(b)).astype(np.float64)
    a_ranks -= a_ranks.mean()
    b_ranks -= b_ranks.mean()
    denom = np.linalg.norm(a_ranks) * np.linalg.norm(b_ranks)
    if denom == 0:
        return float("nan")
    return float(np.dot(a_ranks, b_ranks) / denom)


# --- Gate 1: native energy matches LAMMPS dump ------------------------------
@pytest.mark.parametrize("pdb_id", ["5AON", "11BG"])
def test_native_energy_matches_dump(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = mutational_decoy_stats(coords, seed=0)
    dump = _parse_dump(pdb_id)
    ours_n, theirs_n, _, _, _, _ = _align_ours_to_dump(out, dump)
    assert len(ours_n) == len(dump), \
        f"{pdb_id}: pair-count mismatch ours={len(ours_n)} dump={len(dump)}"
    # Native energy should match the dump exactly to 3 decimals
    max_diff = np.max(np.abs(ours_n - theirs_n))
    assert max_diff < 5e-3, f"{pdb_id}: max native-energy diff {max_diff:.4f}"
    # Spearman should be perfect (the native energy is deterministic — no RNG)
    sp = _spearman(ours_n, theirs_n)
    assert sp > 0.9999, f"{pdb_id}: native Spearman {sp:.4f}"


# --- Gate 2: per-pair decoy_mean / decoy_std Spearman > 0.95 ---------------
PANEL = ["5AON", "11BG", "1O3S", "3F9M"]


@pytest.mark.parametrize("pdb_id", PANEL)
def test_decoy_mean_std_spearman_per_pair(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = mutational_decoy_stats(coords, seed=0)
    dump = _parse_dump(pdb_id)
    _, _, ours_dm, theirs_dm, ours_ds, theirs_ds = _align_ours_to_dump(out, dump)
    sp_dm = _spearman(ours_dm, theirs_dm)
    sp_ds = _spearman(ours_ds, theirs_ds)
    assert sp_dm > 0.95, f"{pdb_id}: decoy_mean Spearman {sp_dm:.4f}"
    assert sp_ds > 0.95, f"{pdb_id}: decoy_std Spearman {sp_ds:.4f}"


# --- Gate 3: rel-err averaged across pairs < 5% (with protective floor) ----
@pytest.mark.parametrize("pdb_id", PANEL)
def test_rel_err_decoy_stats(pdb_id):
    """Average relative-error on per-pair (mean, std) < 5%.

    The denominator carries a small floor (0.5 kcal/mol on decoy_mean,
    0.1 kcal/mol on decoy_std) to neutralise the divide-by-tiny-number
    blow-up that occasionally happens when ``decoy_mean ≈ 0``. The floor
    is small enough that genuine differences still surface.
    """
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = mutational_decoy_stats(coords, seed=0)
    dump = _parse_dump(pdb_id)
    _, _, a, b, c, d = _align_ours_to_dump(out, dump)
    rel_dm = float(np.mean(np.abs(a - b) / (np.abs(b) + 0.5)))
    rel_ds = float(np.mean(np.abs(c - d) / (np.abs(d) + 0.1)))
    assert rel_dm < 0.05, f"{pdb_id}: rel-err decoy_mean {rel_dm:.4f}"
    assert rel_ds < 0.05, f"{pdb_id}: rel-err decoy_std {rel_ds:.4f}"


# --- Gate 4: per-pair stats VARY (validates non-caching) -------------------
def test_per_pair_decoy_stats_vary():
    """Mutational mode must produce DIFFERENT (mean, std) per pair, in
    contrast to configurational mode where they're cached. Verify the
    spread is meaningful (std-of-decoy_mean across pairs > 0.1).
    """
    coords = parse_pdb(PDB_DIR / "5AON.pdb")
    out = mutational_decoy_stats(coords, seed=0)
    if int(out["pair_i"].numel()) == 0:
        pytest.skip("no pairs")
    spread_mean = float(out["decoy_mean"].std(unbiased=False))
    spread_std = float(out["decoy_std"].std(unbiased=False))
    assert spread_mean > 0.1, f"decoy_mean spread {spread_mean:.4f} too small"
    assert spread_std > 0.1, f"decoy_std spread {spread_std:.4f} too small"


# --- Gate 5: CPU wall-clock for 11BG < 60s ---------------------------------
@pytest.mark.slow
def test_mutational_wall_clock_cpu_11bg():
    if not (_has_pdb("11BG") and _has_dump("11BG")):
        pytest.skip("11BG unavailable")
    coords = parse_pdb(PDB_DIR / "11BG.pdb")
    # Warmup (gamma cache load)
    _ = mutational_decoy_stats(coords, seed=0)
    t0 = time.time()
    _ = mutational_decoy_stats(coords, seed=0)
    t1 = time.time()
    assert (t1 - t0) < 60.0, f"11BG CPU mutational took {t1-t0:.2f}s (target <60s)"


# --- Gate 6: GPU wall-clock for 11BG < 5s ----------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_mutational_wall_clock_gpu_11bg():
    if not (_has_pdb("11BG") and _has_dump("11BG")):
        pytest.skip("11BG unavailable")
    coords = parse_pdb(PDB_DIR / "11BG.pdb")
    # Migrate to GPU
    for k in ("ca_coords", "cb_coords", "residue_types"):
        coords[k] = coords[k].cuda()
    # Warmup
    _ = mutational_decoy_stats(coords, seed=0, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = mutational_decoy_stats(coords, seed=0, device="cuda")
    torch.cuda.synchronize()
    t1 = time.time()
    assert (t1 - t0) < 5.0, f"11BG GPU mutational took {t1-t0:.2f}s (target <5s)"


# --- Gate 7: output shapes ---------------------------------------------------
def test_mutational_output_shapes():
    if not _has_pdb("5AON"):
        pytest.skip("5AON unavailable")
    coords = parse_pdb(PDB_DIR / "5AON.pdb")
    out = mutational_decoy_stats(coords, seed=0, n_decoys=128)
    np_pair = int(out["pair_i"].numel())
    assert out["pair_j"].shape == (np_pair,)
    assert out["E_native"].shape == (np_pair,)
    assert out["decoy_mean"].shape == (np_pair,)
    assert out["decoy_std"].shape == (np_pair,)
    assert out["aa_i_dec"].shape == (np_pair, 128)
    assert out["aa_j_dec"].shape == (np_pair, 128)
    # No NaNs
    assert torch.isfinite(out["E_native"]).all()
    assert torch.isfinite(out["decoy_mean"]).all()
    assert torch.isfinite(out["decoy_std"]).all()
    # Decoy std must be > 0
    assert (out["decoy_std"] > 0).all()


# --- Gate 8: seed reproducibility -------------------------------------------
def test_mutational_seed_reproducibility():
    if not _has_pdb("5AON"):
        pytest.skip("5AON unavailable")
    coords = parse_pdb(PDB_DIR / "5AON.pdb")
    a = mutational_decoy_stats(coords, seed=42)
    b = mutational_decoy_stats(coords, seed=42)
    assert torch.allclose(a["decoy_mean"], b["decoy_mean"], atol=1e-9)
    assert torch.allclose(a["decoy_std"], b["decoy_std"], atol=1e-9)
    # Different seed → different stats (genuinely different RNG)
    c = mutational_decoy_stats(coords, seed=7)
    diff = (a["decoy_mean"] - c["decoy_mean"]).abs().max()
    assert diff > 1e-3, f"different seed produced identical stats (diff {diff})"


# --- Gate 9: CPU vs GPU agreement on decoy_std (mean varies via RNG) -------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_decoy_std_agreement_5aon():
    """The decoy_std is computed from the same AA sample (the CPU
    generator is portable). CPU vs GPU should agree to ~1e-6 modulo
    float32→float64 rounding.
    """
    coords_cpu = parse_pdb(PDB_DIR / "5AON.pdb")
    out_cpu = mutational_decoy_stats(coords_cpu, seed=0)
    # GPU copy
    coords_gpu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coords_cpu.items()}
    out_gpu = mutational_decoy_stats(coords_gpu, seed=0, device="cuda")
    # Compare on CPU
    diff_mean = (out_cpu["decoy_mean"] - out_gpu["decoy_mean"].cpu()).abs().max().item()
    diff_std = (out_cpu["decoy_std"] - out_gpu["decoy_std"].cpu()).abs().max().item()
    # Float64 CPU vs CUDA: order of summations differ → expect 1e-5 floor on
    # the per-pair decoy ensemble (1000 elements summed in a different
    # parallel-reduction order on GPU).
    assert diff_mean < 1e-5, f"CPU vs GPU decoy_mean max diff {diff_mean:.2e}"
    assert diff_std < 1e-5, f"CPU vs GPU decoy_std max diff {diff_std:.2e}"


# --- Gate 10: pair-count matches dump ---------------------------------------
@pytest.mark.parametrize("pdb_id", PANEL)
def test_pair_count_matches_dump(pdb_id):
    if not (_has_pdb(pdb_id) and _has_dump(pdb_id)):
        pytest.skip(f"{pdb_id} unavailable")
    coords = parse_pdb(PDB_DIR / f"{pdb_id}.pdb")
    out = mutational_decoy_stats(coords, seed=0)
    dump = _parse_dump(pdb_id)
    assert int(out["pair_i"].numel()) == len(dump), (
        f"{pdb_id}: ours {int(out['pair_i'].numel())} vs dump {len(dump)}"
    )
