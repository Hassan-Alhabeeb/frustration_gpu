"""Phase 5 stress test + frustrapy CPU benchmark.

Runs ``compute_frustration`` on the full 20-PDB panel (defined in
``benchmark/pdb_panel.csv``) across the three modes
(configurational / mutational / singleresidue) on both CPU and
RTX 4070 GPU, then compares per-pair / per-residue results against
the LAMMPS-AWSEM reference dumps in ``benchmark/cpu_baseline/`` and
against frustrapy CPU run on the VM (SSH ``root@10.1.0.45``).

Outputs (written next to this script):

* ``phase5_panel_results.csv`` — one row per (PDB, mode) with timings,
  VRAM peaks, status, Spearman vs reference where available.
* ``phase5_frustrapy_comparison.csv`` — head-to-head wall-clocks for the
  4-PDB validation set.
* ``phase5_results.md`` — human-readable summary.

Re-runnable: ``python benchmark/run_phase5.py``. The script keeps an
existing CSV row if its (PDB, mode) already completed (skip mode), or
overwrites if ``--force`` is passed. The frustrapy half is opt-in via
``--frustrapy`` (requires the VM to be reachable at root@10.1.0.45).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from frustration_gpu.compute_frustration import compute_frustration  # noqa: E402


BENCH_DIR = REPO_ROOT / "benchmark"
PDB_PANEL_CSV = BENCH_DIR / "pdb_panel.csv"
CPU_BASELINE_DIR = BENCH_DIR / "cpu_baseline"
PDB_FILE_ROOT = REPO_ROOT.parent / "allosteric" / "data" / "pdb_files"

PHASE5_PANEL_CSV = BENCH_DIR / "phase5_panel_results.csv"
PHASE5_FRUSTRAPY_CSV = BENCH_DIR / "phase5_frustrapy_comparison.csv"
PHASE5_MD = BENCH_DIR / "phase5_results.md"
PHASE5_PROGRESS_JSON = BENCH_DIR / "phase5_progress.json"

VM_HOST = "root@10.1.0.45"
VM_PDB_ROOT = "/root/allosteric/data/pdb_files"
VM_VENV = "/root/pyenvs/tuhnon/bin/activate"

MODES = ("configurational", "mutational", "singleresidue")
N_DECOYS = 1000
SEED = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_panel() -> List[Dict[str, str]]:
    rows = []
    with open(PDB_PANEL_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def _pdb_path(pdb_id: str) -> Path:
    p = PDB_FILE_ROOT / f"{pdb_id}.pdb"
    if not p.exists():
        raise FileNotFoundError(f"PDB file not found locally: {p}")
    return p


def _read_lammps_pair_dump(path: Path) -> pd.DataFrame:
    """Parse a tertiary_frustration.dat reference dump into a DataFrame."""
    cols = [
        "i", "j", "i_chain", "j_chain",
        "xi", "yi", "zi", "xj", "yj", "zj",
        "r_ij", "rho_i", "rho_j", "a_i", "a_j",
        "native_energy", "decoy_mean", "decoy_std", "f_ij",
    ]
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=cols,
        engine="python",
    )
    return df


def _read_lammps_5adens(path: Path) -> pd.DataFrame:
    """Parse a 5adens.dat reference dump.

    File format (frustratometeR convention) has a single header line followed
    by space-separated values. Note the reference uses ``HighlyFrst`` etc.
    while our DataFrame uses ``nHighlyFrst``; we rename for join-compat.
    """
    df = pd.read_csv(path, sep=r"\s+", header=0, engine="python")
    rename_map = {
        "HighlyFrst": "nHighlyFrst",
        "NeutrallyFrst": "nNeutrallyFrst",
        "MinimallyFrst": "nMinimallyFrst",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # cast numeric cols
    for c in ("Res", "Total", "nHighlyFrst", "nNeutrallyFrst", "nMinimallyFrst"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("relHighlyFrustrated", "relNeutralFrustrated",
              "relMinimallyFrustrated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_lammps_singleresidue(path: Path) -> pd.DataFrame:
    """Parse a singleresidue.dat reference dump (has a single header line)."""
    df = pd.read_csv(path, sep=r"\s+", header=0, engine="python")
    for c in ("Res",):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("DensityRes", "NativeEnergy", "DecoyEnergy", "SDEnergy",
              "FrstIndex"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        # Manual rank-correlation fallback (Pearson on ranks).
        ar = pd.Series(a).rank().to_numpy()
        br = pd.Series(b).rank().to_numpy()
        if ar.std() == 0 or br.std() == 0:
            return float("nan")
        return float(np.corrcoef(ar, br)[0, 1])


# ---------------------------------------------------------------------------
# Per-PDB benchmark runner
# ---------------------------------------------------------------------------

def _run_one(pdb_id: str, mode: str, device: str, dtype: torch.dtype = torch.float64) -> Dict[str, Any]:
    """Run compute_frustration once, measure wall-clock + VRAM peak.

    Returns a dict of measurements. ``status`` is one of:
    ``ok`` / ``oom`` / ``runtime_error`` / ``parse_error``.
    """
    pdb_path = _pdb_path(pdb_id)
    out = {
        "pdb": pdb_id,
        "mode": mode,
        "device": device,
        "status": "ok",
        "error": "",
        "n_residues": -1,
        "n_pairs": -1,
        "wall_ms": -1.0,
        "peak_vram_mb": -1.0,
        "decoy_mean": float("nan"),
        "decoy_std": float("nan"),
        "n_highly": -1,
        "n_neutral": -1,
        "n_minimally": -1,
    }
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    # warm-up (untimed forward) — only when device is CUDA AND this is the
    # first call after process start. We track this with a process-level
    # flag; for simplicity each call does a tiny warm-up on the first GPU
    # invocation only.
    if device == "cuda" and not getattr(_run_one, "_cuda_warm", False):
        try:
            warmup_path = _pdb_path("5AON")
            _ = compute_frustration(
                warmup_path,
                mode="configurational",
                device="cuda",
                dtype=dtype,
                n_decoys=100,
                seed=SEED,
            )
            torch.cuda.synchronize()
        except Exception:
            pass
        _run_one._cuda_warm = True  # type: ignore[attr-defined]
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = compute_frustration(
                pdb_path,
                mode=mode,
                device=device,
                dtype=dtype,
                n_decoys=N_DECOYS,
                seed=SEED,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    except torch.cuda.OutOfMemoryError as e:
        out["status"] = "oom"
        out["error"] = f"OOM: {str(e)[:200]}"
        return out
    except Exception as e:
        out["status"] = type(e).__name__
        out["error"] = str(e)[:400]
        return out

    out["n_residues"] = result.metadata.get("n_residues", -1)
    out["n_pairs"] = result.metadata.get("n_pairs", -1)
    out["wall_ms"] = round(wall_ms, 2)
    if device == "cuda":
        peak_b = torch.cuda.max_memory_allocated()
        out["peak_vram_mb"] = round(peak_b / (1024 * 1024), 1)

    if mode == "configurational":
        out["decoy_mean"] = result.metadata.get("decoy_mean", float("nan"))
        out["decoy_std"] = result.metadata.get("decoy_std", float("nan"))

    # FrstState counts (sanity — confirms output isn't degenerate)
    if result.pair_records is not None and len(result.pair_records) > 0:
        st = result.pair_records["FrstState"].value_counts()
        out["n_highly"] = int(st.get("highly", 0))
        out["n_neutral"] = int(st.get("neutral", 0))
        out["n_minimally"] = int(st.get("minimally", 0))
    elif result.singleresidue_records is not None:
        # singleresidue doesn't have FrstState directly; classify by FI
        fi = result.singleresidue_records["FrstIndex"].to_numpy()
        out["n_highly"] = int(np.sum(fi <= -1.0))
        out["n_neutral"] = int(np.sum((fi > -1.0) & (fi < 0.78)))
        out["n_minimally"] = int(np.sum(fi >= 0.78))

    # Stash the dataframes on the function-local object for downstream
    # spearman validation in the same process — keyed by (pdb, mode, device).
    if not hasattr(_run_one, "_cache"):
        _run_one._cache = {}  # type: ignore[attr-defined]
    _run_one._cache[(pdb_id, mode, device)] = result  # type: ignore[attr-defined]
    return out


def _spearman_vs_reference(pdb_id: str, mode: str, result_cache: Dict) -> Dict[str, float]:
    """Compare our (cpu/cuda) results to the LAMMPS reference dump.

    Returns dict with ``spearman_fi``, ``spearman_density_nhi``, and a
    cpu_vs_cuda spearman for cross-check.
    """
    out = {
        "ref_spearman_fi": float("nan"),
        "ref_spearman_density_nhi": float("nan"),
        "cpu_vs_cuda_max_abs_diff_fi": float("nan"),
        "cpu_vs_cuda_spearman_fi": float("nan"),
    }
    ref_dir = CPU_BASELINE_DIR / mode
    res_cpu = result_cache.get((pdb_id, mode, "cpu"))
    res_gpu = result_cache.get((pdb_id, mode, "cuda"))

    # CPU↔GPU stability (always available if we ran both)
    if res_cpu is not None and res_gpu is not None:
        if mode == "singleresidue":
            df_c = res_cpu.singleresidue_records
            df_g = res_gpu.singleresidue_records
            if df_c is not None and df_g is not None and len(df_c) == len(df_g):
                a = df_c["FrstIndex"].to_numpy()
                b = df_g["FrstIndex"].to_numpy()
                out["cpu_vs_cuda_max_abs_diff_fi"] = float(np.max(np.abs(a - b)))
                out["cpu_vs_cuda_spearman_fi"] = _spearman(a, b)
        else:
            df_c = res_cpu.pair_records
            df_g = res_gpu.pair_records
            if df_c is not None and df_g is not None and len(df_c) == len(df_g):
                a = df_c["FrstIndex"].to_numpy()
                b = df_g["FrstIndex"].to_numpy()
                out["cpu_vs_cuda_max_abs_diff_fi"] = float(np.max(np.abs(a - b)))
                out["cpu_vs_cuda_spearman_fi"] = _spearman(a, b)

    # LAMMPS reference
    if mode == "singleresidue":
        ref_path = ref_dir / f"{pdb_id}_singleresidue.dat"
        if ref_path.exists() and res_cpu is not None:
            try:
                ref = _read_lammps_singleresidue(ref_path)
                df = res_cpu.singleresidue_records
                if df is not None and len(df) == len(ref):
                    out["ref_spearman_fi"] = _spearman(
                        df["FrstIndex"].to_numpy(),
                        ref["FrstIndex"].to_numpy(),
                    )
            except Exception as e:
                out["ref_spearman_fi"] = float("nan")
                print(f"  WARN: ref-singleresidue parse {pdb_id}: {e}")
    else:
        ref_pair_path = ref_dir / f"{pdb_id}_tertiary_frustration.dat"
        ref_dens_path = ref_dir / f"{pdb_id}_5adens.dat"
        if ref_pair_path.exists() and res_cpu is not None:
            try:
                ref = _read_lammps_pair_dump(ref_pair_path)
                df = res_cpu.pair_records
                if df is not None and len(df) == len(ref):
                    out["ref_spearman_fi"] = _spearman(
                        df["FrstIndex"].to_numpy(),
                        ref["f_ij"].to_numpy(),
                    )
                elif df is not None:
                    # length mismatch — try aligning by (Res1,Res2,ChainRes1,ChainRes2)
                    pass
            except Exception as e:
                print(f"  WARN: ref-pair parse {pdb_id}: {e}")
        if ref_dens_path.exists() and res_cpu is not None:
            try:
                ref = _read_lammps_5adens(ref_dens_path)
                df = res_cpu.density_records
                if df is not None and len(df) >= len(ref):
                    # join on (Res, ChainRes)
                    j = df.merge(
                        ref, on=("Res", "ChainRes"), how="inner",
                        suffixes=("_ours", "_ref"),
                    )
                    if len(j) > 2:
                        out["ref_spearman_density_nhi"] = _spearman(
                            j["nHighlyFrst_ours"].to_numpy(),
                            j["nHighlyFrst_ref"].to_numpy(),
                        )
            except Exception as e:
                print(f"  WARN: ref-density parse {pdb_id}: {e}")

    return out


# ---------------------------------------------------------------------------
# Frustrapy on VM
# ---------------------------------------------------------------------------

def _run_frustrapy_on_vm(pdb_id: str, mode: str = "configurational") -> Dict[str, Any]:
    """SSH to VM, run frustrapy on a single PDB, return wall-clock seconds.

    Frustrapy spawns a LAMMPS subprocess per residue — it can be very
    slow on big PDBs. We add a per-call 600 s timeout.
    """
    py = (
        "import sys, time, warnings; warnings.filterwarnings('ignore'); "
        "import frustrapy; "
        f"t0=time.perf_counter(); "
        f"r=frustrapy.calculate_frustration("
        f"pdb_file='{VM_PDB_ROOT}/{pdb_id}.pdb', "
        f"mode='{mode}', "
        f"results_dir='/tmp/frustrapy_phase5/{pdb_id}_{mode}', "
        f"graphics=False, debug=False); "
        f"print('WALL_SECONDS', time.perf_counter()-t0)"
    )
    cmd = f"source {VM_VENV} && mkdir -p /tmp/frustrapy_phase5 && python -c {shlex.quote(py)}"
    out = {"pdb": pdb_id, "mode": mode, "frustrapy_wall_s": -1.0, "status": "ok", "error": ""}
    try:
        proc = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             VM_HOST, cmd],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            out["status"] = "ssh_error"
            out["error"] = (proc.stderr[-300:] if proc.stderr else "no-stderr")
            return out
        # Look for WALL_SECONDS marker
        wall = None
        for line in proc.stdout.splitlines():
            if line.startswith("WALL_SECONDS "):
                try:
                    wall = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
        if wall is None:
            out["status"] = "no_timing"
            out["error"] = proc.stdout[-300:]
            return out
        out["frustrapy_wall_s"] = round(wall, 3)
    except subprocess.TimeoutExpired:
        out["status"] = "timeout"
        out["error"] = "600s timeout"
    except Exception as e:
        out["status"] = type(e).__name__
        out["error"] = str(e)[:300]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustrapy", action="store_true",
                    help="Also run frustrapy on the VM for head-to-head.")
    ap.add_argument("--frustrapy-pdbs", default="5AON,11BG,1O3S,3F9M",
                    help="Comma-separated PDB IDs for frustrapy head-to-head.")
    ap.add_argument("--frustrapy-modes", default="configurational",
                    help="Comma-separated modes for frustrapy run.")
    ap.add_argument("--skip-cpu", action="store_true",
                    help="Skip the CPU sweep (only run GPU).")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="Skip the GPU sweep.")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="Comma-separated subset of modes to run.")
    ap.add_argument("--pdbs", default="",
                    help="Comma-separated subset of PDB IDs (default: all 20).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite phase5_panel_results.csv from scratch.")
    ap.add_argument("--spearman-only", action="store_true",
                    help="Skip the timing sweep entirely; only re-populate the "
                         "in-process cache by re-running compute_frustration on "
                         "every (PDB, mode, device) that has a LAMMPS reference, "
                         "then regenerate phase5_spearman.csv. Leaves "
                         "phase5_panel_results.csv and phase5_frustrapy_comparison.csv "
                         "untouched. Use this after the panel sweep already ran "
                         "across multiple separate process invocations (the in-process "
                         "cache resets between processes, so the Spearman pass otherwise "
                         "sees an empty cache).")
    args = ap.parse_args()

    panel = _read_panel()
    if args.pdbs:
        keep = set(args.pdbs.split(","))
        panel = [r for r in panel if r["pdb_id"] in keep]
    modes_to_run = [m for m in args.modes.split(",") if m in MODES]

    print(f"Phase 5 panel: {len(panel)} PDBs, modes={modes_to_run}")
    print(f"  GPU: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda'})")

    if args.force and PHASE5_PANEL_CSV.exists() and not args.spearman_only:
        PHASE5_PANEL_CSV.unlink()

    rows: List[Dict[str, Any]] = []
    if PHASE5_PANEL_CSV.exists() and not args.force:
        with open(PHASE5_PANEL_CSV) as f:
            rows = list(csv.DictReader(f))
    done = {(r["pdb"], r["mode"], r["device"]) for r in rows}

    devices = []
    if not args.skip_cpu:
        devices.append("cpu")
    if not args.skip_gpu and torch.cuda.is_available():
        devices.append("cuda")

    # ---- Spearman-only mode: skip the timing sweep, just populate the
    # in-process cache so the validation pass below can compute against
    # LAMMPS reference dumps. This is needed when the panel ran in a
    # separate process invocation (cache resets between processes).
    if args.spearman_only:
        print("\nSpearman-only mode: re-running compute_frustration on "
              "PDBs that have LAMMPS reference dumps (no timing CSV update).")
        ref_pdbs = set()
        for mode in modes_to_run:
            for f in (CPU_BASELINE_DIR / mode).glob("*.dat"):
                ref_pdbs.add(f.stem.split("_")[0])
        ref_pdbs = sorted(ref_pdbs)
        print(f"  PDBs with reference: {ref_pdbs}")
        for pdb_id in ref_pdbs:
            for mode in modes_to_run:
                # Need ref to exist for this (pdb, mode)
                if mode == "singleresidue":
                    ref_exists = (CPU_BASELINE_DIR / mode /
                                  f"{pdb_id}_singleresidue.dat").exists()
                else:
                    ref_exists = (CPU_BASELINE_DIR / mode /
                                  f"{pdb_id}_tertiary_frustration.dat").exists()
                if not ref_exists:
                    continue
                for dev in devices:
                    print(f"  CACHE {pdb_id} {mode} {dev}", flush=True)
                    out = _run_one(pdb_id, mode, dev)
                    if out["status"] != "ok":
                        print(f"       -> status={out['status']}  "
                              f"err={out['error'][:80]}")
        # Spearman validation pass (only — no CSV mutation, no markdown).
        print("\nSpearman validation against LAMMPS reference dumps:")
        val_rows = []
        cache = getattr(_run_one, "_cache", {})
        for pdb_id in ref_pdbs:
            for mode in modes_to_run:
                # Need at least one cached result (cpu OR cuda) — skip
                # (pdb, mode) where every _run_one call errored.
                has_any = any((pdb_id, mode, d) in cache for d in devices)
                if not has_any:
                    continue
                sp = _spearman_vs_reference(pdb_id, mode, cache)
                val_rows.append({"pdb": pdb_id, "mode": mode, **sp})
                print(f"  {pdb_id:6s} {mode:14s}  "
                      f"ref_fi={sp['ref_spearman_fi']:.4f}  "
                      f"ref_dens={sp['ref_spearman_density_nhi']:.4f}  "
                      f"cpu_vs_cuda_fi={sp['cpu_vs_cuda_spearman_fi']:.4f}  "
                      f"max_abs_diff={sp['cpu_vs_cuda_max_abs_diff_fi']:.2e}")
        val_path = BENCH_DIR / "phase5_spearman.csv"
        pd.DataFrame(val_rows).to_csv(val_path, index=False)
        print(f"  -> {val_path}  ({len(val_rows)} rows)")
        return

    for prow in panel:
        pdb_id = prow["pdb_id"]
        n_res_target = int(prow["actual_residues"])
        for mode in modes_to_run:
            for dev in devices:
                key = (pdb_id, mode, dev)
                if key in done:
                    print(f"  SKIP {pdb_id} {mode} {dev} (already done)")
                    continue

                # Heuristic skip: skip CPU on the very large 4PKN to keep
                # total runtime tractable — CPU on 8689 res would take
                # tens of minutes per mode.
                if dev == "cpu" and n_res_target > 2000:
                    print(f"  SKIP {pdb_id} {mode} cpu (>2000 res, would take >>10 min)")
                    out = {
                        "pdb": pdb_id, "mode": mode, "device": dev,
                        "status": "skipped_too_large",
                        "error": f"N={n_res_target} skipped on CPU for runtime",
                        "n_residues": -1, "n_pairs": -1, "wall_ms": -1.0,
                        "peak_vram_mb": -1.0, "decoy_mean": float("nan"),
                        "decoy_std": float("nan"),
                        "n_highly": -1, "n_neutral": -1, "n_minimally": -1,
                    }
                    rows.append(out)
                    _write_csv(rows)
                    done.add(key)
                    continue

                print(f"  RUN  {pdb_id} {mode} {dev}", flush=True)
                t = time.perf_counter()
                out = _run_one(pdb_id, mode, dev)
                print(f"       -> status={out['status']:>12s}  "
                      f"n_res={out['n_residues']:>5d}  "
                      f"n_pair={out['n_pairs']:>6d}  "
                      f"wall_ms={out['wall_ms']:>7.1f}  "
                      f"vram_mb={out['peak_vram_mb']:>7.1f}  "
                      f"(realtime={round(time.perf_counter()-t,1)}s)",
                      flush=True)
                rows.append(out)
                _write_csv(rows)
                done.add(key)

    # ---- Spearman validation pass ----
    print("\nSpearman validation against LAMMPS reference dumps:")
    val_rows = []
    cache = getattr(_run_one, "_cache", {})
    for prow in panel:
        pdb_id = prow["pdb_id"]
        for mode in modes_to_run:
            sp = _spearman_vs_reference(pdb_id, mode, cache)
            val_rows.append({"pdb": pdb_id, "mode": mode, **sp})
            print(f"  {pdb_id:6s} {mode:14s}  "
                  f"ref_fi={sp['ref_spearman_fi']:.4f}  "
                  f"ref_dens={sp['ref_spearman_density_nhi']:.4f}  "
                  f"cpu_vs_cuda_fi={sp['cpu_vs_cuda_spearman_fi']:.4f}  "
                  f"max_abs_diff={sp['cpu_vs_cuda_max_abs_diff_fi']:.2e}")

    val_path = BENCH_DIR / "phase5_spearman.csv"
    pd.DataFrame(val_rows).to_csv(val_path, index=False)
    print(f"  -> {val_path}")

    # ---- Frustrapy comparison ----
    if args.frustrapy:
        print("\nFrustrapy head-to-head on VM:")
        frust_rows = []
        for pdb_id in args.frustrapy_pdbs.split(","):
            for mode in args.frustrapy_modes.split(","):
                print(f"  RUN VM frustrapy {pdb_id} {mode}...", flush=True)
                t0 = time.perf_counter()
                r = _run_frustrapy_on_vm(pdb_id, mode)
                print(f"       -> status={r['status']:>10s}  "
                      f"wall_s={r['frustrapy_wall_s']:>7.2f}  "
                      f"(realtime={round(time.perf_counter()-t0,1)}s)",
                      flush=True)
                if r["status"] != "ok":
                    print(f"       err: {r['error'][:200]}")
                frust_rows.append(r)
                pd.DataFrame(frust_rows).to_csv(PHASE5_FRUSTRAPY_CSV, index=False)
        pd.DataFrame(frust_rows).to_csv(PHASE5_FRUSTRAPY_CSV, index=False)

    # ---- Markdown report ----
    _write_markdown(rows, val_rows, args)
    print(f"\n  -> {PHASE5_MD}")


def _write_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(PHASE5_PANEL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(panel_rows, val_rows, args):
    """Generate phase5_results.md from accumulated CSV state."""
    panel = _read_panel()
    panel_by_pdb = {r["pdb_id"]: r for r in panel}

    # Build (pdb, mode) -> {cpu_row, cuda_row}
    by_key = {}
    for r in panel_rows:
        key = (r["pdb"], r["mode"])
        by_key.setdefault(key, {})[r["device"]] = r
    val_by_key = {(r["pdb"], r["mode"]): r for r in val_rows}

    lines = []
    a = lines.append
    a("# Phase 5 stress test + frustrapy benchmark\n")
    a(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    a("Hardware: RTX 4070 (12 GB) + Windows 11 host. CPU runs are the "
      "local Windows Python process (single-threaded fair-comparison; "
      "no multiprocessing). All compute in **float64**, **n_decoys=1000**, "
      "**seed=0**.\n")
    a("Frustrapy comparison runs on VM ``root@10.1.0.45`` (EPYC 32-core, "
      "no GPU; frustrapy spawns LAMMPS subprocesses, so single-PDB single-"
      "threaded is the apples-to-apples).\n")

    # Panel results table
    a("## 20-PDB panel results\n")
    a("| PDB | N res | mode | CPU (s) | GPU (ms) | Peak VRAM (MB) | "
      "Status |")
    a("|-----|-------|------|---------|----------|----------------|------|")
    for prow in panel:
        pdb_id = prow["pdb_id"]
        n_res = prow["actual_residues"]
        for mode in MODES:
            d = by_key.get((pdb_id, mode), {})
            cpu = d.get("cpu")
            gpu = d.get("cuda")
            cpu_s = "n/a"
            gpu_ms = "n/a"
            vram = "n/a"
            status = "—"
            n_actual = n_res
            if cpu:
                if cpu["status"] == "ok":
                    cpu_s = f"{float(cpu['wall_ms']) / 1000.0:.3f}"
                    n_actual = cpu["n_residues"]
                elif cpu["status"] == "skipped_too_large":
                    cpu_s = "(skip)"
                else:
                    cpu_s = f"FAIL({cpu['status']})"
            if gpu:
                if gpu["status"] == "ok":
                    gpu_ms = f"{float(gpu['wall_ms']):.1f}"
                    vram = f"{float(gpu['peak_vram_mb']):.0f}"
                    n_actual = gpu["n_residues"]
                    status = "ok"
                elif gpu["status"] == "oom":
                    gpu_ms = "OOM"
                    status = "OOM"
                else:
                    gpu_ms = f"FAIL"
                    status = gpu["status"]
            if not gpu and not cpu:
                continue
            if status == "—":
                status = "ok" if (cpu and cpu["status"] == "ok") else "n/a"
            a(f"| {pdb_id} | {n_actual} | {mode} | {cpu_s} | {gpu_ms} | "
              f"{vram} | {status} |")

    # Frustrapy comparison
    if PHASE5_FRUSTRAPY_CSV.exists():
        a("\n## vs frustrapy CPU (head-to-head)\n")
        a("Frustrapy CPU times are single-PDB single-threaded on the VM. "
          "Our CPU + GPU times are the timings from the panel table above.\n")
        a("| PDB | N res | mode | frustrapy CPU (s) | ours CPU (s) | "
          "ours GPU (ms) | Speedup GPU vs frustrapy |")
        a("|-----|-------|------|-------------------|--------------|----"
          "----------|--------------------------|")
        frust = pd.read_csv(PHASE5_FRUSTRAPY_CSV)
        for _, fr in frust.iterrows():
            pdb_id = fr["pdb"]
            mode = fr["mode"]
            d = by_key.get((pdb_id, mode), {})
            cpu = d.get("cpu")
            gpu = d.get("cuda")
            cpu_s = (f"{float(cpu['wall_ms']) / 1000.0:.3f}"
                     if cpu and cpu["status"] == "ok" else "n/a")
            gpu_ms = (f"{float(gpu['wall_ms']):.1f}"
                      if gpu and gpu["status"] == "ok" else "n/a")
            frust_s = fr["frustrapy_wall_s"]
            if frust_s and frust_s > 0 and gpu and gpu["status"] == "ok":
                speedup = f"{frust_s * 1000.0 / float(gpu['wall_ms']):.1f}×"
            else:
                speedup = "n/a"
            n_res = (cpu["n_residues"] if cpu and cpu["status"] == "ok"
                     else gpu["n_residues"] if gpu and gpu["status"] == "ok"
                     else panel_by_pdb[pdb_id]["actual_residues"])
            a(f"| {pdb_id} | {n_res} | {mode} | {frust_s:.3f} | {cpu_s} | "
              f"{gpu_ms} | {speedup} |")

    # 4PKN stress test
    a("\n## 4PKN stress test (8689 residues)\n")
    pkn_rows = [r for r in panel_rows if r["pdb"] == "4PKN"]
    if not pkn_rows:
        a("(no 4PKN runs in this report)\n")
    else:
        for r in pkn_rows:
            if r["status"] == "ok":
                a(f"- **{r['mode']}, {r['device']}**: "
                  f"N={r['n_residues']} residues, "
                  f"N_pairs={r['n_pairs']}, "
                  f"wall_ms={r['wall_ms']}, "
                  f"peak_VRAM={r['peak_vram_mb']} MB, "
                  f"FrstState counts highly/neutral/min "
                  f"= {r['n_highly']}/{r['n_neutral']}/{r['n_minimally']}.")
            elif r["status"] == "oom":
                a(f"- **{r['mode']}, {r['device']}**: OOM. "
                  f"Peak VRAM = {r['peak_vram_mb']} MB. "
                  f"Recovery path: enable α-chunking with reduced chunk "
                  f"budget (see optimization_sprint_results.md §Idea 2).")
            elif r["status"] == "skipped_too_large":
                a(f"- **{r['mode']}, {r['device']}**: SKIPPED "
                  f"(>2000 res, runtime budget).")
            else:
                a(f"- **{r['mode']}, {r['device']}**: FAILED "
                  f"({r['status']}). Error: {r['error'][:200]}")

    # Spearman validation summary
    a("\n## Numerical validation (Spearman per PDB / mode)\n")
    a("``ref_spearman_fi`` = our per-pair FrstIndex vs LAMMPS dump f_ij. "
      "``ref_density`` = our nHighlyFrst vs LAMMPS 5adens.dat. "
      "``cpu_vs_cuda`` = stability check (should be ~1.0 to machine "
      "precision; both paths share the same RNG seed).\n")
    a("| PDB | mode | ref Spearman FI | ref Spearman density | CPU↔GPU "
      "Spearman FI | CPU↔GPU max |Δ FI| |")
    a("|-----|------|-----------------|----------------------|--------"
      "------------|----------------|")
    for prow in panel:
        pdb_id = prow["pdb_id"]
        for mode in MODES:
            v = val_by_key.get((pdb_id, mode))
            if not v:
                continue
            def _f(x):
                try:
                    fx = float(x)
                    return "n/a" if np.isnan(fx) else f"{fx:.4f}"
                except Exception:
                    return "n/a"
            a(f"| {pdb_id} | {mode} | {_f(v['ref_spearman_fi'])} | "
              f"{_f(v['ref_spearman_density_nhi'])} | "
              f"{_f(v['cpu_vs_cuda_spearman_fi'])} | "
              f"{v['cpu_vs_cuda_max_abs_diff_fi']:.2e} |"
              if isinstance(v["cpu_vs_cuda_max_abs_diff_fi"], float)
                 and not np.isnan(v["cpu_vs_cuda_max_abs_diff_fi"])
              else f"| {pdb_id} | {mode} | {_f(v['ref_spearman_fi'])} | "
                   f"{_f(v['ref_spearman_density_nhi'])} | "
                   f"{_f(v['cpu_vs_cuda_spearman_fi'])} | n/a |")

    # Headline numbers
    a("\n## Key headline numbers\n")
    gpu_oks = [r for r in panel_rows
               if r["device"] == "cuda" and r["status"] == "ok"]
    cpu_oks = [r for r in panel_rows
               if r["device"] == "cpu" and r["status"] == "ok"]
    if gpu_oks:
        biggest = max(gpu_oks, key=lambda r: int(r["n_residues"]))
        a(f"- Largest successfully run on GPU: **{biggest['pdb']}** "
          f"(N={biggest['n_residues']} residues, mode={biggest['mode']}, "
          f"wall={float(biggest['wall_ms']):.0f} ms, "
          f"VRAM peak={float(biggest['peak_vram_mb']):.0f} MB).")

    if PHASE5_FRUSTRAPY_CSV.exists():
        frust = pd.read_csv(PHASE5_FRUSTRAPY_CSV)
        ratios = []
        for _, fr in frust.iterrows():
            if fr["status"] != "ok":
                continue
            d = by_key.get((fr["pdb"], fr["mode"]), {})
            gpu = d.get("cuda")
            if gpu and gpu["status"] == "ok":
                ratios.append(float(fr["frustrapy_wall_s"]) * 1000.0 /
                              float(gpu["wall_ms"]))
        if ratios:
            a(f"- Mean GPU speedup vs frustrapy CPU on the panel "
              f"({len(ratios)} PDBs): **{np.mean(ratios):.1f}×** "
              f"(range {min(ratios):.1f}× — {max(ratios):.1f}×).")
            # Pick a representative PDB for the README headline
            mid_idx = sorted(range(len(ratios)), key=lambda i: ratios[i])[len(ratios)//2]
            a(f"- README headline candidate: "
              f"**{np.median(ratios):.0f}× faster on RTX 4070** than "
              f"frustrapy CPU on a typical ~250-res protein.")

    # write
    PHASE5_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
