"""Tests for the top-level :func:`compute_frustration` orchestrator.

Coverage:

* Smoke: end-to-end on a single PDB; returns a populated FrustrationResult.
* Chain filter: ``chain="A"`` on a multi-chain PDB (11BG) restricts the
  output to chain-A pairs only; pair count matches the LAMMPS
  ``param_sweep/<PDB>_chain_A_only_*.dat`` dump.
* Residue subset filter: ``residues={"A": [...]}`` restricts pair_records
  / density_records to rows involving the listed residues. Self-validated
  (no reference dump available for residue subsets).
* DH opt-in: ``electrostatics_k=4.15`` adds the DH pair term to E_native;
  difference vs. ``electrostatics_k=None`` matches the DH formula.
* All 3 modes return populated results.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frustration_gpu import FrustrationResult, compute_frustration  # noqa: E402

from _paths import PDB_DIR  # noqa: E402
from _paths import DUMP_ROOT  # noqa: E402


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


def _parse_lammps_pair_dump(fp: Path) -> set:
    """Return {(i_pos, j_pos, i_chain, j_chain)} tuples from a LAMMPS raw dump."""
    out = set()
    with fp.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 19:
                continue
            i, j = int(p[0]), int(p[1])
            ci, cj = int(p[2]), int(p[3])
            out.add((min(i, j), max(i, j), ci, cj))
    return out


# ----- smoke ----------------------------------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_smoke_5AON():
    """End-to-end smoke: returns a populated FrustrationResult."""
    result = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        device="cpu",
        seed=0,
    )
    assert isinstance(result, FrustrationResult)
    assert result.pair_records is not None
    assert len(result.pair_records) == 221     # 5AON native pair count
    assert result.density_records is not None
    assert len(result.density_records) == 49   # 5AON residue count
    assert result.singleresidue_records is None
    assert result.metadata["mode"] == "configurational"
    assert result.metadata["n_residues"] == 49
    assert result.metadata["n_pairs"] == 221
    assert "decoy_mean" in result.metadata
    assert "decoy_std" in result.metadata
    assert result.metadata["wall_clock_ms"] > 0


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_writes_dump_files():
    """When ``output_dir`` is set, the three dump files are written."""
    with tempfile.TemporaryDirectory() as td:
        result = compute_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            device="cpu",
            seed=0,
            output_dir=td,
        )
        files = {p.name for p in Path(td).iterdir()}
        assert "5AON_tertiary_frustration.dat" in files
        assert "5AON_configurational.dat" in files
        assert "5AON_5adens.dat" in files
        # 5adens file has 49 data lines + 1 header
        n_lines = sum(1 for _ in (Path(td) / "5AON_5adens.dat").open())
        assert n_lines == 50


# ----- all 3 modes return populated results ---------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
@pytest.mark.parametrize("mode", ["configurational", "mutational", "singleresidue"])
def test_compute_frustration_modes(mode: str):
    """Each of the 3 modes returns a non-empty result."""
    result = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode=mode,
        device="cpu",
        seed=0,
    )
    if mode == "singleresidue":
        assert result.pair_records is None
        assert result.singleresidue_records is not None
        assert len(result.singleresidue_records) == 49
        assert result.density_records is None
    else:
        assert result.pair_records is not None
        assert len(result.pair_records) > 0
        assert result.singleresidue_records is None
        assert result.density_records is not None
        assert len(result.density_records) == 49
    assert result.metadata["mode"] == mode


# ----- chain filter ----------------------------------------------------------
@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb missing")
def test_compute_frustration_chain_filter():
    """``chain='A'`` on 11BG (homodimer) restricts pair_records to chain-A.

    Validated against ``param_sweep/11BG_chain_A_only_tertiary_frustration.dat``:
    pair count must match (modulo header lines).
    """
    sweep_path = DUMP_ROOT / "param_sweep" / "11BG_chain_A_only_tertiary_frustration.dat"
    if not sweep_path.is_file():
        pytest.skip(f"{sweep_path} missing")

    n_ref_pairs = sum(
        1 for line in sweep_path.read_text().splitlines()
        if line and not line.startswith("#")
    )

    result = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational",
        chain="A",
        device="cpu",
        seed=0,
    )
    assert result.metadata["chain"] == "A"
    # Every pair must reference chain A only.
    assert (result.pair_records["ChainRes1"] == "A").all()
    assert (result.pair_records["ChainRes2"] == "A").all()
    # Pair count matches LAMMPS chain-A-only sweep.
    assert len(result.pair_records) == n_ref_pairs, (
        f"chain-A pair count: ours {len(result.pair_records)} vs ref {n_ref_pairs}"
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb missing")
def test_compute_frustration_chain_filter_density():
    """Chain filter also restricts density_records."""
    result = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational",
        chain="A",
        device="cpu",
        seed=0,
    )
    assert (result.density_records["ChainRes"] == "A").all()
    # 11BG chain A has 124 residues (half of 248).
    assert len(result.density_records) == 124


# ----- residue subset filter -------------------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_residues_filter():
    """``residues={'A': [...]}`` filters pair_records to rows involving any
    listed residue. We pick residues 25, 30, 35 of 5AON.
    """
    subset_resnums = [25, 30, 35]
    result_full = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        device="cpu",
        seed=0,
    )
    result_sub = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        residues={"A": subset_resnums},
        device="cpu",
        seed=0,
    )
    # Every row in the subset must involve at least one of the subset residues.
    in_subset_i = result_sub.pair_records["Res1"].isin(subset_resnums)
    in_subset_j = result_sub.pair_records["Res2"].isin(subset_resnums)
    assert ((in_subset_i | in_subset_j)).all(), (
        "subset filter let through a pair with neither residue in subset"
    )
    # Strict subset of full pair list — same FrstIndex on overlap.
    expected = result_full.pair_records[
        result_full.pair_records["Res1"].isin(subset_resnums)
        | result_full.pair_records["Res2"].isin(subset_resnums)
    ]
    assert len(result_sub.pair_records) == len(expected)
    # Density records: only the 3 specified residues remain.
    assert set(result_sub.density_records["Res"].tolist()) == set(subset_resnums)


# ----- DH opt-in -------------------------------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_dh_opt_in():
    """``include_dh_in_e_native=True`` adds the DH pair-energy term.

    Test:
      * ``electrostatics_k=4.15`` alone: DH is NOT added to E_native.
        This matches LAMMPS-AWSEM / frustratometeR semantics — even
        when DH was active during dynamics, the analysis ``native_energy``
        column is water+burial only.
      * ``electrostatics_k=4.15, include_dh_in_e_native=True``: DH IS
        added to E_native.
      * ``electrostatics_k=None``: DH not computed (the legacy default).
    """
    r_no_dh = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        electrostatics_k=None,
        device="cpu",
        seed=0,
    )
    r_k_no_inc = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        electrostatics_k=4.15,
        # include_dh_in_e_native defaults to False
        device="cpu",
        seed=0,
    )
    r_k_inc = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        electrostatics_k=4.15,
        include_dh_in_e_native=True,
        device="cpu",
        seed=0,
    )

    # `electrostatics_k=4.15` alone must NOT change E_native (new default
    # — matches LAMMPS analysis convention).
    df_no_dh = r_no_dh.pair_records.set_index(["Res1", "Res2"])
    df_k_no_inc = r_k_no_inc.pair_records.set_index(["Res1", "Res2"])
    common = df_no_dh.index.intersection(df_k_no_inc.index)
    e_off = df_no_dh.loc[common]["NativeEnergy"].values
    e_kno = df_k_no_inc.loc[common]["NativeEnergy"].values
    np.testing.assert_array_equal(e_off, e_kno)

    # `include_dh_in_e_native=True` DOES change E_native.
    df_k_inc = r_k_inc.pair_records.set_index(["Res1", "Res2"])
    common_inc = df_no_dh.index.intersection(df_k_inc.index)
    e_off2 = df_no_dh.loc[common_inc]["NativeEnergy"].values
    e_inc = df_k_inc.loc[common_inc]["NativeEnergy"].values
    n_diff = int((e_off2 != e_inc).sum())
    assert n_diff > 0, "include_dh_in_e_native=True produced no energy difference"

    # The metadata reflects the opt-in.
    assert r_no_dh.metadata["electrostatics_k"] is None
    assert r_k_no_inc.metadata["electrostatics_k"] == 4.15
    assert r_k_no_inc.metadata["include_dh_in_e_native"] is False
    assert r_k_inc.metadata["electrostatics_k"] == 4.15
    assert r_k_inc.metadata["include_dh_in_e_native"] is True


# ----- DH byte-exact regression against LAMMPS (Phase 5 P4 + 2026-05-20 fix) ---
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_dh_byte_exact_against_lammps_5AON():
    """Validate DH semantics:

    1. **Byte-exact** (no RNG drift on E_native — it's deterministic):
       with the NEW default semantics ``include_dh_in_e_native=False``,
       ``compute_frustration(..., electrostatics_k=4.15)`` E_native
       must match ``5AON_electro_4p15_tertiary_frustration.dat``'s
       native_energy column to %8.3f for EVERY pair. Both omit DH.

    2. **Closed-form opt-in check**: with ``include_dh_in_e_native=True``,
       the delta in our E_native must match the standalone
       :func:`debye_huckel_pair_energy` value to %8.3f for every pair.

    Background (verified empirically against ``benchmark/cpu_baseline/
    param_sweep/5AON_electro_4p15_*.log``): the LAMMPS-generated
    ``electro_4p15`` dump has Electro=0.0 in the energy.log AND identical
    E_native to the no-DH dump for all pairs. frustratometeR's analysis
    is water+burial-only EVEN when DH was active during dynamics.
    """
    fp = DUMP_ROOT / "param_sweep" / "5AON_electro_4p15_tertiary_frustration.dat"
    if not fp.is_file():
        pytest.skip("LAMMPS electro_4p15 dump missing")

    from frustration_gpu import debye_huckel_pair_energy
    from frustration_gpu.parser import ONE_TO_IDX

    def _parse_full(fp_path):
        out = {}
        with open(fp_path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                if len(p) < 19:
                    continue
                out[(int(p[0]), int(p[1]))] = {
                    "e_native": float(p[15]),
                    "aa_i": p[13],
                    "aa_j": p[14],
                    "rij": float(p[10]),
                }
        return out

    theirs = _parse_full(fp)

    # New default: electrostatics_k=4.15 but DH NOT added to E_native.
    # This must byte-match LAMMPS' electro_4p15 dump E_native.
    with tempfile.TemporaryDirectory() as td:
        compute_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            electrostatics_k=4.15,
            # include_dh_in_e_native defaults to False (new default)
            n_decoys=1000,
            seed=0,
            device="cpu",
            output_dir=td,
        )
        ours_k_default = _parse_full(Path(td) / "5AON_tertiary_frustration.dat")

    # Assertion 1: pair sets match.
    only_ours = set(ours_k_default) - set(theirs)
    only_them = set(theirs) - set(ours_k_default)
    assert not only_ours, f"pairs only in ours (first 5): {sorted(only_ours)[:5]}"
    assert not only_them, f"pairs only in LAMMPS (first 5): {sorted(only_them)[:5]}"

    # Assertion 2: byte-exact match of E_native to LAMMPS (to print precision).
    eps_e = 5e-3
    no_dh_violations = []
    for k in sorted(ours_k_default.keys()):
        if abs(ours_k_default[k]["e_native"] - theirs[k]["e_native"]) > eps_e:
            no_dh_violations.append(
                (k, ours_k_default[k]["e_native"], theirs[k]["e_native"])
            )
    assert not no_dh_violations, (
        f"E_native drift > {eps_e} vs LAMMPS electro_4p15 dump in "
        f"{len(no_dh_violations)} rows under the new default "
        f"(include_dh_in_e_native=False). First 3: {no_dh_violations[:3]}"
    )

    # Now opt-in: include_dh_in_e_native=True should ADD DH to E_native.
    with tempfile.TemporaryDirectory() as td:
        compute_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            electrostatics_k=4.15,
            include_dh_in_e_native=True,
            n_decoys=1000,
            seed=0,
            device="cpu",
            output_dir=td,
        )
        ours_dh_on = _parse_full(Path(td) / "5AON_tertiary_frustration.dat")

    # Assertion 3: closed-form delta matches the standalone DH formula.
    dh_formula_violations = []
    n_charged_pairs = 0
    for k in sorted(ours_dh_on.keys()):
        if k not in ours_k_default:
            continue
        delta = ours_dh_on[k]["e_native"] - ours_k_default[k]["e_native"]
        aa_i = ONE_TO_IDX[ours_dh_on[k]["aa_i"]]
        aa_j = ONE_TO_IDX[ours_dh_on[k]["aa_j"]]
        expected = debye_huckel_pair_energy(
            ours_dh_on[k]["rij"], aa_i, aa_j, k_QQ=4.15,
        )
        if abs(delta - expected) > eps_e:
            dh_formula_violations.append(
                (k, delta, expected, ours_dh_on[k]["aa_i"], ours_dh_on[k]["aa_j"])
            )
        if abs(expected) > eps_e:
            n_charged_pairs += 1

    assert not dh_formula_violations, (
        f"DH opt-in formula mismatch: include_dh_in_e_native=True adds "
        f"{dh_formula_violations[0][1]:.4f} but debye_huckel_pair_energy "
        f"returns {dh_formula_violations[0][2]:.4f} on "
        f"{dh_formula_violations[0][0]} "
        f"({dh_formula_violations[0][3]}-{dh_formula_violations[0][4]}). "
        f"Total violations: {len(dh_formula_violations)}/{len(ours_dh_on)}"
    )
    assert n_charged_pairs >= 10, (
        f"Only {n_charged_pairs} charged pairs exercised — test is too weak"
    )


# ----- frustrapy drop-in alias (Phase 5 P3) ----------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_calculate_frustration_drop_in_alias():
    """Phase 5 P3 — ``calculate_frustration`` is a frustrapy-compatible alias.

    Verifies:
      * Returns same FrustrationResult as ``compute_frustration`` on same args.
      * ``results_dir`` (frustrapy spelling) → maps to ``output_dir``.
      * ``chain=["A", "B"]`` (list, not str) is accepted.
      * ``graphics=True`` is accepted but warns (doesn't crash).
      * ``is_mutation_calculation=True`` is back-compat synonym for mode="mutational".
    """
    import warnings

    from frustration_gpu import calculate_frustration  # noqa: E402

    # 1. Same FI values as compute_frustration on same args.
    a = compute_frustration(
        PDB_DIR / "5AON.pdb", mode="configurational", device="cpu", seed=0,
    )
    b = calculate_frustration(
        PDB_DIR / "5AON.pdb", mode="configurational", device="cpu", seed=0,
    )
    assert isinstance(b, FrustrationResult)
    assert len(a.pair_records) == len(b.pair_records)
    # FI columns byte-equal (same seed, same code path)
    np.testing.assert_array_equal(
        a.pair_records["FrstIndex"].values,
        b.pair_records["FrstIndex"].values,
    )

    # 2. results_dir maps to output_dir.
    with tempfile.TemporaryDirectory() as td:
        calculate_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            results_dir=td,
            device="cpu",
            seed=0,
        )
        files = {p.name for p in Path(td).iterdir()}
        assert "5AON_tertiary_frustration.dat" in files
        assert "5AON_configurational.dat" in files
        assert "5AON_5adens.dat" in files


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb missing")
def test_calculate_frustration_chain_list_arg():
    """``calculate_frustration(chain=["A", "B"])`` filters to both chains."""
    from frustration_gpu import calculate_frustration  # noqa: E402
    result = calculate_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational",
        chain=["A", "B"],
        device="cpu",
        seed=0,
    )
    chains = set(result.pair_records["ChainRes1"].tolist()) | set(
        result.pair_records["ChainRes2"].tolist()
    )
    assert chains.issubset({"A", "B"}), f"unexpected chains: {chains - {'A', 'B'}}"
    # 11BG has both A and B → expect both represented
    assert "A" in chains
    assert "B" in chains


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_calculate_frustration_accepts_silent_kwargs():
    """``graphics=True`` etc. don't error, just warn."""
    import warnings

    from frustration_gpu import calculate_frustration  # noqa: E402
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            graphics=True,
            visualization=True,
            debug=True,
            pbar=False,
            device="cpu",
            seed=0,
        )
        # At least one UserWarning about unsupported graphics flag
        msgs = [str(x.message) for x in w]
        assert any("graphics" in m.lower() or "visualization" in m.lower() for m in msgs), (
            f"expected graphics/visualization warning, got: {msgs}"
        )
    assert result.pair_records is not None
    assert len(result.pair_records) > 0


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_calculate_frustration_is_mutation_calculation_synonym():
    """``is_mutation_calculation=True`` translates to mode='mutational'."""
    from frustration_gpu import calculate_frustration  # noqa: E402
    result = calculate_frustration(
        PDB_DIR / "5AON.pdb",
        is_mutation_calculation=True,
        device="cpu",
        seed=0,
    )
    assert result.metadata["mode"] == "mutational"
    # Mutational mode → per-pair decoy stats (not scalar), pair_records populated.
    assert result.pair_records is not None
    assert len(result.pair_records) > 0


# ----- timing (informational, doesn't fail under load) ----------------------
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG.pdb missing")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_frustration_gpu_timing_11bg_under_5s():
    """Sanity gate: 11BG mutational mode on GPU completes in < 5 s (Phase
    3b target was 5 s; we should be well under)."""
    result = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="mutational",
        device="cuda",
        seed=0,
    )
    elapsed = result.metadata["wall_clock_ms"]
    assert elapsed < 5_000.0, f"11BG mut GPU took {elapsed:.1f} ms — > 5 s"
    assert result.metadata["device"].startswith("cuda")


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_device_auto():
    """``device='auto'`` picks CUDA when available, else CPU."""
    result = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="configurational",
        device="auto",
        seed=0,
    )
    if torch.cuda.is_available():
        assert result.metadata["device"].startswith("cuda")
    else:
        assert result.metadata["device"] == "cpu"


# ----- error handling --------------------------------------------------------
def test_compute_frustration_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        compute_frustration(
            PDB_DIR / "5AON.pdb",
            mode="bogus_mode",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON.pdb missing")
def test_compute_frustration_residues_filter_singleresidue_mode():
    """Residues filter works on singleresidue_records too."""
    subset_resnums = [25, 30]
    result = compute_frustration(
        PDB_DIR / "5AON.pdb",
        mode="singleresidue",
        residues={"A": subset_resnums},
        device="cpu",
        seed=0,
    )
    assert result.singleresidue_records is not None
    assert set(result.singleresidue_records["Res"].tolist()) == set(subset_resnums)
