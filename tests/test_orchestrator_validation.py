"""Audit 2026-05-21 orchestrator-validation tests.

One test (or a tight cluster) per audit finding owned by the
``compute_frustration.py`` orchestrator agent.

Findings covered:

* #2 / #22  — ``n_decoys=0`` / ``n_decoys=1`` rejected with ``ValueError``
* #4        — ``metadata["v_dh"]`` populated when ``electrostatics_k`` is set
* #5        — header-only marker files written when ``n_pairs == 0``
* #7        — output_dir files reflect the ``residues=`` filter
* #8        — ``include_dh_in_e_native=True`` emits ``DeprecationWarning``
* #19       — metadata exposes filtered + ``_unfiltered`` pair/residue counts
* #20       — configurational zero-contact input returns schema-empty result
* #21       — mutational zero-pair branch returns schema-preserving empty DF
* #23       — non-floating ``dtype`` rejected with ``ValueError``
* #25       — chain partial-miss rejected with ``ValueError`` listing available
* #26       — residues= partial-miss emits ``UserWarning`` naming the gaps
* #35       — ``calculate_frustration`` graphics warns once across the process
* #59       — non-int ``precision`` rejected with ``ValueError``

The tests intentionally use the bundled 5AON PDB (so they run in CI on a
fresh clone) and very small ``n_decoys`` where the FI numerics don't
matter — the gate is on validation behaviour, not on physics accuracy.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import PDB_DIR  # noqa: E402

from frustration_gpu import compute_frustration  # noqa: E402
from frustration_gpu.compute_frustration import _CF_WARN_ONCE  # noqa: E402

PDB_5AON = PDB_DIR / "5AON.pdb"


def _has_5aon() -> bool:
    return PDB_5AON.is_file()


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


_ZERO_CONTACT_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.500   1.500   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.500   0.500   0.000  1.00  0.00           C
ATOM      6  N   ALA A   5     200.000   0.000   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   5     201.500   0.000   0.000  1.00  0.00           C
ATOM      8  C   ALA A   5     202.000   1.000   0.000  1.00  0.00           C
ATOM      9  O   ALA A   5     202.500   1.500   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   5     202.500   0.500   0.000  1.00  0.00           C
TER
END
"""


@pytest.fixture
def zero_contact_pdb(tmp_path):
    """Two residues 200 A apart — guarantees zero native contacts."""
    fp = tmp_path / "zero_contact.pdb"
    fp.write_text(_ZERO_CONTACT_PDB)
    return fp


# ---------------------------------------------------------------------------
# Input validation (findings #2/#22, #23, #25, #59)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
@pytest.mark.parametrize("bad", [0, 1, -5])
def test_n_decoys_below_two_raises(bad):
    """Finding #2 + #22: n_decoys < 2 must raise a clean ValueError.

    The FI is a z-score, so an ensemble of <2 is degenerate. Without this
    guard, n_decoys=0 silently produced NaN FrstIndex and n_decoys=1
    produced FrstIndex=0 for every pair via the degenerate-std clamp.
    """
    with pytest.raises(ValueError, match="n_decoys"):
        compute_frustration(PDB_5AON, mode="configurational", n_decoys=bad, device="cpu")


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_n_decoys_non_int_raises():
    with pytest.raises(ValueError, match="n_decoys must be int"):
        compute_frustration(
            PDB_5AON, mode="configurational", n_decoys=10.5, device="cpu",
        )


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
@pytest.mark.parametrize("bad", [torch.int32, torch.int64, torch.bool])
def test_dtype_non_floating_raises(bad):
    """Finding #23: int / bool dtypes must be rejected with a clear ValueError.

    Previously these passed validation and crashed inside parse_pdb with
    a raw RuntimeError about overflow when filling NaN into the coord
    tensors.
    """
    with pytest.raises(ValueError, match="floating torch.dtype"):
        compute_frustration(PDB_5AON, mode="configurational", dtype=bad, device="cpu")


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
@pytest.mark.parametrize("bad", ["3", None, 1.5])
def test_precision_non_int_raises(bad):
    """Finding #59: precision must be a non-negative int, not just any "ordered" value."""
    with pytest.raises(ValueError, match="precision"):
        compute_frustration(
            PDB_5AON, mode="configurational", precision=bad, device="cpu",
        )


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_chain_partial_miss_raises():
    """Finding #25: chain=['A', 'Z'] on a single-chain PDB raises with the missing chains called out."""
    with pytest.raises(ValueError) as exc:
        compute_frustration(
            PDB_5AON, mode="configurational", chain=["A", "Z"],
            n_decoys=10, device="cpu",
        )
    msg = str(exc.value)
    assert "Z" in msg, f"expected missing chain Z reported: {msg!r}"
    assert "available" in msg, f"expected available-chains hint: {msg!r}"


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_chain_bad_type_raises():
    """Finding #25: parser used to accept 'AB' as string membership — orchestrator must validate types."""
    with pytest.raises(ValueError):
        compute_frustration(
            PDB_5AON, mode="configurational", chain=42,  # type: ignore[arg-type]
            n_decoys=10, device="cpu",
        )


# ---------------------------------------------------------------------------
# DH semantics (findings #4 + #8)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_v_dh_metadata_populated_for_electrostatics_k():
    """Finding #4: metadata["v_dh"] is the scalar total native DH energy.

    The docstring promised this field when electrostatics_k was set; the
    old code never populated it. With include_dh_in_e_native=False (the
    default) we ALSO get a v_dh diagnostic without changing E_native.
    """
    r = compute_frustration(
        PDB_5AON, mode="configurational", electrostatics_k=4.15,
        n_decoys=10, seed=0, device="cpu",
    )
    assert "v_dh" in r.metadata, f"metadata keys: {sorted(r.metadata)}"
    v_dh = r.metadata["v_dh"]
    assert isinstance(v_dh, float)
    # 5AON has charged residues — the total can't be exactly zero.
    assert abs(v_dh) > 1e-6, f"v_dh too small to be meaningful: {v_dh!r}"

    # When electrostatics_k is not set, no v_dh field.
    r_off = compute_frustration(
        PDB_5AON, mode="configurational",
        n_decoys=10, seed=0, device="cpu",
    )
    assert "v_dh" not in r_off.metadata


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_include_dh_in_e_native_is_deprecated():
    """Finding #8: include_dh_in_e_native=True emits a DeprecationWarning.

    It remains functional in v0.2.0 (so existing callers don't break),
    but the warning steers users to electrostatics_k alone + metadata.
    """
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        compute_frustration(
            PDB_5AON, mode="configurational",
            electrostatics_k=4.15, include_dh_in_e_native=True,
            n_decoys=10, seed=0, device="cpu",
        )
    dep_msgs = [
        str(w.message) for w in ws if issubclass(w.category, DeprecationWarning)
    ]
    assert any(
        "include_dh_in_e_native" in m and "deprecated" in m.lower()
        for m in dep_msgs
    ), f"expected DeprecationWarning; got: {dep_msgs}"


# ---------------------------------------------------------------------------
# residues= filter consistency (findings #7, #19, #26)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_residues_filter_applies_to_output_files():
    """Finding #7: output_dir dumps must reflect the residues= filter.

    Pre-fix on 5AON with residues={"A":[25,30,35]} returned 27 pair rows
    while 5AON_configurational.dat contained 221 — wholly unfiltered.
    """
    subset = {"A": [25, 30, 35]}
    with tempfile.TemporaryDirectory() as td:
        r = compute_frustration(
            PDB_5AON, mode="configurational", residues=subset,
            n_decoys=10, seed=0, device="cpu", output_dir=td,
        )
        n_returned = len(r.pair_records)
        configurational_dat = Path(td) / "5AON_configurational.dat"
        with configurational_dat.open() as fh:
            n_data_rows = sum(
                1 for ln in fh
                if ln.strip() and not ln.startswith("Res1") and not ln.startswith("#")
            )
        assert n_data_rows == n_returned, (
            f"dump file rows {n_data_rows} != returned pair_records "
            f"{n_returned} — filter was applied to DataFrames but not to "
            f"emitted .dat (audit finding #7)."
        )
        # And 5adens.dat should be filtered too.
        adens_dat = Path(td) / "5AON_5adens.dat"
        with adens_dat.open() as fh:
            n_adens_rows = sum(
                1 for ln in fh
                if ln.strip() and not ln.startswith("Res ChainRes")
            )
        assert n_adens_rows == len(r.density_records)


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_metadata_reports_filtered_and_unfiltered_counts():
    """Finding #19: metadata exposes n_pairs/n_residues post-filter AND
    n_pairs_unfiltered/n_residues_unfiltered for the original counts.
    """
    subset = {"A": [25, 30, 35]}
    r = compute_frustration(
        PDB_5AON, mode="configurational", residues=subset,
        n_decoys=10, seed=0, device="cpu",
    )
    assert r.metadata["n_pairs"] == len(r.pair_records)
    assert r.metadata["n_residues"] == len(r.density_records)
    # The unfiltered fields preserve the original computation size.
    assert r.metadata["n_pairs_unfiltered"] == 221     # 5AON configurational
    assert r.metadata["n_residues_unfiltered"] == 49   # 5AON residue count


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_residues_partial_miss_warns():
    """Finding #26: residues={'A': [25, 9999]} must warn about 9999."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        compute_frustration(
            PDB_5AON, mode="singleresidue", residues={"A": [25, 9999]},
            n_decoys=10, seed=0, device="cpu",
        )
    user_msgs = [
        str(w.message) for w in ws if issubclass(w.category, UserWarning)
    ]
    assert any("9999" in m for m in user_msgs), (
        f"expected partial-miss UserWarning mentioning 9999; got {user_msgs}"
    )


# ---------------------------------------------------------------------------
# Empty / zero-pair handling (findings #5, #20, #21)
# ---------------------------------------------------------------------------


def _pair_columns():
    return [
        "Res1", "Res2", "ChainRes1", "ChainRes2",
        "DensityRes1", "DensityRes2", "AA1", "AA2", "r_ij",
        "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex",
        "Welltype", "FrstState",
    ]


def test_configurational_zero_contact_returns_empty_schema(zero_contact_pdb):
    """Finding #20: zero-contact configurational input must NOT crash.

    The old code propagated a RuntimeError from
    sample_configurational_decoys; mutational and singleresidue already
    returned clean empties on the same input.
    """
    r = compute_frustration(
        zero_contact_pdb, mode="configurational",
        n_decoys=10, seed=0, device="cpu",
    )
    assert r.pair_records is not None
    assert len(r.pair_records) == 0
    assert list(r.pair_records.columns) == _pair_columns()
    assert r.metadata["n_pairs"] == 0


def test_mutational_zero_pair_schema_preserved(zero_contact_pdb):
    """Finding #21: mutational zero-pair branch now matches configurational schema.

    Pre-fix returned a bare ``pd.DataFrame()`` with shape (0, 0) — code
    expecting standard pair columns would silently drop them.
    """
    r = compute_frustration(
        zero_contact_pdb, mode="mutational",
        n_decoys=10, seed=0, device="cpu",
    )
    assert r.pair_records is not None
    assert len(r.pair_records) == 0
    assert list(r.pair_records.columns) == _pair_columns()


def test_empty_pair_mode_writes_header_only_files(zero_contact_pdb, tmp_path):
    """Finding #5: zero-pair pair-mode runs still emit (header-only) files.

    Batch users want a deliberate marker so a downstream "file missing"
    check doesn't conflate "run failed" with "run produced no pairs."
    """
    out_dir = tmp_path / "out"
    compute_frustration(
        zero_contact_pdb, mode="configurational",
        n_decoys=10, seed=0, device="cpu", output_dir=out_dir,
    )
    stem = zero_contact_pdb.stem
    expected = {
        f"{stem}_tertiary_frustration.dat",
        f"{stem}_configurational.dat",
        f"{stem}_5adens.dat",
    }
    actual = {p.name for p in out_dir.iterdir()}
    assert expected.issubset(actual), (
        f"missing expected header-only files; got {actual}"
    )
    # Each file has a header line but no data rows.
    for fname in expected:
        with (out_dir / fname).open() as fh:
            data_rows = [
                ln for ln in fh
                if ln.strip()
                and not ln.startswith("#")
                and not ln.startswith("Res")
            ]
        assert data_rows == [], (
            f"{fname} should be header-only; got data rows {data_rows!r}"
        )


# ---------------------------------------------------------------------------
# calculate_frustration warning behaviour (finding #35)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_calculate_frustration_graphics_warns_once_across_calls():
    """Finding #35: docstring promised a single warning per process.

    The pre-fix code warned on EVERY call. We deduplicate via a
    module-level set; reset it in the test so the assertion is hermetic.
    """
    from frustration_gpu import calculate_frustration  # noqa: E402

    _CF_WARN_ONCE.discard("graphics")
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        for _ in range(3):
            calculate_frustration(
                PDB_5AON, mode="configurational",
                graphics=True,
                n_decoys=10, seed=0, device="cpu",
            )
    n_graphics_warnings = sum(
        1 for w in ws
        if issubclass(w.category, UserWarning)
        and "graphic" in str(w.message).lower()
    )
    assert n_graphics_warnings == 1, (
        f"expected exactly one graphics UserWarning across 3 calls; "
        f"got {n_graphics_warnings}"
    )


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_calculate_frustration_overwrite_false_warns_when_file_exists(tmp_path):
    """Finding #35: overwrite=False should warn (once) when the output
    file already exists; pre-fix the kwarg was silently swallowed.
    """
    from frustration_gpu import calculate_frustration  # noqa: E402

    _CF_WARN_ONCE.discard("overwrite_false")
    # Seed an existing dump file so the heuristic detects an overwrite.
    (tmp_path / "5AON_configurational.dat").write_text("placeholder\n")

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        calculate_frustration(
            PDB_5AON, mode="configurational",
            results_dir=tmp_path, overwrite=False,
            n_decoys=10, seed=0, device="cpu",
        )
    overwrite_msgs = [
        str(w.message) for w in ws
        if issubclass(w.category, UserWarning)
        and "overwrite" in str(w.message).lower()
    ]
    assert len(overwrite_msgs) == 1, (
        f"expected one overwrite UserWarning; got {overwrite_msgs}"
    )


# ---------------------------------------------------------------------------
# Happy-path regression: identical output for the canonical valid input
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_5aon(), reason="5AON.pdb missing")
def test_happy_path_smoke_unchanged():
    """Sanity gate: the canonical valid-input call returns the same shapes
    and same metadata fields as the pre-fix code.
    """
    r = compute_frustration(
        PDB_5AON, mode="configurational",
        n_decoys=10, seed=0, device="cpu",
    )
    assert r.pair_records is not None and len(r.pair_records) == 221
    assert r.density_records is not None and len(r.density_records) == 49
    assert r.metadata["n_residues"] == 49
    assert r.metadata["n_pairs"] == 221
    # New fields exist alongside the old ones.
    assert r.metadata["n_residues_unfiltered"] == 49
    assert r.metadata["n_pairs_unfiltered"] == 221
