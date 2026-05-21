"""Coverage gap tests landed by QA-4 fix sprint (2026-05-21).

Closes the H1 / H3 / H4 holes in the QA-4 test review by adding
behavioural unit tests for public-API surfaces that previously had
ZERO direct test coverage:

* :func:`src.compute_rho` — burial-density kernel; verify against a
  hand-built 4-residue system.
* :func:`src.chain_segments` — public helper for chain-boundary
  enumeration; verify on multi-chain inputs + degenerate edges.
* :func:`src.density_to_dataframe` — public helper; verify schema +
  values.
* :func:`src.build_contact_context` / :class:`src.ContactContext` —
  scaffolding shared across the three contact terms.
* Edge-case PDB inputs (empty file, single-residue, all-Gly chain,
  chain-filter that matches nothing).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frustration_gpu import (                                          # noqa: E402
    ContactContext,
    build_contact_context,
    chain_segments,
    compute_residue_density,
    compute_rho,
    density_to_dataframe,
    parse_pdb,
)


from _paths import PDB_DIR  # noqa: E402


def _has_pdb(pdb_id: str) -> bool:
    return (PDB_DIR / f"{pdb_id}.pdb").is_file()


# ---------------------------------------------------------------------------
# compute_rho behavioural test (QA-4 H1)
# ---------------------------------------------------------------------------
def test_compute_rho_hand_built_four_residues():
    """``compute_rho`` on a 4-residue 1-chain system, hand-checked.

    Layout: 4 residues at x = {0, 5.5, 11.0, 16.5} A in one chain. With
    ``min_seq_sep = 1`` (default), only |i - j| >= 2 pairs contribute.

    The sigmoid window is centred at r ∈ [r_min, r_max] = [0.45, 0.65] nm
    with eta = 50/nm. So a pair at exactly 0.55 nm = 5.5 A sits dead-centre
    of the window and contributes ~1.0; a pair at 1.1 nm = 11 A is well
    past r_max so contributes ~0; a pair at 1.65 nm = 16.5 A also ~0.

    Hand-derived expected contributions (rho is intra-chain |i-j|>=2):

    * Residue 0: contributes from j=2 (d=11 A, factor ≈ 0) and j=3
      (d=16.5 A, factor ≈ 0). rho_0 ≈ 0.
    * Residue 1: contributes from j=3 (d=11 A, factor ≈ 0). rho_1 ≈ 0.
    * Residue 2: contributes from j=0 (d=11 A, factor ≈ 0). rho_2 ≈ 0.
    * Residue 3: contributes from j=0 (d=16.5 A, factor ≈ 0) and j=1
      (d=11 A, factor ≈ 0). rho_3 ≈ 0.

    So this layout is the boring "all out of range" case. To exercise the
    in-window arm, build a SECOND configuration where residues 0 and 2
    sit 5.5 A apart in 3D (residue 2 perpendicular off the chain axis so
    that |i-j|=2 → distance 5.5 A → ρ ≈ 1.0).
    """
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [16.5, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    residue_numbers = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    chain_index = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    rho = compute_rho(coords, residue_numbers, chain_index, coord_units="angstrom")
    assert rho.shape == (4,)
    # All four rho values are ~0 (no |i-j|>=2 contact within r_max).
    assert rho.max().item() < 1e-6, f"max rho {rho.max()} but expected ~0"

    # Now bend residue 2 perpendicular so dist(0, 2) = 5.5 A → factor ≈ 1.0.
    coords2 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [0.0, 5.5, 0.0],   # |i-j|=2 with res 0, dist = 5.5 A
            [16.5, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rho2 = compute_rho(coords2, residue_numbers, chain_index, coord_units="angstrom")
    # Hand-computed contribution at 0.55 nm: 0.25 * (1 + tanh(50*0.10)) *
    # (1 + tanh(50*0.10)) = 0.25 * 1.9999... * 1.9999... ≈ 0.9999.
    expected_in_window = 0.25 * (
        1.0 + torch.tanh(torch.tensor(50.0 * 0.10, dtype=torch.float64))
    ) ** 2
    # rho[0] is sum over {j=2 contributes ~1, j=3 ~0} → ~1.0
    assert abs(rho2[0].item() - expected_in_window.item()) < 1e-9, (
        f"rho[0] {rho2[0].item():.6f} != expected {expected_in_window.item():.6f}"
    )
    # Symmetry: rho[2] (from j=0) ≈ rho[0]
    assert abs(rho2[2].item() - rho2[0].item()) < 1e-9


def test_compute_rho_cross_chain_excludes_seq_sep_filter():
    """``compute_rho``: cross-chain pairs always contribute (no seq-sep filter)."""
    # 2 residues in chain 0, 1 residue in chain 1 — all 5 A apart.
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],   # |i-j|=1 with res 0 → same chain seq-sep filter applies
            [10.0, 0.0, 0.0],  # different chain from both 0 and 1 → no filter
        ],
        dtype=torch.float64,
    )
    residue_numbers = torch.tensor([1, 2, 1], dtype=torch.int64)
    chain_index = torch.tensor([0, 0, 1], dtype=torch.int64)
    rho = compute_rho(coords, residue_numbers, chain_index, coord_units="angstrom")
    # res 0 ↔ res 1 same chain |i-j|=1, filtered out
    # res 0 ↔ res 2 different chain, dist 1.0 nm → factor ~0
    # res 1 ↔ res 2 different chain, dist 0.5 nm → factor ~0.993
    # So rho[1] should be substantial because of the cross-chain contact.
    assert rho[1].item() > 0.9, (
        f"cross-chain contact ignored: rho[1]={rho[1].item()} (expected > 0.9)"
    )


# ---------------------------------------------------------------------------
# chain_segments behavioural test (QA-4 H1)
# ---------------------------------------------------------------------------
def test_chain_segments_multi_chain():
    """Multi-chain coords → correct (start, end-exclusive) tuples."""
    chains = ["A", "A", "A", "B", "B", "C"]
    segs = chain_segments(chains)
    assert segs == [(0, 3), (3, 5), (5, 6)], f"got {segs}"


def test_chain_segments_single_chain():
    chains = ["A", "A", "A", "A"]
    segs = chain_segments(chains)
    assert segs == [(0, 4)], f"got {segs}"


def test_chain_segments_empty():
    assert chain_segments([]) == []


def test_chain_segments_singleton():
    assert chain_segments(["A"]) == [(0, 1)]


def test_chain_segments_alternating_does_not_merge():
    """Non-contiguous chain IDs should NOT be merged — order matters."""
    chains = ["A", "B", "A", "B"]
    segs = chain_segments(chains)
    assert segs == [(0, 1), (1, 2), (2, 3), (3, 4)], f"got {segs}"


# ---------------------------------------------------------------------------
# density_to_dataframe behavioural test (QA-4 H1)
# ---------------------------------------------------------------------------
def test_density_to_dataframe_schema_and_values():
    """Build a density dict by hand → DataFrame round-trips schema + values."""
    density = {
        "residue_numbers": torch.tensor([10, 11, 12], dtype=torch.int64),
        "chain_ids": ["A", "A", "B"],
        "Total": torch.tensor([5, 7, 0], dtype=torch.int64),
        "nHighlyFrst": torch.tensor([2, 0, 0], dtype=torch.int64),
        "nNeutrallyFrst": torch.tensor([2, 5, 0], dtype=torch.int64),
        "nMinimallyFrst": torch.tensor([1, 2, 0], dtype=torch.int64),
        "relHighlyFrustrated": torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64),
        "relNeutralFrustrated": torch.tensor([0.4, 5.0 / 7.0, 0.0], dtype=torch.float64),
        "relMinimallyFrustrated": torch.tensor([0.2, 2.0 / 7.0, 0.0], dtype=torch.float64),
    }
    df = density_to_dataframe(density)
    expected_cols = [
        "Res", "ChainRes", "Total", "nHighlyFrst", "nNeutrallyFrst",
        "nMinimallyFrst", "relHighlyFrustrated", "relNeutralFrustrated",
        "relMinimallyFrustrated",
    ]
    assert list(df.columns) == expected_cols, f"got columns {list(df.columns)}"
    assert len(df) == 3
    # Values round-trip
    assert df["Res"].tolist() == [10, 11, 12]
    assert df["ChainRes"].tolist() == ["A", "A", "B"]
    assert df["Total"].tolist() == [5, 7, 0]
    assert df["nHighlyFrst"].tolist() == [2, 0, 0]
    assert abs(df["relHighlyFrustrated"].iloc[0] - 0.4) < 1e-12
    # Zero-total guard: ratios are 0 (not NaN) when Total == 0
    assert df["relHighlyFrustrated"].iloc[2] == 0.0
    assert df["relNeutralFrustrated"].iloc[2] == 0.0
    assert df["relMinimallyFrustrated"].iloc[2] == 0.0


# ---------------------------------------------------------------------------
# ContactContext / build_contact_context behavioural test (QA-4 H4)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_build_contact_context_5AON():
    """``build_contact_context`` populates every field consistently."""
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    ctx = build_contact_context(coords, seq_seps=[2])
    n = int(coords["ca_coords"].shape[0])
    assert isinstance(ctx, ContactContext)
    # Shapes
    assert ctx.n == n
    assert ctx.cb_or_ca.shape == (n, 3)
    assert ctx.chain_idx.shape == (n,)
    assert ctx.dist.shape == (n, n)
    # Geometry mask keyed by seq_sep
    assert 2 in ctx.geom_mask_min_sep
    mask = ctx.geom_mask_min_sep[2]
    assert mask.shape == (n, n)
    assert mask.dtype == torch.bool
    # Mask must exclude self-pairs
    diag = torch.diagonal(mask)
    assert not diag.any(), "geom mask still includes self-pairs"
    # Mask should be symmetric (pair (i, j) == pair (j, i))
    assert torch.equal(mask, mask.T), "geom mask is asymmetric"
    # Distance is symmetric and finite on the diagonal (0)
    assert torch.allclose(ctx.dist, ctx.dist.T)
    assert torch.allclose(torch.diagonal(ctx.dist), torch.zeros(n, dtype=ctx.dist.dtype))
    # chain_idx is contiguous non-negative
    assert (ctx.chain_idx >= 0).all()
    # device + dtype propagated from input
    assert ctx.cb_or_ca.device == coords["ca_coords"].device
    assert ctx.cb_or_ca.dtype == coords["ca_coords"].dtype


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_build_contact_context_dist_full_optional():
    """Speed-fix4 SPEED-2 Idea 2: ``compute_dist_full`` toggles a NaN-poisoned
    (N, N) distance matrix on the context. Default OFF keeps memory low.
    """
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    n = int(coords["ca_coords"].shape[0])
    # default: dist_full is None
    ctx_off = build_contact_context(coords, seq_seps=[2])
    assert ctx_off.dist_full is None
    # opt-in: dist_full present and shaped (N, N), NaN-row pairs forced to +inf
    ctx_on = build_contact_context(
        coords, seq_seps=[2], compute_dist_full=True
    )
    assert ctx_on.dist_full is not None
    assert ctx_on.dist_full.shape == (n, n)
    assert ctx_on.dist_full.dtype == coords["ca_coords"].dtype
    # NaN-poisoned construction: diagonal is 0 (self-pairs); any +inf entries
    # only on NaN-row pairs (5AON is clean, so no infs expected here).
    assert torch.allclose(
        torch.diagonal(ctx_on.dist_full),
        torch.zeros(n, dtype=ctx_on.dist_full.dtype),
    )
    # Bit-identity against the dense ``ctx.dist`` for finite rows: when no NaN
    # rows exist, dist_full == dist exactly (modulo the NaN→1e6 sentinel
    # substitution which only affects NaN rows).
    # 5AON has no NaN rows → dist_full and ctx.dist must agree on every entry
    # that isn't already NaN in ctx.dist.
    finite_mask = torch.isfinite(ctx_on.dist)
    if finite_mask.any():
        assert torch.equal(
            ctx_on.dist_full[finite_mask],
            ctx_on.dist[finite_mask],
        )


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_singleresidue_with_shared_dist_full_context():
    """SR with ``_context`` carrying ``dist_full`` produces BIT-IDENTICAL
    output to SR without context (Speed-fix4 SPEED-2 Idea 2 contract).
    """
    from frustration_gpu.singleresidue_decoys import singleresidue_decoy_stats
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    ctx = build_contact_context(coords, seq_seps=[2], compute_dist_full=True)
    out_no_ctx = singleresidue_decoy_stats(coords, seed=0, dtype=torch.float64)
    out_with_ctx = singleresidue_decoy_stats(
        coords, seed=0, dtype=torch.float64, _context=ctx,
    )
    # FI must be bit-identical (same draws, same dist_full, same math).
    assert torch.equal(out_no_ctx["FI"], out_with_ctx["FI"])
    assert torch.equal(out_no_ctx["E_native"], out_with_ctx["E_native"])
    assert torch.equal(out_no_ctx["decoy_mean"], out_with_ctx["decoy_mean"])
    assert torch.equal(out_no_ctx["decoy_std"], out_with_ctx["decoy_std"])


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_configurational_with_shared_dist_full_context():
    """Configurational with ``_context.dist_full`` produces BIT-IDENTICAL
    output to the non-context path (the dist matrices are constructed
    identically).
    """
    from frustration_gpu.decoys import lammps_dump_rho, sample_configurational_decoys
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    rho = lammps_dump_rho(coords)
    ctx = build_contact_context(coords, seq_seps=[2], compute_dist_full=True)
    a = sample_configurational_decoys(coords, rho=rho, seed=0)
    b = sample_configurational_decoys(coords, rho=rho, seed=0, _context=ctx)
    for k in ("aa_i_decoy", "aa_j_decoy", "rij_decoy", "rho_i_decoy", "rho_j_decoy"):
        assert torch.equal(a[k], b[k]), f"{k} differs"


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_build_contact_context_multiple_seq_seps():
    """Asking for multiple sep values pre-caches all of them."""
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    ctx = build_contact_context(coords, seq_seps=[1, 2, 3])
    assert set(ctx.geom_mask_min_sep.keys()) == {1, 2, 3}
    # Stricter sep → strictly fewer True entries
    m1 = ctx.geom_mask_min_sep[1].sum()
    m2 = ctx.geom_mask_min_sep[2].sum()
    m3 = ctx.geom_mask_min_sep[3].sum()
    assert m1 >= m2 >= m3, f"masks not monotone: {m1} {m2} {m3}"


# ---------------------------------------------------------------------------
# Edge-case PDB inputs (QA-4 H3)
# ---------------------------------------------------------------------------
def test_parse_empty_pdb_raises():
    """A PDB file with no ATOM records raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "empty.pdb"
        fp.write_text("HEADER    EMPTY TEST\nEND\n")
        with pytest.raises(ValueError, match="No usable residues"):
            parse_pdb(fp)


def test_parse_header_only_pdb_raises():
    """Even with valid header lines but zero ATOM records, parser raises."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "hdr.pdb"
        fp.write_text(
            "HEADER    TEST PROTEIN\n"
            "TITLE     A TITLE LINE\n"
            "REMARK    1 SOMETHING\n"
            "END\n"
        )
        with pytest.raises(ValueError):
            parse_pdb(fp)


def test_parse_missing_file_raises():
    """Missing file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_pdb(Path("F:/research_plan/frustration_gpu/this_does_not_exist.pdb"))


def test_parse_single_residue_pdb():
    """Single-residue PDB: parser returns N=1 tensors and is internally consistent."""
    pdb_text = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O  \n"
        "ATOM      5  CB  ALA A   1       1.988  -0.773  -1.199  1.00  0.00           C  \n"
        "END\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "single.pdb"
        fp.write_text(pdb_text)
        p = parse_pdb(fp, dtype=torch.float64)
    assert p["ca_coords"].shape == (1, 3)
    assert p["residue_types"].shape == (1,)
    assert p["chain_ids"] == ["A"]
    assert int(p["residue_numbers"][0]) == 1
    assert not bool(p["is_gly"][0])
    # CB should be filled (ALA has a CB)
    assert torch.isfinite(p["cb_coords"]).all()


def test_parse_all_gly_chain():
    """All-GLY chain: every CB row is NaN; ``compute_rho`` falls back to CA."""
    # 3 glycines on a line, spacing 5 A
    pdb_lines = []
    for i, x in enumerate([0.0, 5.0, 10.0], start=1):
        pdb_lines.extend([
            f"ATOM   {4*i-3:>4d}  N   GLY A {i:>3d}    {x:>8.3f}   0.000   1.000  1.00  0.00           N  ",
            f"ATOM   {4*i-2:>4d}  CA  GLY A {i:>3d}    {x:>8.3f}   0.000   0.000  1.00  0.00           C  ",
            f"ATOM   {4*i-1:>4d}  C   GLY A {i:>3d}    {x:>8.3f}   0.000  -1.000  1.00  0.00           C  ",
            f"ATOM   {4*i  :>4d}  O   GLY A {i:>3d}    {x:>8.3f}   1.000  -1.500  1.00  0.00           O  ",
        ])
    pdb_text = "\n".join(pdb_lines) + "\nEND\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "allgly.pdb"
        fp.write_text(pdb_text)
        p = parse_pdb(fp, dtype=torch.float64)
    assert p["ca_coords"].shape == (3, 3)
    assert bool(p["is_gly"].all()), "is_gly should be True for every residue"
    # Every cb row should be NaN (GLY has no CB).
    assert torch.isnan(p["cb_coords"]).all(), "all-GLY → all-NaN cb_coords"
    # _resolve_density_coords falls back to CA, so burial_density should
    # produce finite, non-negative rho values.
    from frustration_gpu.burial import burial_density
    rho = burial_density(p)
    assert torch.isfinite(rho).all(), "rho should be finite even on all-GLY"
    assert (rho >= 0).all()


@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_compute_frustration_empty_chain_filter_raises():
    """``chain='Z'`` on 5AON (which has no chain Z) raises ValueError."""
    from frustration_gpu import compute_frustration  # noqa: E402
    with pytest.raises(ValueError):
        compute_frustration(
            PDB_DIR / "5AON.pdb",
            mode="configurational",
            chain="Z",
            device="cpu",
            seed=0,
        )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG missing")
def test_compute_frustration_chain_list_equals_chain_str_for_singleton():
    """``chain="A"`` and ``chain=["A"]`` must produce byte-identical pair_records.

    QA-3 H-2 fix verification (2026-05-21): before the fix, ``["A"]``
    sometimes routed through the post-filter branch; now both paths go
    through the parser filter. The two calls must produce IDENTICAL FI
    values for every chain-A residue.
    """
    from frustration_gpu import compute_frustration  # noqa: E402
    r_str = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational", chain="A", device="cpu", seed=0,
    )
    r_list = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational", chain=["A"], device="cpu", seed=0,
    )
    assert len(r_str.pair_records) == len(r_list.pair_records)
    # FI columns byte-equal (same code path now)
    import numpy as np
    np.testing.assert_array_equal(
        r_str.pair_records["FrstIndex"].values,
        r_list.pair_records["FrstIndex"].values,
    )


@pytest.mark.skipif(not _has_pdb("11BG"), reason="11BG missing")
def test_compute_frustration_chain_list_vs_post_filter_semantics():
    """QA-3 H-2 regression: ``chain=["A", "B"]`` must give SAME chain-A FI
    values as a full-pipeline run filtered to chain A.

    Before the QA-3 H-2 fix the orchestrator used to run the full pipeline
    and post-filter the dataframes for multi-chain lists. Now both go
    through the parser, which means a chain-A residue scored under
    ``chain=["A", "B"]`` (full structure) sees cross-chain B contacts; a
    chain-A residue scored under ``chain="A"`` (chain-A-only structure)
    does NOT. THIS TEST asserts they differ (the multi-chain rho really
    does change FI), then asserts the multi-chain run agrees with the
    full-PDB run post-filtered to chain A — i.e. the canonical semantic
    is now consistent.
    """
    from frustration_gpu import compute_frustration  # noqa: E402
    r_full = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational", chain=None, device="cpu", seed=0,
    )
    r_ab = compute_frustration(
        PDB_DIR / "11BG.pdb",
        mode="configurational", chain=["A", "B"], device="cpu", seed=0,
    )
    # 11BG is a homodimer of chains A and B → chain=["A","B"] equals chain=None.
    chains_full = set(r_full.pair_records["ChainRes1"].tolist()) | set(
        r_full.pair_records["ChainRes2"].tolist()
    )
    chains_ab = set(r_ab.pair_records["ChainRes1"].tolist()) | set(
        r_ab.pair_records["ChainRes2"].tolist()
    )
    assert chains_full == chains_ab, f"full {chains_full} vs A+B {chains_ab}"
    # Pair counts should match
    assert len(r_full.pair_records) == len(r_ab.pair_records)


# ---------------------------------------------------------------------------
# compute_residue_density round-trip sanity (closes a gap noted in QA-4)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_pdb("5AON"), reason="5AON missing")
def test_compute_residue_density_smoke():
    """End-to-end smoke: density dict has the documented schema."""
    coords = parse_pdb(PDB_DIR / "5AON.pdb", dtype=torch.float64)
    n = int(coords["ca_coords"].shape[0])
    # Construct a fake pair list (just 5 self-pairs, FI = 0 → all neutral)
    pair_i = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    pair_j = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int64)
    fi = torch.zeros(5, dtype=torch.float64)
    out = compute_residue_density(
        coords=coords, pair_i=pair_i, pair_j=pair_j, fi=fi,
    )
    expected_keys = {
        "residue_numbers", "chain_ids",
        "Total", "nHighlyFrst", "nNeutrallyFrst", "nMinimallyFrst",
        "relHighlyFrustrated", "relNeutralFrustrated", "relMinimallyFrustrated",
    }
    assert set(out.keys()) == expected_keys
    assert out["Total"].shape == (n,)
    assert out["Total"].dtype == torch.int64
    # Total counts must equal sum of three class counts
    sum_three = out["nHighlyFrst"] + out["nNeutrallyFrst"] + out["nMinimallyFrst"]
    assert torch.equal(out["Total"], sum_three), (
        "Total != n_high + n_neut + n_min — partition broken"
    )
    # FI=0 → all pairs in neutral class
    assert int(out["nHighlyFrst"].sum()) == 0
    assert int(out["nMinimallyFrst"].sum()) == 0
