"""Regression tests for parser hardening fixes landed 2026-05-21.

Covers findings #9, #17, #18, #37, #38, #44, #45, #46, #53, #60, #61 from
``F:/research_plan/New folder/odo.txt``. Each test is a synthetic-PDB-string
reproducer: write a minimal PDB to a tempdir, parse it, assert the bug is
gone.

Important: these tests pin the NEW correct behaviour. Some findings (#38
blank-chain-no-longer-coerced, #53 mixed-resname-now-raises, #37
altloc-B-only-no-longer-dropped) change parser output for previously-
buggy inputs. The intended diff is recorded inline in each docstring
and consolidated in ``docs/v2_fix_parser.md``.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest
import torch

from frustration_gpu.parser import parse_pdb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resnum: int,
    x: float, y: float, z: float,
    *,
    record: str = "ATOM",
    altloc: str = " ",
    icode: str = " ",
    occ: float = 1.00,
    b: float = 0.00,
    element: str | None = None,
) -> str:
    """Format a single PDB ATOM/HETATM line per the column spec.

    Columns (1-indexed for readability):
      1-6   record
      7-11  serial
      13-16 name (left-aligned within columns)
      17    altLoc
      18-20 resName
      22    chainID
      23-26 resSeq
      27    iCode
      31-38 x  (8.3f)
      39-46 y  (8.3f)
      47-54 z  (8.3f)
      55-60 occupancy (6.2f)
      61-66 tempFactor (6.2f)
      77-78 element (right-aligned)
    """
    # Atom name field handling (4 cols, columns 13-16). Per PDB spec, names
    # of 1-3 chars start at column 14; names of 4 chars start at column 13.
    if len(name) >= 4:
        name_field = f"{name:<4s}"
    else:
        name_field = f" {name:<3s}"
    if element is None:
        element = name[0] if name and name[0].isalpha() else " "
    return (
        f"{record:<6s}"
        f"{serial:>5d}"
        f" "
        f"{name_field}"
        f"{altloc:1s}"
        f"{resname:<3s}"
        f" "
        f"{chain:1s}"
        f"{resnum:>4d}"
        f"{icode:1s}"
        f"   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{occ:>6.2f}{b:>6.2f}"
        f"          "
        f"{element:>2s}"
        "\n"
    )


def _backbone(
    serial0: int,
    resname: str,
    chain: str,
    resnum: int,
    cx: float, cy: float, cz: float,
    *,
    record: str = "ATOM",
    altloc: str = " ",
    icode: str = " ",
    occ: float = 1.00,
    include_cb: bool = True,
) -> tuple[str, int]:
    """Return (pdb_text, next_serial) for a complete N/CA/C/O[/CB] residue
    centred at (cx, cy, cz). Trivial bond geometry — enough to parse."""
    s = serial0
    lines = []
    lines.append(_atom_line(s, "N", resname, chain, resnum,
                            cx - 1.0, cy, cz, record=record, altloc=altloc,
                            icode=icode, occ=occ))
    s += 1
    lines.append(_atom_line(s, "CA", resname, chain, resnum,
                            cx, cy, cz, record=record, altloc=altloc,
                            icode=icode, occ=occ))
    s += 1
    lines.append(_atom_line(s, "C", resname, chain, resnum,
                            cx + 0.5, cy + 1.0, cz, record=record,
                            altloc=altloc, icode=icode, occ=occ))
    s += 1
    lines.append(_atom_line(s, "O", resname, chain, resnum,
                            cx - 0.5, cy + 1.5, cz, record=record,
                            altloc=altloc, icode=icode, occ=occ))
    s += 1
    if include_cb and resname != "GLY":
        lines.append(_atom_line(s, "CB", resname, chain, resnum,
                                cx + 0.5, cy - 1.0, cz - 0.5,
                                record=record, altloc=altloc, icode=icode,
                                occ=occ))
        s += 1
    return "".join(lines), s


def _write_and_parse(pdb_text: str, **kwargs):
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "tmp.pdb"
        fp.write_text(pdb_text)
        return parse_pdb(fp, dtype=torch.float64, **kwargs)


# ---------------------------------------------------------------------------
# Finding #9 — HETATM modified amino acids are accepted
# ---------------------------------------------------------------------------

def test_hetatm_mse_is_accepted():
    """#9: HETATM MSE with complete N/CA/C/O/CB should map to MET (idx 12).

    Previously: HETATM was unconditionally rejected at the line level, so a
    PDB whose only selenomethionine was encoded as HETATM raised ValueError
    ("No usable residues parsed"). Now: HETATM lines whose resname is in
    :data:`HETATM_PROMOTED_RESNAMES` are treated as ATOM lines.
    """
    pdb, _ = _backbone(1, "MSE", "A", 1, 0.0, 0.0, 0.0, record="HETATM")
    # Suppress the "non-standard residue mapped" warning we expect.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        d = _write_and_parse(pdb + "END\n")
    assert d["ca_coords"].shape == (1, 3)
    # MSE -> MET, index 12 in the OpenAWSEM gamma order
    assert int(d["residue_types"][0]) == 12


def test_hetatm_water_still_rejected():
    """#9 negative control: HETATM HOH must STILL be dropped."""
    pdb = _atom_line(1, "O", "HOH", "A", 1, 0.0, 0.0, 0.0, record="HETATM")
    # Plus a real ALA so the parser doesn't error on empty.
    real, _ = _backbone(2, "ALA", "A", 2, 5.0, 0.0, 0.0)
    d = _write_and_parse(pdb + real + "END\n")
    # Only ALA survives.
    assert d["ca_coords"].shape == (1, 3)
    assert int(d["residue_numbers"][0]) == 2


# ---------------------------------------------------------------------------
# Finding #17 — TER + same-letter restart
# ---------------------------------------------------------------------------

def test_ter_then_same_chain_letter_creates_new_segment():
    """#17: chain A residue 1, TER, chain A residue 2 must NOT be merged.

    Before the fix the second residue's atoms were silently discarded
    because ``setdefault`` kept the first one's. Now the second segment
    is emitted under chain label ``A#2``, distinct from the first ``A``.
    """
    seg1, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0)
    seg2, _ = _backbone(s, "GLY", "A", 1, 10.0, 0.0, 0.0)
    pdb = seg1 + "TER\n" + seg2 + "END\n"
    d = _write_and_parse(pdb)
    # Two residues, on two distinct chain labels.
    assert d["ca_coords"].shape == (2, 3)
    assert d["chain_ids"] == ["A", "A#2"]


def test_ter_then_different_chain_letter_unchanged():
    """#17 control: ``A ... TER ... B ...`` should behave as before."""
    a, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0)
    b, _ = _backbone(s, "GLY", "B", 1, 10.0, 0.0, 0.0)
    pdb = a + "TER\n" + b + "END\n"
    d = _write_and_parse(pdb)
    assert d["chain_ids"] == ["A", "B"]


def test_chain_filter_accepts_letter_for_both_segments():
    """#17: ``chains=['A']`` should keep both segments of original chain A."""
    seg1, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0)
    seg2, _ = _backbone(s, "VAL", "A", 1, 10.0, 0.0, 0.0)
    pdb = seg1 + "TER\n" + seg2 + "END\n"
    d = _write_and_parse(pdb, chains=["A"])
    assert d["ca_coords"].shape == (2, 3)
    assert d["chain_ids"] == ["A", "A#2"]


# ---------------------------------------------------------------------------
# Finding #18 — insertion codes preserved
# ---------------------------------------------------------------------------

def test_insertion_codes_distinguish_residues():
    """#18: residues A:10A and A:10B parse as two distinct rows."""
    res_a, s = _backbone(1, "ALA", "A", 10, 0.0, 0.0, 0.0, icode="A")
    res_b, _ = _backbone(s, "GLY", "A", 10, 5.0, 0.0, 0.0, icode="B")
    d = _write_and_parse(res_a + res_b + "END\n")
    assert d["ca_coords"].shape == (2, 3)
    # ``residue_numbers`` carries the integer 10 for both, but
    # ``insertion_codes`` distinguishes them.
    assert d["residue_numbers"].tolist() == [10, 10]
    assert d["insertion_codes"] == ["A", "B"]


# ---------------------------------------------------------------------------
# Finding #37 — altloc-B-only residues
# ---------------------------------------------------------------------------

def test_altloc_b_only_residue_is_kept_default_mode():
    """#37: a residue where every atom is altloc B and no A exists must NOT
    silently disappear or raise. In default mode the B record IS the canonical
    position for that residue.
    """
    pdb, _ = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0, altloc="B")
    d = _write_and_parse(pdb + "END\n")
    assert d["ca_coords"].shape == (1, 3)
    assert int(d["residue_numbers"][0]) == 1
    # CA recovered at the altloc-B coords.
    assert torch.allclose(
        d["ca_coords"][0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    )


def test_altloc_a_present_makes_b_silently_dropped():
    """#37 control: when both A and B exist, default mode keeps A."""
    a, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0, altloc="A")
    b, _ = _backbone(s, "ALA", "A", 1, 10.0, 0.0, 0.0, altloc="B")
    d = _write_and_parse(a + b + "END\n")
    assert d["ca_coords"].shape == (1, 3)
    # CA equals altloc-A coords (0, 0, 0), not altloc-B (10, 0, 0).
    assert torch.allclose(
        d["ca_coords"][0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    )


# ---------------------------------------------------------------------------
# Finding #38 — blank chain IDs not coerced to "A"
# ---------------------------------------------------------------------------

def test_blank_chain_kept_distinct_from_chain_A():
    """#38: a blank chain ID must not be silently merged into chain A.

    Before fix: both residues parsed as chain A and the first one's coords
    won. Now: blank-chain residue gets chain label "" (empty string).
    """
    res_blank, s = _backbone(1, "ALA", " ", 1, 0.0, 0.0, 0.0)
    res_a, _ = _backbone(s, "GLY", "A", 1, 10.0, 0.0, 0.0)
    d = _write_and_parse(res_blank + res_a + "END\n")
    assert d["ca_coords"].shape == (2, 3)
    # One blank-chain residue + one A-chain residue
    assert set(d["chain_ids"]) == {"", "A"}


# ---------------------------------------------------------------------------
# Finding #44 — occupancy-aware atom dedup
# ---------------------------------------------------------------------------

def test_duplicate_atom_records_pick_higher_occupancy():
    """#44: two identical atom-name records → the higher-occupancy coord wins.

    Previously the first record always won regardless of occupancy. This
    test crafts the first record with low occupancy (0.30) at (0,0,0) and
    the second with high occupancy (0.70) at (5,0,0). The CA should be at
    (5,0,0) after the fix.
    """
    # Both records have altloc blank so my altloc filter doesn't kick in.
    lines: list[str] = [
        _atom_line(1, "N", "ALA", "A", 1, -1.0, 0.0, 0.0, occ=1.0),
        # Two CAs with different occupancies.
        _atom_line(2, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, occ=0.30),
        _atom_line(3, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0, occ=0.70),
        _atom_line(4, "C", "ALA", "A", 1, 0.5, 1.0, 0.0, occ=1.0),
        _atom_line(5, "O", "ALA", "A", 1, -0.5, 1.5, 0.0, occ=1.0),
        _atom_line(6, "CB", "ALA", "A", 1, 0.5, -1.0, -0.5, occ=1.0),
    ]
    d = _write_and_parse("".join(lines) + "END\n")
    assert d["ca_coords"].shape == (1, 3)
    # Higher-occupancy CA wins.
    assert torch.allclose(
        d["ca_coords"][0], torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
    )


# ---------------------------------------------------------------------------
# Finding #45 — END terminates parsing
# ---------------------------------------------------------------------------

def test_end_record_stops_parsing():
    """#45: atoms appearing AFTER ``END`` must not be parsed.

    Previously only ``ENDMDL`` stopped parsing; ``END`` was ignored.
    """
    res1, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0)
    res2, _ = _backbone(s, "GLY", "A", 2, 5.0, 0.0, 0.0)
    pdb = res1 + "END\n" + res2
    d = _write_and_parse(pdb)
    assert d["ca_coords"].shape == (1, 3)
    assert int(d["residue_numbers"][0]) == 1


# ---------------------------------------------------------------------------
# Finding #46 — non-finite coordinates rejected
# ---------------------------------------------------------------------------

def test_non_finite_coords_are_rejected():
    """#46: a CA with NaN x-coord must be rejected, not silently parsed."""
    # Build a residue where CA has NaN x. We have to format NaN by hand
    # because format spec '%.3f' for float('nan') produces 'nan' (only 3
    # chars). To keep column alignment, hand-emit the line.
    nan_ca = (
        "ATOM      1  CA  ALA A   1         nan   0.000   0.000  1.00  0.00           C  \n"
    )
    # Also emit a valid backup residue so the parser hits the "no usable
    # residues" branch only when our CA is correctly rejected.
    other, _ = _backbone(2, "ALA", "A", 2, 5.0, 0.0, 0.0)
    pdb = nan_ca + other + "END\n"
    d = _write_and_parse(pdb)
    # Only the valid residue (A:2) survives.
    assert d["ca_coords"].shape == (1, 3)
    assert int(d["residue_numbers"][0]) == 2


def test_non_finite_only_pdb_raises():
    """#46: when EVERY residue has non-finite coords, parser must raise."""
    nan_ca = (
        "ATOM      1  CA  ALA A   1         nan   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      3  C   ALA A   1       0.500   1.000   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1      -0.500   1.500   0.000  1.00  0.00           O  \n"
        "ATOM      5  CB  ALA A   1       0.500  -1.000  -0.500  1.00  0.00           C  \n"
        "END\n"
    )
    # Without CA, the strict backbone filter drops the residue.
    with pytest.raises(ValueError):
        _write_and_parse(nan_ca)


# ---------------------------------------------------------------------------
# Finding #53 — mixed resnames at same (chain, resnum, icode)
# ---------------------------------------------------------------------------

def test_mixed_resname_same_key_raises():
    """#53: ALA and GLY records at A:1 must not silently merge into ALA.

    Previously the parser kept the first resname and merged the other's
    atoms in. Now it raises a clear ValueError. Microheterogeneous sites
    must use altloc to disambiguate.
    """
    lines = [
        _atom_line(1, "N", "ALA", "A", 1, -1.0, 0.0, 0.0),
        _atom_line(2, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        # Now a GLY record for the SAME (chain, resnum, icode).
        _atom_line(3, "C", "GLY", "A", 1, 0.5, 1.0, 0.0),
        _atom_line(4, "O", "GLY", "A", 1, -0.5, 1.5, 0.0),
    ]
    with pytest.raises(ValueError, match="conflicting residue names"):
        _write_and_parse("".join(lines) + "END\n")


# ---------------------------------------------------------------------------
# Finding #60 — second MODEL without intervening ENDMDL
# ---------------------------------------------------------------------------

def test_second_model_without_endmdl_stops_parsing():
    """#60: ``MODEL 1 ... MODEL 2 ...`` (no ENDMDL) must stop at MODEL 2.

    Previously the parser only stopped on ENDMDL, so the residues from
    both models were silently appended together.
    """
    res1, s = _backbone(1, "ALA", "A", 1, 0.0, 0.0, 0.0)
    res2, _ = _backbone(s, "GLY", "A", 2, 5.0, 0.0, 0.0)
    pdb = (
        "MODEL        1\n"
        + res1
        + "MODEL        2\n"
        + res2
        + "END\n"
    )
    d = _write_and_parse(pdb)
    assert d["ca_coords"].shape == (1, 3)
    assert int(d["residue_numbers"][0]) == 1


# ---------------------------------------------------------------------------
# Finding #61 — OXT-only terminal residue
# ---------------------------------------------------------------------------

def test_oxt_promotes_to_carbonyl_O_when_O_missing():
    """#61: a terminal residue with OXT but no O is kept (OXT fills O slot).

    Previously the strict-backbone filter dropped these residues because
    "O" was not in the atoms dict.
    """
    lines: list[str] = [
        _atom_line(1, "N", "ALA", "A", 1, -1.0, 0.0, 0.0),
        _atom_line(2, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom_line(3, "C", "ALA", "A", 1, 0.5, 1.0, 0.0),
        # No "O" record — only OXT.
        _atom_line(4, "OXT", "ALA", "A", 1, -0.5, 1.5, 0.0),
        _atom_line(5, "CB", "ALA", "A", 1, 0.5, -1.0, -0.5),
    ]
    d = _write_and_parse("".join(lines) + "END\n")
    assert d["ca_coords"].shape == (1, 3)
    # The "O" tensor slot should be finite (filled from OXT coords).
    assert torch.isfinite(d["o_coords"][0]).all()
    # And it should equal the OXT coords.
    assert torch.allclose(
        d["o_coords"][0],
        torch.tensor([-0.5, 1.5, 0.0], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Validation-PDB golden-anchor regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected_n_res,expected_chains",
    [
        ("5AON", 49, ["A"]),
        ("11BG", 248, ["A", "B"]),
        ("1O3S", 200, ["A"]),
        ("3F9M", 451, ["A"]),
    ],
)
def test_validation_pdbs_unchanged(name, expected_n_res, expected_chains, pdb_dir):
    """Residue counts and chain labels on the bundled validation PDBs are
    UNCHANGED by the 2026-05-21 parser hardening. These are golden anchors —
    any future parser change that moves them must justify itself in the
    fix doc.
    """
    d = parse_pdb(pdb_dir / f"{name}.pdb")
    assert d["ca_coords"].shape[0] == expected_n_res, (
        f"{name}: expected {expected_n_res} residues, got {d['ca_coords'].shape[0]}"
    )
    assert sorted(set(d["chain_ids"])) == expected_chains
