"""PDB parsing to PyTorch tensors for AWSEM frustration analysis.

Extracts CA / N / O / CB coordinates, residue identity and chain assignment
from a PDB file, mapping non-standard residues to their canonical AA where
possible (e.g. MSE -> MET, SEC -> CYS).

HETATM records are *normally* skipped, with one exception: HETATM lines whose
resname is a known modified amino acid (MSE / SEC / PYL / HID / HIE / HIP /
CYX / CYM / ASH / GLH / LYN) are PROMOTED to be treated as ATOM records
(see finding #9 in ``docs/v2_fix_parser.md``). RCSB encodes selenomethionine
as HETATM by convention; dropping it silently mis-counts residues.

AA index convention follows OpenAWSEM's ``gamma_se_map_1_letter`` (the index
used by ``gamma.dat`` and ``burial_gamma.dat``):

    A R N D C Q E G H I L K M F P S T W Y V
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

This is NOT the same as ``se_map_3_letter`` in ``openAWSEM.py`` (that one
indexes residue records, not gamma columns). Be careful when cross-referencing.

Parser hardening fixes (2026-05-21, see ``docs/v2_fix_parser.md``)
------------------------------------------------------------------
This module's audit findings #9, #17, #18, #37, #38, #44, #45, #46, #53, #60,
and #61 were addressed in this pass. The fixes that may affect public output
on edge-case PDBs are:

* **TER + same-letter restart** (#17): when a TER record is followed by ATOM
  records sharing the same chain letter, the second segment is emitted as
  ``"<letter>#<seg_num>"`` (e.g. ``A`` then ``A#2``) so the downstream
  ``_build_chain_index`` treats them as distinct chains. On the 4 bundled
  validation PDBs (5AON / 11BG / 1O3S / 3F9M) no segments repeat, so the
  visible chain IDs are unchanged.
* **Blank chain ID** (#38): blank chain IDs are now kept as the empty
  string ``""`` instead of being silently coerced to ``"A"``. If a real
  ``A`` chain also exists, this prevents a silent merge collision.
* **Insertion codes** (#18): each ``(chain, resnum, icode)`` is a distinct
  residue. Previously this was true at the parser level too, but the bug
  flag here is documenting that **callers must look at ``insertion_codes``
  to distinguish residues with the same integer ``resnum``** — they are
  not collapsed by parse_pdb.
* **Altloc-B-only residues** (#37): when no altloc-A record exists for a
  ``(chain, resnum, icode)`` we now keep the altloc-B record as the
  canonical position (in BOTH default and ``lammps_compat_altloc`` mode).
  Previously this was silently dropped (or raised IndexError).
* **HETATM modified AAs** (#9): HETATM lines whose resname is one of
  ``MSE / SEC / PYL / HID / HIE / HIP / CYX / CYM / ASH / GLH / LYN``
  are now accepted exactly as the equivalent ATOM line would be.
* **First-coord-wins → highest-occupancy-wins** (#44): when multiple
  ``ATOM`` lines have the same atom name within a residue (rare; usually
  the altloc handling catches this), we now pick the one with the highest
  occupancy instead of the first encountered.
* **END terminator** (#45): coordinate parsing stops at the first ``END``
  record (in addition to the existing ``ENDMDL`` stop).
* **Non-finite coordinates** (#46): atoms whose x/y/z parse to NaN or
  Inf are rejected at the line level.
* **Mixed resname collision** (#53): if two ATOM lines for the same
  ``(chain, resnum, icode)`` have different resnames (e.g. ALA + GLY),
  the parser now raises ``ValueError``. Microheterogeneous residues
  must be expressed via altloc.
* **MODEL without ENDMDL** (#60): if a second ``MODEL`` record appears
  with no preceding ``ENDMDL``, parsing stops at that second ``MODEL``.
* **OXT-only terminal residues** (#61): terminal residues that supply an
  ``OXT`` atom but no ``O`` atom now have ``OXT`` promoted to the
  ``O`` slot so the strict backbone filter accepts them.

LAMMPS-compatibility flags (2026-05-20 fix pass)
------------------------------------------------
Three opt-in flags exist to reproduce known frustrapy + LAMMPS-AWSEM behaviours
that diverge from the cleaner "biopython picks altloc-A; drop non-protein
chains; require full backbone" defaults:

* ``keep_incomplete_backbone=False`` (default) — drop residues lacking ANY of
  N / CA / C / O. Matches PDBToCoordinates.py:182-191 in LAMMPS-AWSEM. With
  ``True`` we keep residues with NaN backbone slots (less strict).
* ``include_dna=False`` (default) — DNA chains (DA / DT / DC / DG) are
  dropped entirely (scientifically correct; AWSEM has no DNA force field).
  With ``True`` they are emitted as "dna-placeholder" residues using the
  ``C1'`` atom as a CA proxy. This is opt-in compat ONLY — AWSEM frustration
  on DNA is not physically meaningful.
* ``lammps_compat_altloc=False`` (default) — only altloc A / blank is kept
  (BioPython default). With ``True`` altloc B records are emitted as an
  additional residue inserted right after their altloc A entry; the inserted
  residue inherits the altloc-A chain / resnum so it duplicates the
  containing position. Reproduces the LAMMPS-AWSEM PDBToCoordinates +
  density-iteration pattern that yields consecutive duplicate-density rows
  on PDBs with alt-conformers (e.g. 3F9M).
"""
from __future__ import annotations

from pathlib import Path

import torch

# Three-letter -> one-letter, including the most common non-standard residues
# encountered in PDB files. Anything not in here is dropped.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standards mapped to their canonical form per the PDB
    # MODRES / RCSB conventions used by frustrapy / PDBFixer.
    "MSE": "M",   # selenomethionine
    "SEC": "C",   # selenocysteine
    "PYL": "K",   # pyrrolysine
    "HID": "H", "HIE": "H", "HIP": "H",   # protonation-state variants
    "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E",
    "LYN": "K",
}

# 3-letter codes considered "standard" amino acids (no warning when seen).
# Every other key in THREE_TO_ONE is a non-standard residue that we silently
# coerce to a canonical AA; parse_pdb emits a single UserWarning per (file,
# 3-letter) combination so the user knows the substitution happened.
STANDARD_AA_3LETTERS: frozenset = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})

# OpenAWSEM gamma_se_map_1_letter (the column order in gamma.dat / burial_gamma.dat).
ONE_TO_IDX: dict[str, int] = {
    "A": 0,  "R": 1,  "N": 2,  "D": 3,  "C": 4,
    "Q": 5,  "E": 6,  "G": 7,  "H": 8,  "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

# DNA residue 3-letter codes — used by the opt-in `include_dna` flag.
DNA_RESNAMES: tuple = ("DA", "DT", "DC", "DG", "A", "T", "C", "G", "U", "DU")

# 3-letter codes for modified amino acids that RCSB / many older deposits
# encode as ``HETATM`` records rather than ``ATOM``. Finding #9 (2026-05-21):
# silently dropping these can delete residues from a structure that already
# advertised support via :data:`THREE_TO_ONE`. We promote these specific
# HETATM resnames to be treated as ATOM lines; every other HETATM record
# is still ignored (water, ligands, ions, ...).
HETATM_PROMOTED_RESNAMES: frozenset = frozenset({
    "MSE", "SEC", "PYL",
    "HID", "HIE", "HIP",
    "CYX", "CYM",
    "ASH", "GLH",
    "LYN",
})

# Atom names this module tracks.
TRACKED_ATOMS = ("N", "CA", "C", "O", "CB")
# Backbone atoms required by the strict ``keep_incomplete_backbone=False``
# filter. ``OXT`` is accepted as a fallback for the carbonyl O on terminal
# residues (#61) — see ``_parse_atom_record``.
_OXT_FALLBACK_ATOMS = ("O",)
# Extra atom names used when DNA inclusion is enabled (C1' is the closest
# analogue of CA on a nucleotide; P is the closest analogue of the
# backbone N/C carbonyl; the ′ apostrophe variations are tracked because
# different PDB files use different notations).
DNA_TRACKED_ATOMS = ("C1'", "C1*", "P", "O5'", "O5*", "O3'", "O3*")


def _parse_atom_record(
    line: str,
    *,
    keep_altloc_b: bool = False,
    include_dna: bool = False,
) -> dict | None:
    """Parse an ATOM/HETATM line into a dict, or return None if unusable.

    Parameters
    ----------
    line : str
        Raw PDB record (one line).
    keep_altloc_b : bool
        If True, accept altloc 'B' in addition to ''/'A'. Used by the
        ``lammps_compat_altloc`` mode where altloc-B records are emitted
        as duplicated residues (see :func:`parse_pdb`).
    include_dna : bool
        If True, accept DNA residues (DA / DT / DC / DG) and their
        ``C1'`` atom (treated as a CA-proxy). DNA records ALSO use ATOM
        records, but their atom names + resnames are foreign.

    PDB column spec (0-indexed slicing):
        record  0..6
        serial  6..11
        name    12..16
        altLoc  16
        resName 17..20
        chainID 21
        resSeq  22..26
        iCode   26
        x       30..38
        y       38..46
        z       46..54
        occ     54..60
        b       60..66
        element 76..78
    """
    if len(line) < 54:
        return None
    record = line[0:6].strip()
    resname = line[17:20].strip()
    # Finding #9 (2026-05-21): accept HETATM lines whose resname is a known
    # modified amino acid (MSE / SEC / PYL / HID / HIE / HIP / ...). Every
    # other HETATM (waters, ligands, ions) is still ignored.
    if record == "ATOM":
        pass
    elif record == "HETATM" and resname in HETATM_PROMOTED_RESNAMES:
        pass
    else:
        return None
    name = line[12:16].strip()
    is_dna = resname in DNA_RESNAMES
    if is_dna:
        if not include_dna:
            return None
        # For DNA, accept only the C1' atom (the closest CA analogue). We
        # ignore other DNA atoms entirely — they have no AWSEM equivalent.
        if name not in ("C1'", "C1*"):
            return None
    else:
        # Finding #61 (2026-05-21): accept OXT as a fallback for the
        # carbonyl O on terminal residues. We do NOT also promote it when
        # both O and OXT are present — that case keeps both records but
        # only "O" populates the backbone slot (the "OXT" record is
        # silently ignored by the grouping step below).
        if name not in TRACKED_ATOMS and name != "OXT":
            return None
        if resname not in THREE_TO_ONE:
            return None
    altloc = line[16:17].strip()
    accepted_altlocs = ("", "A") if not keep_altloc_b else ("", "A", "B")
    # When the caller did NOT request altloc-B handling, also accept altloc
    # 'B' (and others) at parse time and let the grouping logic decide what
    # to do — this is what makes finding #37 fixable for B-only residues.
    # We tag the record with the original altloc letter so the grouper can
    # still distinguish A vs B vs other.
    if altloc not in accepted_altlocs and altloc not in ("B", "C", "D", "E"):
        return None
    # Finding #38 (2026-05-21): preserve a blank chain ID as the empty
    # string instead of coercing it to "A". The downstream `_build_chain_index`
    # treats distinct strings as distinct chain ints.
    chain_id = line[21:22].strip()  # may be ""
    try:
        res_seq = int(line[22:26].strip())
        icode = line[26:27].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    # Finding #46 (2026-05-21): reject non-finite coords (NaN / Inf).
    # Without this, a `CA x=NaN` row parsed successfully and propagated NaN
    # into every downstream geometric calculation.
    import math as _math
    if not (_math.isfinite(x) and _math.isfinite(y) and _math.isfinite(z)):
        return None
    # Finding #44 (2026-05-21): occupancy. PDB columns 54..60. Used as
    # the tiebreaker when multiple records exist for the same atom name.
    occupancy = 1.0
    if len(line) >= 60:
        occ_str = line[54:60].strip()
        if occ_str:
            try:
                occ_val = float(occ_str)
                if _math.isfinite(occ_val):
                    occupancy = occ_val
            except ValueError:
                pass
    return {
        "name": name,
        "resname": resname,
        "chain": chain_id,
        "resnum": res_seq,
        "icode": icode,
        "altloc": altloc,
        "is_dna": is_dna,
        "xyz": (x, y, z),
        "occupancy": occupancy,
    }


def parse_pdb(
    pdb_path: str | Path,
    *,
    chains: list[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    keep_incomplete_backbone: bool = False,
    include_dna: bool = False,
    lammps_compat_altloc: bool = False,
) -> dict[str, torch.Tensor | list]:
    """Parse a PDB file into PyTorch tensors.

    Parameters
    ----------
    pdb_path : str | Path
        Path to the PDB file on disk.
    chains : list[str] | None
        If provided, restrict to these chain IDs. ``None`` keeps every chain.
    device, dtype
        Standard torch device / dtype controls. Default is CPU / float32 which
        is what every downstream AWSEM term expects.
    keep_incomplete_backbone : bool
        If False (default), drop any residue that lacks ANY of N / CA / C / O.
        Matches the LAMMPS-AWSEM ``PDBToCoordinates.py`` rule (lines 182-191:
        missing-backbone residues are skipped). Useful for byte-comparable
        output against frustrapy on PDBs with truncated terminal residues.
        If True, retain residues that have a CA but lack other backbone
        atoms (those backbone slots are filled with NaN). Pre-2026-05-20
        behaviour.

        Note: when False, on multi-model PDBs only the first model is
        scanned, so residues missing backbone atoms in *that* model are
        dropped. This is what LAMMPS-AWSEM also does.

    include_dna : bool
        Opt-in compat flag. If True, accept DNA residues (DA / DT / DC /
        DG / A / T / C / G / U / DU) as positional residues using their
        ``C1'`` atom as a CA proxy. They are assigned ``residue_type == -1``
        (sentinel for "non-protein placeholder") and ``is_dna == True``.
        Downstream burial / contact / decoy code MUST ignore residues with
        ``residue_type == -1`` (we do this by masking them out before they
        ever reach a gamma lookup).

        **AWSEM frustration on DNA is not physically meaningful**: there
        is no published gamma table for nucleotide contacts, no validated
        burial / water-mediated parameters, and the C1' atom is geometrically
        non-equivalent to CA. This flag exists ONLY for byte-comparable
        parity with LAMMPS-AWSEM + frustratometeR output on protein-DNA
        complexes such as 1O3S (CAP-DNA, PDB 1O3S).

    lammps_compat_altloc : bool
        Opt-in compat flag. If False (default), only altloc ''/A is kept
        (standard convention; matches BioPython's tied-occupancy
        first-added pick). If True, altloc B records are kept too, AND
        inserted as a separate "shadow" residue immediately after their
        altloc-A counterpart. The shadow residue inherits the altloc-A
        chain + resnum (so the resulting residue list has consecutive
        duplicate ``(chain, resnum)`` tuples at the altloc positions).

        The downstream density emitter detects the altloc-B shadow by
        looking at the ``altloc_b_mask`` returned with the coord dict;
        in the LAMMPS-compatible 5adens emission, the shadow residue is
        assigned the next sequential resnum (= altloc-A's resnum + 1),
        and all subsequent rows are labelled with the unmodified PDB
        resnums (i.e. the duplicate row is INSERTED, not the trailing
        rows pushed). This reproduces the consecutive-duplicate-density
        pattern observed in frustratometeR's 5adens.dat on PDBs such
        as 3F9M (alt-conformers at resnums 9, 27, 42, 48, 107, 155, 243).

        Pair-energy + decoy stats are computed on the FULL coord list
        including shadow residues — meaning a homodimer with 7 altloc-B
        records produces 7 extra residue slots in the contact matrix.
        The shadow residues use the altloc-B atom positions where
        available, falling back to the altloc-A coords when no altloc-B
        record was provided for a given atom (this matches what BioPython
        does when a DisorderedAtom is set to the altloc-B child).

    Returns
    -------
    dict with these keys:
        ``ca_coords``   (N, 3) — alpha-carbon positions, angstroms.
        ``n_coords``    (N, 3) — backbone N (NaN where missing, e.g. chain start).
        ``c_coords``    (N, 3) — backbone C (NaN where missing, e.g. chain end).
        ``o_coords``    (N, 3) — backbone O (NaN where missing).
        ``cb_coords``   (N, 3) — beta-carbon (NaN where missing; always NaN for GLY).
        ``residue_types`` (N,) int64 — index 0..19 in OpenAWSEM gamma order;
            ``-1`` for DNA placeholder residues (when ``include_dna=True``).
        ``chain_ids``   list[str], length N.
        ``residue_numbers`` (N,) int64 — author residue numbers from the PDB.
        ``insertion_codes`` list[str], length N — empty string where absent.
        ``is_gly``      (N,) bool — True for glycine (no CB).
        ``is_dna``      (N,) bool — True for DNA placeholder residues (only
            non-zero when ``include_dna=True``).
        ``is_altloc_b_shadow`` (N,) bool — True for altloc-B shadow rows
            (only non-zero when ``lammps_compat_altloc=True``).

    Notes
    -----
    * Coordinates are kept in **angstroms** to match the PDB convention. This
      differs from OpenMM (which uses nm in its expressions). Conversion to nm
      happens at the burial/contact-term boundary, not here.
    * Multi-model PDBs: the first model encountered is used (we stop at
      the first ``ENDMDL`` line). For NMR ensembles run the file through
      ``pdbselect`` first or split the models.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)

    # collect atoms grouped by (chain_label, resnum, icode, altloc_key)
    # preserving file order. When lammps_compat_altloc=True we keep altloc-A
    # and B in SEPARATE groups so we can insert the altloc-B as a shadow.
    residues: list[dict] = []
    res_index: dict[tuple, int] = {}
    # Track every chain ID observed (across atom records that map to a
    # standard / mapped residue) so we can produce a helpful error if a
    # caller-supplied ``chains`` filter excludes everything.
    chains_seen: set = set()

    # Finding #17 (2026-05-21): TER + non-contiguous same-letter chain
    # segments must NOT be merged. We track a per-original-letter segment
    # counter so that the second "A" segment after a TER becomes chain
    # label ``A#2``, the third ``A#3``, etc. The visible label for the
    # first occurrence of each letter is the unsuffixed letter (so no
    # change for well-behaved PDBs).
    chain_segment_counter: dict[str, int] = {}
    # The current segment label per original chain letter, refreshed
    # whenever we see a TER record for that chain.
    chain_current_segment: dict[str, str] = {}
    # Set of letters retired by the most recent TER record. We only
    # advance a chain's segment number the first time we see an ATOM
    # record from that letter AFTER a TER (so a TER followed by an ATOM
    # for a different letter doesn't trip the counter).
    pending_segment_advance: set = set()

    def _segment_label(orig_letter: str) -> str:
        """Return the (possibly suffixed) chain label for the CURRENT
        segment of ``orig_letter``. Advances the segment counter once
        per post-TER first-seen ATOM."""
        if orig_letter in pending_segment_advance:
            pending_segment_advance.discard(orig_letter)
            n = chain_segment_counter.get(orig_letter, 1) + 1
            chain_segment_counter[orig_letter] = n
            chain_current_segment[orig_letter] = f"{orig_letter}#{n}"
        elif orig_letter not in chain_current_segment:
            chain_segment_counter[orig_letter] = 1
            chain_current_segment[orig_letter] = orig_letter
        return chain_current_segment[orig_letter]

    # Finding #60 (2026-05-21): MODEL/ENDMDL discipline. If MODEL records
    # appear, we keep only the first model. Two cases:
    #  - MODEL N ... ENDMDL ... MODEL N+1: stop at ENDMDL (existing behaviour).
    #  - MODEL N ... MODEL N+1 (missing ENDMDL): stop at second MODEL.
    seen_model = False

    with pdb_path.open("r") as fh:
        for line in fh:
            # Finding #60 (2026-05-21): stop on second MODEL even if no
            # ENDMDL appeared between them.
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if line.startswith("ENDMDL"):  # stop after first model
                break
            # Finding #45 (2026-05-21): END terminates coordinate parsing.
            # Distinct from ENDMDL — atoms after END must be ignored.
            if line.startswith("END") and not line.startswith("ENDMDL"):
                break
            if line.startswith("TER"):
                # Finding #17 (2026-05-21): a TER means the next ATOM
                # records for ANY chain letter previously seen are a
                # new segment. We mark every letter we've seen so far
                # as pending; whichever one shows up next is bumped.
                for k in list(chain_current_segment.keys()):
                    pending_segment_advance.add(k)
                continue
            rec = _parse_atom_record(
                line,
                keep_altloc_b=lammps_compat_altloc,
                include_dna=include_dna,
            )
            if rec is None:
                continue
            orig_chain = rec["chain"]
            # Determine the visible (post-TER-aware) chain label.
            chain_label = _segment_label(orig_chain)
            chains_seen.add(chain_label)
            # When the user passes ``chains=['A']`` and the file contains
            # both segment ``A`` and ``A#2`` after a TER, treat both as
            # matching — the user's literal letter is what they typed.
            if chains is not None and (
                chain_label not in chains and orig_chain not in chains
            ):
                continue
            altloc = rec["altloc"]
            # Default mode: only B-as-shadow when lammps_compat_altloc=True.
            # Otherwise we KEEP altloc-B records here too (tag them as 'B')
            # because finding #37 needs us to keep an altloc-B record when
            # no altloc-A exists for that residue. The B records are
            # silently dropped at the end of grouping IF a corresponding
            # altloc-A record exists.
            altloc_key = "B" if (lammps_compat_altloc and altloc == "B") else "A"
            key = (chain_label, rec["resnum"], rec["icode"], altloc_key)
            if key not in res_index:
                res_index[key] = len(residues)
                residues.append({
                    "chain": chain_label,
                    "resnum": rec["resnum"],
                    "icode": rec["icode"],
                    "resname": rec["resname"],
                    "altloc_key": altloc_key,
                    "is_dna": rec["is_dna"],
                    "atoms": {},
                    # observed altlocs in this residue (used by #37 fallback)
                    "_altlocs_seen": set(),
                })
            ri = res_index[key]
            r = residues[ri]
            r["_altlocs_seen"].add(altloc)
            # Finding #53 (2026-05-21): refuse to merge two different resnames
            # at the same (chain, resnum, icode) key. ALA + GLY at A:1 was
            # silently merged into one ALA residue (the first resname seen).
            if not rec["is_dna"] and r["resname"] != rec["resname"]:
                raise ValueError(
                    f"PDB {pdb_path.name} has conflicting residue names at "
                    f"{chain_label}:{rec['resnum']}{rec['icode'] or ''} "
                    f"({r['resname']} and {rec['resname']}). Microheterogeneous "
                    f"residues must use altloc to disambiguate."
                )
            # Finding #44 (2026-05-21): pick the record with the highest
            # occupancy when multiple records exist for the same atom name
            # within a residue. The atom dict value is now a (xyz, occ) tuple
            # internally; we keep the legacy "atoms[name] = xyz" shape after
            # the grouping pass completes.
            atom_name = rec["name"]
            existing = r["atoms"].get(atom_name)
            if existing is None:
                r["atoms"][atom_name] = (rec["xyz"], rec["occupancy"], altloc)
            else:
                _, existing_occ, existing_altloc = existing
                # Same altloc letter -> use the higher occupancy.
                # Different altloc (B vs A) -> let the altloc logic win.
                if altloc == existing_altloc:
                    if rec["occupancy"] > existing_occ:
                        r["atoms"][atom_name] = (
                            rec["xyz"], rec["occupancy"], altloc
                        )

    # Finding #37 (2026-05-21): handle altloc-B-only residues. In default
    # mode (`lammps_compat_altloc=False`) we kept altloc-B atom records
    # grouped into an "A" key. We now finalize: for each residue,
    # if every atom we collected is altloc 'B' and no altloc 'A' was
    # seen, that's an altloc-B-only residue — keep it. Otherwise filter
    # out the altloc-B atoms (consistent with the previous "A-only"
    # default). In lammps_compat_altloc mode altloc-B atoms live in a
    # separate group keyed by altloc_key == 'B', so this filter only
    # touches the 'A' groups.
    for r in residues:
        if r["altloc_key"] != "A":
            continue
        seen = r["_altlocs_seen"]
        b_only = (not ({"", "A"} & seen)) and ("B" in seen)
        if b_only:
            # Keep the B records — this is the altloc-B-only case.
            continue
        # Otherwise drop any atom whose stored altloc is 'B' (because a
        # better altloc-'' or 'A' record must exist for the residue, even
        # if not for this exact atom name — defensive).
        r["atoms"] = {
            name: tup for name, tup in r["atoms"].items()
            if tup[2] in ("", "A")
        } or r["atoms"]  # if filtering empties the dict, keep what we had

    # Finding #61 (2026-05-21): promote OXT to O when the residue is
    # missing the carbonyl O but has OXT (terminal-O fallback).
    for r in residues:
        if "O" not in r["atoms"] and "OXT" in r["atoms"]:
            r["atoms"]["O"] = r["atoms"]["OXT"]

    # Strip OXT from the atoms dict (it was only useful as a fallback
    # for O — it has no slot in the output tensors).
    for r in residues:
        r["atoms"].pop("OXT", None)

    # Flatten the (xyz, occ, altloc) tuples back to plain xyz tuples
    # for the rest of the pipeline.
    for r in residues:
        r["atoms"] = {
            name: tup[0] if isinstance(tup, tuple) and len(tup) == 3 else tup
            for name, tup in r["atoms"].items()
        }
        # remove the diagnostic set so downstream code doesn't trip on it
        r.pop("_altlocs_seen", None)

    # In lammps_compat_altloc mode, weave the altloc-B groups in
    # immediately after the matching altloc-A residue, AND fill in any
    # backbone atoms the B record didn't provide from the matching A
    # record. Most altloc-B records in real PDBs only re-position the
    # side chain (CB onward); N / C / O come from the shared backbone
    # of the altloc-A record. Without this inheritance the strict
    # backbone filter below would drop every B shadow.
    if lammps_compat_altloc:
        # Finding #37 (2026-05-21): re-tag altloc-B residues with NO
        # matching altloc-A as primary (altloc_key="A") so they aren't
        # treated as shadows. Without this they'd index a stale eq_list
        # entry in _build_lammps_emit_rows or hit IndexError.
        _a_keys = {
            (r["chain"], r["resnum"], r["icode"])
            for r in residues if r["altloc_key"] == "A"
        }
        for r in residues:
            if (
                r["altloc_key"] == "B"
                and (r["chain"], r["resnum"], r["icode"]) not in _a_keys
            ):
                r["altloc_key"] = "A"
        residues = _weave_altloc_b_shadows(residues)
        _inherit_backbone_to_altloc_b(residues)

    # DNA placeholder residues use C1' as CA. Hoist it into the "CA" slot
    # of the atoms dict so the rest of the function treats them uniformly.
    if include_dna:
        for r in residues:
            if r["is_dna"]:
                ca_proxy = r["atoms"].get("C1'") or r["atoms"].get("C1*")
                if ca_proxy is not None:
                    r["atoms"].setdefault("CA", ca_proxy)

    # Drop residues without a CA (only useful AWSEM residues survive).
    residues = [r for r in residues if "CA" in r["atoms"]]

    # Apply the strict-backbone filter, but skip it for DNA placeholders
    # (they only ever have a CA-proxy).
    if not keep_incomplete_backbone:
        residues = [
            r for r in residues
            if r["is_dna"] or all(
                a in r["atoms"] for a in ("N", "CA", "C", "O")
            )
        ]

    if not residues:
        if chains is not None and chains_seen:
            requested = list(chains) if not isinstance(chains, str) else [chains]
            missing = [c for c in requested if c not in chains_seen]
            available = sorted(chains_seen)
            if missing:
                raise ValueError(
                    f"chain {missing!r} not found in {pdb_path}; "
                    f"available chains: {available}"
                )
        raise ValueError(f"No usable residues parsed from {pdb_path}")

    # Warn once per call when we silently coerced non-standard residues to
    # their canonical AA (SEC -> C, MSE -> M, HID/HIE/HIP -> H, etc.). This
    # makes the implicit substitution visible to users who would otherwise
    # be surprised by the gamma-table identity used downstream.
    nonstandard_seen: dict[str, str] = {}
    for r in residues:
        if r["is_dna"]:
            continue
        rn = r["resname"]
        if rn not in STANDARD_AA_3LETTERS and rn in THREE_TO_ONE:
            nonstandard_seen.setdefault(rn, THREE_TO_ONE[rn])
    if nonstandard_seen:
        import warnings as _w
        _details = ", ".join(f"{k}->{v}" for k, v in sorted(nonstandard_seen.items()))
        _w.warn(
            f"non-standard residues in {pdb_path.name} mapped to canonical "
            f"AAs: {_details}. AWSEM uses the canonical gamma table for "
            f"these positions; pass the modified structure through PDBFixer "
            f"if you need different handling.",
            UserWarning,
            stacklevel=2,
        )

    n_res = len(residues)
    nan = float("nan")
    ca = torch.empty((n_res, 3), dtype=dtype)
    n_ = torch.full((n_res, 3), nan, dtype=dtype)
    c_ = torch.full((n_res, 3), nan, dtype=dtype)
    o_ = torch.full((n_res, 3), nan, dtype=dtype)
    cb = torch.full((n_res, 3), nan, dtype=dtype)
    rtypes = torch.empty(n_res, dtype=torch.int64)
    resnums = torch.empty(n_res, dtype=torch.int64)
    is_gly = torch.zeros(n_res, dtype=torch.bool)
    is_dna = torch.zeros(n_res, dtype=torch.bool)
    is_altb = torch.zeros(n_res, dtype=torch.bool)
    chain_ids: list[str] = []
    icodes: list[str] = []

    for i, r in enumerate(residues):
        a = r["atoms"]
        ca[i] = torch.tensor(a["CA"], dtype=dtype)
        if "N" in a:
            n_[i] = torch.tensor(a["N"], dtype=dtype)
        if "C" in a:
            c_[i] = torch.tensor(a["C"], dtype=dtype)
        if "O" in a:
            o_[i] = torch.tensor(a["O"], dtype=dtype)
        if r["is_dna"]:
            # Sentinel index for non-protein placeholder rows.
            rtypes[i] = -1
            is_dna[i] = True
            # No CB; AWSEM gamma lookups must skip these.
        else:
            one = THREE_TO_ONE[r["resname"]]
            rtypes[i] = ONE_TO_IDX[one]
            is_gly[i] = one == "G"
            if "CB" in a and not is_gly[i]:
                cb[i] = torch.tensor(a["CB"], dtype=dtype)
        if r.get("altloc_key", "A") == "B":
            is_altb[i] = True
        resnums[i] = r["resnum"]
        chain_ids.append(r["chain"])
        icodes.append(r["icode"])

    # Build the LAMMPS-compatible 5adens emission rows. Each entry is
    # a triple (chain_label, resnum_label, math_protein_idx):
    #   - When include_dna / lammps_compat_altloc are both False, this
    #     reduces to the identity mapping (chain_ids[i], resnums[i], i)
    #     for protein residues only — same as legacy behaviour.
    #   - When include_dna=True, DNA rows come FIRST (file-order) and
    #     borrow the first N_dna protein math_idx values. This reproduces
    #     frustratometeR's zip-truncation bug on protein-DNA complexes.
    #   - When lammps_compat_altloc=True, altloc-B rows inherit the
    #     ALTLOC-A residue's resnum + 1 (the "next available resnum"
    #     trick) and re-use the altloc-A math_idx — so the emitted row
    #     has the same density value as the preceding (altloc-A) row,
    #     and the next row (originally the following residue) gets
    #     "shifted" one math_idx earlier.
    lammps_emit_rows = _build_lammps_emit_rows(residues, is_dna, is_altb)

    return {
        "ca_coords": ca.to(device),
        "n_coords": n_.to(device),
        "c_coords": c_.to(device),
        "o_coords": o_.to(device),
        "cb_coords": cb.to(device),
        "residue_types": rtypes.to(device),
        "chain_ids": chain_ids,
        "residue_numbers": resnums.to(device),
        "insertion_codes": icodes,
        "is_gly": is_gly.to(device),
        "is_dna": is_dna.to(device),
        "is_altloc_b_shadow": is_altb.to(device),
        "lammps_emit_rows": lammps_emit_rows,
    }


def _build_lammps_emit_rows(
    residues: list[dict],
    is_dna: torch.Tensor,
    is_altb: torch.Tensor,
) -> list[tuple]:
    """Build the 5adens-emission row list that reproduces frustratometeR's
    LAMMPS-compatible output pattern (a zip over equivalences and the
    file-order CA list, where altloc-B CAs are inserted in-line in the
    CA list but NOT in the equivalences list).

    Empirically derived against ``benchmark/cpu_baseline/configurational/
    {1O3S,3F9M}_5adens.dat`` — see the FrustrationGPU memo
    ``docs/lammps_compat_fixes.md`` for the trace-through that confirms
    this model matches the dump's duplicate-density pattern.

    Returns
    -------
    list of (chain_label : str, resnum_label : int, math_protein_idx : int)
        One tuple per output row. ``math_protein_idx`` is the index into
        the SUBSET coord dict (protein-only) that the orchestrator uses
        as the sphere center for density. The list length is the input
        full-list length; the orchestrator applies the zip-cut to
        ``min(N_protein, len(emit_rows))`` at actual emission time.

    Algorithm
    ---------
    Two parallel cursors:

    * ``full_idx`` walks the residues list (including altloc-B shadows
      and DNA placeholder entries). It advances on every entry.
    * ``eq_idx`` walks the "equivalences" stream — unique
      ``(chain, resnum)`` tuples in PDB order. It advances on every
      entry EXCEPT altloc-B shadows (which share the resnum of their
      altloc-A counterpart and so are not new equivalences rows).
    * ``math_idx`` walks the protein-only math view. It advances on
      every entry EXCEPT altloc-B shadows AND DNA placeholders (those
      are excluded from the math subset).

    For each ``full_idx`` we emit one output row:
      * label = current ``(chain, resnum)`` taken from eq (a virtual
        equivalences value at position ``eq_idx``)
      * density = math_protein[ ``math_idx`` ]

    Note that the iteration emits ONE row per FULL-list entry, but the
    orchestrator caps the emission at ``N_protein`` (matching the zip
    truncation behaviour in frustratometeR's source).
    """
    rows: list[tuple] = []
    math_idx = 0
    eq_idx = 0
    # Pre-build the eq list — a unique (chain, resnum) per protein /
    # DNA residue in PDB-file order (skip altloc-B shadows).
    eq_list: list[tuple] = []
    for i, r in enumerate(residues):
        is_altb_i = bool(is_altb[i].item())
        if is_altb_i:
            continue
        eq_list.append((r["chain"], r["resnum"]))
    n_eq = len(eq_list)
    for i in range(len(residues)):
        is_dna_i = bool(is_dna[i].item())
        is_altb_i = bool(is_altb[i].item())
        # Cap eq_idx defensively — past the end means "fall off the zip"
        # which the orchestrator's truncation handles at write time.
        eq_idx_safe = min(eq_idx, n_eq - 1)
        if is_altb_i:
            # Re-use the previously emitted math_idx (the altloc-A's
            # math_idx); label = the NEXT eq entry (matching the
            # frustratometeR pattern: altloc-B "steals" the label of
            # the residue immediately AFTER its altloc-A in the
            # equivalences stream, while keeping the density of the
            # altloc-A position).
            altb_label = eq_list[eq_idx_safe]
            rows.append((altb_label[0], altb_label[1], max(0, math_idx - 1)))
            eq_idx += 1
            # math_idx does NOT advance — the altloc-B row reuses the
            # math density of its altloc-A neighbour.
        elif is_dna_i:
            # DNA placeholder: label from eq stream; math_idx points
            # into the protein math (this row "borrows" the math_idx-th
            # protein CA, mirroring frustratometeR's zip-of-mismatched-
            # lengths bug on protein-DNA complexes).
            label = eq_list[eq_idx_safe]
            rows.append((label[0], label[1], math_idx))
            eq_idx += 1
            math_idx += 1  # DNA still consumes one "slot" in the ca_xyz cursor
        else:
            # Plain protein altloc-A: label = this residue's eq, math
            # advances.
            label = eq_list[eq_idx_safe]
            rows.append((label[0], label[1], math_idx))
            eq_idx += 1
            math_idx += 1
    return rows


def _inherit_backbone_to_altloc_b(residues: list[dict]) -> None:
    """For each altloc-B residue, fill any missing N/CA/C/O/CB atom from
    its matching altloc-A counterpart.

    This mirrors BioPython's behaviour when `DisorderedAtom.disordered_select("B")`
    is called: only the atoms with an altloc-B record switch; everything
    else stays at the altloc-A coords. Crucially, the strict backbone
    filter `keep_incomplete_backbone=False` would otherwise drop every
    B shadow (PDB altloc-B records typically omit the shared N/C/O).
    """
    # Build a map of altloc-A residues for O(1) lookup. Mutates in place.
    a_lookup: dict[tuple, dict] = {}
    for r in residues:
        if r["altloc_key"] == "A":
            a_lookup[(r["chain"], r["resnum"], r["icode"])] = r
    for r in residues:
        if r["altloc_key"] != "B":
            continue
        key = (r["chain"], r["resnum"], r["icode"])
        a = a_lookup.get(key)
        if a is None:
            continue
        # Inherit any backbone atom not present in B from A.
        for atom_name in ("N", "CA", "C", "O", "CB"):
            if atom_name not in r["atoms"] and atom_name in a["atoms"]:
                r["atoms"][atom_name] = a["atoms"][atom_name]


def _weave_altloc_b_shadows(residues: list[dict]) -> list[dict]:
    """Re-order residues so each altloc-B group is inserted right after its
    matching altloc-A group.

    The naive dict-based collection above can leave altloc-B groups at the
    end of `residues` when their atom records appear in a separate block
    of the PDB file (rare; normally PDB-A and PDB-B are adjacent). This
    helper makes the order deterministic and matches the LAMMPS-AWSEM
    PDBToCoordinates iteration order when both altlocs are kept.

    For each altloc-B residue:
      * Find the index in `residues` where the matching altloc-A entry
        sits (same chain + resnum + icode).
      * Splice the B group in immediately after that A entry.
    If no matching A is found, the B residue stays in its original
    position (defensive: shouldn't happen on well-formed PDBs).
    """
    a_index: dict[tuple, int] = {}
    out: list[dict] = []
    for r in residues:
        if r["altloc_key"] == "A":
            a_index[(r["chain"], r["resnum"], r["icode"])] = len(out)
            out.append(r)
        # else: defer; we'll splice it in below.
    # Now insert B residues right after their matching A entry. Walk in
    # the original order to preserve B-after-B ordering for residues
    # carrying multiple altlocs (rare but possible).
    inserts: list[tuple] = []  # (insert_after_idx_in_out, b_residue)
    for r in residues:
        if r["altloc_key"] != "B":
            continue
        key = (r["chain"], r["resnum"], r["icode"])
        if key in a_index:
            inserts.append((a_index[key], r))
        else:
            # Defensive fallback: append at end as its own row.
            inserts.append((len(out) - 1, r))
    # Insert from the back so earlier indices don't shift.
    # Sort by insertion point descending, then by sequence (B-records
    # appear in PDB file order — keep that order on insertion).
    # We process in original PDB order but insert from the back; this
    # preserves the natural B1-then-B2 ordering for residues with two
    # altloc-B entries.
    inserts.sort(key=lambda t: t[0], reverse=True)
    for after_idx, r in inserts:
        out.insert(after_idx + 1, r)
    return out


def chain_segments(chain_ids: list[str]) -> list[tuple]:
    """Return a list of (start, end) index ranges per chain (end exclusive).

    Used by the burial and contact terms to forbid cross-chain ``rho`` neighbours
    when computing the |i - j| > 2 sequence-separation rule (sequence separation
    is intra-chain only)."""
    segs = []
    if not chain_ids:
        return segs
    start = 0
    cur = chain_ids[0]
    for i, c in enumerate(chain_ids[1:], start=1):
        if c != cur:
            segs.append((start, i))
            start = i
            cur = c
    segs.append((start, len(chain_ids)))
    return segs
