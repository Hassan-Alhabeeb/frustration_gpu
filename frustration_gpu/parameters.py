"""Loaders for the AWSEM parameter tables (``gamma.dat``, ``burial_gamma.dat``).

The parameter files were copied byte-exactly from the OpenAWSEM install at
``site-packages/openawsem/parameters/`` into ``src/data/``. Md5 verified at
copy time (see ``PHASE_1_STATUS.md``). The loaders here only read text — no
runtime dependency on the openawsem package.

Layout of ``gamma.dat``
-----------------------
Plain whitespace-numeric file, two columns. OpenAWSEM's ``read_gamma`` slices
the first 210 rows as the ``direct`` block and the rest as the ``mediated``
block. 210 = C(20+1, 2) = the number of unordered pairs over the 20 amino
acids (i.e. the upper triangle of a 20x20 symmetric matrix, including the
diagonal).

Within each block, the iteration order in ``contactTerms.py`` (lines 82-98) is::

    count = 0
    for i in range(20):
        for j in range(i, 20):
            table[i][j] = data[count]
            table[j][i] = data[count]
            count += 1

so ``gamma_direct[count]`` corresponds to the unordered pair ``(i, j)`` with
``i <= j`` in OpenAWSEM AA order ``A R N D C Q E G H I L K M F P S T W Y V``.
The mediated block has two columns: column 0 is protein-mediated, column 1 is
water-mediated.

Layout of ``burial_gamma.dat``
------------------------------
20 rows x 3 columns. Each row is one amino acid (same A-V order); each column
is one of the three burial wells (low/medium/high CB density). Verified by
inspecting the file: 20 lines, 3 floats each.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch

# Default location: F:/research_plan/frustration_gpu/src/data/
_DATA_DIR = Path(__file__).resolve().parent / "data"

# Burial constants from contactTerms.py lines 71-73. Units: dimensionless
# (rho is a count of nearby CB atoms after sigmoid weighting).
BURIAL_KAPPA: float = 4.0
BURIAL_RHO_MIN: tuple = (0.0, 3.0, 6.0)
BURIAL_RHO_MAX: tuple = (3.0, 6.0, 9.0)

# rho-computation constants. Units: nanometers — these match the OpenMM
# expression strings in contactTerms.py. The PDB parser keeps coordinates in
# angstroms, so the burial module multiplies by 0.1 at the boundary.
RHO_R_MIN_NM: float = 0.45        # 4.5 angstroms
RHO_R_MAX_NM: float = 0.65        # 6.5 angstroms
RHO_ETA_PER_NM: float = 50.0      # nm^-1 (== 5.0 / angstrom)
RHO_MIN_SEQ_SEP: int = 1          # |i - j| must be > 1 (excludes self + ±1 only). Matches LAMMPS-AWSEM smart_matrix_lib.h:638 (Opus C++ audit 2026-05-20). Was 2 in Phase 1 — off-by-one fix.


class GammaTables(NamedTuple):
    """Holder for the contact gamma tables.

    Shapes:
        direct      (20, 20) — symmetric, sym(g_direct).
        mediated_protein  (20, 20) — symmetric.
        mediated_water    (20, 20) — symmetric.
    """
    direct: torch.Tensor
    mediated_protein: torch.Tensor
    mediated_water: torch.Tensor


def _gamma_dat_path() -> Path:
    return _DATA_DIR / "gamma.dat"


def _burial_gamma_dat_path() -> Path:
    return _DATA_DIR / "burial_gamma.dat"


def load_burial_gamma(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the burial gamma table as a (20, 3) tensor.

    Rows: amino acid index in OpenAWSEM gamma order (A=0 ... V=19).
    Columns: burial well index 0=low / 1=med / 2=high.

    Reads ``src/data/burial_gamma.dat`` line by line; no dependency on
    NumPy loadtxt semantics. Values are kept at the precision written in the
    file (six significant digits, e.g. ``0.839477E+00``).
    """
    path = _burial_gamma_dat_path()
    rows: list[list[float]] = []
    with path.open("r") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"burial_gamma.dat:{lineno}: expected 3 columns, got "
                    f"{len(parts)} in line: {line!r}"
                )
            row = [float(x) for x in parts]
            # QA-MISC #49: reject NaN / inf in parameter values. A typo'd
            # parameter file would otherwise propagate silently through the
            # burial energy as ``NaN`` or ``inf``.
            for v in row:
                if not (v == v) or v in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"burial_gamma.dat:{lineno}: non-finite value {v!r} "
                        "in parameter table"
                    )
            rows.append(row)
    if len(rows) != 20:
        raise ValueError(f"burial_gamma.dat: expected 20 rows, got {len(rows)}")
    return torch.tensor(rows, dtype=dtype, device=device)


def load_gamma_tables(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GammaTables:
    """Return symmetric (20, 20) gamma tables for direct + mediated contacts.

    The two halves of ``gamma.dat`` are decoded with the same iteration order
    OpenAWSEM uses (i in range(20), j in range(i, 20)). The direct block uses
    column 0; the mediated block keeps both columns (col 0 = protein-mediated,
    col 1 = water-mediated).

    Returned tensors are symmetric (g[i, j] == g[j, i]).
    """
    path = _gamma_dat_path()
    raw: list[list[float]] = []
    # QA-MISC #32: previous loader accepted 1-column lines (silently duplicating
    # the value across both columns) and accepted >420-row files (silently
    # ignoring the extras). Both modes hid file corruption. Be strict:
    # require exactly 2 numeric columns per non-blank line and exactly 420
    # data rows total.
    with path.open("r") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(
                    f"gamma.dat:{lineno}: expected 2 columns, got {len(parts)} "
                    f"in line: {line!r}"
                )
            row = [float(parts[0]), float(parts[1])]
            # QA-MISC #49: reject NaN / inf.
            for v in row:
                if not (v == v) or v in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"gamma.dat:{lineno}: non-finite value {v!r} in "
                        "parameter table"
                    )
            raw.append(row)
    if len(raw) != 420:
        raise ValueError(
            f"gamma.dat: expected exactly 420 data rows, got {len(raw)}"
        )

    direct_block = raw[:210]
    mediated_block = raw[210:420]

    direct = torch.zeros((20, 20), dtype=dtype, device=device)
    med_protein = torch.zeros((20, 20), dtype=dtype, device=device)
    med_water = torch.zeros((20, 20), dtype=dtype, device=device)

    count = 0
    for i in range(20):
        for j in range(i, 20):
            v = direct_block[count][0]
            direct[i, j] = v
            direct[j, i] = v
            count += 1

    count = 0
    for i in range(20):
        for j in range(i, 20):
            v_p = mediated_block[count][0]
            v_w = mediated_block[count][1]
            med_protein[i, j] = v_p
            med_protein[j, i] = v_p
            med_water[i, j] = v_w
            med_water[j, i] = v_w
            count += 1

    return GammaTables(direct=direct, mediated_protein=med_protein, mediated_water=med_water)
