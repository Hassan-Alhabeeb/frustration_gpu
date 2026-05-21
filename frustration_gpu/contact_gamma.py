"""Direct-contact gamma table loader.

Phase 2a parameter loader. Reads the *direct contact* block of
``src/data/gamma.dat`` and returns a symmetric (20, 20) ``γ_direct[aa_i, aa_j]``
tensor in the AWSEM AA order ``A R N D C Q E G H I L K M F P S T W Y V``.

Why a separate loader (parameters.py already has ``load_gamma_tables``)?
-----------------------------------------------------------------------
``parameters.load_gamma_tables`` returns the *raw* gamma values from the file,
suitable for both the direct and mediated terms. The C++ formula
(``fix_backbone.cpp:626-639``) **pre-multiplies the loaded gamma by ``k_water``
at load time**, then uses ``-(γ × θ)`` in the energy expression (no further
prefactor — see ``fix_backbone.cpp:5473``).

We want our direct-contact module to be self-contained and obvious to read,
and we want a single place to document the gamma.dat layout quirks. So this
file:

* Wraps ``load_gamma_tables`` to return only the direct block, and
* Documents the "two identical columns" quirk for posterity.

Layout reminder (from ``parameters.py`` docstring and the C++ loader):

* Rows 0-209 of ``gamma.dat`` = direct contact, upper-triangle iteration order
  ``for i in 0..19: for j in i..19``. Both columns hold the *same* number for
  direct entries — the LAMMPS API just happens to read two columns. The C++
  loader stores ``γ[0]`` and ``γ[1]`` separately then averages them
  (``sigma_gamma_direct = (γ[0] + γ[1]) / 2``); since both numbers are equal
  this is the identity. We sidestep the averaging and use column 0 directly.
* Rows 210-419 = mediated contact (NOT used in Phase 2a). Handled in
  ``parameters.py`` for Phase 2b.

C++ formula reference (``fix_backbone.cpp:5462-5473``)::

    sigma_gamma_direct = (water_gamma_0_direct + water_gamma_1_direct) / 2;
    theta_direct       = 0.25 * (1 + tanh(k*(r - r_min))) * (1 + tanh(k*(r_max - r)));
    water_energy       = -(sigma_gamma_direct * theta_direct + ...);

with ``water_gamma`` pre-multiplied by ``k_water`` at load. So in our notation,
where ``γ_direct[aa_i, aa_j]`` is the *raw* gamma (no k_water folded in):

    V_direct(i, j) = -k_water * γ_direct[aa_i, aa_j] * θ_direct(r_ij)

Note: no factor of 1/2 in the final expression — that was a transcription
error in the original Phase 2a prompt. The "1/2" in the C++ comes from
averaging two identical columns, which collapses to the identity. Verified
against ``5AON_tertiary_frustration.dat`` row 1 (i=1, j=3, S-R, r=5.065,
ρ_i=0.000, ρ_j=0.304): V_water + E_burial(i) + E_burial(j) = -1.003 matches
the dump's ``E_native = -1.003`` only when the "no 1/2" form is used.

Pure PyTorch, single function. ~70 LOC plus this docstring.
"""
from __future__ import annotations

import torch

from .parameters import load_gamma_tables


def load_direct_gamma(
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the direct-contact gamma table as a symmetric (20, 20) tensor.

    Parameters
    ----------
    device : torch device or string, optional
        Destination device. Defaults to CPU.
    dtype : torch.dtype
        Tensor dtype. Defaults to ``float32``. Use ``float64`` for the
        numerical-parity tests against the LAMMPS dump.

    Returns
    -------
    γ_direct : (20, 20) tensor
        ``γ_direct[aa_i, aa_j]`` in AA index order
        ``A R N D C Q E G H I L K M F P S T W Y V``. Symmetric:
        ``γ_direct[i, j] == γ_direct[j, i]``.

    Notes
    -----
    * No ``k_water`` multiplication is applied here — the energy formula in
      :func:`direct_contact_energy` carries the ``k_water`` factor explicitly.
      This matches the user-facing convention (separate ``k_water`` knob) and
      differs from the C++ which folds ``k_water`` into the loaded gamma.
      Numerically identical when ``k_water = 1.0`` (the LAMMPS default), which
      is the only value frustrapy ever uses.
    * Values are read at full file precision (six significant digits, e.g.
      ``-0.33109``). For CPU/GPU agreement to better than 1e-6 use
      ``dtype=torch.float64``.
    """
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    tables = load_gamma_tables(device=target_device, dtype=dtype)
    return tables.direct


def load_mediated_gamma(
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mediated-contact gamma tables (protein, water).

    Rows 210-419 of ``gamma.dat`` contain the mediated contact entries — TWO
    distinct columns (unlike the direct block, whose two columns are
    duplicates):

    * Column 0 → ``γ_med_protein[aa_i, aa_j]`` (protein-mediated).
    * Column 1 → ``γ_med_water[aa_i, aa_j]`` (water-mediated).

    Iteration order: ``for i in 0..19: for j in i..19`` — the same upper-triangle
    walk used by the direct block (``contactTerms.py:82-98``).

    Parameters
    ----------
    device : torch device or string, optional
        Destination device. Defaults to CPU.
    dtype : torch.dtype
        Tensor dtype. Defaults to ``float32``. Use ``float64`` for the
        numerical-parity tests against the LAMMPS dump.

    Returns
    -------
    (gamma_mediated_protein, gamma_mediated_water) : tuple of (20, 20) tensors
        Both symmetric. AA index order
        ``A R N D C Q E G H I L K M F P S T W Y V``.

    Notes
    -----
    * No ``k_water`` multiplication is applied here — same convention as
      :func:`load_direct_gamma`. The user-facing :func:`water_mediated_energy`
      carries ``k_water`` as a separate runtime knob.
    * Values read at full file precision (six significant digits).
    """
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    tables = load_gamma_tables(device=target_device, dtype=dtype)
    return tables.mediated_protein, tables.mediated_water
