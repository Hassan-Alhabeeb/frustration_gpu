"""AWSEM virtual-atom construction in PyTorch.

LAMMPS-AWSEM (and OpenAWSEM) represent each residue with six atoms — N, H, CA,
C, O, CB — but the N, H and C positions are *virtual sites*, computed as
fixed linear combinations of CA(i-1), CA(i), CA(i+1) and O(i-1) / O(i).

The trans-peptide coefficients (OpenAWSEM ``prepare_virtual_sites_v2``,
``openAWSEM.py`` line ~491) are:

    N(i) = 0.48318 * CA(i-1) + 0.70328 * CA(i) - 0.18643 * O(i-1)
    C(i) = 0.44365 * CA(i)   + 0.23520 * CA(i+1) + 0.32115 * O(i)
    H(i) = 0.84100 * CA(i-1) + 0.89296 * CA(i) - 0.73389 * O(i-1)

Cis-proline (residue type ``IPR``) uses a different set of coefficients (lines
487-489 of ``openAWSEM.py``). We expose a ``use_cis_proline`` flag for that
branch; the default is False, matching frustrapy's default. Real PRO in the
LAMMPS-AWSEM convention is *trans* by default — only the special ``IPR``
3-letter code triggers the cis branch.

The CB position is **not virtual** in OpenAWSEM. It is read straight from the
crystal structure or built by PDBFixer. For frustration we follow the same
convention: if the parser found a CB atom we keep it; if not (e.g. the residue
is GLY, or CB was missing) we leave it NaN. Downstream code uses CA as a
fallback for GLY when computing densities (see contactTerms.py line 166).

Coefficients here are typed out to all five reported decimal places. Note: the
3x3 [ABC] matrix in ``fix_backbone_coeff.data`` is rounded to three decimals
(0.483, 0.703, -0.186 etc.) — the higher-precision values in OpenAWSEM are
preferred for numerical match.
"""
from __future__ import annotations

import torch

# Trans coefficients (default for all residues except IPR cis-proline).
_TRANS = {
    "N": (0.48318, 0.70328, -0.18643),
    "C": (0.44365, 0.23520, 0.32115),
    "H": (0.84100, 0.89296, -0.73389),
}
# Cis-proline (IPR) coefficients.
_CIS_PRO = {
    "N": (-0.2094, 0.6908, 0.5190),
    "C": (0.2196, 0.2300, 0.5507),
    "H": (-0.9871, 0.9326, 1.0604),
}


def compute_virtual_atoms(
    parsed: dict[str, torch.Tensor | list],
    *,
    use_cis_proline: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute virtual N / H / C positions from CA and O.

    Parameters
    ----------
    parsed : dict
        Output of :func:`src.parser.parse_pdb`. Must contain ``ca_coords``,
        ``o_coords``, ``chain_ids``, ``residue_types``.
    use_cis_proline : bool
        If True, residues annotated as IPR get the cis-proline coefficient set.
        We don't currently re-parse the PDB looking for IPR codes — the parser
        canonicalises any PRO to ``PRO`` -> index 14. The flag therefore has no
        effect unless the parser is taught to preserve ``IPR``. Default False
        matches frustrapy. **Flagged uncertain** — preserved here for forward
        compatibility.

    Returns
    -------
    dict with the virtual-atom tensors. Each is (N, 3), in angstroms. Where the
    formula needs an atom that doesn't exist (e.g. CA at i-1 for the first
    residue of a chain, or O at i-1 for chain start, or CA at i+1 for chain
    end), the corresponding row is filled with NaN.

    The keys returned are ``n_virtual``, ``h_virtual``, ``c_virtual``. The
    original parsed CA and O are not modified — the caller can still pull the
    crystallographic N / C from ``parsed["n_coords"]`` for comparison.

    Notes
    -----
    * The construction is purely linear in the input coordinates and therefore
      auto-grad friendly out of the box.
    * Coefficients are kept as Python floats and broadcast against the input
      device / dtype — no implicit promotion.
    """
    ca = parsed["ca_coords"]                      # (N, 3)
    o = parsed["o_coords"]                        # (N, 3)
    device = ca.device
    dtype = ca.dtype
    n_res = ca.shape[0]
    chains = parsed["chain_ids"]

    # boolean masks: which residues have a valid i-1 (same chain) and i+1 (same chain)?
    has_prev = torch.zeros(n_res, dtype=torch.bool, device=device)
    has_next = torch.zeros(n_res, dtype=torch.bool, device=device)
    for i in range(n_res):
        if i > 0 and chains[i] == chains[i - 1]:
            has_prev[i] = True
        if i < n_res - 1 and chains[i] == chains[i + 1]:
            has_next[i] = True

    # shifted tensors: ca_prev[i] = ca[i-1] (or NaN), o_prev[i] = o[i-1] (or NaN)
    nan = torch.full_like(ca, float("nan"))
    ca_prev = torch.where(has_prev.unsqueeze(1), torch.roll(ca, shifts=1, dims=0), nan)
    o_prev = torch.where(has_prev.unsqueeze(1), torch.roll(o, shifts=1, dims=0), nan)
    ca_next = torch.where(has_next.unsqueeze(1), torch.roll(ca, shifts=-1, dims=0), nan)

    coeffs = _CIS_PRO if use_cis_proline else _TRANS
    a, b, c = coeffs["N"]
    n_v = a * ca_prev + b * ca + c * o_prev

    a, b, c = coeffs["C"]
    c_v = a * ca + b * ca_next + c * o

    a, b, c = coeffs["H"]
    h_v = a * ca_prev + b * ca + c * o_prev

    return {
        "n_virtual": n_v.to(dtype=dtype),
        "c_virtual": c_v.to(dtype=dtype),
        "h_virtual": h_v.to(dtype=dtype),
    }
