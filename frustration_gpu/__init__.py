"""Pure-PyTorch reimplementation of LAMMPS-AWSEM frustration analyser.

Phase 1 scope (this commit):
    - PDB parser
    - Virtual atom (N / H / C) construction
    - Parameter loaders for gamma.dat / burial_gamma.dat
    - Burial term

Phase 2+ scope (NOT in this commit):
    - Direct + water-mediated contact terms
    - Debye-Huckel
    - Hydrogen-bond / Tertiary_Frustratometer geometry
    - Decoy generation (1000 random AA mutations per native contact)
    - Frustration analyser (Z-score + per-residue density classification)
"""
from ._contact_common import ContactContext, build_contact_context
from .burial import (
    burial_density,
    burial_energy,
    compute_rho,
)
from .compute_frustration import (
    FrustrationResult,
    calculate_frustration,
    compute_frustration,
)
from .contact_gamma import load_direct_gamma, load_mediated_gamma
from .debye_huckel import (
    DH_CHARGES_FLOAT,
    DH_EPSILON,
    DH_K_QQ_DEFAULT,
    DH_K_SCREENING,
    DH_MIN_SEQ_SEP,
    DH_SCREENING_LENGTH_A,
    aa_charge_vector,
    debye_huckel_energy,
    debye_huckel_pair_energy,
)
from .decoys import (
    DEFAULT_CONTACT_CUTOFF_A,
    DEFAULT_N_DECOYS,
    LAMMPS_DUMP_RHO_MIN_SEQ_SEP,
    burial_switch,
    compute_configurational_decoy_energy,
    configurational_decoy_stats,
    lammps_dump_rho,
    sample_configurational_decoys,
    water_theta,
)
from .density import (
    DEFAULT_DENSITY_RATIO_A,
    compute_residue_density,
    density_to_dataframe,
    emit_5adens_dat,
)
from .direct_contact import direct_contact_energy, direct_pair_energy
from .frustration import (
    CLASS_HIGHLY,
    CLASS_MINIMALLY,
    CLASS_NEUTRAL,
    HIGHLY_FRUSTRATED_THRESHOLD,
    MINIMALLY_FRUSTRATED_THRESHOLD,
    WELL_LONG,
    WELL_SHORT,
    WELL_WATER_MEDIATED,
    WELLTYPE_R_SHORT_A,
    WELLTYPE_RHO_WATER,
    classify_frustration,
    compute_frustration_index,
    emit_postprocessed_pair_dat,
    emit_singleresidue_dat,
    emit_tertiary_frustration_dat,
    welltype_from_contact,
)
from .mutational_decoys import (
    PAIR_MIN_SEQ_SEP,
    mutational_decoy_stats,
)
from .parameters import (
    BURIAL_KAPPA,
    BURIAL_RHO_MAX,
    BURIAL_RHO_MIN,
    RHO_ETA_PER_NM,
    RHO_R_MAX_NM,
    RHO_R_MIN_NM,
    GammaTables,
    load_burial_gamma,
    load_gamma_tables,
)
from .parser import (
    ONE_TO_IDX,
    THREE_TO_ONE,
    chain_segments,
    parse_pdb,
)
from .singleresidue_decoys import singleresidue_decoy_stats
from .virtual_atoms import compute_virtual_atoms
from .water_mediated import water_mediated_energy, water_mediated_pair_energy

__all__ = [
    "burial_density",
    "burial_energy",
    "compute_rho",
    "BURIAL_KAPPA",
    "BURIAL_RHO_MIN",
    "BURIAL_RHO_MAX",
    "RHO_R_MIN_NM",
    "RHO_R_MAX_NM",
    "RHO_ETA_PER_NM",
    "GammaTables",
    "load_burial_gamma",
    "load_direct_gamma",
    "load_gamma_tables",
    "load_mediated_gamma",
    "ONE_TO_IDX",
    "THREE_TO_ONE",
    "chain_segments",
    "parse_pdb",
    "compute_virtual_atoms",
    "direct_contact_energy",
    "direct_pair_energy",
    "water_mediated_energy",
    "water_mediated_pair_energy",
    "DEFAULT_CONTACT_CUTOFF_A",
    "DEFAULT_N_DECOYS",
    "LAMMPS_DUMP_RHO_MIN_SEQ_SEP",
    "burial_switch",
    "compute_configurational_decoy_energy",
    "configurational_decoy_stats",
    "lammps_dump_rho",
    "sample_configurational_decoys",
    "water_theta",
    "DH_CHARGES_FLOAT",
    "DH_EPSILON",
    "DH_K_QQ_DEFAULT",
    "DH_K_SCREENING",
    "DH_MIN_SEQ_SEP",
    "DH_SCREENING_LENGTH_A",
    "aa_charge_vector",
    "debye_huckel_energy",
    "debye_huckel_pair_energy",
    "PAIR_MIN_SEQ_SEP",
    "mutational_decoy_stats",
    "singleresidue_decoy_stats",
    "CLASS_HIGHLY",
    "CLASS_MINIMALLY",
    "CLASS_NEUTRAL",
    "HIGHLY_FRUSTRATED_THRESHOLD",
    "MINIMALLY_FRUSTRATED_THRESHOLD",
    "WELL_LONG",
    "WELL_SHORT",
    "WELL_WATER_MEDIATED",
    "WELLTYPE_R_SHORT_A",
    "WELLTYPE_RHO_WATER",
    "classify_frustration",
    "compute_frustration_index",
    "emit_postprocessed_pair_dat",
    "emit_singleresidue_dat",
    "emit_tertiary_frustration_dat",
    "welltype_from_contact",
    "ContactContext",
    "build_contact_context",
    "FrustrationResult",
    "calculate_frustration",
    "compute_frustration",
    "DEFAULT_DENSITY_RATIO_A",
    "compute_residue_density",
    "density_to_dataframe",
    "emit_5adens_dat",
]
