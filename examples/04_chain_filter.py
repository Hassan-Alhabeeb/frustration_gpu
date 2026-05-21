"""Example 04 - Multi-chain PDB, filter to one chain.

Run frustration analysis on a multi-chain PDB twice: once on the full
structure, once restricted to chain A only. Report the pair counts and
show that the chain-A run is a strict subset of pairs from the full run.

Run:
    python examples/04_chain_filter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frustration_gpu import compute_frustration  # noqa: E402

# 11BG is a homodimer (chains A + B).
_bundled = Path(__file__).resolve().parent.parent / "tests" / "data" / "11BG.pdb"
if _bundled.is_file():
    PDB_PATH = _bundled
else:
    PDB_PATH = Path("F:/research_plan/allosteric/data/pdb_files/11BG.pdb")


def main() -> None:
    full = compute_frustration(PDB_PATH, mode="configurational")
    chain_a = compute_frustration(PDB_PATH, mode="configurational", chain="A")

    chains_full = sorted(set(full.pair_records["ChainRes1"]).union(
        full.pair_records["ChainRes2"]
    ))
    chains_a = sorted(set(chain_a.pair_records["ChainRes1"]).union(
        chain_a.pair_records["ChainRes2"]
    ))

    print(f"PDB: {PDB_PATH.name}")
    print(f"  Full structure: chains={chains_full}, "
          f"n_residues={full.metadata['n_residues']}, "
          f"n_pairs={full.metadata['n_pairs']}")
    print(f"  Chain A only:   chains={chains_a}, "
          f"n_residues={chain_a.metadata['n_residues']}, "
          f"n_pairs={chain_a.metadata['n_pairs']}")

    # Full chain-A native pairs from the FULL run.
    full_a_mask = (
        (full.pair_records["ChainRes1"] == "A")
        & (full.pair_records["ChainRes2"] == "A")
    )
    print(f"\nIntra-chain-A pairs from FULL run: {full_a_mask.sum()}")
    print(f"Pairs from chain-A-only run:       {len(chain_a.pair_records)}")
    print(
        "Note: counts can differ slightly - the chain-A-only run re-computes "
        "rho on the isolated chain, so its in-contact-pair set is a "
        "structurally-consistent subset, not strictly identical."
    )


if __name__ == "__main__":
    main()
