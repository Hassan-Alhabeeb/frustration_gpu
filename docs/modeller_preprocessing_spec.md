# Modeller preprocessing spec - for native reimplementation

Reviewer: Opus 4.7, 2026-05-20
Scope: read-only investigation of whether the 3F9M density Spearman gap
(0.27 vs `5adens.dat`, Phase 4 review) is caused by upstream Modeller
preprocessing that frustrapy invokes and we don't.

## TL;DR

**Modeller is NOT actually invoked by the frustrapy version installed at
`C:/Users/7sN/AppData/Local/Programs/Python/Python310/lib/site-packages/frustrapy`.**
The reference 5adens.dat files in `benchmark/cpu_baseline/configurational/`
were produced by a Biopython-only path. Modeller code paths (`MissingAtoms.py`,
`complete_backbone`) exist in the codebase as dead weight - they're defined
but never called from `FrustrationCalculator._prepare_calculation_files`.

This means the Phase 4 review's headline finding ("real Modeller-renumbering
issue") is partially wrong - there is no Modeller renumbering happening
upstream. The 3F9M density gap has a different root cause.

## Frustrapy's Modeller wrapper (legacy/dead code in current build)

`frustrapy/core/scripts/MissingAtoms.py` is a 15-line wrapper:

```python
# Line 5-15
from modeller import *
from modeller.scripts import complete_pdb

PDBToComplete = sys.argv[1]
env = environ()
env.libs.topology.read('${LIB}/top_heav.lib')        # heavy-atom topology
env.libs.parameters.read('${LIB}/par.lib')           # parameters
m = complete_pdb(env, PDBToComplete, transfer_res_num="true")
m.write(file=PDBToComplete+"_completed")
```

Operations Modeller's `complete_pdb` performs (per Modeller docs):
- Reads `top_heav.lib` / `par.lib` for heavy-atom-only topology
- Rebuilds **missing heavy-atom sidechain coordinates** using CHARMM topology
- Builds **missing backbone atoms** (N/CA/C/O) by extrapolating from neighbours
- `transfer_res_num="true"` **preserves PDB author residue numbers** in output
  (so it would NOT renumber alt-conformer residues into separate entries as
  the Phase 4 memo speculated)
- Picks alt-locations: chooses **highest occupancy** (Modeller default;
  ties broken by altloc 'A' alphabetically)
- Drops HETATM by default unless explicitly preserved

Call site (only legacy, no caller):
- `frustrapy/utils/helpers.py:63-83` defines `complete_backbone(pdb)` which
  shells out: `subprocess.run(["python3", missing_atoms_script, pdb.job_dir, pdb_file])`
  then renames `*.pdb_completed` -> `*.pdb`.
- **No call to `complete_backbone` exists anywhere in `frustrapy/analysis/`.**
  Grep confirms: only `__init__.py` re-export and the definition itself.

## What the actual frustrapy pipeline does to 3F9M (Biopython, not Modeller)

Entry point: `frustrapy/analysis/frustration_calculator.py:_prepare_calculation_files`
- Line 422: `run_subprocess(["sh", PdbCoords2Lammps.sh, pdb_base, pdb_base, scripts_dir])`
- That shell script (`AWSEMFiles/AWSEMTools/PdbCoords2Lammps.sh:9`):
  ```
  python3 .../AWSEMTools/PDBToCoordinates.py <pdb> <coord>
  python3 .../AWSEMTools/CoordinatesToWorkLammpsDataFile.py ...
  ```

`PDBToCoordinates.py` behaviour (lines 137-298):
- Uses Biopython `PDBParser(PERMISSIVE=1)` (line 139)
- For each chain, for each residue with `res_id` in `(' ', 'H_MSE', 'H_M3L', 'H_CAS')`
  and all of N/CA/C/O present (line 193, 195):
  - Reads `res['N'].get_coord()`, `res['CA']`, `res['C']`, `res['O']` (line 202-205)
  - Reads `res['CB']` for non-Gly (line 215); for Gly computes virtual H
- **Residues missing any of N/CA/C/O are SILENTLY SKIPPED** (line 190, `continue`)
- **No alt-conformer handling code**: Biopython's `DisorderedAtom.get_coord()`
  returns the selected child's coord. Selection rule
  (`Bio/PDB/Atom.py:560-572`) is **highest occupancy; ties keep first added**.

For 3F9M, the affected residues 9/27/42/46/48/107/155/243 all have altloc
records at occupancy **0.50/0.50** (tied). The PDB file lists altloc 'A'
records before 'B' records (verified: line 50 'A', line 51 'B' for resnum 9).
So Biopython picks 'A' on tie - **identical to our parser's altloc filter**
at `src/parser.py:85`.

## Frustrapy intermediate PDB for 3F9M

No cached intermediate PDB was found:
- `F:/research_plan/frustration_gpu/benchmark/cpu_baseline/` has only `*.dat`
  dumps (5adens, configurational, tertiary_frustration, mutational,
  singleresidue), no `*.pdb`.
- `F:/research_plan/_demo_package/data/pdb_files/3F9M.pdb` is byte-identical
  to `F:/research_plan/allosteric/data/pdb_files/3F9M.pdb` (same 90 altloc
  records).

## Empirical residue count - raw vs LAMMPS vs our parser

Verified counts on `F:/research_plan/allosteric/data/pdb_files/3F9M.pdb`:

| Source | Residue count (chain A) | Resnum range |
|---|---:|---|
| Raw PDB ATOM lines | 451 unique resnums | 4 .. 458 |
| LAMMPS `3F9M_5adens.dat` | 451 rows | 4 .. 458 |
| LAMMPS `3F9M_singleresidue.dat` | 451 rows | 4 .. 458 |
| Our parser (altloc A only) | 451 (matches the test panel) | 4 .. 458 |

`comm -12` (intersection) on the sorted resnum sets between raw PDB and
LAMMPS 5adens gives **451 / 451 - perfect match. No residues added, none
removed, no renumbering.** This rules out the Phase 4 hypothesis that
Modeller "shifts the index by +1 starting at the first alt-conformer".

The 8 altloc-bearing residues (3F9M chain A, CA records):

| Resnum | Resname | Altloc A occ | Altloc B occ |
|---:|---|---:|---:|
| 9   | GLN | 0.50 | 0.50 |
| 27  | GLU | 0.50 | 0.50 |
| 42  | ASP | 0.50 | 0.50 |
| 46  | ARG | 1.00 | (no B) |
| 48  | GLU | 0.50 | 0.50 |
| 107 | MET | 0.50 | 0.50 |
| 155 | ARG | 0.50 | 0.50 |
| 243 | LEU | 0.50 | 0.50 |

Cartesian diff between altloc A and B for CA(9): (-55.540 vs -55.560,
12.932 vs 12.889, -10.814 vs -10.790) - max delta ~0.043 A. Negligible.

## Altloc handling rules - definitive

Both pipelines pick altloc 'A' for 3F9M:
- **Our parser** (`src/parser.py:84-86`): explicit allowlist `altloc in ('', 'A')`,
  so 'A' kept, 'B' silently dropped.
- **Frustrapy's Biopython path**: `DisorderedAtom` selects highest occupancy,
  tie broken by *first added* (Bio/PDB/Atom.py:570 - `if occupancy > self.last_occupancy`,
  strict greater). PDB-file order is A then B, so A wins on every tied residue.

Both pipelines produce the **same CA / CB coordinates** for 3F9M chain A.

## Missing-atom rebuild - what would Modeller add?

Verified: no residue in 3F9M chain A is missing any backbone atom (N/CA/C/O)
or any non-Gly CB. So Modeller's `complete_pdb` would be a no-op on
backbone/CB and only potentially rebuild sidechain atoms beyond CB - which
AWSEM doesn't use (it operates on CA+CB+N+O only via `PDBToCoordinates.py`).

**Modeller cannot be the source of the 3F9M density Spearman gap.**

## Implication: the 0.27 Spearman is something else

Since residue count, resnum mapping, and CA/CB coordinates are
byte-comparable between our parser and frustrapy's reference path, the
0.27 Spearman on `3F9M_5adens.dat` must come from one of:

1. **A latent bug in our `density.py` aggregation** that surfaces on 3F9M
   specifically (e.g. NaN propagation through some residue with partial CB).
2. **Positional vs key-based join in the test** - `test_density_spearman_against_5adens`
   does positional alignment by index 0..n_min (`tests/test_density.py:219-223`).
   If a single residue ordering difference exists (e.g. our parser ordered
   by file-encounter vs ref ordered by 1-indexed chain seq), every residue
   past that point is off-by-one. **This is the most likely cause given
   the count matches.**
3. **A different random-seed or per-pair FI value drift on this PDB**
   propagating into the 3-state classification (highly/neutral/minimally).

Recommendation: replace the positional `ref_keys[:n_min]` join in
`test_density.py:221-227` with a **(resnum, chain) keyed join** before
ranking. If Spearman jumps from 0.27 -> >0.95, the 3F9M gate can be raised
and the comment about "Modeller renumbering" removed. This is consistent
with both `test_configurational_fi_validation[3F9M]` already passing
Spearman > 0.99 on the per-pair physics.

## Implementation plan (no Modeller dependency)

The current parser is **already correct** for 3F9M's needs. No code change
is required to match frustrapy. Specifically:

| Modeller would do | We need | Status |
|---|---|---|
| Pick highest-occupancy altloc, tie-> A | `src/parser.py:85` keeps `altloc in ('', 'A')` | DONE |
| Preserve author resnums (transfer_res_num=true) | parser stores `res_seq` from PDB cols 22-26 verbatim | DONE |
| Rebuild missing backbone | not needed on 3F9M (no gaps); for other PDBs we drop residues missing CA (`src/parser.py:185`) - mirrors `PDBToCoordinates.py:190` | EQUIVALENT |
| Rebuild missing CB | not needed on 3F9M; for other PDBs we leave NaN and downstream uses `_xb_coords` fallback | EQUIVALENT (frustrapy also tolerates - line 213 `has_cb=False; continue with backbone`) |

What WOULD be needed if we ever wanted to support PDBs with truly missing
backbone atoms (not 3F9M's case):
- Implement Engh-Huber idealised geometry rebuilders for N/CA/C/O gaps
  (small dependency-free routine using neighbouring residue geometry).
- Implement standard CB rebuild from N/CA/C using fixed bond lengths /
  angles. Already partially done in `src/virtual_atoms.py` per the
  codebase notes.
- Neither is needed to close the 3F9M Spearman gap.

## Modeller installation check

Result: `python -c "import modeller"` returns `ModuleNotFoundError`. Modeller
is **not installed** on this Windows host. If the user ever wants ground-truth
Modeller output for cross-check, the call to make is:

```bash
# Linux/VM (Modeller needs Linux usually; license required):
python3 MissingAtoms.py /tmp/3F9M_in/3F9M.pdb
# Output: /tmp/3F9M_in/3F9M.pdb_completed
```

But given that the frustrapy build in use does NOT invoke Modeller, this
ground-truth run is **not needed for byte-exact 3F9M reproduction**.

## Conclusion

**The "missing Modeller" Phase 4 finding is incorrect for the installed
frustrapy build.** No Modeller preprocessing happens. The 3F9M density
Spearman gap is most likely a **positional-join artifact in
`test_density.py:test_density_spearman_against_5adens`** or a latent
density-side bug that only surfaces on 3F9M's specific pair distribution.

Replace positional alignment with `(resnum, chain)` keyed join, re-run,
and the gate should rise to >0.95 with no parser change.
