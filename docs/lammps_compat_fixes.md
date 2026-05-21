# LAMMPS / frustrapy compatibility fixes — 2026-05-20

Four targeted fixes to close the remaining parity gaps between our PyTorch
port and `LAMMPS-AWSEM + frustratometeR` output. Each is principled
(traceable to a specific upstream behaviour), documented, and protected
by explicit opt-in flags so the default API remains the cleanest path.

| Fix | File | New default | Opt-in flag |
|---|---|---|---|
| 1 | `compute_frustration.py` | `electrostatics_k` is metadata-only | `include_dh_in_e_native=True` to add DH to E_native |
| 2 | `parser.py` | drop residues missing N/CA/C/O | `keep_incomplete_backbone=True` keeps NaN-slot residues |
| 3 | `parser.py` + density emission | DNA chains dropped | `include_dna=True` to emit DNA placeholder rows |
| 4 | `parser.py` + density emission | altloc-A only | `lammps_compat_altloc=True` inserts altloc-B shadows |

All flags wire through `compute_frustration(...)` and the frustrapy-style
`calculate_frustration(...)` adapter; metadata reports each flag's value.

---

## Fix 1 — DH semantics: `include_dh_in_e_native=False` default

### What changed

Previously, passing `electrostatics_k=4.15` to `compute_frustration` would
ADD the per-pair Debye-Hückel term to `E_native`. As of this commit,
`electrostatics_k=4.15` alone is **metadata-only**; DH is not added to
`E_native` unless the new flag `include_dh_in_e_native=True` is set.

### Why

frustratometeR's analysis pipeline scores the **water + burial**
Hamiltonian only. Even on LAMMPS runs launched with `huckel_flag=true`
and `k_QQ=4.15`, the `energy.log` `Electro.` column is 0.000000 and the
`tertiary_frustration.dat` `native_energy` column is identical to a
`k_QQ=0` run. This was verified empirically:

```
diff benchmark/cpu_baseline/configurational/5AON_tertiary_frustration.dat \
     benchmark/cpu_baseline/param_sweep/5AON_electro_4p15_tertiary_frustration.dat
# → byte-identical
```

The LAMMPS-AWSEM `electrostatics_k` knob controls the **simulation
force** (during dynamics), NOT the **analysis energy** (the
`tertiary_frustration.dat` and `energy.log` outputs). Adding DH to
E_native in the analysis was therefore an over-eager interpretation
of "electrostatics_k means add DH everywhere."

### Validation

`tests/test_compute_frustration.py::test_compute_frustration_dh_byte_exact_against_lammps_5AON`
runs the orchestrator with `electrostatics_k=4.15` and the new default
`include_dh_in_e_native=False`, then byte-checks the resulting
`E_native` column against `5AON_electro_4p15_tertiary_frustration.dat`:

```
Pair count ours: 221, theirs: 221
Max |E_native diff| = 0.000000, violations > 5e-3: 0
```

A second assertion runs with `include_dh_in_e_native=True` and verifies
the delta matches the standalone `debye_huckel_pair_energy()` formula
to %8.3f precision on every charged-residue pair.

---

## Fix 2 — Backbone-completeness filter: `keep_incomplete_backbone=False`

### What changed

The parser previously dropped only residues missing a CA. Now, by
default, residues missing ANY of N / CA / C / O are dropped. To keep
the old (lax) behaviour pass `keep_incomplete_backbone=True`.

### Why

LAMMPS-AWSEM's `PDBToCoordinates.py` (lines 182-191 in the bundled
frustrapy install) skips any residue with `missing_backbone = True`.
AWSEM needs full backbone for virtual CB construction (Engh-Huber
geometry; see `src/virtual_atoms.py`) and for the burial / contact
potentials. A NaN backbone atom poisons everything downstream.

### Empirical note on 1O3S

The original task brief stated 1O3S chain A residues 182–207 are dropped
by LAMMPS because they have incomplete backbone. Empirically (verified
by walking `F:/research_plan/allosteric/data/pdb_files/1O3S.pdb`'s ATOM
records) all 200 chain-A residues have full backbones in this PDB. The
LAMMPS-side truncation of 1O3S to 174 chain-A residues in the 5adens
output is a different mechanism — see Fix 3 below.

This fix is still correct (matches `PDBToCoordinates.py`), it just
doesn't actually reduce the 1O3S row count.

### Validation

Default behaviour preserved on the 4-PDB panel (5AON / 11BG / 1O3S /
3F9M): all four still produce the same residue counts as before (49,
248, 200, 451) because none of those PDBs have residues missing N/CA/C/O.

---

## Fix 3 — DNA inclusion: `include_dna=False` default

### What changed

DNA residues (DA / DT / DC / DG and ribonucleotide variants) are now
dropped by default (correctly — AWSEM has no DNA force field). With
`include_dna=True`, DNA residues are emitted as "placeholder rows" in
the 5adens density file (and only there — they don't participate in
the math), with their C1' atom serving as a CA-proxy for positional
alignment.

### Why frustratometeR's 1O3S 5adens looks weird

frustratometeR's `_calculate_frustration_density` iterates:

```python
positions = pdb.equivalences.iloc[:, 2].tolist()     # 226 entries (incl. DNA)
res_chains = pdb.equivalences.iloc[:, 0].tolist()    # 226 entries
ca_xyz = pdb.atom.loc[pdb.atom["atom_name"] == "CA",
                      ["x","y","z"]].values           # 200 entries (protein-only)
for i, (ca_point, res_num, chain_id) in enumerate(
    zip(ca_xyz, positions, res_chains)
):
    ...
```

This is a **zip-mismatch bug** in frustrapy: `pdb.equivalences` covers
ALL residues (including DNA) but `pdb.atom["atom_name"]=="CA"` only
catches protein CAs (DNA backbones have no CA). Python's `zip` cuts
to the shorter list (200), so the LAST 26 protein chain-A residues
(residues 182–207, which sit AFTER the DNA chains' equivalences entries)
get DROPPED from the 5adens output. AND the labels for the first 26
output rows are MIS-ASSIGNED to the DNA chains (because positions[0..25]
hold DNA-chain labels but ca_xyz[0..25] hold the first 26 protein CAs).

### Our compat reproduction

When `include_dna=True`, the parser produces a `lammps_emit_rows` list
in the order:

```
(chain=B, resnum=-2, math_idx=0)    # first protein CA, labelled as DNA
(chain=B, resnum=-1, math_idx=1)
... 26 DNA rows total
(chain=A, resnum=8,  math_idx=26)   # 27th protein CA, labelled chain A res 8
(chain=A, resnum=181, math_idx=199) # 200th protein CA, labelled res 181
```

The orchestrator's density emission `_project_density_to_lammps_emit`
then writes 200 rows total (= `min(N_protein, N_equiv)` = 200), where
each row uses the density value at the protein math_idx but the chain /
resnum labels from the LAMMPS-bug "zip" view. This reproduces the
duplicate / shifted labels that frustratometeR's broken zip produces.

### Validation

`tests/test_density.py::test_density_spearman_lammps_compat_flags[1O3S]`:

| Run | Spearman vs LAMMPS 5adens (nHighlyFrst, resnum-keyed join) |
|---|---|
| Default | 0.2240 |
| `include_dna=True, lammps_compat_altloc=True` | **0.9992** |

### Limitations (documented in the kwarg docstring)

AWSEM frustration on DNA is not physically meaningful: there is no
published gamma table for nucleotide contacts, no validated burial /
water-mediated parameters, and the C1' atom is geometrically
non-equivalent to CA. This flag exists ONLY for byte-comparable parity
with frustratometeR's output on protein-DNA complexes. The DNA rows
inherit protein densities (a known feature of frustrapy's zip-mismatch
bug, NOT a faithful physical computation).

---

## Fix 4 — altloc-B duplication: `lammps_compat_altloc=False`

### What changed

The parser previously kept only altloc-A (and blank-altloc) records, the
standard convention (and what BioPython picks by default on tied
occupancies). With `lammps_compat_altloc=True`, altloc-B records are
inserted as "shadow" residues immediately after their altloc-A
counterpart. The math layer subsets them out (so per-pair energies are
unchanged), but the density emission iterates over the full residue
list including the shadows.

### Why frustratometeR's 3F9M 5adens shows consecutive-duplicate rows

frustratometeR's emit pattern on 3F9M:

```
Resnum 9:  density(res 9  CA)     # altloc-A position
Resnum 10: density(res 9B CA) ≈ density(res 9 CA)  # altloc-B re-uses next resnum
Resnum 11: density(res 10 CA)     # actual res 10 — shifted into resnum 11 slot
```

So rows at (resnum 9, 10) appear with the SAME density value (because
altloc-A and altloc-B CAs are ~0.04 Å apart, and the 5 Å density
sphere counts identical), and subsequent rows are shifted by one
position relative to PDB-author resnums.

The mechanism: frustrapy / LAMMPS' `pdb.atom` collection (with
disordered-atom iteration) yields BOTH altloc-A and altloc-B CA atoms
in file order, giving `ca_xyz` length 458 for 3F9M. `pdb.equivalences`
yields one entry per unique resnum, length 451. zip cuts to 451 output
rows. The label resnum at output row k = equivalences[k].resnum
(which is the kth unique PDB resnum in file order), but the density is
computed at `ca_xyz[k]` (where positions 5, 24, 40, 47, 103, 151, 235
are the altloc-B copies).

After 7 altlocs in 3F9M chain A, the FILE-position-to-resnum mapping
between ca_xyz and equivalences gradually drifts:
- Resnum 9 (eq[5]) gets density at ca_xyz[5] = res 9 altA CA (correct).
- Resnum 10 (eq[6]) gets density at ca_xyz[6] = res 9 altB CA (≈ res 9).
- Resnum 11 (eq[7]) gets density at ca_xyz[7] = res 10 CA (shifted +1).
- Resnum 28 (eq[24]) gets density at ca_xyz[24] = res 27 altA CA.
- Resnum 29 (eq[25]) gets density at ca_xyz[25] = res 27 altB CA (≈ res 27).
- Resnum 30 (eq[26]) gets density at ca_xyz[26] = res 28 CA (shifted +2).
- … shift grows by +1 after each altloc.

This is the mechanism that produces the empirically observed
duplicate-density pairs at resnums (9, 10), (28, 29), (44, 45), (51, 52),
(160, 161), (249, 250) in `3F9M_5adens.dat`.

### Our compat reproduction

`src/parser.py::_build_lammps_emit_rows` reproduces this pattern in
three steps:

1. The parser ingests altloc-A and altloc-B records into separate
   "residue groups" (keyed by `(chain, resnum, icode, altloc)`).
2. `_weave_altloc_b_shadows` re-orders the residue list so each
   altloc-B entry sits immediately after its altloc-A counterpart.
3. `_inherit_backbone_to_altloc_b` fills missing N/C/O on altloc-B
   shadows from their altloc-A neighbour (most B records in real PDBs
   only re-position the side chain).

`_build_lammps_emit_rows` then walks the full residue list with two
cursors — `math_idx` (advances on protein non-altloc-B rows only) and
`eq_idx` (advances on every row except altloc-B, since equivalences
doesn't include altloc-B). For each row it emits a tuple
`(chain_label, resnum_label, math_protein_idx)` matching frustratometeR's
emit pattern.

The orchestrator's `_project_density_to_lammps_emit` consumes that list
to build the final 5adens output.

### Validation

`tests/test_density.py::test_density_spearman_lammps_compat_flags[3F9M]`:

| Run | Spearman vs LAMMPS 5adens (nHighlyFrst, resnum-keyed join) |
|---|---|
| Default | 0.2736 |
| `lammps_compat_altloc=True` (incl_dna and incl_dh both irrelevant on 3F9M) | **0.9997** |

The first 12 emitted rows of our output now match the reference
duplicate-pair pattern exactly:

```
Res ChainRes  ... nHighlyFrst nNeutrallyFrst nMinimallyFrst
 4  A         ... 2 5 0
 5  A         ... 8 2 0
 6  A         ... 9 4 0
 7  A         ... 9 5 0
 8  A         ... 10 7 0
 9  A         ... 10 16 1     # altloc-A row
10  A         ... 10 16 1     # altloc-B shadow row (duplicate!)
11  A         ... 10 10 1     # actual res 10, shifted into resnum 11 slot
12  A         ... 10 10 2
...
```

### Limitations

The duplicate row at the altloc-B position uses the altloc-A's protein
math_idx (we don't have altloc-B in math, since the math subset is
protein-only). frustratometeR uses the altloc-B's CA coord for that
row's sphere center. The two CAs are typically <0.1 Å apart in well-
refined crystallographic alt-conformers, so the integer-counted sphere
content is identical (verified on 3F9M's 7 altloc residues — all 6
visible-duplicate pairs in the dump are exact int-match across the
altloc-A vs altloc-B pair).

---

## Summary of validation gates

After all 4 fixes are in place:

```
============================================================
187 tests passing (4 new in test_density_spearman_lammps_compat_flags,
on top of the pre-existing baseline; 0 regressions)
============================================================
```

| Gate | Default | With LAMMPS-compat flags | Status |
|---|---|---|---|
| 5AON E_native byte-exact vs `electro_4p15` dump | 0.000 max diff | n/a (test uses default) | ✓ PASS |
| 5AON density Spearman | 1.0000 | 1.0000 | ✓ PASS (≥0.98) |
| 11BG density Spearman | 0.9760 | 0.9760 | ✓ PASS (≥0.95) |
| 1O3S density Spearman | 0.2240 | 0.9992 | ✓ PASS (≥0.90 compat gate) |
| 3F9M density Spearman | 0.2736 | 0.9997 | ✓ PASS (≥0.90 compat gate) |

See `docs/frustrapy_vs_us.md` for the full kwarg-by-kwarg comparison.
