# QA5 — Docs + API surface review

Date: 2026-05-21
Scope: README.md, QUICKSTART.md, VALIDATION.md, CHANGELOG.md, docs/API.md,
docs/lammps_compat_fixes.md, docs/frustrapy_vs_us.md, examples/01..07_*.py,
CITATION.cff, pyproject.toml, LICENSE.

Verdict: **Defensible overall.** The docs are unusually well sourced
(numbers trace to a real CSV; an API-drift regression test is checked in).
Three claims are stated more strongly than the underlying CSV supports
(see CRITICAL/HIGH below). No claim is literally false in the strict
sense, but two would lose points in a PI cross-examination.

Counts: 4 HIGH, 9 MEDIUM, 6 LOW.

---

## CRITICAL (claims that are literally false)

None found. Every numerical claim has a backing artefact under
`benchmark/` or `tests/`.

---

## HIGH (misleading or unverifiable)

### H1. "Configurational FI Spearman = 1.0000 exact" overstated
`CHANGELOG.md:27` and `VALIDATION.md:24` say configurational FI Spearman
is *exactly* 1.0000 (bolded **1.0000** in the table). The backing CSV
`benchmark/phase5_spearman.csv:2-31` shows the actual values are
`0.9999981 ... 0.9999998` — close to but not literally 1.0000. The
rationale in `VALIDATION.md:24` ("FI ranking ... is a monotone function
of E_native") is correct in theory, but in practice 1-2 pair pairs must
be tying differently between us and the LAMMPS dump (E_native rounded to
%8.3f produces ties).

Fix: state the floor (`≥ 0.999998` or `≈ 1.0000`) rather than asserting
literal equality. Drop the word "exact" from `CHANGELOG.md:27`.

### H2. "max |FI_CPU − FI_CUDA| ≤ 1e-14" is over-conservative; CSV shows 0.0
`VALIDATION.md:30,32-46,206` repeatedly claims `≤ 1e-14`. The CSV
`benchmark/phase5_spearman.csv` shows the column
`cpu_vs_cuda_max_abs_diff_fi = 0.0` literally on all 30 combos. The text
on `VALIDATION.md:47` correctly notes "decoy_mean agrees to ~15 sig
digits" with last-ULP examples, but the FI itself is bit-equal because
the ranking is identical and the FI values are computed in the same
order. The doc is technically not wrong (0.0 ≤ 1e-14), but a PI will
ask why we don't just say "0.0" — and the CHANGELOG line 27 *does* say
"= 0.0 literally", contradicting VALIDATION's softer claim. Pick one.

Fix: align `VALIDATION.md:30,46,98,206` to say "= 0.0 literally" matching
`CHANGELOG.md:27` and the example assertion `examples/05_gpu_vs_cpu.py:52`
(which uses `< 1e-3`, much looser than reality).

### H3. "Every code block is a verbatim excerpt from examples/ — they are tested"
`QUICKSTART.md:3` says all code blocks are tested examples. Examining
the examples directory, several QUICKSTART blocks are NOT verbatim from
`examples/`:
- Section 5 ("Try all three modes", `QUICKSTART.md:62-77`) — not a copy
  of `examples/02_three_modes.py` (the example uses separate `summarise_*`
  functions; QUICKSTART inlines `if mode == "singleresidue"`).
- Section 7 DH example (`QUICKSTART.md:108-117`) — not a copy of
  `examples/03_dh_electrostatics.py`.
- Section 8 frustrapy migration (`QUICKSTART.md:127-141`) — close to
  `examples/07_frustrapy_drop_in.py` but not verbatim.

Risk: a doc-vs-example drift will go unnoticed in CI.

Fix: either tighten the language ("excerpts inspired by examples/")
or add a doctest pass over QUICKSTART.md.

### H4. `VALIDATION.md:79-80` LAMMPS-compat density-Spearman numbers don't match CSV
`VALIDATION.md:79-80` (section 4 table) reports:
- 4HON default = 0.1659
- 6F56 default = 0.1397

`benchmark/phase5_spearman.csv:20,29` shows:
- 4HON configurational default = 0.19415
- 6F56 configurational default = 0.13014

For 1O3S (0.2240) and 3F9M (0.2735), the doc matches the CSV. For 4HON
and 6F56 the numbers differ. The "≥0.99" compat-flag column for these
two PDBs is also non-cited (no row in `phase5_spearman.csv` for the
compat-flag run). Likely the 4HON/6F56 default numbers came from a
different (mutational?) run — `phase5_spearman.csv:21` shows 4HON
mutational density = 0.16588 (= "0.1659" rounded). So `VALIDATION.md:79`
is silently quoting the **mutational** density Spearman as the "default"
without saying so.

Fix: re-pull the four rows from the canonical
`benchmark/phase5_spearman.csv` and label the mode explicitly. The
mutational vs configurational density numbers are different and the doc
should not collapse them.

---

## MEDIUM (inconsistency / typo / clarity)

### M1. "187 tests on 4-PDB validation panel" vs "10-PDB panel"
`CHANGELOG.md:19` says "187 tests passing on the **4-PDB** validation
panel". `VALIDATION.md:6,164,180` and `phase5_spearman.csv` clearly use
the **10-PDB** panel (5AON, 11BG, 1O3S, 3F9M, 5N9R, 4C8B, 4HON, 2SKE,
1OS2, 6F56). The 4-PDB subset is only used for the frustrapy head-to-head
because frustrapy was only run on those four. Reword.

### M2. `API.md` signature drift: `debye_huckel_pair_energy`
`docs/API.md:443-451` documents the parameter as `r: float` and omits
`epsilon`, `device`, `dtype`. The live signature
`src/debye_huckel.py:371-382` is `r_ij: torch.Tensor | float` with
three extra kwargs. The `tests/test_api_docs.py` regression test does
include `debye_huckel_pair_energy` in `EXPECTED_DOCUMENTED`, but the
first-positional rename (`r` → `r_ij`) won't be caught by the kwarg-only
matcher.

### M3. `5N9R` mutational CUDA slower than CPU (acknowledge somewhere)
`benchmark/phase5_panel_results.csv:52-53` shows:
- 5N9R mutational CPU: 1721 ms
- 5N9R mutational CUDA: 2290 ms (33% slower)

This single panel row is the only place CUDA loses to CPU in the dataset.
The headline 12 head-to-head combinations all show GPU wins, but a PI
spot-checking the panel CSV will ask why. A one-line note in
`VALIDATION.md:142` ("4HON small spikes outside the head-to-head subset
— see 5N9R") would head off the question.

### M4. "10-PDB CPU baseline panel" but only 4 frustrapy comparisons
`VALIDATION.md:7,165` mentions 126 LAMMPS dump files from a 10-PDB
baseline, but the head-to-head only covers 4 PDBs (5AON, 11BG, 1O3S,
3F9M). The README table at lines 32-45 also only shows 4. Be explicit
that the head-to-head is a 4-PDB sub-panel because frustrapy was
expensive to run.

### M5. "Apache-2.0 ... we re-implement the algorithms from published specifications"
`README.md:137` makes a licensing claim ("we re-implement the algorithms
from published specifications rather than redistributing their source").
This is a defensible position but a PI / IP lawyer might ask for
clarification on the gamma tables (`src/data/gamma.dat`,
`src/data/burial_gamma.dat`). Those are extracted parameter values
originating from OpenAWSEM/LAMMPS-AWSEM — re-shipping numerical
parameter tables is fine under fair use / scientific data norms, but
the LICENSE / README should briefly state where `gamma.dat` /
`burial_gamma.dat` came from.

### M6. `examples/` hardcode absolute Windows paths
All seven example scripts hardcode `F:/research_plan/allosteric/data/pdb_files/`
(`01_basic.py:19`, `02_three_modes.py:16`, `03_dh_electrostatics.py:21`,
`04_chain_filter.py:18`, `05_gpu_vs_cpu.py:21`, `06_batch.py:19`,
`07_frustrapy_drop_in.py:26`). README/QUICKSTART says "Adjust this path"
but a clean-clone user has no PDBs. Add a `curl -O` line in each example
docstring (the QUICKSTART has it but the example scripts don't), or have
a single `_download_panel_pdbs.py` helper that the examples reference.

### M7. API.md `compute_frustration` return-type cell missing for some kwargs
`docs/API.md:78-97` documents the kwarg table but does not mention
`v_dh` in the metadata table (`API.md:251-273`). The compute_frustration
docstring (`src/compute_frustration.py:510-513`) refers to
`metadata["v_dh"]`. Either add the key to the API.md metadata table or
drop the docstring reference if it's not actually populated.

### M8. CHANGELOG omits the "Phase 6 polish" caveat for α-chunking
`CHANGELOG.md:31-34` "Known limitations" mentions
"Alpha-chunking the auxiliary (N,N) tensors will address this; tracked
in `docs/optimization_opportunities.md`." Cross-checked — that doc exists,
good. But README `line 64` says α-chunking is "Phase 6 polish item"
while `README.md:103` lists it as "in progress — Phase 6 polish". Make
the phrasing consistent (it's either planned or in progress, not both).

### M9. CITATION.cff missing frustratometeR primary citation
`CITATION.cff:116-127` lists frustratometeR as a `software` reference but
NOT the Rausch 2021 *Bioinformatics* paper (`README.md:133` cites it
directly). The CITATION.cff should include both the software repo AND
the journal article — these are separate canonical references in the
frustration literature.

---

## LOW (polish)

### L1. README `line 7` test badge says "187_passing" — fine but no CI link
The badge is informational only (`tests/` not `actions/`). A real CI
badge (`gh workflow run`) would be stronger for KAIMRC defense. LOW
because not a claim of accuracy, just polish.

### L2. `examples/05_gpu_vs_cpu.py:52` asserts `< 1e-3` but VALIDATION claims `≤ 1e-14`
The example tolerance is 11 orders of magnitude looser than the actual
agreement. Tightening to e.g. `< 1e-12` would make the example a stronger
demonstration of the claim.

### L3. `QUICKSTART.md:99` comment says "≤ 1e-14" but example assertion is `< 1e-3`
Same as L2 — the in-comment claim doesn't match the assertion.

### L4. `docs/lammps_compat_fixes.md:296` says "163 → 167 tests passing"
Stale test count — the rest of the repo says 187. This is an older
checkpoint count from when only 4 compat tests had been added. Update.

### L5. `pyproject.toml` Apache classifier text is "Apache Software License"
The PyPI classifier is technically `License :: OSI Approved :: Apache
Software License` not `Apache 2.0` — this is the correct trove
classifier so the form is right, but `Apache Software License` covers
1.x and 2.0. Adding a comment that we're 2.0 specifically (also clear
from LICENSE file) would be belt-and-braces.

### L6. CHANGELOG `[Unreleased]` block is empty
Empty `[Unreleased]` is conventional in Keep-a-Changelog format, but
some projects elide it; not a defect.

---

## Top three honesty/accuracy issues

1. **H1** — "Configurational FI Spearman = 1.0000 exact" overstates a
   floor of 0.9999981 (`CHANGELOG.md:27`, `VALIDATION.md:24`).
2. **H4** — VALIDATION section-4 numbers for 4HON / 6F56 do not match
   `phase5_spearman.csv`; appears to silently quote mutational density
   numbers in a configurational-density context.
3. **H3** — QUICKSTART's "every code block ... tested" claim is not
   true today; several blocks are paraphrased.

## Top three consistency issues

1. **M1** — CHANGELOG says "4-PDB validation panel"; everything else
   uses "10-PDB panel" (the 4-PDB subset is only the frustrapy
   head-to-head).
2. **H2** — VALIDATION uses `≤ 1e-14` but CHANGELOG and the CSV both
   say literal `0.0`. Pick one.
3. **L4** — `docs/lammps_compat_fixes.md` still says "163 → 167 tests";
   repo is at 187. Update for consistency with the badge and
   VALIDATION.md.

## Things that surprised me (positive)

- An `test_api_docs.py` regression test that *parses* `docs/API.md` code
  blocks via `ast` and checks them against `inspect.signature`. This is
  unusual diligence — most projects let API docs drift silently.
- Three distinct LAMMPS-compat flags (`include_dna`, `lammps_compat_altloc`,
  `include_dh_in_e_native`) each map to a published `fix_backbone.cpp`
  line or an empirical frustrapy bug, each gated by a default that picks
  the scientifically clean output. This is exactly the right
  default-choice pattern for a "we reproduce the reference, but cleanly"
  package.
- The hardware footnote (`README.md:51`, `VALIDATION.md:146`) honestly
  flags the CPU-vs-CPU column as not apples-to-apples. A weaker port
  would have buried it.

## Things that surprised me (concerning)

- The CSV-vs-doc number drift in H1/H4 suggests the docs were partly
  authored from memory rather than re-pulled from the CSV at write time.
  A small `regenerate_validation_tables.py` script that emits the
  markdown tables straight from `benchmark/phase5_spearman.csv` would
  prevent this whole class of issue.
- `5N9R` mutational CUDA-slower-than-CPU is real (panel CSV) but absent
  from any narrative. A PI will find it on their first audit of
  `phase5_panel_results.csv`.
