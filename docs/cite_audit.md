# Citation + attribution audit

Auditor: read-only review for publication readiness.
Scope: repo state at 2026-05-21 (v0.1.0 CHANGELOG entry).
Files reviewed: `CITATION.cff`, `README.md`, `VALIDATION.md`, `QUICKSTART.md`, `CHANGELOG.md`, `LICENSE`, `pyproject.toml`, all `docs/*.md`, the mirrored C++ tree at `docs/reference_lammps_awsem/`, and citation-bearing comments in `src/*.py`.

## Verdict: NEEDS-FIXES

Five publication-blocking issues. Three are missing citations; two are license/attribution. The CITATION.cff is well-formed but incomplete relative to the canonical dependency set, and the mirrored upstream C++ code in `docs/reference_lammps_awsem/` is currently redistributed with no top-level LICENSE/NOTICE acknowledgement, which is the most serious item.

## 1. Citation completeness checklist

| Work | In CITATION.cff? | In README? | DOI present? | Issue |
|---|---|---|---|---|
| Ferreiro 2007 — Localizing frustration (PNAS) | YES | YES | YES (`10.1073/pnas.0709915104`) | OK |
| Davtyan 2012 — AWSEM-MD (J Phys Chem B) | YES | YES | YES (`10.1021/jp212541y`) | OK |
| Lu 2021 — OpenAWSEM (PLOS Comp Biol) | YES | NO | YES (`10.1371/journal.pcbi.1008308`) | Not mentioned in README primary-refs list; CITATION-only |
| Rausch 2021 — FrustratometeR (Bioinformatics) | PARTIAL | YES | NO | CITATION.cff entry is `type: software` with no DOI, no journal, no year — needs a proper `type: article` entry with DOI `10.1093/bioinformatics/btab176`, Bioinformatics 37(18):3038-3040, and full author list (Rausch, Monteagudo-Mesas, Bibic, Wolynes, Ferreiro). |
| Parra 2016 — Frustratometer 2 server (NAR) | **MISSING** | NO | n/a | DOI `10.1093/nar/gkw304`. Project explicitly inherits the Welltype/FrstState classification from frustratometeR which traces back to this server paper. |
| LAMMPS-AWSEM C++ (adavtyan/awsemmd) | YES (software) | NO | n/a | `repository-code` listed but no LICENSE in `docs/reference_lammps_awsem/` (see §3). |
| Thompson 2022 — LAMMPS (Comp Phys Comm) | **MISSING** | NO | n/a | DOI `10.1016/j.cpc.2021.108171`. The whole VALIDATION.md ("FI Spearman ≥ 0.9975 vs LAMMPS reference") rests on LAMMPS-AWSEM dumps but the LAMMPS engine paper itself is not cited. |
| Paszke 2019 — PyTorch (NeurIPS) | **MISSING** | NO | n/a (NeurIPS proceedings URL) | Hard runtime dependency; tagline literally says "Pure-PyTorch". |
| Cock 2009 — Biopython (Bioinformatics) | **MISSING** | NO | n/a | DOI `10.1093/bioinformatics/btp163`. Listed as runtime dep in `pyproject.toml` and used in `parser.py` (PDB parsing, altloc handling). |
| Sali & Blundell 1993 / Webb & Sali 2016 — Modeller | n/a | n/a | n/a | `docs/modeller_preprocessing_spec.md` explicitly establishes Modeller is **not** invoked by the installed frustrapy build and is not a dependency here. Citation **not required**. |

**Critical missing citations: 4** (Parra 2016 frustratometer-2 server, Thompson 2022 LAMMPS, Paszke 2019 PyTorch, Cock 2009 Biopython).
**Incomplete citations: 1** (Rausch 2021 frustratometeR — needs DOI + journal upgrade).

## 2. License / attribution issues

1. **`docs/reference_lammps_awsem/*.cpp` redistributes upstream LAMMPS-AWSEM source without an accompanying LICENSE.** The files carry their original headers — Sandia Corporation copyright (LAMMPS engine code, GPL-2.0) and Davtyan/Papoian copyrights for the awsemmd plugin — but the repo's only LICENSE file is Apache-2.0 at the root. **GPL-2.0 source distributed inside an Apache-2.0 repository is a license-compatibility flag a journal reviewer or downstream packager will spot.** Two acceptable fixes: (a) drop a `docs/reference_lammps_awsem/LICENSE.GPL-2.0` plus a `README.md` in that directory stating "this code is the upstream reference, mirrored verbatim from adavtyan/awsemmd and lammps/lammps under GPL-2.0; it is included for documentation cross-reference only and is not built or linked by `frustration_gpu`", or (b) move the directory out of the published source distribution (e.g. into a separate `reference/` ignored by `pyproject.toml` packaging) and link to upstream instead. Option (a) is cleaner.

2. **No NOTICE file at repo root.** Apache-2.0 §4(d) instructs adding a NOTICE file when distributing attributions for re-implemented works. The project is "re-implemented from published specifications" (README LL.135-137) — adding a minimal `NOTICE` listing AWSEM (Davtyan 2012), Ferreiro 2007 frustration index, and the frustratometeR classification rules as algorithmic sources would shut down "did you derive from GPL code?" objections.

3. **README §License paragraph is ambiguous.** "we re-implement the algorithms from published specifications rather than redistributing their source" is the right legal posture *for the Python code*, but it is contradicted by the verbatim C++ source files actually shipped in `docs/reference_lammps_awsem/`. Once §2.1 is resolved, this sentence should add: "Verbatim upstream LAMMPS-AWSEM C++ sources are mirrored under `docs/reference_lammps_awsem/` for cross-reference and retain their original GPL-2.0 license; see that directory's LICENSE."

## 3. Code-comment citation gaps (inline references that should be added)

- `src/frustration.py:9-10, 83, 160` — references "Ferreiro 2007" and the (-1.0, 0.78) thresholds. Add the PNAS DOI inline (`# Ferreiro et al. 2007, PNAS 104:19819-19824, doi:10.1073/pnas.0709915104`) at LL.83-84 next to the threshold constants. Currently only the surname+year appears.
- `src/parameters.py:1-35` — gamma.dat / burial_gamma.dat origin is "OpenAWSEM install at site-packages/openawsem/parameters/". Add a one-liner citing Lu 2021 PLOS Comp Biol with DOI for the gamma-table source. Right now the parameter origin is only described, not cited.
- `src/parameters.py:58` and other LAMMPS-source citations (e.g. `fix_backbone.cpp:5089`) — these point at internal file paths. At minimum, a top-of-module note "C++ line citations refer to the mirror under `docs/reference_lammps_awsem/`, copied from adavtyan/awsemmd commit <SHA>" would make every embedded line citation in the codebase fully reproducible.
- `src/debye_huckel.py:1-46` — describes the DH term parameters as matching LAMMPS-AWSEM but does not cite Davtyan 2012 inline.
- `src/water_mediated.py:1-10` — same: describes the AWSEM water-mediated energy but does not cite Davtyan 2012.
- No `src/*.py` file currently cites Rausch 2021 or Parra 2016 although the project inherits the Welltype/FrstState classification from frustratometeR (`src/frustration.py` LL.9-22 explicitly cites `inst/Scripts/RenumFiles.pl`).

## 4. "Inspired by" vs "derived from"

The README says "re-implement the algorithms from published specifications rather than redistributing their source" (legally protective language, correct for the *Python* code) — but the `src/frustration.py` and `src/density.py` modules carry comments like "matches `inst/Scripts/RenumFiles.pl`" and `src/parameters.py` says the gamma tables were "copied byte-exactly from the OpenAWSEM install … md5 verified at copy time." The legally-honest description for these specific items is *derived from* (gamma tables) and *behaviour-equivalent to* (RenumFiles.pl rules). Suggested wording for the journal Methods section is in §6.

## 5. Recommendations before submitting to a journal

In priority order (1 = blocker, 5 = polish):

1. **Add a LICENSE + README to `docs/reference_lammps_awsem/`** acknowledging GPL-2.0 upstream provenance, OR move it out of the distributed source tree. This is the single biggest publication risk.
2. **Add the four missing citations** (Parra 2016, Thompson 2022, Paszke 2019, Cock 2009) to `CITATION.cff` and the README primary-refs list. Upgrade the Rausch 2021 entry to a proper `type: article` entry with DOI.
3. **Add a root-level `NOTICE`** file enumerating algorithmic sources (Ferreiro 2007, Davtyan 2012, Lu 2021, frustratometeR rules) — this is the canonical Apache-2.0 attribution channel and silences "did you re-use GPL code?" objections without claiming you didn't.
4. **Patch the inline code-comment gaps** in §3 — five `src/*.py` files have surname+year citations that need DOIs added next to the constants/thresholds they document.
5. **Note in CHANGELOG `[Unreleased]`** that the citation+attribution audit landed (e.g. "Added: NOTICE, citation upgrades for Parra/Thompson/Paszke/Cock, LICENSE for reference_lammps_awsem mirror").

## 6. Suggested Methods paragraph (draft)

> We reimplemented the AWSEM Hamiltonian (Davtyan et al., 2012, *J. Phys. Chem. B* 116:8494, doi:10.1021/jp212541y) and the Ferreiro local-frustration index (Ferreiro et al., 2007, *PNAS* 104:19819, doi:10.1073/pnas.0709915104) as a pure-PyTorch (Paszke et al., 2019, *NeurIPS*) library, `frustration_gpu`. The water-mediated contact, burial, and Debye-Hückel terms were re-derived from the AWSEM specification and cross-validated against the upstream C++ implementation (LAMMPS-AWSEM, Davtyan & Papoian, `github.com/adavtyan/awsemmd`; LAMMPS engine, Thompson et al., 2022, *Comp. Phys. Comm.* 271:108171, doi:10.1016/j.cpc.2021.108171) on a 10-PDB reference panel: FI Spearman ≥ 0.9975 on every (PDB, mode) combination. The pair-level Welltype categories and the three-state highly/neutral/minimally classification thresholds (-1.0, 0.78) follow the conventions established by frustratometeR (Rausch et al., 2021, *Bioinformatics* 37:3038, doi:10.1093/bioinformatics/btab176) and its predecessor server (Parra et al., 2016, *Nucleic Acids Res.* 44:W356, doi:10.1093/nar/gkw304). PDB parsing uses Biopython (Cock et al., 2009, *Bioinformatics* 25:1422, doi:10.1093/bioinformatics/btp163). On a single RTX 4070, mutational-mode analysis is 30-53× faster than frustrapy CPU for proteins of 200-450 residues, with byte-comparable `tertiary_frustration.dat` / `5adens.dat` output.

## Summary

- Missing critical citations: **4**
- Incomplete citations: **1**
- License / attribution issues: **2** (mirrored GPL C++ without LICENSE; no NOTICE file)
- Top 3 fixes before submission: (i) LICENSE + README inside `docs/reference_lammps_awsem/` covering the GPL-2.0 upstream, (ii) add Parra 2016 + Thompson 2022 + Paszke 2019 + Cock 2009 to CITATION.cff and README, (iii) add a root-level `NOTICE` listing all algorithmic provenance with DOIs.
