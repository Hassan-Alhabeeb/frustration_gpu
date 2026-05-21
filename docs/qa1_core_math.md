# QA-1: Core math modules review

Reviewer: Opus 4.7. Scope: Phases 1-2 modules (parser, virtual_atoms,
parameters, burial, _contact_common, direct_contact, water_mediated,
debye_huckel, contact_gamma).

## Verdict

YELLOW — no critical / wrong-numerics findings against the C++ baseline on
the default protein-only path. One genuine HIGH-severity latent bug
(`-1`-sentinel residue types silently index into amino-acid tables) and a
handful of MEDIUM doc-vs-code drifts that will mislead future readers.
Math is otherwise faithful to `fix_backbone.cpp`.

## Critical (0)

(none)

## High (1)

- [`burial.py:249`, `direct_contact.py:316`, `water_mediated.py:306-307`]
  DNA placeholder residues (`residue_types == -1`) silently index gamma
  tables via Python-style negative indexing. `burial_gamma[aa]`,
  `gamma_direct[aa.unsqueeze(1), aa.unsqueeze(0)]`, and the mediated lookups
  all treat `-1` as "last row" (VAL row), so a caller who hands a parsed
  dict from `parse_pdb(..., include_dna=True)` straight to
  `burial_energy` / `direct_contact_energy` / `water_mediated_energy`
  gets numerically valid but biologically nonsensical output, with no
  warning. Only `compute_frustration.py` filters DNA rows; the public
  energy APIs do not. Suggested fix: add a defensive
  `if (aa < 0).any(): raise ValueError(...)` at the top of each energy
  function, or auto-mask `aa < 0` rows out of the pair mask.
  (`debye_huckel_energy` lucks out: charge vector index 19 = V = 0.0, so
  -1 → 0 charge → no contribution; still worth a defensive guard.)

## Medium (4)

- [`burial.py:5-9, 92-94`] Docstring claims rho is restricted to
  `|i-j| > 2`, but after the 2026-05-20 off-by-one fix the constant
  `RHO_MIN_SEQ_SEP = 1` combined with `seq_diff > min_seq_sep` implements
  `|i-j| >= 2` (i.e. `> 1`). The CODE is correct against
  `smart_matrix_lib.h:638` (`abs(res_no[j]-res_no[i])>1`); the COMMENT is
  stale. Same drift in the `min_seq_sep` parameter docstring at
  `burial.py:91-94` ("Default 2 means pairs with |i - j| > 2") — the
  default is now 1 and the inequality is strict. Fix the prose to
  match the post-fix constant.

- [`_contact_common.py:241-248`] The decoy-replacement comment says
  "distant enough that even after subtraction the result is a normal
  finite number". But two NaN rows both get the SAME decoy value (1e6),
  so their pairwise `diff = 0`, `vector_norm = 0`, not "distant". The
  invariant that protects against `1/r` blowup is layer 2 (the mask
  swap to `fill_value`), not the decoy itself. Reword the comment.

- [`parser.py:148`] `chain_id = line[21:22].strip() or "A"`. PDBs with
  multiple chains where one has a blank chain ID would silently merge
  the blank chain into chain "A". Rare in practice but a confusion
  trap. Either preserve `""` as a distinct chain key or raise on
  ambiguous mixing.

- [`parser.py:600` `chain_segments`] Public helper defined but never
  used in the module set under review. Either remove or document as
  external API. Same module also re-implements chain-index logic
  inline in `burial.py:182-189` rather than calling
  `_contact_common._build_chain_index` — minor duplication risk.

## Low (5)

- [`parameters.py:82, 112`] `load_burial_gamma` and `load_gamma_tables`
  default to `dtype=torch.float32`. For an ML pipeline targeting
  1e-14 parity tests, a float32 default is a discipline lapse — every
  caller must remember to pass `dtype=torch.float64`. Consider flipping
  the default to float64 with a documented one-line override for GPU.

- [`parameters.py:131-133`] `load_gamma_tables` silently duplicates a
  single-column row into both columns (`raw.append([parts[0], parts[0]])`).
  Defensive but masks a real malformed-file condition; raising would
  be safer. Same module: rows with `< 2` columns and `!= 1` columns
  fall through without error.

- [`virtual_atoms.py:99-103`] Python-level loop to build
  `has_prev` / `has_next` masks. Easily vectorizable
  (`has_prev[1:] = chain_t[1:] == chain_t[:-1]`); no correctness
  impact but allocates a chain_index tensor anyway and the loop is
  warm on long inputs.

- [`debye_huckel.py:339`] `inv_r = 1.0 / safe_dist.clamp(min=1e-12)`.
  In autograd mode, `clamp(min=...)` has a sub-gradient discontinuity
  at the cutoff. Not triggered in practice (CB-CB distances are
  bounded below by ~3.8 Å on real proteins and `fill_value=1000` on
  masked entries), so harmless — but the clamp comment says "tiny
  epsilon ≪ float64 precision" while 1e-12 is well above float64
  epsilon (≈2.2e-16). Either tighten the clamp or update the comment.

- [`parser.py:331-335`] When `include_dna=True`, DNA placeholders are
  given a CA-proxy by `setdefault("CA", ...)`. They then survive the
  `if "CA" not in r["atoms"]` filter and reach the `is_dna` branch.
  Fine, but the design tangles "DNA-as-CA-proxy" into the protein
  CA pipeline; the downstream energy modules become the only place
  this leaks. See the HIGH finding for the consequence.

## Notes / FYI items

- The Phase-1.5 sign / averaging / k_water-folding decisions are
  faithfully implemented and documented. The "no 1/2 factor in
  V_direct" call against `fix_backbone.cpp:5462-5473` is correct —
  the C++ averages two identical columns (line 5462's
  `sigma_gamma_direct = (γ_0 + γ_1)/2` followed by `direct_contact = true`
  short-circuit on line 3431) which collapses to the identity.
- `_pair_mask` cross-chain handling diverges semantically from the C++
  DH path (C++ checks only `|i_resno - j_resno| < min_sep`, ignoring
  chain), but because LAMMPS-AWSEM uses GLOBAL residue indexing
  (`res_no[i]` is monotone across chains), the two paths agree in
  practice. Worth a one-line comment in `_pair_mask` to head off future
  confusion.
- HIS = 0 charge convention matches `fix_backbone.cpp:5511-5527`
  exactly. Verified.
- Burial seq-sep logic (`>1`) matches `smart_matrix_lib.h:638` and the
  burial-potential gather at `fix_backbone.cpp:3546`. Verified.
- Water/mediated seq-sep gate (`>=contact_cutoff`, default 2) matches
  `fix_backbone.cpp:7243`. Verified.
- `_pairwise_distance_safe` autograd-safety story is sound: layer-1
  coordinate sanitisation + layer-2 distance-fill substitution. NaN
  gradients cannot escape. Diagnostic `dist` is correctly detached
  (`with torch.no_grad():`). Good design.
- DH `screening_length / k_screening` API matches
  `fix_backbone.cpp:5544-5545` byte-for-byte; epsilon factor included.
- contact_gamma.py wraps `parameters.load_gamma_tables` cleanly; no
  duplicated logic, no precision loss.
- No file handle leaks, no GPU/CPU mixing in any code path I traced.
- All energy modules guard `n < 2` with a typed zero tensor and return
  consistent empty matrices when `return_pair_matrix=True`. Good
  defensive coding.
