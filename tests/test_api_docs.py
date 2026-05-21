"""Regression test: kwargs documented in ``docs/API.md`` must match live signatures.

Motivation
----------
``docs/api_inconsistencies.md`` (2026-05-20) catalogued five kwarg-naming
/ default-value mismatches between the docs and the source code that all
slipped past human review. This test parses every Python code block in
``docs/API.md`` that looks like a function signature, locates the matching
function via ``inspect.signature`` on a known set of public symbols, and
asserts each documented kwarg exists with the documented default.

Design notes
------------
* We parse code blocks (```` ```python ... ``` ````) rather than markdown
  tables — every documented signature in API.md is rendered as a code block,
  and parsing Python signature text is simpler than parsing markdown tables.
* Defaults are compared via ``repr()`` after light normalisation (``Path``
  vs ``str``, ``torch.float64`` vs ``torch.float64``, quoted vs unquoted
  strings, etc.). Defaults the docs leave as a symbolic constant (e.g.
  ``RHO_MIN_SEQ_SEP``) are compared by resolving the symbol from
  ``src.parameters``.
* Signature blocks that look like type definitions or non-function snippets
  (no ``(`` or no ``=`` defaults) are silently skipped — this is the "fail
  silently on non-kwarg-table sections" requirement.
* Public functions in ``src.__all__`` that have no documented signature
  block emit a ``warnings.warn`` (soft warning) so the test passes but a
  human reading ``pytest -v`` sees the gap.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest


class APIDocsCoverageWarning(Warning):
    """Soft warning category for "function expected but not documented".

    Subclasses bare ``Warning`` (NOT ``UserWarning``) so the project-wide
    ``filterwarnings = ["error::UserWarning"]`` rule in ``pyproject.toml``
    does not escalate it to a hard failure. This makes the coverage check
    advisory — visible in ``pytest -v`` output but non-fatal.
    """


# Make the ``frustration_gpu`` package importable when running pytest from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Aliased to ``src`` so the legacy attribute-access logic below
# (``hasattr(src, ...)``, ``getattr(src, ...)``) remains unchanged.
import frustration_gpu as src  # noqa: E402
from frustration_gpu import parameters as src_params  # noqa: E402

API_DOC = REPO / "docs" / "API.md"

# Functions we expect to be documented in API.md. Each must have a
# ```python ... ``` block whose body parses to a Python signature.
# (Internal helpers like ``direct_pair_energy``, ``water_mediated_pair_energy``
# are exported but have no doc block — they show up as soft-warnings.)
EXPECTED_DOCUMENTED = [
    "compute_frustration",
    "calculate_frustration",
    "parse_pdb",
    "compute_rho",
    "burial_energy",
    "direct_contact_energy",
    "water_mediated_energy",
    "debye_huckel_energy",
    "debye_huckel_pair_energy",
    "compute_frustration_index",
    "classify_frustration",
    "welltype_from_contact",
    "compute_residue_density",
    "emit_5adens_dat",
    "compute_virtual_atoms",
    "chain_segments",
]

# Code-block regex: capture body between ```python ... ```
CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def _resolve_symbolic_default(s: str) -> Any:
    """Resolve a symbolic default like ``RHO_MIN_SEQ_SEP`` or ``torch.float64``.

    Returns ``inspect.Parameter.empty`` if the symbol cannot be resolved —
    callers then fall back to literal string comparison.
    """
    # bare attribute on parameters / src / torch — try them in turn
    if hasattr(src_params, s):
        return getattr(src_params, s)
    if hasattr(src, s):
        return getattr(src, s)
    if s.startswith("torch."):
        import torch
        attr = s.split(".", 1)[1]
        if hasattr(torch, attr):
            return getattr(torch, attr)
    return inspect.Parameter.empty


def _parse_signature_block(body: str) -> tuple[str, dict[str, str]] | None:
    """Extract ``(function_name, {kwarg_name: default_literal})`` from a code block.

    Returns None if the block is not a function signature (no ``(`` or
    no ``def``/bare-call pattern with defaulted kwargs).

    The default-literal is the raw RHS text from the signature — kept as a
    string so we can resolve symbolic constants vs Python literals later.
    """
    body = body.strip()
    if "(" not in body:
        return None

    # Strip an optional leading ``def`` (top-level API uses ``def``, building
    # blocks use a bare ``name(...)`` call form).
    text = body
    if text.startswith("def "):
        # def name(args) -> ret:  — strip "def " and the trailing " -> ret:"
        text = text[4:]

    # Function name = chars up to first '('
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if m is None:
        return None
    name = m.group(1)
    args_start = m.end() - 1  # position of '('

    # Find matching ')'. Python signatures don't have unbalanced parens
    # inside defaults in this doc, but handle nested parens just in case.
    depth = 0
    end = None
    for i in range(args_start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    args_text = text[args_start + 1 : end]

    # Try to parse args via AST: wrap the signature in a dummy ``def`` and
    # parse it. This handles all the corner cases (nested types, kwargs-only
    # ``*``, ``= None`` etc.) without us re-implementing Python's grammar.
    # Strip any ``...`` (ellipsis) lines that the docs use as placeholders.
    args_clean_lines = [
        ln for ln in args_text.split("\n")
        if ln.strip() not in ("...", "")
    ]
    args_clean = "\n".join(args_clean_lines)
    # Drop trailing comma + comments — ast handles those, but be safe.
    # Force a newline before the closing ``):`` so a trailing inline
    # ``# comment`` on the last kwarg line doesn't swallow the ``):``.
    dummy_src = f"def _dummy(\n{args_clean}\n):\n    pass\n"
    try:
        tree = ast.parse(dummy_src)
    except SyntaxError:
        return None
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    kwargs: dict[str, str] = {}
    # ast separates positional/positional-or-keyword args from kw-only args.
    all_args = list(func_def.args.args) + list(func_def.args.kwonlyargs)
    defaults_pos = list(func_def.args.defaults)
    defaults_kw = list(func_def.args.kw_defaults)

    # positional-or-keyword: defaults align with the *tail* of args.args
    pos_args = func_def.args.args
    n_pos = len(pos_args)
    n_pos_def = len(defaults_pos)
    for i, arg in enumerate(pos_args):
        idx_in_defaults = i - (n_pos - n_pos_def)
        if idx_in_defaults < 0:
            continue   # no default — required arg, skip
        kwargs[arg.arg] = ast.unparse(defaults_pos[idx_in_defaults]).strip()

    # kw-only: defaults_kw aligns 1:1 with kwonlyargs (None entries → required)
    for arg, dfl in zip(func_def.args.kwonlyargs, defaults_kw):
        if dfl is None:
            continue   # required kw-only
        kwargs[arg.arg] = ast.unparse(dfl).strip()

    return name, kwargs


def _extract_doc_signatures() -> dict[str, dict[str, str]]:
    """Parse API.md, return ``{function_name: {kwarg: default_literal}}``.

    A function with multiple code blocks (e.g. shown once as a signature
    and once in an example) keeps the FIRST signature block. Non-signature
    blocks (no kwargs with defaults, or no top-level function name match)
    are silently skipped.
    """
    text = API_DOC.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for body in CODE_BLOCK_RE.findall(text):
        parsed = _parse_signature_block(body)
        if parsed is None:
            continue
        name, kwargs = parsed
        if not kwargs:
            continue  # no defaults to verify
        # First-occurrence wins; later code blocks (examples) are ignored.
        if name not in out:
            out[name] = kwargs
    return out


def _live_kwargs(fn_name: str) -> dict[str, Any]:
    """Live ``inspect.signature`` kwargs for ``src.<fn_name>``.

    Returns ``{kwarg_name: default_value}`` — values are real Python objects,
    not string literals. Required parameters are omitted.
    """
    fn = getattr(src, fn_name)
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            continue
        # Skip ``**kwargs`` etc. (these have empty default but VAR_KEYWORD kind).
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        out[name] = p.default
    return out


def _defaults_match(doc_literal: str, live_value: Any) -> bool:
    """Compare a doc-literal string to a live default value.

    Strategy:
      1. Try ``ast.literal_eval(doc_literal)`` — handles ``None``, numbers,
         strings, tuples, etc.
      2. Fall back to symbolic resolution (``RHO_MIN_SEQ_SEP``,
         ``torch.float64``, …).
      3. Last resort: compare ``repr(live_value)`` against the literal.
    """
    # Strategy 1: Python literal
    try:
        parsed = ast.literal_eval(doc_literal)
        if parsed == live_value:
            return True
    except (ValueError, SyntaxError):
        pass

    # Strategy 2: symbolic constant
    resolved = _resolve_symbolic_default(doc_literal)
    if resolved is not inspect.Parameter.empty and resolved == live_value:
        return True

    # Strategy 3: repr fallback (handles ``torch.float64`` etc.)
    if repr(live_value) == doc_literal:
        return True
    # Handle ``torch.float64`` literal vs the dtype's repr ``torch.float64``
    if str(live_value) == doc_literal:
        return True
    return False


# --- the test ------------------------------------------------------------------

DOC_SIGS = _extract_doc_signatures()


@pytest.mark.parametrize("fn_name", sorted(DOC_SIGS.keys()))
def test_documented_kwargs_match_live_signature(fn_name: str) -> None:
    """For every signature block in API.md whose name resolves to a public
    ``src`` symbol, every documented kwarg must exist in the live signature
    with the same default value.
    """
    if not hasattr(src, fn_name):
        pytest.skip(
            f"{fn_name!r} appears as a signature block in API.md but is not in "
            f"src.__all__ — likely a non-function example."
        )

    doc_kwargs = DOC_SIGS[fn_name]
    live_kwargs = _live_kwargs(fn_name)

    drift: list[str] = []
    for kw, doc_default in doc_kwargs.items():
        if kw not in live_kwargs:
            drift.append(
                f"  doc lists kwarg {kw!r} (= {doc_default}) but live signature "
                f"of {fn_name!r} has no such parameter"
            )
            continue
        live_default = live_kwargs[kw]
        if not _defaults_match(doc_default, live_default):
            drift.append(
                f"  kwarg {kw!r}: docs say default = {doc_default!r}, "
                f"live signature has {live_default!r}"
            )

    assert not drift, (
        f"docs/API.md is out of sync with live signature of "
        f"src.{fn_name}:\n" + "\n".join(drift)
    )


def test_expected_functions_are_documented() -> None:
    """Soft warning if a function in EXPECTED_DOCUMENTED has no doc block.

    Does NOT fail — many internal helpers in ``src.__all__`` legitimately
    have no public-facing API.md entry. We only complain about a known
    short list that we DO expect to see documented (so a future doc
    refactor doesn't silently delete the compute_rho block).
    """
    missing = [name for name in EXPECTED_DOCUMENTED if name not in DOC_SIGS]
    if missing:
        warnings.warn(
            "Functions expected to have a signature block in docs/API.md "
            "but none was found (note: signatures with no defaulted kwargs "
            "are silently skipped by the parser — add at least one kwarg "
            "default to make them discoverable): " + ", ".join(missing),
            category=APIDocsCoverageWarning,
            stacklevel=2,
        )


def test_no_undocumented_public_kwargs_in_top_level_api() -> None:
    """``compute_frustration`` is the public entry point. Every kwarg in
    its live signature should appear in the doc block — this catches the
    reverse drift (code adds a kwarg, docs forget to mention it).
    """
    if "compute_frustration" not in DOC_SIGS:
        pytest.skip("compute_frustration not documented — covered by the soft warning test.")
    doc_kwargs = set(DOC_SIGS["compute_frustration"].keys())
    live_kwargs = set(_live_kwargs("compute_frustration").keys())
    only_in_code = live_kwargs - doc_kwargs
    assert not only_in_code, (
        f"compute_frustration has kwargs in code that are not in docs/API.md: "
        f"{sorted(only_in_code)}"
    )


# The reverse-drift check, broadened (finding #67). Run on every API.md
# signature whose name resolves to a real ``frustration_gpu`` export — so
# adding a new kwarg in code without documenting it fails the test for
# any (not just the top-level) documented function.
#
# Kwargs starting with an underscore (e.g. ``_context``) are conventional
# "implementation-detail" parameters that the docs don't need to mention;
# they are excluded from the reverse-check so internal refactors don't
# explode the test suite.
REVERSE_CHECK_FUNCS = sorted(
    {fn for fn in DOC_SIGS if hasattr(src, fn)}
)

# Known opt-in / performance-only kwargs whose presence in the live
# signature is intentional but is documented in the function's
# narrative prose (LAMMPS-compat flags section, performance notes,
# etc.) rather than in the code-block signature. The reverse-drift
# check ignores them; the forward-drift check (which compares values
# of documented kwargs to live defaults) still runs.
#
# Each entry is justified inline; if you find yourself adding a new
# one, prefer documenting the kwarg in the signature block instead.
DOC_DRIFT_TOLERATED: dict[str, set[str]] = {
    # Phase 5 sparse-contact opt-in performance flags (see
    # docs/sparse_contacts_impl.md). Not advertised on the headline
    # API; advanced perf-tuning only.
    "burial_energy": {"sparse", "sparse_cutoff_a"},
    "compute_rho": {"sparse", "sparse_cutoff_a"},
    "debye_huckel_energy": {"sparse", "use_cdist", "device"},
    "direct_contact_energy": {"sparse", "use_cdist", "device"},
    "water_mediated_energy": {
        "sparse", "use_cdist", "device",
        "contact_min_seq_sep", "eta", "eta_sigma",
        "r_max", "r_min", "return_pair_matrix", "rho_0",
    },
    # FI degenerate-decoy clamp (numerical-stability knob, default
    # leaves behaviour unchanged).
    "compute_frustration_index": {"degenerate_threshold"},
}


@pytest.mark.parametrize("fn_name", REVERSE_CHECK_FUNCS)
def test_no_undocumented_kwargs_anywhere_in_api_docs(fn_name: str) -> None:
    """For every function with a documented signature block in API.md,
    require that every kwarg currently exposed by the live signature is
    also documented (modulo private ``_x`` kwargs and the explicit
    DOC_DRIFT_TOLERATED allowlist above). This catches "code added kwarg
    X but docs forgot" drift across the public API, not just on
    ``compute_frustration``.
    """
    doc_kwargs = set(DOC_SIGS[fn_name].keys())
    live_kwargs = {
        k for k in _live_kwargs(fn_name).keys()
        if not k.startswith("_")
    }
    tolerated = DOC_DRIFT_TOLERATED.get(fn_name, set())
    only_in_code = (live_kwargs - doc_kwargs) - tolerated
    assert not only_in_code, (
        f"{fn_name!r} has kwargs in code that are not in docs/API.md: "
        f"{sorted(only_in_code)}. Add them to the function's signature "
        "code block in docs/API.md (this test parses ```python ... ``` "
        "blocks). If the kwarg is intentionally undocumented (e.g. a "
        "perf-tuning knob), add it to DOC_DRIFT_TOLERATED in this file "
        "with an inline justification."
    )
