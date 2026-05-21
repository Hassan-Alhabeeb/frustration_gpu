"""Diff two ``phase5_panel_results.csv`` files.

Use case: comparing a candidate run (e.g. after a code change, or on
different hardware) against a baseline. Emits a Markdown summary of
the per-(PDB, mode, device) deltas in wall-clock and status, plus
basic regression flags (status flips, > 2x slowdown).

Usage::

    python benchmark/compare.py baseline.csv candidate.csv [-o report.md]

Both CSVs must follow the schema emitted by ``run_phase5.py``: header
columns including ``pdb``, ``mode``, ``device``, ``status``, ``wall_ms``,
``n_residues``, ``n_pairs``, ``peak_vram_mb``.

Exits 0 if the candidate matches the baseline within tolerance, 1 if
any regression is detected.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Wall-clock delta beyond which we flag as a regression.
SLOWDOWN_FLAG = 2.0  # candidate >= 2x baseline


def _load(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Load a phase5_panel_results.csv into {(pdb, mode, device): row}."""
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row["pdb"], row["mode"], row["device"])
            out[key] = row
    return out


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def compare(baseline_path: Path, candidate_path: Path) -> Tuple[List[str], int]:
    """Return (markdown_lines, exit_code).

    ``exit_code`` is 0 if no regressions, 1 otherwise.
    """
    base = _load(baseline_path)
    cand = _load(candidate_path)

    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    both = sorted(set(base) & set(cand))

    lines: List[str] = []
    a = lines.append
    a(f"# Phase 5 comparison: {candidate_path.name} vs {baseline_path.name}\n")
    a(f"Baseline rows: {len(base)}; candidate rows: {len(cand)}; "
      f"intersection: {len(both)}.\n")

    regressions = 0
    a("## Per-(PDB, mode, device) timing + status\n")
    a("| PDB | mode | device | base ms | cand ms | ratio (cand/base) | base status | "
      "cand status | flag |")
    a("|-----|------|--------|---------|---------|---------|-------------|"
      "-------------|------|")
    for key in both:
        bk = base[key]
        ck = cand[key]
        bm = _to_float(bk.get("wall_ms", "nan"))
        cm = _to_float(ck.get("wall_ms", "nan"))
        if bm > 0 and cm > 0:
            ratio = cm / bm
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio = float("nan")
            ratio_str = "n/a"
        bs = bk.get("status", "?")
        cs = ck.get("status", "?")
        flag = ""
        if bs != cs:
            flag = "STATUS-FLIP"
            regressions += 1
        elif ratio == ratio and ratio >= SLOWDOWN_FLAG:
            flag = f">{SLOWDOWN_FLAG:.0f}x SLOWER"
            regressions += 1
        a(f"| {key[0]} | {key[1]} | {key[2]} | "
          f"{bm:.1f} | {cm:.1f} | {ratio_str} | {bs} | {cs} | {flag} |")

    if only_base:
        a("\n## Rows present only in baseline\n")
        for k in only_base:
            a(f"- `{k[0]}` / `{k[1]}` / `{k[2]}`")
        regressions += len(only_base)  # missing rows are a regression
    if only_cand:
        a("\n## Rows present only in candidate (new coverage)\n")
        for k in only_cand:
            a(f"- `{k[0]}` / `{k[1]}` / `{k[2]}`")

    a("\n## Summary\n")
    if regressions == 0:
        a("OK: no regressions detected.")
    else:
        a(f"REGRESSIONS: {regressions} row(s) flagged.")

    return lines, (1 if regressions else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline", type=Path,
                    help="Path to baseline phase5_panel_results.csv")
    ap.add_argument("candidate", type=Path,
                    help="Path to candidate phase5_panel_results.csv")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="If set, write the Markdown report to this path; "
                         "otherwise print to stdout.")
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 2
    if not args.candidate.exists():
        print(f"candidate not found: {args.candidate}", file=sys.stderr)
        return 2

    lines, exit_code = compare(args.baseline, args.candidate)
    text = "\n".join(lines)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text, encoding="utf-8")
        print(f"-> {args.output}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
