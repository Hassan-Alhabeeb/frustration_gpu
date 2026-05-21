"""Placeholder validation script — to be implemented after user signs off on the plan.

Will load cpu_baseline/<PDB>_frustrapy.npz and gpu_outputs/<PDB>_gpu.npz and compute:
  - per-pair FI matrix: Pearson r, MAE
  - per-residue density: Spearman ρ, MAE, max abs diff
  - top-30 most-frustrated residue overlap
  - timing comparison
  - failure-mode flags (any PDB where outputs diverge beyond tolerance)

Output: results/accuracy_comparison.csv and results/speed_benchmark.csv.

NOT IMPLEMENTED YET. Pending user-approved plan.
"""
raise NotImplementedError("Pending user plan. See README.md.")
