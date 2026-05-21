# Optional: LAMMPS-AWSEM recompile for full float64 precision

**Status**: deferred. Not blocking any phase. Pick up whenever the current 1e-7 precision feels limiting.

**Why**: our PyTorch port matches LAMMPS-AWSEM at 2.7e-7 (5AON) and 1.4e-8 (11BG) relative error on V_water. The residual is entirely **print-truncation in the C++ output**, not any computational difference. To get below 1e-7 we need the raw float64 values out of LAMMPS, which means modifying the print format strings in `fix_backbone.cpp` and recompiling.

## The 4-line patch (already located)

Edit `adavtyan/awsemmd/src/fix_backbone.cpp`:

| Line | Output file | Current | Patch to |
|---|---|---|---|
| **7669** | `energy.log` (V_water, V_burial, etc.) | `"\t%8.6f"` (6-decimal trunc) | `"\t%.15g"` |
| **7670** | `energy.log` (V_total) | `"\t%8.6f\n"` | `"\t%.15g\n"` |
| **5104** | `tertiary_frustration.dat` (configurational + mutational per-pair: coords, rho, native_energy, decoy_mean, std, FI) | `"%5d %5d %3d %3d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %c %c %8.3f %8.3f %8.3f %8.3f\n"` (3-decimal trunc) | Replace each `%8.3f` with `%.15g` (keep `%5d`, `%3d`, `%c` unchanged) |
| **5168** | `tertiary_frustration.dat` (singleresidue mode) | Similar `%8.3f` mix | Replace each `%8.3f` with `%.15g` |

That's the entire patch. Mechanically isolated. Cannot introduce any computational bug — only changes how values are printed.

## Recompile recipe (Linux VM)

```bash
ssh root@10.1.0.45
apt install -y build-essential gfortran libopenmpi-dev    # if not already
cd /tmp
git clone https://github.com/adavtyan/awsemmd.git
git clone -b release https://github.com/lammps/lammps.git lammps

# Apply the 4-line print-format patch to awsemmd/src/fix_backbone.cpp

# Build LAMMPS with AWSEM-MD as a USER package:
cp awsemmd/src/* lammps/src/
cd lammps/src
make yes-MOLECULE yes-USER-AWSEMMD     # adjust based on awsemmd's install instructions
make serial -j32

# Result: lmp_serial binary with the high-precision printf in it
# Drop into frustrapy:
cp lmp_serial /root/pyenvs/tuhnon/lib/python3.13/site-packages/frustrapy/core/scripts/lmp_serial_12_Linux
cp lmp_serial /root/pyenvs/tuhnon/lib/python3.13/site-packages/frustrapy/core/scripts/lmp_serial_3_Linux

# Verify: re-run on 5AON, check that energy.log now has more digits
cd /tmp/awsem_validation
python -c "import frustrapy; frustrapy.calculate_frustration(pdb_file='5AON.pdb', mode='configurational', results_dir='/tmp/awsem_validation/highprec', graphics=False, debug=True)"
cat /tmp/awsem_validation/highprec/5AON.done/energy.log
# Should see V_water like -18.700275955619... (15 significant figures) instead of -18.700281
```

## Re-dump + revalidate after recompile

1. Re-run all 10 panel PDBs × 3 modes + sweeps (the same VM script we already have at `/tmp/awsem_validation/run_dump_8.py`, `run_full_modes.py`, `run_sweeps.py`)
2. Pull new dumps to `benchmark/cpu_baseline_highprec/` (parallel directory; keep current `cpu_baseline/` as historical record)
3. Update validation tests in `tests/test_water_mediated.py` etc.:
   - Change `assert abs(diff) / abs(target) < 0.001` → `< 1e-12`
   - 5AON expected: -18.700275955619 (or whatever the new value is)
   - 11BG expected: -147.390849025519
4. Re-run full test suite — should now pass at machine precision
5. Update `PHASES_ROADMAP.md` Phase 2b numerical results section with the new precision-floor numbers

## What this buys you

**For Phase 2 (deterministic energies):** 1e-7 → 1e-15. Clean float64 floor. Any future deviation is a REAL bug, not print noise.

**For Phase 3+ (stochastic decoy stuff):** modest — RNG noise floor (~3%) dominates regardless. Going from 3e-4 print precision to 1e-15 here matters because we'd see RNG noise CLEARLY instead of having it hidden by print truncation. So debugging Phase 3 issues becomes much easier.

**For Phase 4 (per-residue density):** doesn't matter — densities are integer-counted fractions, already at exact precision.

## Cost estimate

- VM build environment setup: 1-2 hours (if not already configured)
- Patch + compile: 1-2 hours
- Re-dump 10 PDBs × 3 modes: ~30 min (already have the run scripts)
- Re-validate + update tests: 1-2 hours
- **Total: 4-8 hours on the VM**

## When to actually do this

Skip until one of these triggers:
1. A Phase 3 bug looks suspicious and the 3-decimal print precision makes it impossible to diagnose
2. Want to publish results claiming "machine-precision match with LAMMPS-AWSEM" rather than "0.1% match"
3. Phase 2c (DebyeHuckel) validation comes out at marginal precision and we want to verify it's just print truncation not a real bug

Otherwise, current 1e-7 is plenty for shipping a deployment-ready GPU port. The decision should be cost vs. cleanliness.

## Related files (for future reference)

- C++ source mirrored: `docs/reference_lammps_awsem/fix_backbone.cpp` (lines 5104, 5168, 7669, 7670)
- Current dumps: `benchmark/cpu_baseline/`
- Phase 2b validation script that would gain precision: `tests/test_water_mediated.py`
