# Test Data Files

This directory contains example ORCA output files used for testing the workflow parsers.

## Contents

### `orca_direct_example/`
Example output from a direct ORCA geometry optimization run (AmO molecule).

**Key files:**
- `logs` - Main ORCA output file (text format)
- `AmO_orca.engrad` - Energy and gradient file
- `AmO_orca.xyz` - Final optimized geometry
- `AmO_orca_trj.xyz` - Optimization trajectory

**Used by tests:**
- `test_parse_max_forces_direct`
- `test_parse_scf_steps_direct`
- `test_parse_final_energy_direct`
- `test_parse_job_metrics_direct`

### `quacc_example/`
Example output from a quacc ORCA run (NpF3 molecule).

**Key files:**
- `orca.out.gz` - Main ORCA output (gzipped)
- `orca.engrad.gz` - Energy and gradient file (gzipped)
- `results.pkl` - Quacc results pickle file

**Used by tests:**
- `test_parse_max_forces_quacc_gzipped`
- `test_parse_job_metrics_quacc`

## Purpose

These files test that the parser functions in `oact_utilities/utils/analysis.py` can:
1. Extract max forces from ORCA output
2. Extract SCF iteration counts
3. Extract final energies
4. Handle both regular and gzipped files (quacc format)
5. Gracefully handle missing or corrupted files
6. Extract max forces from `.engrad` files

## Original Sources

- `orca_direct_example`: `/data/orca_am66/AmO_done/`
- `quacc_example`: `/data/an66_quacc/NpF3/quacc-2025-12-22-23-40-36-847376-59738/`
