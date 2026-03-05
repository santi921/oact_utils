---
title: "fix: Re-run RKS singlet calculations with UKS for actinide multi-spin"
type: fix
date: 2026-02-27
---

# fix: Re-run RKS Singlet Calculations with UKS for Actinide Multi-Spin

## Overview

Singlet (spin_1) calculations in the multi-spin campaign were run with **RKS** (Restricted Kohn-Sham) instead of **UKS** (Unrestricted Kohn-Sham). This is incorrect for actinide systems where open-shell singlet character is expected. ORCA defaults to RKS for singlets unless explicitly told otherwise; for non-singlet states (spin > 1), ORCA automatically uses UKS.

The fix involves regenerating inputs for 56 completed spin_1 jobs using the **current codebase**, which now correctly adds UKS + symmetry breaking for actinide singlets (via `get_orca_blocks()` auto-detection at `calc.py:509-519`).

## Problem Statement

**Root cause**: The `multi_spin_from_converged_calcs.py` script set up singlet calculations before the automatic UKS detection was added to `get_orca_blocks()`. The `ks_method` parameter was not passed through `wrapper_write_job_folder`, and ORCA defaulted to RKS for all `mult=1` jobs.

**Impact**: 56 completed spin_1 jobs (29 omol + 27 x2c) across 6 ligand categories have incorrect wavefunctions. Energy gaps, population analyses, and geometry optimizations for the singlet state are unreliable.

**Affected data**:

| LOT | COT | Crown | Dithio | Nitrates | carbenes | tris-Cp | Total |
|-----|-----|-------|--------|----------|----------|---------|-------|
| omol | 6 | 2 | 8 | 6 | 1 | 6 | **29** |
| x2c | 8 | 3 | 6 | 5 | 1 | 4 | **27** |

Additionally, 10 failed spin_1 jobs (6 omol, 4 x2c) may have failed **because** of the RKS/UKS issue and should also be regenerated.

## Proposed Solution

Create a `rerun_singlets_uks.py` script that:
1. Reads completed spin_1 geometries from the multi-spin DB
2. Extracts charge from existing ORCA input files in each job directory
3. **Renames** existing `spin_1` folders to `spin_1_rks` (preserving old RKS results)
4. Regenerates inputs in fresh `spin_1` folders using the current code (which auto-adds UKS + symmetry breaking)

Then use existing infrastructure to re-submit, re-collect, and re-analyze. The old RKS data remains accessible in `spin_1_rks` for comparison.

## Technical Approach

### Phase 1: New Re-run Script

**New file**: `oact_utilities/scripts/multi_spin/rerun_singlets_uks.py`

This script:

1. **Connects to DB** at the path provided (default: `data/multi_spin/multi_spin_jobs_v2.sqlite3`)
2. **Queries spin_1 jobs** with `status IN (1, -1)` — both completed and failed
3. **For each job**:
   - Reads `final_coords` and `final_elements` from DB (for completed jobs)
   - For failed jobs without geometry, falls back to reading `orca.xyz` from the job directory
   - Reads charge from existing `orca.inp` in the job directory (parses `* xyz <charge> <mult>` line)
   - Determines `lot` from the DB record (omol or x2c)
   - Determines whether the original was a Sella run (checks for `sella.log` or `opt.traj`)
   - **Renames** existing `spin_1` folder to `spin_1_rks` (preserves old RKS data)
   - Creates a fresh `spin_1` folder
   - Calls `wrapper_write_job_folder()` with the correct parameters to regenerate UKS inputs
4. **Reports** a summary of regenerated vs skipped jobs

**Folder rename logic**:

```python
import shutil

spin_1_dir = row["path"]  # e.g., .../molecule/spin_1
spin_1_rks_dir = spin_1_dir.replace("/spin_1", "/spin_1_rks")

if os.path.exists(spin_1_dir) and not os.path.exists(spin_1_rks_dir):
    shutil.move(spin_1_dir, spin_1_rks_dir)
    # Read charge from the RENAMED directory's orca.inp
    charge = extract_charge_from_orca_inp(spin_1_rks_dir)
elif os.path.exists(spin_1_rks_dir):
    # Already renamed (re-running script is idempotent)
    charge = extract_charge_from_orca_inp(spin_1_rks_dir)
```

**Charge extraction** from existing `orca.inp`:

```python
def extract_charge_from_orca_inp(job_dir: str) -> int | None:
    """Parse charge from ORCA input file's '* xyz <charge> <mult>' line."""
    inp_file = os.path.join(job_dir, "orca.inp")
    if not os.path.exists(inp_file):
        return None
    with open(inp_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("* xyz") or line.startswith("*xyz"):
                parts = line.split()
                # Format: * xyz <charge> <mult>
                if len(parts) >= 4:
                    return int(parts[2])
    return None
```

**Geometry reconstruction** from DB:

```python
import json
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

def atoms_from_db_row(row: dict) -> Atoms | None:
    """Reconstruct ASE Atoms from DB final_coords and final_elements."""
    coords = json.loads(row["final_coords"])
    elements = json.loads(row["final_elements"])

    # Handle atomic numbers vs symbols
    if elements and isinstance(elements[0], (int, str)):
        if isinstance(elements[0], str) and elements[0].isdigit():
            elements = [chemical_symbols[int(e)] for e in elements]
        elif isinstance(elements[0], int):
            elements = [chemical_symbols[e] for e in elements]

    return Atoms(symbols=elements, positions=np.array(coords))
```

**LOT-specific configuration**: The script must replicate the original setup from `multi_spin_from_converged_calcs.py:main()`:

| Parameter | omol | x2c |
|-----------|------|-----|
| `functional` | wB97M-V | PBE0 |
| `simple_input` | omol | x2c |
| `actinide_basis` | ma-def-TZVP | cc_pvtz_x2c.bas (file path) |
| `actinide_ecp` | def-ECP | None |
| `non_actinide_basis` | def2-TZVPD | X2C-TZVPPall |

CLI:
```bash
python rerun_singlets_uks.py <db_path> \
    --orca-exe /path/to/orca \
    --dry-run  # Preview what would be regenerated
    --include-failed  # Also regenerate failed spin_1 jobs
    --cores 24 --n-hours 24 --queue pbatch --allocation dnn-sim
```

### Phase 2: Fix Infrastructure Gaps

Three gaps in the multi-spin infrastructure should be fixed to prevent this class of bug:

#### 2a. Add `ks_method` to `wrapper_write_job_folder`

**File**: `oact_utilities/scripts/multi_spin/multi_spin_from_converged_calcs.py`

Add `ks_method: str | None = None` parameter and pass it through to both `write_orca_inputs()` calls (line 189 Sella path, line 248 ORCA-direct path).

#### 2b. Add `ks_method` to `write_inputs_ase`

**File**: `oact_utilities/utils/create.py`

Add `ks_method: str | None = None` parameter and include it in the generated `orca.py` script's call to `pure_ase_relaxation()`.

#### 2c. Fix potential double-UKS in `get_orca_blocks`

**File**: `oact_utilities/core/orca/calc.py`

If `ks_method="uks"` is passed AND the molecule is an actinide singlet, both the explicit `ks_method` append (line 502) and the auto-detection (line 512) fire, producing `UKS UKS` in the simple input. Add a guard:

```python
# Line 509-512, change to:
needs_symm_break = mult == 1 and has_actinides

if needs_symm_break:
    # Only add UKS if not already explicitly set via ks_method
    if ks_method is None or ks_method.upper() != "UKS":
        orcasimpleinput += " UKS"
    orcablocks.append(
        get_symm_break_block(atoms, charge=charge, all_electron_elements=all_electron_elems)
    )
```

### Phase 3: Add `charge` column to multi-spin DB (Optional Enhancement)

**File**: `oact_utilities/scripts/multi_spin/check_multi_spin.py`

Follow the existing migration pattern (currently at `SCHEMA_VERSION = 2`):

- Bump to `SCHEMA_VERSION = 3`
- Add `_apply_migration_v3()` with `charge INTEGER` column
- Backfill charge from ORCA input files during `--update` runs

This ensures charge data is available for future analysis without requiring the Excel file.

### Phase 4: Re-submit, Re-collect, Re-analyze

These use **existing scripts** with no modifications:

1. **Submit**: `python run_multi_spin.py /p/lustre5/vargas58/maria_benchmarks/multi_spin/ --skip-done`
   - Fresh `spin_1` folders have no completed output, so all will be submitted
   - Old RKS results are preserved in `spin_1_rks` folders

2. **Collect**: `python check_multi_spin.py /p/lustre5/vargas58/maria_benchmarks/multi_spin/ --parse-charges --verbose-results --print-table`
   - The DB `ON CONFLICT(path) DO UPDATE` clause will overwrite old spin_1 results with new UKS data
   - The `spin_1_rks` folders will also be collected as separate entries (different paths)

3. **Analyze**: Run `analyze_spin_states.ipynb`
   - Consider adding a verification cell that checks the ORCA output for `UKS` keyword to confirm the fix

## Acceptance Criteria

### Functional Requirements

- [ ] `rerun_singlets_uks.py` reads spin_1 geometries from DB and regenerates ORCA inputs with UKS
- [ ] Existing `spin_1` folders renamed to `spin_1_rks` before regeneration
- [ ] All 56 completed + 10 failed spin_1 jobs have regenerated inputs in fresh `spin_1` folders
- [ ] Generated `orca.inp` files contain `UKS` keyword in the simple input line
- [ ] Generated `orca.inp` files contain `%scf rotate` symmetry breaking block
- [ ] Charge is correctly extracted from existing ORCA inputs and preserved
- [ ] LOT-specific parameters (functional, basis, ECP) match the originals
- [ ] `--dry-run` mode previews without writing
- [ ] `--include-failed` flag regenerates failed jobs too

### Infrastructure Fixes

- [ ] `wrapper_write_job_folder` accepts and forwards `ks_method`
- [ ] `write_inputs_ase` accepts and forwards `ks_method` in generated script
- [ ] `get_orca_blocks` does not produce duplicate `UKS` keywords
- [ ] Existing tests pass
- [ ] New tests cover `ks_method` propagation through multi-spin path

### Quality Gates

- [ ] All existing tests pass: `pytest tests/`
- [ ] Linting passes: `ruff check . && black --check .`
- [ ] Type checking passes: `mypy oact_utilities/`

## Implementation Files

| File | Action | Description |
|------|--------|-------------|
| `oact_utilities/scripts/multi_spin/rerun_singlets_uks.py` | **CREATE** | Main re-run script |
| `oact_utilities/scripts/multi_spin/multi_spin_from_converged_calcs.py` | EDIT | Add `ks_method` to `wrapper_write_job_folder` |
| `oact_utilities/utils/create.py` | EDIT | Add `ks_method` to `write_inputs_ase` |
| `oact_utilities/core/orca/calc.py` | EDIT | Fix double-UKS guard in `get_orca_blocks` |
| `oact_utilities/scripts/multi_spin/check_multi_spin.py` | EDIT | (Optional) Add charge column migration |
| `tests/test_calculator.py` | EDIT | Add double-UKS prevention test |
| `tests/test_check_multi_spin.py` | EDIT | (Optional) Add charge column test |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Charge extraction fails for some jobs | Low | High | Fall back to Excel file; `--dry-run` to verify first |
| Rename fails (permissions/disk) | Low | Medium | `--dry-run` to verify first; rename is atomic on same filesystem |
| UKS singlets don't converge | Medium | Medium | Current code includes symmetry breaking rotation; monitor SCF convergence |
| Sella path doesn't get UKS | Low | High | Auto-detection fires in `get_orca_blocks()` regardless of call path |

## References

### Internal
- UKS auto-detection: [calc.py:504-519](oact_utilities/core/orca/calc.py#L504-L519)
- Symmetry breaking: [calc.py:286-324](oact_utilities/core/orca/calc.py#L286-L324)
- Original setup: [multi_spin_from_converged_calcs.py:342-625](oact_utilities/scripts/multi_spin/multi_spin_from_converged_calcs.py#L342-L625)
- DB schema migrations: [check_multi_spin.py:224-369](oact_utilities/scripts/multi_spin/check_multi_spin.py#L224-L369)
- KS method plan: [2026-02-26-feat-add-ks-wavefunction-type-flag-plan.md](docs/plans/2026-02-26-feat-add-ks-wavefunction-type-flag-plan.md)

### ORCA
- UKS for singlets requires explicit keyword in ORCA 6.0+
- `%scf rotate` block breaks alpha/beta symmetry to avoid collapsing to RKS solution
- 20-degree rotation angle (~12% LUMO-into-HOMO mixing) matches Q-Chem defaults
