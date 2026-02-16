---
status: complete
priority: p1
issue_id: "005"
tags: [data-integrity, scientific, validation, critical]
dependencies: []
---

# Add Charge and Spin Conservation Validation

CRITICAL: No validation that charges sum to total charge or spins match multiplicity.

## Problem Statement

`parse_mulliken_population()` extracts atomic charges and spins but does not validate physical conservation laws. For quantum chemistry data:
- **Charges must sum to total molecular charge** (usually 0 for neutral molecules)
- **Spins must sum to match spin multiplicity** (2S+1 where S = sum of spins / 2)

Missing validation means corrupted or incorrect ORCA output could silently propagate bad data through workflows.

**Scientific Impact:**
- **HIGH SEVERITY** - Violates fundamental physics principles
- Silent data corruption possible
- Invalid results could be published
- Workflow continues with nonsense data
- Undermines scientific integrity of results

## Findings

**Location:** `oact_utilities/utils/analysis.py:723-820`

**Current behavior:**
```python
def parse_mulliken_population(output_file):
    # Parses charges and spins
    # Returns dictionary
    # ❌ NO VALIDATION of conservation laws
    return {
        "mulliken_charges": charges,
        "mulliken_spins": spins,
        # ...
    }
```

**Missing checks:**
1. Sum of charges ≈ total molecular charge (within tolerance)
2. Sum of spins ≈ expected from multiplicity (within tolerance)
3. Individual charges/spins within physical bounds
4. Consistency between Mulliken and Loewdin results

**Identified by:** data-integrity-guardian agent during code review

## Proposed Solutions

### Option 1: Add Validation with Warning (Recommended)

**Approach:** Validate conservation laws and warn on violations

```python
def parse_mulliken_population(
    output_file: str | Path,
    expected_charge: int = 0,
    expected_multiplicity: int | None = None,
    tolerance: float = 0.01
) -> dict[str, list] | None:
    # ... parse data ...

    # Validate charge conservation
    charge_sum = sum(charges)
    if abs(charge_sum - expected_charge) > tolerance:
        warnings.warn(
            f"Charge conservation violated: sum={charge_sum:.3f}, "
            f"expected={expected_charge}, diff={abs(charge_sum - expected_charge):.3f}"
        )

    # Validate spin if multiplicity known
    if expected_multiplicity is not None:
        expected_spin = (expected_multiplicity - 1) / 2
        spin_sum = sum(spins)
        actual_spin = spin_sum / 2
        if abs(actual_spin - expected_spin) > tolerance:
            warnings.warn(
                f"Spin conservation violated: S={actual_spin:.3f}, "
                f"expected={expected_spin}, multiplicity={expected_multiplicity}"
            )

    # Add validation metadata
    result["validation"] = {
        "charge_sum": charge_sum,
        "charge_valid": abs(charge_sum - expected_charge) <= tolerance,
        "spin_sum": spin_sum,
        "spin_valid": True if expected_multiplicity is None
                     else abs(actual_spin - expected_spin) <= tolerance
    }

    return result
```

**Pros:**
- Catches data corruption immediately
- Provides actionable warnings
- Optional validation (backward compatible)
- Helps debug ORCA output issues
- Scientifically rigorous

**Cons:**
- Requires passing expected charge/multiplicity
- May generate warnings for numerical noise
- Need to tune tolerance parameter

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Strict Validation with Exceptions

**Approach:** Raise exception if conservation laws violated

**Pros:**
- Forces handling of bad data
- Prevents silent corruption
- Clear failure mode

**Cons:**
- May break existing workflows
- Too strict for numerical noise
- Need good error messages

**Effort:** 2 hours

**Risk:** High (breaking change)

---

### Option 3: Post-Processing Validation

**Approach:** Separate validation function called by workflow

**Pros:**
- Separation of concerns
- Optional validation
- Flexible usage

**Cons:**
- Easy to forget to call
- Not automatic
- Extra function call overhead

**Effort:** 3 hours

**Risk:** Medium

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected functions:**
- `parse_mulliken_population()` - Primary function needing validation
- `parse_job_metrics()` - Should pass through validation results
- `check_multi_spin.py` - Should use validation in workflow

**Conservation laws:**

**Charge conservation:**
```
Σ(atomic charges) = total molecular charge
For neutral molecule: Σ(charges) ≈ 0 (within ~0.01e)
```

**Spin conservation:**
```
S_total = Σ(atomic spins) / 2
Multiplicity = 2S + 1
Example: doublet (M=2) → S=0.5 → Σ(spins)=1.0
Example: quintet (M=5) → S=2.0 → Σ(spins)=4.0
```

**Tolerance values:**
- Charge: ±0.01e (standard for population analysis)
- Spin: ±0.05 (DFT can have larger deviations)

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** data-integrity-guardian agent finding
- **ORCA manual:** Population analysis section
- **Reference:** Mulliken, R.S. J. Chem. Phys. 1955, 23, 1833
- **Related:** Test file shows quintet (M=5, S=2, sum≈4.0)

## Acceptance Criteria

- [x] Charge conservation validated (sum vs expected)
- [x] Spin conservation validated (sum vs multiplicity)
- [x] Tolerance configurable
- [x] Clear warning messages with actual vs expected values
- [x] Validation results stored in return dictionary
- [x] Tests added for conservation violations
- [x] Documentation updated with validation behavior
- [x] Integration tested with check_multi_spin.py workflow

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (data-integrity-guardian agent)

**Actions:**
- Identified missing conservation law validation
- Reviewed quantum chemistry fundamentals
- Analyzed test data for expected values
- Verified test file shows ~4.0 spin sum (quintet)
- Researched appropriate tolerance values

**Learnings:**
- Critical for scientific integrity
- Standard practice in quantum chemistry codes
- Should be automatic, not optional
- Helps catch ORCA convergence issues
- Mulliken populations can have numerical noise

## Notes

- **CRITICAL** for scientific validity
- Should fail/warn on bad data, not silently continue
- Consider making this a standard validation pattern
- Add similar checks for other parsed quantities (energies, forces)
- Document expected behavior in scientific workflow guide

### 2026-02-16 - Implementation Complete

**By:** Claude Code

**Actions:**
- Implemented validate_charge_spin_conservation() function in analysis.py
- Added validation checks for charge sum (expected=0, tolerance=0.01)
- Added validation checks for spin sum based on multiplicity
- Integrated validation into parse_mulliken_population()
- Issues UserWarning when conservation violated (non-blocking)
- All 77 tests pass
- Committed and pushed to feature/mulliken-population-analysis

**Learnings:**
- Charge conservation: sum(charges) ≈ molecular charge
- Spin conservation: sum(spins)/2 ≈ (multiplicity-1)/2
- Used warnings instead of exceptions to avoid breaking existing workflows
- Validation helps detect ORCA parsing errors
