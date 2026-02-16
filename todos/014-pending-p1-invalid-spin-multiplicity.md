---
status: pending
priority: p1
issue_id: "014"
tags: [data-integrity, scientific, validation, critical]
dependencies: ["005"]
---

# Validate Spin Multiplicity Before Database Storage

CRITICAL: No validation that spin multiplicity is physically valid before storage.

## Problem Statement

Workflow stores spin multiplicity in database without validation. Invalid values (e.g., spin=0, negative spins, non-integer multiplicity) could propagate through entire workflow, leading to:
- **Invalid ORCA calculations** (crash or garbage results)
- **Wasted HPC resources** (failed jobs)
- **Scientific errors** (wrong spin state studied)
- **Silent failures** (bad data propagates undetected)

**Scientific Impact:**
- **HIGH SEVERITY** - Invalid quantum chemistry inputs
- Computational waste (job failures)
- Wrong scientific results
- Undermines data integrity

## Findings

**Location:** Multiple files handling spin multiplicity

**Current behavior:**
```python
# architector.py:485-492 - Fixed recently but no validation
spin = int(spin_val) + 1  # Convert uhf to multiplicity

# ❌ No validation:
# - spin must be positive integer
# - spin must be odd or even based on electron count
# - spin must be ≥ 1 (singlet minimum)

# Stored in database without checking validity
```

**Invalid spin examples:**
- `spin = 0` → Invalid (multiplicity must be ≥ 1)
- `spin = -1` → Invalid (negative multiplicity nonsense)
- `spin = 1.5` → Invalid (must be integer)
- `spin = 100` → Likely error (physically unreasonable)

**Physical constraints:**
```
Multiplicity (2S+1) must be:
1. Positive integer ≥ 1
2. Singlet (S=0, M=1): Even number of electrons
3. Doublet (S=0.5, M=2): Odd number of electrons
4. Consistent with electron count parity

Valid: 1 (singlet), 2 (doublet), 3 (triplet), 5 (quintet), ...
Invalid: 0, negative, non-integer, unreasonably high
```

**Identified by:** data-integrity-guardian agent during code review

## Proposed Solutions

### Option 1: Add Validation Function (Recommended)

**Approach:** Validate spin multiplicity before storage

```python
def validate_spin_multiplicity(
    spin: int,
    n_electrons: int | None = None,
    max_reasonable: int = 11
) -> int:
    """Validate spin multiplicity is physically valid.

    Args:
        spin: Spin multiplicity (2S+1)
        n_electrons: Total electron count (optional, for parity check)
        max_reasonable: Maximum reasonable multiplicity (default 11 = undectet)

    Returns:
        Validated spin multiplicity

    Raises:
        ValueError: If spin multiplicity is invalid

    Examples:
        >>> validate_spin_multiplicity(1)  # Singlet - OK
        1
        >>> validate_spin_multiplicity(3)  # Triplet - OK
        3
        >>> validate_spin_multiplicity(0)  # Invalid
        ValueError: Spin multiplicity must be ≥ 1, got 0
        >>> validate_spin_multiplicity(5, n_electrons=60)  # Check parity
        ValueError: Spin multiplicity 5 (odd) incompatible with 60 electrons (even)
    """
    # Must be positive integer
    if not isinstance(spin, int):
        raise ValueError(f"Spin multiplicity must be integer, got {type(spin).__name__}")

    if spin < 1:
        raise ValueError(f"Spin multiplicity must be ≥ 1, got {spin}")

    # Check reasonable range
    if spin > max_reasonable:
        warnings.warn(
            f"Spin multiplicity {spin} is very high (> {max_reasonable}). "
            "This may be an error."
        )

    # Check parity if electron count known
    if n_electrons is not None:
        # Odd electrons → even multiplicity (doublet, quartet, ...)
        # Even electrons → odd multiplicity (singlet, triplet, quintet, ...)
        expected_parity = "even" if n_electrons % 2 == 1 else "odd"
        actual_parity = "even" if spin % 2 == 0 else "odd"

        if expected_parity != actual_parity:
            raise ValueError(
                f"Spin multiplicity {spin} ({actual_parity}) incompatible with "
                f"{n_electrons} electrons ({expected_parity} expected)"
            )

    return spin

# Usage in architector.py:
if spin_column and spin_column in chunk.columns:
    spin_val = row.get(spin_column)
    if not pd.isna(spin_val):
        spin = int(spin_val) + 1  # Convert uhf to multiplicity

        # Validate before storing
        try:
            spin = validate_spin_multiplicity(spin, n_electrons=None)
        except ValueError as e:
            logger.error(f"Invalid spin multiplicity for row {idx}: {e}")
            continue  # Skip this job

        # Now safe to use
        job_data["spin"] = spin
```

**Pros:**
- Catches invalid data before database storage
- Prevents wasted ORCA calculations
- Clear error messages
- Scientifically rigorous
- Optional electron count check

**Cons:**
- Need electron count for full validation
- May need to tune max_reasonable threshold
- Extra validation overhead (minimal)

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Database Constraint

**Approach:** Add CHECK constraint to database schema

```sql
CREATE TABLE jobs (
    spin INTEGER NOT NULL CHECK (spin >= 1 AND spin <= 11),
    ...
);
```

**Pros:**
- Enforced at database level
- Cannot be bypassed
- Standard SQL pattern

**Cons:**
- Doesn't explain *why* invalid
- No electron parity check
- Less informative errors
- Can't do complex validation

**Effort:** 1 hour

**Risk:** Low

---

### Option 3: Combined Approach

**Approach:** Validation function + database constraint

**Pros:**
- Defense in depth
- Database enforces minimum
- Code provides detailed validation
- Best of both worlds

**Cons:**
- Two places to maintain
- Slightly more code

**Effort:** 3 hours

**Risk:** Very Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Spin multiplicity fundamentals:**

```
Multiplicity (M) = 2S + 1
where S = total spin angular momentum

M=1 (S=0): Singlet - all electrons paired
M=2 (S=1/2): Doublet - one unpaired electron
M=3 (S=1): Triplet - two unpaired electrons
M=4 (S=3/2): Quartet - three unpaired electrons
M=5 (S=2): Quintet - four unpaired electrons
...
M=11 (S=5): Undectet - ten unpaired electrons
```

**Validation rules:**
1. M must be positive integer ≥ 1
2. M and electron count parity must match:
   - Odd electrons → Even M (2, 4, 6, ...)
   - Even electrons → Odd M (1, 3, 5, ...)
3. M ≤ N_electrons + 1 (maximum all unpaired)
4. Reasonable maximum (M ≤ 11 for most chemistry)

**Affected code locations:**
- `architector.py:485-492` - Spin multiplicity calculation from uhf
- `architector_workflow.py` - Workflow database creation
- `check_multi_spin.py` - Multi-spin state workflows

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** data-integrity-guardian agent finding
- **Related:** Issue #005 (charge/spin conservation validation)
- **Spin states:** https://en.wikipedia.org/wiki/Spin_multiplicity
- **ORCA manual:** Spin multiplicity section

## Acceptance Criteria

- [ ] Validation function implemented
- [ ] Spin multiplicity validated before database storage
- [ ] Positive integer check (≥ 1)
- [ ] Reasonable maximum check (≤ 11 with warning)
- [ ] Electron parity check (optional, if n_electrons available)
- [ ] Clear error messages
- [ ] Tests for invalid spin values
- [ ] Tests for electron parity mismatches
- [ ] Documentation updated
- [ ] Integration with workflow tested

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (data-integrity-guardian agent)

**Actions:**
- Identified missing spin multiplicity validation
- Reviewed quantum chemistry spin fundamentals
- Analyzed validation requirements
- Designed validation function with parity checks
- Researched reasonable spin ranges

**Learnings:**
- Spin multiplicity validation is critical for QC calculations
- Must check both value range and electron parity
- Invalid spin → ORCA job fails immediately
- Should validate at workflow input, not at ORCA submission
- Standard practice in computational chemistry codes

## Notes

- **CRITICAL** for scientific validity
- Prevents expensive ORCA failures
- Should be validated at multiple points:
  1. CSV import (architector.py)
  2. Database storage
  3. Job submission (belt and suspenders)
- **Depends on #005** for full conservation validation
- Document spin state conventions in scientific guide
