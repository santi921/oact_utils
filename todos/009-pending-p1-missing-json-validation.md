---
status: pending
priority: p1
issue_id: "009"
tags: [data-integrity, validation, json, critical]
dependencies: []
---

# Add JSON Deserialization Validation

CRITICAL: No validation when reading JSON from database, risk of code execution.

## Problem Statement

`check_multi_spin.py` stores Mulliken charges/spins as JSON strings in SQLite, then deserializes them without validation. Malicious or corrupted JSON could cause:
- **Code execution** if using `eval()` instead of `json.loads()`
- **Type confusion** (expecting list, get dict)
- **Silent data corruption** from invalid JSON
- **Workflow crashes** from unexpected data types

**Security & Data Integrity Impact:**
- **HIGH SEVERITY** - Potential code injection
- Database corruption propagates silently
- Workflow failures from type mismatches
- No schema validation for JSON content

## Findings

**Location:** `oact_utilities/scripts/multi_spin/check_multi_spin.py`

**Current pattern:**
```python
# Storing (safe):
conn.execute(
    "INSERT INTO jobs (..., mulliken_charges) VALUES (?, ...)",
    (json.dumps(charges), ...)
)

# Reading (UNSAFE - no validation):
row = conn.execute("SELECT mulliken_charges FROM jobs WHERE id=?").fetchone()
charges = json.loads(row[0])  # ❌ No validation
# Assumes: charges is list of floats
# Reality: Could be anything - dict, string, null, invalid JSON
```

**Vulnerabilities:**
1. No type checking (list vs dict vs null)
2. No element validation (floats vs strings)
3. No length validation (empty list? too many elements?)
4. No error handling for invalid JSON
5. Silent failures propagate bad data

**Attack scenarios:**
- Database corrupted by bug → invalid JSON → workflow crash
- Manual database edit → wrong data type → silent errors
- Concurrent writes → partial JSON → json.loads() fails
- Future code change → schema mismatch → type errors

**Identified by:** data-integrity-guardian agent during code review

## Proposed Solutions

### Option 1: Validation Functions (Recommended)

**Approach:** Create validation wrappers for JSON deserialization

```python
from typing import Any
import json

def validate_charge_spin_list(
    data: Any,
    field_name: str,
    expected_length: int | None = None
) -> list[float]:
    """Validate and return a list of charges or spins.

    Args:
        data: JSON-decoded data to validate
        field_name: Name for error messages ('mulliken_charges', etc.)
        expected_length: Expected list length (number of atoms)

    Returns:
        Validated list of floats

    Raises:
        ValueError: If validation fails
    """
    if data is None:
        return []

    if not isinstance(data, list):
        raise ValueError(
            f"{field_name} must be a list, got {type(data).__name__}"
        )

    # Validate each element
    validated = []
    for i, item in enumerate(data):
        if not isinstance(item, (int, float)):
            raise ValueError(
                f"{field_name}[{i}] must be numeric, got {type(item).__name__}: {item}"
            )
        validated.append(float(item))

    # Validate length if specified
    if expected_length is not None and len(validated) != expected_length:
        raise ValueError(
            f"{field_name} length mismatch: got {len(validated)}, "
            f"expected {expected_length}"
        )

    return validated

def safe_load_json_field(
    json_string: str | None,
    field_name: str,
    validator: callable
) -> Any:
    """Safely load and validate a JSON field from database.

    Args:
        json_string: JSON string from database (may be None)
        field_name: Field name for error messages
        validator: Validation function to apply

    Returns:
        Validated data

    Raises:
        ValueError: If JSON invalid or validation fails
    """
    if json_string is None or json_string == "":
        return None

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {field_name}: {e}")

    return validator(data, field_name)

# Usage:
row = conn.execute("SELECT mulliken_charges, n_atoms FROM jobs WHERE id=?").fetchone()
charges = safe_load_json_field(
    row[0],
    "mulliken_charges",
    lambda d, name: validate_charge_spin_list(d, name, expected_length=row[1])
)
```

**Pros:**
- Explicit validation at deserialization
- Clear error messages
- Type safety guaranteed
- Prevents data corruption propagation
- Reusable for all JSON fields

**Cons:**
- Need to add validation calls everywhere
- Extra code to maintain
- Performance overhead (minimal)

**Effort:** 3-4 hours

**Risk:** Low

---

### Option 2: Pydantic Models

**Approach:** Use Pydantic for schema validation

```python
from pydantic import BaseModel, Field

class MullikenData(BaseModel):
    charges: list[float] = Field(..., description="Atomic charges")
    spins: list[float] = Field(..., description="Atomic spins")
    elements: list[str]
    indices: list[int]

    @validator('charges', 'spins')
    def validate_numeric_list(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Must be list of numbers")
        return v

# Usage:
data = json.loads(row[0])
validated = MullikenData(**data)  # Automatic validation
```

**Pros:**
- Professional validation framework
- Automatic type conversion
- Rich validation options
- IDE support

**Cons:**
- New dependency
- Learning curve
- Overkill for simple validation

**Effort:** 4-5 hours

**Risk:** Medium

---

### Option 3: JSON Schema Validation

**Approach:** Use jsonschema library

**Pros:**
- Standard JSON Schema format
- Comprehensive validation

**Cons:**
- Extra dependency
- More complex than needed
- Verbose schemas

**Effort:** 3-4 hours

**Risk:** Medium

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected fields:**
- `mulliken_charges` - List[float]
- `mulliken_spins` - List[float]
- `loewdin_charges` - List[float]
- `loewdin_spins` - List[float]

**Validation requirements:**
1. Must be valid JSON
2. Must be a list (not dict, string, null)
3. All elements must be numeric (int or float)
4. Length should match number of atoms
5. Values should be physically reasonable (charges typically -3 to +3, spins -1 to +1)

**Error handling:**
```python
try:
    charges = safe_load_json_field(row[0], "mulliken_charges", validator)
except ValueError as e:
    logger.error(f"Invalid data for job {job_id}: {e}")
    # Mark job as failed? Skip? Raise?
```

**Performance impact:**
- Validation adds ~0.1ms per field
- Negligible compared to database query time
- Can cache validation results

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** data-integrity-guardian agent finding
- **Pydantic:** https://docs.pydantic.dev/
- **jsonschema:** https://python-jsonschema.readthedocs.io/

## Acceptance Criteria

- [ ] All JSON deserialization goes through validation
- [ ] Type checking (list of floats)
- [ ] Length validation (matches n_atoms)
- [ ] Clear error messages on validation failure
- [ ] Tests for invalid JSON scenarios
- [ ] Tests for type mismatches
- [ ] Tests for corrupted data
- [ ] Documentation of JSON schema
- [ ] Error handling policy defined

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (data-integrity-guardian agent)

**Actions:**
- Identified missing JSON validation
- Analyzed potential vulnerabilities
- Researched validation approaches
- Designed lightweight validation system
- Drafted implementation examples

**Learnings:**
- JSON deserialization without validation is dangerous
- Type confusion can cause silent failures
- Standard practice: always validate untrusted data
- Database is untrusted (can be manually edited)
- Validation should happen at boundary (deserialization)

## Notes

- **CRITICAL** for data integrity
- Prevents many classes of bugs
- Should be standard practice for all JSON fields
- Consider adding schema documentation
- Add validation tests to test suite
