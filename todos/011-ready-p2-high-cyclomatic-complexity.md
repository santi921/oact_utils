---
status: ready
priority: p2
issue_id: "011"
tags: [code-quality, refactoring, maintainability]
dependencies: []
---

# Refactor High Cyclomatic Complexity in parse_mulliken_population

Code quality: Function complexity score of 15 exceeds threshold of 10.

## Problem Statement

`parse_mulliken_population()` has a cyclomatic complexity of 15, exceeding the recommended maximum of 10. High complexity makes code harder to understand, test, and maintain.

**Impact:**
- Difficult to understand logic flow
- Hard to write comprehensive tests
- Higher bug probability
- Maintenance burden
- Code review challenges

## Findings

**Location:** `oact_utilities/utils/analysis.py:723-820`

**Complexity contributors:**
- Multiple nested if statements
- Two parsing modes (Mulliken vs Loewdin)
- Two element format variations ("Np:" vs "F :")
- Error handling throughout
- Multiple early returns
- Complex loop logic

**Cyclomatic complexity breakdown:**
- Base: 1
- If statements: ~8
- Nested conditions: ~4
- Loop branches: ~2
- **Total: 15** (threshold: 10)

**Identified by:** pattern-recognition-specialist agent during code review

## Proposed Solutions

### Option 1: Extract Parsing Functions (Recommended)

**Approach:** Break into smaller, focused functions

```python
def _parse_population_section(
    lines: list[str],
    start_idx: int,
    section_type: str
) -> dict[str, list]:
    """Parse a single population analysis section (Mulliken or Loewdin).

    Returns:
        Dictionary with charges, spins, elements, indices
    """
    charges = []
    spins = []
    elements = []
    indices = []

    i = start_idx
    while i < len(lines):
        line = lines[i]

        # Stop at blank line or next section
        if not line.strip() or "---" in line:
            break

        parts = line.split()
        if len(parts) < 4:
            i += 1
            continue

        # Parse element (handles both formats)
        element, idx = _parse_element_and_index(parts)
        if element is None:
            i += 1
            continue

        # Extract charge and spin
        charge = float(parts[-2])
        spin = float(parts[-1])

        elements.append(element)
        indices.append(idx)
        charges.append(charge)
        spins.append(spin)

        i += 1

    return {
        f"{section_type}_charges": charges,
        f"{section_type}_spins": spins,
        "elements": elements,
        "indices": indices,
    }

def _parse_element_and_index(parts: list[str]) -> tuple[str | None, int | None]:
    """Parse element symbol and index from split line.

    Handles both formats:
    - "0 Np:" (2-char element)
    - "0 F :" (1-char element)
    """
    try:
        idx = int(parts[0])

        # Format 1: "Np:" (2-char element with colon)
        if parts[1].endswith(":"):
            element = parts[1][:-1]
            return element, idx

        # Format 2: "F :" (1-char element, separate colon)
        if len(parts) > 2 and parts[2] == ":":
            element = parts[1]
            return element, idx

        return None, None
    except (ValueError, IndexError):
        return None, None

def parse_mulliken_population(output_file: str | Path) -> dict[str, list] | None:
    """Extract Mulliken/Loewdin population analysis.

    Now much simpler with extracted functions.
    """
    with _smart_open(output_file) as f:
        lines = list(f)

    result = {}

    # Find and parse Mulliken section
    for i, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line:
            mulliken_data = _parse_population_section(lines, i + 2, "mulliken")
            result.update(mulliken_data)
            break

    # Find and parse Loewdin section
    for i, line in enumerate(lines):
        if "LOEWDIN ATOMIC CHARGES" in line:
            loewdin_data = _parse_population_section(lines, i + 2, "loewdin")
            result.update({
                "loewdin_charges": loewdin_data["loewdin_charges"],
                "loewdin_spins": loewdin_data["loewdin_spins"],
            })
            break

    return result if result else None
```

**Pros:**
- **Reduces complexity:** Main function → 5, helpers → 3-4 each
- Easier to understand each piece
- Better testability
- Reusable functions
- Clear separation of concerns

**Cons:**
- More functions to maintain
- Slightly more code overall
- Need to coordinate between functions

**Effort:** 3-4 hours

**Risk:** Low

---

### Option 2: State Machine Pattern

**Approach:** Use explicit state machine for parsing

**Pros:**
- Very clear logic flow
- Easy to extend
- Handles edge cases well

**Cons:**
- More code
- Overkill for this use case
- Harder for team to understand

**Effort:** 5-6 hours

**Risk:** Medium

---

### Option 3: Accept Current Complexity

**Approach:** Leave as-is but add comprehensive tests

**Pros:**
- No refactoring needed
- Function works correctly

**Cons:**
- Maintains technical debt
- Hard to modify in future
- Still difficult to test thoroughly

**Effort:** 0 hours

**Risk:** Low (short-term), High (long-term)

## Recommended Action

Approved during triage - proceed with Option 1 (see Proposed Solutions)

## Technical Details

**Current complexity breakdown:**
```python
def parse_mulliken_population(...):  # CC = 15
    # 1. Open file (CC +1 for error handling)
    # 2. Loop through lines (CC +1)
    # 3. Find Mulliken section (CC +1)
    # 4. Parse Mulliken loop (CC +1)
    #    - Check element format 1 (CC +1)
    #    - Check element format 2 (CC +1)
    #    - Validate parts length (CC +1)
    # 5. Find Loewdin section (CC +1)
    # 6. Parse Loewdin loop (CC +1)
    #    - Check element format 1 (CC +1)
    #    - Check element format 2 (CC +1)
    #    - Validate parts length (CC +1)
    # 7. Return handling (CC +1)
    # 8. Error cases (CC +2)
    # Total: ~15
```

**After refactoring:**
```python
parse_mulliken_population()  # CC = 5
_parse_population_section()  # CC = 4
_parse_element_and_index()   # CC = 3
```

**Testing improvements:**
- Can test element parsing independently
- Can test section parsing with mock data
- Can test main function with integration tests
- Each function has clear contract

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** pattern-recognition-specialist agent finding
- **Cyclomatic Complexity:** https://en.wikipedia.org/wiki/Cyclomatic_complexity
- **Refactoring book:** Martin Fowler

## Acceptance Criteria

- [ ] Main function complexity ≤ 10
- [ ] Helper functions complexity ≤ 5 each
- [ ] All tests still pass
- [ ] New unit tests for extracted functions
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Code review passed

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (pattern-recognition-specialist agent)

**Actions:**
- Measured cyclomatic complexity (15)
- Identified complexity contributors
- Designed refactoring with extracted functions
- Drafted simplified implementation

**Learnings:**
- Function does too much (parsing + format handling + validation)
- Extract Method refactoring is appropriate
- Smaller functions easier to test and understand
- Should set CC limit in linter (ruff, flake8)

## Notes

- **P2** because code works, but improvement valuable
- Good opportunity to improve code quality
- Should add complexity checks to CI
- Consider using `radon` or `flake8-complexity` in pre-commit
- Document complexity guidelines in CLAUDE.md

### 2026-02-16 - Approved for Work

**By:** Claude Triage System

**Actions:**
- Issue approved during triage session
- Status changed from pending → ready
- Ready to be picked up and worked on
