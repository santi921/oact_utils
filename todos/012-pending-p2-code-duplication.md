---
status: pending
priority: p2
issue_id: "012"
tags: [code-quality, duplication, refactoring]
dependencies: ["011"]
---

# Extract Duplicate Mulliken Parsing Logic

Code duplication: Mulliken and Loewdin parsing logic nearly identical.

## Problem Statement

`parse_mulliken_population()` contains two nearly-identical code blocks for parsing Mulliken and Loewdin sections. The only difference is the section header text and result key names. This violates DRY principle and increases maintenance burden.

**Impact:**
- Code duplication (~40 lines duplicated)
- Bug fixes need to be applied twice
- Inconsistent behavior between sections
- Increases maintenance burden
- Harder to add new population analysis types

## Findings

**Location:** `oact_utilities/utils/analysis.py:723-820`

**Duplicate pattern:**
```python
# Block 1: Mulliken parsing
for i, line in enumerate(lines):
    if "MULLIKEN ATOMIC CHARGES" in line:
        # ... ~30 lines of parsing logic ...

# Block 2: Loewdin parsing (nearly identical)
for i, line in enumerate(lines):
    if "LOEWDIN ATOMIC CHARGES" in line:
        # ... ~30 lines of parsing logic (duplicated) ...
```

**Differences:**
- Section header text: "MULLIKEN" vs "LOEWDIN"
- Result keys: "mulliken_charges" vs "loewdin_charges"
- Everything else is identical

**Identified by:** pattern-recognition-specialist and code-simplicity-reviewer agents

## Proposed Solutions

### Option 1: Extract Common Parsing Function (Recommended)

**Approach:** Create single function that handles both sections

```python
def _parse_population_section(
    lines: list[str],
    section_name: str,
    result_prefix: str
) -> dict[str, list] | None:
    """Parse a population analysis section (Mulliken or Loewdin).

    Args:
        lines: All lines from ORCA output
        section_name: Section header to find (e.g., "MULLIKEN ATOMIC CHARGES")
        result_prefix: Prefix for result keys (e.g., "mulliken")

    Returns:
        Dictionary with {prefix}_charges, {prefix}_spins, elements, indices
    """
    # Find section
    section_start = None
    for i, line in enumerate(lines):
        if section_name in line:
            section_start = i
            break

    if section_start is None:
        return None

    # Parse section (common logic)
    charges = []
    spins = []
    elements = []
    indices = []

    for line in lines[section_start + 2:]:
        if not line.strip() or "---" in line:
            break

        parts = line.split()
        if len(parts) < 4:
            continue

        # Parse element (handles both formats)
        idx = int(parts[0])
        if parts[1].endswith(":"):
            element = parts[1][:-1]
        elif len(parts) > 2 and parts[2] == ":":
            element = parts[1]
        else:
            continue

        charge = float(parts[-2])
        spin = float(parts[-1])

        indices.append(idx)
        elements.append(element)
        charges.append(charge)
        spins.append(spin)

    return {
        f"{result_prefix}_charges": charges,
        f"{result_prefix}_spins": spins,
        "elements": elements,
        "indices": indices,
    }

def parse_mulliken_population(output_file: str | Path) -> dict[str, list] | None:
    """Extract Mulliken/Loewdin population analysis."""
    with _smart_open(output_file) as f:
        lines = list(f)

    # Parse Mulliken
    mulliken_data = _parse_population_section(
        lines,
        "MULLIKEN ATOMIC CHARGES",
        "mulliken"
    )

    # Parse Loewdin
    loewdin_data = _parse_population_section(
        lines,
        "LOEWDIN ATOMIC CHARGES",
        "loewdin"
    )

    # Combine results
    if mulliken_data:
        result = mulliken_data
        if loewdin_data:
            result.update({
                "loewdin_charges": loewdin_data["loewdin_charges"],
                "loewdin_spins": loewdin_data["loewdin_spins"],
            })
        return result
    elif loewdin_data:
        return loewdin_data

    return None
```

**Pros:**
- **Eliminates ~40 lines of duplication**
- Bug fixes applied once
- Consistent behavior guaranteed
- Easy to add new population types (Hirshfeld, etc.)
- Better testability

**Cons:**
- Slightly more abstraction
- Need to pass section name and prefix

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Keep Duplication, Add Tests

**Approach:** Accept duplication but ensure both paths tested

**Pros:**
- No refactoring needed
- Explicit code is sometimes clearer

**Cons:**
- Maintains technical debt
- Bug fixes must be applied twice
- Risk of divergence

**Effort:** 0 hours

**Risk:** Medium (long-term)

---

### Option 3: Combined with Complexity Refactoring

**Approach:** Do this as part of #011 (cyclomatic complexity refactoring)

**Pros:**
- Solves multiple issues at once
- More comprehensive refactoring
- Better overall result

**Cons:**
- Larger change
- More testing needed

**Effort:** 4-5 hours (combined with #011)

**Risk:** Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Duplication metrics:**
- Lines duplicated: ~40 lines
- Duplication ratio: 50% of function is duplicate code
- Similarity: 95% identical between blocks

**Affected sections:**
1. Section search loop (identical)
2. Element parsing logic (identical)
3. Charge/spin extraction (identical)
4. Result building (differs only in key names)

**After refactoring:**
- Lines saved: ~40 lines
- LOC reduction: ~20%
- Functions: 1 instead of inline duplication

**Testing requirements:**
- Test extracted function with Mulliken data
- Test extracted function with Loewdin data
- Test main function with both sections
- Test main function with only Mulliken
- Test main function with only Loewdin

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** pattern-recognition-specialist + code-simplicity-reviewer findings
- **Related:** Issue #011 (cyclomatic complexity) - can be done together
- **DRY principle:** https://en.wikipedia.org/wiki/Don%27t_repeat_yourself

## Acceptance Criteria

- [ ] Duplicate parsing logic extracted to single function
- [ ] Both Mulliken and Loewdin use same parsing function
- [ ] All tests pass (existing tests unchanged)
- [ ] New unit test for extracted function
- [ ] Code review passed
- [ ] Documentation updated
- [ ] Future population types easy to add

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (pattern-recognition-specialist + code-simplicity-reviewer)

**Actions:**
- Identified duplicate parsing blocks
- Measured duplication (~40 lines, 50% of function)
- Analyzed differences (only section name and key prefix)
- Designed refactoring to eliminate duplication
- Noted synergy with complexity refactoring (#011)

**Learnings:**
- Copy-paste duplication common in parsing code
- Can parameterize section name and key prefix
- Should catch this in code review
- Consider adding duplication detection to CI (ruff, pylint)

## Notes

- **P2** because code works, but improvement valuable
- **Depends on #011** - Can be done together for efficiency
- Good opportunity to practice DRY principle
- Should add duplication detection to linter config
- Document code quality standards in CLAUDE.md
