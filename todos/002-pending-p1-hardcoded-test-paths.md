---
status: pending
priority: p1
issue_id: "002"
tags: [code-review, testing, portability, critical]
dependencies: []
---

# Remove Hardcoded Absolute Paths in test_analysis.py

CRITICAL: Tests will fail on any machine other than the developer's laptop.

## Problem Statement

`test_analysis.py` uses hardcoded absolute paths (`/Users/santiagovargas/dev/oact_utils/...`) instead of relative paths. This makes tests non-portable and will cause CI/CD failures and issues for other developers.

**Impact:**
- Tests fail on CI systems
- Tests fail for other developers
- Blocks collaborative development
- Violates testing best practices

## Findings

**Location:** `tests/test_analysis.py:8, 11`

```python
res_no_traj = get_rmsd_start_final(
    "/Users/santiagovargas/dev/oact_utils/tests/files/no_traj"  # ❌ Hardcoded
)
res_traj = get_rmsd_start_final(
    "/Users/santiagovargas/dev/oact_utils/tests/files/traj/"  # ❌ Hardcoded
)
```

**Root cause:**
- Used absolute path instead of relative path from test file
- Missing use of `Path(__file__).parent` pattern for test file location

**Identified by:** kieran-python-reviewer agent during code review

## Proposed Solutions

### Option 1: Use Path(__file__).parent (Recommended)

**Approach:**
```python
from pathlib import Path

def test_get_rmsd_start_final():
    test_dir = Path(__file__).parent / "files"
    res_no_traj = get_rmsd_start_final(str(test_dir / "no_traj"))
    res_traj = get_rmsd_start_final(str(test_dir / "traj"))
```

**Pros:**
- Standard Python pattern for test paths
- Works on all platforms and machines
- Self-contained and portable

**Cons:**
- None

**Effort:** 3 minutes

**Risk:** Very Low

---

### Option 2: Use pytest fixtures with tmp_path

**Approach:** Copy test data to pytest's tmp_path fixture

**Pros:**
- Isolated test environment
- Each test run uses fresh copies

**Cons:**
- More complex
- Unnecessary overhead for read-only test data
- Overkill for this use case

**Effort:** 15 minutes

**Risk:** Low

## Recommended Action

**To be filled during triage.**

## Technical Details

**Affected files:**
- `tests/test_analysis.py:8` - First hardcoded path
- `tests/test_analysis.py:11` - Second hardcoded path

**Required changes:**
```python
# Add at top of file
from pathlib import Path

# Update test function
def test_get_rmsd_start_final():
    test_dir = Path(__file__).parent / "files"
    res_no_traj = get_rmsd_start_final(str(test_dir / "no_traj"))
    res_traj = get_rmsd_start_final(str(test_dir / "traj"))
    # ... rest of test
```

**Test data location:**
- `tests/files/no_traj/` - Exists ✓
- `tests/files/traj/` - Exists ✓

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** kieran-python-reviewer agent finding
- **Python Path docs:** https://docs.python.org/3/library/pathlib.html

## Acceptance Criteria

- [ ] All absolute paths replaced with relative paths
- [ ] Tests pass locally
- [ ] Tests pass on CI (if CI configured)
- [ ] Code follows project testing patterns
- [ ] Pre-commit hooks pass

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (kieran-python-reviewer agent)

**Actions:**
- Identified hardcoded paths during code review
- Verified test data files exist at correct relative locations
- Confirmed this is a blocking P1 issue for portability

**Learnings:**
- Common mistake when writing tests quickly
- Should add linting rule to catch absolute paths in tests
- Consider pytest plugin to catch this automatically

## Notes

- This is a **BLOCKING** issue for PR merge
- Affects portability and CI/CD
- Simple fix but critical for collaborative development
