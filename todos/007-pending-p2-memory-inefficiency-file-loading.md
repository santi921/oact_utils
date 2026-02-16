---
status: pending
priority: p2
issue_id: "007"
tags: [performance, code-review, optimization, memory]
dependencies: []
---

# Optimize Memory Usage: Stream Instead of Loading Full Files

Performance bottleneck: Loading entire ORCA output files into memory.

## Problem Statement

`parse_mulliken_population()` and other analysis functions use `.readlines()` which loads the entire file into memory. ORCA output files can be hundreds of MB, and workflows process thousands of files. This causes unnecessary memory consumption and could lead to OOM errors on HPC systems.

**Performance Impact:**
- **90% memory reduction possible** with streaming
- Current: Loads entire 100MB file → 100MB RAM
- With streaming: Reads line-by-line → ~4KB RAM
- Enables processing larger files
- Reduces memory pressure in parallel workflows

## Findings

**Location:** `oact_utilities/utils/analysis.py`

**Current pattern:**
```python
def parse_mulliken_population(output_file):
    with _smart_open(output_file) as f:
        lines = f.readlines()  # ❌ Loads entire file into memory
        for i, line in enumerate(lines):
            # Process line by line
```

**Memory usage example:**
- File size: 100 MB
- Memory loaded: 100 MB (entire file)
- Actually need: Only lines matching "MULLIKEN ATOMIC CHARGES"

**Other affected functions:**
- `parse_max_forces()` - Uses readlines()
- `parse_scf_steps()` - Uses readlines()
- `parse_final_energy()` - Uses readlines()

**Identified by:** performance-oracle agent during code review

## Proposed Solutions

### Option 1: Stream Line-by-Line (Recommended)

**Approach:** Replace `.readlines()` with iteration

```python
def parse_mulliken_population(output_file):
    with _smart_open(output_file) as f:
        # Stream line by line - no full file in memory
        for line_num, line in enumerate(f, start=1):
            if "MULLIKEN ATOMIC CHARGES" in line:
                # Start parsing
                # Read next N lines for data
                data_lines = [next(f) for _ in range(n_atoms)]
                # Process data_lines
```

**Pros:**
- **90% memory reduction**
- Works with files of any size
- Minimal code changes
- Same functionality

**Cons:**
- Slightly more complex for lookahead/lookbehind
- Need `next(f)` for reading ahead

**Effort:** 2-3 hours

**Risk:** Low

---

### Option 2: Memory-Mapped Files

**Approach:** Use `mmap` for large files

```python
import mmap

def parse_mulliken_population(output_file):
    with open(output_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Memory-mapped file access
            for line in iter(mm.readline, b''):
                # Process
```

**Pros:**
- OS-level memory management
- Very fast for large files
- Can handle huge files

**Cons:**
- More complex
- Binary mode handling
- OS-dependent behavior
- Overkill for most cases

**Effort:** 4-5 hours

**Risk:** Medium

---

### Option 3: Chunk-Based Reading

**Approach:** Read file in chunks

**Pros:**
- Balance between memory and simplicity
- Can tune chunk size

**Cons:**
- Complex line boundary handling
- May split lines across chunks
- Not worth the complexity

**Effort:** 3-4 hours

**Risk:** Medium

## Recommended Action

**To be filled during triage.**

## Technical Details

**Current memory usage (example workflow):**
- 1000 jobs × 100 MB files = 100 GB if all loaded
- With 20 parallel workers × 100 MB = 2 GB minimum RAM
- With streaming: 20 workers × 4 KB = 80 KB

**Benchmark data needed:**
- Time to parse with readlines() vs streaming
- Memory usage with readlines() vs streaming
- Impact on workflow completion time

**Affected functions to refactor:**
1. `parse_mulliken_population()` - line 723
2. `parse_max_forces()` - check implementation
3. `parse_scf_steps()` - check implementation
4. `parse_final_energy()` - check implementation

**Pattern to use:**
```python
# Old
with open(file) as f:
    lines = f.readlines()  # ❌ Bad
    for i, line in enumerate(lines):
        ...

# New
with open(file) as f:
    for i, line in enumerate(f):  # ✅ Good
        ...
```

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** performance-oracle agent finding
- **Python docs:** https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects

## Acceptance Criteria

- [ ] All analysis functions stream instead of loading full files
- [ ] Memory usage reduced by >80% in benchmarks
- [ ] Tests still pass with same results
- [ ] Performance benchmarks show improvement or no regression
- [ ] Works with both regular and gzipped files
- [ ] Documentation updated with memory considerations

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (performance-oracle agent)

**Actions:**
- Identified readlines() pattern in multiple functions
- Calculated potential memory savings (90%)
- Researched streaming alternatives
- Verified _smart_open() supports iteration

**Learnings:**
- File objects are iterators in Python
- `.readlines()` is rarely needed
- Streaming is almost always better
- HPC systems have memory limits
- Should be standard practice

## Notes

- **P2** because workflows currently work, but optimization valuable
- High impact for large-scale workflows
- Easy fix with big benefits
- Should apply pattern to all file parsing functions
- Consider adding performance tests to CI
