---
title: Fix job status updates with completion callback
type: fix
date: 2026-02-16
---

# Fix Job Status Updates with Completion Callback

## Overview

Jobs are currently incorrectly marked as COMPLETED when they should be FAILED or TIMEOUT. The issue stems from `parse_job_metrics()` only checking for "ORCA TERMINATED NORMALLY" without verifying the job actually completed successfully. Additionally, jobs should self-report their completion status rather than relying on external polling.

## Problem Statement

### Bug: Incomplete Termination Checking

Currently in [oact_utilities/utils/analysis.py:1185](oact_utilities/utils/analysis.py#L1185):
```python
success = "ORCA TERMINATED NORMALLY" in content
```

This naive check misses critical failure modes:
- ❌ Jobs that time out (file not modified in hours_cutoff)
- ❌ Jobs with "aborting the run" messages
- ❌ Jobs with "Error" in the last 5 lines

Meanwhile, `check_file_termination()` in [oact_utilities/utils/status.py:15](oact_utilities/utils/status.py#L15) already implements robust checking for all these cases but isn't being used.

### Missing Feature: Job Self-Reporting

Jobs currently require external polling via `update_job_status()` to update their completion status. This creates:
- ⏱️ Delayed status updates (polling interval)
- 🔄 Overhead from manual status checks
- 🐛 Risk of jobs marked RUNNING indefinitely if polling fails

## Proposed Solution

### Part 1: Fix `parse_job_metrics()` Termination Check

Replace the simple string check with `check_file_termination()` which properly detects:
- ✅ Normal termination ("ORCA TERMINATED NORMALLY")
- ✅ Aborted runs ("aborting the run")
- ✅ Error states ("Error" in last 5 lines)
- ✅ Timeouts (file modification time > hours_cutoff)

### Part 2: Add Job Completion Callback

Create a completion callback system where jobs automatically report their status upon finishing:

**Flow:**
```
Submit job → Mark RUNNING → Job runs → Job finishes →
Job script calls report_job_completion() → check_file_termination() →
Update DB with COMPLETED/FAILED/TIMEOUT + metrics
```

**Benefits:**
- ✅ Immediate status updates (no polling delay)
- ✅ Jobs self-report their ground truth status
- ✅ Reduces need for external status monitoring
- ✅ Automatic metric extraction on completion

## Technical Approach

### 1. Fix `parse_job_metrics()` Function

**File:** `oact_utilities/utils/analysis.py`

**Changes:**
```python
# Before (line ~1185):
success = "ORCA TERMINATED NORMALLY" in content

# After:
from ..utils.status import check_file_termination
termination_status = check_file_termination(output_file, hours_cutoff=6)
success = (termination_status == 1)  # 1 = normal termination
is_timeout = (termination_status == -2)  # -2 = timeout
```

**Return value updates:**
```python
return {
    "max_forces": max_forces,
    "scf_steps": scf_steps,
    "final_energy": final_energy,
    "success": success,
    "is_timeout": is_timeout,  # NEW
    "termination_status": termination_status,  # NEW
    "mulliken_population": mulliken_pop,
}
```

### 2. Update `update_job_status()` to Handle Timeouts

**File:** `oact_utilities/workflows/architector_workflow.py`

**Changes around line 553:**
```python
# Before:
new_status = JobStatus.COMPLETED if metrics["success"] else JobStatus.FAILED

# After:
if metrics.get("is_timeout", False):
    new_status = JobStatus.TIMEOUT
elif metrics["success"]:
    new_status = JobStatus.COMPLETED
else:
    new_status = JobStatus.FAILED
```

### 3. Create Job Completion Callback Function

**File:** `oact_utilities/workflows/architector_workflow.py` (new function)

```python
def report_job_completion(
    db_path: str | Path,
    job_id: int,
    job_dir: str | Path,
    extract_metrics: bool = True,
    unzip: bool = False,
    hours_cutoff: float = 6.0,
) -> JobStatus:
    """Callback function for jobs to report their completion status.

    This should be called at the end of each job script to automatically
    update the database with the job's final status and metrics.

    Args:
        db_path: Path to the workflow SQLite database.
        job_id: Database ID of the job.
        job_dir: Directory containing job output files.
        extract_metrics: If True, extract and store max_forces, scf_steps, etc.
        unzip: If True, handle gzipped output files (quacc).
        hours_cutoff: Timeout threshold in hours for stale files.

    Returns:
        The final status of the job (COMPLETED, FAILED, or TIMEOUT).

    Example:
        # Add to end of job script:
        python -c "
        from oact_utilities.workflows import report_job_completion
        report_job_completion('workflow.db', job_id=123, job_dir='.')
        "
    """
    try:
        with ArchitectorWorkflow(db_path) as workflow:
            status = update_job_status(
                workflow=workflow,
                job_dir=job_dir,
                job_id=job_id,
                extract_metrics=extract_metrics,
                unzip=unzip,
            )
            return status
    except Exception as e:
        # Log error but don't crash (job already finished)
        import sys
        print(f"Error reporting job completion: {e}", file=sys.stderr)
        return JobStatus.FAILED
```

### 4. Create CLI Utility for Job Completion

**File:** `oact_utilities/workflows/report_completion.py` (new file)

```python
#!/usr/bin/env python
"""CLI utility for jobs to report their completion status."""

import argparse
import sys
from pathlib import Path

from .architector_workflow import report_job_completion


def main():
    parser = argparse.ArgumentParser(
        description="Report job completion to workflow database"
    )
    parser.add_argument("db_path", type=str, help="Path to workflow database")
    parser.add_argument("job_id", type=int, help="Job database ID")
    parser.add_argument(
        "--job-dir",
        type=str,
        default=".",
        help="Job directory (default: current directory)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metric extraction (faster)",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Handle gzipped output files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Timeout threshold in hours (default: 6)",
    )

    args = parser.parse_args()

    try:
        status = report_job_completion(
            db_path=args.db_path,
            job_id=args.job_id,
            job_dir=args.job_dir,
            extract_metrics=not args.no_metrics,
            unzip=args.unzip,
            hours_cutoff=args.timeout,
        )
        print(f"Job {args.job_id} reported as: {status.value}")
        sys.exit(0 if status == "completed" else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
```

### 5. Update HPC Job Templates

**Files to modify:**
- `oact_utilities/utils/hpc.py` - `write_flux_job()` and `write_slurm_job()`

**Add completion callback to job templates:**

```bash
# Flux job template (end of script):
# ... existing job commands ...

# Report completion to workflow database
if [ -n "${WORKFLOW_DB}" ] && [ -n "${JOB_ID}" ]; then
    python -m oact_utilities.workflows.report_completion \
        "${WORKFLOW_DB}" "${JOB_ID}" --job-dir "${PWD}" \
        || echo "Warning: Failed to report job completion" >&2
fi
```

```bash
# SLURM job template (end of script):
# ... existing job commands ...

# Report completion to workflow database
if [ -n "${WORKFLOW_DB}" ] && [ -n "${JOB_ID}" ]; then
    python -m oact_utilities.workflows.report_completion \
        "${WORKFLOW_DB}" "${JOB_ID}" --job-dir "${PWD}" \
        || echo "Warning: Failed to report job completion" >&2
fi
```

### 6. Update Job Submission to Pass Metadata

**File:** `oact_utilities/workflows/submit_jobs.py`

**Changes:**
- Pass `WORKFLOW_DB` and `JOB_ID` as environment variables when submitting jobs
- Add optional `enable_completion_callback` parameter (default: True)

```python
# When writing job file:
env_vars = {
    "WORKFLOW_DB": str(db_path),
    "JOB_ID": str(job_record.id),
    # ... other vars ...
}

# In Flux:
flux_batch_cmd = [
    "flux", "batch",
    "--setenv=WORKFLOW_DB=" + env_vars["WORKFLOW_DB"],
    "--setenv=JOB_ID=" + env_vars["JOB_ID"],
    # ... other flags ...
]

# In SLURM:
sbatch_cmd = [
    "sbatch",
    "--export=ALL,WORKFLOW_DB=" + env_vars["WORKFLOW_DB"],
    "--export=ALL,JOB_ID=" + env_vars["JOB_ID"],
    # ... other flags ...
]
```

## Implementation Phases

### Phase 1: Fix Core Bug (High Priority)
- [x] Update `parse_job_metrics()` to use `check_file_termination()`
- [x] Update `update_job_status()` to handle TIMEOUT status
- [x] Add `is_timeout` and `termination_status` to metrics dict
- [x] Write unit tests for the fixed termination logic
- [ ] Test manually on existing job directories

**Success criteria:**
- ✅ Jobs with "Error" in output are marked FAILED
- ✅ Jobs with stale modification times are marked TIMEOUT
- ✅ Jobs with "aborting the run" are marked FAILED
- ✅ Only true successful completions are marked COMPLETED

### Phase 2: Add Completion Callback (Medium Priority)
- [ ] Implement `report_job_completion()` function
- [ ] Create `report_completion.py` CLI utility
- [ ] Add completion callback to Flux job template
- [ ] Add completion callback to SLURM job template
- [ ] Update `submit_jobs.py` to pass WORKFLOW_DB and JOB_ID
- [ ] Add integration tests

**Success criteria:**
- Jobs automatically update their status on completion
- Database correctly reflects job status within seconds of completion
- Failed jobs (crash before callback) still detectable via polling

### Phase 3: Testing & Documentation (Low Priority)
- [ ] Test with real ORCA jobs on Tuolumne (Flux)
- [ ] Test timeout detection (modify file times manually)
- [ ] Test concurrent job completions (database locking)
- [ ] Update workflow documentation with completion callback
- [ ] Add examples to QUICKSTART.md

## Edge Cases & Handling

### 1. Job Crashes Before Callback
**Problem:** Job crashes/killed before reaching completion callback.
**Solution:** Keep existing polling mechanism as fallback. Dashboard can detect jobs stuck in RUNNING state.

### 2. Database Locking (Concurrent Updates)
**Problem:** Multiple jobs finish simultaneously, causing DB lock contention.
**Solution:** Already handled by `_execute_with_retry()` with exponential backoff (line 88-115 in architector_workflow.py).

### 3. Network/DB Connection Failures
**Problem:** Compute node can't reach database (network issues, permissions).
**Solution:** Wrap callback in try/except, log error but don't fail the job. Polling fallback will catch it.

### 4. Stale RUNNING Jobs (Timeout Detection)
**Problem:** Jobs marked RUNNING but actually timed out/crashed.
**Solution:** Dashboard can detect jobs RUNNING longer than expected, manually update via `update_job_status()`.

### 5. Job ID Not Available
**Problem:** Job script doesn't receive JOB_ID env var.
**Solution:** Make completion callback optional. If WORKFLOW_DB or JOB_ID unset, skip callback (backwards compatible).

## Testing Plan

### Unit Tests

**File:** `tests/test_workflows.py`

```python
def test_parse_job_metrics_uses_check_file_termination():
    """Ensure parse_job_metrics calls check_file_termination."""
    # Test with normal termination
    # Test with timeout (old file)
    # Test with error in last 5 lines
    # Test with "aborting the run"

def test_update_job_status_handles_timeout():
    """Ensure TIMEOUT status is set correctly."""
    # Mock metrics with is_timeout=True
    # Verify status set to TIMEOUT

def test_report_job_completion():
    """Test job completion callback."""
    # Create test database
    # Call report_job_completion
    # Verify status updated correctly
```

### Integration Tests

**File:** `tests/test_integration_workflow.py`

```python
def test_end_to_end_job_completion():
    """Test complete workflow with completion callback."""
    # Create workflow DB
    # Submit mock job
    # Simulate job completion (write output files)
    # Call report_job_completion
    # Verify DB updated with correct status and metrics
```

### Manual Testing

1. **Test on real ORCA jobs:**
   - Submit jobs to Tuolumne Flux
   - Verify completion callback runs
   - Check database for correct status updates

2. **Test timeout detection:**
   - Create old output file (touch -t 202601010000 output.out)
   - Call parse_job_metrics
   - Verify TIMEOUT status

3. **Test error handling:**
   - Create output with "Error" in last 5 lines
   - Call parse_job_metrics
   - Verify FAILED status

## Success Metrics

### Correctness
- ✅ 100% of successful jobs marked COMPLETED
- ✅ 0% of timeout jobs marked COMPLETED (should be TIMEOUT)
- ✅ 0% of error jobs marked COMPLETED (should be FAILED)

### Performance
- ✅ Status updates within 5 seconds of job completion (with callback)
- ✅ No database lock errors under concurrent load (10+ jobs finishing simultaneously)

### Reliability
- ✅ Fallback polling still works if callback fails
- ✅ No job script failures due to callback errors

## Dependencies & Risks

### Dependencies
- Existing `check_file_termination()` function (already implemented)
- SQLite database with WAL mode (already enabled)
- Flux/SLURM environment variables support (standard)

### Risks
- **Low**: Database connection issues from compute nodes
  - Mitigation: Try/except around callback, polling fallback
- **Low**: Performance impact of DB writes from many jobs
  - Mitigation: WAL mode + retry logic already handles this
- **Medium**: Backwards compatibility with existing workflows
  - Mitigation: Make callback optional (only if env vars set)

## Documentation Requirements

### Update Files
1. `oact_utilities/workflows/README.md`
   - Add section on job completion callback
   - Explain how jobs self-report status

2. `oact_utilities/workflows/QUICKSTART.md`
   - Show example of completion callback in action
   - Explain environment variables (WORKFLOW_DB, JOB_ID)

3. Docstrings
   - `report_job_completion()` - comprehensive docstring with examples
   - `parse_job_metrics()` - update to mention check_file_termination usage
   - `update_job_status()` - update to mention timeout handling

### Example Usage

```python
# In job submission script:
from oact_utilities.workflows import ArchitectorWorkflow, submit_jobs

with ArchitectorWorkflow("workflow.db") as wf:
    ready_jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)

    # Submit with completion callback enabled (default)
    submit_jobs(
        workflow=wf,
        jobs=ready_jobs[:10],
        scheduler="flux",
        enable_completion_callback=True,  # NEW parameter
    )

# Jobs will automatically update their status on completion
# No need for manual polling with update_job_status()
```

## Future Enhancements

### Phase 4: Real-time Progress Updates (Future)
- Jobs report progress during optimization (geometry cycle count)
- Dashboard shows live progress bars for running jobs
- Requires intermediate checkpointing

### Phase 5: Failure Analysis (Future)
- Automatically parse error messages and categorize failures
- Suggest fixes for common errors (SCF convergence, memory, etc.)
- Integration with error recovery strategies

## Acceptance Criteria

### Part 1: Bug Fix
- [x] `parse_job_metrics()` uses `check_file_termination()` instead of simple string check
- [x] Timeout jobs are correctly marked as TIMEOUT (not COMPLETED or FAILED)
- [x] Error jobs are correctly marked as FAILED
- [x] Unit tests pass for all termination scenarios
- [ ] Manual testing confirms correct status detection

### Part 2: Completion Callback
- [ ] `report_job_completion()` function implemented and tested
- [ ] CLI utility `report_completion.py` works correctly
- [ ] Flux job templates include completion callback
- [ ] SLURM job templates include completion callback
- [ ] Job submission passes WORKFLOW_DB and JOB_ID env vars
- [ ] Integration tests pass
- [ ] Real ORCA jobs on Tuolumne correctly self-report completion
- [ ] Database remains consistent under concurrent updates
- [ ] Backwards compatible (works without callback if env vars unset)

## References

### Internal Code
- [oact_utilities/utils/status.py:15](oact_utilities/utils/status.py#L15) - `check_file_termination()`
- [oact_utilities/utils/analysis.py:1095](oact_utilities/utils/analysis.py#L1095) - `parse_job_metrics()`
- [oact_utilities/workflows/architector_workflow.py:520](oact_utilities/workflows/architector_workflow.py#L520) - `update_job_status()`
- [oact_utilities/utils/hpc.py](oact_utilities/utils/hpc.py) - Job template writers

### Related Issues
- Code review issue 007: Memory optimization (streaming file reads)
- Database schema versioning system (commit 16ad45e)
