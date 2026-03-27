---
title: "fix: Pre-submission disk check to prevent re-submitting completed jobs"
type: fix
date: 2026-03-27
---

# Pre-Submission Disk Check to Prevent Re-Submitting Completed Jobs

## Overview

Add a disk-based completion check before job submission to prevent re-submitting jobs that already completed on disk but whose DB status was never updated (e.g., `dashboard --update` was not run).

## Problem Statement

`filter_jobs_for_submission()` trusts the DB status entirely. If a job completed on disk but the DB still says `to_run` (user forgot to run `--update`, or the process crashed before updating), the job is re-submitted and `prepare_job_directory()` overwrites `orca.inp`. Once the job re-runs, the prior `orca.out` results are destroyed.

This is a real workflow gap: every older script-based submission path (`launch_flux_jobs`, `launch_flux_sella_jobs`) has a `skip_done` guard. The DB-managed workflow submission is the only path without one.

## Proposed Solution

Add a `_filter_completed_on_disk()` function that checks each candidate job's directory for completed output before submission. Jobs found completed on disk are auto-updated to COMPLETED in the DB and excluded from submission. Jobs found failed are auto-updated to FAILED.

This follows the existing `_filter_marker_jobs()` pattern: a post-filter step that runs after `filter_jobs_for_submission()` but before `mark_jobs_as_running()`.

## Technical Considerations

### Where to insert the check

In `submit_jobs.py`, after `_filter_marker_jobs()` and before `mark_jobs_as_running()`:
- Parsl flow: between lines ~932 and ~940
- Traditional flow: between lines ~1329 and ~1337

### Performance on Lustre

- **No directory**: `os.path.exists()` is a single stat call (~1-5ms). For first-time submissions where most directories do not exist, the check adds <3 seconds for 500 jobs.
- **Directory exists**: `check_job_termination()` streams the last 10 lines via `deque(f, maxlen=10)`. For a 100MB ORCA output on Lustre: ~200-500ms per file.
- **Worst case (500 existing dirs)**: Sequential ~100-250s, parallel with 4 workers ~25-65s. Acceptable for a guard before a multi-hour batch.
- **Optimization**: Short-circuit if `job_dir` is NULL (no directory ever created) or directory does not exist.

### Edge cases

| Scenario | `check_job_termination` returns | Action |
|---|---|---|
| No directory exists | N/A (short-circuit) | Submit normally |
| Directory exists, no .out file | 0 (running/unknown) | Submit normally |
| Completed on disk | 1 | Update DB to COMPLETED, skip submission |
| Failed on disk | -1 | Update DB to FAILED, skip submission |
| Timeout on disk | -2 | Submit normally (user wants to retry) |
| Truncated/corrupted output | 0 (safe default) | Submit normally |
| Sella job, `sella_status.txt` says CONVERGED | 1 | Update DB to COMPLETED, skip |

### Content-before-age rule

The `check_job_termination()` function already implements the content-before-age priority (documented in `docs/solutions/logic-errors/recheck-completed-timeout-bug.md`). A completed job is never incorrectly treated as timed out. No new logic needed for this.

### What about `--force` override?

Sometimes users intentionally want to re-run completed jobs (e.g., with different ORCA settings). Add a `--skip-disk-check` flag to both CLI paths to bypass this guard.

## Acceptance Criteria

- [ ] Jobs that completed on disk but have `to_run` status in DB are detected and skipped
- [ ] Detected completed jobs are auto-updated to COMPLETED in the DB
- [ ] Detected failed jobs are auto-updated to FAILED in the DB
- [ ] Jobs with timeout status on disk (-2) are submitted normally (user wants retry)
- [ ] Jobs with no directory or no output file are submitted normally
- [ ] `--skip-disk-check` flag bypasses the guard
- [ ] Check runs in both Parsl and traditional submission paths
- [ ] Print summary: "Skipped N jobs already completed on disk (updated DB)"
- [ ] No performance regression for first-time submissions (no directories exist)

## Success Metrics

- Zero accidental re-submissions of completed jobs in normal workflow
- <5 second overhead for typical submission batches (200-500 jobs, most without directories)

## Dependencies and Risks

- **Depends on**: `check_job_termination()` in `utils/status.py` (already battle-tested)
- **Risk**: False positives (job incorrectly skipped). Mitigated by the conservative return values -- `check_job_termination` returns 0 (submit normally) for ambiguous cases.
- **Risk**: Performance on re-submission batches where all directories exist. Mitigated by parallelization if needed (pattern exists in dashboard.py).

## References

- [submit_jobs.py:351-403](oact_utilities/workflows/submit_jobs.py#L351-L403) -- `_filter_marker_jobs()` (pattern to follow)
- [status.py:183-265](oact_utilities/utils/status.py#L183-L265) -- `check_job_termination()` (the check to call)
- [jobs.py:6](oact_utilities/utils/jobs.py#L6) -- `launch_flux_sella_jobs()` with `skip_done=True` (prior art)
- `docs/solutions/logic-errors/recheck-completed-timeout-bug.md` -- content-before-age rule
