---
title: Add Parsl-based batch submission for exclusive node systems
type: feat
date: 2026-02-06
---

# Add Parsl-based batch submission for exclusive node systems

## Overview

Extend the job submission system to support concurrent execution of multiple ORCA jobs on exclusive HPC nodes using Parsl's HighThroughputExecutor. This addresses use cases where systems allocate entire nodes exclusively, making it efficient to run multiple jobs concurrently on the same node rather than submitting individual single-job scripts.

## Problem Statement / Motivation

**Current Behavior:**
- `submit_jobs.py` creates one Flux/SLURM script per job
- Each job typically runs on its own node
- Works well for shared systems with fine-grained scheduling

**Problem:**
On exclusive node systems (e.g., certain DoD HPC systems, Frontier-class machines):
- Each job allocation gets an entire node (64+ cores)
- Running a single 16-core ORCA job wastes 48+ cores
- Need to pack multiple jobs onto the same node for efficiency
- Current approach requires manually writing multi-job scripts

**Existing Solution (not integrated):**
`run_jobs_quacc_wave2.py` demonstrates Parsl-based concurrent execution:
- Uses `HighThroughputExecutor` to manage multiple workers per node
- Implements CPU pinning for NUMA affinity
- Dynamically provisions nodes via SlurmProvider
- Campaign-specific, not integrated with workflow system

**Desired Behavior:**
- Allow users to specify resource allocation (nodes, cores, time)
- Intelligently pack jobs onto exclusive nodes
- Integrate with existing `ArchitectorWorkflow` database
- Trust DB status for job filtering (no file activity checking)
- Maintain same CLI patterns and configuration as current system

**Implementation Constraints:**
- Flux: Single-node execution only (Parsl doesn't support Flux scale-out)
- SLURM: Multi-node support using existing wave2 configuration patterns

## Proposed Solution

Extend `oact_utilities/workflows/submit_jobs.py` with Parsl support by:

1. **Add `--use-parsl` flag** to enable Parsl-based submission mode
2. **Create Parsl configuration builders** for Flux (single-node) and SLURM (multi-node)
3. **Integrate with existing workflow** for status tracking and job filtering (DB-only)
4. **Fix concurrent execution** using `as_completed()` instead of blocking loops
5. **Two-phase rollout**: Flux implementation first, then SLURM using wave2 patterns

## Technical Approach

### Architecture

```
submit_jobs.py
│
├── submit_batch() [existing]
│   └── Traditional: One job script per job
│
└── submit_batch_parsl() [new]
    ├── Build Parsl config (inline for Flux/SLURM)
    ├── Filter jobs (DB status only - trust workflow)
    ├── Validate resource capacity
    ├── Submit python_app futures to Parsl
    ├── Monitor futures concurrently (as_completed)
    └── Cleanup Parsl executor

Phase 1: Flux (single-node only)
Phase 2: SLURM (reuse wave2 config for multi-node)
```

### System Presets

Define common HPC node configurations in a new module:

**File:** `oact_utilities/workflows/parsl_config.py`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class SystemPreset:
    """HPC system configuration preset."""
    name: str
    cores_per_node: int
    cores_per_worker: int
    max_workers_per_node: int
    scheduler: Literal["flux", "slurm"]
    default_queue: str
    default_allocation: str

SYSTEM_PRESETS = {
    "tuolumne_64core": SystemPreset(
        name="Tuolumne 64-core nodes",
        cores_per_node=64,
        cores_per_worker=16,
        max_workers_per_node=4,
        scheduler="flux",
        default_queue="pbatch",
        default_allocation="dnn-sim",
    ),
    "frontier_128core": SystemPreset(
        name="Frontier 128-core nodes",
        cores_per_node=128,
        cores_per_worker=16,
        max_workers_per_node=8,
        scheduler="slurm",
        default_queue="frontier",
        default_allocation="ODEFN5169CYFZ",
    ),
    # ... more presets
}

def build_parsl_config(
    preset: str | SystemPreset,
    max_nodes: int,
    walltime_hours: int,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
) -> Config:
    """Build Parsl Config from system preset."""
    # Implementation details
    pass
```

### Parsl Executor Configuration

**Flux Support:**

```python
from parsl.providers import FluxProvider  # If available
from parsl.launchers import SimpleLauncher

# Note: FluxProvider may not exist in Parsl - need to verify
# Alternative: Use subprocess to submit flux batch scripts
```

**SLURM Support** (following wave2 pattern):

```python
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

provider = SlurmProvider(
    qos=preset.default_queue,
    account=preset.default_allocation,
    nodes_per_block=1,
    init_blocks=2,
    min_blocks=1,
    max_blocks=max_nodes,
    walltime=f"{walltime_hours:02d}:00:00",
    scheduler_options=(
        f"#SBATCH --ntasks-per-node={preset.cores_per_node}\n"
        f"#SBATCH --cpus-per-task=1\n"
    ),
    exclusive=True,
    launcher=SimpleLauncher(),
    worker_init=worker_init_commands,
    parallelism=1.0,
)

executor = HighThroughputExecutor(
    label="parsl_htex",
    cores_per_worker=preset.cores_per_worker,
    max_workers_per_node=preset.max_workers_per_node,
    provider=provider,
)
```

### Job Filtering (DB Status Only)

**Implementation in `submit_batch_parsl()`:**

Trust the workflow database as the source of truth for job status. The existing `get_jobs_by_status()` API already filters correctly.

```python
def filter_jobs_for_submission(
    workflow: ArchitectorWorkflow,
    num_jobs: int,
    max_fail_count: int | None = None,
) -> list[JobRecord]:
    """Filter jobs that are ready to submit.

    Args:
        workflow: ArchitectorWorkflow instance
        num_jobs: Number of jobs to return
        max_fail_count: Skip jobs with fail_count >= this value

    Returns:
        List of JobRecords ready for submission
    """
    # Get ready jobs (DB is source of truth)
    ready_jobs = workflow.get_jobs_by_status([JobStatus.TO_RUN, JobStatus.READY])

    # Apply fail_count filter if specified
    if max_fail_count is not None:
        original_count = len(ready_jobs)
        ready_jobs = [j for j in ready_jobs if j.fail_count < max_fail_count]
        skipped = original_count - len(ready_jobs)
        if skipped > 0:
            print(f"Skipped {skipped} jobs with fail_count >= {max_fail_count}")

    # Limit to requested count
    jobs_to_submit = ready_jobs[:num_jobs]

    print(f"Found {len(ready_jobs)} ready jobs, submitting {len(jobs_to_submit)}")
    return jobs_to_submit
```

**Note:** If DB status is stale, run `dashboard.py --update` first to sync from filesystem.

### Parsl Job Wrapper

**File:** `oact_utilities/workflows/submit_jobs.py` (add to module)

```python
from parsl import python_app

@python_app
def orca_job_wrapper(
    job_id: int,
    job_dir: str,
    orca_config: dict,
    n_cores: int,
) -> dict:
    """Execute ORCA job within Parsl worker.

    This runs as a Parsl python_app, executing directly on the worker node.
    Parsl handles CPU affinity and worker management automatically.

    Args:
        job_id: Workflow database job ID
        job_dir: Absolute path to job directory
        orca_config: ORCA configuration dictionary
        n_cores: Number of cores for ORCA

    Returns:
        Dict with job_id, status, metrics
    """
    import os
    import subprocess
    import time
    from pathlib import Path

    job_dir_path = Path(job_dir)
    input_file = job_dir_path / "orca.inp"

    # Verify input file exists
    if not input_file.exists():
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Input file not found: {input_file}",
        }

    # Get ORCA path from config
    orca_cmd = orca_config.get("orca_path", "orca")

    # Let Parsl handle CPU affinity - no manual pinning needed
    os.environ["OMP_NUM_THREADS"] = "1"

    start_time = time.time()

    try:
        # Run ORCA
        result = subprocess.run(
            [orca_cmd, str(input_file)],
            cwd=job_dir,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per job
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            return {
                "job_id": job_id,
                "status": "completed",
                "wall_time": elapsed,
            }
        else:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": f"ORCA exited with code {result.returncode}",
                "stderr": result.stderr[:500],  # Truncate
            }

    except subprocess.TimeoutExpired:
        return {
            "job_id": job_id,
            "status": "timeout",
            "error": "Job exceeded 2 hour timeout",
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }
```

### Main Submission Function

**File:** `oact_utilities/workflows/submit_jobs.py` (add new function)

```python
def submit_batch_parsl(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    num_jobs: int,
    system_preset: str,
    max_nodes: int,
    walltime_hours: int,
    job_dir_pattern: str = "job_{orig_index}",
    orca_config: OrcaConfig | None = None,
    setup_func: Callable | None = None,
    n_cores: int = 16,
    queue: str | None = None,
    allocation: str | None = None,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    dry_run: bool = False,
    max_fail_count: int | None = None,
) -> list[int]:
    """Submit batch of jobs using Parsl for concurrent execution.

    Args:
        workflow: ArchitectorWorkflow instance
        root_dir: Root directory for job directories
        num_jobs: Total number of jobs to submit
        system_preset: System preset name (e.g., "tuolumne_64core")
        max_nodes: Maximum nodes to provision
        walltime_hours: Job walltime in hours
        job_dir_pattern: Pattern for job directory names
        orca_config: ORCA configuration
        setup_func: Optional setup function per job
        n_cores: Cores per ORCA job
        queue: Override preset queue
        allocation: Override preset allocation
        conda_env: Conda environment name
        conda_base: Conda base path
        ld_library_path: Override LD_LIBRARY_PATH
        dry_run: Prepare but don't submit
        max_fail_count: Skip jobs with fail_count >= this value

    Returns:
        List of submitted job IDs
    """
    import parsl
    from .parsl_config import SYSTEM_PRESETS, build_parsl_config

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Load system preset
    if system_preset not in SYSTEM_PRESETS:
        raise ValueError(f"Unknown preset: {system_preset}. Available: {list(SYSTEM_PRESETS.keys())}")

    preset = SYSTEM_PRESETS[system_preset]

    # Validate resource capacity
    total_cores = max_nodes * preset.cores_per_node
    max_concurrent = total_cores // n_cores

    if num_jobs > max_concurrent:
        print(f"Warning: {num_jobs} jobs requested but only {max_concurrent} can run concurrently")
        print(f"  ({max_nodes} nodes × {preset.cores_per_node} cores/node ÷ {n_cores} cores/job)")

    # Merge ORCA config
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}
    if "orca_path" not in config or config.get("orca_path") is None:
        config["orca_path"] = DEFAULT_ORCA_PATHS.get(
            preset.scheduler.lower(), DEFAULT_ORCA_PATHS["flux"]
        )

    # Filter jobs for submission (DB status only)
    jobs_to_submit = filter_jobs_for_submission(
        workflow,
        num_jobs=num_jobs,
        max_fail_count=max_fail_count,
    )

    if not jobs_to_submit:
        print("No jobs available for submission after filtering")
        return []

    # Limit to requested job count
    jobs_to_submit = filtered_jobs[:num_jobs]
    print(f"\nPreparing {len(jobs_to_submit)} jobs for Parsl submission...")

    # Prepare job directories
    print("Setting up job directories...")
    for i, job in enumerate(jobs_to_submit, 1):
        job_dir = prepare_job_directory(
            job,
            root_dir,
            job_dir_pattern=job_dir_pattern,
            orca_config=config,
            n_cores=n_cores,
            setup_func=setup_func,
        )
        print(f"  [{i}/{len(jobs_to_submit)}] Prepared {job_dir}")

    if dry_run:
        print("\n[DRY RUN] Would submit to Parsl executor")
        print(f"  System: {preset.name}")
        print(f"  Max nodes: {max_nodes}")
        print(f"  Walltime: {walltime_hours}h")
        print(f"  Workers per node: {preset.max_workers_per_node}")
        return [j.id for j in jobs_to_submit]

    # Build Parsl configuration
    print(f"\nBuilding Parsl config for {preset.name}...")
    parsl_config = build_parsl_config(
        preset=preset,
        max_nodes=max_nodes,
        walltime_hours=walltime_hours,
        conda_env=conda_env,
        conda_base=conda_base,
        ld_library_path=ld_library_path,
        queue=queue or preset.default_queue,
        allocation=allocation or preset.default_allocation,
    )

    # Initialize Parsl
    parsl.clear()
    parsl.load(parsl_config)
    print("Parsl executor loaded successfully")

    # Submit futures
    print(f"\nSubmitting {len(jobs_to_submit)} jobs to Parsl...")
    futures = []

    for job in jobs_to_submit:
        job_dir_name = job_dir_pattern.replace("{orig_index}", str(job.orig_index)).replace("{id}", str(job.id))
        job_dir_abs = (root_dir / job_dir_name).resolve()

        future = orca_job_wrapper(
            job_id=job.id,
            job_dir=str(job_dir_abs),
            orca_config=dict(config),
            n_cores=n_cores,
        )
        futures.append((job.id, future))

    # Mark jobs as running
    submitted_ids = [j.id for j in jobs_to_submit]
    workflow.mark_jobs_as_running(submitted_ids)
    print(f"Marked {len(submitted_ids)} jobs as RUNNING in database")

    # Monitor futures concurrently (CRITICAL: use as_completed, not sequential loop)
    print("\nMonitoring job execution...")
    print("(Press Ctrl+C for graceful shutdown)\n")

    completed_ids = []
    failed_ids = []

    # Create future->job_id mapping for concurrent completion
    from concurrent.futures import as_completed
    futures_map = {future: job_id for job_id, future in futures}

    try:
        # as_completed() yields futures as they finish (concurrent, not sequential!)
        for future in as_completed(futures_map.keys()):
            job_id = futures_map[future]
            try:
                result = future.result()

                if result["status"] == "completed":
                    workflow.update_status(job_id, JobStatus.COMPLETED)
                    completed_ids.append(job_id)
                    print(f"✓ Job {job_id} completed ({len(completed_ids)}/{len(futures)} done)")
                elif result["status"] == "timeout":
                    workflow.update_status(job_id, JobStatus.TIMEOUT, error_message=result.get("error"))
                    failed_ids.append(job_id)
                    print(f"⏱ Job {job_id} timeout")
                else:
                    workflow.update_status(job_id, JobStatus.FAILED, error_message=result.get("error"))
                    workflow._execute_with_retry(
                        "UPDATE structures SET fail_count = COALESCE(fail_count, 0) + 1 WHERE id = ?",
                        (job_id,)
                    )
                    failed_ids.append(job_id)
                    print(f"✗ Job {job_id} failed: {result.get('error', 'Unknown error')[:100]}")

            except Exception as e:
                workflow.update_status(job_id, JobStatus.FAILED, error_message=str(e))
                workflow._execute_with_retry(
                    "UPDATE structures SET fail_count = COALESCE(fail_count, 0) + 1 WHERE id = ?",
                    (job_id,)
                )
                failed_ids.append(job_id)
                print(f"✗ Job {job_id} exception: {str(e)[:100]}")

    except KeyboardInterrupt:
        print("\n\nGraceful shutdown requested...")

    finally:
        # Cleanup Parsl
        print("\nCleaning up Parsl executor...")
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()
        except Exception as e:
            print(f"Warning: Parsl cleanup failed: {e}")

        try:
            parsl.clear()
        except Exception:
            pass

    print(f"\nSubmission complete:")
    print(f"  ✓ Completed: {len(completed_ids)}")
    print(f"  ✗ Failed: {len(failed_ids)}")
    print(f"  Total: {len(submitted_ids)}")

    return submitted_ids
```

### CLI Integration

**File:** `oact_utilities/workflows/submit_jobs.py` (update `main()`)

```python
def main():
    """Main entry point for job submission script."""
    parser = argparse.ArgumentParser(
        description="Submit architector workflow jobs to HPC scheduler"
    )
    parser.add_argument("db_path", help="Path to workflow SQLite database")
    parser.add_argument("root_dir", help="Root directory for job directories")

    # Submission mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--use-parsl",
        action="store_true",
        help="Use Parsl for concurrent execution on exclusive nodes",
    )

    # Traditional batch options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of jobs to submit (default: 10). For Parsl mode, this is the total job count.",
    )

    # Parsl-specific options
    parsl_group = parser.add_argument_group("Parsl Options (--use-parsl)")
    parsl_group.add_argument(
        "--system-preset",
        choices=["tuolumne_64core", "frontier_128core", "custom"],
        help="System preset for Parsl configuration (required with --use-parsl)",
    )
    parsl_group.add_argument(
        "--max-nodes",
        type=int,
        default=2,
        help="Maximum nodes for Parsl to provision (default: 2)",
    )
    parsl_group.add_argument(
        "--conda-base",
        default="/usr/WS1/vargas58/miniconda3",
        help="Conda base path (default: /usr/WS1/vargas58/miniconda3)",
    )

    # ... existing args ...

    args = parser.parse_args()

    # Validate Parsl mode
    if args.use_parsl and not args.system_preset:
        parser.error("--system-preset is required when using --use-parsl")

    # Build ORCA config
    orca_config: OrcaConfig = {
        "functional": args.functional,
        "simple_input": args.simple_input,
        "actinide_basis": args.actinide_basis,
        "actinide_ecp": args.actinide_ecp,
        "non_actinide_basis": args.non_actinide_basis,
        "scf_MaxIter": args.scf_maxiter,
        "nbo": args.nbo,
        "opt": args.opt,
    }
    if args.orca_path:
        orca_config["orca_path"] = args.orca_path

    # Open workflow
    try:
        workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Submit based on mode
    if args.use_parsl:
        submitted_ids = submit_batch_parsl(
            workflow=workflow,
            root_dir=args.root_dir,
            num_jobs=args.batch_size,
            system_preset=args.system_preset,
            max_nodes=args.max_nodes,
            walltime_hours=args.n_hours,
            job_dir_pattern=args.job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            queue=args.queue,
            allocation=args.allocation,
            conda_env=args.conda_env,
            conda_base=args.conda_base,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
        )
    else:
        submitted_ids = submit_batch(
            workflow=workflow,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            scheduler=args.scheduler,
            job_dir_pattern=args.job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            n_hours=args.n_hours,
            queue=args.queue,
            allocation=args.allocation,
            conda_env=args.conda_env,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
        )

    print(f"\nTotal jobs submitted: {len(submitted_ids)}")
    workflow.close()
```

## Implementation Phases

### Phase 1: Flux Implementation (Single-Node)

**Goal:** Get Parsl working on Tuolumne with Flux for single-node concurrent execution.

**Tasks:**
- [x] Add parsl dependency to pyproject.toml
- [x] Add `build_parsl_config_flux()` inline function in `submit_jobs.py`
- [x] Implement `orca_job_wrapper()` python_app (reuse existing recipes where possible)
- [x] Implement simplified `filter_jobs_for_submission()` (DB status only)
- [x] Add `submit_batch_parsl()` main function with `as_completed()` loop
- [x] Add `--use-parsl` flag to CLI with Flux-specific args
- [x] Test locally: imports work, CLI parses correctly, dry-run works
- [ ] Test on Tuolumne: 10 jobs, 1 node, 4 workers
- [ ] Verify CPU affinity and concurrent execution

**Acceptance Criteria:**
- [ ] Jobs execute concurrently via Parsl on Flux (single node)
- [ ] `as_completed()` provides true concurrency (not sequential blocking)
- [ ] DB status updated as jobs complete
- [ ] Traditional submission still works (backward compatible)
- [ ] Graceful Ctrl+C shutdown works
- [ ] ORCA runs successfully with Parsl-managed CPU affinity

**Testing Strategy:**
- Start with minimal test: 5 jobs, 30-second timeout each
- Verify completion time is ~30 seconds (not 150 seconds if sequential)
- Check CPU usage confirms multiple ORCA processes running

### Phase 2: SLURM Implementation (Multi-Node)

**Goal:** Add SLURM support using existing `run_jobs_quacc_wave2.py` configuration patterns.

**Tasks:**
- [ ] Copy `base_config()` from `run_jobs_quacc_wave2.py` to `submit_jobs.py`
- [ ] Adapt for workflow integration (use `ArchitectorWorkflow` instead of custom dict)
- [ ] Add SLURM-specific CLI args (reuse wave2 patterns)
- [ ] Test on DoD SLURM system: 20 jobs, 2 nodes, 8 workers/node
- [ ] Verify multi-node provisioning and job distribution

**Acceptance Criteria:**
- [ ] Parsl provisions multiple SLURM nodes dynamically
- [ ] Jobs distributed across nodes as workers become available
- [ ] Same `as_completed()` monitoring works for SLURM
- [ ] Configuration matches proven wave2 patterns (cores, scheduler options)
- [ ] CLI supports both Flux and SLURM via `--scheduler` arg

**Documentation:**
- [ ] Add section to `workflows/README.md` explaining when to use Parsl
- [ ] Update CLAUDE.md with Parsl examples
- [ ] Document Flux single-node limitation
- [ ] Add troubleshooting guide (CPU affinity, timeout issues)

## Acceptance Criteria

### Functional Requirements

- [ ] User can submit jobs via Parsl with `--use-parsl` flag
- [ ] Flux: Single-node concurrent execution (4 workers on 64-core node)
- [ ] SLURM: Multi-node execution using wave2 config patterns
- [ ] Resource validation prevents over-subscription
- [ ] Jobs filtered based on DB status only (trust workflow database)
- [ ] Jobs with high fail_count skipped if `--max-fail-count` specified
- [ ] ORCA jobs execute concurrently using `as_completed()` (not sequential blocking)
- [ ] Job status updated in database as jobs complete
- [ ] Graceful shutdown on Ctrl+C
- [ ] Parsl executor cleaned up after submission
- [ ] Works with both Flux and SLURM schedulers

### Non-Functional Requirements

- [ ] No breaking changes to existing `submit_batch()` API
- [ ] CLI remains intuitive and follows existing patterns
- [ ] Error messages are clear and actionable
- [ ] Dry-run mode works for Parsl submissions
- [ ] Performance: Can prepare 100+ jobs in < 30 seconds
- [ ] Robustness: Handles partial failures gracefully

### Quality Gates

- [ ] Type hints on all new functions
- [ ] Docstrings in Google/NumPy style
- [ ] Code passes black formatting
- [ ] Code passes ruff linting
- [ ] Code passes mypy type checking
- [ ] Unit tests for core functions (job filtering, concurrent execution)
- [ ] Integration test for Parsl config building (mocked)
- [ ] Verify as_completed() provides true concurrency (not blocking)

## Success Metrics

**Efficiency Gains:**
- Reduce core wastage on exclusive nodes from 75% to < 10%
- Enable 4x concurrent jobs on 64-core nodes (16 cores/job)
- Throughput: 100+ jobs/hour on 10-node allocation

**Usability:**
- Single flag (`--use-parsl`) to enable feature
- System presets eliminate manual configuration
- Same workflow database and monitoring tools

**Reliability:**
- Failed jobs auto-increment fail_count
- DB status tracking prevents duplicate submissions
- Concurrent execution via as_completed() (not blocking)
- Graceful shutdown preserves job state

## Dependencies & Risks

### Dependencies

**Python Packages:**
- `parsl >= 2024.1` (need to verify Flux support)
- Existing: `sqlite3`, `pathlib`, `subprocess`, `time`

**External Systems:**
- Flux scheduler (Tuolumne HPC)
- SLURM scheduler (DoD systems)
- ORCA 6.0+ quantum chemistry engine

**Internal Modules:**
- `oact_utilities.workflows.architector_workflow` (ArchitectorWorkflow, JobStatus)
- `oact_utilities.workflows.submit_jobs` (existing functions)
- `oact_utilities.utils.status` (optional, for enhanced monitoring)

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parsl lacks Flux scale-out | Known | Low | Single-node Flux is sufficient for testing |
| CPU affinity conflicts | Medium | High | Let Parsl manage affinity (test on real hardware) |
| Database lock contention | Low | Medium | Existing WAL mode + retry logic handles this |
| Job status drift | Medium | Low | Run dashboard --update regularly to sync DB |
| Blocking future.result() loop | Known | High | Use as_completed() for true concurrency |
| Parsl version incompatibility | Low | High | Pin Parsl version in requirements.txt |
| Worker provisioning delays | Medium | Low | Set appropriate init_blocks and parallelism |

**Risk Mitigation Actions:**
1. Verify Parsl Flux support before Phase 3
2. Test CPU affinity on actual HPC nodes (not just local)
3. Monitor database lock frequency during high-load testing
4. Document Parsl version requirements in README

## Alternative Approaches Considered

### 1. Manual Job Packing (No Parsl)

**Approach:** Write Flux/SLURM scripts that run multiple ORCA jobs sequentially or via GNU Parallel.

**Pros:**
- No new dependencies
- Full control over execution order
- Simpler architecture

**Cons:**
- Manual CPU pinning error-prone
- No dynamic scheduling if jobs fail
- Hard to scale to many nodes
- Requires rewriting hpc.py job writers

**Decision:** Rejected. Parsl provides better abstraction and error handling.

### 2. Separate Parsl-Only Module

**Approach:** Create `submit_jobs_parsl.py` as standalone module, don't modify existing code.

**Pros:**
- Zero risk of breaking existing workflows
- Cleaner separation of concerns
- Can evolve independently

**Cons:**
- Code duplication (prepare_job_directory, etc.)
- Two CLI interfaces to maintain
- Users need to remember which command to use

**Decision:** Rejected. User prefers extending existing module with flag.

### 3. Always-On Parsl Mode

**Approach:** Use Parsl for all submissions, remove traditional mode.

**Pros:**
- Simpler codebase (no mode selection)
- Single path to maintain
- Better long-term architecture

**Cons:**
- Breaking change for existing users
- Overkill for single-job submissions
- Parsl overhead unnecessary on shared systems

**Decision:** Rejected. Keep both modes for flexibility.

## Future Considerations

### Extensibility

**Custom Executors:**
Allow users to provide custom Parsl executor configurations via config file:

```yaml
# ~/.oact_utils_parsl.yaml
custom_executor:
  name: "my_system"
  cores_per_node: 96
  cores_per_worker: 12
  max_workers_per_node: 8
  scheduler: "slurm"
  queue: "my_queue"
  allocation: "my_account"
```

**Multi-Site Execution:**
Support job submission across multiple HPC sites simultaneously using Parsl's multi-executor support.

**Dynamic Resource Scaling:**
Implement auto-scaling based on queue depth (add/remove nodes dynamically).

### Long-Term Vision

**1. Real-Time Dashboard Integration:**
Integrate with `dashboard.py` to show:
- Live worker status (provisioning, idle, busy)
- Job execution progress bars
- Resource utilization metrics

**2. Intelligent Job Scheduling:**
Use machine learning to predict job runtime and optimize packing:
- Prioritize fast jobs to maximize throughput
- Group similar jobs to reduce startup overhead
- Balance load across NUMA domains

**3. Fault Tolerance:**
Implement automatic retry with checkpoint/restart:
- Save ORCA intermediate files (gbw, hess)
- Resume failed jobs from checkpoint
- Exponential backoff for transient failures

**4. Multi-Objective Optimization:**
Support user preferences like:
- Minimize walltime (throughput)
- Minimize cost (core-hours)
- Maximize reliability (fewer large jobs)

## Documentation Plan

### User Documentation

**Update:** `oact_utilities/workflows/README.md`

Add section "Using Parsl for Exclusive Node Systems":
```markdown
## Using Parsl for Exclusive Node Systems

On HPC systems that allocate exclusive nodes, running multiple ORCA jobs
concurrently on the same node improves efficiency. The `--use-parsl` flag
enables Parsl-based submission for this use case.

### Quick Start

```bash
python -m oact_utilities.workflows.submit_jobs \
  workflow.db jobs/ \
  --use-parsl \
  --system-preset tuolumne_64core \
  --batch-size 100 \
  --max-nodes 10 \
  --n-hours 4
```

### System Presets

- `tuolumne_64core`: LLNL Tuolumne (64 cores/node, Flux)
- `frontier_128core`: OLCF Frontier (128 cores/node, SLURM)

### How It Works

1. Provisions up to `--max-nodes` nodes via Parsl
2. Runs `cores_per_node / cores_per_job` jobs concurrently per node
3. Dynamically schedules jobs as workers become available
4. Updates workflow database in real-time

### When to Use Parsl

Use Parsl (`--use-parsl`) when:
- HPC system allocates entire nodes exclusively
- Running many small jobs (< 2 hours each)
- Want to maximize core utilization

Use traditional mode (default) when:
- Jobs run on shared nodes
- Each job needs full node resources
- Simpler submission preferred
```

**Create:** `examples/parsl_submission_example.py`

Show end-to-end example of Parsl-based workflow submission.

**Update:** `CLAUDE.md`

Add section under "Common Tasks for Claude" > "HPC Job Submission":
```markdown
### Parsl-Based Concurrent Execution

For exclusive node systems, use Parsl to pack multiple jobs per node:

```python
from oact_utilities.workflows.submit_jobs import submit_batch_parsl

submitted = submit_batch_parsl(
    workflow=workflow,
    root_dir="jobs/",
    num_jobs=100,
    system_preset="tuolumne_64core",
    max_nodes=10,
    walltime_hours=4,
    dry_run=True,  # Test first
)
```

Key differences from traditional submission:
- Provisions nodes upfront (max_nodes parameter)
- Runs multiple jobs per node concurrently
- Monitors job completion concurrently via as_completed()
- Uses DB status only (no file scanning)
```

### Developer Documentation

**Create:** `oact_utilities/workflows/parsl_config.py` (module docstring)

```python
"""Parsl configuration presets for HPC systems.

This module provides pre-configured Parsl executors for common HPC systems
used by the oact_utilities workflow system. System presets encapsulate:

- Node hardware specs (cores per node)
- Optimal worker configuration (cores per worker, max workers)
- Scheduler-specific settings (Flux vs SLURM)
- Default queue and allocation names

Usage:
    from oact_utilities.workflows.parsl_config import SYSTEM_PRESETS, build_parsl_config

    preset = SYSTEM_PRESETS["tuolumne_64core"]
    config = build_parsl_config(preset, max_nodes=5, walltime_hours=2)

    parsl.load(config)

Adding Custom Presets:
    custom = SystemPreset(
        name="My System",
        cores_per_node=128,
        cores_per_worker=16,
        max_workers_per_node=8,
        scheduler="slurm",
        default_queue="batch",
        default_allocation="proj123",
    )
    SYSTEM_PRESETS["my_system"] = custom
"""
```

## References & Research

### Internal References

**Current Implementation:**
- `oact_utilities/workflows/submit_jobs.py` - Traditional job submission
- `oact_utilities/workflows/architector_workflow.py:315-399` - Status tracking API
- `oact_utilities/scripts/run_jobs_quacc_wave2.py` - Parsl template with CPU pinning
- `oact_utilities/utils/hpc.py` - Legacy job file writers
- `oact_utilities/utils/status.py` - File-based status checking

**Database Schema:**
- `oact_utilities/utils/architector.py:190-209` - Table definition with fail_count

**Configuration Patterns:**
- `oact_utilities/workflows/submit_jobs.py:20-66` - OrcaConfig TypedDict, defaults

### External References

**Parsl Documentation:**
- [Parsl User Guide](https://parsl.readthedocs.io/)
- [HighThroughputExecutor](https://parsl.readthedocs.io/en/stable/userguide/executors.html#high-throughput-executor)
- [SLURM Provider](https://parsl.readthedocs.io/en/stable/stubs/parsl.providers.SlurmProvider.html)

**Best Practices:**
- Parsl best practices for HPC: [2024 guide](https://parsl-project.org/tutorials/)
- CPU affinity with Parsl: Managed automatically by HTEx workers
- Parsl error handling: Use future.result() with try/except

### Related Work

**Existing Campaigns:**
- `oact_utilities/scripts/wave_one/` - Early batch submission patterns
- `oact_utilities/scripts/wave_two/` - Parsl-based SLURM campaigns
- `oact_utilities/scripts/multi_spin/` - Multi-state workflows

**Recent Commits:**
- `73973eb` - Fixed LD_LIBRARY_PATH default behavior
- `d66330c`, `550b4f5` - Edge cases in status checking
- `2942a8f` - Flux time format fix
- `bb5a12f` - Set log and work dirs in job writers

## Appendix: Example Usage

### Basic Parsl Submission

```bash
# Submit 100 jobs to Tuolumne using Parsl
python -m oact_utilities.workflows.submit_jobs \
  workflow.db \
  jobs/ \
  --use-parsl \
  --system-preset tuolumne_64core \
  --batch-size 100 \
  --max-nodes 10 \
  --n-hours 4 \
  --allocation dnn-sim \
  --conda-env py10mpi
```

### Dry Run First

```bash
# Test configuration without submitting
python -m oact_utilities.workflows.submit_jobs \
  workflow.db \
  jobs/ \
  --use-parsl \
  --system-preset frontier_128core \
  --batch-size 50 \
  --max-nodes 5 \
  --n-hours 2 \
  --dry-run
```

### With ORCA Configuration

```bash
# Custom ORCA settings
python -m oact_utilities.workflows.submit_jobs \
  workflow.db \
  jobs/ \
  --use-parsl \
  --system-preset tuolumne_64core \
  --batch-size 200 \
  --max-nodes 20 \
  --n-cores 16 \
  --functional wB97M-V \
  --actinide-basis ma-def-TZVP \
  --scf-maxiter 600
```

### Skip Chronic Failures

```bash
# Skip jobs that have failed 3+ times
python -m oact_utilities.workflows.submit_jobs \
  workflow.db \
  jobs/ \
  --use-parsl \
  --system-preset tuolumne_64core \
  --batch-size 100 \
  --max-nodes 10 \
  --max-fail-count 3
```

### Python API Usage

```python
from pathlib import Path
from oact_utilities.workflows import ArchitectorWorkflow
from oact_utilities.workflows.submit_jobs import submit_batch_parsl

# Open workflow
with ArchitectorWorkflow("workflow.db") as workflow:
    # Submit batch via Parsl
    submitted = submit_batch_parsl(
        workflow=workflow,
        root_dir=Path("jobs"),
        num_jobs=100,
        system_preset="tuolumne_64core",
        max_nodes=10,
        walltime_hours=4,
        n_cores=16,
        max_fail_count=3,
    )

    print(f"Submitted {len(submitted)} jobs")
```

### Monitoring After Submission

```bash
# Dashboard updates status from filesystem
python -m oact_utilities.workflows.dashboard \
  workflow.db \
  --update jobs/ \
  --show-metrics

# Check for failed jobs
python -m oact_utilities.workflows.dashboard \
  workflow.db \
  --show-failed

# Reset failures and resubmit
python -m oact_utilities.workflows.dashboard \
  workflow.db \
  --reset-failed \
  --max-retries 3

# Resubmit with Parsl
python -m oact_utilities.workflows.submit_jobs \
  workflow.db \
  jobs/ \
  --use-parsl \
  --system-preset tuolumne_64core \
  --batch-size 50 \
  --max-nodes 5 \
  --max-fail-count 3
```
