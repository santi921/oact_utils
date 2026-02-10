"""Quick script to check status and metrics for a single job directory."""

from pathlib import Path

from oact_utilities.utils.analysis import parse_job_metrics
from oact_utilities.utils.status import check_job_termination

job_dir = "jobs_parsl/job_2372"

# Check termination status
status_code = check_job_termination(job_dir)
status_map = {1: "SUCCESS", -1: "FAILED", -2: "TIMEOUT", 0: "RUNNING/INCOMPLETE"}
print(f"Job dir:    {job_dir}")
print(f"Termination: {status_map.get(status_code, 'UNKNOWN')} ({status_code})")

# Parse metrics
if Path(job_dir).exists():
    metrics = parse_job_metrics(job_dir, unzip=False)
    print(f"Success:     {metrics.get('success')}")
    print(f"Energy:      {metrics.get('final_energy')}")
    print(f"Max forces:  {metrics.get('max_forces')}")
    print(f"SCF steps:   {metrics.get('scf_steps')}")
else:
    print(f"Directory {job_dir} not found")
