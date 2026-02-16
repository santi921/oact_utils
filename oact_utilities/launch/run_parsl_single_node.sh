#!/bin/bash
# =============================================================================
# Single-node Parsl submission for architector workflow jobs.
#
# This script runs on a single allocated node (e.g. from a Flux or SLURM
# interactive/batch allocation) and uses Parsl's LocalProvider to fan out
# multiple ORCA workers concurrently on that node.
#
# Usage (Flux):
#   flux batch -N1 -n 64 -q pbatch -B dnn-sim -t 480m run_parsl_single_node.sh
#
# Usage (SLURM):
#   sbatch run_parsl_single_node.sh
# =============================================================================

#SBATCH --account=ODEFN5169CYFZ
#SBATCH --time=168:00:00
#SBATCH --qos=frontier
#SBATCH --constraint=standard
#SBATCH --job-name=parsl-single-node
#SBATCH --nodes=1

# ---- Configuration (edit these) ----
DB_PATH="/path/to/workflow.db"
ROOT_DIR="/path/to/job_output_dir"

BATCH_SIZE=100
MAX_WORKERS=4
CORES_PER_WORKER=16
CONDA_ENV="py10mpi"
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
JOB_TIMEOUT=604800          # 7 days per job
MAX_FAIL_COUNT=3

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
# -----------------------------------

source ~/.bashrc
conda activate "${CONDA_ENV}"

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --scheduler flux \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --conda-env "${CONDA_ENV}" \
    --conda-base "${CONDA_BASE}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}"
