#!/bin/bash
# =============================================================================
# Multi-node Parsl submission for architector workflow jobs.
#
# This script acts as a lightweight coordinator that runs on a login or
# single compute node. Parsl's SlurmProvider automatically submits worker
# jobs to SLURM, scaling from init_blocks up to max_blocks nodes as needed.
#
# Usage:
#   sbatch run_parsl_multi_node.sh          # run coordinator as a batch job
#   bash run_parsl_multi_node.sh            # or run directly on a login node
# =============================================================================

#SBATCH --account=ODEFN5169CYFZ
#SBATCH --time=168:00:00
#SBATCH --qos=frontier
#SBATCH --constraint=standard
#SBATCH --job-name=parsl-multi-node
#SBATCH --nodes=1

# ---- Configuration (edit these) ----
DB_PATH="/path/to/workflow.db"
ROOT_DIR="/path/to/job_output_dir"

BATCH_SIZE=500
MAX_WORKERS=4              # workers per node
CORES_PER_WORKER=16        # cores per worker (must match ORCA nprocs)
CONDA_ENV="py10mpi"
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
JOB_TIMEOUT=72000          # 20 hours per job
MAX_FAIL_COUNT=3

# SLURM scale-out settings
MAX_BLOCKS=10              # max SLURM nodes Parsl can provision
INIT_BLOCKS=2              # nodes requested at startup
MIN_BLOCKS=1               # minimum nodes to keep alive
WALLTIME_HOURS=24          # walltime per worker node allocation
QOS="frontier"
ACCOUNT="ODEFN5169CYFZ"

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
    --scheduler slurm \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --conda-env "${CONDA_ENV}" \
    --conda-base "${CONDA_BASE}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --max-blocks "${MAX_BLOCKS}" \
    --init-blocks "${INIT_BLOCKS}" \
    --min-blocks "${MIN_BLOCKS}" \
    --walltime-hours "${WALLTIME_HOURS}" \
    --qos "${QOS}" \
    --account "${ACCOUNT}" \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}"
