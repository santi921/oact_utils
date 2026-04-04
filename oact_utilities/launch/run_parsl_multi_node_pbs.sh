#!/bin/bash
# =============================================================================
# Multi-node Parsl submission for architector workflow jobs on PBS Pro.
#
# This script acts as a lightweight coordinator that runs inside a small PBS
# allocation. Parsl's PBSProProvider automatically submits worker jobs to PBS,
# scaling from init_blocks up to max_blocks blocks as needed.
#
# Usage:
#   qsub run_parsl_multi_node_pbs.sh
# =============================================================================

#PBS -N parsl-multi-node
#PBS -A YOUR_ACCOUNT
#PBS -q YOUR_QUEUE
#PBS -l walltime=168:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -j oe

# ---- Configuration (edit these) ----
DB_PATH="/path/to/workflow.db"
ROOT_DIR="/path/to/job_output_dir"
ORCA_PATH="/path/to/orca"

BATCH_SIZE=500
MAX_WORKERS=12
CORES_PER_WORKER=16
CONDA_ENV="py10mpi"
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
LD_LIBRARY_PATH_OVERRIDE=""
JOB_TIMEOUT=432000
MAX_FAIL_COUNT=3

# PBS Pro scale-out settings
NODES_PER_BLOCK=39
MAX_BLOCKS=10
INIT_BLOCKS=2
MIN_BLOCKS=1
WALLTIME_HOURS=120
QUEUE="YOUR_QUEUE"
ACCOUNT="YOUR_ACCOUNT"

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=500
# -----------------------------------

cd "${PBS_O_WORKDIR}"
source ~/.bashrc
conda activate "${CONDA_ENV}"

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --scheduler pbspro \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --queue "${QUEUE}" \
    --account "${ACCOUNT}" \
    --conda-env "${CONDA_ENV}" \
    --conda-base "${CONDA_BASE}" \
    ${LD_LIBRARY_PATH_OVERRIDE:+--ld-library-path "${LD_LIBRARY_PATH_OVERRIDE}"} \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --nodes-per-block "${NODES_PER_BLOCK}" \
    --max-blocks "${MAX_BLOCKS}" \
    --init-blocks "${INIT_BLOCKS}" \
    --min-blocks "${MIN_BLOCKS}" \
    --walltime-hours "${WALLTIME_HOURS}" \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks \
    --orca-path "${ORCA_PATH}"
