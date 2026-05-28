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

GLOBUS_TRANSFER_FLAG=""
if [ "$1" = "--globus-transfer" ]; then
    GLOBUS_TRANSFER_FLAG="--globus-transfer"
fi

# ---- Configuration (edit these) ----
DB_PATH="/path/to/workflow.db"
ROOT_DIR="/path/to/job_output_dir"
ORCA_PATH="/path/to/orca"
JOB_PREFIX=""  # optional stable prefix reused across coordinator requeues

BATCH_SIZE=500
# Reserve full nodes but intentionally leave some cores idle for memory headroom.
MAX_WORKERS=8
CORES_PER_WORKER=8
CPUS_PER_NODE=192
CONDA_ENV="py10mpi"
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
LD_LIBRARY_PATH_OVERRIDE=""
JOB_TIMEOUT=432000
MAX_FAIL_COUNT=3

# W&B online monitoring (optional -- leave empty to disable)
WANDB_PROJECT=""        # e.g. "actinide-campaign"
WANDB_RUN_NAME=""       # display name in W&B UI (default: db filename stem)
WANDB_RUN_ID=""         # resume an existing run across batches

# Globus backup (optional). Endpoint values may also be provided by the
# environment. Use GLOBUS_TRANSFER_REFRESH_TOKEN for campaign runs.
export GLOBUS_SOURCE_ENDPOINT_ID="${GLOBUS_SOURCE_ENDPOINT_ID:-}"
export GLOBUS_DESTINATION_ENDPOINT_ID="${GLOBUS_DESTINATION_ENDPOINT_ID:-}"
export GLOBUS_DEST_ROOT="${GLOBUS_DEST_ROOT:-}"
export GLOBUS_CLIENT_ID="${GLOBUS_CLIENT_ID:-}"
export GLOBUS_TRANSFER_REFRESH_TOKEN="${GLOBUS_TRANSFER_REFRESH_TOKEN:-}"
export GLOBUS_CLIENT_SECRET="${GLOBUS_CLIENT_SECRET:-}"  # optional
GLOBUS_CONNECT_PERSONAL_BIN="${GLOBUS_CONNECT_PERSONAL_BIN:-globusconnectpersonal}"

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

if [ "${GLOBUS_TRANSFER_FLAG}" = "--globus-transfer" ]; then
    GLOBUS_CONNECT_JOB_ID="${SLURM_JOB_ID:-${PBS_JOBID:-${FLUX_JOB_ID:-$$}}}"
    nohup "${GLOBUS_CONNECT_PERSONAL_BIN}" -start >"/tmp/globusconnectpersonal_${GLOBUS_CONNECT_JOB_ID}.log" 2>&1 &
fi

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --scheduler pbspro \
    ${GLOBUS_TRANSFER_FLAG} \
    ${JOB_PREFIX:+--job-prefix "${JOB_PREFIX}"} \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --cpus-per-node "${CPUS_PER_NODE}" \
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
    --orca-path "${ORCA_PATH}" \
    ${WANDB_PROJECT:+--wandb-project "${WANDB_PROJECT}"} \
    ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
    ${WANDB_RUN_ID:+--wandb-run-id "${WANDB_RUN_ID}"}
