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
#
# Enable Globus backup:
#   bash run_parsl_single_node.sh --globus-transfer
#   sbatch run_parsl_single_node.sh --globus-transfer
# =============================================================================

#SBATCH --account=ODEFN5169CYFZ
#SBATCH --time=168:00:00
#SBATCH --qos=frontier
#SBATCH --constraint=standard
#SBATCH --job-name=parsl-single-node
#SBATCH --nodes=1

GLOBUS_TRANSFER_FLAG=""
if [ "$1" = "--globus-transfer" ]; then
    GLOBUS_TRANSFER_FLAG="--globus-transfer"
fi

# ---- Configuration (edit these) ----
DB_PATH="/path/to/workflow.db"
ROOT_DIR="/path/to/job_output_dir"

BATCH_SIZE=100
MAX_WORKERS=4
CORES_PER_WORKER=16
CONDA_ENV="py10mpi"
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
LD_LIBRARY_PATH_OVERRIDE=""  # (optional) set to override LD_LIBRARY_PATH in job scripts
JOB_TIMEOUT=604800          # 7 days per job
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

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=500
# -----------------------------------

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
    --scheduler flux \
    ${GLOBUS_TRANSFER_FLAG} \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --conda-env "${CONDA_ENV}" \
    --conda-base "${CONDA_BASE}" \
    ${LD_LIBRARY_PATH_OVERRIDE:+--ld-library-path "${LD_LIBRARY_PATH_OVERRIDE}"} \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks \
    ${WANDB_PROJECT:+--wandb-project "${WANDB_PROJECT}"} \
    ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
    ${WANDB_RUN_ID:+--wandb-run-id "${WANDB_RUN_ID}"}
