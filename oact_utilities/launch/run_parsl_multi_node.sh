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
DB_PATH="/path/to/workflow.db"  #(update USE ACTINIDES DB)
ROOT_DIR="/path/to/job_output_dir" #(update)
ORCA_PATH="/path/to/orca"          #(update -- must be absolute path to ORCA binary)
JOB_PREFIX=""                     # optional stable prefix reused across coordinator requeues

BATCH_SIZE=500          # number of jobs to dispatch at once (update)
MAX_WORKERS=4              # workers per node (update)
CORES_PER_WORKER=20        # cores per worker (update)
CPUS_PER_NODE=""          # optional: reserve more scheduler cores/node than active workers use
CONDA_ENV="py10mpi" # (update)
CONDA_BASE="/usr/WS1/vargas58/miniconda3"
LD_LIBRARY_PATH_OVERRIDE=""  # (optional) set to override LD_LIBRARY_PATH in job scripts
JOB_TIMEOUT=432000          # 5 days per job
MAX_FAIL_COUNT=3

# W&B online monitoring (optional -- leave empty to disable)
WANDB_PROJECT=""        # e.g. "actinide-campaign"
WANDB_RUN_NAME=""       # display name in W&B UI (default: db filename stem)
WANDB_RUN_ID=""         # resume an existing run across batches

# SLURM scale-out settings
NODES_PER_BLOCK=1          # nodes per SLURM block (>1 enables multi-node with SrunLauncher)
MAX_BLOCKS=10              # max SLURM blocks Parsl can provision
INIT_BLOCKS=2              # blocks requested at startup
MIN_BLOCKS=1               # minimum blocks to keep alive
WALLTIME_HOURS=120          # walltime per block allocation (update)
QOS="frontier"
ACCOUNT="ODEFN5169CYFZ"

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

# TODO: we might need to add NBO Path if we decide to use it later

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --scheduler slurm \
    ${JOB_PREFIX:+--job-prefix "${JOB_PREFIX}"} \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    ${CPUS_PER_NODE:+--cpus-per-node "${CPUS_PER_NODE}"} \
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
    --qos "${QOS}" \
    --account "${ACCOUNT}" \
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
    #--dry-run # Uncomment to do a dry run (Parsl will spin up and prepare directories but not actually submit jobs)
