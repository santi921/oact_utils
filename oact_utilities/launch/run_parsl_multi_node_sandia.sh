#!/bin/bash
# =============================================================================
# Multi-node Parsl coordinator for Sandia CTS1/TLCC2 (attaway / ecl).
#
# This coordinator runs on a login node or as a lightweight sbatch job.
# Parsl SlurmProvider automatically submits worker blocks. Each block's
# worker_init does the full bootstrap (module load, MPI_ROOT, OMPI_MCA),
# so the coordinator only needs Python -- not OpenMPI.
#
# Usage (batch -- coordinator as a small sbatch job):
#   sbatch run_parsl_multi_node_sandia.sh
#
# Usage (interactive login node):
#   conda activate oact
#   bash run_parsl_multi_node_sandia.sh
# =============================================================================

#SBATCH --account=fy250086
#SBATCH --time=168:00:00
#SBATCH --qos=normal
#SBATCH --partition=attaway
#SBATCH --job-name=parsl-coord-sandia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ---- Configuration (edit these) ----
DB_PATH="/tscratch/${USER}/working_dbs/actinides_dod_47106_chunk4.db"
ROOT_DIR="/tscratch/${USER}/jobs_scratch/"
ORCA_PATH="/home/${USER}/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
CONDA_ENV="oact"
CONDA_BASE="/home/${USER}/miniconda3"    # adjust if conda is elsewhere

BATCH_SIZE=500
MAX_WORKERS=3
CORES_PER_WORKER=12         # CTS1 attaway: 36 cores / 3 workers
JOB_TIMEOUT=172800          # 48 hours per job (seconds)
MAX_FAIL_COUNT=10

# SLURM scale-out settings
NODES_PER_BLOCK=1           # nodes per Parsl block (>1 uses SrunLauncher)
MAX_BLOCKS=5                # max simultaneous SLURM allocations
INIT_BLOCKS=2               # blocks requested at startup
MIN_BLOCKS=1                # minimum blocks kept alive
WALLTIME_HOURS=48           # walltime per block allocation
QOS="normal"
ACCOUNT="fy250086"
PARTITION="attaway"
OPENMPI_MODULE="aue/openmpi/4.1.6-gcc-12.3.0"

# W&B online monitoring (optional -- leave empty to disable)
WANDB_PROJECT=""
WANDB_RUN_NAME=""
WANDB_RUN_ID=""

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

# Activate Python env (coordinator needs python; workers get their own env via worker_init)
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --hpc-site sandia \
    --scheduler slurm \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --n-cores "${CORES_PER_WORKER}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --orca-path "${ORCA_PATH}" \
    --nodes-per-block "${NODES_PER_BLOCK}" \
    --max-blocks "${MAX_BLOCKS}" \
    --init-blocks "${INIT_BLOCKS}" \
    --min-blocks "${MIN_BLOCKS}" \
    --walltime-hours "${WALLTIME_HOURS}" \
    --qos "${QOS}" \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --openmpi-module "${OPENMPI_MODULE}" \
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
