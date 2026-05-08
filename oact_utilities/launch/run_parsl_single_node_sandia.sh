#!/bin/bash
# =============================================================================
# Single-node Parsl submission for Sandia CTS1/TLCC2 (attaway / ecl).
#
# Parsl LocalProvider forks workers from this script's shell, so they inherit
# the OpenMPI env and OMPI_MCA settings set below -- no module load inside
# workers.
#
# Usage (interactive allocation):
#   salloc -N1 -p attaway -A fy250086 -t 8:00:00
#   conda activate oact            # activate Python env first
#   bash run_parsl_single_node_sandia.sh
#
# Usage (batch):
#   sbatch run_parsl_single_node_sandia.sh
# =============================================================================

#SBATCH --account=fy250086
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --partition=attaway
#SBATCH --job-name=parsl-single-sandia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36

# ---- Configuration (edit these) ----
DB_PATH="/tscratch/${USER}/working_dbs/actinides_dod_47106_chunk4.db"
ROOT_DIR="/tscratch/${USER}/jobs_scratch/"
ORCA_PATH="/home/${USER}/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
CONDA_ENV="oact"
CONDA_BASE="/home/${USER}/miniconda3"    # adjust if conda is elsewhere

BATCH_SIZE=200
MAX_WORKERS=3
CORES_PER_WORKER=12         # CTS1 attaway: 36 cores / 3 workers
JOB_TIMEOUT=172800          # 48 hours per job (seconds)
MAX_FAIL_COUNT=10

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

# OpenMPI transport -- disable PSM2/Omni-Path, force TCP
module load aue/openmpi/4.1.6-gcc-12.3.0
export OMPI_MCA_pml=ob1
export OMPI_MCA_mtl='^psm2'
export OMPI_MCA_btl='tcp,self,vader'

# Activate Python env if not already active (no-op if already active)
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
    --max-blocks 1 \
    --orca-path "${ORCA_PATH}" \
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
