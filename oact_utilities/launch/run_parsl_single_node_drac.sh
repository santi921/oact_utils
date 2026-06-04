#!/bin/bash
# =============================================================================
# Single-node Parsl submission for Digital Research Alliance of Canada clusters
# (Fir / Narval / Nibi / Rorqual / Trillium).
#
# ONE SLURM allocation occupies a node (or part of one) and Parsl packs
# MAX_WORKERS ORCA jobs onto it concurrently, each using CORES_PER_WORKER cores.
# This is the node-packing alternative to per-molecule sbatch -- fewer jobs
# (relieves the 1000-job/account cap), at the cost of worse backfill than many
# short jobs. Request a PARTIAL node to schedule faster (see ---ntasks below).
#
# Parsl LocalProvider forks workers from this script's shell, so they inherit the
# module-loaded ORCA chain and the activated venv set up below -- no module load
# inside workers.
#
# Usage (batch):
#   sbatch run_parsl_single_node_drac.sh
# Usage (interactive):
#   salloc --account=def-yqw --nodes=1 --ntasks=64 --time=3:00:00
#   bash run_parsl_single_node_drac.sh
# =============================================================================

#SBATCH --account=def-yqw
#SBATCH --time=3:00:00            # keep tight; <3h tier backfills on the most nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64      # cores to grab: 64 = partial Fir node (schedules
                                  # faster); 192 = whole node. Must be >= MAX_WORKERS*CORES_PER_WORKER.
#SBATCH --job-name=parsl-drac
# No --qos, no --partition: DRAC auto-assigns the partition by time/cores.

set -euo pipefail

# ---- Configuration (edit these) ----
DB_PATH="${HOME}/projects/def-yqw/${USER}/working_dbs/campaign.db"
ROOT_DIR="${HOME}/scratch/jobs_scratch"          # heavy I/O -> scratch (Lustre)
VENV_PATH="${HOME}/projects/def-yqw/${USER}/oact-env"   # built by examples/drac/setup_venv.sh
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"

MAX_WORKERS=4               # concurrent ORCA jobs on the node
CORES_PER_WORKER=16         # cores per ORCA job (== %pal nprocs); 4*16=64 == --ntasks
BATCH_SIZE=200              # molecules to pull from the DB this run
JOB_TIMEOUT=10800           # 3h per ORCA job (seconds); <= the SBATCH --time
MAX_FAIL_COUNT=10

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

# Modules first (ORCA chain), then the venv -- workers inherit this via fork.
module load ${MODULE_LOAD}
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
# Clear PYTHONPATH so a stray module (e.g. a sticky ipykernel) can't shadow the
# venv's numpy/pandas with mismatched module copies. Workers inherit this.
unset PYTHONPATH

# Resolve the module's ORCA binary now ($EBROOTORCA expands in this shell).
# Parsl's ASE/quacc calculator needs a concrete path, not the literal env var.
ORCA_PATH="${EBROOTORCA}/orca"

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --hpc-site drac \
    --scheduler slurm \
    --allocation def-yqw \
    --orca-path "${ORCA_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --n-cores "${CORES_PER_WORKER}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    --max-blocks 1 \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks
