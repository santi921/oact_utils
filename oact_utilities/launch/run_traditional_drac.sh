#!/bin/bash
# =============================================================================
# Traditional (non-Parsl) submission for Digital Research Alliance of Canada
# clusters: Fir / Narval / Nibi / Rorqual / Trillium.
#
# Why traditional, not Parsl: DRAC scheduling rewards MANY SHORT single-node
# jobs that ride backfill into idle gaps (jobs under 3h get the largest node
# pool). Traditional mode submits one independent sbatch per molecule, which is
# exactly that shape. --hpc-site drac is SLURM-only and rejects --use-parsl.
#
# Run this on a LOGIN node (it needs the venv to run submit_jobs, and submit_jobs
# shells out to `sbatch`). Each per-job script it writes module-loads ORCA and
# runs $EBROOTORCA/orca itself -- no conda, no srun/mpirun, no qos/partition.
#
# submit_jobs submits BATCH_SIZE jobs then exits. Re-run it to top up the queue
# as jobs finish (crontab is disabled on Fir; re-run by hand or a watch loop).
# DRAC caps 1000 queued+running jobs per account -- keep BATCH_SIZE under that.
#
# Usage:
#   bash run_traditional_drac.sh
# =============================================================================

set -euo pipefail

# ---- Configuration (edit these) ----
DB_PATH="${HOME}/projects/def-yqw/${USER}/working_dbs/campaign.db"
ROOT_DIR="${HOME}/scratch/jobs_scratch"          # heavy I/O -> scratch (Lustre)
ACCOUNT="def-yqw"                                # your RAP; Default RAP keeps the
                                                 # fairshare hit off the sponsor's RAC
VENV_PATH="${HOME}/projects/def-yqw/${USER}/oact-env"   # built by examples/drac/setup_venv.sh
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"   # per-job ORCA chain

BATCH_SIZE=200              # sbatch jobs to submit this run (keep < 1000 in flight)
N_CORES=16                  # cores per job == %pal nprocs in the .inp
N_HOURS=3                   # keep <=3h to ride backfill on the largest node pool
MAX_FAIL_COUNT=10
MAX_ATOMS=                  # optional: only submit molecules with natoms <= this

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

# Activate the venv so `python -m oact_utilities...` runs (login node only).
# Per-job scripts load their own ORCA module chain; this is just for the driver.
module load StdEnv/2023 "${PYTHON_MODULE}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# No --orca-path (uses module's $EBROOTORCA/orca), no --use-parsl, no --qos/--partition.
python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --hpc-site drac \
    --scheduler slurm \
    --allocation "${ACCOUNT}" \
    --module-load "${MODULE_LOAD}" \
    --venv-path "${VENV_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --n-cores "${N_CORES}" \
    --n-hours "${N_HOURS}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    ${MAX_ATOMS:+--max-atoms "${MAX_ATOMS}"} \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks
