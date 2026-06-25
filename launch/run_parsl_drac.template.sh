#!/bin/bash
# ===TEMPLATE-DOC-START (this block is stripped from generated lane scripts)
# TEMPLATE -- do not sbatch this file directly. Generate per-lane scripts:
#
#   python oact_utilities/scripts/stratify_lanes.py <db> [--cluster fir] \
#       [--ntasks 64] \
#       --template launch/run_parsl_drac.template.sh --outdir launch/generated/
#
# Double-brace placeholders are filled per lane by the generator (atom band,
# cores/worker, workers, ntasks, walltime, per-job timeout, DB path). Everything
# else is yours to edit ONCE: account, modules, venv, ROOT_DIR, basis, array.
#
# Requires a submit_jobs that supports --min-atoms/--max-atoms (branch
# feat/drac-min-atoms-lanes). Generate from a checkout/venv that has it.
#
# Single-node Parsl: one SLURM allocation occupies the reserved cores and Parsl
# packs MAX_WORKERS ORCA jobs onto it, each using CORES_PER_WORKER cores.
# ===TEMPLATE-DOC-END
# Generated lane '{{CLASS}}/{{LANE}}' -- regenerate with stratify_lanes.py; do not hand-edit.

#SBATCH --account=def-yqw
#SBATCH --time={{TIME}}                  # lane tier; keep tight for better backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={{NTASKS}}     # = MAX_WORKERS * CORES_PER_WORKER
#SBATCH --job-name=parsl-{{CLASS}}-{{LANE}}
#SBATCH --mem-per-cpu=3900M              # node ratio; scales mem with cores reserved
#SBATCH --array=1-25                     # allocations for this lane (tune by hand)
# No --qos, no --partition: DRAC auto-assigns the partition by time/cores.

set -euo pipefail

# ---- Configuration (edit these once; shared across lanes) ----
DB_PATH="{{DB_PATH}}"
ROOT_DIR="${HOME}/scratch/oact_jobs/jobs_parsl/"        # heavy I/O -> scratch (Lustre); one tree across lanes
VENV_PATH="${HOME}/projects/def-yqw/${USER}/oact-env"   # built by examples/drac/setup_venv.sh
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"

# ---- Lane parameters (filled by the generator) ----
MAX_WORKERS={{MAX_WORKERS}}             # concurrent ORCA jobs on the node
CORES_PER_WORKER={{CORES_PER_WORKER}}   # cores per ORCA job (== %pal nprocs)
MIN_ATOMS={{MIN_ATOMS}}                 # closed atom band [MIN_ATOMS, MAX_ATOMS]
MAX_ATOMS={{MAX_ATOMS}}
JOB_TIMEOUT={{JOB_TIMEOUT}}             # per-ORCA-job cap (s); set below --time
BATCH_SIZE=20                           # molecules pulled from the DB per allocation
MAX_FAIL_COUNT=10

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# --------------------------------------------------------------

# Modules first (ORCA chain), then the venv -- workers inherit this via fork.
module load ${MODULE_LOAD}
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
# Clear PYTHONPATH so a sticky module (e.g. ipykernel) can't shadow venv numpy.
unset PYTHONPATH

# Resolve the module's ORCA binary now ($EBROOTORCA expands in this shell).
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
    --min-atoms "${MIN_ATOMS}" \
    --max-atoms "${MAX_ATOMS}" \
    --max-blocks 1 \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks \
    --no-parsl-monitoring \
    --clean-on-complete
    # --purge-on-fail   # RE-ENABLE only once lanes are stable: during tuning it
                       # writes .do_not_rerun.json on OOM victims and deletes
                       # their orca.out evidence (permanently skipped thereafter).
