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
#SBATCH --time=3:00:00            # BACKFILL role: <3h tier backfills on the most nodes.
                                  # MAIN-QUEUE role: raise to 24:00:00 and set MAX_ATOMS=0.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64      # cores to grab: 64 = partial Fir node (schedules
                                  # faster); 192 = whole node. Must be >= MAX_WORKERS*CORES_PER_WORKER.
#SBATCH --mem-per-cpu=3900M       # node ratio (~4GB/core on Fir/Narval/Nibi/Rorqual);
                                  # WITHOUT this DRAC gives a tiny default -> ORCA OOM-killed.
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
MAX_ATOMS=40                # BACKFILL knob: only submit molecules with natoms <= this so
                            # jobs stay short and ride the <3h backfill pool. 30-50 is a
                            # good range. Set 0 (or empty) for NO cap -- pair that with a
                            # 24h+ SBATCH --time for the big-molecule "main" queue.

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

# --- memory + core sizing (auto-derived from the SLURM allocation) -----------
# ORCA's %maxcore is per MPI rank; with MAX_WORKERS jobs running at once the
# memory sum must fit the allocation or the kernel OOM-kills ranks -- which
# surfaces (confusingly) as ORCA "aborting the run". Derive --mem-per-job from
# the allocation so it can't be hand-mis-set.
_cores_needed=$(( MAX_WORKERS * CORES_PER_WORKER ))
_alloc_cpus="${SLURM_CPUS_ON_NODE:-}"

# Core-oversubscription guard: more worker cores than the allocation has means
# ORCA's ranks fight over too few cores (intermittent MPI-startup aborts).
if [ -n "${_alloc_cpus}" ] && [ "${_cores_needed}" -gt "${_alloc_cpus}" ]; then
    echo "ERROR: MAX_WORKERS*CORES_PER_WORKER=${_cores_needed} > allocated cores ${_alloc_cpus}." >&2
    echo "       Lower MAX_WORKERS/CORES_PER_WORKER or raise #SBATCH --ntasks-per-node." >&2
    exit 1
fi

# Per-job memory budget = node memory / workers. SLURM exposes the allocation as
# SLURM_MEM_PER_NODE (from --mem) or SLURM_MEM_PER_CPU (from --mem-per-cpu).
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    _alloc_mem_mb="${SLURM_MEM_PER_NODE}"
elif [ -n "${SLURM_MEM_PER_CPU:-}" ] && [ -n "${_alloc_cpus}" ]; then
    _alloc_mem_mb=$(( SLURM_MEM_PER_CPU * _alloc_cpus ))
else
    _alloc_mem_mb=""
fi

if [ -n "${_alloc_mem_mb}" ]; then
    MEM_PER_JOB=$(( _alloc_mem_mb / MAX_WORKERS ))
    echo "auto --mem-per-job=${MEM_PER_JOB} MB  (alloc ${_alloc_mem_mb} MB / ${MAX_WORKERS} workers)"
else
    # Not inside an allocation (e.g. dry test on a login node). Fall back to the
    # node-ratio share so %maxcore is still clamped.
    MEM_PER_JOB=$(( CORES_PER_WORKER * 3900 ))
    echo "WARNING: no SLURM memory in env; using fallback --mem-per-job=${MEM_PER_JOB} MB." >&2
fi

# MAX_ATOMS=0 (or empty) means no size cap. argparse rejects --max-atoms 0, so
# normalize 0 -> empty and drop the flag entirely; any positive value caps size.
[ "${MAX_ATOMS:-0}" = "0" ] && MAX_ATOMS=""

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
    --mem-per-job "${MEM_PER_JOB}" \
    ${MAX_ATOMS:+--max-atoms "${MAX_ATOMS}"} \
    --max-blocks 1 \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks
