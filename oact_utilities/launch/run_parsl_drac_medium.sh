#!/bin/bash
# =============================================================================
# DRAC single-node Parsl -- MEDIUM lane (41 <= natoms <= 60).
# One of four size-stratified lanes (small/medium/big/huge). Claims a disjoint
# band via --min-atoms/--max-atoms, so it runs CONCURRENTLY with the others.
#
# Lane geometry on a Rorqual node (768 GB / 192 cores, --mem-per-cpu=3900M),
# every lane FULLY PACKED (192 cores, no idle): per-job mem = CORES_PER_WORKER*3900.
#   small  <= 40  : 24 x 8  -> ~31 GB/job
#   medium 41-60  : 12 x 16 -> ~62 GB/job   <- this file
#   big    61-80  : 6 x 32  -> ~125 GB/job
#   huge   81-100 : 4 x 48  -> ~187 GB/job
#
# MEDIUM rationale: measured avg runtime for 41-60 atoms is ~3.6h; 12 workers at
# 62 GB/job is 2x the budget that OOM-killed a 55-atom job at 24 workers. WATCH
# the 56-60 atom top end -- if it shows oom_kill, drop MAX_WORKERS to 10 (~75 GB)
# or move the 56-60 slice up to the big lane. 12h wall keeps it in the broad
# backfill tier (avg 3.6h fits comfortably; the >12h SCF tail requeues).
# =============================================================================

#SBATCH --account=def-yqw
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192      # whole node, fully packed: 12 workers * 16 cores
#SBATCH --mem-per-cpu=3900M        # 192 * 3900 = 748,800 MB ~= 731 GB; fits the 768 GB node
#SBATCH --job-name=parsl-drac-medium
#SBATCH --array=1-40               # concurrent allocations; resubmit until this band drains

# No --qos, no --partition: DRAC auto-assigns the partition by time/cores.

set -euo pipefail

# ---- Configuration (edit these) ----
DB_PATH="${HOME}/working_dbs/campaign.db"        # point at your workflow DB
ROOT_DIR="${HOME}/scratch/jobs_scratch"          # heavy I/O -> scratch (Lustre)
VENV_PATH="${HOME}/oact-env"
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"

MAX_WORKERS=12
CORES_PER_WORKER=16         # == %pal nprocs; 12*16 = 192 (full pack -> ~62 GB/job)
MIN_ATOMS=41                # band: 41 <= natoms <= 60
MAX_ATOMS=60
BATCH_SIZE=36               # ~3 waves at ~3.6h over a 12h wall. submit_jobs claims ALL of
                            # these RUNNING up front (no loop), so size it to fill the wall.
                            # Overshoot orphans the tail as RUNNING (clear with dashboard
                            # --recover-orphans); undershoot idles cores.
JOB_TIMEOUT=42000           # 11h40m: Parsl kills stragglers before the 12h wall
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
unset PYTHONPATH

export OMPI_MCA_hwloc_base_binding_policy=none
export OMP_NUM_THREADS=1

# Resolve the module's ORCA binary now ($EBROOTORCA expands in this shell).
ORCA_PATH="${EBROOTORCA}/orca"

# --- memory + core sizing (auto-derived from the SLURM allocation) -----------
_cores_needed=$(( MAX_WORKERS * CORES_PER_WORKER ))
_alloc_cpus="${SLURM_CPUS_ON_NODE:-}"
if [ -n "${_alloc_cpus}" ] && [ "${_cores_needed}" -gt "${_alloc_cpus}" ]; then
    echo "ERROR: MAX_WORKERS*CORES_PER_WORKER=${_cores_needed} > allocated cores ${_alloc_cpus}." >&2
    echo "       Lower MAX_WORKERS/CORES_PER_WORKER or raise #SBATCH --ntasks-per-node." >&2
    exit 1
fi
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
    MEM_PER_JOB=$(( 748800 / MAX_WORKERS ))
    echo "WARNING: no SLURM memory in env; using fallback --mem-per-job=${MEM_PER_JOB} MB." >&2
fi

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
    ${MIN_ATOMS:+--min-atoms "${MIN_ATOMS}"} \
    ${MAX_ATOMS:+--max-atoms "${MAX_ATOMS}"} \
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
