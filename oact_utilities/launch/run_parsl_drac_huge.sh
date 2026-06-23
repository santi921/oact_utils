#!/bin/bash
# =============================================================================
# DRAC single-node Parsl -- HUGE lane (81 <= natoms <= 100).
# One of four size-stratified lanes (small/medium/big/huge). Claims a disjoint
# band via --min-atoms/--max-atoms, so it runs CONCURRENTLY with the others.
#
# Lane geometry on a Rorqual node (768 GB / 192 cores, --mem-per-cpu=3900M),
# every lane FULLY PACKED (192 cores, no idle): per-job mem = CORES_PER_WORKER*3900.
#   small  <= 40  : 24 x 8  -> ~31 GB/job
#   medium 41-60  : 12 x 16 -> ~62 GB/job
#   big    61-80  : 6 x 32  -> ~125 GB/job
#   huge   81-100 : 4 x 48  -> ~187 GB/job   <- this file
#
# HUGE rationale: measured avg runtime for 81-100 atoms is ~7.1h (and likely
# higher -- the old completions were clipped at a ~13.9h timeout), so it needs
# the 24h wall. 4 workers x 48 cores fully packs the node at ~187 GB/job.
# WARNING: 187 GB/job is EXTRAPOLATED, not measured (no per-job RSS data exists).
# Watch the first batch: `grep -l oom_kill slurm-*.out`. If it OOMs, drop
# MAX_WORKERS to 3 (~244 GB/job; 144 cores used, 48 idle -- the one lane where
# underpacking for memory is justified).
# =============================================================================

#SBATCH --account=def-yqw
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192      # whole node, fully packed: 4 workers * 48 cores
#SBATCH --mem-per-cpu=3900M        # 192 * 3900 = 748,800 MB ~= 731 GB; fits the 768 GB node
#SBATCH --job-name=parsl-drac-huge
#SBATCH --array=1-40               # concurrent allocations; resubmit until this band drains

# No --qos, no --partition: DRAC auto-assigns the partition by time/cores.

set -euo pipefail

# ---- Configuration (edit these) ----
DB_PATH="${HOME}/working_dbs/campaign.db"        # point at your workflow DB
ROOT_DIR="${HOME}/scratch/jobs_scratch"          # heavy I/O -> scratch (Lustre)
VENV_PATH="${HOME}/oact-env"
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"

MAX_WORKERS=4
CORES_PER_WORKER=48         # == %pal nprocs; 4*48 = 192 (full pack -> ~187 GB/job)
MIN_ATOMS=81                # band: 81 <= natoms <= 100
MAX_ATOMS=100
BATCH_SIZE=8                # ~2 waves at ~7-8h over a 24h wall. submit_jobs claims ALL of
                            # these RUNNING up front (no loop), so size it to fill the wall.
                            # Overshoot orphans the tail as RUNNING (clear with dashboard
                            # --recover-orphans); undershoot idles cores.
JOB_TIMEOUT=84600           # 23.5h: Parsl kills stragglers before the 24h wall
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
