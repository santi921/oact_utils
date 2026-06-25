#!/bin/bash
# ===TEMPLATE-DOC-START (this block is stripped from generated lane scripts)
# TEMPLATE -- do not sbatch this file directly. Generate per-lane scripts:
#
#   python oact_utilities/scripts/stratify_lanes.py <db> --cluster <name> \
#       --venv-path '<venv>' --root-dir '<scratch>/jobs_parsl/' \
#       --template launch/run_parsl_drac.template.sh --outdir launch/generated/
#
# ONE template for every cluster. The generator fills:
#   - lane geometry (atom band, cores/worker, workers, ntasks, walltime, batch)
#   - per-cluster paths (DB_PATH, ROOT_DIR, VENV_PATH)
#   - SIMPLE_INPUT: omol for actinide DBs, omol_base for non-actinide (auto from
#     the metal column, or forced with --simple-input)
# Cluster differences are CLI args: --cluster (node size), --venv-path/--root-dir,
# --ntasks (partial node). Edit account/modules/basis here ONCE.
#
# Requires a submit_jobs with --min-atoms/--max-atoms + --mem-per-job
# (branch feat/drac-min-atoms-lanes).
# ===TEMPLATE-DOC-END
# Generated lane '{{CLASS}}/{{LANE}}' -- regenerate with stratify_lanes.py; do not hand-edit.

#SBATCH --account=def-yqw
#SBATCH --time={{TIME}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={{NTASKS}}     # = MAX_WORKERS * CORES_PER_WORKER
#SBATCH --mem-per-cpu=3900M              # node ratio (~3900 MB/core on the 192c/768GB nodes)
#SBATCH --job-name=parsl-{{CLASS}}-{{LANE}}
#SBATCH --array=1-20                     # concurrent allocations; resubmit until the band drains
# No --qos, no --partition: DRAC auto-assigns the partition by time/cores.

set -euo pipefail

# ---- Configuration (DB_PATH/ROOT_DIR/VENV_PATH/SIMPLE_INPUT filled per run) ----
DB_PATH="{{DB_PATH}}"
ROOT_DIR="{{ROOT_DIR}}"     # heavy I/O -> scratch (Lustre)
VENV_PATH="{{VENV_PATH}}"   # per-cluster; built by examples/drac/setup_venv.sh
PYTHON_MODULE="python/3.11"
MODULE_LOAD="StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"   # standard CVMFS chain

# ---- Lane parameters (filled by the generator) ----
MAX_WORKERS={{MAX_WORKERS}}
CORES_PER_WORKER={{CORES_PER_WORKER}}   # == %pal nprocs; MAX_WORKERS*this = full pack
MIN_ATOMS={{MIN_ATOMS}}                 # band: MIN_ATOMS <= natoms <= MAX_ATOMS
MAX_ATOMS={{MAX_ATOMS}}
BATCH_SIZE={{BATCH_SIZE}}               # workers * waves; submit_jobs claims ALL of these RUNNING
                                        # up front (no loop). Overshoot orphans the tail (clear with
                                        # dashboard --recover-orphans), undershoot idles cores.
JOB_TIMEOUT={{JOB_TIMEOUT}}             # Parsl kills stragglers ~30 min before the wall
MAX_FAIL_COUNT=10

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="{{SIMPLE_INPUT}}"         # omol (actinide) / omol_base (non-actinide)
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
    # --purge-on-fail   # RE-ENABLE only once lanes are stable: during tuning it
                       # writes .do_not_rerun.json on OOM victims and deletes evidence.
