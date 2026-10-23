#!/bin/bash
# =============================================================================
# NERSC Perlmutter CPU single-node Parsl -- size-banded lanes on WHOLE NODES.
# Every lane runs on an exclusive node (default QOS regular). Lanes differ only
# in natoms band and worker geometry, and claim DISJOINT bands via
# --min-atoms/--max-atoms, so all of them run CONCURRENTLY on one DB. Edit the
# lane table in lane_config() to retune; nothing else needs to change.
#
# Usage (from the directory where logs/ should live):
#   bash run_parsl_perlmutter_lanes.sh submit                 # sbatch every lane
#   bash run_parsl_perlmutter_lanes.sh submit medium large    # sbatch some lanes
#   TEST_ONLY=1 bash run_parsl_perlmutter_lanes.sh submit     # validate, queue nothing
#   QOS=preempt bash run_parsl_perlmutter_lanes.sh submit small
# Smoke test inside an interactive allocation:
#   salloc -A m3278 -C cpu -q interactive -N 1 --ntasks-per-node=128 \
#          --cpus-per-task=2 -t 0:30:00
#   LANE=small BATCH_SIZE=8 bash run_parsl_perlmutter_lanes.sh
#
# Perlmutter CPU node: 2x EPYC 7763, 128 cores / 256 hyperthreads, ~484 GB
# usable. SLURM counts HYPERTHREADS as CPUs. Request tasks (--ntasks-per-node)
# not a single fat task: ORCA's mpirun takes its slot count from SLURM's task
# count, so --ntasks=1 --cpus-per-task=N fails with "not enough slots".
# =============================================================================

#SBATCH --account=m3278
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128      # whole node: 128 cores
#SBATCH --cpus-per-task=2          # both hyperthreads of each core
#SBATCH --job-name=parsl-pm-lane
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=ALL

set -euo pipefail

# ---- Lane table (edit these) ----
# Band counts (solvent_round1_MTD_reactive_even3_actinides, 8,576 rows at start):
#   <60: 1,931   60-80: 4,726   >=81: 1,919 (max natoms 124)
# small is <60 rather than DRAC's <=40 because this DB has only 75 rows <=40.
# Per-rank memory is ~3.2 GB in every lane (node memory / workers, 85% clamp).
# BATCH_SIZE: submit_jobs claims ALL of these RUNNING up front, so size it to
# ~(walltime / avg job hours) * MAX_WORKERS. ARRAY = concurrent allocations.
lane_config() {
    case "$1" in
        small)   # ~60 GB/job
            MAX_WORKERS=8;  CORES_PER_WORKER=16; MIN_ATOMS=;   MAX_ATOMS=59
            WALLTIME=12:00:00; BATCH_SIZE_DEFAULT=24; ARRAY=1-4 ;;
        medium)  # ~120 GB/job
            MAX_WORKERS=4;  CORES_PER_WORKER=32; MIN_ATOMS=60; MAX_ATOMS=80
            WALLTIME=24:00:00; BATCH_SIZE_DEFAULT=12; ARRAY=1-8 ;;
        large)   # ~160 GB/job; 126 of 128 cores used
            MAX_WORKERS=3;  CORES_PER_WORKER=42; MIN_ATOMS=81; MAX_ATOMS=
            WALLTIME=24:00:00; BATCH_SIZE_DEFAULT=6;  ARRAY=1-4 ;;
        *)
            echo "ERROR: unknown lane '$1' (small|medium|large)" >&2
            exit 1 ;;
    esac
}
ALL_LANES="small medium large"

# ---- Paths + ORCA settings (edit these) ----
BASE="${PSCRATCH}/oact_calcs"
DB_PATH="${BASE}/dbs/solvent_round1_MTD_reactive_even3_actinides.db"
ROOT_DIR="${BASE}/jobs/solvated_reactive_test_v0/jobs_parsl"
ENV_PREFIX="${PSCRATCH}/envs/oact"  # = `echo $CONDA_PREFIX` inside (oact)
ORCA_DIR="${PSCRATCH}/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg"
MAX_FAIL_COUNT=10

FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

# --- submit mode (login node): one sbatch per lane, re-using this file -------
if [ "${1:-}" = "submit" ]; then
    shift
    mkdir -p logs
    for lane in ${*:-${ALL_LANES}}; do
        lane_config "${lane}"
        sbatch ${TEST_ONLY:+--test-only} \
            --qos="${QOS:-regular}" \
            --job-name="parsl-pm-${lane}" \
            --time="${WALLTIME}" \
            --array="${ARRAY}" \
            --export="ALL,LANE=${lane}" \
            "$0"
    done
    exit 0
fi

# --- run mode (inside the allocation) ---------------------------------------
LANE="${LANE:?set LANE=small|medium|large, or run: bash $0 submit}"
lane_config "${LANE}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_DEFAULT}}"
# Parsl kills stragglers 30 min before the wall so their rows reset cleanly.
IFS=: read -r _h _m _s <<< "${WALLTIME}"
JOB_TIMEOUT=$(( 10#${_h} * 3600 + 10#${_m} * 60 + 10#${_s} - 1800 ))
echo "lane=${LANE}: ${MAX_WORKERS} x ${CORES_PER_WORKER} cores, natoms [${MIN_ATOMS:-0}, ${MAX_ATOMS:-inf}], batch ${BATCH_SIZE}, job timeout ${JOB_TIMEOUT}s"

# Activate by prefix, not `conda activate`: that runs the base install's
# python, and $PSCRATCH purges files unaccessed for 8 weeks, which can leave
# the base stdlib broken while the env itself still works. Workers inherit this.
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
# NERSC injects /opt/nersc/pymon; clear it so nothing shadows env packages.
unset PYTHONPATH

# ORCA's shared build links OpenMPI 4.1.x. NERSC's openmpi module is 5.x only,
# so mpirun/libmpi come from the conda env (conda-forge openmpi=4.1). The env's
# lib/ also carries the gcc-13 libstdc++ (GLIBCXX_3.4.32) ORCA needs.
ORCA_PATH="${ORCA_DIR}/orca"
export PATH="${ORCA_DIR}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ORCA_DIR}:${LD_LIBRARY_PATH:-}"

# The Parsl job wrapper forces OMPI_MCA_btl=self,tcp on every ORCA run (all
# sites). Perlmutter nodes have several NICs, so a local rank connects from an
# IP its peer does not advertise, Open MPI drops it ("unexpected IP address" in
# orca.err), and ORCA hangs forever at STARTUP CALCULATIONS. Every rank is on
# this node: pin TCP (BTL + mpirun's out-of-band channel) to loopback. pml=ob1
# keeps libfabric/UCX out. Parsl pins each worker to whole cores; ranks float
# inside that cpuset.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_tcp_if_include=127.0.0.1/8
export OMPI_MCA_oob_tcp_if_include=127.0.0.1/8
export OMPI_MCA_hwloc_base_binding_policy=none
export OMP_NUM_THREADS=1

# --- preflight: fail in seconds, not after the allocation is spent ----------
_ompi_ver=$(mpirun --version 2>/dev/null || true)
if [[ "${_ompi_ver}" != *"(Open MPI) 4.1."* ]]; then
    echo "ERROR: mpirun is not Open MPI 4.1.x: $(command -v mpirun || echo none)" >&2
    echo "       conda install -p ${ENV_PREFIX} -c conda-forge 'openmpi=4.1.*'" >&2
    exit 1
fi
_missing=$(ldd "${ORCA_DIR}/orca_scf_mpi" 2>&1 | grep "not found" || true)
if [ -n "${_missing}" ]; then
    echo "ERROR: unresolved ORCA libraries:" >&2
    echo "${_missing}" >&2
    exit 1
fi
# Several coordinators on different nodes share this SQLite DB. That is only
# safe if the filesystem enforces cluster-wide locks ('flock'); 'localflock'
# locks per node and silently corrupts the DB.
_mnt_opts=$(findmnt -no OPTIONS --target "${DB_PATH}" || true)
if [[ ",${_mnt_opts}," != *",flock,"* ]]; then
    echo "ERROR: ${DB_PATH} is not on a cluster-wide 'flock' mount." >&2
    echo "       Mount options: ${_mnt_opts:-<none>}" >&2
    exit 1
fi

# --- core + memory sizing (SLURM_CPUS_ON_NODE counts hyperthreads) -----------
_cores_needed=$(( MAX_WORKERS * CORES_PER_WORKER ))
_alloc_cores=$(( ${SLURM_CPUS_ON_NODE:-256} / 2 ))
if [ "${_cores_needed}" -gt "${_alloc_cores}" ]; then
    echo "ERROR: MAX_WORKERS*CORES_PER_WORKER=${_cores_needed} > ${_alloc_cores} physical cores." >&2
    exit 1
fi
# No --mem in the header: a node exposes ~1.9 GB per hyperthread (~484 GB, not
# 512), so any explicit --mem near the top is rejected. Whole-node QOS get all
# of it. Use whatever SLURM granted.
if [ "${SLURM_MEM_PER_NODE:-0}" -gt 0 ]; then
    _alloc_mem_mb="${SLURM_MEM_PER_NODE}"
elif [ "${SLURM_MEM_PER_CPU:-0}" -gt 0 ]; then
    _alloc_mem_mb=$(( SLURM_MEM_PER_CPU * ${SLURM_CPUS_ON_NODE:-256} ))
else
    _alloc_mem_mb=$(( _alloc_cores * 3700 ))
    echo "NOTE: SLURM exported no memory size; assuming ${_alloc_mem_mb} MB (3700 MB/core)." >&2
fi
MEM_PER_JOB=$(( _alloc_mem_mb / MAX_WORKERS ))
echo "auto --mem-per-job=${MEM_PER_JOB} MB  (${_alloc_mem_mb} MB / ${MAX_WORKERS} workers)"

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --use-parsl \
    --hpc-site perlmutter \
    --scheduler slurm \
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
