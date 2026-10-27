#!/bin/bash
# =============================================================================
# NERSC Perlmutter CPU single-node Parsl -- PREEMPT lane (natoms <= 60).
# One of three size-stratified lanes, one per QOS. Each claims a DISJOINT natoms
# band via --min-atoms/--max-atoms, so all three run CONCURRENTLY on one DB.
#
#   preempt  <= 60  : whole node, 8 x 16 cores  -> ~60 GB/job   <- this file
#   shared   61-72  : half node,  2 x 32 cores  -> ~120 GB/job
#   regular  >= 73  : whole node, 4 x 32 cores  -> ~120 GB/job
#
# Band counts (solvent_round1_MTD_reactive_even3_actinides, 8,576 rows):
#   <=60: 2,163   61-72: 1,757   >=73: 4,656
#
# PREEMPT rationale: charged 0.5 after the first 2 h (minimum 2 h charge) but
# killable after 2 h. Smallest molecules finish fastest, so a preemption loses
# the least work here. On SIGTERM the coordinator resets its in-flight jobs to
# TO_RUN, and --requeue puts the allocation back in the queue to claim more.
#
# Perlmutter CPU node: 2x EPYC 7763, 128 cores / 256 hyperthreads, 512 GB.
# SLURM counts HYPERTHREADS as CPUs. Request tasks (--ntasks-per-node) not a
# single fat task: ORCA's mpirun takes its slot count from SLURM's task count,
# so --ntasks=1 --cpus-per-task=N fails with "not enough slots".
#
# Submit from ${PSCRATCH}/oact_calcs (the log paths are relative):
#   mkdir -p logs && sbatch <oact_utils checkout>/oact_utilities/launch/run_parsl_perlmutter_preempt.sh
# Smoke test inside an interactive allocation:
#   salloc -A m3278 -C cpu -q interactive -N 1 --ntasks-per-node=128 \
#          --cpus-per-task=2 -t 1:00:00
#   BATCH_SIZE=8 bash run_parsl_perlmutter_preempt.sh
# =============================================================================

#SBATCH --account=m3278
#SBATCH --constraint=cpu
#SBATCH --qos=preempt
#SBATCH --time=12:00:00            # preempt minimum is 2h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128      # whole node: 128 cores
#SBATCH --cpus-per-task=2          # both hyperthreads of each core
#SBATCH --requeue                  # preempted -> back in queue, claims new jobs
#SBATCH --open-mode=append         # keep logs across requeues
#SBATCH --job-name=parsl-pm-preempt
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --array=1-4                # concurrent allocations; raise once the smoke test passes

set -euo pipefail

# ---- Configuration (edit these) ----
BASE="${PSCRATCH}/oact_calcs"
DB_PATH="${BASE}/dbs/solvent_round1_MTD_reactive_even3_actinides.db"
ROOT_DIR="${BASE}/jobs/solvated_reactive_test_v0/jobs_parsl"
# $PSCRATCH purges files unaccessed for 8 weeks; prefer installs under the
# purge-exempt /global/common/software/<project>. Both are env-overridable.
ENV_PREFIX="${ENV_PREFIX:-${PSCRATCH}/envs/oact}"  # = `echo $CONDA_PREFIX` inside (oact)
ORCA_DIR="${ORCA_DIR:-${PSCRATCH}/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg}"

MAX_WORKERS=8
CORES_PER_WORKER=16         # physical cores == %pal nprocs; 8*16 = 128 (full node)
MIN_ATOMS=                  # no floor
MAX_ATOMS=60                # band: natoms <= 60
BATCH_SIZE="${BATCH_SIZE:-24}"  # ~3 waves over 12h. submit_jobs claims ALL of these
                            # RUNNING up front, so size it to fill the wall.
JOB_TIMEOUT=42000           # per-job cap from each job's own start, not an
                            # allocation deadline; jobs running at the wall are
                            # reset to TO_RUN by submit_jobs' SIGTERM handler
MAX_FAIL_COUNT=10

# ORCA settings
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

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
# A scratch purge leaves partial installs, so check the pieces actually used.
if ! python -c "import oact_utilities, parsl" 2>/dev/null; then
    echo "ERROR: ${ENV_PREFIX}/bin/python cannot import oact_utilities + parsl (purged env?)" >&2
    exit 1
fi
if [ ! -x "${ORCA_PATH}" ] || [ ! -x "${ORCA_DIR}/orca_leanscf_mpi" ]; then
    echo "ERROR: ORCA binaries missing under ${ORCA_DIR} (purged?)" >&2
    exit 1
fi
_ompi_ver=$(mpirun --version 2>/dev/null || true)
if [[ "${_ompi_ver}" != *"(Open MPI) 4.1."* ]]; then
    echo "ERROR: mpirun is not Open MPI 4.1.x: $(command -v mpirun || echo none)" >&2
    echo "       conda install -p ${ENV_PREFIX} -c conda-forge 'openmpi=4.1.*'" >&2
    exit 1
fi
_missing=$(ldd "${ORCA_DIR}/orca_leanscf_mpi" 2>&1 | grep "not found" || true)
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
# of it; a shared job gets its CPU share. Use whatever SLURM granted.
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
