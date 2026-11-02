#!/bin/bash
# =============================================================================
# Pre-optimization smoke test on LBNL Lawrencium (LRC): 10 tiny jobs (6 AN66 + 4 water),
# one arm per submission, same ORCA settings as run_parsl_single_node_lrc.sh.
#
#   sbatch run_preopt_smoke_lrc.sh baseline   # plain ORCA
#   sbatch run_preopt_smoke_lrc.sh gxtb       # g-xTB pre-opt, then ORCA
#   sbatch run_preopt_smoke_lrc.sh mlip       # OAct v4 MLIP (CPU) pre-opt, then ORCA
#   DRY_RUN=1 bash run_preopt_smoke_lrc.sh gxtb   # prepare job dirs only (login node ok)
#
# One-time setup on the LRC login node (compute nodes may have no internet):
#   export S=/global/scratch/users/$USER/preopt_smoke; mkdir -p $S/logs
#   scp smoke.db v4_main_inference.pt  <you>@lrc-xfer.lbl.gov:$S/       # from the Mac
#   # g-xTB (Linux build), checksum verified:
#   cd $S && U=https://raw.githubusercontent.com/grimme-lab/g-xtb/main/binaries
#   curl -LO $U/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz
#   curl -LO $U/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz.sha256
#   sha256sum -c xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz.sha256
#   mkdir -p xtb_gxtb && tar -xJf xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz --strip-components=1 -C xtb_gxtb
#   # code: clone/pull the branch, install into the env, add fairchem for the mlip arm
#   git clone -b feat/preopt-gxtb-mlip https://github.com/santi921/oact_utils.git $S/../oact_utils   # or git pull
#   conda activate oact && pip install -e $S/../oact_utils   # mlip arm: also pip install fairchem-core
#
# The SLURM header is yours (account/partition/qos). Two deliberate changes: no --requeue (a requeued
# coordinator would resubmit jobs the DB already marked running) and --time 02:00 (tiny jobs).
# =============================================================================

#SBATCH --account=pc_aditik
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=preopt-smoke
#SBATCH --output=/global/scratch/users/santiagovargas/preopt_smoke/logs/%x_%j.out

set -euo pipefail

ARM="${1:?usage: sbatch run_preopt_smoke_lrc.sh baseline|gxtb|mlip}"
case "${ARM}" in baseline|gxtb|mlip) ;; *) echo "unknown arm ${ARM}" >&2; exit 2;; esac

# ---- Configuration (edit these) ----
SMOKE_DIR="/global/scratch/users/${USER}/preopt_smoke"
REPO_DIR="${REPO_DIR:-/global/scratch/users/${USER}/oact_utils}"
CONDA_ENV="oact"
ORCA_DIR="/global/scratch/users/${USER}/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg"
MODULE_LOAD="gcc/11.4.0 openmpi/4.1.6 miniforge3/25.9.1"
GXTB_PATH="${SMOKE_DIR}/xtb_gxtb/bin/xtb"
MLIP_CKPT="${SMOKE_DIR}/v4_main_inference.pt"

MAX_WORKERS=2               # concurrent ORCA jobs
CORES_PER_WORKER=2          # %pal nprocs per job; 2*2 = 4 = --cpus-per-task
JOB_TIMEOUT=3600            # s per job (ORCA plus pre-opt)
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="omol"
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"
SCF_MAXITER=600
# -----------------------------------

module load ${MODULE_LOAD}
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
unset PYTHONPATH

# Same runtime linking as run_parsl_single_node_lrc.sh (gcc-13 libstdc++ for ORCA, no IB).
export PATH="${ORCA_DIR}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ORCA_DIR}:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=vader,self
export OACT_UTILITIES_REPO_ROOT="${REPO_DIR}"

# ---- preflight: fail here, not 20 minutes into a job ----
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
[ -x "${ORCA_DIR}/orca" ] || fail "ORCA not found at ${ORCA_DIR}/orca"
for need in orca_ndoint orca_propint orca_leanscf_mpi; do
    [ -e "${ORCA_DIR}/${need}" ] || fail "${need} missing from ${ORCA_DIR} (incomplete ORCA install)"
done
[ -f "${SMOKE_DIR}/smoke.db" ] || fail "${SMOKE_DIR}/smoke.db missing (scp it first)"
python - <<PY || fail "python preflight failed (message above)"
import importlib.util, pathlib, sys
import oact_utilities
here = pathlib.Path(oact_utilities.__file__).resolve()
repo = pathlib.Path("${REPO_DIR}").resolve()
if repo not in here.parents:
    sys.exit(f"oact_utilities is imported from {here}, expected under {repo}: pip install -e {repo}")
if importlib.util.find_spec("oact_utilities.core.preopt") is None:
    sys.exit(
        f"the checkout at {repo} has no pre-opt code: run "
        f"git -C {repo} fetch origin && git -C {repo} switch feat/preopt-gxtb-mlip && git -C {repo} pull"
    )
PY

PREOPT_ARGS=()
case "${ARM}" in
gxtb)
    [ -x "${GXTB_PATH}" ] || fail "g-xTB binary missing: ${GXTB_PATH} (see setup in this script's header)"
    "${GXTB_PATH}" --help 2>&1 | grep -q -- "--gxtb" || fail "${GXTB_PATH} is not a g-xTB build (no --gxtb)"
    PREOPT_ARGS=(--preopt gxtb --gxtb-path "${GXTB_PATH}")
    ;;
mlip)
    [ -f "${MLIP_CKPT}" ] || fail "checkpoint missing: ${MLIP_CKPT}"
    python -c "import fairchem.core" 2>/dev/null || fail "fairchem-core not installed in env ${CONDA_ENV}"
    PREOPT_ARGS=(--preopt mlip --mlip-checkpoint "${MLIP_CKPT}" --mlip-device cpu)
    ;;
esac

# ---- per-arm DB copy (resumable: never overwritten) ----
DB="${SMOKE_DIR}/smoke_${ARM}.db"
[ -f "${DB}" ] || cp "${SMOKE_DIR}/smoke.db" "${DB}"
ROOT="${SMOKE_DIR}/jobs_${ARM}"
mkdir -p "${ROOT}" "${SMOKE_DIR}/run_${ARM}"
cd "${SMOKE_DIR}/run_${ARM}"   # Parsl runinfo/ lands here

_alloc_mem_mb="${SLURM_MEM_PER_NODE:-$(( CORES_PER_WORKER * MAX_WORKERS * 3900 ))}"
MEM_PER_JOB=$(( _alloc_mem_mb / MAX_WORKERS ))
echo "arm=${ARM} db=${DB} mem-per-job=${MEM_PER_JOB} MB cpus=${SLURM_CPUS_ON_NODE:-unset}"

python -m oact_utilities.workflows.submit_jobs \
    "${DB}" "${ROOT}" \
    --use-parsl --hpc-site sandia --scheduler slurm --max-blocks 1 \
    --orca-path "${ORCA_DIR}/orca" \
    --batch-size 10 \
    --max-workers "${MAX_WORKERS}" --cores-per-worker "${CORES_PER_WORKER}" \
    --n-cores "${CORES_PER_WORKER}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --mem-per-job "${MEM_PER_JOB}" \
    --functional "${FUNCTIONAL}" --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" --ks-method uks \
    --no-parsl-monitoring \
    ${DRY_RUN:+--dry-run} \
    ${PREOPT_ARGS[@]+"${PREOPT_ARGS[@]}"}

python -m oact_utilities.workflows.dashboard "${DB}" --show-metrics || true
echo "done: compare arms with  python compare_arms.py ${SMOKE_DIR}"
