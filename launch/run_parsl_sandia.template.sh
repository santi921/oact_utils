#!/bin/bash
# ===TEMPLATE-DOC-START (this block is stripped from generated lane scripts)
# TEMPLATE -- do not sbatch this file directly. Generate per-lane scripts:
#
#   python oact_utilities/scripts/stratify_lanes.py <db> --cluster sandia \
#       --root-dir 'jobs_rnfs/' \
#       --template launch/run_parsl_sandia.template.sh --outdir launch/generated/
#
# Sandia (Attaway): conda env, Omni-Path MPI, longpri 48h, 36-core whole node.
# Two lanes (--cluster sandia scheme): fast 6 workers x 6c, slow 3 workers x 12c
# -- both pack the full 36-core node. SIMPLE_INPUT auto: omol (actinide) /
# omol_base (non-actinide). The generator fills the {{...}} tokens; edit the
# account/qos/conda/MPI bits here ONCE.
# ===TEMPLATE-DOC-END
# Generated lane '{{CLASS}}/{{LANE}}' -- regenerate with stratify_lanes.py; do not hand-edit.

#SBATCH --job-name=oact-{{CLASS}}-{{LANE}}
#SBATCH --nodes=1
#SBATCH --qos=longpri
#SBATCH --partition=batch
#SBATCH --account=fy250086P
#SBATCH --time={{TIME}}
#SBATCH --output=logs/orca_{{LANE}}_%A_%a.out
#SBATCH --error=logs/orca_{{LANE}}_%A_%a.err
#SBATCH --array=1-100                    # independent allocations; resubmit until the band drains

set -euo pipefail

# ---- conda env (Sandia uses conda, not a venv) ----
source /projects/netpub/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate oact

# ---- MPI / ORCA (Omni-Path; match ORCA's shared OpenMPI build) ----
module load aue/openmpi/4.1.6-gcc-12.3.0
export MPI_ROOT="$(dirname "$(dirname "$(which mpirun)")")"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_pml=ob1
export OMPI_MCA_mtl=^psm2
export OMPI_MCA_btl=tcp,self,vader
ORCA_PATH="/home/svargas/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"

# ---- Configuration ----
DB_PATH="{{DB_PATH}}"
ROOT_DIR="{{ROOT_DIR}}"     # heavy I/O -> working FS

# ---- Lane parameters (filled by the generator) ----
N_CORES={{NTASKS}}                       # whole Sandia node = 36c (MAX_WORKERS*CORES_PER_WORKER)
MAX_WORKERS={{MAX_WORKERS}}
CORES_PER_WORKER={{CORES_PER_WORKER}}    # == %pal nprocs
MIN_ATOMS={{MIN_ATOMS}}                  # band: MIN_ATOMS <= natoms <= MAX_ATOMS
MAX_ATOMS={{MAX_ATOMS}}
BATCH_SIZE={{BATCH_SIZE}}                # workers * waves; submit_jobs claims ALL of these RUNNING
                                         # up front -- overshoot orphans the tail (reclaim with
                                         # dashboard --recover-orphans), undershoot idles cores.
N_HOURS={{N_HOURS}}                      # Parsl block walltime (h); matches SBATCH --time
JOB_TIMEOUT={{JOB_TIMEOUT}}              # per-ORCA-job cap (s); ~30 min under the wall
MAX_FAIL_COUNT=20

# ORCA settings
SCF_MAXITER=600
FUNCTIONAL="wB97M-V"
SIMPLE_INPUT="{{SIMPLE_INPUT}}"          # omol (actinide) / omol_base (non-actinide)
ACTINIDE_BASIS="ma-def-TZVP"
ACTINIDE_ECP="def-ECP"
NON_ACTINIDE_BASIS="def2-TZVPD"

cd "${SLURM_SUBMIT_DIR}"

python -m oact_utilities.workflows.submit_jobs \
    "${DB_PATH}" \
    "${ROOT_DIR}" \
    --scheduler slurm \
    --hpc-site sandia \
    --use-parsl \
    --max-blocks 1 \
    --conda-env oact \
    --orca-path "${ORCA_PATH}" \
    --n-cores "${N_CORES}" \
    --n-hours "${N_HOURS}" \
    --max-workers "${MAX_WORKERS}" \
    --cores-per-worker "${CORES_PER_WORKER}" \
    --batch-size "${BATCH_SIZE}" \
    --job-timeout "${JOB_TIMEOUT}" \
    --max-fail-count "${MAX_FAIL_COUNT}" \
    ${MIN_ATOMS:+--min-atoms "${MIN_ATOMS}"} \
    ${MAX_ATOMS:+--max-atoms "${MAX_ATOMS}"} \
    --functional "${FUNCTIONAL}" \
    --simple-input "${SIMPLE_INPUT}" \
    --actinide-basis "${ACTINIDE_BASIS}" \
    --actinide-ecp "${ACTINIDE_ECP}" \
    --non-actinide-basis "${NON_ACTINIDE_BASIS}" \
    --scf-maxiter "${SCF_MAXITER}" \
    --ks-method uks \
    --clean-on-complete
