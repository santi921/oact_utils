#!/bin/bash
# ORCA module-chain + MPI validation job for Fir (DRAC).
#
# Validates end-to-end, before wiring anything into oact_utilities:
#   1. the orca/6.1.0 module chain loads,
#   2. $EBROOTORCA resolves to the install,
#   3. ORCA spawns its OWN MPI across --ntasks ranks (never srun/mpirun),
#   4. the run reaches "ORCA TERMINATED NORMALLY".
#
# Submit from a dir containing h2o_mpi_test.inp:
#   sbatch --account=def-yqw orca_fir_test.sh
# (or set --account below). Then check h2o_mpi_test.out.

#SBATCH --job-name=orca-fir-validate
#SBATCH --account=def-yqw          # Default RAP (opportunistic, lowest priority)
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # MUST equal '%pal nprocs 4 end' in the .inp
#SBATCH --mem-per-cpu=3000M        # <= ~4000 MB/core ratio on Fir (768000M/192)
#SBATCH --time=0-00:20             # short -> rides backfill, tiny fairshare hit
#SBATCH --output=%x-%j.out
# No --partition, no --qos on DRAC (auto-assigned; no preemptible QOS exists).

set -euo pipefail

echo "host        : $(hostname)"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-?}"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-?}"

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0

echo "EBROOTORCA  : ${EBROOTORCA:-<unset! module did not load>}"
ORCA_BIN="${EBROOTORCA}/orca"
echo "orca binary : $ORCA_BIN"
[ -x "$ORCA_BIN" ] || { echo "ERROR: orca not executable at $ORCA_BIN"; exit 1; }

INP="h2o_mpi_test.inp"
OUT="h2o_mpi_test.out"

# Run inside node-local NVMe scratch, copy results back. This is the production
# I/O pattern (keeps the heavy .tmp churn off Lustre). For this tiny test it
# mostly just exercises the staging mechanics.
WORK="${SLURM_TMPDIR:-.}/orca_validate"
mkdir -p "$WORK"
cp "$INP" "$WORK/"
cd "$WORK"

# Full path, NOT mpirun/srun: ORCA launches MPI itself from %pal nprocs.
"$ORCA_BIN" "$INP" > "$OUT" 2>&1 || true

cd - >/dev/null
cp "$WORK/$OUT" "./$OUT"

echo
echo "== tail of $OUT =="
tail -n 15 "./$OUT"
echo
if grep -q "ORCA TERMINATED NORMALLY" "./$OUT"; then
    echo "PASS: ORCA terminated normally; module chain + MPI spawn verified."
else
    echo "FAIL: did not find 'ORCA TERMINATED NORMALLY'. Inspect $OUT."
    echo "      Common fixes: MPI fabric env on Fir (InfiniBand NDR):"
    echo "        export OMPI_MCA_mtl='^mxm'; export OMPI_MCA_pml='^yalla'"
    exit 1
fi
