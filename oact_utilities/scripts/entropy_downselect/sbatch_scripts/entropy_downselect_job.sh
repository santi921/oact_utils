#!/bin/bash
#SBATCH -A m5250
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -J entropy_sel
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/selection_output/logs/entropy_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/selection_output/logs/entropy_%j.err

set -euo pipefail

FEATURES_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/features_output"
OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/selection_output"

mkdir -p "${OUTPUT_DIR}/logs"

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128

echo "Node: $(hostname), CPUs: $(nproc)"
date

python -m oact_utilities.scripts.entropy_downselect \
    --features-dir "${FEATURES_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-select 250000 \
    --seed-size 128 \
    --random-seed 42 \
    --regularization 1e-6 \
    --batch-size 1000 \
    --pool-factor 5 \
    --checkpoint-every 10000 \
    --resume

echo "Done."
date
