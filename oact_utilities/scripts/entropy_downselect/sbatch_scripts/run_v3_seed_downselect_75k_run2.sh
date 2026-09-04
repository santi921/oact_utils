#!/bin/bash
#SBATCH -A m5293
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 4
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J v3_downselect_75k_run2
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v3_seed_downselect_75k_run2/slurm_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v3_seed_downselect_75k_run2/slurm_%j.err

# Entropy downselect + in-loop structure optimization: select 75k structures from the
# 17.4M candidate pool (features_output_v3, v3 model features), seeded with the
# combined OMOL train_4M + v3 act/nonact seed covariance (seed_features_v3_combined_4M.npy),
# excluding indices already selected in v2_seed_downselect (500k) and
# v3_seed_downselect_30k_run1 (30k) -- both are already in the dataset.
# Run2 changes vs run1: 30 opt steps (was 15), max_disp 2.0 A (was 1.2), and the
# per-step delta_logdet trajectory is recorded in optimization_report.npz.

set -euo pipefail

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

PS="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect"
OUTPUT_DIR="${PS}/v3_seed_downselect_75k_run2"
SEED="${PS}/seed_features_v3_combined_4M.npy"
MODEL="/pscratch/sd/i/ishan_a/open_actinides/runs/202607-0100-1905-0cc9/checkpoints/final/inference_ckpt.pt"
EXCLUDE_V2="/global/homes/i/ishan_a/oact_utils/data/entropy_downselect/v2_seed_downselect/selected_indices.npy"
EXCLUDE_30K="/global/homes/i/ishan_a/oact_utils/data/entropy_downselect/v3_seed_downselect_30k_run1/selected_indices.npy"
mkdir -p "${OUTPUT_DIR}"

echo "Node: $(hostname)"
nvidia-smi -L || true
date

python -m oact_utilities.scripts.entropy_downselect.entropy_downselect_optimize \
    --features-dir "${PS}/features_output_v3" \
    --seed-features "${SEED}" \
    --lmdb-dir "${PS}/lmdb_inference" \
    --model-path "${MODEL}" \
    --exclude-indices "${EXCLUDE_V2}" "${EXCLUDE_30K}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-select 75000 \
    --batch-size 1000 \
    --pool-factor 5 \
    --checkpoint-every 10000 \
    --regularization 1e-6 \
    --n-gpus 4 \
    --max-atoms 1024 \
    --opt-max-steps 30 \
    --opt-max-disp 2.0 \
    --opt-step-size 0.1 \
    --opt-min-dist 0.7

echo "Done"
date
