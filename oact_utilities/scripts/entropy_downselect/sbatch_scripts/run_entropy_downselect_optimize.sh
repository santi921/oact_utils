#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J entropy_opt
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect_optimized/slurm_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect_optimized/slurm_%j.err

# Entropy downselect with in-loop structure optimization (GPU).
#
# Runs the greedy entropy downselect and, as each structure is selected, perturbs its
# atomic positions through the fairchem model to increase its marginal delta-log-det,
# committing the optimized feature/geometry.
#
# This is GPU-bound (~n_select * opt_max_steps model fwd+bwd). Start with a small POC
# (reduce --n-select and add --limit) before launching a large run. Use --opt-top-n to
# optimize only the most informative selections.

set -euo pipefail

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

PS="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect"
OUTPUT_DIR="${PS}/v2_seed_downselect_optimized"
SEED="/global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb/seed_dataset_filtered_lmdb_features.npy"
MODEL="/pscratch/sd/i/ishan_a/open_actinides/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"
mkdir -p "${OUTPUT_DIR}"

echo "Node: $(hostname)"
nvidia-smi -L || true
date

python -m oact_utilities.scripts.entropy_downselect.entropy_downselect_optimize \
    --features-dir "${PS}/features_output" \
    --seed-features "${SEED}" \
    --lmdb-dir "${PS}/lmdb_inference" \
    --model-path "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-select 500000 \
    --batch-size 1000 \
    --pool-factor 5 \
    --checkpoint-every 10000 \
    --regularization 1e-6 \
    --opt-max-steps 5 \
    --opt-max-disp 0.3 \
    --opt-step-size 0.05 \
    --opt-min-dist 0.7

echo "Done"
date
