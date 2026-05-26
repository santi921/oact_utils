#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J seed_infer
#SBATCH -o /global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb/infer_%j.out
#SBATCH -e /global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb/infer_%j.err

set -euo pipefail

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

LMDB="/global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb/data.lmdb"
OUTPUT="/global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb"
MODEL="/pscratch/sd/i/ishan_a/open_actinides/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
date

python /global/homes/i/ishan_a/oact_utils/oact_utilities/scripts/entropy_downselect/run_lmdb_inference.py \
    "${LMDB}" \
    -o "${OUTPUT}" \
    --model-path "${MODEL}" \
    --max-atoms 8192 \
    --num-workers 4 \
    --device cuda \
    --resume

echo "Done"
date
