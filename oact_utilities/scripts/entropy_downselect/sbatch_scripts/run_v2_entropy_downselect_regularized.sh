#!/bin/bash
#SBATCH -A m5250
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J v2_entropy_reg
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect_regularized/slurm_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect_regularized/slurm_%j.err

set -euo pipefail

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect_regularized"
mkdir -p "${OUTPUT_DIR}"

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
date

python /global/homes/i/ishan_a/oact_utils/oact_utilities/scripts/entropy_downselect/entropy_downselect.py \
    --features-dir /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/features_output \
    --seed-features /global/homes/i/ishan_a/oact_utils/data/v2_data/seed_dataset_filtered_lmdb/seed_dataset_filtered_lmdb_features.npy \
    --output-dir "${OUTPUT_DIR}" \
    --n-select 500000 \
    --batch-size 1000 \
    --pool-factor 5 \
    --checkpoint-every 10000 \
    --regularization 1e-6 \
    --lmdb-dir /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference

echo "Done"
date
