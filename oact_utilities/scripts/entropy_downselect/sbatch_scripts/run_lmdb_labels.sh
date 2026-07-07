#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 06:00:00
#SBATCH -J lmdb_labels
#SBATCH -o /global/homes/i/ishan_a/oact_utils/data/entropy_downselect/slurm_labels_%j.out
#SBATCH -e /global/homes/i/ishan_a/oact_utils/data/entropy_downselect/slurm_labels_%j.err

# Label the entropy-downselect optimized structures with energy + forces from the v1
# fairchem checkpoint.
#
# 2 nodes x 4 GPUs = 8 tasks. Each srun runs one task per GPU; task R of 8 processes a
# strided 1/8 shard of the dataset in parallel and writes {stem}_labels_rankR.npz. After
# all ranks finish, a single-process --merge concatenates the shards (sorted by structure
# index) into {stem}_labels.npz. The two datasets are labeled sequentially, each using all
# 8 GPUs. Re-submit with RESUME=1 to skip structures already written in the shards.

set -euo pipefail

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="/global/homes/i/ishan_a/oact_utils/data/entropy_downselect"
MODEL="/pscratch/sd/i/ishan_a/open_actinides/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"
MAX_ATOMS=1024
NUM_WORKERS=8
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAG="--resume"; fi

MODULE="oact_utilities.scripts.entropy_downselect.run_lmdb_labels"

DATASETS=(
    "v2_seed_downselect_optimized"
    "v2_seed_downselect_optimized_aggressive"
)

date
nvidia-smi -L || true

for ds in "${DATASETS[@]}"; do
    LMDB="${BASE}/${ds}/optimized_structures.lmdb"
    OUT="${BASE}/${ds}"
    echo "=== Labeling ${ds} ==="

    srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        python -m "${MODULE}" "${LMDB}" -o "${OUT}" \
        --model-path "${MODEL}" --max-atoms "${MAX_ATOMS}" \
        --num-workers "${NUM_WORKERS}" ${RESUME_FLAG}

    echo "=== Merging ${ds} shards ==="
    python -m "${MODULE}" "${LMDB}" -o "${OUT}" --merge --world-size 8

    echo "=== Done ${ds} ==="
    date
done

echo "All done"
date
