#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 4
#SBATCH -N 1
#SBATCH -t 01:45:00
#SBATCH -J seed_infer_v3_4M
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M/logs/seed4m_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M/logs/seed4m_%j.err

# Same as run_seed_inference_v3.sh, but for OMOL train_4M instead of train_1M, split
# across all 8 GPUs (2 node submissions) since act/train and nonact/train features are
# already computed (in seed_features_v3/) and don't need to be redone here.
#
# Submit twice:
#   sbatch run_seed_inference_v3_4M.sh 0   # train_4M ranks 0-3 of world-size 8
#   sbatch run_seed_inference_v3_4M.sh 1   # train_4M ranks 4-7 of world-size 8
#
# After both complete, combine with the existing act/nonact chunks into one array:
#   ln -s /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3/act_train_features.npy \
#       /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M/
#   ln -s /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3/nonact_train_features.npy \
#       /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M/
#   python -m oact_utilities.scripts.entropy_downselect.build_seed_features \
#       /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M \
#       -o /global/u2/i/ishan_a/oact_utils/data/entropy_downselect/seed_features_v3_combined_4M.npy

set -euo pipefail

BATCH_IDX="${1:?Usage: sbatch run_seed_inference_v3_4M.sh <0|1>}"

OMOL_SRC="/pscratch/sd/i/ishan_a/OMOL/4M/train_4M"
OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_4M"
MODEL_PATH="/pscratch/sd/i/ishan_a/open_actinides/runs/202607-0100-1905-0cc9/checkpoints/final/inference_ckpt.pt"
WORLD_SIZE=8

if [ "$BATCH_IDX" -eq 0 ]; then
    RANKS=(0 1 2 3)
elif [ "$BATCH_IDX" -eq 1 ]; then
    RANKS=(4 5 6 7)
else
    echo "Invalid BATCH_IDX: $BATCH_IDX (must be 0 or 1)"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

module load conda
conda activate fairchemV2
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

echo "Node: $(hostname), Batch: ${BATCH_IDX}"
nvidia-smi --query-gpu=index,name --format=csv,noheader
date

PIDS=()
for i in "${!RANKS[@]}"; do
    RANK="${RANKS[$i]}"
    STEM="omol4m_rank${RANK}"
    LOG="${OUTPUT_DIR}/logs/${STEM}.log"

    echo "Starting GPU ${i}: ${STEM} (rank ${RANK}/${WORLD_SIZE})"
    CUDA_VISIBLE_DEVICES=${i} python -m oact_utilities.scripts.entropy_downselect.run_asedb_inference \
        "${OMOL_SRC}" \
        -o "${OUTPUT_DIR}" \
        --stem "${STEM}" \
        --model-path "${MODEL_PATH}" \
        --max-atoms 1024 \
        --num-workers 4 \
        --device cuda \
        --rank "${RANK}" \
        --world-size "${WORLD_SIZE}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    STEM="omol4m_rank${RANKS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "${STEM} completed successfully"
    else
        echo "${STEM} FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo "All done. ${FAILED} failures."
date
exit ${FAILED}
