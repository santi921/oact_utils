#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 4
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -J seed_infer_v3
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3/logs/seed_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3/logs/seed_%j.err

# Builds a combined seed feature set (OMOL train_1M + v3 act/nonact train) for entropy
# downselect, from fairchem-native ase_db data (*.aselmdb + metadata.npz), via
# run_asedb_inference.py. train_1M (1M structures) is split into 6 strided shards across
# both node submissions; act/train and nonact/train each get a dedicated GPU.
#
# Submit twice:
#   sbatch run_seed_inference_v3.sh 0   # train_1M ranks 0-3 of world-size 6
#   sbatch run_seed_inference_v3.sh 1   # train_1M ranks 4-5 of world-size 6, act, nonact
#
# After both complete, combine chunks into one array:
#   python -m oact_utilities.scripts.entropy_downselect.build_seed_features \
#       /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3 \
#       -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3_combined.npy

set -euo pipefail

BATCH_IDX="${1:?Usage: sbatch run_seed_inference_v3.sh <0|1>}"

OMOL_SRC="/pscratch/sd/i/ishan_a/OMOL/4M/train_1M"
ACT_SRC="/global/u2/i/ishan_a/oact_utils/data/v3_data/act/train"
NONACT_SRC="/global/u2/i/ishan_a/oact_utils/data/v3_data/nonact/train"
OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3"
MODEL_PATH="/pscratch/sd/i/ishan_a/open_actinides/runs/202607-0100-1905-0cc9/checkpoints/final/inference_ckpt.pt"
WORLD_SIZE=6

if [ "$BATCH_IDX" -eq 0 ]; then
    SRCS=("${OMOL_SRC}" "${OMOL_SRC}" "${OMOL_SRC}" "${OMOL_SRC}")
    STEMS=("omol_rank0" "omol_rank1" "omol_rank2" "omol_rank3")
    RANKS=(0 1 2 3)
    WORLD_SIZES=(${WORLD_SIZE} ${WORLD_SIZE} ${WORLD_SIZE} ${WORLD_SIZE})
elif [ "$BATCH_IDX" -eq 1 ]; then
    SRCS=("${OMOL_SRC}" "${OMOL_SRC}" "${ACT_SRC}" "${NONACT_SRC}")
    STEMS=("omol_rank4" "omol_rank5" "act_train" "nonact_train")
    RANKS=(4 5 0 0)
    WORLD_SIZES=(${WORLD_SIZE} ${WORLD_SIZE} 1 1)
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
for i in "${!SRCS[@]}"; do
    STEM="${STEMS[$i]}"
    LOG="${OUTPUT_DIR}/logs/${STEM}.log"

    echo "Starting GPU ${i}: ${STEM} (rank ${RANKS[$i]}/${WORLD_SIZES[$i]})"
    CUDA_VISIBLE_DEVICES=${i} python -m oact_utilities.scripts.entropy_downselect.run_asedb_inference \
        "${SRCS[$i]}" \
        -o "${OUTPUT_DIR}" \
        --stem "${STEM}" \
        --model-path "${MODEL_PATH}" \
        --max-atoms 1024 \
        --num-workers 4 \
        --device cuda \
        --rank "${RANKS[$i]}" \
        --world-size "${WORLD_SIZES[$i]}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    STEM="${STEMS[$i]}"
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
