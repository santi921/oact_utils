#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -G 4
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J lmdb_infer
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/features_output_v3/logs/infer_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/features_output_v3/logs/infer_%j.err

set -euo pipefail

BATCH_IDX="${1:?Usage: sbatch run_lmdb_inference_job.sh <0|1>}"
BASE_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference"
OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/features_output_v3"
MODEL_PATH="/pscratch/sd/i/ishan_a/open_actinides/runs/202607-0100-1905-0cc9/checkpoints/final/inference_ckpt.pt"

if [ "$BATCH_IDX" -eq 0 ]; then
    LMDB_FILES=(
        "${BASE_DIR}/sane_full_4_7_2026/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-08_09-02-54/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-09_15-59-25/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-10_08-41-47/data.lmdb"
    )
    TASK_OFFSET=0
elif [ "$BATCH_IDX" -eq 1 ]; then
    LMDB_FILES=(
        "${BASE_DIR}/full_sample_2026-04-20_10-56-01/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-20_10-11-27/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-20_15-30-30/data.lmdb"
        "${BASE_DIR}/full_sample_2026-04-22_14-33-09/data.lmdb"
    )
    TASK_OFFSET=4
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
for i in "${!LMDB_FILES[@]}"; do
    task_idx=$((TASK_OFFSET + i))
    LMDB_FILE="${LMDB_FILES[$i]}"
    STEM=$(basename "$(dirname "${LMDB_FILE}")")
    LOG="${OUTPUT_DIR}/logs/task${task_idx}_${STEM}.log"

    echo "Starting task ${task_idx} GPU ${i}: ${STEM}"
    CUDA_VISIBLE_DEVICES=${i} python -m oact_utilities.scripts.entropy_downselect.run_lmdb_inference \
        "${LMDB_FILE}" \
        -o "${OUTPUT_DIR}" \
        --model-path "${MODEL_PATH}" \
        --max-atoms 8192 \
        --num-workers 4 \
        --device cuda \
        --resume \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    task_idx=$((TASK_OFFSET + i))
    if wait "${PIDS[$i]}"; then
        echo "Task ${task_idx} completed successfully"
    else
        echo "Task ${task_idx} FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo "All done. ${FAILED} failures."
date
exit ${FAILED}
