#!/bin/bash
#SBATCH -A m5250
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J pkl2lmdb
#SBATCH -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference/logs/pkl2lmdb_%j.out
#SBATCH -e /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference/logs/pkl2lmdb_%j.err

set -euo pipefail

PICKLE_FILES=(
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/second_sample/sane_full_4_7_2026.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/third_sample/full_sample_2026-04-08_09-02-54.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/fourth_sample/full_sample_2026-04-09_15-59-25.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/fifth_sample/full_sample_2026-04-10_08-41-47.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/six_sample/full_sample_2026-04-20_10-56-01.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/seven_sample/full_sample_2026-04-20_10-11-27.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/eight_sample/full_sample_2026-04-20_15-30-30.pkl"
    "/global/cfs/cdirs/m4292/mgt/open_Ac_gen/full_prod/nine_sample/full_sample_2026-04-22_14-33-09.pkl"
)

OUTPUT_DIR="/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

module load python
conda activate /global/cfs/cdirs/m4292/mgt/conda-env
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"

echo "Node: $(hostname), CPUs: $(nproc)"
date

# All 8 in parallel -- parsing is fast (~1 min), bottleneck is pickle load (~90s)
# Peak RAM: 8 x ~33 GB = ~264 GB (fits in 512 GB node)
PIDS=()
for i in "${!PICKLE_FILES[@]}"; do
    PKL="${PICKLE_FILES[$i]}"
    STEM=$(basename "${PKL}" .pkl)
    OUTPUT="${OUTPUT_DIR}/${STEM}.lmdb"
    LOG="${OUTPUT_DIR}/logs/task_${i}_${STEM}.log"

    echo "Starting task ${i}: ${STEM}"
    python -m oact_utilities.scripts.pkl_to_lmdb \
        "${PKL}" \
        "${OUTPUT}" \
        --chunk-size 50000 \
        --create-metadata \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting for completion..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "Task ${i} completed successfully"
    else
        echo "Task ${i} FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo "All tasks finished. ${FAILED} failures."
date
exit ${FAILED}
