#!/bin/bash
#SBATCH -A m5250
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:30:00
#SBATCH -J lmdb_eval
#SBATCH -o /global/homes/i/ishan_a/oact_utils/data/eval_final_ckpt/slurm_eval_%j.out
#SBATCH -e /global/homes/i/ishan_a/oact_utils/data/eval_final_ckpt/slurm_eval_%j.err

# Evaluate a fairchem checkpoint on the v3 act/nonact val sets: energy (per-atom) +
# force MAE. 1 node x 4 GPUs = 4 tasks; each rank R of 4 evaluates a strided 1/4 shard of
# every val set and writes {name}_evalstats_rankR.npz (running sums). A single-process
# --merge then reduces the shards and prints the MAE table. Both datasets fit easily on a
# single GPU (~4 min total), so this multi-GPU launch is just for speed / reruns.

set -euo pipefail

module load conda
conda activate fairchemV3
export PYTHONPATH="/global/u2/i/ishan_a/oact_utils:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="/global/homes/i/ishan_a/oact_utils/data/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"
OUT="/global/homes/i/ishan_a/oact_utils/data/eval_final_ckpt"
MAX_ATOMS=1024
NUM_WORKERS=8

VAL_DIRS=(
    "/global/homes/i/ishan_a/oact_utils/data/v3_data/act/val"
    "/global/homes/i/ishan_a/oact_utils/data/v3_data/nonact/val"
)

MODULE="oact_utilities.scripts.entropy_downselect.run_lmdb_eval"

mkdir -p "${OUT}"
date
nvidia-smi -L || true

echo "=== Evaluating $(basename "${MODEL}") on ${#VAL_DIRS[@]} val sets ==="
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} \
    python -m "${MODULE}" "${VAL_DIRS[@]}" -o "${OUT}" \
    --model-path "${MODEL}" --max-atoms "${MAX_ATOMS}" --num-workers "${NUM_WORKERS}"

echo "=== Reducing shards and reporting MAE ==="
python -m "${MODULE}" "${VAL_DIRS[@]}" -o "${OUT}" --merge --world-size 4

echo "All done"
date
