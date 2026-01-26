#!/bin/bash
#SBATCH --account=ODEFN51701626
#SBATCH --time=168:00:00
#SBATCH --qos=standard
#SBATCH --constraint=standard
#SBATCH --job-name=parsl-coordinator

source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate omol-orca

cd /p/home/ritwik/dev/oact_utils/oact_utilities

python scripts/run_jobs_quacc_wave2.py \
    --root_data_dir /p/home/ritwik/dev/oact_utils/data/big_benchmark/ \
    --calc_root_dir /p/app/projects/nga-frontier/ritwik/temp-orca/big_benchmark \
    --orca_cmd /p/home/ritwik/orca_6_1_0_avx2/orca \
    --nprocs 16 \
    --max_blocks 10 \
    --walltime_hours 160 \
    --conda_env omol-orca \
    --conda_base /p/home/ritwik/miniconda3
