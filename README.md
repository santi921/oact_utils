How to run "wave 2" on slurm

1. Download the entire folder here: `https://github.com/santi921/oact_utils/tree/main/data/big_benchmark`
2. Download orca.6.1
3. You're going to use four scripts: `job_writer_wave2.py` , `check_jobs_wave2.py` , `analysis_wave2_sp.py`, and `run_jobs_wave2.py` . In each you want to update variables I highlighted to update for your HPC systems/install. In particular you'll want to check the replicates value. If we're really trying to stress test the queue/systems, play with that.
4. From here you can just successively run `job_writer_wave2.py` into `run_jobs_wave2.py` into `check_jobs_wave2.py`. Check the jobs written by the writer to make sure single jobs work then run_jobs will launch all of them in the root job folder. Check_jobs will give you status.
5. That's pretty much it to get the calcs.
6. Finally, You'll want to run `analysis_wave2_sp.py` to get energy, grad, and timing values, this will give you a set of np files in the save_dir. You can send me those so I can compare vs. tuolumne.

full-runner-parsl --type_runner local --move_results --clean --restart --preprocess_compressed --orca_2mkl_cmd /Users/santiagovargas/Documents/orca_6_0_1_macosx_arm64_openmpi411/orca_2mkl --multiwfn_cmd multiwfn --job_file /Users/santiagovargas/dev/runner_jobs/rmechdb.txt
