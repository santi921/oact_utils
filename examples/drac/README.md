# DRAC cluster validation kit

Hands-on probes to run on a Digital Research Alliance of Canada cluster (Fir
first) before wiring `oact_utilities` into it. Each is read-only or
self-cleaning. Transfer this directory to the cluster (git pull or scp).

Account: use `--account=def-yqw` (Default RAP, lowest-priority opportunistic
backfill, protects the sponsor's RAC priority).

## 0. Build the Python environment (virtualenv, NOT conda)

DRAC HOME has a ~500K **file-count** quota separate from bytes; a conda install
alone can eat ~half of it (that is the "out of space" you hit). Use a virtualenv
in PROJECT instead. Run on a LOGIN node (needs internet for niche PyPI deps):

```bash
# Fir (direct egress):
bash setup_venv.sh "$HOME/projects/def-yqw/$USER/oact-env" python/3.11
# Rorqual/Narval/Tamia (restricted egress -- pass the proxy module 3rd):
bash setup_venv.sh "$HOME/links/projects/def-yqw/$USER/oact-env" python/3.11 httpproxy
```

Calcul Quebec clusters block outbound internet; the 3rd arg loads `httpproxy`
(after the script's `module --force purge`) so pip can reach PyPI and git can
reach GitHub. Omit it where egress is direct.

It loads modules first, builds the venv, installs the scientific core from the
wheelhouse (`--no-index`) and quacc/sella/parsl/etc from PyPI (pinned so they
cannot yank the optimized numpy), installs `oact_utilities` editable, verifies
imports, and prints `diskusage_report` so you can confirm you are under the
inode limit. Activate it in jobs with `source <venv>/bin/activate` AFTER the
module loads. On Trillium, build in HOME (project/home are read-only in jobs).

It also installs `qtaim_generator` (git) + pymatgen, which powers the QTAIM
metrics in `analysis.parse_generator_data` (the `generator_data` DB column).
Only the ORCA-output text parser is used, so no Multiwfn/rdkit/lmdb are needed;
pymatgen is required only because `qtaim_gen`'s package init imports it.

## 1. Storage probe -- where can the DB and job dirs live?

```bash
# login node: shared Lustre mounts (home, scratch, project, nearline)
bash probe_storage.sh

# add a path to also list its largest immediate subdirs
bash probe_storage.sh "$HOME/project/def-yqw"
```

Watch the **inode / file-count** columns in `diskusage_report`, not just bytes:
that is what a conda env exhausts on the 500K-file HOME limit. Re-run inside a
job (below) to also capture `$SLURM_TMPDIR`.

## 2. SQLite concurrency test -- is the mount safe for the workflow DB?

Mirrors `architector_workflow.py` exactly (DELETE journal, `BEGIN IMMEDIATE`,
backoff+jitter). Run it on each candidate mount; compare commits + throughput.

```bash
# login node, Lustre mounts:
python sqlite_lock_test.py --db-dir "$HOME/scratch"          --workers 16
python sqlite_lock_test.py --db-dir "$HOME/project/def-yqw"  --workers 16
```

Pure stdlib -- no env needed. DELETE committing 100% with low retries = safe
here. WAL reported "unusable" on Lustre is expected and confirms DELETE-only.

## 3. ORCA module-chain + MPI test

Confirms `orca/6.1.0` loads, `$EBROOTORCA` resolves, and ORCA spawns its own
MPI across 4 ranks (never srun/mpirun), reaching "ORCA TERMINATED NORMALLY".

```bash
sbatch --account=def-yqw orca_fir_test.sh   # runs h2o_mpi_test.inp
# then:
cat orca-fir-validate-*.out
```

## 4. In-job probe (captures node-local NVMe)

`$SLURM_TMPDIR` only exists during a job. Grab an interactive node and re-run
the storage + DB probes there to measure the NVMe scratch:

```bash
salloc --account=def-yqw --nodes=1 --ntasks=8 --mem-per-cpu=3000M --time=0:30:00
# once on the node:
bash probe_storage.sh
python sqlite_lock_test.py --db-dir "$SLURM_TMPDIR" --workers 16
exit
```

## What each result decides

| Probe | Decides |
|-------|---------|
| storage probe | which mount hosts the venv (file-count headroom) and final results |
| sqlite test on Lustre | whether multi-node Parsl can share one DB on scratch/project |
| sqlite test on `$SLURM_TMPDIR` | the single-node "live DB on NVMe, copy back at end" pattern |
| orca test | the exact module chain for the production job script + Parsl `worker_init` |
