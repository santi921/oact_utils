# Pre-optimization (g-xTB or MLIP) before ORCA

An optional stage that relaxes each geometry with g-xTB or a fairchem MLIP on the compute worker, then hands the
relaxed structure to the ORCA step the workflow already runs (single point, ORCA native opt, or Sella). Default is
off; nothing changes unless `--preopt` is passed.

## Quick start

```bash
# g-xTB
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --preopt gxtb --gxtb-path /path/to/xtb \
    --orca-path /path/to/orca

# MLIP (OAct checkpoint file, or a fairchem registry name)
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --preopt mlip --mlip-checkpoint /path/to/v4_main_inference.pt \
    --orca-path /path/to/orca
```

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--preopt {gxtb,mlip}` | off | Pre-optimizer |
| `--preopt-fmax EV_ANG` | 0.05 | Largest per-atom force at which the pre-opt stops (eV/Angstrom) |
| `--preopt-max-steps N` | 200 | xtb ANCopt cycles or LBFGS steps |
| `--gxtb-path PATH` | `xtb` on PATH | g-xTB build of xtb (only with `gxtb`) |
| `--mlip-checkpoint X` | none | OAct `.pt` path or registry name; required with `mlip` |
| `--mlip-device {cpu,cuda}` | cpu | Device for the MLIP |
| `--preopt-on-fail {fail,continue}` | fail | `fail` stops the job before ORCA; `continue` runs ORCA on the raw geometry and records the error |
| `--skip-unsupported` | off | Skip rows with elements the pre-optimizer cannot handle instead of aborting the submit |

The same keys exist in `OrcaConfig` (`preopt`, `preopt_fmax`, `preopt_max_steps`, `gxtb_path`, `mlip_checkpoint`,
`mlip_device`, `preopt_on_fail`) for programmatic use. The tool paths are parameters; nothing is hardcoded.

## How it runs

The pre-opt runs at execution time on the worker, never at prep time on the login node.

1. `prepare_job_directory()` writes the usual `orca.inp` (and `run_sella.py` for Sella), plus `input.xyz`,
   `preopt_config.json` and the static shim `run_preopt.py`. No value is interpolated into code.
2. The job script (Flux, SLURM, Sandia, DRAC) or the Parsl wrapper runs `python run_preopt.py` first.
3. The shim relaxes `input.xyz`, writes `preopt.xyz` and `preopt_status.json`, and rewrites the `*xyz` block of
   `orca.inp`. The unchanged ORCA or Sella step then starts from the relaxed geometry.
4. Exit code is 0 on success, and on failure only with `--preopt-on-fail continue`. Otherwise the job script stops
   before ORCA (`|| exit 1`) and the Parsl wrapper returns a failed result with the recorded reason.

Files left in the job directory:

| File | Keep? | Content |
| ---- | ----- | ------- |
| `input.xyz` | yes | Raw geometry the pre-opt started from |
| `preopt.xyz` | yes | Relaxed geometry (absent when the pre-opt failed) |
| `preopt_status.json` | yes | Method, converged, final fmax (eV/Angstrom), steps, energy (eV), RMSD from input (Angstrom), seconds, error, provenance |
| `preopt_config.json`, `run_preopt.py`, `preopt_driver.log` | yes | Shim, its parameters, its log |
| `preopt_gxtb/` | no | xtb scratch (`xtbrestart`, `xtbopt.log`, per-pass dirs); removed by `clean.py --clean-tmp` |

## Status and database

- `check_job_termination()` reads `preopt_status.json`: a recorded error returns failed (-1), because a job that dies
  in the pre-opt never writes `orca.out`. A pre-opt that failed under `continue` is not a job failure.
- Six columns are written for completed jobs in both Parsl mode and `dashboard --update --extract-metrics`
  (through `_METRIC_COLUMNS`): `preopt_method` (TEXT), `preopt_converged` (INTEGER 0/1), `preopt_fmax` (eV/Angstrom),
  `preopt_energy_ev` (eV), `preopt_rmsd` (Angstrom), `preopt_seconds` (s). Older databases gain them on open.
- The pre-opt energy is from g-xTB or the MLIP, not DFT. Never compare it with ORCA energies.

## Element coverage

An MLIP does not raise on an element it was never trained on: any Z below its embedding size returns a number.
Coverage is therefore checked explicitly from the per-element energy references the model was fitted with.

| Pre-optimizer | Supported elements | Source |
| ------------- | ------------------ | ------ |
| g-xTB | Z = 1-103 (GFN2-xTB and GFN-FF stop at Rn and cannot do actinides) | fixed range |
| OAct `.pt` checkpoint | Z = 1-103 except Rn (Z = 86) | energy task `element_references`, tolerance 1e-6 Eh (Rn is fit noise of ~1e-10 in v3/v4, exactly 0 in v1) |
| Registry model (`uma-s-1p1`, `esen-sm-conserving-all-omol`, ...) | Z = 1-83, no actinides | `predict_unit.atom_refs["omol"]` keys |

A nonzero reference is necessary, not sufficient: it does not prove the element is well represented in training data.

Checks happen twice. At submit time, `filter_jobs_for_submission()` compares the DB `elements` column (no geometry
load) against the pre-optimizer's coverage, prints a per-element count, and either aborts before any job is prepared
or, with `--skip-unsupported`, leaves the rows `to_run`. Rows are never marked failed and `fail_count` is untouched.
On the worker the check runs again before the model loads, in case the DB is stale.

## g-xTB details

- Command: `xtb in.xyz --gxtb --chrg Q [--uhf 2S] --iterations 1000 --opt <level> --cycles N`. Singlets run restricted
  (any `--uhf`, even 0, forces UHF).
- The optimizer level is the loosest xtb level whose gradient-norm criterion is within 5% of `--preopt-fmax`. ANCopt
  converges on the norm of the full gradient, which bounds the largest per-atom force. `0.05` selects `normal`,
  `0.01` selects `vtight`.
- The result is verified with a gradient single point restarted from the optimizer's `xtbrestart`; `preopt_fmax` is the
  measured value, and `converged` requires it to be at or below the requested fmax. A fresh SCF on an open-shell
  actinide can land on a different state, which is why the check restarts from the optimizer's density.
- If the first SCF fails, the guess is seeded from an SCF at 3000 K and the optimization reruns at the default
  temperature (noted in `preopt_status.json`).
- Provenance (xtb version line) is stored in `detail`.

## Installing g-xTB

g-xTB is a separate build of xtb: <https://github.com/grimme-lab/g-xtb>, directory `binaries/`
(`xtb-6.7.1-gxtb-<date>-{linux-x86_64,macos-arm64,macos-x86_64,windows-x86_64}`, each with a `.sha256` file).

```bash
# 1. download the archive and its .sha256 for your platform, then verify
shasum -a 256 xtb-6.7.1-gxtb-*-macos-arm64.tar.gz     # compare with the .sha256 file
# 2. extract to a stable directory (not $PSCRATCH, which purges files unaccessed for 8 weeks;
#    on NERSC use /global/common/software/<project>)
mkdir -p ~/software/xtb_gxtb && tar -xzf xtb-6.7.1-gxtb-*-macos-arm64.tar.gz --strip-components=1 -C ~/software/xtb_gxtb
# 3. macOS only: clear the download quarantine flag or Gatekeeper kills the binary
xattr -dr com.apple.quarantine ~/software/xtb_gxtb
# 4. check that it is the g-xTB build
~/software/xtb_gxtb/bin/xtb --help | grep -- --gxtb
export GXTB_PATH=~/software/xtb_gxtb/bin/xtb
```

Pass the same path to `submit_jobs` as `--gxtb-path`. On HPC nodes set `ulimit -s unlimited` and `OMP_STACKSIZE=4G` if
you run xtb by hand; the driver sets both for its own runs.

## Tests

Tests are stratified so CI needs no binary, model or GPU, and a developer machine can run everything natively.

| Layer | Marker | Needs | Runs in CI |
| ----- | ------ | ----- | ---------- |
| Mock | none | nothing (fake `xtb` script in `tests/conftest.py`; ASE EMT standing in for the MLIP) | yes |
| g-xTB native | `gxtb` | a g-xTB `xtb` from `GXTB_PATH` or `PATH` | skipped |
| MLIP native | `mlip` | `examples/benchmarks/v4_main_inference.pt` (or `OACT_TEST_CKPT`) and a registry model (`OACT_TEST_REGISTRY_MODEL`, default `esen-sm-conserving-all-omol`; needs a Hugging Face login for gated models) | skipped |
| End to end | `gxtb` | g-xTB plus a local ORCA (`ORCA_PATH` or `orca` on PATH); runs the shim, then ORCA PM3 | skipped |

```bash
pytest tests/test_preopt_*.py                       # mock layer, plus native tests that can run
GXTB_PATH=~/software/xtb_gxtb/bin/xtb ORCA_PATH=/path/to/orca pytest tests/test_preopt_*.py -rs
pytest tests/test_preopt_*.py -m "not gxtb and not mlip"   # mock layer only
```

`-rs` prints the skip reason of every native test. The Parsl wrapper is tested through its underlying function
(`orca_job_wrapper.func`) with a fake ORCA script and the real shim, so no Parsl runtime is needed.

Local ORCA caveats seen on a macOS install: the PM3 end-to-end test needs `orca_ndoint`, and templates that print
dipoles need `orca_propint`; an install missing either fails with "No such file or directory". On macOS, clear the
quarantine flag from the whole ORCA directory (`xattr -dr com.apple.quarantine <dir>`) or its binaries are killed.

## Adding a pre-optimizer to an existing workflow

1. Install g-xTB (above) or choose an MLIP checkpoint. For an MLIP, check coverage first: a registry model will reject
   every actinide row, an OAct checkpoint covers Z = 1-103 except Rn.
2. Dry run to see the element filter and the generated files: add `--dry-run`. Inspect one job directory
   (`run_preopt.py`, `preopt_config.json`, `input.xyz`).
3. Submit with `--preopt ...`. Add `--skip-unsupported` when the DB mixes covered and uncovered elements.
4. Watch `preopt_status.json` and the `preopt_*` columns. A high `preopt_rmsd` or `preopt_converged = 0` points at a
   structure the pre-optimizer handled poorly.
5. On a cluster, use a path that survives scratch purges for the binary or checkpoint, and pass it with
   `--gxtb-path` / `--mlip-checkpoint`.
