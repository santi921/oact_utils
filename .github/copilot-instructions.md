# Copilot instructions for oact_utils

Summary

- Short goal: help AI agents be productive quickly in this repository.
- Focus: ORCA-focused workflows, HPC job generation (Flux & SLURM), quacc/ASE integration, and test/dev conventions.

Quick actions

- Install dev env: `pip install -e .` and `pip install -e "[dev]"`.
- Run linters/formatters: `pre-commit install` then `pre-commit run --all-files` (or run `black .`, `ruff check . --fix`, `mypy oact_utilities/`).
- Run tests: `pytest tests/` (ORCA is not available in CI—tests mock ORCA where needed).
- Run tests before committing: always run `pytest tests/` and ensure all tests pass locally before creating a PR or committing changes. It's recommended to run `pre-commit run --all-files` (which can be configured to include `pytest`) as part of your commit workflow.

Where to look first (high value files)

- `CLAUDE.md` — project overview and workflows (good starting point).
- `README.md` — practical Wave 2 run instructions and example commands.
- `oact_utilities/core/orca/recipes.py` — add new ORCA calculation types here; follow existing recipe patterns like `single_point_calculation`.
- `oact_utilities/core/orca/_base.py` and `calc.py` — calculator setup and helper utilities.
- `oact_utilities/utils/hpc.py` — Flux/SLURM job script generation patterns.
- `oact_utilities/utils/jobs.py` — job launching utilities (how jobs are submitted and monitored).
- `oact_utilities/utils/analysis.py` — parsing ORCA outputs, extracting energies/timings.
- `oact_utilities/launch/run_parsl_coordinator.sh` and `scripts/run_jobs_quacc_wave2.py` — examples of Slurm/Parsl job orchestration.
- `scripts/` — runnable pipeline examples (wave_one, wave_two, multi_spin). Use these as templates but be careful: they often contain hardcoded paths.

Project-specific conventions & patterns

- Type hints and docstrings are required for public functions; use Google or NumPy style.
- Prefer small reusable 'recipe' functions for calculation types (see `recipes.py`).
- HPC support aims for both Flux (Tuolumne) and SLURM; job writers accept configurable parameters—look at `hpc.py` for supported flags.
- Tests: add unit tests to `tests/`; use fixtures and mock ORCA interactions (do not rely on ORCA being installed in CI).
- Large `data/` folders are working datasets and not packaged; be cautious modifying them.

Common developer workflows

- Local development: `pip install -e .`, then use pre-commit hooks and run `pytest` locally.
- Creating an HPC job set: update `scripts/job_writer_wave2.py` variables (replicates, paths), run `job_writer`, then `run_jobs_wave2.py`, then `check_jobs_wave2.py` for status.
- For ParSL/Slurm runs, review `run_parsl_coordinator.sh` for environment and conda activation examples.

Do's & Don'ts for AI agents (clear, actionable)

- Do: Suggest concrete edits that follow existing file patterns (e.g., add a recipe in `recipes.py` with same function signature and tests in `tests/test_<name>.py`).
- Do: Update `CLAUDE.md` or `README.md` when you discover actionable workflows or missing steps. In this repo we will use both claude and copilot so update .github and CLAUDE.md with learnings
- Don't: Add code that depends on ORCA binaries in tests; instead add mocks or integration docs that instruct how to test on HPC.
- Don't: Replace pipeline example scripts with library code — scripts are examples and may contain hardcoded environment-specific settings.

When unsure — ask for clarifying info

- Ask which HPC system (Flux or Slurm) the contributor targets, what ORCA version they intend to use, and whether they can run integration tests on an HPC node.

Suggested PR checklist for AI-generated changes

- Add/modify unit tests covering behavior (mocks for ORCA calls)
- Run `pre-commit run --all-files` and fix all lint/format issues
- Add brief docs in `CLAUDE.md` or `README.md` for any new major workflows or config variables

Questions for you

- Do you want this file to be more or less prescriptive (examples vs. checklists)?
- Any additional files or workflows I should reference in this guide?

---

Update: Created from `CLAUDE.md`, `README.md`, and repository inspection. Please review and tell me what to clarify or expand.
