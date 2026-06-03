#!/bin/bash
# Build the oact_utilities virtualenv on a DRAC cluster (Fir-first).
#
# Why a venv and not conda: DRAC HOME has a ~500K FILE-COUNT quota separate from
# its byte quota. A conda install alone can eat ~half of it; that is the "out of
# space" you have hit. A virtualenv writes far fewer files and is the Alliance's
# recommended approach.
#
# Usage:
#   bash setup_venv.sh <venv_dir> [python_module]
# Example (put the venv in PROJECT, persistent + backed up, NOT scratch):
#   bash setup_venv.sh "$HOME/projects/def-yqw/$USER/oact-env" python/3.11
#
# Run on a LOGIN node (needs internet for the niche PyPI packages). On Trillium,
# build in HOME (project/home are read-only inside jobs there).

set -euo pipefail

VENV_DIR="${1:?usage: setup_venv.sh <venv_dir> [python_module]}"
PY_MODULE="${2:-python/3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CORE_PKGS=(numpy scipy pandas ase tqdm)        # in the wheelhouse (--no-index)
# install_pkg tries the wheelhouse first then PyPI, so order/source is handled.
# quacc and spyrmsd are imported by the package but NOT declared in pyproject.toml,
# so `pip install -e . --no-deps` does not pull them -- list them explicitly here.
# matplotlib (analysis plotting), lmdb (architector LMDB path), tabulate
# (multi_spin script) are real but non-blocking deps; included for completeness.
NICHE_PKGS=(quacc sella parsl periodictable wandb spyrmsd matplotlib lmdb tabulate)

# --- guards -----------------------------------------------------------------
case "$VENV_DIR" in
    *scratch*) echo "REFUSING: venv under scratch will be purged (>60 days) and break."
               echo "Put it in PROJECT instead."; exit 1 ;;
esac

# pip stages builds in $TMPDIR and caches in $PIP_CACHE_DIR / $XDG_CACHE_HOME.
# A stale SLURM_TMPDIR (e.g. /localscratch/<user>.<oldjob>.0 left in the
# environment by a previous allocation, or exported in ~/.bashrc) is node-local
# and not writable from a login node, breaking pip with "[Errno 13] Permission
# denied". Repoint all three at a per-run dir we know is writable on THIS node:
# a valid SLURM_TMPDIR if present, else /tmp.
if [ -n "${SLURM_TMPDIR:-}" ] && [ -w "${SLURM_TMPDIR:-/nonexistent}" ]; then
    _tmp_base="$SLURM_TMPDIR"
else
    _tmp_base="/tmp"
fi
_build_root="$(mktemp -d "$_tmp_base/oact-venv-build.XXXXXX")"
export TMPDIR="$_build_root"
export PIP_CACHE_DIR="$_build_root/pip-cache"
export XDG_CACHE_HOME="$_build_root/cache"
trap 'rm -rf "$_build_root"' EXIT

# A leaked interactive/JupyterHub session env exports PIP_PREFIX/PIP_TARGET set
# to the job's localscratch, which redirects pip installs OUT of the venv into
# a now-dead directory -> "[Errno 13] Permission denied". Clear them so pip
# installs into the activated venv.
unset PIP_PREFIX PIP_TARGET

echo "venv target : $VENV_DIR"
echo "python module: $PY_MODULE"
echo "repo root   : $REPO_ROOT"
echo "build temp   : $_build_root (TMPDIR + pip/xdg cache)"
echo

# --- 1. modules FIRST, then create + activate the venv ----------------------
# (DRAC docs: never `module load` while a venv is active; load everything first.)
module purge
module load StdEnv/2023 "$PY_MODULE"

echo "## wheelhouse availability (informational):"
avail_wheels "${CORE_PKGS[@]}" "${NICHE_PKGS[@]}" pyyaml pymatgen monty cclib 2>&1 \
    | sed 's/^/    /' || echo "    (avail_wheels not available)"
echo

virtualenv --no-download "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# --- 2. core scientific stack from the wheelhouse (ABI-correct) -------------
# Per package: prefer the wheelhouse (--no-index); fall back to PyPI if absent.
install_pkg() {
    local pkg="$1"; shift
    if pip install --no-index "$pkg" "$@" 2>/dev/null; then
        echo "  [wheelhouse] $pkg"
    else
        echo "  [pypi]       $pkg"
        pip install "$pkg" "$@"
    fi
}

echo "## installing core stack"
for p in "${CORE_PKGS[@]}"; do install_pkg "$p"; done

# Pin the core so PyPI deps of the niche packages cannot yank/upgrade the
# wheelhouse numpy/scipy out from under you.
CONSTRAINTS="$VENV_DIR/core-constraints.txt"
pip freeze --local > "$CONSTRAINTS"
echo "  core pinned -> $CONSTRAINTS"

# --- 3. niche deps (PyPI on login node), constrained to the core versions ---
echo "## installing niche stack (constrained to core pins)"
for p in "${NICHE_PKGS[@]}"; do install_pkg "$p" -c "$CONSTRAINTS"; done

# --- 3b. QTAIM generator metrics (analysis.parse_generator_data) ------------
# oact_utilities uses ONLY qtaim_generator's ORCA-output text parser, which
# pulls no external tools (no Multiwfn). But importing qtaim_gen triggers
# qtaim_gen/__init__.py -> _pymatgen_compat -> pymatgen, so pymatgen is a hard
# import requirement (rdkit/lmdb are only for the generation paths we skip).
# git+ install needs internet -> login node only.
echo "## installing QTAIM generator metrics path (qtaim_generator + pymatgen)"
install_pkg pymatgen -c "$CONSTRAINTS"
pip install "git+https://github.com/santi921/qtaim_generator.git" -c "$CONSTRAINTS"

# --- 4. the package itself, editable, deps already satisfied ----------------
echo "## installing oact_utilities (editable)"
pip install -e "$REPO_ROOT" --no-deps -c "$CONSTRAINTS"

# --- 5. snapshot for replicating on the other clusters ----------------------
REQ="$VENV_DIR/oact-requirements.txt"
pip freeze --local > "$REQ"
echo "  requirements snapshot -> $REQ"

# --- 6. verify ---------------------------------------------------------------
echo
echo "## import check"
python -c "import ase, quacc, sella, parsl, pandas, numpy, scipy, periodictable, tqdm; \
import oact_utilities; print('core imports OK')"
python -c "from qtaim_gen.source.core.parse_orca import parse_orca_output; \
from oact_utilities.utils.analysis import GENERATOR_AVAILABLE; \
print('generator metrics available:', GENERATOR_AVAILABLE)"

echo
echo "## inode usage (this is the quota that historically failed):"
diskusage_report 2>&1 | sed 's/^/    /' || echo "    (diskusage_report unavailable)"

echo
echo "Done. Activate in job scripts with:"
echo "    module load StdEnv/2023 $PY_MODULE orca/6.1.0"
echo "    source $VENV_DIR/bin/activate"
