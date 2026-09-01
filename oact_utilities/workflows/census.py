"""DB-free census of ORCA job-directory corpora, keyed on metal and status.

Walks every job directory under one or more roots and emits a single flat table
(one row per job directory) plus a printed running count of what the corpus is
doing, broken down by status and by metal class (actinide vs non-actinide).
Nothing is read from a workflow database: every column is reconstructed from the
files on disk, so a corpus that was moved, re-rooted, or split across machines
still tallies correctly.

The column set is the union of what
``notebooks/baseline_profiling_actinides.ipynb`` and
``notebooks/check_initial_dbs_oact.ipynb`` consume, minus the columns that only
a workflow DB can supply (``id``, ``fail_count``, ``worker_id``,
``generator_data``, ``source_db``). ``orig_index`` survives only when the
directory name encodes it.

Where each column comes from:

* ``orca.inp``   -- elements, natoms, charge, spin, functional, requested procs
* ``orca.out``   -- status, scf_steps, wall_time, n_cores, final_energy,
                    Mulliken/Loewdin populations (uses the ``orca_metrics.json``
                    cache written by ``parse_job_metrics`` when it is fresh)
* ``orca.engrad``-- energy, per-atom force magnitudes, final coordinates, and
                    the actinide-neighbor force statistics from notebook 2
* directory name -- ``orig_index``

Forces are stored in ORCA's native Eh/Bohr, matching the workflow DB's
``max_forces`` column. Multiply by ``EH_BOHR_TO_EV_ANG`` for eV/Angstrom.

Two force columns exist on purpose and are not the same quantity.
``max_forces`` is what the workflow DB stores -- ORCA's own reported MAX
gradient *component*, or Sella's final fmax on a Sella job -- so it lines up
with the notebooks' DB-sourced plots. ``force_max`` is the largest per-atom
gradient *norm* computed from ``orca.engrad``, which is what the
``force_baselines`` tables hold. ``max_forces`` falls back to ``force_max``
when the output did not report one, and ``final_energy`` falls back to
``engrad_energy`` the same way.

At 400k jobs the table lands near 100 MB of parquet and reads back in well
under a second, but materialising every column costs a few hundred MB of pandas
memory (the path and ``elements`` strings dominate). Pass ``columns=[...]`` to
``read_parquet`` for analysis work.

Usage:
    python -m oact_utilities.workflows.census jobs_a/ jobs_b/ -o census.parquet
    python -m oact_utilities.workflows.census --roots-file roots.json -o out.parquet
    python -m oact_utilities.workflows.census jobs/ -o fast.parquet --no-metrics
    python -m oact_utilities.workflows.census jobs/ -o out.db --format sqlite
"""

from __future__ import annotations

import argparse
import array
import glob
import gzip
import json
import math
import os
import re
import sqlite3
import sys
import time
import warnings
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.orca.calc import ACTINIDE_LIST
from ..utils.analysis import parse_job_metrics
from ..utils.status import parse_failure_reason, pull_log_file
from .clean import MARKER_FILENAME
from .inventory import (
    _STATUS_COMPLETED,
    _STATUS_FAILED,
    _STATUS_ORDER,
    _STATUS_RUNNING,
    _STATUS_TIMEOUT,
    _classify_job_status,
)

# The per-element basis-function table lives in utils/basis.py on branches that
# have split it out of core.orca.calc, and in core.orca.calc on those that have
# not. The two tables are identical; prefer the leaf module when it is there.
try:
    from ..utils.basis import BASIS_DICT
except ImportError:
    from ..core.orca.calc import BASIS_DICT

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 1 Eh/Bohr = 51.42206313 eV/Angstrom (the conversion both notebooks apply).
EH_BOHR_TO_EV_ANG = 51.42206313
BOHR_TO_ANG = 0.529177210903

# ORCA geometry convergence thresholds on the max gradient component (Eh/Bohr),
# the three lines notebook 2 draws over every force histogram.
CONV_TIGHT = 3e-4
CONV_NORMAL = 1e-3
CONV_LOOSE = 3e-3

# Default radius for "atoms coordinated to the metal centre", matching the
# neighbor_cutoff recorded in the force_baselines DBs.
DEFAULT_NEIGHBOR_CUTOFF_ANG = 4.0

_ACTINIDES = frozenset(ACTINIDE_LIST)

# Metal centre picking is tiered rather than a plain highest-Z scan so that a
# heavy p-block ligand atom (Bi, Pb, Te) can never outrank the real centre.
# Within a tier the highest atomic number wins. Tier 2 is the "beyond polonium"
# heavy block these campaigns run alongside the actinides.
_TIER_LANTHANIDE = frozenset("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
_TIER_HEAVY_P = frozenset("Po At Rn Fr Ra".split())
_TIER_D_BLOCK = frozenset(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Hf Ta W Re Os Ir Pt Au Hg".split()
)
_TIER_OTHER_METAL = frozenset("Li Be Na Mg Al K Ca Rb Sr In Sn Cs Ba Tl Pb Bi".split())
_METAL_TIERS = (
    _ACTINIDES,
    _TIER_HEAVY_P,
    _TIER_LANTHANIDE,
    _TIER_D_BLOCK,
    _TIER_OTHER_METAL,
)

# Atomic numbers, index == Z. Element 0 is the ASE "X" placeholder.
_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_Z_OF = {sym: z for z, sym in enumerate(_SYMBOLS)}

_ORIG_INDEX_RE = re.compile(r"(?:^|_)job_(\d+)(?:_|$)")
_IDX_SUFFIX_RE = re.compile(r"idx(\d+)(?:_|$)")
_TRAILING_INT_RE = re.compile(r"(\d+)$")
# A trailing number preceded by q/m is the charge or spin placeholder from
# render_job_dir_pattern, not an index. Names like "UO2C4H6_q0_m5" must yield
# None rather than 5: orig_index is the join key back to a workflow DB, so a
# wrong value is far worse than a missing one.
_CHARGE_SPIN_TAIL_RE = re.compile(r"_[qm]-?\d+$")

# Row schema. One ordered list drives the parquet schema, the SQLite DDL, and
# the CSV header, so the three output formats can never drift apart.
_FIELDS: tuple[tuple[str, str], ...] = (
    # identity
    ("root", "str"),
    ("job_name", "str"),
    ("job_dir", "str"),
    ("orig_index", "int"),
    # status
    ("status", "str"),
    ("termination_code", "int"),
    ("optimizer", "str"),
    ("failure_reason", "str"),
    ("marker", "bool"),
    ("age_hours", "float"),
    # composition
    ("elements", "str"),
    ("formula", "str"),
    ("natoms", "int"),
    ("charge", "int"),
    ("spin", "int"),
    ("n_basis", "int"),
    ("metal", "str"),
    ("metal_class", "str"),
    ("ligand_elements", "str"),
    ("n_ligand_types", "int"),
    # calculation setup
    ("functional", "str"),
    ("simple_input", "str"),
    ("nprocs_requested", "int"),
    # metrics from orca.out
    ("final_energy", "float"),
    ("scf_steps", "int"),
    ("wall_time", "float"),
    ("n_cores", "int"),
    ("sella_steps", "int"),
    ("max_forces", "float"),
    ("metal_mulliken_charge", "float"),
    ("metal_mulliken_spin", "float"),
    ("metal_loewdin_charge", "float"),
    ("metal_loewdin_spin", "float"),
    ("charge_conserved", "bool"),
    ("spin_conserved", "bool"),
    # forces from orca.engrad (Eh/Bohr)
    ("engrad_energy", "float"),
    ("force_max", "float"),
    ("force_mean", "float"),
    ("force_median", "float"),
    ("metal_force", "float"),
    ("ligand_force_max", "float"),
    ("ligand_force_mean", "float"),
    ("n_neighbors", "int"),
    ("neighbor_force_max", "float"),
    ("neighbor_force_mean", "float"),
    ("frac_conv_tight", "float"),
    ("frac_conv_normal", "float"),
    ("frac_conv_loose", "float"),
)
_FIELD_NAMES = tuple(name for name, _ in _FIELDS)

_METAL_CLASS_ACTINIDE = "actinide"
_METAL_CLASS_NON_ACTINIDE = "non_actinide"


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------


def metal_class(symbol: str | None) -> str | None:
    """Classify a metal symbol as ``actinide`` or ``non_actinide``.

    Mirrors ``scripts/stratify_lanes.py:is_actinide`` and the two-way split both
    source notebooks use for their ``ALL_DATASETS`` panels.

    Args:
        symbol: Element symbol, or None when no metal centre was identified.

    Returns:
        ``"actinide"``, ``"non_actinide"``, or None when ``symbol`` is None.
    """
    if symbol is None:
        return None
    return _METAL_CLASS_ACTINIDE if symbol in _ACTINIDES else _METAL_CLASS_NON_ACTINIDE


def pick_metal(symbols: list[str]) -> str | None:
    """Identify the metal centre of a structure from its element symbols.

    Scans the metal tiers in priority order (actinide, heavy p-block Po-Ra,
    lanthanide, d-block, other metal) and returns the highest-Z member of the
    first tier that appears. The tiering keeps a heavy ligand atom (Bi, Pb, Te)
    from outranking the real centre, which a plain highest-Z scan would do.

    Args:
        symbols: Element symbols in the structure.

    Returns:
        The metal centre's symbol, or None when the structure holds no metal.
    """
    present = set(symbols)
    for tier in _METAL_TIERS:
        candidates = present & tier
        if candidates:
            return max(candidates, key=lambda s: _Z_OF.get(s, 0))
    return None


def hill_formula(symbols: list[str]) -> str:
    """Return the Hill-ordered molecular formula for a list of element symbols.

    Carbon first, hydrogen second, then everything else alphabetically. When no
    carbon is present every element is ordered alphabetically.

    Args:
        symbols: Element symbols, one per atom.

    Returns:
        Formula string such as ``"C6H6NpO2"``, or ``""`` for no atoms.
    """
    counts = Counter(symbols)
    if not counts:
        return ""
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
    ordered.extend(sorted(s for s in counts if s not in ordered))
    parts = []
    for sym in ordered:
        n = counts[sym]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def count_basis(symbols: Iterable[str]) -> int | None:
    """Sum the per-element basis-function counts for a structure.

    Equivalent to ``utils.basis.count_basis_functions(symbols, strict=False)``,
    inlined here so census does not depend on which module holds the table on a
    given branch. Returns None rather than raising when a symbol is missing, so
    one unparseable structure cannot abort a 400k-job scan.

    Args:
        symbols: Element symbols, one per atom.

    Returns:
        Total basis-function count, or None if any symbol is not in the table.
    """
    total = 0
    for symbol in symbols:
        n = BASIS_DICT.get(symbol)
        if n is None:
            return None
        total += n
    return total


def _parse_orig_index(job_name: str) -> int | None:
    """Recover ``orig_index`` from a job directory name, or None.

    Handles every pattern ``job_dir_patterns.render_job_dir_pattern`` can emit
    that carries the index: ``job_17``, ``host_job_17``, ``prefix_job_17_x``,
    the ``idx`` suffix used by the real corpora
    (``barfoot_BN4C6H8O2_q2_m1_idx8761``), and a plain trailing integer.

    Returns None when the name ends in a charge/spin placeholder and nothing
    else (``UO2C4H6_q0_m5``): that trailing number is the multiplicity, and
    reporting it as an index would silently corrupt a join back to the DB.
    """
    match = _ORIG_INDEX_RE.search(job_name)
    if match:
        return int(match.group(1))
    match = _IDX_SUFFIX_RE.search(job_name)
    if match:
        return int(match.group(1))
    if _CHARGE_SPIN_TAIL_RE.search(job_name):
        return None
    match = _TRAILING_INT_RE.search(job_name)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def _open_text(path: Path):
    """Open a text file, transparently decompressing a ``.gz`` suffix."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


def parse_inp(inp_path: Path) -> dict:
    """Extract structure and calculation setup from an ORCA input file.

    Reads the ``!`` simple-input line, ``%pal nprocs``, and the ``* xyz C M``
    coordinate block. Handles plain and gzipped inputs. Unlike
    ``analysis.parse_charge_mult_from_inp`` this also returns the element list,
    which is the only DB-free source of composition for a job that never ran.

    Args:
        inp_path: Path to ``orca.inp`` or ``orca.inp.gz``.

    Returns:
        Dict with keys ``symbols``, ``charge``, ``spin``, ``simple_input``,
        ``functional``, ``nprocs_requested``. Missing pieces are None / empty.
    """
    result: dict = {
        "symbols": [],
        "charge": None,
        "spin": None,
        "simple_input": None,
        "functional": None,
        "nprocs_requested": None,
    }
    try:
        with _open_text(inp_path) as f:
            in_coords = False
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if in_coords:
                    if stripped.startswith("*"):
                        break
                    symbol = stripped.split()[0]
                    # ORCA allows "C(1)" style labels and dummy atoms.
                    symbol = symbol.split("(")[0].split(":")[0]
                    if symbol in _Z_OF:
                        result["symbols"].append(symbol)
                    continue
                if stripped.startswith("!"):
                    simple = stripped.lstrip("!").strip()
                    result["simple_input"] = simple
                    tokens = simple.split()
                    if tokens:
                        result["functional"] = tokens[0]
                elif stripped.lower().startswith("nprocs"):
                    tokens = stripped.split()
                    if len(tokens) >= 2 and tokens[1].isdigit():
                        result["nprocs_requested"] = int(tokens[1])
                elif stripped.startswith("* xyz") or stripped.startswith("*xyz"):
                    tokens = stripped.split()
                    offset = 2 if tokens[0] == "*" else 1
                    try:
                        result["charge"] = int(tokens[offset])
                        result["spin"] = int(tokens[offset + 1])
                    except (IndexError, ValueError):
                        pass
                    in_coords = True
    except OSError:
        pass
    return result


def parse_engrad(engrad_path: Path) -> dict:
    """Read an ORCA ``.engrad`` file into energy, gradient, and geometry.

    Same format as ``analysis.get_engrad`` but streams from a file handle so a
    gzipped quacc ``orca.engrad.gz`` is read without a temporary file, which
    matters at 400k jobs.

    Args:
        engrad_path: Path to ``orca.engrad`` or ``orca.engrad.gz``.

    Returns:
        Dict with ``energy`` (Eh), ``gradient`` (flat Eh/Bohr list),
        ``symbols``, and ``coords_bohr`` (flat list). Empty on a parse failure.
    """
    energy: float | None = None
    gradient: list[float] = []
    symbols: list[str] = []
    coords: list[float] = []
    try:
        with _open_text(engrad_path) as f:
            f_iter = iter(f)
            for line in f_iter:
                if "The current total energy in Eh" in line:
                    next(f_iter)
                    energy = float(next(f_iter).strip())
                elif "The current gradient in Eh/bohr" in line:
                    next(f_iter)
                    for grad_line in f_iter:
                        if grad_line.strip() == "#":
                            break
                        gradient.append(float(grad_line.strip()))
                elif "The atomic numbers and current coordinates in Bohr" in line:
                    next(f_iter)
                    for coord_line in f_iter:
                        if not coord_line.strip():
                            break
                        parts = coord_line.split()
                        z = int(parts[0])
                        symbols.append(_SYMBOLS[z] if z < len(_SYMBOLS) else "X")
                        coords.extend(float(x) for x in parts[1:4])
    except (OSError, ValueError, IndexError, StopIteration, EOFError):
        return {}
    return {
        "energy": energy,
        "gradient": gradient,
        "symbols": symbols,
        "coords_bohr": coords,
    }


def force_stats(
    symbols: list[str],
    coords_bohr: list[float],
    gradient: list[float],
    metal: str | None,
    neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF_ANG,
) -> dict:
    """Reduce a gradient to the per-job force statistics both notebooks plot.

    Splits the per-atom gradient norms into the metal centre, the ligand atoms,
    and the ligand atoms coordinated to the metal (within ``neighbor_cutoff``),
    which is the actinide-neighbor decomposition notebook 2 reads out of the
    ``force_baselines`` DBs. Also reports the fraction of atoms under each ORCA
    convergence threshold.

    Args:
        symbols: Element symbol per atom.
        coords_bohr: Flat ``[x, y, z, ...]`` coordinates in Bohr.
        gradient: Flat ``[gx, gy, gz, ...]`` gradient in Eh/Bohr.
        metal: Metal centre symbol, or None to skip the metal/neighbor split.
        neighbor_cutoff: Coordination radius in Angstrom.

    Returns:
        Dict of the ``force_*`` / ``metal_force`` / ``ligand_force_*`` /
        ``n_neighbors`` / ``neighbor_force_*`` / ``frac_conv_*`` columns. Empty
        when the gradient is missing or inconsistent with the atom count.
    """
    natoms = len(symbols)
    if natoms == 0 or len(gradient) != 3 * natoms:
        return {}

    norms = [
        math.sqrt(
            gradient[3 * i] ** 2 + gradient[3 * i + 1] ** 2 + gradient[3 * i + 2] ** 2
        )
        for i in range(natoms)
    ]
    ordered = sorted(norms)
    mid = natoms // 2
    median = ordered[mid] if natoms % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    stats: dict = {
        "force_max": ordered[-1],
        "force_mean": sum(norms) / natoms,
        "force_median": median,
        "frac_conv_tight": sum(1 for v in norms if v <= CONV_TIGHT) / natoms,
        "frac_conv_normal": sum(1 for v in norms if v <= CONV_NORMAL) / natoms,
        "frac_conv_loose": sum(1 for v in norms if v <= CONV_LOOSE) / natoms,
    }

    if metal is None or metal not in symbols:
        return stats

    metal_idx = symbols.index(metal)
    stats["metal_force"] = norms[metal_idx]

    ligand_idx = [i for i in range(natoms) if i != metal_idx]
    if ligand_idx:
        ligand_norms = [norms[i] for i in ligand_idx]
        stats["ligand_force_max"] = max(ligand_norms)
        stats["ligand_force_mean"] = sum(ligand_norms) / len(ligand_norms)

    if len(coords_bohr) != 3 * natoms:
        return stats

    cutoff_bohr = neighbor_cutoff / BOHR_TO_ANG
    cutoff_sq = cutoff_bohr * cutoff_bohr
    mx, my, mz = coords_bohr[3 * metal_idx : 3 * metal_idx + 3]
    neighbor_norms = []
    for i in ligand_idx:
        dx = coords_bohr[3 * i] - mx
        dy = coords_bohr[3 * i + 1] - my
        dz = coords_bohr[3 * i + 2] - mz
        if dx * dx + dy * dy + dz * dz <= cutoff_sq:
            neighbor_norms.append(norms[i])
    stats["n_neighbors"] = len(neighbor_norms)
    if neighbor_norms:
        stats["neighbor_force_max"] = max(neighbor_norms)
        stats["neighbor_force_mean"] = sum(neighbor_norms) / len(neighbor_norms)
    return stats


# ---------------------------------------------------------------------------
# Per-job scan
# ---------------------------------------------------------------------------


def _find(names: list[str], plain: str, gz: str) -> str | None:
    """Return the plain filename if present, else the gzipped one, else None."""
    if plain in names:
        return plain
    if gz in names:
        return gz
    return None


def _metal_population(
    population: dict | None, metal: str | None
) -> dict[str, float | bool | None]:
    """Pull the metal centre's charge and spin out of a population analysis dict.

    Args:
        population: The ``mulliken_population`` dict from ``parse_job_metrics``.
        metal: Metal centre symbol.

    Returns:
        Dict of the four ``metal_*`` population columns plus the two
        conservation flags. Values are None when unavailable.
    """
    out: dict[str, float | bool | None] = {}
    if not population:
        return out

    validation = population.get("validation") or {}
    out["charge_conserved"] = validation.get("charge_valid")
    out["spin_conserved"] = validation.get("spin_valid")

    elements = population.get("elements") or []
    if metal is None or metal not in elements:
        return out
    idx = elements.index(metal)
    for key, column in (
        ("mulliken_charges", "metal_mulliken_charge"),
        ("mulliken_spins", "metal_mulliken_spin"),
        ("loewdin_charges", "metal_loewdin_charge"),
        ("loewdin_spins", "metal_loewdin_spin"),
    ):
        values = population.get(key) or []
        if idx < len(values):
            out[column] = values[idx]
    return out


def _label_from_code(code: int) -> str:
    """Map a ``check_file_termination`` code to a status label.

    Only called for a directory that already has real ORCA output, so the
    ``to_run`` case ``inventory._classify_job_status`` handles cannot arise.
    """
    if code == 1:
        return _STATUS_COMPLETED
    if code == -1:
        return _STATUS_FAILED
    if code == -2:
        return _STATUS_TIMEOUT
    return _STATUS_RUNNING


def scan_job(
    job_dir: Path,
    root: Path,
    with_metrics: bool = True,
    with_forces: bool = True,
    hours_cutoff: int = 24,
    neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF_ANG,
    recompute: bool = False,
) -> dict:
    """Build one census row from a single job directory.

    Composition always comes from ``orca.inp`` when present, falling back to the
    ``orca.engrad`` atom list for a directory whose input was purged. Status is
    the same on-disk classification ``inventory.py --status`` performs: taken
    from the metrics tier's termination code when that tier runs (it is the
    same code, already computed and cached), and from
    ``inventory._classify_job_status`` otherwise.

    Args:
        job_dir: The job directory to scan.
        root: The root it was found under (recorded in the ``root`` column).
        with_metrics: Read ``orca.out`` for scf_steps / wall_time / n_cores /
            final_energy / populations. This is the expensive tier; it uses the
            ``orca_metrics.json`` cache when that cache is fresh.
        with_forces: Read ``orca.engrad`` for energies and force statistics.
        hours_cutoff: Hours of inactivity before a job with no termination
            signal is classified as timed out rather than running.
        neighbor_cutoff: Metal coordination radius in Angstrom.
        recompute: Bypass the ``orca_metrics.json`` cache and re-read the output.

    Returns:
        A dict keyed by ``_FIELD_NAMES``. Unavailable columns are None.
    """
    row: dict = dict.fromkeys(_FIELD_NAMES)
    row["root"] = str(root)
    row["job_name"] = job_dir.name
    row["job_dir"] = str(job_dir)
    row["orig_index"] = _parse_orig_index(job_dir.name)

    try:
        names = os.listdir(job_dir)
    except OSError:
        row["status"] = "unreadable"
        return row

    row["marker"] = MARKER_FILENAME in names
    row["optimizer"] = "sella" if "run_sella.py" in names else "orca"

    # -- composition ------------------------------------------------------
    inp_name = _find(names, "orca.inp", "orca.inp.gz")
    if inp_name is None:
        inp_name = next((n for n in names if n.endswith((".inp", ".inp.gz"))), None)
    symbols: list[str] = []
    if inp_name is not None:
        inp = parse_inp(job_dir / inp_name)
        symbols = inp["symbols"]
        row["charge"] = inp["charge"]
        row["spin"] = inp["spin"]
        row["simple_input"] = inp["simple_input"]
        row["functional"] = inp["functional"]
        row["nprocs_requested"] = inp["nprocs_requested"]

    # -- forces and engrad energy ----------------------------------------
    engrad_name = None
    if with_forces:
        engrad_name = _find(names, "orca.engrad", "orca.engrad.gz")
        if engrad_name is None:
            engrad_name = next(
                (n for n in names if n.endswith((".engrad", ".engrad.gz"))), None
            )
    engrad: dict = {}
    if engrad_name is not None:
        engrad = parse_engrad(job_dir / engrad_name)
        if not symbols:
            symbols = engrad.get("symbols") or []

    if symbols:
        row["elements"] = ";".join(symbols)
        row["formula"] = hill_formula(symbols)
        row["natoms"] = len(symbols)
        row["n_basis"] = count_basis(symbols)
        metal = pick_metal(symbols)
        row["metal"] = metal
        row["metal_class"] = metal_class(metal)
        ligands = sorted({s for s in symbols if s != metal and s != "H"})
        row["ligand_elements"] = ",".join(ligands)
        row["n_ligand_types"] = len(ligands)
    else:
        metal = None

    if engrad:
        row["engrad_energy"] = engrad.get("energy")
        row.update(
            force_stats(
                engrad.get("symbols") or symbols,
                engrad.get("coords_bohr") or [],
                engrad.get("gradient") or [],
                metal,
                neighbor_cutoff,
            )
        )

    # -- output file: age, expensive metrics, status ----------------------
    try:
        out_path = pull_log_file(str(job_dir))
    except (FileNotFoundError, PermissionError):
        out_path = None

    age_hours: float | None = None
    if out_path is not None:
        try:
            age_hours = (time.time() - os.path.getmtime(out_path)) / 3600.0
            row["age_hours"] = age_hours
        except OSError:
            pass

    if with_metrics and out_path is not None:
        # quacc dirs carry only orca.out.gz; parse_job_metrics needs to be told.
        unzip = out_path.endswith(".gz")
        # parse_job_metrics is annotated dict[str, float | int | None] but also
        # carries the population dict and error strings; widen it here.
        metrics: dict[str, Any] = parse_job_metrics(
            job_dir,
            unzip=unzip,
            hours_cutoff=hours_cutoff,
            recompute=recompute,
        )
        row["final_energy"] = metrics.get("final_energy")
        row["scf_steps"] = metrics.get("scf_steps")
        row["wall_time"] = metrics.get("wall_time")
        row["n_cores"] = metrics.get("nprocs")
        row["sella_steps"] = metrics.get("sella_steps")
        row["max_forces"] = metrics.get("max_forces")
        row.update(_metal_population(metrics.get("mulliken_population"), metal))

        # parse_job_metrics already determined termination (and cached it), so
        # reuse that code rather than re-tailing the output. That matters on a
        # gzipped corpus: a .out.gz tail costs a full decompress, and running it
        # twice per job doubles the scan. The cached code was computed under
        # whatever hours_cutoff was in force when the cache was written, so the
        # stale-file heuristic is re-applied here against the current one.
        status_code = int(metrics.get("termination_status") or 0)
        if status_code == 0 and age_hours is not None and age_hours > hours_cutoff:
            status_code = -2
        status_label = _label_from_code(status_code)
    else:
        status_label, status_code = _classify_job_status(job_dir, hours_cutoff)

    row["status"] = status_label
    row["termination_code"] = status_code

    if out_path is not None and status_label in (_STATUS_FAILED, _STATUS_TIMEOUT):
        try:
            row["failure_reason"] = parse_failure_reason(out_path)
        except OSError:
            pass

    # max_forces / final_energy are the DB-shaped columns and come from the
    # output; fill them from the engrad tier when the output reported neither.
    if row["max_forces"] is None:
        row["max_forces"] = row["force_max"]
    if row["final_energy"] is None:
        row["final_energy"] = row["engrad_energy"]

    return row


def iter_job_dirs(roots: list[Path], limit: int | None = None):
    """Yield ``(root, job_dir)`` for every immediate subdirectory of each root.

    Args:
        roots: Root directories whose immediate subdirectories are job dirs.
        limit: Optional cap on job directories taken from each root (testing).

    Yields:
        ``(root, job_dir)`` pairs, roots in the order given, dirs sorted.
    """
    for root in roots:
        try:
            with os.scandir(root) as it:
                entries = sorted(e.path for e in it if e.is_dir(follow_symlinks=False))
        except OSError as exc:
            print(f"  Warning: cannot list {root}: {exc}", file=sys.stderr)
            continue
        if limit is not None:
            entries = entries[:limit]
        for path in entries:
            yield root, Path(path)


# ---------------------------------------------------------------------------
# Chunked writers (parquet / sqlite / csv)
# ---------------------------------------------------------------------------

_SQL_TYPES = {"str": "TEXT", "int": "INTEGER", "float": "REAL", "bool": "INTEGER"}


def census_schema():
    """The fixed pyarrow schema every census parquet carries.

    Declared up front rather than inferred from data, so an all-null column in
    one shard cannot come back with a different type and break a later merge.
    Shards written by different runs are therefore always concatenable.
    """
    return pa.schema(
        [
            pa.field(
                name,
                {
                    "str": pa.string(),
                    "int": pa.int64(),
                    "float": pa.float64(),
                    "bool": pa.bool_(),
                }[kind],
            )
            for name, kind in _FIELDS
        ]
    )


class _ParquetSink:
    """Append-only parquet writer with a fixed schema, flushed in row chunks."""

    def __init__(self, path: Path, chunk_size: int) -> None:
        self.schema = census_schema()
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.chunk_size = chunk_size
        self.buffer: list[dict] = []

    def write(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        columns = {name: [r.get(name) for r in self.buffer] for name in _FIELD_NAMES}
        self.writer.write_table(pa.table(columns, schema=self.schema))
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


class _SqliteSink:
    """Append-only SQLite writer, committed in row chunks."""

    def __init__(self, path: Path, chunk_size: int) -> None:
        if path.exists():
            path.unlink()
        self.conn = sqlite3.connect(path)
        columns = ", ".join(f"{n} {_SQL_TYPES[k]}" for n, k in _FIELDS)
        self.conn.execute(f"CREATE TABLE census ({columns})")
        self.conn.execute("CREATE INDEX idx_status ON census(status)")
        self.conn.execute("CREATE INDEX idx_metal ON census(metal)")
        self.insert = (
            f"INSERT INTO census ({', '.join(_FIELD_NAMES)}) "
            f"VALUES ({', '.join('?' * len(_FIELD_NAMES))})"
        )
        self.chunk_size = chunk_size
        self.buffer: list[tuple] = []

    def write(self, row: dict) -> None:
        self.buffer.append(tuple(row.get(name) for name in _FIELD_NAMES))
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.conn.executemany(self.insert, self.buffer)
        self.conn.commit()
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()


class _CsvSink:
    """Append-only CSV writer with the schema's column order as its header."""

    def __init__(self, path: Path, chunk_size: int) -> None:
        import csv

        self.handle = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(_FIELD_NAMES))
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow({n: row.get(n) for n in _FIELD_NAMES})

    def close(self) -> None:
        self.handle.close()


def _make_sink(path: Path, fmt: str, chunk_size: int):
    """Build the output sink for ``fmt``, falling back to SQLite without pyarrow."""
    if fmt == "parquet":
        if not PYARROW_AVAILABLE:
            fallback = path.with_suffix(".db")
            print(
                f"  pyarrow not installed; writing SQLite to {fallback} instead",
                file=sys.stderr,
            )
            return _SqliteSink(fallback, chunk_size), fallback
        return _ParquetSink(path, chunk_size), path
    if fmt == "sqlite":
        return _SqliteSink(path, chunk_size), path
    return _CsvSink(path, chunk_size), path


# ---------------------------------------------------------------------------
# Streaming summary ("the running count")
# ---------------------------------------------------------------------------


class Summary:
    """Running tallies accumulated as census rows stream past.

    Holds only counters plus compact per-metal ``array`` buffers of natoms,
    wall_time, and max_forces, so a 400k-job corpus summarises in tens of MB
    instead of materialising every row.
    """

    def __init__(self) -> None:
        self.total = 0
        self.by_root: Counter = Counter()
        self.root_status: Counter = Counter()  # (root, status)
        self.status: Counter = Counter()
        self.class_status: Counter = Counter()  # (metal_class, status)
        self.metal_status: Counter = Counter()  # (metal, status)
        self.metal_to_class: dict[str, str] = {}
        self.failures: Counter = Counter()
        self.no_metal = 0
        self.natoms: dict[str, array.array] = {}
        self.wall_time: dict[str, array.array] = {}
        self.max_forces: dict[str, array.array] = {}
        self.conv_normal: Counter = Counter()  # metal_class -> jobs at/below 1e-3
        self.conv_total: Counter = Counter()
        # A job directory must appear exactly once. A repeat means the roots
        # overlapped (one nested in another) or a merge re-ingested a shard --
        # both silently double-count, so they are surfaced, not swallowed.
        # Hashes rather than the paths themselves: at 800k jobs a set of full
        # job_dir strings costs well over 100 MB, a set of their hashes ~25 MB.
        # A 64-bit collision across 1M rows is ~3e-8 likely and would only
        # nudge a warning count, never the table.
        self._seen_job_dirs: set[int] = set()
        self.duplicate_job_dirs: int = 0

    def add(self, row: dict) -> None:
        self.total += 1
        job_dir = row.get("job_dir")
        if job_dir is not None:
            key = hash(job_dir)
            if key in self._seen_job_dirs:
                self.duplicate_job_dirs += 1
            else:
                self._seen_job_dirs.add(key)
        root = row["root"]
        status = row["status"] or "unknown"
        metal = row["metal"] or "(none)"
        mclass = row["metal_class"] or "(none)"

        self.by_root[root] += 1
        self.root_status[(root, status)] += 1
        self.status[status] += 1
        self.class_status[(mclass, status)] += 1
        self.metal_status[(metal, status)] += 1
        self.metal_to_class[metal] = mclass
        if row["metal"] is None:
            self.no_metal += 1
        if row["failure_reason"]:
            self.failures[row["failure_reason"][:120]] += 1

        if status != "completed":
            return
        if row["natoms"] is not None:
            self.natoms.setdefault(metal, array.array("i")).append(row["natoms"])
        if row["wall_time"] is not None:
            self.wall_time.setdefault(metal, array.array("d")).append(row["wall_time"])
        mf = row["max_forces"]
        if mf is not None:
            self.max_forces.setdefault(metal, array.array("d")).append(mf)
            self.conv_total[mclass] += 1
            if mf <= CONV_NORMAL:
                self.conv_normal[mclass] += 1


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of a pre-sorted list."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _status_columns(summary: Summary) -> list[str]:
    """Status labels present in the corpus, in the canonical report order."""
    seen = set(summary.status)
    ordered = [s for s in _STATUS_ORDER if s in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _print_report(summary: Summary, top: int = 15) -> None:
    """Print the running count: status by root, by metal class, and by metal."""
    statuses = _status_columns(summary)
    width = max(12, *(len(s) for s in statuses)) if statuses else 12

    print()
    print("=" * 78)
    print(f"CENSUS: {summary.total:,} job directories")
    print("=" * 78)

    if not summary.total:
        print("No job directories found.")
        return

    header = "".join(f"{s:>{width}}" for s in statuses)
    print(f"\n{'root':<40}{'total':>10}{header}")
    print("-" * (50 + width * len(statuses)))
    for root, n in summary.by_root.most_common():
        label = root if len(root) <= 39 else "..." + root[-36:]
        cells = "".join(f"{summary.root_status[(root, s)]:>{width},}" for s in statuses)
        print(f"{label:<40}{n:>10,}{cells}")
    cells = "".join(f"{summary.status[s]:>{width},}" for s in statuses)
    print(f"{'ALL':<40}{summary.total:>10,}{cells}")

    print(f"\n{'metal class':<40}{'total':>10}{header}")
    print("-" * (50 + width * len(statuses)))
    classes = sorted({k[0] for k in summary.class_status})
    for mclass in classes:
        total = sum(summary.class_status[(mclass, s)] for s in statuses)
        cells = "".join(
            f"{summary.class_status[(mclass, s)]:>{width},}" for s in statuses
        )
        print(f"{mclass:<40}{total:>10,}{cells}")

    print("\n=== Completed jobs by metal ===")
    print(
        f"{'metal':<8}{'class':<14}{'completed':>11}{'natoms':>18}"
        f"{'wall_time (hr)':>26}{'max_force med':>16}"
    )
    print(
        f"{'':<8}{'':<14}{'':>11}{'min/med/max':>18}"
        f"{'med / p95 / max':>26}{'(Eh/Bohr)':>16}"
    )
    print("-" * 93)
    metals = sorted(
        {m for m, s in summary.metal_status if s == "completed"},
        key=lambda m: -summary.metal_status[(m, "completed")],
    )
    for metal in metals:
        n = summary.metal_status[(metal, "completed")]
        na = sorted(summary.natoms.get(metal, []))
        wt = sorted(v / 3600.0 for v in summary.wall_time.get(metal, []))
        mf = sorted(summary.max_forces.get(metal, []))
        mclass = summary.metal_to_class.get(metal, "(none)")
        na_txt = f"{na[0]}/{int(_quantile(na, 0.5))}/{na[-1]}" if na else "-"
        wt_txt = (
            f"{_quantile(wt, 0.5):.2f} / {_quantile(wt, 0.95):.2f} / {wt[-1]:.2f}"
            if wt
            else "-"
        )
        mf_txt = f"{_quantile(mf, 0.5):.2e}" if mf else "-"
        print(f"{metal:<8}{mclass:<14}{n:>11,}{na_txt:>18}{wt_txt:>26}{mf_txt:>16}")

    if summary.conv_total:
        print(
            "\n=== Force convergence of completed jobs (max_force <= 1e-3 Eh/Bohr) ==="
        )
        for mclass in sorted(summary.conv_total):
            total = summary.conv_total[mclass]
            good = summary.conv_normal[mclass]
            print(
                f"  {mclass:<16} {good:>8,} / {total:<8,} "
                f"({good / total * 100:5.1f}%)"
            )

    if summary.duplicate_job_dirs:
        unique = summary.total - summary.duplicate_job_dirs
        print()
        print("!" * 78)
        print(
            f"  WARNING: {summary.duplicate_job_dirs:,} of {summary.total:,} rows "
            f"repeat a job directory ({unique:,} unique)."
        )
        print("  Either the roots overlap (one nested inside another) or a merge")
        print("  re-ingested a shard -- e.g. --merge on a directory that already")
        print("  holds a previous merge output. Every count above is inflated.")
        print("!" * 78)

    if summary.no_metal:
        print(f"\nStructures with no identifiable metal centre: {summary.no_metal:,}")

    if summary.failures:
        print(f"\n=== Top {top} failure reasons ===")
        for reason, count in summary.failures.most_common(top):
            print(f"  [{count:>6,}x] {reason}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_census(
    roots: list[Path],
    out_path: Path,
    fmt: str = "parquet",
    with_metrics: bool = True,
    with_forces: bool = True,
    workers: int = 8,
    hours_cutoff: int = 24,
    neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF_ANG,
    recompute: bool = False,
    limit: int | None = None,
    chunk_size: int = 20000,
) -> tuple[Summary, Path]:
    """Scan every job directory under ``roots`` and stream one table to disk.

    Rows are written in chunks as they complete, so peak memory stays flat
    regardless of corpus size. Summary tallies accumulate alongside.

    Args:
        roots: Root directories whose immediate subdirectories are job dirs.
        out_path: Output file path.
        fmt: ``parquet``, ``sqlite``, or ``csv``.
        with_metrics: Read ``orca.out`` (scf_steps, wall_time, n_cores,
            populations). The expensive tier.
        with_forces: Read ``orca.engrad`` (energies and force statistics).
        workers: Parallel scan threads.
        hours_cutoff: Running-vs-timeout threshold in hours.
        neighbor_cutoff: Metal coordination radius in Angstrom.
        recompute: Bypass the ``orca_metrics.json`` cache.
        limit: Cap job directories taken from each root (testing).
        chunk_size: Rows buffered before each flush to disk.

    Returns:
        ``(summary, written_path)``. The path differs from ``out_path`` only
        when a parquet request fell back to SQLite.
    """
    sink, written = _make_sink(out_path, fmt, chunk_size)
    summary = Summary()
    pairs = list(iter_job_dirs(roots, limit=limit))

    def work(pair: tuple[Path, Path]) -> dict:
        root, job_dir = pair
        return scan_job(
            job_dir,
            root,
            with_metrics=with_metrics,
            with_forces=with_forces,
            hours_cutoff=hours_cutoff,
            neighbor_cutoff=neighbor_cutoff,
            recompute=recompute,
        )

    progress = (
        tqdm(total=len(pairs), desc="Scanning", unit="job")
        if tqdm is not None
        else None
    )
    # Submit in blocks rather than one 400k-wide pool.map: that would build a
    # Future per job up front, and the futures alone outweigh the rows.
    block = max(chunk_size, workers * 64)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(pairs), block):
                for row in pool.map(work, pairs[start : start + block]):
                    summary.add(row)
                    sink.write(row)
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        sink.close()

    return summary, written


def expand_parquet_inputs(patterns: list[str]) -> list[Path]:
    """Resolve merge inputs: files, directories, or glob patterns.

    A directory contributes every ``*.parquet`` inside it (non-recursive).
    Results are de-duplicated by resolved path and sorted, so passing both a
    directory and one of its files cannot double-count a shard.

    Args:
        patterns: Paths, directories, or shell-style globs.

    Returns:
        Sorted, de-duplicated parquet file paths.
    """
    found: set[Path] = set()
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        path = Path(expanded)
        if path.is_dir():
            found.update(p.resolve() for p in sorted(path.glob("*.parquet")))
        elif path.is_file():
            found.add(path.resolve())
        else:
            matches = [Path(m).resolve() for m in sorted(glob.glob(expanded))]
            if not matches:
                print(f"  Warning: no match for {pattern}", file=sys.stderr)
            found.update(m for m in matches if m.is_file())
    return sorted(found)


def merge_census(
    inputs: list[Path],
    out_path: Path,
    fmt: str = "parquet",
    chunk_size: int = 20000,
    batch_size: int = 20000,
) -> tuple[Summary, Path]:
    """Concatenate census shards written by separate (possibly parallel) runs.

    Every shard carries the same declared schema (see :func:`census_schema`), so
    the merge is a straight concatenation -- no schema unification, no type
    reconciliation. Shards stream through in record batches, so nothing scales
    with the number of shards and the corpus is never held whole; the combined
    running count is tallied on the same pass. Measured peak RSS grows
    sublinearly -- 0.63 GB at 200k rows, 0.78 GB at 400k, 0.87 GB at 800k --
    against a 0.23 GB floor for the interpreter and imports alone. The batch
    buffers dominate; only the duplicate-detection hash set grows with rows
    (~25 MB per 800k). Tuning ``chunk_size`` barely moves it.

    The ``root`` column survives the merge, so a merged table can still be
    grouped back per source root.

    Args:
        inputs: Census parquet files to concatenate.
        out_path: Combined output path.
        fmt: Output format for the merged table.
        chunk_size: Rows buffered before each flush to disk.
        batch_size: Rows read per input batch.

    Returns:
        ``(summary, written_path)`` for the combined table.

    Raises:
        ValueError: If a shard's schema does not match, or if ``out_path`` is
            also one of the inputs.
    """
    if not PYARROW_AVAILABLE:
        raise ValueError("merging parquet shards requires pyarrow")

    expected = census_schema()
    resolved_out = out_path.resolve()
    for path in inputs:
        if path == resolved_out:
            raise ValueError(
                f"output {out_path} is also an input; pick a different -o path"
            )
        schema = pq.read_schema(path)
        if not schema.equals(expected):
            missing = sorted(set(expected.names) - set(schema.names))
            extra = sorted(set(schema.names) - set(expected.names))

            def _brief(names: list[str]) -> str:
                head = ", ".join(names[:5])
                return head + (
                    f", ... (+{len(names) - 5} more)" if len(names) > 5 else ""
                )

            detail = ""
            if missing:
                detail += f" missing=[{_brief(missing)}]"
            if extra:
                detail += f" unexpected=[{_brief(extra)}]"
            raise ValueError(
                f"{path} is not a census table written by this version"
                f" ({len(schema.names)} columns, expected {len(expected.names)})"
                f"{detail}"
            )

    sink, written = _make_sink(out_path, fmt, chunk_size)
    summary = Summary()
    progress = tqdm(total=len(inputs), desc="Merging", unit="shard") if tqdm else None
    try:
        for path in inputs:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    summary.add(row)
                    sink.write(row)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        sink.close()

    return summary, written


def _load_roots(args: argparse.Namespace) -> list[Path]:
    """Collect root directories from positional args and/or ``--roots-file``.

    The roots file may hold a JSON list of paths or one path per line
    (``#`` comments and blanks ignored).
    """
    roots: list[str] = list(args.roots)
    if args.roots_file is not None:
        text = args.roots_file.read_text()
        stripped = text.lstrip()
        if stripped.startswith("["):
            roots.extend(str(p) for p in json.loads(text))
        else:
            roots.extend(
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            )

    resolved: list[Path] = []
    for r in roots:
        path = Path(os.path.expanduser(r))
        if not path.is_dir():
            print(f"Error: not a directory: {path}", file=sys.stderr)
            sys.exit(1)
        resolved.append(path)
    if not resolved:
        print("Error: no root directories given", file=sys.stderr)
        sys.exit(1)
    return resolved


def main() -> None:
    """CLI entry point for the DB-free job-directory census."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.census",
        description=(
            "Census every job directory under one or more roots: status, metal "
            "(actinide vs non-actinide), composition, energies, and forces. "
            "Reads only the files on disk -- no workflow database."
        ),
    )
    parser.add_argument(
        "roots",
        nargs="*",
        help="Root directories whose immediate subdirectories are job dirs. "
        "With --merge these are instead the census parquet shards to combine "
        "(files, directories, or globs).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Combine census parquet shards from earlier runs into one table "
        "instead of scanning job directories. Use this after running several "
        "census processes in parallel on different roots. Every shard carries "
        "the same declared schema, so the merge is a plain concatenation; it "
        "streams in batches (constant memory) and reprints the combined "
        "running count. The `root` column survives, so a merged table can "
        "still be grouped per source root.",
    )
    parser.add_argument(
        "--roots-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="File holding the root list: a JSON array or one path per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output table path",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "sqlite", "csv"),
        default="parquet",
        help="Output format (default: parquet; falls back to sqlite without pyarrow)",
    )

    tiers = parser.add_argument_group("what to extract")
    tiers.add_argument(
        "--no-forces",
        action="store_true",
        help="Skip orca.engrad: no engrad energy, no per-atom force statistics, "
        "no neighbor decomposition. Forces and energies are extracted by default.",
    )
    tiers.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip the full orca.out read: no scf_steps, wall_time, n_cores, or "
        "populations. Status, composition, metal, and (unless --no-forces) engrad "
        "energies and forces are still extracted. Faster on a cold corpus, but "
        "SLOWER on a gzipped one whose orca_metrics.json caches are already "
        "warm: without metrics the status check has to tail orca.out.gz itself, "
        "which costs a full decompress that the cache would have avoided.",
    )
    tiers.add_argument(
        "--recompute",
        action="store_true",
        help="Bypass each job's orca_metrics.json cache and re-read orca.out",
    )
    tiers.add_argument(
        "--neighbor-cutoff",
        type=float,
        default=DEFAULT_NEIGHBOR_CUTOFF_ANG,
        metavar="ANG",
        help=f"Metal coordination radius in Angstrom "
        f"(default: {DEFAULT_NEIGHBOR_CUTOFF_ANG})",
    )

    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel scan threads (default: 8)"
    )
    parser.add_argument(
        "--hours-cutoff",
        type=int,
        default=24,
        metavar="H",
        help="Hours of inactivity before a job with no termination signal is "
        "classified as timed out instead of running (default: 24)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        metavar="N",
        help="Rows buffered before each flush to disk (default: 20000)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        metavar="N",
        help="How many failure reasons to list (default: 15)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N job directories per root, for testing",
    )
    args = parser.parse_args()

    # parse_job_metrics warns per job on a charge/spin conservation violation;
    # at corpus scale that is noise, and the columns record it instead.
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="oact_utilities.utils.analysis"
    )

    started = time.time()

    if args.merge:
        for unsupported, flag in (
            (args.roots_file is not None, "--roots-file"),
            (args.no_forces, "--no-forces"),
            (args.no_metrics, "--no-metrics"),
            (args.recompute, "--recompute"),
            (args.debug is not None, "--debug"),
        ):
            if unsupported:
                parser.error(f"{flag} does not apply with --merge")
        inputs = expand_parquet_inputs(args.roots)
        if not inputs:
            parser.error("--merge needs at least one census parquet to combine")
        print(f"Merging {len(inputs)} census shard(s):")
        for path in inputs:
            print(f"  {path}")
        try:
            summary, written = merge_census(
                inputs,
                args.output,
                fmt=args.format,
                chunk_size=args.chunk_size,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_report(summary, top=args.top)
        elapsed = time.time() - started
        print(
            f"\nMerged {summary.total:,} rows x {len(_FIELDS)} columns from "
            f"{len(inputs)} shard(s) to {written}"
            f"  ({written.stat().st_size / 1e6:.1f} MB)"
        )
        print(f"Merged in {elapsed:.1f}s")
        return

    roots = _load_roots(args)
    summary, written = run_census(
        roots,
        args.output,
        fmt=args.format,
        with_metrics=not args.no_metrics,
        with_forces=not args.no_forces,
        workers=args.workers,
        hours_cutoff=args.hours_cutoff,
        neighbor_cutoff=args.neighbor_cutoff,
        recompute=args.recompute,
        limit=args.debug,
        chunk_size=args.chunk_size,
    )
    _print_report(summary, top=args.top)

    elapsed = time.time() - started
    rate = summary.total / elapsed if elapsed > 0 else 0.0
    print(
        f"\nWrote {summary.total:,} rows x {len(_FIELDS)} columns to {written}"
        f"  ({written.stat().st_size / 1e6:.1f} MB)"
    )
    print(f"Scanned in {elapsed:.1f}s ({rate:.0f} jobs/s, {args.workers} workers)")


if __name__ == "__main__":
    main()
