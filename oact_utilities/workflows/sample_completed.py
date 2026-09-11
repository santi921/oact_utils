"""Sample completed structures from one job-directory root, without a workflow DB.

Walks the immediate subdirectories of a single root (the same crawl
``census.py`` performs), keeps the jobs whose ORCA output terminated normally
(or whose Sella optimisation converged), filters them by atom count and metal
centre, draws a random sample, and writes the final geometry of each pick to
either a fresh workflow DB (``.db``; rows are ``to_run`` so the set can go
straight back through ``submit_jobs``) or an extended-XYZ file (``.xyz``).

Composition, charge, and spin come from ``orca.inp``. The final geometry is the
first of these that exists and whose atom list equals the input's:

1. ``orca.xyz`` (or ``<input stem>.xyz``): the optimised geometry an ORCA
   ``OPT`` run or ``sella_runner`` writes. ``*_trj.xyz`` trajectories and any
   other ``.xyz`` in the directory are ignored.
2. ``orca.engrad``: the geometry at which the last gradient was evaluated,
   which for a single point is the input geometry. Stored in Bohr; converted.
3. ``orca.inp``: the input geometry, for a single point that wrote no engrad.

The size and metal filters run before the termination check because parsing
``orca.inp`` is far cheaper than tailing an ``orca.out.gz``, so the
``not_completed`` count in the report covers only size- and metal-eligible
directories, not the whole root.

Usage:
    python -m oact_utilities.workflows.sample_completed jobs/ -o sample.db -n 200
    python -m oact_utilities.workflows.sample_completed jobs/ -o sample.db -n 150 \\
        --even-actinides --min-atoms 10 --max-atoms 80
    python -m oact_utilities.workflows.sample_completed jobs/ -o sample.xyz -n 50 \\
        --metals U Np
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from ..utils.architector import _init_db, _insert_row
from ..utils.status import check_job_termination
from .census import (
    BOHR_TO_ANG,
    _open_text,
    _parse_orig_index,
    iter_job_dirs,
    metal_class,
    parse_engrad,
    parse_inp,
    pick_metal,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# A completed job is recognised from content (ORCA TERMINATED NORMALLY, or a
# CONVERGED sella_status.txt), so the running-vs-timeout threshold never changes
# which directories this tool keeps. Any value works; match the census default.
_HOURS_CUTOFF = 24

_METAL_CLASS_ACTINIDE = "actinide"

# ORCA puts the final energy on the .xyz comment line:
# "Coordinates from ORCA-job orca E -670.534993289315"
_XYZ_ENERGY_RE = re.compile(r"\bE\s+(-?\d+\.\d+)")

# Columns the output workflow DB carries beyond the standard structures schema.
EXTRA_COLUMNS: dict[str, str] = {
    "metal": "TEXT",
    "metal_class": "TEXT",
    "geometry_source": "TEXT",
    "source_job_dir": "TEXT",
    "source_orig_index": "INTEGER",
    "ref_final_energy": "REAL",
}

# Why a directory was or was not kept, in report order.
REASON_OK = "candidate"
REASON_NOT_COMPLETED = "not_completed"
REASON_SIZE = "size_filter"
REASON_METAL = "metal_filter"
REASON_NO_INPUT = "no_input"
_REASON_ORDER = (
    REASON_OK,
    REASON_NOT_COMPLETED,
    REASON_SIZE,
    REASON_METAL,
    REASON_NO_INPUT,
)

_INP_NAMES = ("orca.inp", "orca.inp.gz")
_INP_SUFFIXES = (".inp", ".inp.gz")
_ENGRAD_NAMES = ("orca.engrad", "orca.engrad.gz")
_ENGRAD_SUFFIXES = (".engrad", ".engrad.gz")


@dataclass
class Candidate:
    """A completed job that passed the size and metal filters."""

    job_dir: Path
    symbols: list[str]
    charge: int | None
    spin: int | None
    metal: str | None
    metal_class: str | None
    source_orig_index: int | None

    @property
    def natoms(self) -> int:
        """Atom count as ORCA saw it (metal centre included)."""
        return len(self.symbols)


@dataclass
class Sample(Candidate):
    """A candidate whose final geometry was recovered."""

    coords: list[list[float]] = field(default_factory=list)
    geometry_source: str = ""
    ref_final_energy: float | None = None


# ---------------------------------------------------------------------------
# File discovery and geometry readers
# ---------------------------------------------------------------------------


def _pick_file(
    names: list[str], preferred: tuple[str, ...], suffixes: tuple[str, ...]
) -> str | None:
    """Return the first of ``preferred`` in ``names``, else the first name ending in ``suffixes``."""
    for name in preferred:
        if name in names:
            return name
    return next((n for n in sorted(names) if n.endswith(suffixes)), None)


def _find_final_xyz(names: list[str], inp_name: str) -> str | None:
    """Locate the final-geometry XYZ that ORCA or ``sella_runner`` wrote.

    ORCA names it after the input file (``orca.inp`` -> ``orca.xyz``,
    ``AmO_orca.inp`` -> ``AmO_orca.xyz``) and ``sella_runner`` always writes
    ``orca.xyz``. Any other ``.xyz`` (a ``*_trj.xyz`` trajectory, a starting
    structure someone dropped in) is not the final geometry and is ignored.
    """
    stem = inp_name[: -len(".gz")] if inp_name.endswith(".gz") else inp_name
    stem = stem[: -len(".inp")]
    for name in ("orca.xyz", "orca.xyz.gz", f"{stem}.xyz", f"{stem}.xyz.gz"):
        if name in names:
            return name
    return None


def _parse_coordinate_lines(
    lines: Iterable[str],
) -> tuple[list[str], list[list[float]]]:
    """Split ``El x y z`` lines into symbols and coordinates, skipping junk lines."""
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        # ORCA allows "C(1)" style labels and "C:" dummy-atom markers.
        symbol = parts[0].split("(")[0].split(":")[0]
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue
        symbols.append(symbol)
        coords.append(xyz)
    return symbols, coords


def read_xyz_geometry(path: Path) -> tuple[list[str], list[list[float]], float | None]:
    """Read a standard XYZ file (count, comment, coordinates); gz-transparent.

    Args:
        path: ``.xyz`` or ``.xyz.gz`` file.

    Returns:
        ``(symbols, coords_ang, energy_eh)``. ``energy_eh`` is parsed from an
        ORCA-style ``E <value>`` comment when present. Empty lists when the
        file is missing or its atom count does not match its body.
    """
    try:
        with _open_text(path) as f:
            lines = list(f)
    except OSError:
        return [], [], None
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    try:
        natoms = int(lines[idx].strip())
        comment = lines[idx + 1]
    except (IndexError, ValueError):
        return [], [], None
    symbols, coords = _parse_coordinate_lines(lines[idx + 2 : idx + 2 + natoms])
    if len(symbols) != natoms:
        return [], [], None
    match = _XYZ_ENERGY_RE.search(comment)
    return symbols, coords, float(match.group(1)) if match else None


def read_inp_geometry(path: Path) -> tuple[list[str], list[list[float]]]:
    """Read the ``* xyz C M`` coordinate block of an ORCA input; gz-transparent.

    ``census.parse_inp`` returns the element list only; this is the companion
    that also keeps the coordinates, used as the last-resort geometry source
    for a single point that wrote no engrad.

    Args:
        path: ``orca.inp`` or ``orca.inp.gz``.

    Returns:
        ``(symbols, coords_ang)``; empty on a missing file or no block.
    """
    body: list[str] = []
    try:
        with _open_text(path) as f:
            in_coords = False
            for line in f:
                stripped = line.strip()
                if in_coords:
                    if stripped.startswith("*"):
                        break
                    body.append(stripped)
                elif stripped.startswith(("* xyz", "*xyz")):
                    in_coords = True
    except OSError:
        return [], []
    return _parse_coordinate_lines(body)


def final_geometry(
    job_dir: Path, symbols: list[str]
) -> tuple[str, list[list[float]], float | None] | None:
    """Recover the final geometry of a completed job.

    Tries, in order, the ORCA/sella ``.xyz`` output, the ``.engrad``
    coordinates (Bohr, converted to Angstrom), and the ``orca.inp`` block. A
    source is accepted only when its atom list equals ``symbols`` (the input's),
    so a stale or foreign file can never be paired with the wrong composition.

    Args:
        job_dir: The completed job directory.
        symbols: Element symbols from ``orca.inp``, in order.

    Returns:
        ``(source, coords_ang, energy_eh)`` with ``source`` one of ``"xyz"``,
        ``"engrad"``, ``"inp"``; None when no source agrees with ``symbols``.
        ``energy_eh`` is the engrad energy when the engrad matched, else the
        ``E`` value on an ORCA ``.xyz`` comment line, else None.
    """
    try:
        names = os.listdir(job_dir)
    except OSError:
        return None

    engrad: dict = {}
    engrad_name = _pick_file(names, _ENGRAD_NAMES, _ENGRAD_SUFFIXES)
    if engrad_name is not None:
        engrad = parse_engrad(job_dir / engrad_name)
    engrad_ok = bool(engrad) and engrad.get("symbols") == symbols
    energy: float | None = engrad.get("energy") if engrad_ok else None

    inp_name = _pick_file(names, _INP_NAMES, _INP_SUFFIXES)

    if inp_name is not None:
        xyz_name = _find_final_xyz(names, inp_name)
        if xyz_name is not None:
            xyz_symbols, coords, xyz_energy = read_xyz_geometry(job_dir / xyz_name)
            if xyz_symbols == symbols:
                return "xyz", coords, energy if energy is not None else xyz_energy

    if engrad_ok:
        flat = engrad.get("coords_bohr") or []
        if len(flat) == 3 * len(symbols):
            coords = [
                [
                    flat[3 * i] * BOHR_TO_ANG,
                    flat[3 * i + 1] * BOHR_TO_ANG,
                    flat[3 * i + 2] * BOHR_TO_ANG,
                ]
                for i in range(len(symbols))
            ]
            return "engrad", coords, energy

    if inp_name is not None:
        inp_symbols, coords = read_inp_geometry(job_dir / inp_name)
        if inp_symbols == symbols:
            return "inp", coords, energy

    return None


# ---------------------------------------------------------------------------
# Candidate scan
# ---------------------------------------------------------------------------


def scan_candidate(
    job_dir: Path,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    metals: frozenset[str] | None = None,
    actinides_only: bool = False,
) -> tuple[Candidate | None, str]:
    """Classify one job directory, returning ``(candidate, reason)``.

    ``candidate`` is set only when ``reason == REASON_OK``. Composition is read
    from ``orca.inp`` and the filters run before the termination check, which
    is the expensive step: tailing ``orca.out`` costs a full decompress on a
    gzipped quacc directory.

    Args:
        job_dir: The job directory.
        min_atoms: Keep only ``natoms >= min_atoms`` (metal centre included).
        max_atoms: Keep only ``natoms <= max_atoms``.
        metals: Keep only structures whose metal centre is in this set.
        actinides_only: Keep only actinide-centred structures.

    Returns:
        ``(Candidate, REASON_OK)`` for a kept job, else ``(None, reason)``.
    """
    try:
        names = os.listdir(job_dir)
    except OSError:
        return None, REASON_NO_INPUT
    inp_name = _pick_file(names, _INP_NAMES, _INP_SUFFIXES)
    if inp_name is None:
        return None, REASON_NO_INPUT
    inp = parse_inp(job_dir / inp_name)
    symbols: list[str] = inp["symbols"]
    if not symbols:
        return None, REASON_NO_INPUT

    natoms = len(symbols)
    if min_atoms is not None and natoms < min_atoms:
        return None, REASON_SIZE
    if max_atoms is not None and natoms > max_atoms:
        return None, REASON_SIZE

    metal = pick_metal(symbols)
    mclass = metal_class(metal)
    if actinides_only and mclass != _METAL_CLASS_ACTINIDE:
        return None, REASON_METAL
    if metals is not None and metal not in metals:
        return None, REASON_METAL

    try:
        completed = check_job_termination(str(job_dir), hours_cutoff=_HOURS_CUTOFF) == 1
    except OSError:
        completed = False
    if not completed:
        return None, REASON_NOT_COMPLETED

    return (
        Candidate(
            job_dir=job_dir,
            symbols=symbols,
            charge=inp["charge"],
            spin=inp["spin"],
            metal=metal,
            metal_class=mclass,
            source_orig_index=_parse_orig_index(job_dir.name),
        ),
        REASON_OK,
    )


def collect_candidates(
    root: Path,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    metals: frozenset[str] | None = None,
    actinides_only: bool = False,
    workers: int = 8,
    limit: int | None = None,
) -> tuple[list[Candidate], Counter]:
    """Scan every immediate subdirectory of ``root`` for eligible completed jobs.

    Args:
        root: Directory whose immediate subdirectories are job dirs.
        min_atoms: See :func:`scan_candidate`.
        max_atoms: See :func:`scan_candidate`.
        metals: See :func:`scan_candidate`.
        actinides_only: See :func:`scan_candidate`.
        workers: Parallel scan threads (the work is I/O bound).
        limit: Cap on directories scanned, for testing.

    Returns:
        ``(candidates, reasons)`` where ``reasons`` counts every directory by
        the ``REASON_*`` label it received.
    """
    dirs = [job_dir for _, job_dir in iter_job_dirs([root], limit=limit)]
    reasons: Counter = Counter()
    candidates: list[Candidate] = []

    def work(job_dir: Path) -> tuple[Candidate | None, str]:
        return scan_candidate(job_dir, min_atoms, max_atoms, metals, actinides_only)

    progress = (
        tqdm(total=len(dirs), desc="Scanning", unit="job") if tqdm is not None else None
    )
    # Submit in blocks so a large root does not build every future up front.
    block = max(1000, workers * 64)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(dirs), block):
                for candidate, reason in pool.map(work, dirs[start : start + block]):
                    reasons[reason] += 1
                    if candidate is not None:
                        candidates.append(candidate)
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return candidates, reasons


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def resolve_sample(candidate: Candidate) -> Sample | None:
    """Attach the final geometry to a candidate, or None if none agrees."""
    found = final_geometry(candidate.job_dir, candidate.symbols)
    if found is None:
        return None
    source, coords, energy = found
    return Sample(
        **vars(candidate),
        coords=coords,
        geometry_source=source,
        ref_final_energy=energy,
    )


def select_samples(
    candidates: list[Candidate],
    n_samples: int | None,
    even_actinides: bool = False,
    seed: int | None = 0,
    resolve: Callable[[Candidate], Sample | None] = resolve_sample,
) -> tuple[list[Sample], int]:
    """Draw the sample, replacing any pick whose geometry cannot be recovered.

    Plain mode shuffles the pool once and walks it until ``n_samples`` are
    resolved. ``even_actinides`` restricts the pool to actinide-centred
    structures, groups it by metal, and draws round-robin over the metals in
    symbol order, so every actinide present gets an equal share; a metal whose
    pool runs dry drops out of the rotation and the others keep filling the
    quota. Both modes are deterministic for a given ``seed``.

    Args:
        candidates: Output of :func:`collect_candidates`.
        n_samples: How many to draw; None takes every resolvable candidate.
        even_actinides: Equal per-actinide allocation (see above).
        seed: Seed for the shuffle; None gives a fresh random order.
        resolve: Geometry resolver, swappable for tests.

    Returns:
        ``(samples, n_unresolved)`` where ``n_unresolved`` counts candidates
        skipped because no geometry source agreed with their input.
    """
    rng = random.Random(seed)
    unresolved = 0
    selected: list[Sample] = []

    def full() -> bool:
        return n_samples is not None and len(selected) >= n_samples

    def draw(pool: list[Candidate]) -> Iterator[Sample]:
        nonlocal unresolved
        for candidate in pool:
            sample = resolve(candidate)
            if sample is None:
                unresolved += 1
                continue
            yield sample

    if even_actinides:
        groups: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            if (
                candidate.metal_class == _METAL_CLASS_ACTINIDE
                and candidate.metal is not None
            ):
                groups.setdefault(candidate.metal, []).append(candidate)
        for metal in sorted(groups):
            rng.shuffle(groups[metal])
        iters = {metal: draw(groups[metal]) for metal in sorted(groups)}
        active = list(iters)
        while active and not full():
            for metal in list(active):
                if full():
                    break
                try:
                    selected.append(next(iters[metal]))
                except StopIteration:
                    active.remove(metal)
    else:
        pool = list(candidates)
        rng.shuffle(pool)
        for sample in draw(pool):
            selected.append(sample)
            if full():
                break

    return selected, unresolved


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def geometry_block(symbols: list[str], coords: list[list[float]]) -> str:
    """Headerless XYZ block, the storage form the workflow DB parses most safely."""
    return "\n".join(
        f"{symbol}  {x:.8f}  {y:.8f}  {z:.8f}"
        for symbol, (x, y, z) in zip(symbols, coords)
    )


def write_workflow_db(samples: list[Sample], out_path: Path) -> None:
    """Write the sample as a fresh workflow DB with every row ``to_run``.

    ``orig_index`` is the position in the sample (unique, so it is safe in a
    ``job_{orig_index}`` directory pattern); the source job's own index, when
    its directory name carried one, is kept in ``source_orig_index``.

    Args:
        samples: Resolved samples.
        out_path: Path of the SQLite DB to create.
    """
    conn = _init_db(out_path, extra_columns=EXTRA_COLUMNS)
    try:
        for index, sample in enumerate(samples):
            _insert_row(
                conn,
                orig_index=index,
                elements=";".join(sample.symbols),
                natoms=sample.natoms,
                geometry=geometry_block(sample.symbols, sample.coords),
                status="to_run",
                charge=sample.charge,
                spin=sample.spin,
                extra_values={
                    "metal": sample.metal,
                    "metal_class": sample.metal_class,
                    "geometry_source": sample.geometry_source,
                    "source_job_dir": str(sample.job_dir),
                    "source_orig_index": sample.source_orig_index,
                    "ref_final_energy": sample.ref_final_energy,
                },
            )
        conn.commit()
    finally:
        conn.close()


def write_extxyz(samples: list[Sample], out_path: Path) -> None:
    """Write the sample as one extended-XYZ file, one frame per structure.

    Per-frame ``info`` carries ``sample_index``, ``charge``, ``spin``,
    ``metal``, ``metal_class``, ``geometry_source``, ``source_job_dir``,
    ``source_orig_index``, and ``ref_final_energy_eh``; keys with no value
    are omitted because extxyz cannot encode None.

    Args:
        samples: Resolved samples.
        out_path: Path of the ``.xyz`` / ``.extxyz`` file to create.
    """
    frames = []
    for index, sample in enumerate(samples):
        atoms = Atoms(symbols=sample.symbols, positions=sample.coords)
        info = {
            "sample_index": index,
            "charge": sample.charge,
            "spin": sample.spin,
            "metal": sample.metal,
            "metal_class": sample.metal_class,
            "geometry_source": sample.geometry_source,
            "source_job_dir": str(sample.job_dir),
            "source_orig_index": sample.source_orig_index,
            "ref_final_energy_eh": sample.ref_final_energy,
        }
        atoms.info.update({k: v for k, v in info.items() if v is not None})
        frames.append(atoms)
    ase_write(str(out_path), frames, format="extxyz")


def _print_report(
    root: Path,
    reasons: Counter,
    candidates: list[Candidate],
    samples: list[Sample],
    unresolved: int,
    elapsed: float,
) -> None:
    """Print the scan tally, the geometry-source split, and a per-metal table."""
    total = sum(reasons.values())
    print(f"\nScanned {total:,} job directories under {root} in {elapsed:.1f}s")
    for reason in _REASON_ORDER:
        if reasons[reason]:
            print(f"  {reason:<15}{reasons[reason]:>10,}")

    if not samples:
        return

    sources = Counter(s.geometry_source for s in samples)
    split = ", ".join(f"{k} {v:,}" for k, v in sorted(sources.items()))
    print(f"\nSelected {len(samples):,} structures (geometry from: {split})")
    if unresolved:
        print(
            f"  {unresolved:,} candidate(s) skipped: no geometry source agreed "
            "with orca.inp"
        )

    available = Counter(c.metal or "(none)" for c in candidates)
    picked = Counter(s.metal or "(none)" for s in samples)
    classes = {c.metal or "(none)": c.metal_class or "(none)" for c in candidates}
    natoms: dict[str, list[int]] = {}
    for sample in samples:
        natoms.setdefault(sample.metal or "(none)", []).append(sample.natoms)

    print(f"\n{'metal':<8}{'class':<14}{'available':>11}{'selected':>10}{'natoms':>14}")
    print("-" * 57)
    for metal in sorted(available, key=lambda m: (-picked[m], -available[m], m)):
        sizes = natoms.get(metal)
        size_txt = f"{min(sizes)}-{max(sizes)}" if sizes else "-"
        print(
            f"{metal:<8}{classes[metal]:<14}{available[metal]:>11,}"
            f"{picked[metal]:>10,}{size_txt:>14}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.sample_completed",
        description=(
            "Sample completed structures from a root of job directories and "
            "write their final geometries to a workflow DB (.db) or an "
            "extended-XYZ file (.xyz). Reads only the files on disk."
        ),
    )
    parser.add_argument(
        "root", help="Directory whose immediate subdirectories are job dirs"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output file: .db writes a workflow DB (rows to_run), "
        ".xyz/.extxyz writes extended XYZ",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help="How many structures to draw (default: every eligible one)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=None,
        metavar="N",
        help="Keep only structures with at least N atoms, metal included",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        metavar="N",
        help="Keep only structures with at most N atoms, metal included",
    )
    parser.add_argument(
        "--even-actinides",
        action="store_true",
        help="Restrict to actinide-centred structures and split the draw "
        "equally across the actinide elements present (round-robin; a metal "
        "that runs out yields its share to the others)",
    )
    parser.add_argument(
        "--metals",
        nargs="+",
        default=None,
        metavar="SYM",
        help="Keep only structures whose metal centre is one of these symbols "
        "(combines with --even-actinides)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed; the same seed on the same root reproduces the "
        "sample (default: 0)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel scan threads (default: 8)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Scan only the first N job directories, for testing",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    args = parser.parse_args(argv)

    root = Path(os.path.expanduser(args.root))
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    out: Path = args.output
    if out.suffix not in (".db", ".xyz", ".extxyz"):
        parser.error("output must end in .db (workflow DB) or .xyz/.extxyz")
    if out.exists():
        if not args.overwrite:
            parser.error(f"{out} exists; pass --overwrite to replace it")
        out.unlink()
    if args.n_samples is not None and args.n_samples < 1:
        parser.error("--n-samples must be at least 1")
    if (
        args.min_atoms is not None
        and args.max_atoms is not None
        and args.min_atoms > args.max_atoms
    ):
        parser.error("--min-atoms is larger than --max-atoms")

    started = time.time()
    candidates, reasons = collect_candidates(
        root,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        metals=frozenset(args.metals) if args.metals else None,
        actinides_only=args.even_actinides,
        workers=args.workers,
        limit=args.debug,
    )
    samples, unresolved = select_samples(
        candidates,
        args.n_samples,
        even_actinides=args.even_actinides,
        seed=args.seed,
    )
    _print_report(root, reasons, candidates, samples, unresolved, time.time() - started)

    if not samples:
        print("Error: no completed structures matched the filters", file=sys.stderr)
        return 1
    if args.n_samples is not None and len(samples) < args.n_samples:
        print(
            f"Warning: only {len(samples):,} of the requested {args.n_samples:,} "
            "structures were available",
            file=sys.stderr,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".db":
        write_workflow_db(samples, out)
    else:
        write_extxyz(samples, out)
    print(f"\nWrote {len(samples):,} structures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
