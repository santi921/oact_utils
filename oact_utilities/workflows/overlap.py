"""Cross-corpus ORCA input overlap check (read-only, no database).

Given two root job directories, walks every job directory (immediate
subdirectory) under each root, parses the ORCA input file's coordinate block
(``* xyz <charge> <mult>`` ... ``*``), and reports which structures appear in
one root, the other, or both. Use it to confirm two campaigns are not
repeating calculations.

Two structures are "the same" when charge, spin multiplicity, and geometry
match. Geometry is compared translation-invariantly: coordinates are centered
on their centroid, rounded to ``--decimals`` decimal places (default 3, i.e.
1e-3 Angstrom), and the atom lines are sorted so atom ordering does not
matter. Rotated/permuted-but-rotated duplicates are NOT detected -- this is an
exact-source check aimed at catching the same CSV row submitted twice, not a
conformer search. A looser formula-level key (element counts + charge + spin)
is also tallied to flag "same composition, different geometry" near-misses.

Inputs are discovered as direct children of each job dir: ``orca.inp`` is
preferred, then any ``*.inp`` / ``*.inp.gz`` (gzipped quacc outputs are
handled). Job dirs with no parseable input are counted and reported, never
silently dropped.

Usage:
    python -m oact_utilities.workflows.overlap jobs_a/ jobs_b/
    python -m oact_utilities.workflows.overlap jobs_a/ jobs_b/ --csv overlap.csv
    python -m oact_utilities.workflows.overlap jobs_a/ jobs_b/ --decimals 4 --top 40
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# ORCA input parsing
# ---------------------------------------------------------------------------


@dataclass
class ParsedStructure:
    """Structure identity parsed from one job directory's ORCA input."""

    job_dir: Path
    inp_path: Path
    charge: int
    mult: int
    natoms: int
    formula: str
    geom_key: str  # charge + mult + canonical geometry hash
    formula_key: str  # charge + mult + element counts (no geometry)


def _read_text(path: Path) -> str:
    """Read a possibly-gzipped text file, replacing undecodable bytes."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as fh:
            return fh.read()
    return path.read_text(errors="replace")


def find_input_file(job_dir: Path) -> Path | None:
    """Locate the ORCA input file among a job directory's direct children.

    Prefers ``orca.inp`` / ``orca.inp.gz`` (what submit_jobs writes / quacc
    gzips), then falls back to the alphabetically first ``*.inp`` /
    ``*.inp.gz``. Only direct children are considered so per-step Sella
    replays in subdirectories are not mistaken for the job's own input.
    """
    try:
        names = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    except OSError:
        return None
    for preferred in ("orca.inp", "orca.inp.gz"):
        if preferred in names:
            return job_dir / preferred
    for name in names:
        if name.endswith((".inp", ".inp.gz")):
            return job_dir / name
    return None


def parse_coordinate_block(
    text: str,
) -> tuple[int, int, list[tuple[str, float, float, float]]] | None:
    """Parse the ``* xyz <charge> <mult>`` block from ORCA input text.

    Returns ``(charge, mult, atoms)`` where atoms is a list of
    ``(element, x, y, z)`` tuples, or None if no inline coordinate block is
    found (e.g. ``* xyzfile`` inputs) or it cannot be parsed.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "*" and parts[1].lower() == "xyz":
            try:
                charge, mult = int(parts[2]), int(parts[3])
            except ValueError:
                return None
            atoms: list[tuple[str, float, float, float]] = []
            for atom_line in lines[i + 1 :]:
                stripped = atom_line.strip()
                if stripped == "*":
                    return (charge, mult, atoms) if atoms else None
                if not stripped:
                    continue
                fields = stripped.split()
                if len(fields) < 4:
                    return None
                try:
                    atoms.append(
                        (
                            fields[0],
                            float(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                        )
                    )
                except ValueError:
                    return None
            return None  # block never closed with '*'
    return None


def canonicalize(
    charge: int,
    mult: int,
    atoms: list[tuple[str, float, float, float]],
    decimals: int,
) -> tuple[str, str, str]:
    """Build the geometry key, formula key, and formula string for a structure.

    Geometry is centered on its centroid and rounded to ``decimals`` places,
    then atom lines are sorted, so the key is invariant to translation and
    atom ordering (not rotation). Returns ``(geom_key, formula_key, formula)``.
    """
    n = len(atoms)
    cx = sum(a[1] for a in atoms) / n
    cy = sum(a[2] for a in atoms) / n
    cz = sum(a[3] for a in atoms) / n

    def fmt(v: float) -> str:
        r = round(v, decimals)
        if r == 0.0:
            r = 0.0  # normalize -0.0
        return f"{r:.{decimals}f}"

    atom_lines = sorted(
        f"{el} {fmt(x - cx)} {fmt(y - cy)} {fmt(z - cz)}" for el, x, y, z in atoms
    )
    counts = Counter(el for el, _, _, _ in atoms)
    formula = "".join(
        f"{el}{cnt}" if cnt > 1 else el for el, cnt in sorted(counts.items())
    )
    geom_blob = f"q={charge} m={mult}\n" + "\n".join(atom_lines)
    geom_key = hashlib.sha1(geom_blob.encode()).hexdigest()[:16]
    formula_key = f"{formula}|q={charge}|m={mult}"
    return geom_key, formula_key, formula


def parse_job_dir(job_dir: Path, decimals: int) -> ParsedStructure | str:
    """Parse one job directory into a ParsedStructure.

    Returns a skip-reason string instead when no input is found or the
    coordinate block cannot be parsed.
    """
    inp = find_input_file(job_dir)
    if inp is None:
        return "no_inp"
    try:
        text = _read_text(inp)
    except OSError:
        return "unreadable_inp"
    parsed = parse_coordinate_block(text)
    if parsed is None:
        return "unparseable_block"
    charge, mult, atoms = parsed
    geom_key, formula_key, formula = canonicalize(charge, mult, atoms, decimals)
    return ParsedStructure(
        job_dir=job_dir,
        inp_path=inp,
        charge=charge,
        mult=mult,
        natoms=len(atoms),
        formula=formula,
        geom_key=geom_key,
        formula_key=formula_key,
    )


# ---------------------------------------------------------------------------
# Per-root scan
# ---------------------------------------------------------------------------


@dataclass
class RootScan:
    """All parsed structures under one root, plus skip accounting."""

    root: Path
    structures: list[ParsedStructure]
    skips: Counter  # skip reason -> count

    @property
    def by_geom(self) -> dict[str, list[ParsedStructure]]:
        """Map geometry key -> the job dirs under this root carrying it."""
        out: dict[str, list[ParsedStructure]] = defaultdict(list)
        for s in self.structures:
            out[s.geom_key].append(s)
        return out


def scan_root(
    root: Path, decimals: int, workers: int, limit: int | None, label: str
) -> RootScan:
    """Parse every job directory (immediate subdirectory) under ``root``."""
    root = root.resolve()
    job_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    if limit is not None:
        job_dirs = job_dirs[:limit]

    scan = RootScan(root=root, structures=[], skips=Counter())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_job_dir, jd, decimals): jd for jd in job_dirs}
        completed = as_completed(futures)
        if tqdm is not None:
            completed = tqdm(
                completed, total=len(futures), desc=f"Parsing {label}", unit="job"
            )
        for fut in completed:
            result = fut.result()
            if isinstance(result, str):
                scan.skips[result] += 1
            else:
                scan.structures.append(result)
    scan.structures.sort(key=lambda s: s.job_dir.name)
    return scan


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` (fall back to name)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _print_root_summary(scan: RootScan, label: str) -> None:
    """Print job/parse/unique counts and internal duplicates for one root."""
    geom_map = scan.by_geom
    dup_groups = {k: v for k, v in geom_map.items() if len(v) > 1}
    n_extra = sum(len(v) - 1 for v in dup_groups.values())
    print(f"\n--- Root {label}: {scan.root} ---")
    print(f"  Job directories parsed: {len(scan.structures)}")
    for reason, cnt in sorted(scan.skips.items()):
        print(f"  Skipped ({reason}): {cnt}")
    print(f"  Unique structures (geometry+charge+spin): {len(geom_map)}")
    print(
        f"  Internal duplicates: {len(dup_groups)} groups "
        f"({n_extra} redundant job dirs)"
    )
    for group in sorted(dup_groups.values(), key=len, reverse=True)[:10]:
        dirs = ", ".join(_rel(s.job_dir, scan.root) for s in group)
        print(f"    [{group[0].formula} q={group[0].charge} m={group[0].mult}] {dirs}")
    if len(dup_groups) > 10:
        print(f"    ... and {len(dup_groups) - 10} more duplicate groups")


def print_report(scan_a: RootScan, scan_b: RootScan, top: int) -> set[str]:
    """Print the full comparison report; return the overlapping geometry keys."""
    print(f"\n{'=' * 70}")
    print("  ORCA Input Structure Overlap Report")
    print(f"{'=' * 70}")

    _print_root_summary(scan_a, "A")
    _print_root_summary(scan_b, "B")

    geom_a, geom_b = scan_a.by_geom, scan_b.by_geom
    overlap = set(geom_a) & set(geom_b)
    only_a = set(geom_a) - overlap
    only_b = set(geom_b) - overlap

    print("\n--- Cross-root comparison (geometry+charge+spin) ---")
    print(f"  Unique to A:     {len(only_a)}")
    print(f"  Unique to B:     {len(only_b)}")
    print(f"  In BOTH (overlapping calculations): {len(overlap)}")

    if overlap:
        print(f"\n--- Overlapping structures (showing up to {top}) ---")
        print("  formula | charge | mult | natoms | A job dir(s) | B job dir(s)")
        shown = sorted(overlap, key=lambda k: geom_a[k][0].job_dir.name)[:top]
        for key in shown:
            rep = geom_a[key][0]
            a_dirs = ";".join(_rel(s.job_dir, scan_a.root) for s in geom_a[key])
            b_dirs = ";".join(_rel(s.job_dir, scan_b.root) for s in geom_b[key])
            print(
                f"  {rep.formula} | {rep.charge} | {rep.mult} | {rep.natoms} | "
                f"{a_dirs} | {b_dirs}"
            )
        if len(overlap) > top:
            print(f"  ... and {len(overlap) - top} more (use --csv for the full list)")

    # Same composition + charge + spin but different geometry: not the same
    # calculation, but worth eyeballing when hunting duplicates.
    formula_a = {s.formula_key for s in scan_a.structures}
    formula_b = {s.formula_key for s in scan_b.structures}
    overlap_formulas = {geom_a[k][0].formula_key for k in overlap}
    near_miss = (formula_a & formula_b) - overlap_formulas
    print(
        f"\n  Formula-level near-misses (same formula+charge+spin in both roots, "
        f"different geometry): {len(near_miss)}"
    )
    return overlap


def write_csv(
    scan_a: RootScan, scan_b: RootScan, overlap: set[str], csv_path: Path
) -> None:
    """Write one row per parsed structure across both roots."""
    header = [
        "root",
        "job_dir",
        "inp_file",
        "formula",
        "charge",
        "mult",
        "natoms",
        "geom_key",
        "in_both",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, scan in (("A", scan_a), ("B", scan_b)):
            for s in scan.structures:
                writer.writerow(
                    [
                        label,
                        _rel(s.job_dir, scan.root),
                        s.inp_path.name,
                        s.formula,
                        s.charge,
                        s.mult,
                        s.natoms,
                        s.geom_key,
                        int(s.geom_key in overlap),
                    ]
                )
    print(f"\nPer-structure table written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the two-root ORCA input overlap check."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.overlap",
        description=(
            "Compare the ORCA input structures (orca.inp coordinate blocks) in "
            "two root job directories and report unique vs. overlapping "
            "calculations. Read-only: no database, no deletion."
        ),
    )
    parser.add_argument("root_a", type=Path, help="First root of job directories")
    parser.add_argument("root_b", type=Path, help="Second root of job directories")
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        metavar="N",
        help="Coordinate rounding for geometry matching, in decimal places "
        "of Angstrom (default: 3)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a per-structure CSV (both roots, with overlap flag)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Max overlapping structures to print (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel parse workers (default: 8)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N job directories per root for testing",
    )
    args = parser.parse_args()

    for root in (args.root_a, args.root_b):
        if not root.is_dir():
            print(f"Error: root directory not found: {root}", file=sys.stderr)
            sys.exit(1)

    scan_a = scan_root(args.root_a, args.decimals, args.workers, args.debug, "A")
    scan_b = scan_root(args.root_b, args.decimals, args.workers, args.debug, "B")
    overlap = print_report(scan_a, scan_b, top=args.top)
    if args.csv is not None:
        write_csv(scan_a, scan_b, overlap, args.csv)


if __name__ == "__main__":
    main()
