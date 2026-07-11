"""One-time merge of two ORCA job-directory roots: SRC (A) into DEST (B).

Walks every job directory (immediate subdirectory) under both roots,
classifies each from on-disk content alone (no database) using the same
checks as inventory.py / the dashboard, then merges A into B by directory
name. B is the final home: after --execute, every kept job lives under B.

Rules:

* Final-state jobs (completed / failed) are always kept.
  - failed jobs receive ``.do_not_rerun.json`` (same file clean.py writes)
    once they land in B. This blocks resubmission by submit_jobs and makes
    dashboard --update mark the row FAILED, which is exactly the desired end
    state. Contents are KEPT.
  - completed jobs get NO marker: a normally-terminated orca.out is its own
    proof of completion. (``.do_not_rerun.json`` in particular must never be
    written to a completed dir -- the dashboard treats it as an override and
    would flip the row to FAILED on the next ``--update``.)
* Incomplete jobs (to_run / running / timeout) are kept by default; pass
  ``--drop-incomplete`` to delete them from BOTH roots instead.
* Name collisions (same job dir name in A and B) are cross-checked with
  overlap.py's canonical structure hash (charge + spin + geometry) before
  anything is deleted -- same name is NOT assumed to mean same molecule.
  The hash is taken from the INPUT FILE echo at the top of orca.out when
  available (the authoritative record of what actually ran, immune to a
  swapped-in inp), falling back to orca.inp otherwise. A copy whose inp and
  out-echo hashes DISAGREE (misplaced input file?) is never resolved
  automatically. Only when both sides' hashes are known and match is the
  collision resolved: completed beats failed beats incomplete; within the
  same tier the copy with the newer files (max mtime inside the dir) wins;
  exact ties keep B. The losing copy is deleted on --execute. When hashes
  differ or cannot be verified, BOTH copies are left untouched and reported
  for manual resolution (exception: with --drop-incomplete, two incomplete
  copies are still dropped -- that removal is status-based, not dedup).
* An entirely-empty folder (no file with content) colliding with a non-empty
  one is superseded by the non-empty copy -- nothing to verify, nothing to
  lose. Two empty folders keep B's.
* The report's input-file health table counts, per root: empty/missing/
  unparseable inp files, dirs whose empty inp could later be re-grafted from
  a duplicate (empty inp but other files valid), hashes recovered via the
  orca.out echo, and inp-vs-out disagreements.

Merging is by directory NAME (e.g. ``barfoot_BN4C6H8O2_q2_m1_idx8761``).
The same structure submitted under two different idx values is NOT collapsed
here -- use workflows/overlap.py to find those.

DRY RUN is the default: nothing is moved, deleted, or markered without
--execute. The report shows, per root, how many jobs would be kept / dropped
and how every collision would resolve.

Usage:
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT \\
        --drop-incomplete --csv plan.csv
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT --execute
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oact_utilities.workflows.clean import (
    _ORCA_ATOM_RE,
    MARKER_FILENAME,
    _extract_failure_info,
    is_marker_blocked,
)
from oact_utilities.workflows.inventory import (
    _STATUS_COMPLETED,
    _STATUS_FAILED,
    _STATUS_ORDER,
    _classify_job_status,
)
from oact_utilities.workflows.overlap import (
    _read_text,
    canonicalize,
    find_input_file,
    parse_coordinate_block,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

_FINAL_STATES = frozenset({_STATUS_COMPLETED, _STATUS_FAILED})

# Plan actions.
ACT_MOVE = "move_to_dest"  # A only: move A -> B
ACT_KEEP_DEST = "keep_dest"  # B only (or B wins): B stays
ACT_REPLACE_DEST = "replace_dest"  # collision, A wins: delete B copy, move A
ACT_DISCARD_SRC = "discard_src"  # collision, B wins: delete A copy
ACT_DROP_SRC = "drop_src"  # incomplete, dropped from A
ACT_DROP_DEST = "drop_dest"  # incomplete, dropped from B
ACT_DROP_BOTH = "drop_both"  # collision, both incomplete, dropped
ACT_SKIP_MISMATCH = "skip_hash_mismatch"  # collision, different/unverifiable
# structures under one name: BOTH copies left untouched, resolve manually

# Input-file states (why a geom_key can be None).
INP_OK = "ok"  # parsed and hashed
INP_MISSING = "missing"  # no *.inp file in the job dir
INP_EMPTY = "empty"  # *.inp exists but is empty (killed mid-write)
INP_UNPARSEABLE = "unparseable"  # *.inp has content but no valid coord block


@dataclass
class JobInfo:
    """One job directory: path, status, mtime, and structure hashes.

    ``inp_geom_key`` is overlap.py's canonical structure hash (charge + spin +
    centered/rounded/sorted geometry) of the orca.inp; ``inp_state`` says why
    it can be None (INP_* constants). ``out_geom_key`` is the same hash
    recovered from the INPUT FILE echo at the top of orca.out -- the
    authoritative record of what actually ran, immune to a swapped-in inp.
    ``n_nonempty_files`` counts files with content anywhere in the dir --
    zero means the folder is effectively empty (nothing recoverable).
    """

    path: Path
    status: str
    newest_mtime: float
    inp_geom_key: str | None
    inp_state: str
    out_geom_key: str | None
    n_nonempty_files: int

    @property
    def geom_key(self) -> str | None:
        """Effective structure hash: orca.out echo wins over orca.inp."""
        return self.out_geom_key if self.out_geom_key is not None else self.inp_geom_key

    @property
    def hash_source(self) -> str:
        """Where the effective hash came from: out / inp / none."""
        if self.out_geom_key is not None:
            return "out"
        if self.inp_geom_key is not None:
            return "inp"
        return "none"

    @property
    def self_inconsistent(self) -> bool:
        """True when inp and out-echo hashes both parse but DISAGREE.

        Strong signal that the orca.inp sitting in this folder is not the
        input this job actually ran (e.g. a file moved in from elsewhere).
        """
        return (
            self.inp_geom_key is not None
            and self.out_geom_key is not None
            and self.inp_geom_key != self.out_geom_key
        )

    @property
    def is_empty(self) -> bool:
        """True when the dir holds no file with any content."""
        return self.n_nonempty_files == 0

    @property
    def empty_inp_otherwise_valid(self) -> bool:
        """True for the graft-candidate pattern: empty inp, other real files."""
        return self.inp_state == INP_EMPTY and self.n_nonempty_files > 0


@dataclass
class PlanEntry:
    """Planned outcome for one job name across the two roots."""

    name: str
    info_a: JobInfo | None
    info_b: JobInfo | None
    action: str
    reason: str

    @property
    def survivor_status(self) -> str | None:
        """Status of the copy that ends up in DEST (None if dropped)."""
        if self.action in (ACT_MOVE, ACT_REPLACE_DEST):
            assert self.info_a is not None
            return self.info_a.status
        if self.action in (ACT_KEEP_DEST, ACT_DISCARD_SRC):
            assert self.info_b is not None
            return self.info_b.status
        return None


def _walk_stats(job_dir: Path) -> tuple[float, int]:
    """Return ``(newest_mtime, n_nonempty_files)`` for ``job_dir`` (recursive).

    Symlinks are skipped. mtime falls back to the directory's own mtime when
    it holds no stat-able files.
    """
    newest = 0.0
    n_nonempty = 0
    for dirpath, _dirnames, filenames in os.walk(job_dir, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if os.path.islink(fp):
                    continue
                st = os.stat(fp)
            except OSError:
                continue
            newest = max(newest, st.st_mtime)
            if st.st_size > 0:
                n_nonempty += 1
    if newest == 0.0:
        try:
            newest = job_dir.stat().st_mtime
        except OSError:
            pass
    return newest, n_nonempty


def _hash_text(text: str, decimals: int) -> str | None:
    """Canonical structure hash of ORCA input text, or None if unparseable."""
    parsed = parse_coordinate_block(text)
    if parsed is None:
        return None
    charge, mult, atoms = parsed
    geom_key, _formula_key, _formula = canonicalize(charge, mult, atoms, decimals)
    return geom_key


def _hash_input(job_dir: Path, decimals: int) -> tuple[str | None, str]:
    """Hash the job's ORCA input; return ``(geom_key, inp_state)``.

    Same discovery + canonical hash as overlap.py, but distinguishes WHY a
    hash is unavailable: missing file, empty file (killed mid-write), or
    content with no parseable coordinate block.
    """
    inp = find_input_file(job_dir)
    if inp is None:
        return None, INP_MISSING
    try:
        text = _read_text(inp)
    except OSError:
        return None, INP_MISSING
    if not text.strip():
        return None, INP_EMPTY
    key = _hash_text(text, decimals)
    return (key, INP_OK if key is not None else INP_UNPARSEABLE)


# ORCA echoes the full input file near the top of the .out as "|  N> <line>".
_ECHO_LINE_RE = re.compile(r"^\|\s*\d+>\s?(.*)$")
# Stop scanning for/collecting the echo after this many lines: the echo starts
# ~200 lines in, so a much larger cap only guards against degenerate files.
_ECHO_MAX_LINES = 20000


def _find_output_file(job_dir: Path) -> Path | None:
    """Locate the main ORCA output among a job dir's direct children.

    Prefers ``orca.out`` / ``orca.out.gz``, then the alphabetically first
    other ``*.out`` / ``*.out.gz``. Atomic-density sub-run outputs
    (``orca_atom*.out``) are never candidates -- their echoed input has no
    coordinate block.
    """
    try:
        names = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    except OSError:
        return None
    for preferred in ("orca.out", "orca.out.gz"):
        if preferred in names:
            return job_dir / preferred
    for name in names:
        if name.endswith((".out", ".out.gz")) and not _ORCA_ATOM_RE.match(name):
            return job_dir / name
    return None


def _parse_input_echo(out_path: Path) -> str | None:
    """Reconstruct the input file from the INPUT FILE echo in an ORCA output.

    Streams only the head of the file (echo sits right after the banner) and
    stops at ``****END OF INPUT****``. Returns the de-prefixed input text, or
    None when no echo lines are found (truncated/corrupt output).
    """
    opener = gzip.open if out_path.name.endswith(".gz") else open
    lines: list[str] = []
    try:
        with opener(out_path, "rt", errors="replace") as fh:  # type: ignore[operator]
            for i, line in enumerate(fh):
                if i > _ECHO_MAX_LINES:
                    break
                if "****END OF INPUT****" in line:
                    break
                m = _ECHO_LINE_RE.match(line)
                if m:
                    lines.append(m.group(1))
    except OSError:
        return None
    return "\n".join(lines) + "\n" if lines else None


def _hash_output_echo(job_dir: Path, decimals: int) -> str | None:
    """Structure hash from the orca.out input echo, or None if unavailable."""
    out = _find_output_file(job_dir)
    if out is None:
        return None
    text = _parse_input_echo(out)
    if text is None:
        return None
    return _hash_text(text, decimals)


def scan_root(
    root: Path, hours_cutoff: int, workers: int, label: str, decimals: int = 3
) -> dict[str, JobInfo]:
    """Classify every job directory (immediate subdirectory) under ``root``."""
    job_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())

    def scan_one(jd: Path) -> JobInfo:
        status, _code = _classify_job_status(jd, hours_cutoff)
        inp_geom_key, inp_state = _hash_input(jd, decimals)
        out_geom_key = _hash_output_echo(jd, decimals)
        newest_mtime, n_nonempty = _walk_stats(jd)
        return JobInfo(
            path=jd,
            status=status,
            newest_mtime=newest_mtime,
            inp_geom_key=inp_geom_key,
            inp_state=inp_state,
            out_geom_key=out_geom_key,
            n_nonempty_files=n_nonempty,
        )

    out: dict[str, JobInfo] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one, jd): jd for jd in job_dirs}
        completed = as_completed(futures)
        if tqdm is not None:
            completed = tqdm(
                completed, total=len(futures), desc=f"Classifying {label}", unit="job"
            )
        for fut in completed:
            info = fut.result()
            out[info.path.name] = info
    return out


def _tier(status: str) -> int:
    """Keep-priority tier: completed > failed > incomplete."""
    if status == _STATUS_COMPLETED:
        return 3
    if status == _STATUS_FAILED:
        return 2
    return 1


def _decide(
    info_a: JobInfo | None, info_b: JobInfo | None, drop_incomplete: bool
) -> tuple[str, str]:
    """Return ``(action, reason)`` for one job name."""
    if info_a is not None and info_b is None:
        if drop_incomplete and info_a.status not in _FINAL_STATES:
            return ACT_DROP_SRC, f"{info_a.status} in A only, --drop-incomplete"
        return ACT_MOVE, f"{info_a.status} in A only"
    if info_b is not None and info_a is None:
        if drop_incomplete and info_b.status not in _FINAL_STATES:
            return ACT_DROP_DEST, f"{info_b.status} in B only, --drop-incomplete"
        return ACT_KEEP_DEST, f"{info_b.status} in B only"

    assert info_a is not None and info_b is not None

    # Collision. An entirely-empty folder (no file with content) is superseded
    # by a non-empty one: there is nothing to hash-verify and nothing to lose.
    if info_a.is_empty or info_b.is_empty:
        if info_a.is_empty and info_b.is_empty:
            return ACT_DISCARD_SRC, "both folders empty, keep B"
        winner = info_b if info_a.is_empty else info_a
        if drop_incomplete and winner.status not in _FINAL_STATES:
            return (
                ACT_DROP_BOTH,
                f"{'A' if info_a.is_empty else 'B'} empty, non-empty side "
                f"{winner.status}; --drop-incomplete",
            )
        if info_a.is_empty:
            return ACT_DISCARD_SRC, "A folder empty, non-empty B supersedes"
        return ACT_REPLACE_DEST, "B folder empty, non-empty A supersedes"

    both_incomplete = (
        info_a.status not in _FINAL_STATES and info_b.status not in _FINAL_STATES
    )

    # A copy whose orca.inp disagrees with its own orca.out echo (misplaced
    # input file?) is never allowed to silently win or lose a collision --
    # skip and flag for manual inspection. Status-based --drop-incomplete of
    # two incomplete copies still applies.
    if info_a.self_inconsistent or info_b.self_inconsistent:
        if drop_incomplete and both_incomplete:
            return (
                ACT_DROP_BOTH,
                f"A={info_a.status} vs B={info_b.status}, both incomplete, "
                "--drop-incomplete (inp/out disagreement moot, dropped on status)",
            )
        sides = "+".join(
            s
            for s, i in (("A", info_a), ("B", info_b))
            if i is not None and i.self_inconsistent
        )
        return (
            ACT_SKIP_MISMATCH,
            f"A={info_a.status} vs B={info_b.status}, inp/out echo DISAGREE "
            f"in {sides} (misplaced orca.inp?); both copies kept, resolve manually",
        )

    # Never assume same name means same molecule: only resolve (i.e. delete
    # one copy) when both structure hashes are known AND match. The hash
    # comes from the orca.out input echo when available (what actually ran),
    # else the orca.inp. Exception: with --drop-incomplete, two incomplete
    # copies are both dropped regardless -- that removal is status-based.
    same_hash = info_a.geom_key is not None and info_a.geom_key == info_b.geom_key
    if not same_hash:
        if drop_incomplete and both_incomplete:
            return (
                ACT_DROP_BOTH,
                f"A={info_a.status} vs B={info_b.status}, both incomplete, "
                "--drop-incomplete (hashes differ but both dropped on status)",
            )
        hash_note = (
            "structure hashes DIFFER "
            f"(A from {info_a.hash_source}, B from {info_b.hash_source})"
            if info_a.geom_key is not None and info_b.geom_key is not None
            else "structure hash unverifiable "
            f"(A inp {info_a.inp_state}/out echo "
            f"{'ok' if info_a.out_geom_key else 'unavailable'}, "
            f"B inp {info_b.inp_state}/out echo "
            f"{'ok' if info_b.out_geom_key else 'unavailable'})"
        )
        return (
            ACT_SKIP_MISMATCH,
            f"A={info_a.status} vs B={info_b.status}, {hash_note}; "
            "both copies kept, resolve manually",
        )

    ta, tb = _tier(info_a.status), _tier(info_b.status)
    a_wins = (ta, info_a.newest_mtime) > (tb, info_b.newest_mtime)
    winner = info_a if a_wins else info_b
    reason = (
        f"A={info_a.status} vs B={info_b.status}, same structure hash, "
        f"{'A' if a_wins else 'B'} wins "
        f"({'higher tier' if ta != tb else 'newer files' if info_a.newest_mtime != info_b.newest_mtime else 'tie, B kept'})"
    )
    if drop_incomplete and winner.status not in _FINAL_STATES:
        return ACT_DROP_BOTH, reason + "; winner incomplete, --drop-incomplete"
    return (ACT_REPLACE_DEST if a_wins else ACT_DISCARD_SRC), reason


def build_plan(
    jobs_a: dict[str, JobInfo],
    jobs_b: dict[str, JobInfo],
    drop_incomplete: bool,
) -> list[PlanEntry]:
    """Build the per-job merge plan across the union of names in A and B."""
    plan: list[PlanEntry] = []
    for name in sorted(set(jobs_a) | set(jobs_b)):
        info_a, info_b = jobs_a.get(name), jobs_b.get(name)
        action, reason = _decide(info_a, info_b, drop_incomplete)
        plan.append(
            PlanEntry(
                name=name, info_a=info_a, info_b=info_b, action=action, reason=reason
            )
        )
    return plan


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def _marker_needed(entry: PlanEntry) -> tuple[Path, str] | None:
    """Return ``(dir_to_check, marker_filename)`` if the survivor needs one.

    Only FAILED survivors are markered -- a completed job's normally-terminated
    orca.out is its own proof of completion, so it gets no marker. The check
    runs against the CURRENT location of the surviving copy (A's dir when it
    has not moved yet), since the marker file travels with the directory on
    move.
    """
    if entry.survivor_status != _STATUS_FAILED:
        return None
    src_info = (
        entry.info_a if entry.action in (ACT_MOVE, ACT_REPLACE_DEST) else entry.info_b
    )
    assert src_info is not None
    if is_marker_blocked(src_info.path):
        return None
    return src_info.path, MARKER_FILENAME


def _write_failed_marker(job_dir: Path, source_root: Path) -> None:
    """Write ``.do_not_rerun.json`` into a failed ``job_dir`` (contents kept)."""
    data: dict[str, str | int | None] = {
        "generated_by": "python -m oact_utilities.scripts.merge_job_roots",
        "date": datetime.now(tz=timezone.utc).isoformat(),
        "detected_status": _STATUS_FAILED,
        "merged_from": str(source_root),
        **_extract_failure_info(job_dir),
    }
    (job_dir / MARKER_FILENAME).write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_plan(
    plan: list[PlanEntry], dest_root: Path, src_root: Path, execute: bool
) -> tuple[Counter, int, list[str]]:
    """Apply (or preview) the plan. Returns (action counts, markers, errors)."""
    counts: Counter = Counter()
    markers_written = 0
    errors: list[str] = []

    for entry in plan:
        counts[entry.action] += 1
        marker_info = _marker_needed(entry)

        if execute:
            try:
                if entry.action == ACT_MOVE:
                    assert entry.info_a is not None
                    shutil.move(str(entry.info_a.path), str(dest_root / entry.name))
                elif entry.action == ACT_REPLACE_DEST:
                    assert entry.info_a is not None and entry.info_b is not None
                    shutil.rmtree(str(entry.info_b.path))
                    shutil.move(str(entry.info_a.path), str(dest_root / entry.name))
                elif entry.action == ACT_DISCARD_SRC:
                    assert entry.info_a is not None
                    shutil.rmtree(str(entry.info_a.path))
                elif entry.action == ACT_DROP_SRC:
                    assert entry.info_a is not None
                    shutil.rmtree(str(entry.info_a.path))
                elif entry.action == ACT_DROP_DEST:
                    assert entry.info_b is not None
                    shutil.rmtree(str(entry.info_b.path))
                elif entry.action == ACT_DROP_BOTH:
                    assert entry.info_a is not None and entry.info_b is not None
                    shutil.rmtree(str(entry.info_a.path))
                    shutil.rmtree(str(entry.info_b.path))
            except (OSError, shutil.Error) as e:
                errors.append(f"{entry.name}: {entry.action} failed: {e}")
                continue

        if marker_info is not None:
            markers_written += 1
            if execute:
                origin = (
                    src_root
                    if entry.action in (ACT_MOVE, ACT_REPLACE_DEST)
                    else dest_root
                )
                try:
                    _write_failed_marker(dest_root / entry.name, origin)
                except (OSError, PermissionError) as e:
                    errors.append(f"{entry.name}: marker write failed: {e}")

    return counts, markers_written, errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _status_counts(jobs: dict[str, JobInfo]) -> Counter:
    """Status -> count for one scanned root."""
    return Counter(info.status for info in jobs.values())


def print_report(
    src_root: Path,
    dest_root: Path,
    jobs_a: dict[str, JobInfo],
    jobs_b: dict[str, JobInfo],
    plan: list[PlanEntry],
    counts: Counter,
    markers: int,
    errors: list[str],
    execute: bool,
    top: int,
) -> None:
    """Print the merge summary."""
    verb = "" if execute else "would be "
    print(f"\n{'=' * 70}")
    print("  Job Root Merge Report (A -> B)")
    print(f"{'=' * 70}")
    print(f"  Folder A (source): {src_root}  ({len(jobs_a)} job dirs)")
    print(f"  Folder B (dest):   {dest_root}  ({len(jobs_b)} job dirs)")

    print("\n--- On-disk status ---")
    print("  status | A | B")
    ca, cb = _status_counts(jobs_a), _status_counts(jobs_b)
    for status in _STATUS_ORDER:
        print(f"  {status} | {ca.get(status, 0)} | {cb.get(status, 0)}")

    # Per-root input-file health: why a hash can be missing, plus the
    # graft-candidate pattern (empty inp, everything else valid).
    print("\n--- Input-file health ---")
    print("  metric | A | B")
    health_rows = [
        ("inp parsed + hashed", lambda j: j.inp_state == INP_OK),
        ("inp EMPTY file", lambda j: j.inp_state == INP_EMPTY),
        ("inp missing", lambda j: j.inp_state == INP_MISSING),
        ("inp unparseable", lambda j: j.inp_state == INP_UNPARSEABLE),
        (
            "empty inp, other files valid (graft candidates)",
            lambda j: j.empty_inp_otherwise_valid,
        ),
        (
            "hash recovered from orca.out echo (inp unusable)",
            lambda j: j.inp_geom_key is None and j.out_geom_key is not None,
        ),
        (
            "inp/out echo DISAGREE (misplaced inp?)",
            lambda j: j.self_inconsistent,
        ),
        ("entirely EMPTY folders", lambda j: j.is_empty),
    ]
    for label, pred in health_rows:
        na = sum(1 for j in jobs_a.values() if pred(j))
        nb = sum(1 for j in jobs_b.values() if pred(j))
        print(f"  {label} | {na} | {nb}")

    # Hash cross-check over ALL name collisions, independent of the action
    # taken: same name does not imply same molecule. Categories are
    # mutually exclusive, checked in this order.
    collisions = [e for e in plan if e.info_a is not None and e.info_b is not None]
    n_empty = n_inconsistent = n_same_hash = n_diff_hash = n_unverifiable = 0
    for e in collisions:
        assert e.info_a is not None and e.info_b is not None
        if e.info_a.is_empty or e.info_b.is_empty:
            n_empty += 1
        elif e.info_a.self_inconsistent or e.info_b.self_inconsistent:
            n_inconsistent += 1
        elif e.info_a.geom_key is None or e.info_b.geom_key is None:
            n_unverifiable += 1
        elif e.info_a.geom_key == e.info_b.geom_key:
            n_same_hash += 1
        else:
            n_diff_hash += 1

    kept_from_a = counts[ACT_MOVE] + counts[ACT_REPLACE_DEST]
    kept_from_b = counts[ACT_KEEP_DEST] + counts[ACT_DISCARD_SRC]
    print("\n--- Plan ---")
    print(f"  Jobs {verb}kept from A (moved into B): {kept_from_a}")
    print(f"  Jobs {verb}kept from B (stay in place): {kept_from_b}")
    print(
        f"  Name collisions: {len(collisions)} "
        f"(A wins {counts[ACT_REPLACE_DEST]}, B wins {counts[ACT_DISCARD_SRC]}, "
        f"both dropped {counts[ACT_DROP_BOTH]}, "
        f"skipped {counts[ACT_SKIP_MISMATCH]})"
    )
    print("\n--- Collision structure cross-check (same name != same molecule) ---")
    print(f"  Same name, same structure hash:        {n_same_hash}")
    print(f"  Same name, DIFFERENT structure hash:   {n_diff_hash}")
    print(f"  Same name, hash unverifiable:          {n_unverifiable}")
    print(f"  Same name, inp/out echo disagreement:  {n_inconsistent}")
    print(f"  Empty folder vs non-empty (supersede): {n_empty}")
    if counts[ACT_SKIP_MISMATCH]:
        print(
            f"  -> {counts[ACT_SKIP_MISMATCH]} collision(s) left untouched in "
            "both roots (no deletion without a verified hash match)."
        )
    print(
        f"\n  Incomplete {verb}dropped from A: "
        f"{counts[ACT_DROP_SRC] + counts[ACT_DROP_BOTH]}"
    )
    print(
        f"  Incomplete {verb}dropped from B: "
        f"{counts[ACT_DROP_DEST] + counts[ACT_DROP_BOTH]}"
    )
    print(f"  Failed-job .do_not_rerun.json markers {verb}written: {markers}")

    if collisions:
        print(f"\n--- Collisions (showing up to {top}) ---")
        print("  name | action | reason")
        shown = sorted(collisions, key=lambda e: e.action != ACT_SKIP_MISMATCH)
        for e in shown[:top]:
            print(f"  {e.name} | {e.action} | {e.reason}")
        if len(collisions) > top:
            print(f"  ... and {len(collisions) - top} more (use --csv for all)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if not execute:
        print(
            "\n  DRY RUN -- nothing was moved, deleted, or markered. "
            "Re-run with --execute to apply."
        )


def write_csv(plan: list[PlanEntry], csv_path: Path) -> None:
    """Write the full per-job plan to CSV."""
    header = [
        "name",
        "status_a",
        "status_b",
        "newest_mtime_a",
        "newest_mtime_b",
        "geom_key_a",
        "geom_key_b",
        "hash_source_a",
        "hash_source_b",
        "inp_state_a",
        "inp_state_b",
        "action",
        "reason",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for e in plan:
            writer.writerow(
                [
                    e.name,
                    e.info_a.status if e.info_a else "",
                    e.info_b.status if e.info_b else "",
                    f"{e.info_a.newest_mtime:.0f}" if e.info_a else "",
                    f"{e.info_b.newest_mtime:.0f}" if e.info_b else "",
                    (e.info_a.geom_key or "") if e.info_a else "",
                    (e.info_b.geom_key or "") if e.info_b else "",
                    e.info_a.hash_source if e.info_a else "",
                    e.info_b.hash_source if e.info_b else "",
                    e.info_a.inp_state if e.info_a else "",
                    e.info_b.inp_state if e.info_b else "",
                    e.action,
                    e.reason,
                ]
            )
    print(f"\nPer-job plan written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the one-time A -> B job-root merge."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.scripts.merge_job_roots",
        description=(
            "Merge job root A into job root B (B is the final home). "
            "Final-state jobs (completed/failed) are kept, failed ones are "
            "markered with .do_not_rerun.json; "
            "collisions keep the better/newer copy. DB-blind, name-based. "
            "Dry run by default."
        ),
    )
    parser.add_argument("src_root", type=Path, help="Root A: jobs move FROM here")
    parser.add_argument("dest_root", type=Path, help="Root B: jobs live HERE after")
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Delete to_run/running/timeout jobs from BOTH roots instead of "
        "keeping them. CANNOT tell a live ORCA run from a dead one -- only "
        "use when nothing is executing.",
    )
    parser.add_argument(
        "--hours-cutoff",
        type=int,
        default=24,
        metavar="H",
        help="Hours of inactivity before a job with output but no termination "
        "is classified timeout instead of running (default: 24)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the full per-job plan to this CSV",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Max collisions to print (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel scan workers (default: 8)"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        metavar="N",
        help="Coordinate rounding for the collision input-hash cross-check, "
        "in decimal places of Angstrom (default: 3, matches overlap.py)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move/delete/marker (default: dry-run preview only)",
    )
    args = parser.parse_args()

    for root in (args.src_root, args.dest_root):
        if not root.is_dir():
            print(f"Error: root directory not found: {root}", file=sys.stderr)
            sys.exit(1)
    src_root = args.src_root.resolve()
    dest_root = args.dest_root.resolve()
    if src_root == dest_root:
        print("Error: SRC and DEST are the same directory", file=sys.stderr)
        sys.exit(1)

    mode = "EXECUTE" if args.execute else "DRY RUN (preview only)"
    print("!" * 70)
    print(f"  JOB ROOT MERGE [{mode}] -- DB-BLIND, name-based")
    print(f"  A (source): {src_root}")
    print(f"  B (dest):   {dest_root}")
    print("  Collision losers are DELETED on --execute, but only when both")
    print("  copies' orca.inp structure hashes match; mismatches are kept and")
    print("  reported. --drop-incomplete also deletes to_run/running/timeout")
    print("  jobs from both roots and cannot tell a live ORCA process from a")
    print("  dead one.")
    print("!" * 70)

    jobs_a = scan_root(src_root, args.hours_cutoff, args.workers, "A", args.decimals)
    jobs_b = scan_root(dest_root, args.hours_cutoff, args.workers, "B", args.decimals)
    plan = build_plan(jobs_a, jobs_b, args.drop_incomplete)
    counts, markers, errors = execute_plan(plan, dest_root, src_root, args.execute)
    print_report(
        src_root,
        dest_root,
        jobs_a,
        jobs_b,
        plan,
        counts,
        markers,
        errors,
        args.execute,
        args.top,
    )
    if args.csv is not None:
        write_csv(plan, args.csv)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
