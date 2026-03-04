"""Delete job folders for spin=1 jobs that were reset to to_run.

Reads the reset job list produced by the spin=1 reset operation
(format: one ``job_{orig_index},`` per line, with ``# <section>`` headers),
then deletes the corresponding directories so they are recreated cleanly
on the next submission run.

Usage
-----
    python -m oact_utilities.scripts.delete_reset_jobs \\
        data/oact_db_test/reset_spin1_jobs.txt \\
        --actinides-dir /path/to/actinides/jobs \\
        --non-actinides-dir /path/to/non_actinides/jobs

    # Preview without deleting
    python -m oact_utilities.scripts.delete_reset_jobs \\
        data/oact_db_test/reset_spin1_jobs.txt \\
        --actinides-dir /path/to/actinides/jobs \\
        --non-actinides-dir /path/to/non_actinides/jobs \\
        --dry-run

    # Save not-found paths to a file for debugging
    python -m oact_utilities.scripts.delete_reset_jobs \\
        data/oact_db_test/reset_spin1_jobs.txt \\
        --actinides-dir /path/to/actinides/jobs \\
        --non-actinides-dir /path/to/non_actinides/jobs \\
        --not-found-file missing_jobs.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_reset_list(txt_path: Path) -> dict[str, list[str]]:
    """Parse reset job list into a dict keyed by section name.

    Args:
        txt_path: Path to the reset jobs txt file.

    Returns:
        Dict mapping section label (e.g. 'actinides') to list of job folder
        names (e.g. ['job_0', 'job_1', ...]).
    """
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # e.g. "# actinides (4077 jobs)"
                current_section = line.lstrip("#").strip().split()[0]
                sections[current_section] = []
            elif current_section is not None:
                # e.g. "job_42,"
                sections[current_section].append(line.rstrip(","))

    return sections


def delete_jobs(
    job_names: list[str],
    jobs_dir: Path,
    dry_run: bool,
) -> tuple[int, list[str]]:
    """Delete job directories from *jobs_dir*.

    Args:
        job_names: List of folder names to delete (e.g. ['job_0', 'job_1']).
        jobs_dir: Root directory containing the job folders.
        dry_run: If True, only print what would be deleted without removing anything.

    Returns:
        Tuple of (deleted_count, list of missing folder paths).
    """
    deleted = 0
    missing: list[str] = []

    for name in job_names:
        folder = jobs_dir / name
        if folder.exists():
            if dry_run:
                print(f"  [dry-run] would delete {folder}")
            else:
                shutil.rmtree(folder)
            deleted += 1
        else:
            missing.append(str(folder))

    return deleted, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete job folders for spin=1 jobs reset to to_run"
    )
    ap.add_argument("reset_list", type=Path, help="Path to the reset jobs txt file")
    ap.add_argument(
        "--actinides-dir",
        type=Path,
        default=None,
        help="Root jobs directory for actinide jobs",
    )
    ap.add_argument(
        "--non-actinides-dir",
        type=Path,
        default=None,
        help="Root jobs directory for non-actinide jobs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing anything",
    )
    ap.add_argument(
        "--not-found-file",
        type=Path,
        default=None,
        help="Write paths of missing (not found) job folders to this file for debugging",
    )
    args = ap.parse_args()

    if args.actinides_dir is None and args.non_actinides_dir is None:
        ap.error("Provide at least one of --actinides-dir or --non-actinides-dir")

    sections = parse_reset_list(args.reset_list)

    dir_map: dict[str, Path | None] = {
        "actinides": args.actinides_dir,
        "non_actinides": args.non_actinides_dir,
    }

    if args.dry_run:
        print("DRY RUN - no files will be deleted\n")

    total_deleted = 0
    all_missing: list[str] = []

    for section, jobs_dir in dir_map.items():
        job_names = sections.get(section, [])
        if not job_names:
            print(f"[{section}] No jobs found in reset list - skipping")
            continue
        if jobs_dir is None:
            print(f"[{section}] No directory provided - skipping {len(job_names)} jobs")
            continue
        if not jobs_dir.exists():
            print(f"[{section}] Directory not found: {jobs_dir} - skipping")
            continue

        print(f"[{section}] Scanning {len(job_names)} jobs in {jobs_dir} ...")
        deleted, missing = delete_jobs(job_names, jobs_dir, dry_run=args.dry_run)

        if missing:
            print(f"[{section}] Not found ({len(missing)}):")
            for path in missing:
                print(f"  {path}")

        action = "Would delete" if args.dry_run else "Deleted"
        print(f"[{section}] {action}: {deleted}  |  Not found: {len(missing)}\n")
        total_deleted += deleted
        all_missing.extend(missing)

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"Total - {action}: {total_deleted}  |  Not found: {len(all_missing)}")

    if all_missing and args.not_found_file:
        args.not_found_file.write_text("\n".join(all_missing) + "\n")
        print(f"Not-found paths written to {args.not_found_file}")


if __name__ == "__main__":
    main()
