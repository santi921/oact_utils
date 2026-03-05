#!/usr/bin/env python3
"""
Move ground state calculations from a source root directory into the multi_spin
folder structure, adding them as spin_<ground_state> folders.

Usage:
    python move_ground_state_calcs.py /path/to/source_root /path/to/multi_spin_root \\
        --lot x2c --excel /path/to/dataset.xlsx --dry-run

The script expects source folders structured as:
    <source_root>/<donor_category>/<category>/<molecule>/

And will move them to (flattening out donor_category):
    <multi_spin_root>/<lot>/<category>/<molecule>/spin_<ground_state_mult>/

The ground state multiplicity is read from the Excel file's 'Multiplicity' column.

Example:
    # Move x2c ground state calcs
    python move_ground_state_calcs.py \\
        /p/lustre5/vargas58/maria_benchmarks/wave_2_x2c_opt_filtered \\
        /p/lustre5/vargas58/maria_benchmarks/multi_spin \\
        --lot x2c --excel /path/to/dataset.xlsx --dry-run

    # Move omol ground state calcs
    python move_gs.py /p/lustre5/vargas58/maria_benchmarks/wave2_omol_opt_sella /p/lustre5/vargas58/maria_benchmarks/multi_spin --lot omol --excel /usr/workspace/vargas58/multi_spin_data/dataset.xlsx --dry-run --verbose
"""
from __future__ import annotations

import argparse
import os
import shutil
from typing import Any

import pandas as pd


def sanitize_key(key: str) -> str:
    """Remove - or ECP from the key, and replace spaces with underscores."""
    key = key.replace("-", "").replace("ECP", "").replace("ZORA", "")
    return key


def read_and_process_data(data_xlsx: str) -> pd.DataFrame:
    """Read and process the Excel file with spin/multiplicity data."""
    # ensure to get all tabs of the excel file
    data_df = pd.read_excel(data_xlsx, sheet_name=None)
    # flatten dataframe to a single table
    data_df = pd.concat(data_df.values(), ignore_index=True)
    # remove all rows with only Nans
    data_df = data_df.dropna(how="all")
    # remove unnamed columns
    data_df = data_df.loc[:, ~data_df.columns.str.contains("^Unnamed")]
    # clean rows with missing multiplicity
    list_data = [
        "CCDC ID",
        "Actinide",
        "Formula",
        "Charge ",
        "Multiplicity",
        "Multiplicity_List",
    ]
    data_df = data_df.dropna(subset=list_data)
    # convert multiplicity to int
    data_df["Multiplicity"] = data_df["Multiplicity"].astype(int)
    return data_df


def find_molecule_folders(
    source_root: str, donor_category: str, category: str
) -> dict[str, str]:
    """
    Find molecule folders in the source directory.

    Expects structure: <source_root>/<donor_category>/<category>/<molecule>/

    Returns a dict mapping molecule name -> full path.
    """
    cat_path = os.path.join(source_root, donor_category, category)
    if not os.path.isdir(cat_path):
        return {}

    molecule_folders = {}
    for entry in os.listdir(cat_path):
        entry_path = os.path.join(cat_path, entry)
        if os.path.isdir(entry_path):
            molecule_folders[entry] = entry_path
    return molecule_folders


def find_donor_categories(source_root: str) -> list[str]:
    """Find donor category folders (Hard_donor, Soft_donor, etc.) in source root."""
    if not os.path.isdir(source_root):
        return []
    return [
        entry
        for entry in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, entry))
    ]


def find_categories_in_donor(source_root: str, donor_category: str) -> list[str]:
    """Find category folders within a donor category."""
    donor_path = os.path.join(source_root, donor_category)
    if not os.path.isdir(donor_path):
        return []
    return [
        entry
        for entry in os.listdir(donor_path)
        if os.path.isdir(os.path.join(donor_path, entry))
    ]


def strip_functional_suffix(name: str) -> str:
    """Strip functional suffixes like _PBE0, _TPSSH, _PBE from molecule name."""
    suffixes = ["_PBE0", "_TPSSH", "_PBE", "_pbe0", "_tpssh", "_pbe"]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def get_ground_state_mult(df: pd.DataFrame, molecule_name: str) -> int | None:
    """
    Get the ground state multiplicity for a molecule from the Excel data.

    Tries matching with:
    1. Sanitized key directly
    2. Sanitized key with functional suffix stripped (_PBE0, _TPSSH, _PBE)
    """
    # Try direct match with sanitized key
    sanitized = sanitize_key(molecule_name)
    row = df[df["Formula"] == sanitized]
    if not row.empty:
        return int(row["Multiplicity"].values[0])

    # Try stripping functional suffix then sanitizing
    stripped = strip_functional_suffix(molecule_name)
    if stripped != molecule_name:
        sanitized_stripped = sanitize_key(stripped)
        row = df[df["Formula"] == sanitized_stripped]
        if not row.empty:
            return int(row["Multiplicity"].values[0])

    return None


def move_ground_state_folders(
    source_root: str,
    dest_root: str,
    excel_path: str,
    lot: str,
    donor_categories: list[str] | None = None,
    categories: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    copy_mode: bool = False,
) -> dict[str, Any]:
    """
    Move (or copy) ground state calculation folders into the multi_spin structure.

    Args:
        source_root: Root directory containing donor category folders
                     (e.g., /path/to/wave_2_x2c_opt_filtered/<donor>/<category>/<molecule>/)
        dest_root: Root directory for multi_spin calculations
        excel_path: Path to Excel file with multiplicity data
        lot: Level of theory (e.g., "omol" or "x2c") - determines destination subfolder
        donor_categories: List of donor categories to process (None = auto-detect)
        categories: List of categories to process (None = auto-detect from each donor)
        dry_run: If True, only print what would be done
        verbose: Print detailed info
        copy_mode: If True, copy instead of move

    Returns:
        Summary dict with counts of moved, skipped, and failed folders
    """
    # Load Excel data
    print(f"Loading Excel data from {excel_path}...")
    df = read_and_process_data(excel_path)
    print(f"  Found {len(df)} entries in Excel file")

    # Auto-detect donor categories if not specified
    if donor_categories is None:
        donor_categories = find_donor_categories(source_root)
        print(f"Auto-detected donor categories: {donor_categories}")

    summary: dict[str, Any] = {
        "moved": 0,
        "skipped_exists": 0,
        "skipped_no_mult": 0,
        "failed": 0,
        "details": [],
    }

    action = "copy" if copy_mode else "move"
    action_past = "copied" if copy_mode else "moved"

    # Pre-flight check: collect all planned moves and check for conflicts
    print("\nPre-flight check: scanning for conflicts...")
    planned_moves: list[tuple[str, str, str, str, int]] = (
        []
    )  # (mol_name, src, dest, category, mult)
    conflicts: list[tuple[str, str]] = []  # (mol_name, dest_path)

    for donor_cat in donor_categories:
        cats_to_process = categories
        if cats_to_process is None:
            cats_to_process = find_categories_in_donor(source_root, donor_cat)

        for category in cats_to_process:
            molecule_folders = find_molecule_folders(source_root, donor_cat, category)
            if not molecule_folders:
                continue

            for mol_name, src_path in sorted(molecule_folders.items()):
                ground_mult = get_ground_state_mult(df, mol_name)
                if ground_mult is None:
                    summary["skipped_no_mult"] += 1
                    summary["details"].append(
                        {
                            "molecule": mol_name,
                            "lot": lot,
                            "category": category,
                            "status": "skipped_no_mult",
                        }
                    )
                    continue

                spin_folder = f"spin_{ground_mult}"
                dest_path = os.path.join(
                    dest_root, lot, category, mol_name, spin_folder
                )

                if os.path.exists(dest_path):
                    conflicts.append((mol_name, dest_path))
                else:
                    planned_moves.append(
                        (mol_name, src_path, dest_path, category, ground_mult)
                    )

    # If any conflicts found, report and abort
    if conflicts:
        print(f"\nERROR: {len(conflicts)} destination(s) already exist!")
        print("The following destinations would be overwritten:\n")
        for mol_name, dest_path in conflicts:
            print(f"  CONFLICT: {mol_name}")
            print(f"            -> {dest_path}")
        print(
            "\nAborting to prevent data loss. Remove existing folders first if intended."
        )
        summary["skipped_exists"] = len(conflicts)
        return summary

    print(f"  No conflicts found. {len(planned_moves)} folder(s) to {action}.")

    # Now perform the actual moves
    for mol_name, src_path, dest_path, category, ground_mult in planned_moves:
        spin_folder = f"spin_{ground_mult}"

        if dry_run:
            print(f"  DRY-RUN: Would {action} {src_path}")
            print(f"           -> {dest_path}")
            summary["moved"] += 1
            summary["details"].append(
                {
                    "molecule": mol_name,
                    "lot": lot,
                    "category": category,
                    "spin": ground_mult,
                    "status": f"would_{action}",
                    "src": src_path,
                    "dest": dest_path,
                }
            )
        else:
            try:
                # Ensure parent directory exists
                parent_dir = os.path.dirname(dest_path)
                os.makedirs(parent_dir, exist_ok=True)

                if copy_mode:
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.move(src_path, dest_path)

                print(f"  {action_past.upper()}: {mol_name} -> {spin_folder}")
                if verbose:
                    print(f"           {src_path}")
                    print(f"        -> {dest_path}")
                summary["moved"] += 1
                summary["details"].append(
                    {
                        "molecule": mol_name,
                        "lot": lot,
                        "category": category,
                        "spin": ground_mult,
                        "status": action_past,
                        "src": src_path,
                        "dest": dest_path,
                    }
                )
            except Exception as e:
                print(f"  FAILED: {mol_name} - {e}")
                summary["failed"] += 1
                summary["details"].append(
                    {
                        "molecule": mol_name,
                        "lot": lot,
                        "category": category,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    return summary


def print_summary(summary: dict[str, Any], dry_run: bool = False) -> None:
    """Print a summary of the move operation."""
    action = "Would move" if dry_run else "Moved"
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{action}: {summary['moved']}")
    print(f"Skipped (already exists): {summary['skipped_exists']}")
    print(f"Skipped (no multiplicity in Excel): {summary['skipped_no_mult']}")
    print(f"Failed: {summary['failed']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move ground state calculations into multi_spin folder structure"
    )
    parser.add_argument(
        "source_root",
        help="Root folder containing category folders (e.g., /path/to/wave_2_x2c_opt/)",
    )
    parser.add_argument(
        "dest_root",
        help="Root folder for multi_spin calculations",
    )
    parser.add_argument(
        "--lot",
        required=True,
        choices=["omol", "x2c"],
        help="Level of theory (determines destination subfolder)",
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to Excel file with multiplicity data",
    )
    parser.add_argument(
        "--donor-categories",
        nargs="+",
        default=None,
        help="Donor categories to process, e.g., Hard_donor Soft_donor (default: auto-detect)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to process within each donor (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually move files, only print what would be done",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.source_root):
        parser.error(
            f"Source root does not exist or is not a directory: {args.source_root}"
        )
    if not os.path.isfile(args.excel):
        parser.error(f"Excel file does not exist: {args.excel}")

    # Run the move operation
    summary = move_ground_state_folders(
        source_root=args.source_root,
        dest_root=args.dest_root,
        excel_path=args.excel,
        lot=args.lot,
        donor_categories=args.donor_categories,
        categories=args.categories,
        dry_run=args.dry_run,
        verbose=args.verbose,
        copy_mode=args.copy,
    )

    print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
