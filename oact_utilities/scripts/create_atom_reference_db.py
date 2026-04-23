#!/usr/bin/env python3
"""Create workflow databases of single-atom DFT jobs for Po through Lr.

Produces two databases:
  - atom_reference_non_actinides.db : Po, At, Rn, Fr, Ra  (5 elements)
  - atom_reference_actinides.db    : Ac through Lr        (15 elements)

Each element gets one row per spin state, from the ground-state multiplicity
down to the minimum allowed (M=1 for even electrons, M=2 for odd electrons)
in steps of 2. All atoms are neutral (charge=0).

Usage:
    python create_atom_reference_db.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oact_utilities.utils.architector import _init_db, _insert_row

# Ground-state spin multiplicities (2S+1) from atomic electron configurations.
# Po-Ra: post-lanthanide p/s-block.
# Ac-Lr: actinides via Hund's rule on 5f/6d.
# Note: Bk-No decrease from Am/Cm because 5f fills past half-shell.
NON_ACTINIDE_GS_MULT: dict[str, int] = {
    "Po": 3,  # [Xe] 4f14 5d10 6s2 6p4 - 2 unpaired
    "At": 2,  # [Xe] 4f14 5d10 6s2 6p5 - 1 unpaired
    "Rn": 1,  # [Xe] 4f14 5d10 6s2 6p6 - closed shell
    "Fr": 2,  # [Rn] 7s1               - 1 unpaired
    "Ra": 1,  # [Rn] 7s2               - closed shell
}

ACTINIDE_GS_MULT: dict[str, int] = {
    "Ac": 2,  # [Rn] 6d1 7s2           - 1 unpaired
    "Th": 3,  # [Rn] 6d2 7s2           - 2 unpaired
    "Pa": 4,  # [Rn] 5f2 6d1 7s2       - 3 unpaired
    "U": 5,  # [Rn] 5f3 6d1 7s2       - 4 unpaired
    "Np": 6,  # [Rn] 5f4 6d1 7s2       - 5 unpaired
    "Pu": 7,  # [Rn] 5f6 7s2           - 6 unpaired
    "Am": 8,  # [Rn] 5f7 7s2           - 7 unpaired
    "Cm": 9,  # [Rn] 5f7 6d1 7s2       - 8 unpaired
    "Bk": 6,  # [Rn] 5f9 7s2           - 5 unpaired (past half-fill)
    "Cf": 5,  # [Rn] 5f10 7s2          - 4 unpaired
    "Es": 4,  # [Rn] 5f11 7s2          - 3 unpaired
    "Fm": 3,  # [Rn] 5f12 7s2          - 2 unpaired
    "Md": 2,  # [Rn] 5f13 7s2          - 1 unpaired
    "No": 1,  # [Rn] 5f14 7s2          - closed 5f shell
    "Lr": 2,  # [Rn] 5f14 7s2 7p1      - 1 unpaired
}


def spin_states(gs_mult: int) -> list[int]:
    """Return multiplicities from gs_mult down to M_min in steps of 2.

    M_min is 1 if gs_mult is odd (even number of electrons),
    or 2 if gs_mult is even (odd number of electrons).
    """
    m_min = 1 if gs_mult % 2 == 1 else 2
    return list(range(gs_mult, m_min - 1, -2))


def atom_xyz(symbol: str) -> str:
    """Return a standard single-atom XYZ string at the origin."""
    return f"1\n{symbol}\n{symbol}  0.000000  0.000000  0.000000\n"


def _populate_db(
    db_path: Path,
    elements: dict[str, int],
    label: str,
) -> int:
    """Create a database and populate it with atom rows.

    Args:
        db_path: Path to the output SQLite database file.
        elements: Mapping of element symbol to ground-state multiplicity.
        label: Label for log output (e.g., "actinides").

    Returns:
        Number of rows inserted.
    """
    if db_path.exists():
        print(f"  Overwriting existing {db_path}")
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _init_db(db_path)

    total = 0
    for orig_index, (symbol, gs_mult) in enumerate(elements.items()):
        states = spin_states(gs_mult)
        xyz = atom_xyz(symbol)
        for mult in states:
            _insert_row(
                conn,
                orig_index=orig_index,
                elements=symbol,
                natoms=1,
                geometry=xyz,
                status="to_run",
                charge=0,
                spin=mult,
            )
            total += 1
        print(f"  {symbol:2s}  GS={gs_mult}  spin states: {states}")

    conn.commit()
    conn.close()
    print(f"  -> {db_path.name}: {total} rows ({len(elements)} elements)\n")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[3] / "data",
        help="Directory for output databases (default: data/)",
    )
    args = parser.parse_args()

    print("Non-actinides (Po-Ra):")
    n1 = _populate_db(
        args.output_dir / "atom_reference_non_actinides.db",
        NON_ACTINIDE_GS_MULT,
        "non-actinides",
    )

    print("Actinides (Ac-Lr):")
    n2 = _populate_db(
        args.output_dir / "atom_reference_actinides.db",
        ACTINIDE_GS_MULT,
        "actinides",
    )

    print(f"Total: {n1 + n2} rows across 2 databases.")


if __name__ == "__main__":
    main()
