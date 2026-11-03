"""Build a tiny workflow DB for a real pre-optimization smoke test: 6 AN66 diatomics + 4 distorted waters.

The AN66 rows are copied (charge, spin multiplicity, geometry) from the AN66 geopt DB. The waters are
the same molecule at different O-H lengths and H-O-H angles so the pre-opt has real work to do.

Usage:
    python make_smoke_db.py --an66-db data/converted_workflows/an66_geopt.db --out smoke.db
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from oact_utilities.utils.architector import create_workflow_db

# Low-to-moderate spin AN66 members: (elements as stored in the DB)
AN66_PICK = ["Th;O", "Th;F", "Th;Cl", "Th;Br", "U;O", "U;F"]
# (O-H length in Angstrom, H-O-H angle in degrees); the relaxed geometry is about (0.96, 104.5)
WATERS = [(0.90, 95.0), (1.10, 125.0), (1.25, 110.0), (1.00, 150.0)]


def water_xyz(r: float, angle_deg: float) -> str:
    half = math.radians(angle_deg) / 2
    x, z = r * math.sin(half), r * math.cos(half)
    return f"O 0.0 0.0 0.0\nH {x:.6f} 0.0 {z:.6f}\nH {-x:.6f} 0.0 {z:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--an66-db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.an66_db}?mode=ro", uri=True)
    rows = []
    for els in AN66_PICK:
        r = conn.execute(
            "SELECT charge, spin, geometry FROM structures WHERE elements = ? "
            "ORDER BY natoms, id LIMIT 1",
            (els,),
        ).fetchone()
        if r is None:
            raise SystemExit(f"{els} not found in {args.an66_db}")
        rows.append({"charge": r[0], "spin": r[1], "geometry": r[2].strip()})
    rows += [{"charge": 0, "spin": 1, "geometry": water_xyz(r, a)} for r, a in WATERS]

    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "smoke.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        create_workflow_db(
            csv_path=str(csv),
            db_path=str(args.out),
            geometry_column="geometry",
            spin_column="spin",  # already 2S+1; the default column is "uhf"
        )
    print(
        f"wrote {args.out}: {len(rows)} jobs ({len(AN66_PICK)} AN66 + {len(WATERS)} water)"
    )


if __name__ == "__main__":
    main()
