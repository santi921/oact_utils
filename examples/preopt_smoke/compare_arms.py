"""Compare the baseline / gxtb / mlip arms of the pre-optimization smoke test, job by job.

Reads smoke_<arm>.db from --smoke-dir (read-only) and prints a pipe table per job plus a per-arm
summary. Energies are ORCA final energies (Hartree), differences are in mHartree against the
baseline arm, forces are ORCA max gradient components (Eh/Bohr), wall time is s.

Usage:
    python compare_arms.py /global/scratch/users/$USER/preopt_smoke
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ARMS = ["baseline", "gxtb", "mlip"]
COLS = (
    "id, elements, spin, status, scf_steps, final_energy, max_forces, wall_time, "
    "error_message, preopt_method, preopt_converged, preopt_fmax, preopt_rmsd, preopt_seconds"
)


def load(db: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(f"SELECT {COLS} FROM structures ORDER BY id", conn)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("smoke_dir", type=Path)
    args = ap.parse_args()

    frames = {}
    for arm in ARMS:
        db = args.smoke_dir / f"smoke_{arm}.db"
        if db.exists():
            frames[arm] = load(db)
    if not frames:
        raise SystemExit(f"no smoke_<arm>.db in {args.smoke_dir}")

    print("per-arm summary")
    print("arm | completed | failed | other | mean SCF steps | mean wall time (s)")
    for arm, df in frames.items():
        done = df[df.status == "completed"]
        print(
            f"{arm} | {len(done)}/{len(df)} | {int((df.status == 'failed').sum())} | "
            f"{int((~df.status.isin(['completed', 'failed'])).sum())} | "
            f"{done.scf_steps.mean():.1f} | {done.wall_time.mean():.1f}"
        )

    base = frames.get("baseline")
    print(
        "\nper-job: id | system | 2S+1 | arm | status | SCF steps | E (Eh) | dE vs baseline (mEh) | "
        "max force (Eh/Bohr) | wall (s) | preopt conv | preopt fmax (eV/A) | preopt rmsd (A) | preopt (s)"
    )
    ref = base.set_index("id") if base is not None else None
    for arm, df in frames.items():
        for r in df.itertuples():
            de = ""
            if ref is not None and arm != "baseline" and r.id in ref.index:
                e0 = ref.loc[r.id, "final_energy"]
                if pd.notna(r.final_energy) and pd.notna(e0):
                    de = f"{(r.final_energy - e0) * 1000:.3f}"
            print(
                f"{r.id} | {r.elements} | {r.spin} | {arm} | {r.status} | {r.scf_steps} | "
                f"{r.final_energy} | {de} | {r.max_forces} | {r.wall_time} | {r.preopt_converged} | "
                f"{r.preopt_fmax} | {r.preopt_rmsd} | {r.preopt_seconds}"
            )
            if r.status == "failed" and r.error_message:
                print(f"    error: {r.error_message}")


if __name__ == "__main__":
    main()
