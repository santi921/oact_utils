import sqlite3
from pathlib import Path

import pandas as pd
from ase.io import read

from oact_utilities.utils.architector import chunk_architector_csv


def test_chunk_architector_writes_xyz_and_manifest(tmp_path: Path):
    # prepare a small CSV with three XYZ blocks
    xyz1 = """2
mol1
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
    xyz2 = """3
mol2
O 0.0 0.0 0.0
H 0.0 0.76 0.58
H 0.0 -0.76 0.58
"""
    xyz3 = """1
mol3
He 0.0 0.0 0.0
"""

    df = pd.DataFrame({"aligned_csd_core": [xyz1, xyz2, xyz3]})
    csv_path = tmp_path / "arch.csv"
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    db_path = outdir / "manifest.db"
    manifest = chunk_architector_csv(csv_path, outdir, chunk_size=2, db_path=db_path)

    assert manifest.exists()

    # chunk_0 should have 2 frames, chunk_1 should have 1 frame
    chunk0 = outdir / "chunk_0.xyz"
    chunk1 = outdir / "chunk_1.xyz"
    assert chunk0.exists()
    assert chunk1.exists()

    structures0 = list(read(chunk0, index=":", format="xyz"))
    structures1 = list(read(chunk1, index=":", format="xyz"))

    assert len(structures0) == 2
    assert len(structures1) == 1

    # manifest should contain three entries mapping to chunk files
    man_df = pd.read_csv(manifest)
    assert len(man_df) == 3
    assert set(man_df["chunk_file"]) == {"chunk_0.xyz", "chunk_1.xyz"}

    # verify DB was created and has three rows with status 'ready'
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM structures")
    count = cur.fetchone()[0]
    assert count == 3
    cur.execute("SELECT DISTINCT(status) FROM structures")
    statuses = {r[0] for r in cur.fetchall()}
    assert statuses == {"ready"}
    conn.close()
