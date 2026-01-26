import pickle
from pathlib import Path

import pandas as pd
import pytest

from oact_utilities.utils.architector import chunk_architector_to_lmdb


def test_chunk_architector_writes_lmdb(tmp_path: Path):
    lmdb = pytest.importorskip("lmdb")
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

    lmdb_path = tmp_path / "arch_lmdb"
    lmdb_path = chunk_architector_to_lmdb(csv_path, lmdb_path, chunk_size=2)

    env = lmdb.open(str(lmdb_path))
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = list(cursor.iternext(values=False))
        assert len(keys) == 3

        # check one record content
        val = txn.get(keys[0])
        rec = pickle.loads(val)
        assert rec["status"] == "ready"
        assert "geometry" in rec
        assert rec["natoms"] >= 1
