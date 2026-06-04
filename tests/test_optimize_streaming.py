"""Tests for the streaming LMDB writer + diagnostics log used by the optimizing
entropy downselect (the resume/crash-consistency machinery), without a GPU.

Importing the module pulls fairchem (for the featurizer), but these tests never
instantiate the model -- they exercise only the streaming/resume helpers.
"""

from __future__ import annotations

import pickle

import lmdb
import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from oact_utilities.scripts.entropy_downselect.entropy_downselect_optimize import (  # noqa: E402
    _DiagnosticsLog,
    _jsonable,
    _StreamingLmdbWriter,
)


def _atoms(n: int, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    return Atoms("H" * n, positions=rng.standard_normal((n, 3)))


def _read_lmdb(path) -> tuple[int, dict]:
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    out = {}
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        for r in range(length):
            out[r] = pickle.loads(txn.get(f"{r}".encode("ascii")))
    env.close()
    return length, out


def test_streaming_writer_basic(tmp_path):
    path = tmp_path / "out.lmdb"
    w = _StreamingLmdbWriter(path, resume=False, flush_every=3)
    for r in range(10):
        w.put(r, _atoms(2 + r, r))
    w.finalize(10)
    w.close()

    length, got = _read_lmdb(path)
    assert length == 10
    assert set(got) == set(range(10))
    assert [len(got[r]) for r in range(10)] == [2 + r for r in range(10)]
    # sibling metadata.npz is written by the hook, not the writer; not checked here.


def test_streaming_writer_resume_overwrite(tmp_path):
    """Resume reopens the LMDB; re-emitted ranks overwrite, earlier ranks persist."""
    path = tmp_path / "out.lmdb"
    w = _StreamingLmdbWriter(path, resume=False, flush_every=2)
    for r in range(5):
        w.put(r, _atoms(2, r))  # 2 atoms each
    w.flush()  # simulate on_checkpoint flush
    w.close()  # simulate crash (no finalize)

    w2 = _StreamingLmdbWriter(path, resume=True, flush_every=2)
    for r in range(3, 10):
        w2.put(r, _atoms(7, r))  # re-emit 3,4 (overwrite) + new 5..9, 7 atoms each
    w2.finalize(10)
    w2.close()

    length, got = _read_lmdb(path)
    assert length == 10
    assert [len(got[r]) for r in range(10)] == [2, 2, 2, 7, 7, 7, 7, 7, 7, 7]


def test_diaglog_dedup_keeps_last(tmp_path):
    path = tmp_path / "diag.jsonl"
    d = _DiagnosticsLog(path, resume=False, flush_every=4)
    for r in range(10):
        d.append({"rank": r, "natoms": 2, "val": float(r)})
    d.flush()

    d2 = _DiagnosticsLog(path, resume=True)
    for r in range(5, 10):
        d2.append({"rank": r, "natoms": 2, "val": float(r + 100)})
    d2.close()

    recs = d2.load_deduped()
    assert [r["rank"] for r in recs] == list(range(10))
    assert recs[2]["val"] == 2.0
    assert recs[7]["val"] == 107.0  # last write wins


def test_diaglog_nan_and_types_roundtrip(tmp_path):
    path = tmp_path / "diag.jsonl"
    d = _DiagnosticsLog(path, resume=False)
    d.append(
        {
            "rank": np.int64(0),
            "natoms": 3,
            "x": np.float64("nan"),
            "b": np.bool_(True),
            "s": "skipped",
        }
    )
    d.close()
    rec = d.load_deduped()[0]
    assert rec["rank"] == 0
    assert np.isnan(rec["x"])
    assert rec["b"] is True
    assert rec["s"] == "skipped"


def test_jsonable_casts_numpy():
    out = _jsonable({"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(False)})
    assert isinstance(out["a"], float) and out["a"] == 1.5
    assert isinstance(out["b"], int) and out["b"] == 3
    assert out["c"] is False
