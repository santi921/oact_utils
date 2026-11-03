"""preopt_* workflow-DB columns: schema migration, writers, and the status-file reader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from oact_utilities.core.preopt._common import PreoptResult, preopt_db_fields
from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows.architector_workflow import (
    _METRIC_COLUMNS,
    _PREOPT_COLUMNS,
    ArchitectorWorkflow,
)


def _db(tmp_path: Path) -> Path:
    csv = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "charge": [0],
            "spin": [1],
            "aligned_csd_core": ["O 0 0 0.1\nH 0 0.7 -0.5\nH 0 -0.7 -0.5"],
        }
    ).to_csv(csv, index=False)
    return Path(
        create_workflow_db(
            csv_path=str(csv),
            db_path=str(tmp_path / "w.db"),
            geometry_column="aligned_csd_core",
        )
    )


def test_columns_are_in_the_single_metrics_tuple() -> None:
    assert set(_PREOPT_COLUMNS) <= set(_METRIC_COLUMNS)
    assert len(_PREOPT_COLUMNS) == 6


def test_reader_empty_without_status(tmp_path: Path) -> None:
    assert preopt_db_fields(tmp_path) == {}


def test_reader_maps_success_and_failure(tmp_path: Path) -> None:
    PreoptResult(
        method="gxtb",
        converged=True,
        fmax_final=0.004,
        energy_ev=-2080.0,
        rmsd_from_input=0.4,
        seconds=1.5,
    ).write(tmp_path)
    got = preopt_db_fields(tmp_path)
    assert got["preopt_method"] == "gxtb" and got["preopt_converged"] == 1
    assert got["preopt_fmax"] == 0.004 and got["preopt_seconds"] == 1.5
    PreoptResult(
        method="mlip", error="X: boom", detail="continued on raw geometry"
    ).write(tmp_path)
    got = preopt_db_fields(tmp_path)
    assert got["preopt_method"] == "mlip" and got["preopt_converged"] == 0
    assert got["preopt_fmax"] is None


def test_migration_adds_columns_and_bulk_writer_stores_them(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with ArchitectorWorkflow(str(db)) as wf:
        wf.update_job_metrics_bulk(
            [
                {
                    "job_id": 1,
                    "preopt_method": "gxtb",
                    "preopt_converged": 1,
                    "preopt_fmax": 0.01,
                }
            ]
        )
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(structures)")}
    assert set(_PREOPT_COLUMNS) <= cols
    row = conn.execute(
        "SELECT preopt_method, preopt_converged, preopt_fmax FROM structures"
    ).fetchone()
    assert row == ("gxtb", 1, 0.01)
