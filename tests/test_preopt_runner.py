"""Pre-opt shim and in-job runner: orca.inp rewriting, exit codes, status file."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_preopt_gxtb import NATIVE, WATER, needs_gxtb

from oact_utilities.core.preopt._common import PreoptResult, parse_xyz
from oact_utilities.core.preopt.runner import (
    PREOPT_CMD,
    rewrite_inp_geometry,
    write_preopt_shim,
)
from oact_utilities.utils.create import read_geom_from_inp_file

REPO = Path(__file__).resolve().parents[1]
INP = "! PM3 EnGrad\n%pal\n nprocs 1\nend\n*xyz 0 1\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.48\nH 0.0 -0.76 -0.48\n*\n"
DISTORTED = "3\n\nO 0 0 0.3\nH 0 1.2 -0.7\nH 0 -1.2 -0.7\n"


def _job(tmp_path: Path, xyz: str = WATER, inp: str = INP) -> Path:
    (tmp_path / "orca.inp").write_text(inp)
    return tmp_path


def _run_shim(job: Path, **env: str) -> subprocess.CompletedProcess:
    full = os.environ | {"OACT_UTILITIES_REPO_ROOT": str(REPO), **env}
    return subprocess.run(
        [sys.executable, "run_preopt.py"],
        cwd=job,
        env=full,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_rewrite_inp_geometry_replaces_block_keeps_rest(tmp_path: Path) -> None:
    job = _job(tmp_path)
    rewrite_inp_geometry(job / "orca.inp", DISTORTED)
    text = (job / "orca.inp").read_text()
    assert text.startswith("! PM3 EnGrad\n%pal") and text.rstrip().endswith("*")
    atoms = read_geom_from_inp_file(str(job / "orca.inp"), ase_format_tf=True)
    assert atoms.positions[1][1] == pytest.approx(1.2)
    assert atoms.charge == 0 and atoms.spin == 1


def test_rewrite_inp_geometry_errors(tmp_path: Path) -> None:
    job = _job(tmp_path)
    with pytest.raises(ValueError, match="atoms"):
        rewrite_inp_geometry(job / "orca.inp", "2\n\nH 0 0 0\nH 0 0 1\n")
    (job / "bad.inp").write_text("! PM3\n")
    with pytest.raises(ValueError, match="no '\\*xyz'"):
        rewrite_inp_geometry(job / "bad.inp", WATER)


def test_preopt_cmd_stops_job_on_failure() -> None:
    assert "run_preopt.py" in PREOPT_CMD and "|| exit 1" in PREOPT_CMD
    assert PREOPT_CMD.endswith("\n")


def test_shim_writes_static_files(tmp_path: Path) -> None:
    shim = write_preopt_shim(tmp_path, WATER, "gxtb", 0, 1, job_id=5)
    assert shim.name == "run_preopt.py" and os.access(shim, os.X_OK)
    assert (tmp_path / "input.xyz").read_text() == WATER
    assert '"job_id": 5' in (tmp_path / "preopt_config.json").read_text()
    assert "WATER" not in shim.read_text() and "xtb" not in shim.read_text()


def test_shim_success_with_fake_xtb(fake_xtb: str, tmp_path: Path) -> None:
    job = _job(tmp_path)
    write_preopt_shim(job, DISTORTED, "gxtb", 0, 1, gxtb_path=fake_xtb)
    proc = _run_shim(job)
    assert proc.returncode == 0, proc.stderr
    res = PreoptResult.read(job)
    assert res and res.ok and res.converged
    assert (job / "preopt.xyz").exists()
    atoms = read_geom_from_inp_file(str(job / "orca.inp"), ase_format_tf=True)
    assert atoms.positions[1][1] == pytest.approx(
        1.2
    )  # fake xtb returns input unchanged


@pytest.mark.parametrize(("on_fail", "code"), [("fail", 1), ("continue", 0)])
def test_shim_failure_exit_codes(
    fake_xtb: str, tmp_path: Path, on_fail: str, code: int
) -> None:
    job = _job(tmp_path)
    write_preopt_shim(job, WATER, "gxtb", 0, 1, gxtb_path=fake_xtb, on_fail=on_fail)
    assert _run_shim(job, FAKE_XTB_MODE="bad_exit").returncode == code
    res = PreoptResult.read(job)
    assert res and not res.ok and res.error.startswith("RuntimeError")
    assert (job / "orca.inp").read_text() == INP
    assert not (job / "preopt.xyz").exists()
    assert ("continued on raw geometry" in res.detail) == (on_fail == "continue")


def test_shim_unsupported_element(fake_xtb: str, tmp_path: Path) -> None:
    job = _job(tmp_path)
    write_preopt_shim(job, "1\n\nRf 0 0 0\n", "gxtb", 0, 1, gxtb_path=fake_xtb)
    assert _run_shim(job).returncode == 1
    res = PreoptResult.read(job)
    assert res and "UnsupportedElementsError" in res.error
    assert res.detail == "unsupported_elements"


def test_shim_mlip_without_checkpoint_fails_cleanly(tmp_path: Path) -> None:
    job = _job(tmp_path)
    write_preopt_shim(job, WATER, "mlip", 0, 1)
    assert _run_shim(job).returncode == 1
    res = PreoptResult.read(job)
    assert res and "mlip_checkpoint" in res.error


@pytest.mark.gxtb
@needs_gxtb
def test_native_shim_relaxes_distorted_water(tmp_path: Path) -> None:
    job = _job(tmp_path)
    write_preopt_shim(job, DISTORTED, "gxtb", 0, 1, gxtb_path=NATIVE, threads=2)
    proc = _run_shim(job)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    res = PreoptResult.read(job)
    assert res and res.converged and res.fmax_final <= 0.05
    atoms = read_geom_from_inp_file(str(job / "orca.inp"), ase_format_tf=True)
    oh = abs(atoms.positions[1] - atoms.positions[0])
    assert 0.9 < ((atoms.positions[1] - atoms.positions[0]) ** 2).sum() ** 0.5 < 1.05
    assert parse_xyz((job / "preopt.xyz").read_text())[0] == ["O", "H", "H"]
    assert oh.sum() > 0
