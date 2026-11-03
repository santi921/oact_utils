"""Pre-opt wiring: job directory prep, job scripts, submit-time element filter, status."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_preopt_gxtb import NATIVE, needs_gxtb

from oact_utilities.core.preopt.elements import UnsupportedElementsError
from oact_utilities.core.preopt.runner import PREOPT_CMD
from oact_utilities.utils.status import check_job_termination, preopt_failure
from oact_utilities.workflows import submit_jobs
from oact_utilities.workflows.submit_jobs import (
    filter_unsupported_preopt_elements,
    prepare_job_directory,
    write_flux_job_file,
    write_slurm_drac_job_file,
    write_slurm_job_file,
    write_slurm_sandia_job_file,
)

REPO = Path(__file__).resolve().parents[1]
ORCA = os.environ.get("ORCA_PATH") or shutil.which("orca")
CFG = {"orca_path": "/fake/orca", "preopt": "gxtb", "gxtb_path": "/fake/xtb"}


def _record(id_: int = 1, geometry: str | None = None, elements: str = "H;H"):
    rec = MagicMock()
    rec.id, rec.orig_index, rec.job_dir = id_, 42, None
    rec.geometry = geometry or "H 0.0 0.0 0.0\nH 0.0 0.0 0.74"
    rec.charge, rec.spin, rec.elements = 0, 1, elements
    return rec


def test_prepare_writes_shim_only_when_enabled(tmp_path: Path) -> None:
    off = prepare_job_directory(
        _record(), tmp_path / "off", orca_config={"orca_path": "/f/orca"}
    )
    assert not (off / "run_preopt.py").exists()
    on = prepare_job_directory(_record(), tmp_path / "on", orca_config=CFG, n_cores=6)
    assert (on / "run_preopt.py").exists() and (on / "input.xyz").exists()
    cfg = (on / "preopt_config.json").read_text()
    assert '"method": "gxtb"' in cfg and '"threads": 6' in cfg and '"job_id": 1' in cfg
    assert (on / "orca.inp").exists()


@pytest.mark.parametrize("flag", [True, False])
def test_job_scripts_prepend_preopt(tmp_path: Path, flag: bool) -> None:
    for name in "abcd":
        (tmp_path / name).mkdir()
    scripts = [
        write_flux_job_file(tmp_path / "a", preopt=flag),
        write_slurm_job_file(tmp_path / "b", preopt=flag),
        write_slurm_sandia_job_file(tmp_path / "c", preopt=flag, orca_path="/f/orca"),
        write_slurm_drac_job_file(tmp_path / "d", preopt=flag, venv_path="/venv"),
    ]
    for script in scripts:
        text = script.read_text()
        assert (PREOPT_CMD in text) == flag
        if flag:
            assert text.index(PREOPT_CMD) < text.index("orca.inp")


def test_drac_preopt_requires_venv(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    with pytest.raises(ValueError, match="preopt"):
        write_slurm_drac_job_file(tmp_path / "d", preopt=True)


def test_filter_noop_without_preopt() -> None:
    jobs = [_record(elements="U;O")]
    assert filter_unsupported_preopt_elements(jobs, {}) is jobs


def test_filter_gxtb_aborts_then_skips(capsys: pytest.CaptureFixture) -> None:
    ok, bad = _record(1, elements="U;O;O"), _record(2, elements="Rf;O")
    with pytest.raises(UnsupportedElementsError) as exc:
        filter_unsupported_preopt_elements([ok, bad], {"preopt": "gxtb"})
    assert exc.value.elements == ["Rf"]
    out = capsys.readouterr().out
    assert "Rf: 1" in out and "--skip-unsupported" in out
    kept = filter_unsupported_preopt_elements([ok, bad], {"preopt": "gxtb"}, True)
    assert kept == [ok]


def test_filter_mlip_uses_checkpoint_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        submit_jobs, "get_supported_elements", lambda ckpt: frozenset({"H", "O", "Cu"})
    )
    jobs = [_record(1, elements="Cu;O"), _record(2, elements="U;O")]
    kept = filter_unsupported_preopt_elements(
        jobs, {"preopt": "mlip", "mlip_checkpoint": "m.pt"}, skip_unsupported=True
    )
    assert [j.id for j in kept] == [1]
    with pytest.raises(ValueError, match="mlip_checkpoint"):
        filter_unsupported_preopt_elements(jobs, {"preopt": "mlip"})


def test_filter_keeps_rows_without_elements(capsys: pytest.CaptureFixture) -> None:
    jobs = [_record(1, elements="")]
    assert filter_unsupported_preopt_elements(jobs, {"preopt": "gxtb"}) == jobs
    assert "not pre-checked" in capsys.readouterr().out


def test_status_preopt_failure_is_job_failure(tmp_path: Path) -> None:
    (tmp_path / "preopt_status.json").write_text(
        '{"method": "gxtb", "error": "RuntimeError: boom", "detail": ""}'
    )
    assert preopt_failure(str(tmp_path)) == "RuntimeError: boom"
    assert check_job_termination(str(tmp_path)) == -1


def test_status_continued_on_raw_is_not_failure(tmp_path: Path) -> None:
    (tmp_path / "preopt_status.json").write_text(
        '{"method": "gxtb", "error": "RuntimeError: boom", '
        '"detail": "opt=tight; continued on raw geometry"}'
    )
    assert preopt_failure(str(tmp_path)) is None
    assert check_job_termination(str(tmp_path)) == 0


def test_status_success_and_missing_file(tmp_path: Path) -> None:
    assert preopt_failure(str(tmp_path)) is None
    (tmp_path / "preopt_status.json").write_text('{"method": "gxtb", "error": ""}')
    assert preopt_failure(str(tmp_path)) is None


def test_status_normal_termination_beats_stale_preopt_failure(tmp_path: Path) -> None:
    (tmp_path / "preopt_status.json").write_text(
        '{"method": "gxtb", "error": "RuntimeError: boom", "detail": ""}'
    )
    (tmp_path / "orca.out").write_text("...\n****ORCA TERMINATED NORMALLY****\n")
    assert check_job_termination(str(tmp_path)) == 1


def test_failure_reason_comes_from_preopt_status(tmp_path: Path) -> None:
    from oact_utilities.workflows.dashboard import _probe_unlinked_job
    from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

    job = prepare_job_directory(_record(), tmp_path, orca_config=CFG)
    (job / "preopt_status.json").write_text(
        '{"method": "mlip", "error": "UnsupportedElementsError: no U"}'
    )
    expected = "Pre-optimization failed: UnsupportedElementsError: no U"
    assert _probe_unlinked_job(job, hours_cutoff=24) == (True, -1, expected)

    rec = _record()
    rec.job_dir = str(job)
    wf = MagicMock()
    assert _skip_finished_on_disk([rec], tmp_path, "job_{orig_index}", wf) == []
    assert wf.update_status.call_args.kwargs["error_message"] == expected


@pytest.mark.parametrize("cfg", [{"orca_path": "/f/orca"}, CFG])
def test_reprepare_clears_stale_preopt_result(tmp_path: Path, cfg: dict) -> None:
    job = prepare_job_directory(_record(), tmp_path, orca_config=CFG)
    (job / "preopt_status.json").write_text('{"method": "gxtb", "error": "boom"}')
    (job / "preopt.xyz").write_text("stale")
    prepare_job_directory(_record(), tmp_path, orca_config=cfg)
    assert not (job / "preopt_status.json").exists()
    assert not (job / "preopt.xyz").exists()


def test_cli_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for argv, msg in [
        (["--preopt", "mlip"], "requires --mlip-checkpoint"),
        (["--skip-unsupported"], "require --preopt"),
    ]:
        monkeypatch.setattr(sys, "argv", ["submit_jobs", "x.db", str(tmp_path), *argv])
        with pytest.raises(SystemExit) as exc:
            submit_jobs.main()
        assert exc.value.code == 2, msg


@pytest.mark.gxtb
@needs_gxtb
@pytest.mark.skipif(ORCA is None, reason="no local ORCA: set ORCA_PATH")
def test_native_end_to_end_gxtb_then_orca_pm3(tmp_path: Path) -> None:
    """g-xTB pre-opt of a distorted water, then the unchanged ORCA PM3 step reads preopt geometry."""
    rec = _record(
        geometry="O 0.0 0.0 0.3\nH 0.0 1.2 -0.7\nH 0.0 -1.2 -0.7", elements="O;H;H"
    )
    cfg = {
        "orca_path": ORCA,
        "simple_input": "pm3",
        "preopt": "gxtb",
        "gxtb_path": NATIVE,
    }
    job = prepare_job_directory(rec, tmp_path, orca_config=cfg, n_cores=1)
    env = os.environ | {"OACT_UTILITIES_REPO_ROOT": str(REPO)}
    pre = subprocess.run(
        [sys.executable, "run_preopt.py"],
        cwd=job,
        env=env,
        capture_output=True,
        text=True,
    )
    assert pre.returncode == 0, pre.stdout + pre.stderr
    with open(job / "orca.out", "w") as fo:
        subprocess.run(
            [ORCA, "orca.inp"],
            cwd=job,
            stdout=fo,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    assert check_job_termination(str(job)) == 1
    assert "ORCA TERMINATED NORMALLY" in (job / "orca.out").read_text()
    inp = (job / "orca.inp").read_text()
    assert "1.2" not in inp
