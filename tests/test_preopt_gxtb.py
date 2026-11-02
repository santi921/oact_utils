"""g-xTB pre-optimization: mock (fake xtb) tests plus native tests that need the real binary."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from oact_utilities.core.preopt._common import PreoptResult, parse_xyz
from oact_utilities.core.preopt.elements import UnsupportedElementsError
from oact_utilities.core.preopt.gxtb import (
    OPT_LEVELS,
    read_engrad,
    run_gxtb_preopt,
    select_opt_level,
)

WATER = "3\n\nO 0 0 0.12\nH 0 0.76 -0.48\nH 0 -0.76 -0.48\n"

# Behaviour switches via FAKE_XTB_MODE: ok | scf_fail_first | bad_exit | not_converged
FAKE_XTB = textwrap.dedent(
    """\
    #!{python}
    import os, sys
    from pathlib import Path

    args = sys.argv[1:]
    if args == ["--version"]:
        print("      |   xtb version 6.7.1 (fake)   |")
        sys.exit(0)
    mode = os.environ.get("FAKE_XTB_MODE", "ok")
    Path("args.log").write_text(" ".join(args))
    xyz = Path(args[0]).read_text().splitlines()
    n = int(xyz[0])
    if "--opt" in args and mode == "scf_fail_first" and not Path("xtbrestart").exists():
        print("SCF not converged in 1000 cycles")
        sys.exit(1)
    if mode == "bad_exit":
        sys.exit(3)
    Path("xtbrestart").write_text("restart")
    if "--opt" in args:
        out = [xyz[0], "", *xyz[2 : 2 + n]]
        Path("xtbopt.xyz").write_text("\\n".join(out) + "\\n")
        Path("xtbopt.log").write_text(
            "\\n".join(f"{{n}}\\n energy: -5.{{i}} gnorm: 0.01\\n" for i in range(4))
        )
        Path("xtbrestart").write_text("restart")
        if mode == "not_converged":
            Path("NOT_CONVERGED").write_text("")
    if "--grad" in args:
        g = os.environ.get("FAKE_XTB_GRAD", "1e-5")
        lines = ["#", str(n), "#", "-5.0"] + ["0.0"] * 0
        lines = ["#", str(n), "#", "-5.0", "#"]
        lines += [g, "0.0", "0.0"] * n
        Path("in.engrad").write_text("\\n".join(lines) + "\\n")
    print("normal termination of xtb")
    """
)


@pytest.fixture()
def fake_xtb(tmp_path: Path) -> str:
    path = tmp_path / "fake_xtb"
    path.write_text(FAKE_XTB.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_select_opt_level_bounds_fmax() -> None:
    assert select_opt_level(0.05) == "normal"
    assert select_opt_level(0.01) == "vtight"
    assert select_opt_level(0.0001) == "extreme"
    assert select_opt_level(10.0) == "crude"
    for f in (0.3, 0.05, 0.02, 0.01):
        thr_ev = OPT_LEVELS[select_opt_level(f)] * 51.42206313
        assert thr_ev <= f * 1.05


def test_read_engrad_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "e.engrad"
    p.write_text(
        "#\n2\n#\n-1.5\n#\n" + "\n".join(["0.1", "0", "0", "0", "0.2", "0"]) + "\n"
    )
    e, g = read_engrad(p)
    assert e == -1.5
    assert g.shape == (2, 3) and g[1, 1] == 0.2


def test_mock_success_singlet_is_restricted(fake_xtb: str, tmp_path: Path) -> None:
    xyz, res = run_gxtb_preopt(
        WATER, 0, 1, tmp_path / "w", xtb_path=fake_xtb, fmax=0.05
    )
    assert res.ok and res.converged
    assert res.method == "gxtb" and "opt=normal" in res.detail
    assert res.n_steps == 4
    assert res.fmax_final == pytest.approx(1e-5 * 51.42206313)
    assert res.rmsd_from_input == pytest.approx(0.0, abs=1e-6)
    args = (tmp_path / "w" / "opt" / "args.log").read_text()
    assert "--gxtb" in args and "--uhf" not in args and "--opt normal" in args
    assert "--grad" in (tmp_path / "w" / "sp" / "args.log").read_text()
    assert (tmp_path / "w" / "sp" / "xtbrestart").exists()
    assert len(parse_xyz(xyz)[0]) == 3


def test_mock_open_shell_passes_uhf(fake_xtb: str, tmp_path: Path) -> None:
    run_gxtb_preopt(WATER, 0, 3, tmp_path / "w", xtb_path=fake_xtb)
    assert "--uhf 2" in (tmp_path / "w" / "opt" / "args.log").read_text()


def test_mock_not_converged_flag(fake_xtb: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_XTB_MODE", "not_converged")
    _, res = run_gxtb_preopt(WATER, 0, 1, tmp_path / "w", xtb_path=fake_xtb)
    assert res.ok and not res.converged


def test_mock_high_force_not_converged(
    fake_xtb: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_XTB_GRAD", "0.01")
    _, res = run_gxtb_preopt(WATER, 0, 1, tmp_path / "w", xtb_path=fake_xtb, fmax=0.05)
    assert res.ok and not res.converged and res.fmax_final > 0.05


def test_mock_scf_failure_uses_seed(fake_xtb: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_XTB_MODE", "scf_fail_first")
    _, res = run_gxtb_preopt(WATER, 0, 1, tmp_path / "w", xtb_path=fake_xtb)
    assert res.ok and "scf_seed" in res.detail
    assert "--etemp 3000" in (tmp_path / "w" / "scf_seed" / "args.log").read_text()


def test_mock_xtb_failure_recorded_not_raised(
    fake_xtb: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_XTB_MODE", "bad_exit")
    xyz, res = run_gxtb_preopt(WATER, 0, 1, tmp_path / "w", xtb_path=fake_xtb)
    assert not res.ok and res.error.startswith("RuntimeError")
    assert xyz == WATER


def test_unsupported_element_raises(fake_xtb: str, tmp_path: Path) -> None:
    rf = "1\n\nRf 0 0 0\n"
    with pytest.raises(UnsupportedElementsError) as exc:
        run_gxtb_preopt(rf, 0, 1, tmp_path / "w", xtb_path=fake_xtb)
    assert exc.value.elements == ["Rf"]


def test_status_json_roundtrip(tmp_path: Path) -> None:
    res = PreoptResult(method="gxtb", converged=True, fmax_final=0.01, energy_ev=-1.0)
    res.write(tmp_path)
    assert PreoptResult.read(tmp_path) == res
    assert PreoptResult.read(tmp_path / "missing") is None


def _native_xtb() -> str | None:
    cand = os.environ.get("GXTB_PATH") or shutil.which("xtb")
    if not cand:
        return None
    import subprocess

    try:
        out = subprocess.run(
            [cand, "--help"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cand if "--gxtb" in out else None


NATIVE = _native_xtb()
needs_gxtb = pytest.mark.skipif(
    NATIVE is None, reason="no g-xTB xtb binary: set GXTB_PATH or put it on PATH"
)


@pytest.mark.gxtb
@needs_gxtb
def test_native_water_converges_and_is_stable(tmp_path: Path) -> None:
    xyz, res = run_gxtb_preopt(WATER, 0, 1, tmp_path / "w", xtb_path=NATIVE, fmax=0.05)
    assert res.ok and res.converged
    assert res.fmax_final <= 0.05
    assert res.energy_ev == pytest.approx(-76.4376 * 27.2114, abs=0.05)
    assert "6.7.1" in res.detail


@pytest.mark.gxtb
@needs_gxtb
def test_native_open_shell_radical(tmp_path: Path) -> None:
    oh = "2\n\nO 0 0 0\nH 0 0 1.1\n"
    _, res = run_gxtb_preopt(oh, 0, 2, tmp_path / "oh", xtb_path=NATIVE, fmax=0.05)
    assert res.ok and res.converged


@pytest.mark.gxtb
@needs_gxtb
def test_native_actinide_uo2(tmp_path: Path) -> None:
    uo2 = "3\n\nU 0 0 0\nO 0 0 1.9\nO 0 0 -1.9\n"
    xyz, res = run_gxtb_preopt(uo2, 2, 1, tmp_path / "u", xtb_path=NATIVE, fmax=0.05)
    assert res.ok, res.error
    assert res.converged
    _, p = parse_xyz(xyz)
    assert 1.5 < abs(p[1, 2] - p[0, 2]) < 2.2
