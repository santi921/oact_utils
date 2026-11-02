"""Shared test fixtures."""

from __future__ import annotations

import stat
import sys
import textwrap
from pathlib import Path

import pytest

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
