"""Run a pre-optimization inside a job directory, ahead of the ORCA step.

``prepare_job_directory`` writes ``input.xyz``, ``preopt_config.json`` and the static shim
``run_preopt.py`` (``write_preopt_shim``). On the compute worker the job script runs
``python run_preopt.py`` before ORCA / Sella. The shim relaxes ``input.xyz``, writes ``preopt.xyz``
and ``preopt_status.json``, and rewrites the coordinate block of ``orca.inp`` so the unchanged
ORCA or Sella step starts from the relaxed geometry.

Exit code is 0 on success, and on failure only when ``on_fail == "continue"`` (ORCA then runs on
the raw geometry and ``preopt_status.json`` records the error); otherwise 1, so the job script
stops before ORCA.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Literal

from oact_utilities.core.preopt._common import (
    PREOPT_XYZ,
    STATUS_FILE,
    PreoptResult,
    parse_xyz,
)
from oact_utilities.core.preopt.elements import UnsupportedElementsError

CONFIG_FILE = "preopt_config.json"
INPUT_XYZ = "input.xyz"
SHIM_FILE = "run_preopt.py"
# Prepended to the job command by every job-file writer.
PREOPT_CMD = "python run_preopt.py > preopt_driver.log 2>&1 || exit 1\n"


def rewrite_inp_geometry(inp_path: str | Path, xyz: str) -> None:
    """Replace the coordinate block of an ORCA input (``*xyz Q M`` ... ``*``) with ``xyz``.

    Raises:
        ValueError: when the file has no ``*xyz`` block or the atom count differs.
    """
    symbols, coords = parse_xyz(xyz)
    lines = Path(inp_path).read_text().splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith(("* xyz", "*xyz"))),
        None,
    )
    if start is None:
        raise ValueError(f"no '*xyz' block in {inp_path}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip() == "*"), None
    )
    if end is None:
        raise ValueError(f"unterminated '*xyz' block in {inp_path}")
    if end - start - 1 != len(symbols):
        raise ValueError(
            f"{inp_path} has {end - start - 1} atoms, pre-opt geometry has {len(symbols)}"
        )
    block = [
        f"{s:<3} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    Path(inp_path).write_text(
        "\n".join(lines[: start + 1] + block + lines[end:]) + "\n"
    )


def run_preopt_in_job_dir(
    job_dir: str | Path,
    method: Literal["gxtb", "mlip"],
    charge: int,
    mult: int,
    fmax: float = 0.05,
    max_steps: int = 200,
    gxtb_path: str = "xtb",
    mlip_checkpoint: str | None = None,
    mlip_device: str = "cpu",
    on_fail: Literal["fail", "continue"] = "fail",
    threads: int = 1,
    job_id: str | int | None = None,
) -> int:
    """Pre-optimize ``input.xyz`` in ``job_dir``; return the process exit code (see module doc)."""
    job = Path(job_dir)
    # A stale status from an earlier attempt must not read as this run's outcome.
    (job / STATUS_FILE).unlink(missing_ok=True)
    xyz = (job / INPUT_XYZ).read_text()
    try:
        if method == "gxtb":
            from oact_utilities.core.preopt.gxtb import run_gxtb_preopt

            out_xyz, result = run_gxtb_preopt(
                xyz,
                charge,
                mult,
                job / "preopt_gxtb",
                xtb_path=gxtb_path,
                fmax=fmax,
                max_cycles=max_steps,
                threads=threads,
            )
        elif method == "mlip":
            if not mlip_checkpoint:
                raise ValueError("mlip pre-optimization requires mlip_checkpoint")
            import torch

            torch.set_num_threads(threads)
            from oact_utilities.core.preopt.mlip import run_mlip_preopt

            out_xyz, result = run_mlip_preopt(
                xyz,
                charge,
                mult,
                mlip_checkpoint,
                device=mlip_device,
                fmax=fmax,
                max_steps=max_steps,
                job_id=job_id,
            )
        else:
            raise ValueError(f"unknown pre-opt method {method!r}")
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- every failure must reach preopt_status.json
        result = PreoptResult(method=method, error=f"{type(exc).__name__}: {exc}"[:500])
        if isinstance(exc, UnsupportedElementsError):
            result.detail = "unsupported_elements"
        out_xyz = xyz

    if result.ok:
        (job / PREOPT_XYZ).write_text(out_xyz)
        rewrite_inp_geometry(job / "orca.inp", out_xyz)
    if on_fail == "continue" and not result.ok:
        result.detail = (result.detail + "; continued on raw geometry").lstrip("; ")
    result.write(job)
    print(
        f"[preopt] {method} ok={result.ok} converged={result.converged} "
        f"fmax={result.fmax_final} steps={result.n_steps} error={result.error!r}"
    )
    return 0 if result.ok or on_fail == "continue" else 1


def write_preopt_shim(
    outputdir: str | Path,
    xyz: str,
    method: Literal["gxtb", "mlip"],
    charge: int,
    mult: int,
    fmax: float = 0.05,
    max_steps: int = 200,
    gxtb_path: str = "xtb",
    mlip_checkpoint: str | None = None,
    mlip_device: str = "cpu",
    on_fail: Literal["fail", "continue"] = "fail",
    threads: int = 1,
    job_id: str | int | None = None,
) -> Path:
    """Write ``input.xyz``, ``preopt_config.json`` and the static ``run_preopt.py`` shim.

    The shim reads the JSON and calls ``run_preopt_in_job_dir``; no value is interpolated into
    code. Returns the path to the shim.
    """
    outputdir = Path(outputdir)
    (outputdir / INPUT_XYZ).write_text(xyz)
    config: dict[str, Any] = {
        "method": method,
        "charge": charge,
        "mult": mult,
        "fmax": fmax,
        "max_steps": max_steps,
        "gxtb_path": gxtb_path,
        "mlip_checkpoint": mlip_checkpoint,
        "mlip_device": mlip_device,
        "on_fail": on_fail,
        "threads": threads,
        "job_id": job_id,
    }
    (outputdir / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    shim = outputdir / SHIM_FILE
    shim.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python
            \"\"\"Generated pre-optimization runner. Reads preopt_config.json.\"\"\"
            import json
            import os
            import sys
            from pathlib import Path

            repo_root = os.environ.get("OACT_UTILITIES_REPO_ROOT")
            if repo_root:
                sys.path.insert(0, repo_root)

            from oact_utilities.core.preopt.runner import run_preopt_in_job_dir

            config = json.loads(Path("preopt_config.json").read_text())
            sys.exit(run_preopt_in_job_dir(job_dir=".", **config))
            """
        )
    )
    shim.chmod(0o700)
    return shim
