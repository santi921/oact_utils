"""Tests for sella_runner.py - focus on step-0 pre-seeding from existing orca.engrad."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

MINIMAL_ENGRAD = """\
#
# Number of atoms
#
 2
#
# The current total energy in Eh
#
   -1.234567890000
#
# The current gradient in Eh/bohr
#
       0.000000100000
       0.000000200000
       0.000123456789
      -0.000000100000
      -0.000000200000
      -0.000123456789
#
# The atomic numbers and current coordinates in Bohr
#
   1     0.0000000    0.0000000   -0.7000000
   1     0.0000000    0.0000000    0.7000000
"""

MINIMAL_ORCA_OUT = "ORCA TERMINATED NORMALLY\n"

MINIMAL_ORCA_INP = """\
! wB97M-V def2-TZVPD EnGrad
%pal nprocs 4 end
* xyz 0 1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
*
"""


def _write_orca_inp(job_dir: Path) -> None:
    (job_dir / "orca.inp").write_text(MINIMAL_ORCA_INP)


def _write_seed_files(job_dir: Path) -> None:
    (job_dir / "orca.engrad").write_text(MINIMAL_ENGRAD)
    (job_dir / "orca.out").write_text(MINIMAL_ORCA_OUT)


def _make_mock_atoms():
    atoms = MagicMock()
    atoms.copy.return_value = MagicMock()
    atoms.get_forces.return_value = np.zeros((2, 3))
    return atoms


def _run_with_mocks(job_dir: Path):
    """Run run_sella_optimization with all external dependencies mocked."""
    fake_results = {"energy": -1.234567890000, "forces": np.zeros((2, 3))}

    mock_atoms = _make_mock_atoms()
    mock_calc = MagicMock()
    mock_calc.template.read_results.return_value = fake_results
    mock_opt = MagicMock()
    mock_opt.nsteps = 3
    mock_opt.run.return_value = True

    with (
        patch(
            "oact_utilities.core.orca.sella_runner.read_geom_from_inp_file",
            return_value=mock_atoms,
        ),
        patch("ase.calculators.orca.ORCA", return_value=mock_calc),
        patch("ase.calculators.orca.OrcaProfile"),
        patch("sella.Sella", return_value=mock_opt),
        patch("ase.io.write"),
    ):
        from oact_utilities.core.orca.sella_runner import run_sella_optimization

        run_sella_optimization(
            job_dir=str(job_dir),
            charge=0,
            mult=1,
            orcasimpleinput="! wB97M-V def2-TZVPD EnGrad",
            orcablocks="%pal nprocs 4 end",
        )

    return mock_calc, mock_opt


class TestSellaRunnerPreseed:
    def test_preseed_called_when_engrad_and_out_exist(self, tmp_path):
        """read_results is called to pre-seed the calc when engrad+out are present."""
        _write_orca_inp(tmp_path)
        _write_seed_files(tmp_path)

        mock_calc, _ = _run_with_mocks(tmp_path)

        mock_calc.template.read_results.assert_called_once_with(tmp_path)

    def test_preseed_sets_calc_atoms(self, tmp_path):
        """calc.atoms is set to atoms.copy() when pre-seeding."""
        _write_orca_inp(tmp_path)
        _write_seed_files(tmp_path)

        mock_calc, _ = _run_with_mocks(tmp_path)

        assert mock_calc.atoms is not None

    def test_no_preseed_when_engrad_absent(self, tmp_path):
        """read_results is NOT called when orca.engrad is missing."""
        _write_orca_inp(tmp_path)

        mock_calc, _ = _run_with_mocks(tmp_path)

        mock_calc.template.read_results.assert_not_called()

    def test_no_preseed_when_out_absent(self, tmp_path):
        """read_results is NOT called when orca.out is missing (engrad alone not enough)."""
        _write_orca_inp(tmp_path)
        (tmp_path / "orca.engrad").write_text(MINIMAL_ENGRAD)

        mock_calc, _ = _run_with_mocks(tmp_path)

        mock_calc.template.read_results.assert_not_called()

    def test_preseed_failure_is_non_fatal(self, tmp_path):
        """If read_results raises, optimization continues without pre-seeding."""
        _write_orca_inp(tmp_path)
        _write_seed_files(tmp_path)

        mock_atoms = _make_mock_atoms()
        mock_calc = MagicMock()
        mock_calc.template.read_results.side_effect = ValueError("parse error")
        mock_opt = MagicMock()
        mock_opt.nsteps = 1
        mock_opt.run.return_value = True

        with (
            patch(
                "oact_utilities.core.orca.sella_runner.read_geom_from_inp_file",
                return_value=mock_atoms,
            ),
            patch("ase.calculators.orca.ORCA", return_value=mock_calc),
            patch("ase.calculators.orca.OrcaProfile"),
            patch("sella.Sella", return_value=mock_opt),
            patch("ase.io.write"),
        ):
            from oact_utilities.core.orca.sella_runner import run_sella_optimization

            run_sella_optimization(
                job_dir=str(tmp_path),
                charge=0,
                mult=1,
                orcasimpleinput="! wB97M-V def2-TZVPD EnGrad",
                orcablocks="%pal nprocs 4 end",
            )

        mock_opt.run.assert_called_once()


def _write_opt_traj(job_dir: Path, n_frames: int) -> None:
    """Write a minimal opt.traj with n_frames H2 geometries."""
    from ase import Atoms
    from ase.io.trajectory import Trajectory

    traj = Trajectory(str(job_dir / "opt.traj"), "w")
    for i in range(n_frames):
        # Slightly vary bond length so frames are distinguishable
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74 + 0.01 * i]])
        traj.write(a)
    traj.close()


def _run_and_capture(job_dir: Path, max_steps: int = 100):
    """Run with mocks and return (mock_calc, mock_opt, sella_constructor_args)."""
    fake_results = {"energy": -1.234567890000, "forces": np.zeros((2, 3))}

    mock_calc = MagicMock()
    mock_calc.template.read_results.return_value = fake_results
    mock_opt = MagicMock()
    mock_opt.nsteps = 2
    mock_opt.run.return_value = True

    sella_init = MagicMock(return_value=mock_opt)
    mock_inp_atoms = _make_mock_atoms()

    with (
        patch(
            "oact_utilities.core.orca.sella_runner.read_geom_from_inp_file",
            return_value=mock_inp_atoms,
        ),
        patch("ase.calculators.orca.ORCA", return_value=mock_calc),
        patch("ase.calculators.orca.OrcaProfile"),
        patch("sella.Sella", sella_init),
        patch("ase.io.write"),
    ):
        from oact_utilities.core.orca.sella_runner import run_sella_optimization

        run_sella_optimization(
            job_dir=str(job_dir),
            charge=0,
            mult=1,
            orcasimpleinput="! wB97M-V def2-TZVPD EnGrad",
            orcablocks="%pal nprocs 4 end",
            max_steps=max_steps,
        )

    return mock_calc, mock_opt, sella_init, mock_inp_atoms


class TestSellaRunnerRestart:
    def test_restart_reduces_step_budget(self, tmp_path):
        """Prior steps in opt.traj reduce steps_remaining passed to opt.run()."""
        _write_orca_inp(tmp_path)
        _write_opt_traj(tmp_path, n_frames=5)  # frame 0 + 4 steps

        _, mock_opt, _, _ = _run_and_capture(tmp_path, max_steps=100)

        call = mock_opt.run.call_args
        assert call.kwargs["steps"] == 100 - 4

    def test_restart_skips_preseed(self, tmp_path):
        """On restart, the engrad preseed is skipped even if files exist."""
        _write_orca_inp(tmp_path)
        _write_seed_files(tmp_path)
        _write_opt_traj(tmp_path, n_frames=3)  # 2 prior steps

        mock_calc, _, _, _ = _run_and_capture(tmp_path)

        mock_calc.template.read_results.assert_not_called()

    def test_restart_uses_last_frame_not_inp(self, tmp_path):
        """When restart fires, orca.inp geometry is NOT the atoms handed to Sella."""
        _write_orca_inp(tmp_path)
        _write_opt_traj(tmp_path, n_frames=4)

        _, _, sella_init, mock_inp_atoms = _run_and_capture(tmp_path)

        sella_atoms_arg = sella_init.call_args.args[0]
        assert sella_atoms_arg is not mock_inp_atoms

    def test_single_frame_traj_treated_as_fresh(self, tmp_path):
        """opt.traj with only the initial frame is not a restart."""
        _write_orca_inp(tmp_path)
        _write_seed_files(tmp_path)
        _write_opt_traj(tmp_path, n_frames=1)

        mock_calc, mock_opt, _, _ = _run_and_capture(tmp_path, max_steps=50)

        mock_calc.template.read_results.assert_called_once()
        assert mock_opt.run.call_args.kwargs["steps"] == 50

    def test_no_traj_is_fresh_run(self, tmp_path):
        """No opt.traj means fresh run: full step budget, orca.inp geometry."""
        _write_orca_inp(tmp_path)

        _, mock_opt, _, _ = _run_and_capture(tmp_path, max_steps=75)

        assert mock_opt.run.call_args.kwargs["steps"] == 75

    def test_corrupt_traj_falls_back_to_fresh(self, tmp_path):
        """Unreadable opt.traj does not crash; runner falls back to orca.inp."""
        _write_orca_inp(tmp_path)
        (tmp_path / "opt.traj").write_bytes(b"\x00\x01\x02not a trajectory")

        _, mock_opt, sella_init, mock_inp_atoms = _run_and_capture(
            tmp_path, max_steps=20
        )

        assert mock_opt.run.call_args.kwargs["steps"] == 20
        assert sella_init.call_args.args[0] is mock_inp_atoms

    def test_step_budget_floored_at_one(self, tmp_path):
        """If prior_steps >= max_steps, allow one more step to check convergence."""
        _write_orca_inp(tmp_path)
        _write_opt_traj(tmp_path, n_frames=51)  # 50 prior steps

        _, mock_opt, _, _ = _run_and_capture(tmp_path, max_steps=50)

        assert mock_opt.run.call_args.kwargs["steps"] == 1
