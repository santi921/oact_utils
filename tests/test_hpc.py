from oact_utilities.utils import hpc


def test_write_flux_writes_job_in_root(tmp_path):
    d = tmp_path / "job1"
    d.mkdir()
    (d / "mol.inp").write_text("input")

    hpc.write_flux_no_template(str(d), two_step=False)
    assert (d / "flux_job.flux").exists()


def test_write_flux_no_template_sella_writes_in_root(tmp_path):
    d = tmp_path / "job2"
    d.mkdir()
    (d / "mol.inp").write_text("input")

    hpc.write_flux_no_template_sella_ase(str(d), two_step=False)
    assert (d / "flux_job.flux").exists()
