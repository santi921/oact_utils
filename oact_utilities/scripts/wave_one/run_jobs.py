import os
from oact_utilities.utils.jobs import launch_flux_jobs


# roots
# root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"

if __name__ == "__main__":

    dry = True
    skip_done = False
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift/"
    )
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift_non_ma/"
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_M062x/"
    )
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_tpss/"
    )
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_pbe0/"
    )
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pwcvtz_omol_M062x/"
    )
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_M062x/"
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_pbe0/"
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_tpss/"
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
