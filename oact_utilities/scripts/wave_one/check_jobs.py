from oact_utilities.utils.status import check_sucessful_jobs

if __name__ == "__main__":

    flux_tf = True
    check_many = False

    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift/"
    )
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift_non_ma/"
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_M062x/"
    )
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_tpss/"
    )
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_pbe0/"
    )
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = (
        "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pwcvtz_omol_M062x/"
    )
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_M062x/"
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_pbe0/"
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_tpss/"
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
