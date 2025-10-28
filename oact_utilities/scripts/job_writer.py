from oact_utilities.utils.create import write_flux_no_template, write_jobs


def main():

    actinide_basis = "ma-def-TZVP"
    actinide_ecp = "def-ECP"
    non_actinide_basis = "def2-TZVPD"
    template_file = "/Users/santiagovargas/dev/oact_utils/data/data/template_orca.inp" # 300 step
    root_dir = "/Users/santiagovargas/dev/oact_utils/data/data/test_templateles_flux/"
    cores = 10

    # baseline OMol
    write_jobs(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        root_dir=root_dir,
        cores=cores,
        ref_geom_file="/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
        ref_multiplicity_file="/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt",
        two_step=None
    )


    write_flux_no_template(
        root_dir=root_dir,
        two_step=False,
        n_cores=cores,
        n_hours=2,
        queue="pbatch",
        allocation="dnn-sim"
    )
    