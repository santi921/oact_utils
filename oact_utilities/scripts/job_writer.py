import os 
from oact_utilities.utils.create import write_flux_no_template, write_jobs


def write_flux_orca_an66(
        actinide_basis: str,
        actinide_ecp: str,
        non_actinide_basis: str,
        template_file: str,
        root_dir: str,
        cores: int,
        safety: bool = True,
        n_hours: int = 24,
        allocation: str = "dnn-sim",
        two_step: bool = False, 
        queue: str = "pbatch",
        ref_geom_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
        ref_multiplicity_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt"
    ):

    # throw fit if ref geom or multiplicity files don't exist
    if not os.path.exists(ref_geom_file):
        raise FileNotFoundError(f"Reference geometry file {ref_geom_file} does not exist.")
    if not os.path.exists(ref_multiplicity_file):
        raise FileNotFoundError(f"Reference multiplicity file {ref_multiplicity_file} does not exist.")
    # make folder if not there 
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    # baseline OMol
    write_jobs(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        root_dir=root_dir,
        cores=cores,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step
    )

    if safety:
        cores +=2


    write_flux_no_template(
        root_dir=root_dir,
        two_step=two_step,
        n_cores=cores,
        n_hours=n_hours,
        queue=queue,
        allocation=allocation
    )

if __name__ == "__main__":


    cores = 10
    two_step = None
    n_hours = 4
    ref_geom_file = "./ref_geoms.txt"
    ref_multiplicity_file = "./ref_multiplicity.txt"

    
    
    ################################## OMOL BLOCK ##################################

    # 1) baseline omol 
    actinide_basis = "ma-def-TZVP"
    actinide_ecp = "def-ECP"
    non_actinide_basis = "def2-TZVPD"
    template_file = "/usr/workspace/vargas58/orca_test/templates/omol_REFERENCE.inp" # 300 step
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
   
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

    # 2) non-ma basis. omol_600_pmodel_shift_non_ma
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift_non_ma/"
    actinide_basis = "def-TZVP"
    template_file = "/usr/workspace/vargas58/orca_test/templates/omol_600_pmodel_shift.inp" # 600 step
    
    
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )
    
    
    # 3) omol_600_pmodel_shift
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift/"
    actinide_basis = "ma-def-TZVP"
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

    ################################## Relativistic BLOCK ##################################
    # all electronic basis sets
    actinide_ecp = None

    ################################## X2C BLOCK ##################################
    # x2c_omol.inp  x2c_omol_tpss.inp  x2c_omol_pbe0.inp 
    
    non_actinide_basis = "X2C-TZVPPall"


    actinide_basis = "/usr/workspace/vargas58/orca_test/basis_sets/cc_pvtz_x2c.bas"
    template_file = "/usr/workspace/vargas58/orca_test/templates/x2c_omol.inp" 
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_M062x/"
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )


    template_file = "/usr/workspace/vargas58/orca_test/templates/x2c_omol_tpss.inp" 
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_tpss/"
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

    template_file = "/usr/workspace/vargas58/orca_test/templates/x2c_omol_pbe0.inp" 
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_pbe0/"
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

    # diff basis set 
    # x2c_omol.inp
    actinide_basis = "/usr/workspace/vargas58/orca_test/basis_sets/cc_pwcvtz_x2c.bas"
    template_file = "/usr/workspace/vargas58/orca_test/templates/x2c_omol.inp" 
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pwcvtz_omol_M062x/"
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )



    ################################## DK3 BLOCK ##################################
    # dk3 M06x
    actinide_ecp = None
    actinide_basis = "SARC-DKH-TZVPP"
    non_actinide_basis = "DKH-def2-TZVPP"
    
    template_file = "/usr/workspace/vargas58/orca_test/templates/dk3_omol.inp" # 300 step
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_M062x/"
    
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

    template_file = "/usr/workspace/vargas58/orca_test/templates/dk3_omol_pbe0.inp" # 300 step
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_pbe0/"
    # dk3 PBE0
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )


    template_file = "/usr/workspace/vargas58/orca_test/templates/dk3_omol_tpss.inp" # 300 step
    root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_tpss/"
    # dk3 TPSS
    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        template_file=template_file,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours, 
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
    )

# root_dirs
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift_non_ma/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_M062x/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_tpss/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_pbe0/"   
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pwcvtz_omol_M062x/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_M062x/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_pbe0/"
    #root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_tpss/"
