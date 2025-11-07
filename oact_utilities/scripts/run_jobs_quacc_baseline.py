import pickle as pkl 
from oact_utilities.core.orca.recipes import ase_relaxation
from oact_utilities.utils.create import * 

def jobs_wrapper_an66(
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = 1000,
    nprocs: int = 12,
    orca_cmd: str = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
    ref_geom_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
    ref_multiplicity_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt",
    root_directory: str = "./test_quacc_baseline/",
):


    df_multiplicity_ase = process_multiplicity_file(ref_multiplicity_file)
    dict_geoms_ase = process_geometry_file(ref_geom_file, ase_format_tf=True)
    job_list = df_multiplicity_ase["molecule"].tolist()
    spin_list = df_multiplicity_ase["multiplicity"].tolist()
    
    # create folder if it does not exist
    if not os.path.exists(root_directory):
        os.makedirs(root_directory)
    
    
    for ind, job in enumerate(job_list):
        atoms = dict_geoms_ase[job]
        nbo_tf = False
        charge = 0
        mult = spin_list[ind]
        root_directory_job = os.path.join(root_directory, job)
        if not os.path.exists(root_directory_job):
            os.makedirs(root_directory_job)
        
        res_dict = ase_relaxation(
            atoms=atoms,
            charge=charge,
            spin_multiplicity=mult,
            functional=functional,
            simple_input=simple_input,
            scf_MaxIter=scf_MaxIter,
            outputdir=root_directory_job,
            orca_cmd=orca_cmd,
            nbo=nbo_tf,
            nprocs=nprocs,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis
        )
    
        # convert res_dict to normal dict
        res_dict = dict(res_dict)
        # save res_dict as json
        save_loc = dict(res_dict)["dir_name"]
        with open(os.path.join(save_loc, "results.pkl"), "wb") as f:
            pkl.dump(res_dict, f)
    
if __name__ == "__main__":
    jobs_wrapper_an66()
