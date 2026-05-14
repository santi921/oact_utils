import os
import pandas as pd
import pathlib
import concurrent.futures
import traceback
from datetime import datetime
import copy
import hashlib
import numpy as np
import flux.job
from executorlib import FluxJobExecutor as Executor
from architector import convert_io_molecule
import flux
from architector.io_ptable import actinides, lanthanides

outpath = "outputs"
input_path = "metal_ox_lig_randsample_1e5_0.pkl"
main_debug_log_path = None

# 60 minute timeout in seconds
TIMEOUT_SECONDS = 60 * 60
store = 100

handle = flux.Flux()
rs = flux.resource.status.ResourceStatusRPC(handle).get()
rl = flux.resource.list.resource_list(handle).get()
print("NODELIST: ", rs.nodelist, " #CORES: ", rl.all.ncores)

cwd = os.path.abspath(".")
# Total Workers
# max_workers = 24
max_workers = rl.all.ncores
# max_workers = int(os.environ.get("MAX_WORKERS", max_workers))

metal_swap_dict = dict(zip(actinides, lanthanides))
metal_swap_dict.update(
    {
        "Fr": "Cs",
        "Ra": "Ba",
        "At": "I",
        "Po": "Te",
    }
)

def debug_log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [pid={os.getpid()}] {message}"
    print(line, flush=True)
    if main_debug_log_path:
        try:
            with open(main_debug_log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            # Never crash logging path if filesystem/log append fails.
            pass



def extract_features_from_molecules(
    molecule_strings,
    calc,
):
    """Extract UMA pre-activation features for a list of molecule strings.

    Parameters
    ----------
    molecule_strings : Sequence[str]
        Molecules accepted by ``convert_io_molecule`` (for example xyz strings).
    calc : ASE calculator
        Calculator that stores ``activation_key`` in ``calc.results``.
    """

    features = []

    for mol_str in molecule_strings:
        mol = convert_io_molecule(mol_str)
        atoms = mol.ase_atoms
        metal = atoms.get_chemical_symbols()[0]
        atoms.symbols[0] = metal_swap_dict[metal]
        atoms.info["charge"] = int(mol.charge)
        atoms.info["spin"] = int(mol.uhf + 1)
        atoms.calc = calc
        atoms.get_potential_energy()
        features.append(atoms.calc.results["uma_last_layer_pre_activation"][0])

    return features


def generate_distorted_xyz_variants(
    mol2string: str,
    *,
    n_variants: int = 5,
    min_distortion: float = 0.05,
    max_distortion: float = 0.3,
    seed_prefix: str = "",
):
    """Generate xyz variants by random Cartesian distortions."""
    mol = convert_io_molecule(mol2string)
    atoms = mol.ase_atoms
    base_symbols = atoms.get_chemical_symbols()
    base_positions = atoms.get_positions()
    variants = []

    for i in range(n_variants):
        seed_str = f"{seed_prefix}|{i}"
        seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        max_disp = float(rng.uniform(min_distortion, max_distortion))

        directions = rng.normal(size=base_positions.shape)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit_dirs = directions / norms
        magnitudes = rng.uniform(0.0, max_disp, size=(base_positions.shape[0], 1))
        displaced = base_positions + unit_dirs * magnitudes

        lines = [str(len(base_symbols)), f"distorted_variant={i} max_disp={max_disp:.6f}"]
        for sym, xyz in zip(base_symbols, displaced):
            lines.append(f"{sym} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}")
        xyzstr = "{}\n".format("\n".join(lines))
        newatoms = convert_io_molecule(xyzstr).ase_atoms
        mol.ase_atoms = newatoms
        variants.append(mol.write_mol2("max_disp_{:.6f}_variant_{}.mol2".format(max_disp, 
        i), writestring=True))

    return variants


def create_calc():
    import os

    # debug_log("create_calc: starting calculator initialization")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    MODEL_PATH = "/pscratch/sd/i/ishan_a/open_actinides/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"

    def _get_torch_model_from_predictor(predictor):
        for attr in ("module", "model", "inference_model", "_model"):
            model = getattr(predictor, attr, None)
            if isinstance(model, torch.nn.Module):
                return model
        for _, value in vars(predictor).items():
            if isinstance(value, torch.nn.Module):
                return value
        raise AttributeError(
            "Could not find underlying torch.nn.Module on predictor. "
            "Inspect `dir(predictor)` and update _get_torch_model_from_predictor()."
        )

    def _find_last_linear(module):
        last_name = None
        last_layer = None
        for name, submodule in module.named_modules():
            if isinstance(submodule, torch.nn.Linear):
                last_name, last_layer = name, submodule
        if last_layer is None:
            raise RuntimeError(
                "No torch.nn.Linear found in model; choose a different hook target."
            )
        return last_name, last_layer

    class UMAActivationsCalculator(FAIRChemCalculator):
        """FAIRChem calculator wrapper that stores last-layer pre-activations."""

        def __init__(
            self,
            predict_unit,
            *,
            activation_key: str = "uma_last_layer_pre_activation",
            store: str = "numpy",
        ):
            super().__init__(predict_unit)
            self._activation_key = activation_key
            self._store = store

            torch_model = _get_torch_model_from_predictor(self.predictor)
            self._hook_layer_name, self._hook_layer = _find_last_linear(torch_model)

            self._last_pre_activation: torch.Tensor | None = None

        def _hook_fn(self):
            def fn(_mod, inp, _out):
                if (
                    isinstance(inp, (tuple, list))
                    and len(inp) > 0
                    and torch.is_tensor(inp[0])
                ):
                    x = inp[0]
                elif torch.is_tensor(inp):
                    x = inp
                else:
                    x = None
                self._last_pre_activation = None if x is None else x.detach()

            return fn

        def calculate(self, atoms, properties=("energy",), system_changes=()):
            handle = self._hook_layer.register_forward_hook(self._hook_fn())
            try:
                super().calculate(atoms, list(properties), list(system_changes))
            finally:
                handle.remove()

            x = self._last_pre_activation
            if x is None:
                self.results[self._activation_key] = None
            else:
                x_cpu = x.to("cpu")
                self.results[self._activation_key] = (
                    x_cpu if self._store == "torch_cpu" else x_cpu.numpy()
                )

            self.results[f"{self._activation_key}_layer"] = self._hook_layer_name
            self.results[f"{self._activation_key}_shape"] = (
                None if x is None else tuple(x.shape)
            )

    def load_uma_activation_calculator(
        model_path: str,
        device: str = "cpu",
        activation_key: str = "uma_last_layer_pre_activation",
        store: str = "numpy",
    ):
        predictor = load_predict_unit(path=model_path, device=device)
        return UMAActivationsCalculator(
            predictor,
            activation_key=activation_key,
            store=store,
        )

    calc = load_uma_activation_calculator(model_path=MODEL_PATH)
    # debug_log("create_calc: calculator initialized successfully")
    return calc


def run_static(inp_dict):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import pandas as pd
    import time
    from architector import build_complex

    start = time.time()
    complex_mol = []
    job_name = inp_dict.get("name", "<unknown>")
    stage = "calc_init"
    # debug_log(f"run_static[{job_name}]: started")

    try:
        calc = create_calc()
        stage = "build_complex"
        complex_mol = build_complex(inp_dict)  # Run architector
        # debug_log(
        #     f"run_static[{job_name}]: build_complex finished with {len(complex_mol)} entries"
        # )

        stage = "feature_extraction"
        ttime = time.time() - start
        dfrows = []
        mol_lst = []
        for key, val in complex_mol.items():
            del val["ase_atoms"]  # Don't store ASE atoms in dataframe, just the mol2string.
            base_row = copy.deepcopy(val)
            base_row["total_walltime"] = ttime
            base_row["gen_unique_name"] = key
            base_row["name"] = job_name
            base_row["fail"] = False
            base_row["exception"] = ""
            base_row["is_distorted"] = False
            base_row["distortion_variant_index"] = -1
            dfrows.append(base_row)
            mol_lst.append(base_row["mol2string"])

            distorted_variants = generate_distorted_xyz_variants(
                base_row["mol2string"],
                n_variants=5,
                min_distortion=0.05,
                max_distortion=0.3,
                seed_prefix=f"{job_name}|{key}",
            )
            for variant_idx, distorted_mol2string in enumerate(
                distorted_variants
            ):
                drow = copy.deepcopy(val)
                drow["total_walltime"] = ttime
                drow["gen_unique_name"] = f"{key}_distort_{variant_idx}"
                drow["name"] = job_name
                drow["fail"] = False
                drow["exception"] = ""
                drow["is_distorted"] = True
                drow["distortion_variant_index"] = variant_idx
                drow["mol2string"] = distorted_mol2string
                dfrows.append(drow)
                mol_lst.append(distorted_mol2string)

        # debug_log(
        #     f"run_static[{job_name}]: extracting features for {len(mol_lst)} molecules "
        #     f"({len(complex_mol)} base + {len(mol_lst) - len(complex_mol)} distorted)"
        # )
        features = extract_features_from_molecules(mol_lst, calc)
        for i, row in enumerate(dfrows):
            row["features"] = features[i]
        resultsdf = pd.DataFrame(dfrows)
        # # Don't save the ase_atoms.
        # if "ase_atoms" in resultsdf.columns:
        #     resultsdf.drop("ase_atoms", inplace=True, axis=1)

        stage = "write_output"
        out_file = os.path.join(outpath, job_name + ".pkl")
        resultsdf.to_pickle(out_file)
        # debug_log(f"run_static[{job_name}]: success, wrote {out_file}")
        return True
    except Exception as e:  # Catch all manner of errors.
        end = time.time()
        tb = traceback.format_exc()
        fail_path = os.path.join(outpath, job_name + "_failed.txt")
        debug_log(f"run_static[{job_name}]: FAILED at stage={stage}: {e}")
        with open(fail_path, "w") as file1:
            file1.write("Total_Time: {}\n".format(end - start))
            file1.write("Stage: {}\n".format(stage))
            file1.write("{}".format(e))
            file1.write("\n\nTraceback:\n")
            file1.write(tb)
        return False


if __name__ == "__main__":
    outpath = os.path.abspath(outpath)
    os.makedirs(outpath, exist_ok=True)
    main_debug_log_path = os.path.join(outpath, "_executor_debug.log")
    debug_log(f"Main start: cwd={cwd}, outpath={outpath}, input={input_path}")
    try:
        # Load input dataframe
        indf = pd.read_pickle(input_path)
        debug_log(f"Loaded input dataframe with {len(indf)} rows")

        # Shuffle
        indf = indf.sample(frac=1, random_state=42)
        debug_log(f"Sampled dataframe now has {len(indf)} rows")

        # Add index as name of job from input dataframe.
        newindf_rows = []
        for i, row in indf.iterrows():
            inp_dict = row["architector_input"]
            inp_dict["name"] = input_path.replace(".pkl", "") + "_" + str(i)
            newindf_rows.append(inp_dict)

        # Check the output path to not duplicate
        # Finished/failed architector runs.
        op = pathlib.Path(outpath)
        done_list = [
            p.name.replace(".pkl", "").replace("_failed.txt", "") for p in op.glob("*")
        ]
        to_do = []
        for d in newindf_rows:
            if d["name"] not in done_list:
                to_do.append(d)
        debug_log(
            f"Prepared jobs: total_candidates={len(newindf_rows)}, "
            f"already_done={len(done_list)}, to_submit={len(to_do)}"
        )

        with flux.job.FluxExecutor() as flux_exe:
            with (
                Executor(
                    max_workers=max_workers,  # total number of cores available to the Executor
                    resource_dict={
                        "cores": 1,
                        "threads_per_core": 1,
                        "cwd": cwd,
                    },
                    flux_executor=flux_exe,
                    block_allocation=False,  # no init_function payload
                ) as exe
            ):
                # Run it
                futs = []
                fut_to_name = {}
                debug_log(
                    f"Submitting {len(to_do)} jobs with max_workers={max_workers}"
                )
                for td in to_do:
                    fut = exe.submit(run_static, td)
                    futs.append(fut)
                    fut_to_name[fut] = td.get("name", "<unknown>")
                done_count = 0
                all_done = 0
                ok_count = 0
                fail_count = 0
                debug_log("Tracking completions...")
                # with tqdm(total=len(to_do)) as pbar:
                for done in concurrent.futures.as_completed(futs):
                    # pbar.update(1)  # Update pbar
                    done_count += 1
                    all_done += 1
                    name = fut_to_name.get(done, "<unknown>")
                    try:
                        result = done.result()
                    except Exception as e:
                        fail_count += 1
                        debug_log(
                            f"Future exception for job {name}: {e}\n{traceback.format_exc()}"
                        )
                    else:
                        if result:
                            ok_count += 1
                        else:
                            fail_count += 1
                        # debug_log(
                        #     f"Job {name} finished with result={result}. "
                        #     f"ok={ok_count}, failed={fail_count}, total={all_done}"
                        # )
                    if done_count == store:
                        done_count = 0
                        debug_log(
                            "Progress checkpoint: Total Done: {} (ok={}, failed={})".format(
                                all_done, ok_count, fail_count
                            )
                        )
    except Exception as e:
        debug_log(f"FATAL main exception: {e}\n{traceback.format_exc()}")
        raise
