def quality_check(atoms, hof_references):
    """
    Classify whether the provided atoms object meets our quality checks to be
    included in the final LMDB. A system is usable if:
abs(referenced_energy) < 150, corresponds to the 99.8 percentile
fmax < 50eV/A
s_squared_dev check
integrated densities electrons are consistent, 0.001
"final exchange deviates considerably" not in warnings
    """
    try:
        atomic_numbers = atoms.get_atomic_numbers()
        ref_energy = hof_references[atomic_numbers].sum()
        total_energy = atoms.get_potential_energy()
        ref_energy_err = abs(total_energy - ref_energy)
        assert (
            ref_energy_err < 150 or ref_energy_err / len(atomic_numbers) < 10
        ), "energy deviates too much from reference"
        assert len(atoms) > 1, "1 atom system"
        assert atoms.info["fmax"] < 50, "fmax > 50"
        assert spin_contamination_check(atoms), "s_squared_dev violation"
        assert abs(atoms.info["charge"]) <= 10, "abs(charge) > 10"
        assert atoms.info["spin"] <= 23, "spin > 23"


        num_electrons = atoms.info["num_electrons"]
        num_alpha = (num_electrons + atoms.info["spin"] - 1) // 2
        num_beta = (num_electrons - atoms.info["spin"] + 1) // 2
        alpha_electrons, beta_electrons, total_electrons = atoms.info[
            "integrated_densities"
        ]
        assert np.isclose(
            alpha_electrons, num_alpha, atol=0.001
        ), "alpha electrons inconsistent"
        assert np.isclose(
            beta_electrons, num_beta, atol=0.001
        ), "beta electrons inconsistent"
        assert np.isclose(
            total_electrons, num_electrons, atol=0.001
        ), "Total electrons inconsistent"


        for warning_message in atoms.info["warnings"]:
            assert (
                "final exchange deviates considerably" not in warning_message
            ), "deviation warning"


        assert all(
            gap > 0 for gap in atoms.info["homo_lumo_gap"]
        ), "negative HOMO-LUMO gap"


        return True, None
    except AssertionError as e:
        return False, str(e)
