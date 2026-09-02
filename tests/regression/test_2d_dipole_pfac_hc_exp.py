"""Dipole, PFAC, HC and exponential test."""

import numpy as np
from tests import magnetic_potential_coordinate_array
from tests.example_scenario import run_example


def test_2d_dipole_pfac_hc_exp(regression_approx):
    """Test 2D simulation with dipole, PFAC, HC and exponential."""
    # Arrange.
    expected_coeff_norm = 8.958928750315398e-09
    expected_coeff_max = 1.6925448646901912e-09
    expected_coeff_min = -3.784848641445992e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_example(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        integrator="exponential",
        initialize_from_equilibrium=False,
    )

    # Assert.
    coeff_array = magnetic_potential_coordinate_array(simulation)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == regression_approx(expected_coeff_norm)
    assert actual_coeff_max == regression_approx(expected_coeff_max)
    assert actual_coeff_min == regression_approx(expected_coeff_min)
    assert actual_n_coeffs == expected_n_coeffs
