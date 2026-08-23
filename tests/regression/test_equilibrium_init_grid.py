"""Grid-based equilibrium initialization test."""

import numpy as np
import pytest
from tests import magnetic_potential_coordinate_array
from tests.example_scenario import run_example


@pytest.mark.native_hwm_precision
def test_equilibrium_init_grid(regression_approx):
    """Test grid-based simulation with equilibrium initialization."""
    # Arrange.
    # HWM winds are rotated from geographic into dipole coordinates.
    expected_coeff_norm = 1.3445084598553368e-08
    expected_coeff_max = 1.5164526056028662e-09
    expected_coeff_min = -5.642220202358395e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_example(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        initialize_from_equilibrium=True,
        boundary_jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
    )

    # Assert.
    coeff_array = magnetic_potential_coordinate_array(simulation)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]
    resistance = simulation.data.input_series.datasets["conductance"]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == regression_approx(expected_coeff_norm)
    assert actual_coeff_max == regression_approx(expected_coeff_max)
    assert actual_coeff_min == regression_approx(expected_coeff_min)
    assert actual_n_coeffs == expected_n_coeffs
    assert "CS_log_conductance_magnitude" in resistance
    assert "CS_log_hall_to_pedersen_ratio" in resistance
