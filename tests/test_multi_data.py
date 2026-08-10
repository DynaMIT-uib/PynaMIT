"""Multi-data test module."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


@pytest.mark.apexpy_precision
@pytest.mark.native_hwm_precision
def test_multi_data(regression_approx):
    """Test simulation with multiple data points."""
    # Arrange.
    expected_coeff_norm = 3.2859775703370715e-08
    expected_coeff_max = 1.3334897880074204e-08
    expected_coeff_min = -1.0749905655413092e-08
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=15,
        dt=5,
        Nmax=10,
        Mmax=8,
        Ncs=20,
        main_field_kind="igrf",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        equilibrium_initialization=True,
        boundary_jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
        integrator="exponential",
        multi_data=True,
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
