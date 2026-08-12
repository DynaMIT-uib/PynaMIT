"""IGRF, PFAC, HC, and wind test."""

import numpy as np
import pytest

from pynamit.workflows.example import run_example
from tests import magnetic_potential_coordinate_array


@pytest.mark.apexpy_precision
@pytest.mark.native_hwm_precision
def test_2d_igrf_pfac_hc_wind(regression_approx):
    """Test 2D simulation with IGRF, PFAC, HC, and wind."""
    # Arrange.
    expected_coeff_norm = 1.0496896083297977e-08
    expected_coeff_max = 3.524782738723694e-09
    expected_coeff_min = -1.9956990712000052e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_example(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="igrf",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        equilibrium_initialization=False,
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
