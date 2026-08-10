"""Dipole and PFAC test."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


def test_2d_dipole_pfac():
    """Test 2D simulation with dipole and PFAC."""
    # Arrange.
    expected_coeff_norm = 1.2064700204724392e-08
    expected_coeff_max = 2.4632594321555923e-09
    expected_coeff_min = -4.719596570090808e-09
    expected_n_coeffs = 240

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=10,
        Ncs=20,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
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

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)
