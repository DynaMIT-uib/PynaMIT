"""Dipole, PFAC and DOP853 test."""

import numpy as np

from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


def test_2d_dipole_pfac_dop853(pynamit_approx):
    """Test 2D simulation with dipole, PFAC and DOP853."""
    # Arrange.
    expected_coeff_norm = 1.2049082557908e-08
    expected_coeff_max = 2.469631942439048e-09
    expected_coeff_min = -4.747215466647287e-09
    expected_n_coeffs = 240

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=10,
        Ncs=20,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        integrator="DOP853",
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

    assert actual_coeff_norm == pynamit_approx(expected_coeff_norm)
    assert actual_coeff_max == pynamit_approx(expected_coeff_max)
    assert actual_coeff_min == pynamit_approx(expected_coeff_min)
    assert actual_n_coeffs == pynamit_approx(expected_n_coeffs)
