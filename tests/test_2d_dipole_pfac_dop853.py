"""Dipole, PFAC and DOP853 test."""

import os
import tempfile

from pynamit.default_run import run_pynamit
import numpy as np


def test_2d_dipole_pfac_dop853(pynamit_approx):
    """Test 2D simulation with dipole, PFAC and DOP853."""
    # Arrange.
    expected_coeff_norm = 1.1342052545869681e-08
    expected_coeff_max = 8.006258968163613e-10
    expected_coeff_min = -5.063807785683825e-09
    expected_n_coeffs = 240

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=10,
        Ncs=20,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        ignore_PFAC=False,
        integrator="DOP853",
        steady_state_initialization=False,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
        )
    )

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
