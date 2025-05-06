"""Dipole, PFAC, and HC test."""

import os
import tempfile
import pytest

from pynamit.default_run import run_pynamit
import numpy as np


def test_2d_dipole_pfac_hc():
    """Test 2D simulation with dipole, PFAC, and HC."""
    # Arrange.
    expected_coeff_norm = 8.777569574333691e-08
    expected_coeff_max = 1.192784007093201e-09
    expected_coeff_min = -3.125107861526552e-09
    expected_n_coeffs = 201

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=5e-4,
        Nmax=5,
        Mmax=3,
        Ncs=18,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.timeseries.datasets["state"]["SH_m_ind"].values,
            dynamics.timeseries.datasets["state"]["SH_m_imp"].values,
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

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)
