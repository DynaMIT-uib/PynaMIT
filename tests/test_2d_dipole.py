"""Dipole test."""

import os
import tempfile
import pytest

from pynamit.default_run import run_pynamit
import numpy as np


def test_2d_dipole():
    """Test 2D simulation with dipole."""
    # Arrange.
    expected_coeff_norm = 1.233894552105508e-07
    expected_coeff_max = 7.169514594656011e-10
    expected_coeff_min = -4.890615507043153e-09
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
        Ncs=60,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        steady_state_initialization=False,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values,
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values,
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
