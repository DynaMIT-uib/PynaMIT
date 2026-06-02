"""Grid-based steady state initialization test."""

import os
import tempfile
import pytest

from pynamit.default_run import run_pynamit
import numpy as np


def test_steady_state_init_grid():
    """Test grid-based simulation with steady state initialization."""
    # Arrange.
    expected_coeff_norm = 1.3116741266438326e-08
    expected_coeff_max = 1.715597619760354e-09
    expected_coeff_min = -4.859489437731163e-09
    expected_n_coeffs = 228

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        use_wind=True,
        steady_state_initialization=True,
        vector_jr=False,
        vector_conductance=False,
        vector_u=False,
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

    # pyHWM uses single precision, relax tolerances for wind tests.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)
