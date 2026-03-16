"""Regularization test module."""

import pytest

from pynamit.simulation.runner import run_pynamit
import numpy as np
from pynamit.simulation.settings import MainfieldKind

@pytest.mark.wind
def test_regularization():
    """Test simulation with regularization."""
    # Arrange.
    expected_coeff_norm = 1.3111421667172157e-08
    expected_coeff_max = 1.7160298767949959e-09
    expected_coeff_min = -4.857200379874152e-09
    expected_n_coeffs = 228

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        jr_lambda=1e-3,
        conductance_lambda=1e-3,
        u_lambda=1e-3,
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
