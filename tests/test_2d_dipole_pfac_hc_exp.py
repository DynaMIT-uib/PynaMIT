"""Dipole, PFAC, HC and exponential test."""

import pytest

from pynamit.simulation.runner import run_pynamit
import numpy as np
from pynamit.simulation.settings import IntegratorKind, MainfieldKind


def test_2d_dipole_pfac_hc_exp():
    """Test 2D simulation with dipole, PFAC, HC and exponential."""
    # Arrange.
    expected_coeff_norm = 9.289094165656056e-09
    expected_coeff_max = 1.5061102212509041e-09
    expected_coeff_min = -3.9449152012536495e-09
    expected_n_coeffs = 228

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        integrator=IntegratorKind.EXPONENTIAL,
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

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=2e-9)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=2e-9)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=2e-9)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)
