"""Steady state initialization test module."""

import os
import tempfile
import pytest

from pynamit.simulation.runner import run_pynamit
import numpy as np
from pynamit.simulation.settings import MainfieldKind


def test_steady_state_init():
    """Test simulation with steady state initialization."""
    # Arrange.
    expected_coeff_norm = 1.3120048541771941e-08
    expected_coeff_max = 1.7170964863338117e-09
    expected_coeff_min = -4.858577603591746e-09
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
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
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


def test_impose_steady_state_at_current_time(tmp_path):
    """Imposed steady state should overwrite the live state at the current time."""
    run_directory = tmp_path / "steady_state_impose"

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
        steady_state_initialization=False,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        run_directory=run_directory,
    )

    psi_ss, m_ind_ss = dynamics.impose_steady_state(quiet=True)

    state_entry = dynamics.output_timeseries.get_entry("state", dynamics.current_time)
    steady_entry = dynamics.output_timeseries.get_entry("steady_state", dynamics.current_time)

    np.testing.assert_allclose(np.asarray(state_entry["m_ind"]), np.asarray(m_ind_ss))
    np.testing.assert_allclose(np.asarray(steady_entry["m_ind"]), np.asarray(m_ind_ss))
    if psi_ss is not None:
        np.testing.assert_allclose(np.asarray(state_entry["psi"]), np.asarray(psi_ss))
        np.testing.assert_allclose(np.asarray(steady_entry["psi"]), np.asarray(psi_ss))
