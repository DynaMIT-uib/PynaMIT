"""Multi-data test module."""

import os
import tempfile
import datetime
import pytest

from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.settings import DynamicsSettings
from pynamit.data import get_conductance_inputs, get_jr_inputs, get_wind_inputs
import numpy as np


def _get_state_coeff_array(dynamics: Dynamics) -> np.ndarray:
    state_ds = dynamics.output_timeseries.datasets["state"]
    return np.hstack(
        (
            state_ds["SH_m_ind"].values[-1],
            state_ds["SH_m_imp"].values[-1],
        )
    )


def _build_multi_data_dynamics(run_directory: str) -> Dynamics:
    """Construct the same multi-data wind case as ``test_multi_data``."""
    settings = DynamicsSettings(
        run_directory=run_directory,
        Nmax=10,
        Mmax=8,
        Ncs=20,
        mainfield_kind="igrf",
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        integrator="exponential",
        conductance_interpolation_mode="legacy_eta_linear",
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
    )
    dynamics = Dynamics(settings)

    date = datetime.datetime(2001, 5, 12, 21, 45)
    time = np.linspace(0.0, 15.0, 4)

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon
    hall, pedersen, conductance_lat, conductance_lon = get_conductance_inputs(
        date,
        conductance_lat,
        conductance_lon,
        time,
    )

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    jr, jr_lat, jr_lon = get_jr_inputs(date, jr_lat, jr_lon, time)

    wind_inputs = get_wind_inputs(date, wind=True, time=time)
    assert wind_inputs is not None
    u_theta, u_phi, u_lat, u_lon, weights = wind_inputs

    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        time=time,
    )
    dynamics.set_jr(
        jr,
        lat=jr_lat,
        lon=jr_lon,
        time=time,
    )
    dynamics.set_u(
        u_theta=u_theta,
        u_phi=u_phi,
        lat=u_lat,
        lon=u_lon,
        sqrt_weights=weights,
        time=time,
    )
    return dynamics


@pytest.mark.wind
def test_multi_data():
    """Test simulation with multiple data points."""
    # Arrange.
    expected_coeff_norm = 2.5686566061400986e-08
    expected_coeff_max = 6.133350112801935e-09
    expected_coeff_min = -8.876382135048725e-09
    expected_n_coeffs = 228

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=15,
        dt=5,
        Nmax=10,
        Mmax=8,
        Ncs=20,
        mainfield_kind="igrf",
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        integrator="exponential",
        multi_data=True,
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


@pytest.mark.wind
def test_multi_data_restart_matches_direct_run(tmp_path):
    """Restarting from saved files should match the uninterrupted final state."""
    expected_coeff_norm = 2.5686566061400986e-08
    expected_coeff_max = 6.133350112801935e-09
    expected_coeff_min = -8.876382135048725e-09
    expected_n_coeffs = 228

    direct_run_directory = str(tmp_path / "multi_data_direct")
    restart_run_directory = str(tmp_path / "multi_data_restart")

    direct = _build_multi_data_dynamics(direct_run_directory)
    direct.evolve_to_time(
        t=15.0,
        dt=5.0,
        sampling_step_interval=1,
        saving_sample_interval=1,
        quiet=True,
        steady_state_initialization=True,
    )
    direct_coeffs = _get_state_coeff_array(direct)

    partial = _build_multi_data_dynamics(restart_run_directory)
    partial.evolve_to_time(
        t=10.0,
        dt=5.0,
        sampling_step_interval=1,
        saving_sample_interval=1,
        quiet=True,
        steady_state_initialization=True,
    )

    resumed = Dynamics.from_directory(restart_run_directory)
    resumed.evolve_to_time(
        t=15.0,
        dt=5.0,
        sampling_step_interval=1,
        saving_sample_interval=1,
        quiet=True,
        steady_state_initialization=True,
    )
    resumed_coeffs = _get_state_coeff_array(resumed)

    np.testing.assert_allclose(resumed_coeffs, direct_coeffs, rtol=1e-9, atol=1e-12)

    actual_coeff_norm = np.linalg.norm(resumed_coeffs)
    actual_coeff_max = np.max(resumed_coeffs)
    actual_coeff_min = np.min(resumed_coeffs)
    actual_n_coeffs = resumed_coeffs.shape[0]

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == expected_n_coeffs
