from __future__ import annotations

import numpy as np

from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.settings import DynamicsSettings


def test_dynamics_set_fac_accepts_batched_time_series():
    settings = DynamicsSettings(Nmax=2, Mmax=2, Ncs=6, t0="2001-05-12 21:45:00")
    dynamics = Dynamics(settings, benchmark_mode=True)

    lat = np.array([[10.0, 10.0], [20.0, 20.0]])
    lon = np.array([[0.0, 10.0], [0.0, 10.0]])
    fac = np.array([[1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6], [5.0e-6, 6.0e-6, 7.0e-6, 8.0e-6]])
    time = np.array([0.0, 10.0])

    dynamics.set_FAC(fac, lat=lat, lon=lon, time=time)

    jr_dataset = dynamics.input_timeseries.datasets["jr"]
    np.testing.assert_allclose(jr_dataset.time.values, time)
    assert int(jr_dataset.sizes["time"]) == 2
