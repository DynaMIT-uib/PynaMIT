"""Example figure-layout script using the package visualization path."""

from __future__ import annotations

import datetime

import numpy as np

import pynamit
from dipole import Dipole
from lompe import conductance
import pyamps
import pyhwm2014


RE = 6371.2e3
RI = RE + 110e3
LATITUDE_BOUNDARY = 35
WIND_FACTOR = 1.0


def build_example_dynamics() -> pynamit.Dynamics:
    """Build a toy dynamics object and populate one current state entry."""
    filename_prefix = "figlayout"
    nmax, mmax, ncs = 14, 14, 30
    rk = RI / np.cos(np.deg2rad(np.r_[0:80:int(80 / nmax)])) ** 2
    date = datetime.datetime(2001, 5, 12, 21, 45)
    kp = 5

    dipole = Dipole(date.year)
    dynamics = pynamit.Dynamics(
        filename_prefix=filename_prefix,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        RI=RI,
        mainfield_kind="dipole",
        FAC_integration_steps=rk,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=LATITUDE_BOUNDARY,
    )

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon
    hall, pedersen = conductance.hardy_EUV(
        conductance_lon,
        conductance_lat,
        kp,
        date,
        starlight=1,
        dipole=True,
    )
    dynamics.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    amps = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
    jr = amps.get_upward_current(mlat=jr_lat, mlt=dipole.mlon2mlt(jr_lon, date)) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0.0
    dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

    hwm14 = pyhwm2014.HWM142D(
        alt=110.0,
        ap=[35, 35],
        glatlim=[-89.0, 88.0],
        glatstp=3.0,
        glonlim=[-180.0, 180.0],
        glonstp=8.0,
        option=6,
        verbose=False,
        ut=date.hour,
        day=date.timetuple().tm_yday,
    )
    u_theta = -hwm14.Vwind.flatten() * WIND_FACTOR
    u_phi = hwm14.Uwind.flatten() * WIND_FACTOR
    u_lat, u_lon = np.meshgrid(hwm14.glatbins, hwm14.glonbins, indexing="ij")
    dynamics.set_u(
        u_theta=u_theta,
        u_phi=u_phi,
        lat=u_lat,
        lon=u_lon,
        sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90.0 - u_lat.flatten()))), (2, 1)),
    )

    dynamics.current_time = np.float64(0.0)
    dynamics.state.update(dynamics.input_manager, float(dynamics.current_time), interpolation=True)

    m_ind = np.zeros(dynamics.state.solution_space.index_length)
    e_coeffs, m_imp = dynamics.state.calculate_noind_coeffs()
    dynamics.add_state_to_timeseries("state", m_ind, e_coeffs, m_imp)
    return dynamics


if __name__ == "__main__":
    current_date = datetime.datetime(2001, 5, 12, 21, 45)
    noon_longitude = Dipole(current_date.year).mlt2mlon(12, current_date)
    pynamit.debugplot(
        build_example_dynamics(),
        title="figlayout",
        filename=None,
        noon_longitude=noon_longitude,
    )
