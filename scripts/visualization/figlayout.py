"""Build and plot a representative steady-state simulation."""

import datetime

import dipole
from lompe import conductance
import numpy as np
import pyamps
import pyhwm2014  # https://github.com/rilma/pyHWM14

import pynamit
from pynamit.visualization.results import plot_state_diagnostics


RE = 6371.2e3
RI = RE + 110e3
LATITUDE_BOUNDARY = 35
WIND_FACTOR = 1

run_directory = "figlayout"
Nmax, Mmax, Ncs = 14, 14, 30
fac_integration_radii = RI / np.cos(np.deg2rad(np.r_[0 : 80 : int(80 / Nmax)])) ** 2
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
dipole_model = dipole.Dipole(date.year)
noon_longitude = dipole_model.mlt2mlon(12, date)

simulation = pynamit.Simulation(
    run_directory=run_directory,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    main_field_kind="dipole",
    fac_integration_radii=fac_integration_radii,
    enable_pfac_coupling=True,
    enable_interhemispheric_coupling=True,
    interhemispheric_coupling_latitude=LATITUDE_BOUNDARY,
)

model_grid = simulation.geometry.model_grid
hall, pedersen = conductance.hardy_EUV(
    model_grid.lon, model_grid.lat, Kp, date, starlight=1, dipole=True
)
simulation.set_conductance(hall, pedersen, lat=model_grid.lat, lon=model_grid.lon)

amps = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = (
    amps.get_upward_current(mlat=model_grid.lat, mlt=dipole_model.mlon2mlt(model_grid.lon, date))
    * 1e-6
)
jr[np.abs(model_grid.lat) < 50] = 0
simulation.set_jr(jr, lat=model_grid.lat, lon=model_grid.lon)

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
simulation.set_neutral_wind(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
)

simulation.impose_steady_state(time=0.0, save=True, quiet=True)
plot_state_diagnostics(
    simulation, title="State diagnostic summary", filename=None, noon_longitude=noon_longitude
)
