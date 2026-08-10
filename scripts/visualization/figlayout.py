"""Build and plot a representative equilibrium simulation."""

import datetime

import dipole
import numpy as np
import pyamps
import pyhwm2014  # https://github.com/rilma/pyHWM14

import pynamit
from pynamit.external_input_contracts import ExternalInputRequest
from pynamit.external_inputs import get_conductance_inputs
from pynamit.visualization.results import plot_output_diagnostics

RE = 6371.2e3
RI = RE + 110e3
LATITUDE_BOUNDARY = 35
WIND_FACTOR = 1

run_directory = "figlayout"
Nmax, Mmax, Ncs = 14, 14, 30
fac_integration_radii = RI / np.cos(np.deg2rad(np.r_[0 : 80 : int(80 / Nmax)])) ** 2
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5

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
    t0=str(date),
)

model_grid = simulation.geometry.model_grid
dipole_model = dipole.Dipole(simulation.geometry.main_field.epoch)
noon_longitude = dipole_model.mlt2mlon(12, date)
geographic_lat, geographic_lon = dipole_model.mag2geo(model_grid.lat, model_grid.lon)
request = ExternalInputRequest.from_model_coordinates(
    model_grid.lat,
    model_grid.lon,
    geographic_lat=geographic_lat,
    geographic_lon=geographic_lon,
    coordinate_system=simulation.geometry.main_field.horizontal_coordinate_system,
    model_epoch=simulation.geometry.main_field.epoch,
    grid_id="figlayout-model-grid",
)
hall, pedersen, _, _ = get_conductance_inputs(date, None, None, None, request=request, kp=Kp)
simulation.set_conductance(hall, pedersen, lat=model_grid.lat, lon=model_grid.lon)

amps = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = (
    amps.get_upward_current(mlat=model_grid.lat, mlt=dipole_model.mlon2mlt(model_grid.lon, date))
    * 1e-6
)
jr[np.abs(model_grid.lat) < 50] = 0
simulation.set_boundary_jr(jr, lat=model_grid.lat, lon=model_grid.lon)

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

simulation.impose_equilibrium(time=0.0, save=True, quiet=True)
plot_output_diagnostics(
    simulation, title="Output diagnostic summary", filename=None, noon_longitude=noon_longitude
)
