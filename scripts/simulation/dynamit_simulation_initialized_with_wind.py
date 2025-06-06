"""Simulation with initialized wind input."""

import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014  # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

WIND_FACTOR = 1  # Scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

filename_prefix = "wind_step"
Nmax, Mmax, Ncs = 50, 50, 50
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2  # int(80 / Nmax)])) ** 2
print(len(rk))

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date)  # Noon longitude
noon_mlon = d.mlt2mlon(12, date)  # Noon longitude

# Set up simulation object.
dynamics = pynamit.Dynamics(
    filename_prefix=filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="igrf",
    FAC_integration_steps=rk,
    ignore_PFAC=False,
    connect_hemispheres=True,
    latitude_boundary=latitude_boundary,
    ih_constraint_scaling=1e-5,
    t0=str(date),
)

# Get and set jr input.
jr_lat = dynamics.state.grid.lat
jr_lon = dynamics.state.grid.lon
apx = apexpy.Apex(refh=(RI - RE) * 1e-3, date=2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI - RE) * 1e-3)  # Fix this
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # Filter low latitude jr

# Get and set wind input.
hwm14Obj = pyhwm2014.HWM142D(
    alt=110.0,
    ap=[35, 35],
    glatlim=[-88.5, 88.5],
    glatstp=1.5,
    glonlim=[-180.0, 180.0],
    glonstp=3.0,
    option=6,
    verbose=False,
    ut=date.hour + date.minute / 60,
    day=date.timetuple().tm_yday,
)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten() * WIND_FACTOR, hwm14Obj.Uwind.flatten() * WIND_FACTOR)
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")
# u_lat, u_lon, u_phi, u_theta = (
#     np.load("ulat.npy"),
#     np.load("ulon.npy"),
#     np.load("uphi.npy"),
#     np.load("utheta.npy"),
# )
# u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing="ij")
# dynamics.set_u(
#     u_theta=u_theta,
#     u_phi=u_phi,
#     lat=u_lat,
#     lon=u_lon,
#     weights=np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)),
# )

# Get and set conductance input.
conductance_lat = dynamics.state.grid.lat
conductance_lon = dynamics.state.grid.lon

Kp = 5
hall_aurora, pedersen_aurora = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=False
)
dynamics.set_conductance(hall_aurora, pedersen_aurora, lat=conductance_lat, lon=conductance_lon)


# Initialize with zero jr.
dynamics.set_jr(jr=jr * 0, lat=jr_lat, lon=jr_lon)

dynamics.impose_steady_state()

# Turn jr on and evolve.
dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)


dynamics.evolve_to_time(10 * 60)
