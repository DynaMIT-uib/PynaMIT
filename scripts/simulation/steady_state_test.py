"""Test steady state simulation."""

import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014  # https://github.com/rilma/pyHWM14
import matplotlib.pyplot as plt
import datetime
import pyamps
import apexpy

dataset_filename_prefix = "ss_test"
Nmax, Mmax, Ncs = 15, 15, 16
latitude_boundary = 40
RE = 6371.2e3
RI = RE + 110e3
rk = RI / np.cos(np.deg2rad(np.r_[0:70:5])) ** 2

date = datetime.datetime(2001, 5, 12, 21, 0)
Kp = 5
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date)  # Noon longitude
noon_mlon = d.mlt2mlon(12, date)  # Noon longitude

# Set up simulation object.
dynamics = pynamit.Dynamics(
    dataset_filename_prefix=dataset_filename_prefix,
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
)

print("made dynamics object")

# Get and set conductance input.
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=False
)
dynamics.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

# Get and set jr input.
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
apx = apexpy.Apex(refh=(RI - RE) * 1e-3, date=2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI - RE) * 1e-3)  # fix this
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

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

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")

dynamics.set_u(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    weights=np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)),
)

dynamics.evolve_to_time(100)

mv = dynamics.state.steady_state_m_ind()


fig, ax = plt.subplots()
ax.plot(mv)
ax.plot(dynamics.timeseries["state"].SH_m_ind.values[-1, :])
plt.show()
