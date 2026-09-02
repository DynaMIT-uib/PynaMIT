"""Simulate induction at high latitudes."""

import os

os.environ["MKL_NUM_THREADS"] = "185"
os.environ["NUMEXPR_NUM_THREADS"] = "185"
os.environ["OMP_NUM_THREADS"] = "185"
import numpy as np
import pynamit
from lompe import conductance
import dipole
import apexpy
import pyamps
import pyhwm2014  # https://github.com/rilma/pyHWM14

# import matplotlib.pyplot as plt
import datetime

simulation_directory = "data/brn_hl"
Nmax, Mmax, Ncs = 80, 80, 90
interhemispheric_coupling_latitude = 45
RE = 6371.2e3
RI = RE + 110e3
rk = RI / np.cos(np.deg2rad(np.r_[0:70:1])) ** 2

date = datetime.datetime(2001, 6, 1, 0, 0)
Kp = 4
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date)  # Noon longitude
noon_mlon = d.mlt2mlon(12, date)  # Noon longitude

# Set up simulation object.
simulation = pynamit.Simulation(
    simulation_directory=simulation_directory,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    main_field_kind="igrf",
    fac_integration_radii=rk,
    enable_pfac_coupling=True,
    enable_interhemispheric_coupling=True,
    interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
    interhemispheric_electric_field_weight=1e-5,
    t0=str(date),
)

print(datetime.datetime.now(), "made simulation object")

# Get and set conductance input.
conductance_lat = simulation.geometry.model_grid.lat
conductance_lon = simulation.geometry.model_grid.lon
hall, pedersen = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=False
)
simulation.set_conductance(
    pedersen=pedersen, hall=hall, lat=conductance_lat, lon=conductance_lon, reg_lambda=0.0001
)

print(datetime.datetime.now(), "setting jr")
# Set zero jr input.
jr_lat = simulation.geometry.model_grid.lat
jr_lon = simulation.geometry.model_grid.lon
simulation.set_boundary_jr(np.zeros_like(jr_lat), lat=jr_lat, lon=jr_lon)

print(datetime.datetime.now(), "setting wind")
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

simulation.set_neutral_wind(
    u_theta=u_theta,
    u_phi=u_phi,
    lat=u_lat,
    lon=u_lon,
    sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
)

print(datetime.datetime.now(), "calculating steady state")
simulation.evolve_to_time(0)
mv = simulation.response.steady_state_m_ind()
simulation.response.set_coeffs(m_ind=mv)
print(datetime.datetime.now(), "simulating")
simulation.evolve_to_time(0)

# Get and set jr input.
apx = apexpy.Apex(refh=(RI - RE) * 1e-3, date=2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
a = pyamps.AMPS(400, 5, -5, d.tilt(date), 100, minlat=50)
jr = a.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
simulation.set_boundary_jr(jr, lat=jr_lat, lon=jr_lon)


simulation.evolve_to_time(421)  # save simulation object with new m_ind


# a.make_multipanel_output_figure()


# fig, ax = plt.subplots()
# ax.plot(mv)
# ax.plot(simulation.data.output_series['state'].SH_m_ind.values[-1, :])
# plt.show()
