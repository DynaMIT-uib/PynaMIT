"""Simple test."""

import numpy as np
import pynamit
from lompe import conductance
import dipole

# import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

RE = 6371.2e3
RI = RE + 110e3
interhemispheric_coupling_latitude = 40

WIND_FACTOR = 1  # Scale wind by this factor
FLOAT_ERROR_MARGIN = 1e-6

run_directory = "test_data"
Nmax, Mmax, Ncs = 30, 30, 30
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2  # int(80 / Nmax)])) ** 2
print(len(rk))

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date)  # Noon longitude
noon_mlon = d.mlt2mlon(12, date)  # Noon longitude

# Set up simulation object.
simulation = pynamit.Simulation(
    run_directory=run_directory,
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

# Get and set jr input.
jr_lat = simulation.geometry.model_grid.lat
jr_lon = simulation.geometry.model_grid.lon
apx = apexpy.Apex(refh=(RI - RE) * 1e-3, date=2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI - RE) * 1e-3)  # Fix this
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # Filter low latitude jr
simulation.set_jr(jr, lat=jr_lat, lon=jr_lon)

# Get and set wind input.
# hwm14Obj = pyhwm2014.HWM142D(
#     alt=110.0,
#     ap=[35, 35],
#     glatlim=[-89.0, 88.0],
#     glatstp=3.0,
#     glonlim=[-180.0, 180.0],
#     glonstp=8.0,
#     option=6,
#     verbose=False,
#     ut=date.hour + date.minute / 60,
#     day=date.timetuple().tm_yday,
# )

# u = (
#     -hwm14Obj.Vwind.flatten() * WIND_FACTOR,
#     hwm14Obj.Uwind.flatten() * WIND_FACTOR,
# )
# u_lat, u_lon = np.meshgrid(
#     hwm14Obj.glatbins,
#     hwm14Obj.glonbins,
#     indexing="ij"
# )
u_lat, u_lon, u_phi, u_theta = (
    np.load("ulat.npy"),
    np.load("ulon.npy"),
    np.load("uphi.npy"),
    np.load("utheta.npy"),
)
u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing="ij")
simulation.set_neutral_wind(
    u_theta=u_theta.flatten(),
    u_phi=u_phi.flatten(),
    lat=u_lat.flatten(),
    lon=u_lon.flatten(),
    sqrt_weights=np.tile(np.sqrt(np.sin(np.deg2rad(90 - u_lat.flatten()))), (2, 1)),
)

# Get and set conductance input.
conductance_lat = simulation.geometry.model_grid.lat
conductance_lon = simulation.geometry.model_grid.lon

# STEP = 2 # number of seconds between each conductance update


Kp = 4
hall_aurora, pedersen_aurora = conductance.hardy_EUV(
    conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=False
)
simulation.set_conductance(hall_aurora, pedersen_aurora, lat=conductance_lat, lon=conductance_lon)

simulation.input_selection("conductance")
simulation.input_selection("jr")
simulation.input_selection("u")


Cfd1 = simulation.calculate_fd_curl_matrix(stencil_size=1)
Cfd2 = simulation.calculate_fd_curl_matrix(stencil_size=2)
Csh1 = simulation.calculate_sh_curl_matrix(helmholtz=True)
Csh2 = simulation.calculate_sh_curl_matrix(helmholtz=False)

E = simulation.response.get_E()


curlE_fd1 = Cfd1.dot(E)
curlE_fd2 = Cfd2.dot(E)
curlE_sh1 = Csh1.dot(E)
curlE_sh2 = Csh2.dot(E)


simulation.evolve_to_time(120)
