import numpy as np
import pynamit
from lompe import conductance
import dipole
#import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 35

WIND_FACTOR = 1 # scale wind by this factor

result_filename = 'flareish.ncdf'
Nmax, Mmax, Ncs = 50, 50, 70
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 2]))**2 #int(80 / Nmax)])) ** 2
print(len(rk))

date = datetime.datetime(2001, 5, 12, 21, 0)
Kp   = 5
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
noon_mlon = d.mlt2mlon(12, date) # noon longitude
#hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
#                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

#u_phi   =  hwm14Obj.Uwind
#u_theta = -hwm14Obj.Vwind
#u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_lat, u_lon, u_phi, u_theta = np.load('ulat.npy'), np.load('ulon.npy'), np.load('uphi.npy'), np.load('utheta.npy')
u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')

i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSProjection(Ncs)
u_int = i2d_csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, i2d_csp.arr_theta, i2d_csp.arr_phi)
u_east_int, u_north_int, u_r_int = u_int

i2d = pynamit.I2D(result_filename = result_filename, sh = i2d_sh, csp = i2d_csp, RI = RI, mainfield_kind = 'igrf', FAC_integration_steps = rk,
                                       ignore_PFAC = False, connect_hemispheres = True, latitude_boundary = latitude_boundary,
                                       zero_jr_at_dip_equator = True, ih_constraint_scaling = 1e-5)

csp_grid = pynamit.Grid(90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, csp_grid)
csp_b_evaluator = pynamit.FieldEvaluator(i2d.state.mainfield, csp_grid, RI)


## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat, lon)
plt_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = False)
i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

apx = apexpy.Apex(refh = (RI - RE) * 1e-3, date = 2020)
mlat, mlon = apx.geo2apex(csp_grid.lat, csp_grid.lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)

_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI-RE)*1e-3) # fix this

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = a.get_upward_current(mlat = mlat, mlt = mlt) / csp_b_evaluator.br * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_u(-u_north_int * WIND_FACTOR, u_east_int * WIND_FACTOR)
i2d.state.set_FAC(jparallel, csp_i2d_evaluator)

i2d.evolve_to_time(180)

print('increasing conductance')
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, F107 = 300, starlight = 1, dipole = False)
i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

i2d.evolve_to_time(360)

