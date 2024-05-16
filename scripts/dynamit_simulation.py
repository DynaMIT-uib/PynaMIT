import numpy as np
import pynamit
from pynamit import debugplot
from lompe import conductance
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import os

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 35

WIND_FACTOR = 1 # scale wind by this factor

Nmax, Mmax, Ncs = 45, 45, 65
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 20]))**2 #int(80 / Nmax)])) ** 2
print(len(rk))
rk = {'steps':rk}
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)

# u_phi   =  hwm14Obj.Uwind
# u_theta = -hwm14Obj.Vwind
# u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_lat, u_lon, u_phi, u_theta = np.load('ulat.npy'), np.load('ulon.npy'), np.load('uphi.npy'), np.load('utheta.npy')

i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSProjection(Ncs)
u_int = i2d_csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, i2d_csp.arr_theta, i2d_csp.arr_phi)
u_east_int, u_north_int, u_r_int = u_int

i2d = pynamit.I2D(i2d_sh, i2d_csp, RI, mainfield_kind = 'dipole', FAC_integration_parameters = rk, 
                                       ignore_PFAC = False, connect_hemispheres = True, latitude_boundary = latitude_boundary,
                                       zero_jr_at_dip_equator = True)

csp_grid = pynamit.grid.Grid(RI, 90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_i2d_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, csp_grid)


## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.grid.Grid(RI, lat, lon)
plt_i2d_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = -a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) / i2d.state.sinI * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_u(-u_north_int * WIND_FACTOR, u_east_int * WIND_FACTOR)
i2d.state.set_FAC(jparallel, csp_i2d_evaluator)


dt = 5e-4
count = 0
totalsteps = 500001
plotsteps = 200
fig_directory = 'figs/'

filecount = 0
time = 0.
while True:

    time = time + dt
    title = 't = {:.2f} s'.format(time)
    
    fn = os.path.join(fig_directory, 'dynamit_' + str(filecount).zfill(3) + '.png')

    if count % plotsteps == 0:
        print('count = {} saving {}'.format(count, fn))
        debugplot(i2d, title = title, filename = fn, noon_longitude = noon_longitude)
        filecount+=1

    i2d.state.evolve_Br(dt)

    count += 1
    if count > totalsteps:
        break




