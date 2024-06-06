""" Script to check if the interhemispheric connection works """

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance
import pyhwm2014 # https://github.com/rilma/pyHWM14
import cartopy.crs as ccrs
import os

PLOT_WIND = False # True to make a plot of the wind field
SIMULATE = True

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

# MODEL PARAMETERS
Nmax, Mmax, Ncs = 25, 15, 50
print('we need a check that the poloidal field calculation is high enough resoultion compared ot SH ')


rk = RI / np.cos(np.deg2rad(np.linspace(0, 70, int(360 / (Nmax  + .5)) + 1))) ** 2
#rk = np.hstack((rk, np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)[5:] / RE))

rk = {'steps':rk}


# PARAMETERS FOR EMPIRICAL MODELS:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude

hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)
u_phi   =  hwm14Obj.Uwind
u_theta = -hwm14Obj.Vwind
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSProjection(Ncs)

u_int = i2d_csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, i2d_csp.arr_theta, i2d_csp.arr_phi)
u_east_int, u_north_int, u_r_int = u_int

if PLOT_WIND:
    fig, ax = plt.subplots(figsize=(10, 7),
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
    ax.coastlines()
    Q = ax.quiver(u_lon.flatten(), u_lat.flatten(), u_phi.flatten(), -u_theta.flatten(), color='blue', transform=ccrs.PlateCarree())
    ax.quiver(i2d_csp.arr_phi, 90 - i2d_csp.arr_theta, u_east_int, u_north_int, color = 'red', scale = Q.scale, transform=ccrs.PlateCarree() )


## PLOT PARAMETERS
fig_directory = 'figs/'
Blevels = np.linspace(-5, 5, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT

i2d = pynamit.I2D(i2d_sh, i2d_csp, RI, mainfield_kind = 'dipole', FAC_integration_parameters = rk, 
                                       ignore_PFAC = False, connect_hemispheres = True, latitude_boundary = latitude_boundary)

csp_grid = pynamit.Grid(90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_i2d_evaluator = pynamit.BasisEvaluator(i2d_sh, csp_grid)
csp_b_evaluator = pynamit.FieldEvaluator(i2d.state.mainfield, csp_grid, RI)

## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat, lon)
plt_i2d_evaluator = pynamit.BasisEvaluator(i2d_sh, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) / csp_b_evaluator.br * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_u(-u_north_int, u_east_int)
i2d.state.set_FAC(jparallel, csp_i2d_evaluator)

GBr = plt_i2d_evaluator.scaled_G(i2d_sh.n / RI)
Br_I2D = GBr.dot(i2d.state.m_imp_to_B_pol.dot(i2d.state.m_imp.coeffs))


if SIMULATE:
    dt = 5e-4
    totalsteps = 200001
    ## PLOT PARAMETERS
    plotsteps = 500
    fig_directory = 'figs/'
    Blevels = np.linspace(-50, 50, 22) * 1e-9 # color levels for Br
    levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
    c_levels = np.linspace(0, 20, 100) # color levels for conductance
    Wlevels = np.r_[-512.5:512.5:5]
    Philevels = np.r_[-212.5:212.5:2.5]


    ## RUN SIMULATION
    coeffs = []
    count = 0
    filecount = 1
    time = 0
    while True:

        i2d.state.evolve_Br(dt)
        time = time + dt
        coeffs.append(i2d.state.m_ind.coeffs)
        count += 1
        #print(count, time, (i2d.state.m_ind.coeffs * i2d.state.m_ind_to_Br)[:3])

        if count % plotsteps == 0:
            print(count, time, (i2d.state.m_ind.coeffs * i2d.state.m_ind_to_Br)[:3])
            fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
            filecount +=1
            title = 't = {:.3} s'.format(time)
            Br = i2d.state.get_Br(plt_i2d_evaluator)
            fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br.reshape(plt_grid.lat.shape) , title = title, returnplot = True, 
                                                       levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

            W = i2d.state.get_W(plt_i2d_evaluator) * 1e-3

            i2d.state.update_Phi_and_EW()
            Phi = i2d.state.get_Phi(plt_i2d_evaluator) * 1e-3

            #paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
            #paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
            plt.savefig(fn)
            plt.close()

        if count > totalsteps:
            break

else:
    fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br_I2D.reshape(plt_grid.lat.shape), returnplot = True, 
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')



    plt.show()
    plt.close()