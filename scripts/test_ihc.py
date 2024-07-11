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

result_filename_prefix = 'ihc_test'

# MODEL PARAMETERS
Nmax, Mmax, Ncs = 25, 15, 50
print('we need a check that the poloidal field calculation is high enough resoultion compared ot SH ')


rk = RI / np.cos(np.deg2rad(np.linspace(0, 70, int(360 / (Nmax  + .5)) + 1))) ** 2
#rk = np.hstack((rk, np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)[5:] / RE))


# PARAMETERS FOR EMPIRICAL MODELS:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude


## PLOT PARAMETERS
fig_directory = 'figs/'
Blevels = np.linspace(-5, 5, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
i2d = pynamit.Dynamics(result_filename_prefix = result_filename_prefix,
                       Nmax = Nmax,
                       Mmax = Mmax,
                       Ncs = Ncs,
                       RI = RI,
                       mainfield_kind = 'dipole',
                       FAC_integration_steps = rk,
                       ignore_PFAC = False,
                       connect_hemispheres = True,
                       latitude_boundary = latitude_boundary)

## CONDUCTANCE INPUT
conductance_lat = i2d.num_grid.lat
conductance_lon = i2d.num_grid.lon
hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = True)
i2d.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

## jr INPUT
jr_lat = i2d.num_grid.lat
jr_lon = i2d.num_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jr = a.get_upward_current(mlat = jr_lat, mlt = d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr
i2d.set_jr(jr, lat = jr_lat, lon = jr_lon)

## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)

u = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
i2d.set_u(u, lat = u_lat, lon = u_lon)

i2d.update_conductance()
i2d.update_u()
i2d.update_jr()
i2d.state.impose_constraints()
i2d.state.update_Phi_and_W()

## SET UP PLOTTING GRID AND EVALUATORS
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat = lat, lon = lon)
i2d_sh = pynamit.SHBasis(Nmax, Mmax)
plt_i2d_evaluator = pynamit.BasisEvaluator(i2d_sh, plt_grid)

GBr = plt_i2d_evaluator.scaled_G(i2d_sh.n / RI)
Br_I2D = GBr.dot(i2d.state.m_imp_to_B_pol.dot(i2d.state.m_imp.coeffs))


if PLOT_WIND:
    u_basis_evaluator = pynamit.BasisEvaluator(i2d_sh, pynamit.Grid(lat = u_lat, lon = u_lon))

    u_theta_sh = pynamit.Vector(i2d_sh, basis_evaluator = u_basis_evaluator, grid_values = u[0])
    u_phi_sh   = pynamit.Vector(i2d_sh, basis_evaluator = u_basis_evaluator, grid_values = u[1])

    u_theta_int = u_theta_sh.to_grid(i2d.basis_evaluator)
    u_phi_int   = u_phi_sh.to_grid(i2d.basis_evaluator)

    fig, ax = plt.subplots(figsize=(10, 7),
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude = lon0)})
    ax.coastlines()
    Q = ax.quiver(u_lon.flatten(), u_lat.flatten(), u[1].flatten(), -u[0].flatten(), color='blue', transform=ccrs.PlateCarree())
    ax.quiver(i2d.num_grid.lon, i2d.num_grid.lat, u_phi_int, -u_theta_int, color = 'red', scale = Q.scale, transform=ccrs.PlateCarree() )


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

            i2d.state.update_Phi_and_W()
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