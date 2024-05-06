""" Script to check if the PFAC calculation gives a reasonable result """

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import secsy
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance
import os

COMPARE_TO_SECS = True
SIMULATE_DYNAMIC_RESPONSE = False

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3

# MODEL PARAMETERS
Nmax, Mmax, Ncs = 25, 20, 30

## PLOT PARAMETERS
fig_directory = 'figs/'
Blevels = np.linspace(-5, 5, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSprojection(Ncs)
i2d = pynamit.I2D(i2d_sh, i2d_csp, RI, mainfield_kind = 'dipole', FAC_integration_parameters = {'steps':np.logspace(np.log10(RI), np.log10(7 * RE), 11)}, ignore_PFAC = False)

## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.grid.grid(RI, lat, lon)
plt_sh_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d_sh, plt_grid)

## CONDUCTANCE AND FAC INPUT:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude
hall, pedersen = conductance.hardy_EUV(i2d.num_grid.lon, i2d.num_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = -a.get_upward_current(mlat = i2d.num_grid.lat, mlt = d.mlon2mlt(i2d.num_grid.lon, date)) / i2d.state.sinI * 1e-6
jparallel[np.abs(i2d.num_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_FAC(jparallel)
GBr = plt_sh_evaluator.G * i2d_sh.n / i2d.num_grid.RI
Br_I2D = GBr.dot(i2d.state.shc_PFAC)


if SIMULATE_DYNAMIC_RESPONSE:

    fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br_I2D.reshape(plt_grid.lat.shape), returnplot = True, 
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

    plt.savefig('figs/PFAC_steady_state.png')


    # manipulate GTB to remove the r x grad(T) part:
    GrxgradT = -i2d.num_grid.Gdf * i2d.num_grid.RI
    i2d.state.GTB = i2d.state.GTB - GrxgradT # subtract GrxgradT off


    ## RUN SIMULATION
    plotsteps = 400
    fig_directory = 'figs/'
    dt = 1e-3
    totalsteps = 2001
    coeffs = []
    count = 0
    filecount = 1
    time = 0
    while True:

        i2d.state.evolve_Br(dt)
        time = time + dt
        coeffs.append(i2d.state.shc_VB)
        count += 1
        #print(count, time, i2d.shc_Br[:3])

        if count % plotsteps == 0:
            print(count, time, i2d.state.shc_Br[:3])
            fn = os.path.join(fig_directory, 'PFAC_' + str(filecount).zfill(3) + '.png')
            filecount +=1
            title = 't = {:.3} s'.format(time)
            Br = i2d.state.get_Br(plt_grid)
            fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br.reshape(plt_grid.lat.shape) , title = title, returnplot = True, 
                                                       levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

            W = plt_sh_evaluator.to_grid(i2d.state.shc_EW) * 1e-3

            shc_Phi = i2d.num_grid.vector_to_shc_cf.dot(np.hstack(i2d.state.get_E(i2d.num_grid))) # find coefficients for electric potential
            Phi = plt_sh_evaluator.to_grid(shc_Phi) * 1e-3

            #paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
            #paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
            plt.savefig(fn)






if COMPARE_TO_SECS:
    print('Building SECS matrices. This takes some time (and memory) because of global grids...')
    secsI = jparallel * i2d.state.sinI * i2d_csp.unit_area * i2d.num_grid.RI**2 # SECS amplitudes are downward current density times area
    lat, lon = plt_grid.lat.flatten(), plt_grid.lon.flatten()
    r = np.full(lat.size, i2d.num_grid.RI - 1)
    lat_secs, lon_secs = i2d.num_grid.lat, i2d.num_grid.lon
    Be, Bn, Br = i2d.state.bphi, - i2d.state.btheta, i2d.state.br
    Ge, Gn, Gu = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(lat, lon, r, lat_secs, lon_secs, Be, Bn, Br, RI = i2d.num_grid.RI)


    Br_SECS = Gu.dot(secsI)

    fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br_SECS.reshape(plt_grid.lat.shape), returnplot = True, 
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

    fig, ax = plt.subplots(figsize = (10, 10))
    ax.scatter(Br_SECS, Br_I2D)
    brm = np.max(np.abs(Br_I2D))
    ax.plot([-brm, brm], [-brm, brm], 'r-')
    ax.set_aspect('equal')
    ax.set_xlim(-brm, brm)
    ax.set_xlabel('straight tilted SECS')
    ax.set_ylabel('Spherical harmonics')


    plt.show()
