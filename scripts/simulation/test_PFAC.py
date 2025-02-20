"""Script to check if the PFAC calculation gives a reasonable result"""

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

dataset_filename_prefix = 'PFAC_test'

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
dynamics = pynamit.Dynamics(dataset_filename_prefix = dataset_filename_prefix,
                            Nmax = Nmax,
                            Mmax = Mmax,
                            Ncs = Ncs,
                            RI = RI,
                            mainfield_kind = 'dipole',
                            FAC_integration_steps = np.logspace(np.log10(RI), np.log10(7 * RE), 11),
                            ignore_PFAC = False)

## CONDUCTANCE INPUT
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude

conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = True)
dynamics.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

## jr INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jr = a.get_upward_current(mlat = jr_lat, mlt = d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr
dynamics.set_jr(jr, lat = jr_lat, lon = jr_lon)

dynamics.update_conductance()
dynamics.update_jr()
dynamics.state.update_m_imp()
dynamics.state.update_E()

## SET UP PLOTTING GRID AND EVALUATORS
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat = lat, lon = lon)
plt_state_evaluator = pynamit.BasisEvaluator(dynamics.state_basis, plt_grid)

G_Br = plt_state_evaluator.scaled_G(dynamics.state.m_ind_to_Br / dynamics.state.RI)
Br = G_Br.dot(dynamics.state.m_imp_to_B_pol.dot(dynamics.state.m_imp.coeffs))


if SIMULATE_DYNAMIC_RESPONSE:

    fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br.reshape(plt_grid.lat.shape), returnplot = True,
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

    plt.savefig('figs/PFAC_steady_state.png')
    plt.close()


    # manipulate GTB to remove the r x grad(T) part:
    GrxgradT = -dynamics.state_basis_evaluator.Gdf * RI
    dynamics.state.GTB = dynamics.state.GTB - GrxgradT # subtract GrxgradT off


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

        dynamics.state.evolve_Br(dt)
        time = time + dt
        coeffs.append(dynamics.state.m_ind.coeffs)
        count += 1
        #print(count, time, (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3])

        if count % plotsteps == 0:
            print(count, time, (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3])
            fn = os.path.join(fig_directory, 'PFAC_' + str(filecount).zfill(3) + '.png')
            filecount +=1
            title = 't = {:.3} s'.format(time)
            Br = dynamics.state.get_Br(plt_state_evaluator)
            fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br.reshape(plt_grid.lat.shape) , title = title, returnplot = True,
                                                       levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

            W = dynamics.state.get_W(plt_state_evaluator) * 1e-3

            dynamics.state.update_E()
            Phi = dynamics.state.get_Phi(plt_state_evaluator) * 1e-3

            #paxn.contour(dynamics.state_grid.lat.flatten()[nnn], (dynamics.state_grid.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxs.contour(dynamics.state_grid.lat.flatten()[sss], (dynamics.state_grid.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
            #paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
            #paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
            plt.savefig(fn)
            plt.close()






if COMPARE_TO_SECS:
    print('Building SECS matrices. This takes some time (and memory) because of global grids...')
    secsI = -jr * dynamics.cs_basis.unit_area * RI**2 # SECS amplitudes are downward current density times area
    lat, lon = plt_grid.lat.flatten(), plt_grid.lon.flatten()
    r = np.full(lat.size, RI - 1)
    lat_secs, lon_secs = dynamics.state_grid.lat, dynamics.state_grid.lon
    b_evaluator = pynamit.FieldEvaluator(dynamics.state.mainfield, pynamit.Grid(lat = lat_secs, lon = lon_secs), RI)
    Be, Bn, Br = b_evaluator.bphi, - b_evaluator.btheta, b_evaluator.br
    Ge, Gn, Gu = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(lat, lon, r, lat_secs, lon_secs, Be, Bn, Br, RI = RI)


    Br_SECS = Gu.dot(secsI)

    fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br_SECS.reshape(plt_grid.lat.shape), returnplot = True,
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

    fig, ax = plt.subplots(figsize = (10, 10))
    ax.scatter(Br_SECS, Br)
    brm = np.max(np.abs(Br))
    ax.plot([-brm, brm], [-brm, brm], 'r-')
    ax.set_aspect('equal')
    ax.set_xlim(-brm, brm)
    ax.set_xlabel('straight tilted SECS')
    ax.set_ylabel('Spherical harmonics')


    plt.show()
    plt.close()
