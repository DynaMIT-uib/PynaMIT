"""Script for running and debugging PynaMIT simulation."""

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import os
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance

reload(pynamit)

IGNORE_PFAC = False
CONNECT_HEMISPHERES = False

# Define simulation parameters.
Nmax, Mmax, Ncs = 15, 3, 20
RI = (6371.2 + 110) * 1e3
dt = 5e-4
totalsteps = 20001

dataset_filename_prefix = "simulation_test"

# Define plotting parameters.
plotsteps = 200
fig_directory = "figs/"
Blevels = np.linspace(-50, 50, 22) * 1e-9  # color levels for Br
levels = np.linspace(-0.9, 0.9, 22)  # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100)  # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:2.5]

# Set up simulation object.
dynamics = pynamit.Dynamics(
    dataset_filename_prefix=dataset_filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="dipole",
    ignore_PFAC=IGNORE_PFAC,
    connect_hemispheres=CONNECT_HEMISPHERES,
)

# Get and set conductance input.
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date)  # noon longitude

conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(
    dynamics.state_grid.lon, dynamics.state_grid.lat, Kp, date, starlight=1, dipole=True
)
dynamics.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

# Get and set jr input.
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

dynamics.update_conductance()
dynamics.update_jr()
dynamics.state.update_m_imp()
dynamics.state.update_E()


# Make an integration matrix.

# cnm = SHIndices(Nmax, Mmax).setNmin(1).MleN()
# snm = SHIndices(Nmax, Mmax).setNmin(1).MleN().Mge(1)
# Equivalent to pynamit.get_Schmidt_normalization(cnm).T?
# cS =  (2 * cnm.n.T + 1) / (4 * np.pi * RI**2)
# Equivalent to pynamit.get_Schmidt_normalization(snm).T?
# sS =  (2 * snm.n.T + 1) / (4 * np.pi * RI**2)
# Ginv = (
#     dynamics.Gnum.T * np.vstack((cS, sS))
#     * dynamics.cs_basis.unit_area
# )
# gg = Ginv.dot(dynamics.Gnum)

# Set up plotting grid and evaluators.
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat=lat, lon=lon)
plt_state_evaluator = pynamit.BasisEvaluator(dynamics.state_basis, plt_grid)
nnn = plt_grid.lat.flatten() > 50
sss = plt_grid.lat.flatten() < -50


# Run the simulation.
coeffs = []
count = 0
filecount = 1
time = 0
while True:
    dynamics.state.evolve_Br(dt)
    time = time + dt
    coeffs.append(dynamics.state.m_ind.coeffs)
    count += 1
    # print(
    #    count,
    #    time,
    #    (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3],
    # )

    if count % plotsteps == 0:
        print(
            count, time, (dynamics.state.m_ind.coeffs * dynamics.state.m_ind_to_Br)[:3]
        )
        fn = os.path.join(fig_directory, "new_" + str(filecount).zfill(3) + ".png")
        filecount += 1
        title = "t = {:.3} s".format(time)
        Br = dynamics.state.get_Br(plt_state_evaluator)
        fig, paxn, paxs, axg = pynamit.globalplot(
            plt_grid.lon,
            plt_grid.lat,
            Br.reshape(plt_grid.lat.shape),
            title=title,
            returnplot=True,
            levels=Blevels,
            cmap="bwr",
            noon_longitude=lon0,
            extend="both",
        )

        W = dynamics.state.get_W(plt_state_evaluator) * 1e-3

        dynamics.state.update_E()
        Phi = dynamics.state.get_Phi(plt_state_evaluator) * 1e-3

        # paxn.contour(
        #    dynamics.state_grid.lat.flatten()[nnn],
        #    (dynamics.state_grid.lon.flatten() - lon0)[nnn] / 15,
        #    W[nnn],
        #    colors="black",
        #    levels=Wlevels,
        #    linewidths=0.5,
        # )
        # paxs.contour(
        #    dynamics.state_grid.lat.flatten()[sss],
        #    (dynamics.state_grid.lon.flatten() - lon0)[sss] / 15,
        #    W[sss],
        #    colors="black",
        #    levels=Wlevels,
        #    linewidths=0.5,
        # )
        paxn.contour(
            plt_grid.lat.flatten()[nnn],
            (plt_grid.lon.flatten() - lon0)[nnn] / 15,
            Phi[nnn],
            colors="black",
            levels=Philevels,
            linewidths=0.5,
        )
        paxs.contour(
            plt_grid.lat.flatten()[sss],
            (plt_grid.lon.flatten() - lon0)[sss] / 15,
            Phi[sss],
            colors="black",
            levels=Philevels,
            linewidths=0.5,
        )
        plt.savefig(fn)
        plt.close()

    if count > totalsteps:
        break
