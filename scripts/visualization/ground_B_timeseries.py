"""Ground magnetic field time series visualization."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pynamit
from pynamit.math.constants import RE
import dipole
import datetime
import apexpy
from pynamit.simulation.data import SimulationData

periods = [50, 25, 10, 5, 1]
prefixes = [
    Path("../simulation/oscillations") / f"{str(p).zfill(2)}s"
    for p in periods
]
simulation_data_list = [SimulationData.from_prefix(prefix) for prefix in prefixes]
state_data_list = [simulation_data.get_dataset("state") for simulation_data in simulation_data_list]
m_ind_name = simulation_data_list[0].get_data_var_name("state", "m_ind")
m_imp_name = simulation_data_list[0].get_data_var_name("state", "m_imp")

settings = simulation_data_list[0].settings
RI = settings.RI
sh_basis = simulation_data_list[0].sh_basis
mean_free_degrees = sh_basis.scalar_degrees(mean_free=True)

t0 = datetime.datetime.strptime(settings.t0, "%Y-%m-%d %H:%M:%S")
d = dipole.Dipole(t0.year)
a = apexpy.Apex(t0.year)

# Construct plot grid in mlt/mlat, then convert to glat/glon.
mlt, mlat = np.meshgrid([4, 9, 12, 15, 20], [-80, -60, -20, 20, 60, 80][::-1], indexing="ij")

Nrows = mlt.shape[1]
Ncols = mlt.shape[0]

mlon = d.mlt2mlon(mlt, t0)
glat, glon, _ = a.apex2geo(mlat, mlon, 0)
glat, glon = glat.flatten(), glon.flatten()

ground_grid = pynamit.Grid(lat=glat, lon=glon)
# Removed BasisEvaluator. Logic will use basis and grid directly.

m_ind_to_Bh_ground = -(mean_free_degrees + 1) * (RE / RI) ** mean_free_degrees
m_ind_to_Br_ground = (
    mean_free_degrees
    * (mean_free_degrees + 1)
    * (RE / RI) ** (mean_free_degrees - 1)
)


fig, axes = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)

for state_data in state_data_list:
    # Calculate the time series.
    m_ind = state_data[m_ind_name].values.T

    Br = (
        sh_basis.get_evaluation_matrix(ground_grid, mean_free=True)
        * m_ind_to_Br_ground.reshape((1, -1))
    ).dot(m_ind)
    Bh = (
        -sh_basis.get_gradient_matrix(ground_grid, mean_free=True)
        * m_ind_to_Bh_ground.reshape((1, -1))
    ).dot(m_ind)
    Btheta, Bphi = np.split(Bh, 2, axis=0)

    ii, jj = np.unravel_index(np.arange(len(glat)), mlt.shape)
    for i in range(len(glat)):
        axes[jj[i], ii[i]].plot(state_data.time.values, Br[i] * 1e9, label="$B_r$")
        # ax.plot(
        #    state_data.time.values,
        #    Btheta[i] * 1e9,
        #    label="$B_\\theta$"
        # )
        # ax.plot(
        #    state_data.time.values,
        #    Bphi[i] * 1e9,
        #    label="$B_\phi$"
        # )
        if jj[i] == 0:
            axes[jj[i], ii[i]].set_title("MLT$ = " + str(mlt[ii[i], jj[i]]) + "$")

        if ii[i] == Ncols - 1:
            axes[jj[i], ii[i]].set_ylabel(
                "mlat$ = " + str(mlat[ii[i], jj[i]]) + r"^\circ$", rotation=270, labelpad=15
            )

            # axes[jj[i], ii[i]].set_title(
            #    "mlat = " + str(mlat[ii[i], jj[i]]), loc="right"
            # )
            axes[jj[i], ii[i]].yaxis.set_label_position("right")


fig, axes = plt.subplots(ncols=5, nrows=5, sharex=True)

for state_data in state_data_list:
    # calculate the time series:
    m_ind = state_data[m_ind_name].values.T

    for i in range(25):
        axes.flatten()[i].plot(
            state_data.time.values, state_data[m_imp_name].values[:, i], label="$B_r$"
        )


# axes[0, 0].legend(frameon = False)


fig, axesw = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)
fig, axesA = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)
fig, axesphi = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)


for p, state_data in zip(periods, state_data_list):
    sd = state_data.sel(time=slice(200, None))
    t = sd.time.values

    G_fourier = np.vstack(
        (np.ones_like(t), np.cos(t / p * 2 * np.pi), np.sin(t / p * 2 * np.pi))
    ).T

    m_ind = sd[m_ind_name].values.T
    Br = (
        sh_basis.get_evaluation_matrix(ground_grid, mean_free=True)
        * m_ind_to_Br_ground.reshape((1, -1))
    ).dot(m_ind)

    # Fit the wave parameters.
    m = np.linalg.lstsq(G_fourier, Br.T)[0]
    A = np.sqrt(m[1] ** 2 + m[2] ** 2)
    phi = np.rad2deg(np.arctan2(-m[2], m[1]))

    ii, jj = np.unravel_index(np.arange(len(glat)), mlt.shape)
    for i in range(len(glat)):
        axesw[jj[i], ii[i]].plot(t, Br[i] * 1e9, label="$B_r$")
        axesw[jj[i], ii[i]].plot(t, G_fourier.dot(m.T[i]) * 1e9, linestyle="--")
        axesA[jj[i], ii[i]].scatter(p, A[i] * 1e9, color="black", marker="x")
        axesphi[jj[i], ii[i]].scatter(p, phi[i], color="black")

        # ax.plot(
        #    state_data.time.values,
        #    Btheta[i] * 1e9,
        #    label="$B_\\theta$"
        # )
        # ax.plot(
        #    state_data.time.values,
        #    Bphi[i] * 1e9,
        #    label="$B_\phi$"
        # )

        if jj[i] == 0:
            axesw[jj[i], ii[i]].set_title("MLT$ = " + str(mlt[ii[i], jj[i]]) + "$")
            axesA[jj[i], ii[i]].set_title("MLT$ = " + str(mlt[ii[i], jj[i]]) + "$")
            axesphi[jj[i], ii[i]].set_title("MLT$ = " + str(mlt[ii[i], jj[i]]) + "$")

        if ii[i] == Ncols - 1:
            axesw[jj[i], ii[i]].set_ylabel(
                "mlat$ = " + str(mlat[ii[i], jj[i]]) + r"^\circ$", rotation=270, labelpad=15
            )
            axesw[jj[i], ii[i]].yaxis.set_label_position("right")
            axesA[jj[i], ii[i]].set_ylabel(
                "mlat$ = " + str(mlat[ii[i], jj[i]]) + r"^\circ$", rotation=270, labelpad=15
            )
            axesA[jj[i], ii[i]].yaxis.set_label_position("right")
            axesphi[jj[i], ii[i]].set_ylabel(
                "mlat$ = " + str(mlat[ii[i], jj[i]]) + r"^\circ$", rotation=270, labelpad=15
            )
            axesphi[jj[i], ii[i]].yaxis.set_label_position("right")


plt.show()
