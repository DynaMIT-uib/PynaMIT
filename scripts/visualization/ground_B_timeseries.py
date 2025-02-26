"""Ground magnetic field time series visualization."""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pynamit
from pynamit.math.constants import RE
from pynamit.math.constants import mu0
import dipole
import datetime
import apexpy

periods = [50, 25, 10, 5, 1]
state_data_list = [
    xr.load_dataset("../simulation/oscillations/" + str(p).zfill(2) + "s_state.ncdf")
    for p in periods
]
settings_list = [
    xr.load_dataset("../simulation/oscillations/" + str(p).zfill(2) + "s_settings.ncdf")
    for p in periods
]

RI = settings_list[0].RI
sh_basis = pynamit.SHBasis(settings_list[0].Nmax, settings_list[0].Mmax)

t0 = datetime.datetime.strptime(settings_list[0].t0, "%Y-%m-%d %H:%M:%S")
d = dipole.Dipole(t0.year)
a = apexpy.Apex(t0.year)

# Construct plot grid in mlt/mlat, then convert to glat/glon.
mlt, mlat = np.meshgrid([4, 9, 12, 15, 20], [-80, -60, -20, 20, 60, 80][::-1], indexing="ij")

Nrows = mlt.shape[1]
Ncols = mlt.shape[0]

mlon = d.mlt2mlon(mlt, t0)
glat, glon, _ = a.apex2geo(mlat, mlon, 0)
glat, glon = glat.flatten(), glon.flatten()

# Calculate conversion factors.
m_ind_to_Br = -(RI**2) * sh_basis.laplacian(RI)
m_imp_to_jr = RI / mu0 * sh_basis.laplacian(RI)
W_to_dBr_dt = 1 / RI
m_ind_to_Jeq = -RI / mu0 * sh_basis.Ve_to_delta_V


ground_grid = pynamit.Grid(lat=glat, lon=glon)
ground_evaluator = pynamit.BasisEvaluator(sh_basis, ground_grid)

m_ind_to_Bh_ground = -(sh_basis.n + 1) * (RE / RI) ** sh_basis.n
m_ind_to_Br_ground = sh_basis.n * (sh_basis.n + 1) * (RE / RI) ** (sh_basis.n - 1)


fig, axes = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)

for state_data in state_data_list:
    # Calculate the time series.
    m_ind = state_data.SH_m_ind.values.T

    Br = (ground_evaluator.G * m_ind_to_Br_ground.reshape((1, -1))).dot(m_ind)
    Bh = (-ground_evaluator.G_grad * m_ind_to_Bh_ground.reshape((1, -1))).dot(m_ind)
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
    m_ind = state_data.SH_m_ind.values.T

    for i in range(25):
        axes.flatten()[i].plot(
            state_data.time.values, state_data["SH_m_imp"].values[:, i], label="$B_r$"
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

    m_ind = sd.SH_m_ind.values.T
    Br = (ground_evaluator.G * m_ind_to_Br_ground.reshape((1, -1))).dot(m_ind)

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
