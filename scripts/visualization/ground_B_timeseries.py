"""Ground magnetic field time series visualization."""

import datetime
from pathlib import Path

import apexpy
import dipole
import kompe
import matplotlib.pyplot as plt
import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.geomagnetism import decimal_year
from pynamit.storage import ArtifactStore

periods = [50, 25, 10, 5, 1]
DATA_DIRECTORY = Path("../simulation/oscillations")


def _load_period_dataset(period, name):
    run_directory = DATA_DIRECTORY / f"{period:02d}s"
    dataset = ArtifactStore(run_directory).load_dataset(name)
    if dataset is None:
        raise FileNotFoundError(f"No {name!r} artifact found in run directory {run_directory}.")
    return dataset


dynamic_data_list = [_load_period_dataset(p, "dynamic") for p in periods]
settings_list = [_load_period_dataset(p, "settings") for p in periods]

RI = settings_list[0].RI
sh_basis = kompe.SHBasis(settings_list[0].Nmax, settings_list[0].Mmax)

t0 = datetime.datetime.strptime(settings_list[0].t0, "%Y-%m-%d %H:%M:%S")
d = dipole.Dipole(decimal_year(t0))
a = apexpy.Apex(t0)

# Construct plot grid in mlt/mlat, then convert to glat/glon.
mlt, mlat = np.meshgrid([4, 9, 12, 15, 20], [-80, -60, -20, 20, 60, 80][::-1], indexing="ij")

Nrows = mlt.shape[1]
Ncols = mlt.shape[0]

mlon = d.mlt2mlon(mlt, t0)
glat, glon, _ = a.apex2geo(mlat, mlon, 0)
glat, glon = glat.flatten(), glon.flatten()

ground_grid = kompe.Grid(lat=glat, lon=glon)
ground_evaluator = kompe.SphericalTransform(sh_basis, ground_grid)

induced_Br_to_Bh_ground = -((EARTH_RADIUS_M / RI) ** sh_basis.n) / sh_basis.n
induced_Br_to_Br_ground = (EARTH_RADIUS_M / RI) ** (sh_basis.n - 1)


fig, axes = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)

for dynamic_data in dynamic_data_list:
    # Calculate the time series.
    induced_Br = dynamic_data.SH_induced_Br.values.T

    Br = (ground_evaluator.scalar_coeffs_to_grid * induced_Br_to_Br_ground.reshape((1, -1))).dot(
        induced_Br
    )
    Bh = (
        -ground_evaluator.scalar_coeffs_to_gridded_gradient
        * induced_Br_to_Bh_ground.reshape((1, -1))
    ).dot(induced_Br)
    Btheta, Bphi = np.split(Bh, 2, axis=0)

    ii, jj = np.unravel_index(np.arange(len(glat)), mlt.shape)
    for i in range(len(glat)):
        axes[jj[i], ii[i]].plot(dynamic_data.time.values, Br[i] * 1e9, label="$B_r$")
        # ax.plot(
        #    dynamic_data.time.values,
        #    Btheta[i] * 1e9,
        #    label="$B_\\theta$"
        # )
        # ax.plot(
        #    dynamic_data.time.values,
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

for dynamic_data in dynamic_data_list:
    for i in range(25):
        axes.flatten()[i].plot(
            dynamic_data.time.values, dynamic_data["SH_boundary_jr"].values[:, i], label="$j_r$"
        )


# axes[0, 0].legend(frameon = False)


fig, axesw = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)
fig, axesA = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)
fig, axesphi = plt.subplots(ncols=Ncols, nrows=Nrows, sharex=True)


for p, dynamic_data in zip(periods, dynamic_data_list, strict=True):
    sd = dynamic_data.sel(time=slice(200, None))
    t = sd.time.values

    G_fourier = np.vstack(
        (np.ones_like(t), np.cos(t / p * 2 * np.pi), np.sin(t / p * 2 * np.pi))
    ).T

    induced_Br = sd.SH_induced_Br.values.T
    Br = (ground_evaluator.scalar_coeffs_to_grid * induced_Br_to_Br_ground.reshape((1, -1))).dot(
        induced_Br
    )

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
        #    dynamic_data.time.values,
        #    Btheta[i] * 1e9,
        #    label="$B_\\theta$"
        # )
        # ax.plot(
        #    dynamic_data.time.values,
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
