"""Simulation."""

import numpy as np
import pynamit
import dipole
import datetime
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix
import os
import cartopy.crs as ccrs
from polplot import Polarplot
import matplotlib.pyplot as plt



RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40
dt = 300

FLOAT_ERROR_MARGIN = 1e-6

dataset_filename_prefix = "mage-forcing"
Nmax, Mmax, Ncs = 20, 20, 20
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2

date = datetime.datetime(2001, 5, 12, 17, 0)
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date)  # Noon longitude
noon_mlon = d.mlt2mlon(12, date)  # Noon longitude

# Set up simulation object.
dynamics = pynamit.Dynamics(
    dataset_filename_prefix=dataset_filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="dipole",
    FAC_integration_steps=rk,
    ignore_PFAC=False,
    connect_hemispheres=True,
    latitude_boundary=latitude_boundary,
    ih_constraint_scaling=1e-5,
    t0=str(date),
)

mage_dir = "./mage_data/"
mage_tag = "msphere"

gsph = msph.GamsphPipe(mage_dir, mage_tag, doFast=False)
nstep = gsph.sFin

mixFiles = os.path.join(mage_dir, "%s.mix.h5" % (mage_tag))

for step in range(0, nstep):
    ion_north = remix.remix(mixFiles, step)
    ion_north.init_vars("NORTH")

    ion_south = remix.remix(mixFiles, step)
    ion_south.init_vars("SOUTH")

    if step == 0:
        # In units of RionE?
        north_r = ion_north.ion["R"]
        south_r = ion_south.ion["R"]

        # Uses convention from remix.efield().
        north_theta = np.rad2deg(np.arcsin(north_r))
        south_theta = np.rad2deg(np.arcsin(south_r))

        north_phi = np.rad2deg(ion_north.ion["THETA"])
        south_phi = np.rad2deg(ion_south.ion["THETA"])

        # Theta and phi are staggered, so we need to shift them by half
        # a grid point.
        north_theta_centered = (
            north_theta[:-1, :-1] + 0.5 * np.diff(north_theta, axis=0)[:, :-1]
        )
        north_phi_centered = (
            north_phi[:-1, :-1] + 0.5 * np.diff(north_phi, axis=1)[:-1, :]
        )
        south_theta_centered = (
            south_theta[:-1, :-1] + 0.5 * np.diff(south_theta, axis=0)[:, :-1]
        )
        south_phi_centered = (
            south_phi[:-1, :-1] + 0.5 * np.diff(south_phi, axis=1)[:-1, :]
        )

        full_theta_centered = np.concatenate(
            (north_theta_centered, 180 - south_theta_centered)
        )
        full_phi_centered = np.concatenate(
            (north_phi_centered, south_phi_centered)
        )

    # Fetch the ionospheric fields.
    north_current = ion_north.variables["current"]["data"]
    south_current = ion_south.variables["current"]["data"]
    full_current = np.concatenate((north_current, south_current))

    north_conductance_pedersen = ion_north.variables["sigmap"]["data"]
    south_conductance_pedersen = ion_south.variables["sigmap"]["data"]
    full_conductance_pedersen = np.concatenate(
        (north_conductance_pedersen, south_conductance_pedersen)
    )

    north_conductance_hall = ion_north.variables["sigmah"]["data"]
    south_conductance_hall = ion_south.variables["sigmah"]["data"]
    full_conductance_hall = np.concatenate(
        (north_conductance_hall, south_conductance_hall)
    )

    # Get and set jr input.
    dynamics.set_jr(
        full_current.flatten(), theta=full_theta_centered.flatten(), phi=full_phi_centered.flatten(), time=dt * step
    )
    dynamics.set_conductance(
        full_conductance_hall.flatten(),
        full_conductance_pedersen.flatten(),
        theta=full_theta_centered.flatten(),
        phi=full_phi_centered.flatten(),
        time=dt * step,
    )

    plotting = False
    if plotting:

        # PLOTTING
        fig = plt.figure(figsize=(10, 6))

        paxn_imposed = Polarplot(plt.subplot2grid((2, 4), (0, 0)))
        paxs_imposed = Polarplot(plt.subplot2grid((2, 4), (0, 1)))
        paxn_induced = Polarplot(plt.subplot2grid((2, 4), (0, 2)))
        paxs_induced = Polarplot(plt.subplot2grid((2, 4), (0, 3)))

        global_projection = ccrs.PlateCarree(central_longitude=0)
        gax_imposed = plt.subplot2grid((2, 2), (1, 0), projection=global_projection, rowspan=2)
        gax_induced = plt.subplot2grid((2, 2), (1, 1), projection=global_projection, rowspan=2)

        for ax in [gax_imposed, gax_induced]:
            ax.coastlines(zorder=2, color="grey")

        lat = 90 - full_theta_centered
        lon = full_phi_centered
        #jr_kwargs = {
        #    "cmap": plt.cm.bwr,
        #    "levels": np.linspace(-0.95, 0.95, 22) / 6 * 1e-6,
        #    "extend": "both",
        #}

        jr_kwargs = {
            "cmap": plt.cm.bwr,
            "levels": np.linspace(-0.95, 0.95, 22) / 6 * 1e-6,
            "extend": "both",
        }

        contours = {}
        contours["jr "] = gax_imposed.contourf(
            lon, lat, full_current.reshape(lon.shape), transform=ccrs.PlateCarree(), **jr_kwargs
        )

        # north:
        nnn = lat > 50
        contours["jr_n"] = paxn_imposed.contourf(
            lat[nnn], lon[nnn] / 15, full_current.reshape(lon.shape)[nnn], **jr_kwargs
        )

        # south:
        sss = lat < -50
        contours["jr_s"] = paxs_imposed.contourf(
            lat[sss], lon[sss] / 15, full_current.reshape(lon.shape)[sss], **jr_kwargs
        )

        plt.show()

final_time = 3600  # seconds
dynamics.evolve_to_time(final_time)
