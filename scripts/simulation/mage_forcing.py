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


RE = 6381e3
RI = 6.5e3
latitude_boundary = 35
dt = 300

dataset_filename_prefix = "mage-forcing"
Nmax, Mmax, Ncs = 30, 30, 30
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2

date = datetime.datetime(2013, 3, 17, 10)
d = dipole.Dipole(date.year)

# Set up simulation object.
dynamics = pynamit.Dynamics(
    dataset_filename_prefix=dataset_filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    mainfield_kind="dipole",
    FAC_integration_steps=rk,
    ignore_PFAC=True,
    connect_hemispheres=False,
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
        north_theta_centered = north_theta[:-1, :-1] + 0.5 * np.diff(north_theta, axis=0)[:, :-1]
        north_phi_centered = north_phi[:-1, :-1] + 0.5 * np.diff(north_phi, axis=1)[:-1, :]
        south_theta_centered = south_theta[:-1, :-1] + 0.5 * np.diff(south_theta, axis=0)[:, :-1]
        south_phi_centered = south_phi[:-1, :-1] + 0.5 * np.diff(south_phi, axis=1)[:-1, :]

        full_theta_centered = np.concatenate((north_theta_centered, 180 - south_theta_centered))
        full_phi_centered = np.concatenate((north_phi_centered, south_phi_centered))

    # Fetch the ionospheric fields.
    north_current = ion_north.variables["current"]["data"] * 1e-6 # Convert from muA/m^2 to A/m^2
    south_current = ion_south.variables["current"]["data"] * 1e-6
    full_current = np.concatenate((north_current, south_current))

    north_conductance_pedersen = ion_north.variables["sigmap"]["data"]
    south_conductance_pedersen = ion_south.variables["sigmap"]["data"]
    full_conductance_pedersen = np.concatenate(
        (north_conductance_pedersen, south_conductance_pedersen)
    )

    north_conductance_hall = ion_north.variables["sigmah"]["data"]
    south_conductance_hall = ion_south.variables["sigmah"]["data"]
    full_conductance_hall = np.concatenate((north_conductance_hall, south_conductance_hall))

    # Get and set jr input.
    dynamics.set_FAC(
        full_current.flatten(),
        theta=full_theta_centered.flatten(),
        phi=full_phi_centered.flatten(),
        time=dt * step,
    )
    dynamics.set_conductance(
        full_conductance_hall.flatten(),
        full_conductance_pedersen.flatten(),
        theta=full_theta_centered.flatten(),
        phi=full_phi_centered.flatten(),
        time=dt * step,
    )

    plotting = True
    if plotting:
        dynamics.select_timeseries_data("jr")
        minlat = 0
        # PLOTTING
        fig = plt.figure(figsize=(10, 6))

        paxn_input = Polarplot(plt.subplot2grid((2, 4), (0, 0)), minlat=minlat)
        paxs_input = Polarplot(plt.subplot2grid((2, 4), (0, 1)), minlat=minlat)
        paxn_interpolated = Polarplot(plt.subplot2grid((2, 4), (0, 2)), minlat=minlat)
        paxs_interpolated = Polarplot(plt.subplot2grid((2, 4), (0, 3)), minlat=minlat)

        global_projection = ccrs.PlateCarree(central_longitude=0)
        gax_input = plt.subplot2grid((2, 2), (1, 0), projection=global_projection, rowspan=2)
        gax_interpolated = plt.subplot2grid((2, 2), (1, 1), projection=global_projection, rowspan=2)

        for ax in [gax_input, gax_interpolated]:
            ax.coastlines(zorder=2, color="grey")

        lat = 90 - full_theta_centered
        lon = full_phi_centered
        jr_kwargs = {
           "cmap": plt.cm.bwr,
           "levels": np.linspace(-0.95, 0.95, 22) / 6 * 1e-6,
           "extend": "both",
        }

        # jr input:
        contours_input = {}
        contours_input["jr "] = gax_input.contourf(
            lon, lat, full_current.reshape(lon.shape), transform=ccrs.PlateCarree(), **jr_kwargs
        )

        # north:
        nnn = lat > minlat
        contours_input["jr_n"] = paxn_input.contourf(
            lat[nnn], lon[nnn] / 15, full_current.reshape(lon.shape)[nnn], **jr_kwargs
        )

        # south:
        sss = lat < -minlat
        contours_input["jr_s"] = paxs_input.contourf(
            lat[sss], lon[sss] / 15, full_current.reshape(lon.shape)[sss], **jr_kwargs
        )

        lat, lon = np.linspace(-89.9, 89.9, 60), np.linspace(-180, 180, 100)
        lat, lon = np.meshgrid(lat, lon)
        plt_grid = pynamit.Grid(lat=lat, lon=lon)
        plt_evaluator = pynamit.BasisEvaluator(dynamics.state.jr_basis, plt_grid)
        b_evaluator = pynamit.FieldEvaluator(dynamics.mainfield, plt_grid, RI)

        fac_output = plt_evaluator.basis_to_grid(dynamics.state.jr.coeffs) / b_evaluator.br

        # jr interpolated:
        contours_interpolated = {}
        contours_interpolated["jr "] = gax_interpolated.contourf(
            lon, lat, fac_output.reshape(lon.shape), transform=ccrs.PlateCarree(), **jr_kwargs
        )

        # north:
        nnn = lat > minlat
        contours_interpolated["jr_n"] = paxn_interpolated.contourf(
            lat[nnn], lon[nnn] / 15, fac_output.reshape(lon.shape)[nnn], **jr_kwargs
        )

        # south:
        sss = lat < -minlat
        contours_interpolated["jr_s"] = paxs_interpolated.contourf(
            lat[sss], lon[sss] / 15, fac_output.reshape(lon.shape)[sss], **jr_kwargs
        )

        plt.show()

final_time = 3600  # seconds
dynamics.evolve_to_time(final_time)
