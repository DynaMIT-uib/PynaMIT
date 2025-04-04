"""Simulation."""

import numpy as np
import pynamit
import dipole
import datetime
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix
import os


RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40
dt = 300

FLOAT_ERROR_MARGIN = 1e-6

dataset_filename_prefix = "mage-forcing"
Nmax, Mmax, Ncs = 10, 10, 10
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
        ).flatten()
        north_phi_centered = (
            north_phi[:-1, :-1] + 0.5 * np.diff(north_phi, axis=1)[:-1, :]
        ).flatten()
        south_theta_centered = (
            south_theta[:-1, :-1] + 0.5 * np.diff(south_theta, axis=0)[:, :-1]
        ).flatten()
        south_phi_centered = (
            south_phi[:-1, :-1] + 0.5 * np.diff(south_phi, axis=1)[:-1, :]
        ).flatten()

    # Fetch the ionospheric fields.
    north_current = ion_north.variables["current"]["data"].flatten()
    south_current = ion_south.variables["current"]["data"].flatten()

    north_conductance_pedersen = ion_north.variables["sigmap"]["data"].flatten()
    south_conductance_pedersen = ion_south.variables["sigmap"]["data"].flatten()

    north_conductance_hall = ion_north.variables["sigmah"]["data"].flatten()
    south_conductance_hall = ion_south.variables["sigmah"]["data"].flatten()

    # Get and set jr input.
    dynamics.set_jr(
        north_current, theta=north_theta_centered, phi=north_phi_centered, time=dt * step
    )
    dynamics.set_conductance(
        north_conductance_hall,
        north_conductance_pedersen,
        theta=north_theta_centered,
        phi=north_phi_centered,
        time=dt * step,
    )


final_time = 3600  # seconds
dynamics.evolve_to_time(final_time)
