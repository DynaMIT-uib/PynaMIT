"""Simulation."""

import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014  # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy
import h5py as h5
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
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2  # int(80 / Nmax)])) ** 2
print(len(rk))

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

mage_dir = ("./mage_data/")
mage_tag = "msphere"

gsph = msph.GamsphPipe(mage_dir,mage_tag,doFast=False)
nstep = gsph.sFin

mixFiles = os.path.join(mage_dir,"%s.mix.h5"%(mage_tag))

for step in range(0, nstep):
    ion_north = remix.remix(mixFiles, step)
    ion_north.init_vars('NORTH')

    ion_south = remix.remix(mixFiles, step)
    ion_south.init_vars('SOUTH')

    if step == 0:
        # In units of RionE?
        north_r = ion_north.ion["R"]
        south_r = ion_south.ion["R"]

        # Uses convention from remix.efield().
        north_theta = np.rad2deg(np.arcsin(north_r)).flatten()
        south_theta = np.rad2deg(np.arcsin(south_r)).flatten()

        north_phi = np.rad2deg(ion_north.ion["THETA"]).flatten()
        south_phi = np.rad2deg(ion_south.ion["THETA"]).flatten()

    # Fetch the ionospheric fields.
    north_current = ion_north.variables["current"]["data"]
    south_current = ion_south.variables["current"]["data"]

    north_conductance_pedersen = ion_north.variables["sigmap"]["data"]
    south_conductance_pedersen = ion_south.variables["sigmap"]["data"]

    north_conductance_hall = ion_north.variables["sigmah"]["data"]
    south_conductance_hall = ion_south.variables["sigmah"]["data"]

    print("Step %d" % step)
    print(north_theta.shape())
    print(north_phi.shape())
    print(north_current.shape())
    print(north_conductance_hall.shape())

    print(south_theta.shape())
    print(south_phi.shape())
    print(south_current.shape())
    print(south_conductance_hall.shape())
    # Get and set jr input.
    dynamics.set_jr(north_current, theta=north_theta, phi=north_phi, time=dt*step)
    dynamics.set_conductance(north_conductance_hall, north_conductance_pedersen, theta=north_theta, phi=north_phi, time=dt*step)


final_time = 120  # seconds
dynamics.evolve_to_time(final_time)
