""" Script to check if the PFAC calculation gives a reasonable result """

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import os
import secsy
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance

reload(pynamit)
RE = 6371.2e3


# MODEL PARAMETERS
Nmax, Mmax, Ncs = 45, 20, 40

## PLOT PARAMETERS
fig_directory = 'figs/'
Blevels = np.linspace(-2, 2, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
i2d = pynamit.I2D(Nmax, Mmax, Ncs, B0 = 'dipole', FAC_integration_parameters = {'steps':np.logspace(np.log10(RE + 110.e3), np.log10(7 * RE), 21)})


## CONDUCTANCE AND FAC INPUT:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude
hall, pedersen = conductance.hardy_EUV(i2d.phi, 90 - i2d.theta, Kp, date, starlight = 1, dipole = True)
i2d.set_conductance(hall, pedersen)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = a.get_upward_current(mlat = 90 - i2d.theta, mlt = d.mlon2mlt(i2d.phi, date)) / i2d.sinI * 1e-6
jparallel[np.abs(90 - i2d.theta) < 50] = 0 # filter low latitude FACs


i2d.set_FAC(jparallel)

print('Building SECS matrices. This takes some time (and memory) because of global grids...')
secsI = jparallel * i2d.sinI * i2d.csp.unit_area * i2d.RI**2 # SECS amplitudes are radial current density times area
lat, lon = i2d.lat.flatten(), i2d.lon.flatten()
r = np.full(lat.size, i2d.RI - 1)
lat_secs, lon_secs = 90 - i2d.theta, i2d.phi
Be, Bn, Br = i2d.bphi, - i2d.btheta, i2d.br
Ge, Gn, Gu = secsy.get_CF_SECS_B_G_matrices_for_inclined_field(lat, lon, r, lat_secs, lon_secs, Be, Bn, Br, RI = i2d.RI)


Br_SECS = Gu.dot(secsI)

GBr = i2d.Gplt * i2d.n / i2d.RI
Br_I2D = GBr.dot(i2d.shc_PFAC)


fig, paxn, paxs, axg =  pynamit.globalplot(i2d.lon, i2d.lat, Br_I2D.reshape(i2d.lat.shape), returnplot = True, 
                                           levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

fig, paxn, paxs, axg =  pynamit.globalplot(i2d.lon, i2d.lat, Br_SECS.reshape(i2d.lat.shape), returnplot = True, 
                                           levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')

fig, ax = plt.subplots()
ax.scatter(Br_SECS, Br_I2D)
ax.set_xlabel('straight tilted SECS')
ax.set_ylabel('Spherical harmonics')
plt.show()
