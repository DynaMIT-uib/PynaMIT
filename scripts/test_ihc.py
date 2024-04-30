""" Script to check if the interhemispheric connection works """

from importlib import reload
import pynamit
import dipole
import numpy as np
import datetime
import secsy
import pyamps
import matplotlib.pyplot as plt
from lompe import conductance

reload(pynamit)
RE = 6371.2e3
RI = RE + 110e3

# MODEL PARAMETERS
Nmax, Mmax, Ncs = 15, 15, 20

## PLOT PARAMETERS
fig_directory = 'figs/'
Blevels = np.linspace(-5, 5, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
i2d_sha = pynamit.sha(Nmax, Mmax)
i2d_csp = pynamit.CSprojection(Ncs)
i2d = pynamit.I2D(i2d_sha, i2d_csp, RI, mainfield_kind = 'dipole', FAC_integration_parameters = {'steps':np.logspace(np.log10(RI), np.log10(7 * RE), 11)}, 
                                        ignore_PNAF = False, connect_hemispheres = True, latitude_boundary = 50)

## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.grid.grid(RI, lat, lon)
plt_grid.construct_G(i2d_sha)

## CONDUCTANCE AND FAC INPUT:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude
hall, pedersen = conductance.hardy_EUV(i2d.num_grid.lon, i2d.num_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = -a.get_upward_current(mlat = i2d.num_grid.lat, mlt = d.mlon2mlt(i2d.num_grid.lon, date)) / i2d.state.sinI * 1e-6
jparallel[np.abs(i2d.num_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_FAC(jparallel)


GBr = plt_grid.G * i2d_sha.n / i2d.num_grid.RI
Br_I2D = GBr.dot(i2d.state.shc_PFAC)


fig, paxn, paxs, axg =  pynamit.globalplot(plt_grid.lon, plt_grid.lat, Br_I2D.reshape(plt_grid.lat.shape), returnplot = True, 
                                           levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')



plt.show()