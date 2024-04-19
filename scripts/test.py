""" Script to running and debugging pynamit simulation """

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


print('todo: the magnetic field models should be built-in')
dd = dipole.Dipole(2020) 
def B0_dipole(r, theta, phi): 
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    r, theta, phi = r.flatten(), theta.flatten(), phi.flatten()
    size = r.size
    Bn, Br = dd.B(90 - theta, r * 1e-3)
    return(np.vstack((Br, -Bn, np.zeros(r.size))))

# SIMULATION PARAMETERS
Nmax, Mmax, Ncs = 45, 3, 40
dt = 5e-4
totalsteps = 3001

## PLOT PARAMETERS
plotsteps = 200
fig_directory = 'figs/'
Blevels = np.linspace(-300, 300, 22) * 1e-9 # color levels for Br
levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
c_levels = np.linspace(0, 20, 100) # color levels for conductance
Wlevels = np.r_[-512.5:512.5:5]
Philevels = np.r_[-212.5:212.5:5]

## SET UP SIMULATION OBJECT
i2d = pynamit.I2D(Nmax, Mmax, Ncs, B0 = B0_dipole)

## CONDUCTANCE AND FAC INPUT:
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
lon0 = d.mlt2mlon(12, date) # noon longitude
hall, pedersen = conductance.hardy_EUV(i2d.phi, 90 - i2d.theta, Kp, date, starlight = 1, dipole = True)
i2d.set_conductance(hall, pedersen)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
ju = a.get_upward_current(mlat = 90 - i2d.theta, mlt = d.mlon2mlt(i2d.phi, date)) * 1e-6
ju[np.abs(90 - i2d.theta) < 50] = 0 # filter low latitude FACs
ju[i2d.theta < 90] = -ju[i2d.theta < 90] # we need the current to refer to magnetic field direction, so changing sign in the north since the field there points down 
i2d.set_FAC(ju)


# make an integration matrix
#sinmphi = 

cS =  (2 * cnm.n.T + 1) / (4 * np.pi * i2d.RI**2) #pynamit.get_Schmidt_normalization(cnm).T
sS =  (2 * snm.n.T + 1) / (4 * np.pi * i2d.RI**2) #pynamit.get_Schmidt_normalization(snm).T
Ginv = i2d.Gnum.T * np.vstack((cS, sS)) * i2d.csp.unit_area



gg = Ginv.dot(i2d.Gnum)


print(3/0)





## RUN SIMULATION
coeffs = []
count = 0
filecount = 1
time = 0
while True:

    i2d.evolve_Br(dt)
    time = time + dt
    coeffs.append(i2d.shc_VB)
    count += 1
    #print(count, time, i2d.shc_Br[:3])

    if count % plotsteps == 0:
        print(count, time, i2d.shc_Br[:3])
        fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
        filecount +=1
        title = 't = {:.3} s'.format(time)
        Br = i2d.get_Br()
        fig, paxn, paxs, axg =  pynamit.visualization.globalplot(i2d.lon, i2d.lat, Br.reshape(i2d.lat.shape) , title = title, returnplot = True, 
                                           levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
        W = i2d.Gplt.dot(i2d.shc_EW) * 1e-3

        GTE  = i2d.Gdf.T.dot(np.hstack( i2d.get_E()) )
        shc_Phi = i2d.GTGdf_inv.dot(GTE) # find coefficients for electric potential
        Phi = i2d.Gplt.dot(shc_Phi) * 1e-3


        nnn = i2d.lat.flatten() >  50
        sss = i2d.lat.flatten() < -50
        #paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
        #paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
        paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
        paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
        plt.savefig(fn)

    if count > totalsteps:
        break



