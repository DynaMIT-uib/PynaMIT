import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pynamit
from pynamit.various.constants import RE
from pynamit.various.constants import mu0
import dipole
import datetime
import apexpy

state_data_list = [xr.load_dataset('../scripts/oscillations_40s_state.ncdf'), 
                   xr.load_dataset('../scripts/oscillations_20s_state.ncdf'), 
                   xr.load_dataset('../scripts/oscillations_10s_state.ncdf'),
                   xr.load_dataset('../scripts/oscillations_05s_state.ncdf')]
settings_list   = [xr.load_dataset('../scripts/oscillations_40s_settings.ncdf'), 
                   xr.load_dataset('../scripts/oscillations_20s_settings.ncdf'), 
                   xr.load_dataset('../scripts/oscillations_10s_settings.ncdf'),
                   xr.load_dataset('../scripts/oscillations_05s_settings.ncdf')]

RI = settings_list[0].RI
sh_basis = pynamit.SHBasis(settings_list[0].Nmax, settings_list[0].Mmax)

t0 = datetime.datetime.strptime(settings_list[0].t0, '%Y-%m-%d %H:%M:%S')
d = dipole.Dipole(t0.year)
a = apexpy.Apex(t0.year)

# construct plot grid in mlt/mlat, then convert to glat/glon:
mlt, mlat = np.meshgrid([4, 9, 12, 15, 20], [-80, -60, -20, 20, 60, 80][::-1], indexing = 'ij')

Nrows = mlt.shape[1]
Ncols = mlt.shape[0]

mlon = d.mlt2mlon(mlt, t0)
glat, glon, _ = a.apex2geo(mlat, mlon, 0)
glat, glon = glat.flatten(), glon.flatten()

# conversion factors:
m_ind_to_Br  = -sh_basis.n
laplacian    = -sh_basis.n * (sh_basis.n + 1) / RI**2
m_imp_to_jr  =  laplacian * RI / mu0
W_to_dBr_dt  = -laplacian * RI
m_ind_to_Jeq =  RI / mu0 * (2 * sh_basis.n + 1) / (sh_basis.n + 1)


ground_grid = pynamit.Grid(lat = glat, lon = glon)
ground_evaluator = pynamit.BasisEvaluator(sh_basis, ground_grid)

m_ind_to_Bh_ground = (RE / RI) ** sh_basis.n
m_ind_to_Br_ground = -sh_basis.n * (RE / RI) ** (sh_basis.n - 1)


fig, axes = plt.subplots(ncols = Ncols, nrows = Nrows, sharex = True)

for state_data in state_data_list:
    # calculate the time series:
    m_ind = state_data.SH_m_ind.values.T

    Br = (ground_evaluator.G   * m_ind_to_Br_ground.reshape((1, -1))).dot(m_ind)
    Bh = (-ground_evaluator.G_grad * m_ind_to_Bh_ground.reshape((1, -1))).dot(m_ind)
    Btheta, Bphi = np.split(Bh, 2, axis = 0)


    ii, jj = np.unravel_index(np.arange(len(glat)), mlt.shape)
    for i in range(len(glat)):
        axes[jj[i], ii[i]].plot(state_data.time.values, Br[i] * 1e9, label = '$B_r$')
        #ax.plot(state_data.time.values, Btheta[i] * 1e9, label = '$B_\\theta$')
        #ax.plot(state_data.time.values, Bphi[i] * 1e9, label = '$B_\phi$')
        if jj[i] == 0:
            axes[jj[i], ii[i]].set_title('MLT$ = ' + str(mlt[ii[i], jj[i]]) + '$')

        if ii[i] == Ncols - 1:
            axes[jj[i], ii[i]].set_ylabel('mlat$ = ' + str(mlat[ii[i], jj[i]]) + '^\circ$', rotation=270, labelpad=15)

            #axes[jj[i], ii[i]].set_title('mlat = ' + str(mlat[ii[i], jj[i]]), loc = 'right')
            axes[jj[i], ii[i]].yaxis.set_label_position("right")
    


fig, ax = plt.subplots(ncols = 5, nrows = 5, sharex = True)

for state_data in state_data_list:
    # calculate the time series:
    m_ind = state_data.SH_m_ind.values.T


    for i in range(25):
        ax.flatten()[i].plot(state_data.time.values, state_data['SH_m_imp'].values[:, i], label = '$B_r$')


#axes[0, 0].legend(frameon = False)

plt.show()