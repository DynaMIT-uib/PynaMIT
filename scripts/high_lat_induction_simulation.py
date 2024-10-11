import os
os.environ["MKL_NUM_THREADS"] = "180" 
os.environ["NUMEXPR_NUM_THREADS"] = "180" 
os.environ["OMP_NUM_THREADS"] = "180" 
import numpy as np
import pynamit
from lompe import conductance
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
#import matplotlib.pyplot as plt
import datetime
import pyamps
import apexpy

dataset_filename_prefix = 'data/brn_hl'
Nmax, Mmax, Ncs = 80, 80, 90
latitude_boundary = 45
RE = 6371.2e3
RI = RE + 110e3
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 1]))**2

date = datetime.datetime(2001, 6, 1, 0, 0)
Kp   = 4
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
noon_mlon = d.mlt2mlon(12, date) # noon longitude

## SET UP SIMULATION OBJECT
dynamics = pynamit.Dynamics(dataset_filename_prefix = dataset_filename_prefix,
                            Nmax = Nmax,
                            Mmax = Mmax,
                            Ncs = Ncs,
                            RI = RI,
                            mainfield_kind = 'igrf',
                            FAC_integration_steps = rk,
                            ignore_PFAC = False,
                            connect_hemispheres = True,
                            latitude_boundary = latitude_boundary,
                            ih_constraint_scaling = 1e-5,
                            t0 = str(date))

print(datetime.datetime.now(), 'made dynamics object')

## CONDUCTANCE INPUT
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = False)
dynamics.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon, reg_lambda = 0.0001)

print(datetime.datetime.now(), 'setting jr')
## jr INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
dynamics.set_jr(np.zeros_like(jr_lat), lat = jr_lat, lon = jr_lon)

print(datetime.datetime.now(), 'setting wind')
## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-88.5, 88.5], glatstp = 1.5,
                             glonlim=[-180., 180.], glonstp = 3., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

dynamics.set_u(u_theta = u_theta, u_phi = u_phi, lat = u_lat, lon = u_lon, weights = np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)))

print(datetime.datetime.now(), 'calculating steady state')
dynamics.evolve_to_time(0)
mv = dynamics.state.steady_state_m_ind()
dynamics.state.set_coeffs(m_ind = mv)
print(datetime.datetime.now(), 'simulating')
dynamics.evolve_to_time(0)
dynamics.evolve_to_time(421) # save dynamics object with new m_ind









#a.make_multipanel_output_figure()




#fig, ax = plt.subplots()
#ax.plot(mv)
#ax.plot(dynamics.timeseries['state'].SH_m_ind.values[-1, :])
#plt.show()


