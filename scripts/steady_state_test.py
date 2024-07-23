import numpy as np
import pynamit
from lompe import conductance
import dipole
#import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

dataset_filename_prefix = 'ss_test3'
Nmax, Mmax, Ncs = 30, 30, 30
latitude_boundary = 40
RE = 6371.2e3
RI = RE + 110e3
rk = RI / np.cos(np.deg2rad(np.r_[0: 70: 5]))**2

date = datetime.datetime(2001, 5, 12, 21, 0)
Kp   = 5
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
                            zero_jr_at_dip_equator = True,
                            ih_constraint_scaling = 1e-5)

print('made dynamics object')

## CONDUCTANCE INPUT
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = False)
dynamics.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

## jr INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
apx = apexpy.Apex(refh = (RI - RE) * 1e-3, date = 2020)
mlat, mlon = apx.geo2apex(jr_lat, jr_lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)
_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI-RE)*1e-3) # fix this
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jr = a.get_upward_current(mlat = mlat, mlt = mlt) * 1e-6
jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr
dynamics.set_jr(jr, lat = jr_lat, lon = jr_lon)

## WIND INPUT
#hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
#                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

#u_phi   =  hwm14Obj.Uwind
#u_theta = -hwm14Obj.Vwind
#u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_lat, u_lon, u_phi, u_theta = np.load('ulat.npy'), np.load('ulon.npy'), np.load('uphi.npy'), np.load('utheta.npy')
u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')
u = (u_theta.flatten(), u_phi.flatten())
dynamics.set_u(u, lat = u_lat, lon = u_lon)

dynamics.evolve_to_time(0)

m_ind_ss = dynamics.steady_state_m_ind()

# numerical curl:
#Dc = dynamics.get_finite_difference_curl_matrix()
#E = np.atleast_2d(np.hstack((dynamics.state.get_E()))).T
#curl_E_num = Dc.dot(E)

# SH curl
#curlE_SH_coeffs = -dynamics.state.W.coeffs * dynamics.state.W_to_dBr_dt

#curl_E_SH = dynamics.state_basis_evaluator.basis_to_grid(curlE_SH_coeffs)

#GVJ = dynamics.state.G_m_ind_to_JS
#GTJ = dynamics.state.G_m_imp_to_JS

#import scipy.sparse as sp
#br, bt, bp = dynamics.state.b_evaluator.br, dynamics.state.b_evaluator.btheta, dynamics.state.b_evaluator.bphi
eP, eH = dynamics.state.etaP, dynamics.state.etaH
#C00 = sp.diags(eP * (bp**2 + br**2))
#C01 = sp.diags(eP * (-bt * bp) + eH * br)
#C10 = sp.diags(eP * (-bt * bp) - eH * br)
#C11 = sp.diags(eP * (bt**2 + br**2))
#C = sp.vstack((sp.hstack((C00, C01)), sp.hstack((C10, C11))))
#
#uxb = np.hstack((dynamics.state.uxB_theta, dynamics.state.uxB_phi))
#
#GcCGVJ = Dc.dot(C).dot(GVJ)
#GcCGTJ = Dc.dot(C).dot(GTJ)
#
#import xarray as xr
#m_imp = xr.load_dataset(dataset_filename_prefix).SH_m_imp.values[0]
#m_ind_ss = np.linalg.pinv(GcCGVJ, rcond = 0).dot(Dc.dot(uxb) - GcCGTJ.dot(m_imp))

# calculate electric field with steady-state coefficients:
Js_ind, Je_ind = np.split(dynamics.state.G_m_ind_to_JS.dot(m_ind_ss), 2, axis = 0)
Js_imp, Je_imp = np.split(dynamics.state.G_m_imp_to_JS.dot(dynamics.state.m_imp.coeffs), 2, axis = 0)
Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp
Eth = eP * (dynamics.state.b00 * Jth + dynamics.state.b01 * Jph) + eH * ( dynamics.state.b_evaluator.br * Jph)
Eph = eP * (dynamics.state.b10 * Jth + dynamics.state.b11 * Jph) + eH * (-dynamics.state.b_evaluator.br * Jth)
Eth -= dynamics.state.uxB_theta
Eph -= dynamics.state.uxB_phi

E_cf_coeff, E_df_coeff = dynamics.state.basis_evaluator.grid_to_basis(np.hstack((Eth, Eph)), helmholtz = True)



#import matplotlib.pyplot as plt
#fig, axes = plt.subplots(nrows = 2)
#axes[0].scatter(dynamics.state_grid.lon, dynamics.state_grid.lat, c = curl_E_num)
#axes[1].scatter(dynamics.state_grid.lon, dynamics.state_grid.lat, c = curl_E_SH)
#plt.show()
#

