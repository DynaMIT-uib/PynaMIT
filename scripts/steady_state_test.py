import numpy as np
import pynamit
from lompe import conductance
import dipole
#import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps
import apexpy

result_filename = 'ss_test3.ncdf'
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
#hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
#                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)

#u_phi   =  hwm14Obj.Uwind
#u_theta = -hwm14Obj.Vwind
#u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
u_lat, u_lon, u_phi, u_theta = np.load('ulat.npy'), np.load('ulon.npy'), np.load('uphi.npy'), np.load('utheta.npy')
u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')
u_grid = pynamit.Grid(u_lat, u_lon)

i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSProjection(Ncs)
u_int = i2d_csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, i2d_csp.arr_theta, i2d_csp.arr_phi)
u_east_int, u_north_int, u_r_int = u_int

i2d = pynamit.I2D(result_filename = result_filename, sh = i2d_sh, csp = i2d_csp, RI = RI, mainfield_kind = 'igrf', FAC_integration_steps = rk,
                                    ignore_PFAC = False, connect_hemispheres = True, latitude_boundary = latitude_boundary,
                                    zero_jr_at_dip_equator = True, ih_constraint_scaling = 1e-5)
print('made i2d object')


csp_grid = pynamit.Grid(90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_i2d_evaluator = pynamit.BasisEvaluator(i2d_sh, csp_grid)
csp_b_evaluator = pynamit.FieldEvaluator(i2d.state.mainfield, csp_grid, RI)


## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat, lon)
plt_i2d_evaluator = pynamit.BasisEvaluator(i2d_sh, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = False)
i2d.set_conductance(hall, pedersen, csp_grid)

apx = apexpy.Apex(refh = (RI - RE) * 1e-3, date = 2020)
mlat, mlon = apx.geo2apex(csp_grid.lat, csp_grid.lon, (RI - RE) * 1e-3)
mlt = d.mlon2mlt(mlon, date)

_, noon_longitude, _ = apx.apex2geo(0, noon_mlon, (RI-RE)*1e-3) # fix this

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = a.get_upward_current(mlat = mlat, mlt = mlt) / csp_b_evaluator.br * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.set_u(-u_north_int, u_east_int, u_grid)
i2d.set_FAC(jparallel, csp_grid)


i2d.evolve_to_time(0)

m_ind_ss = i2d.steady_state_m_ind()

# numerical curl:
#Dc = i2d.get_finite_difference_curl_matrix()
#E = np.atleast_2d(np.hstack((i2d.state.get_E()))).T
#curl_E_num = Dc.dot(E)

# SH curl
#i2d.state.update_Phi_and_EW()
#curlE_SH_coeffs = -i2d.state.EW.coeffs * i2d.state.EW_to_dBr_dt
#curl_E_SH = csp_i2d_evaluator.basis_to_grid(curlE_SH_coeffs)

#GVJ = i2d.state.G_m_ind_to_JS
#GTJ = i2d.state.G_m_imp_to_JS

#import scipy.sparse as sp
#br, bt, bp = i2d.state.b_evaluator.br, i2d.state.b_evaluator.btheta, i2d.state.b_evaluator.bphi
eP, eH = i2d.state.etaP, i2d.state.etaH
#C00 = sp.diags(eP * (bp**2 + br**2))
#C01 = sp.diags(eP * (-bt * bp) + eH * br)
#C10 = sp.diags(eP * (-bt * bp) - eH * br)
#C11 = sp.diags(eP * (bt**2 + br**2))
#C = sp.vstack((sp.hstack((C00, C01)), sp.hstack((C10, C11))))
#
#uxb = np.hstack((i2d.state.uxB_theta, i2d.state.uxB_phi))
#
#GcCGVJ = Dc.dot(C).dot(GVJ)
#GcCGTJ = Dc.dot(C).dot(GTJ)
#
#import xarray as xr
#m_imp = xr.load_dataset(result_filename).SH_coefficients_imposed.values[0]
#m_ind_ss = np.linalg.pinv(GcCGVJ, rcond = 0).dot(Dc.dot(uxb) - GcCGTJ.dot(m_imp))

# calculate electric field with steady-state coefficients:
Js_ind, Je_ind = np.split(i2d.state.G_m_ind_to_JS.dot(m_ind_ss), 2, axis = 0)
Js_imp, Je_imp = np.split(i2d.state.G_m_imp_to_JS.dot(i2d.state.m_imp.coeffs), 2, axis = 0)
Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp
Eth = eP * (i2d.state.b00 * Jth + i2d.state.b01 * Jph) + eH * ( i2d.state.b_evaluator.br * Jph)
Eph = eP * (i2d.state.b10 * Jth + i2d.state.b11 * Jph) + eH * (-i2d.state.b_evaluator.br * Jth)
Eth -= i2d.state.uxB_theta
Eph -= i2d.state.uxB_phi

E_cf_coeff, E_df_coeff = i2d.state.basis_evaluator.grid_to_basis((Eth, Eph), helmholtz = True)



#import matplotlib.pyplot as plt
#fig, axes = plt.subplots(nrows = 2)
#axes[0].scatter(csp_grid.lon, csp_grid.lat, c = curl_E_num)
#axes[1].scatter(csp_grid.lon, csp_grid.lat, c = curl_E_SH)
#plt.show()
#

