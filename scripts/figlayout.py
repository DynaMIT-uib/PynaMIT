import matplotlib.pyplot as plt
import numpy as np
from polplot import Polarplot
import cartopy.crs as ccrs
import pynamit


# PLACEHOLDER -- DEFINE TOY MODEL OBJECT -- PLACEHOLDER
from lompe import conductance
import dipole
import pyhwm2014 # https://github.com/rilma/pyHWM14
import datetime
import pyamps

RE = 6371.2e3
RI = RE + 110e3
latitude_boundary = 40

Nmax, Mmax, Ncs = 25, 25, 50
rk = RI / np.cos(np.deg2rad(np.r_[0: 60: int(180 / Nmax) - 1])) ** 2
rk = {'steps':rk}
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)
u_phi   =  hwm14Obj.Uwind
u_theta = -hwm14Obj.Vwind
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
i2d_sh = pynamit.SHBasis(Nmax, Mmax)
i2d_csp = pynamit.CSProjection(Ncs)
u_int = i2d_csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, i2d_csp.arr_theta, i2d_csp.arr_phi)
u_east_int, u_north_int, u_r_int = u_int

i2d = pynamit.I2D(i2d_sh, i2d_csp, RI, mainfield_kind = 'dipole', FAC_integration_parameters = rk, 
                                       ignore_PFAC = False, connect_hemispheres = True, latitude_boundary = latitude_boundary)

csp_grid = pynamit.grid.Grid(RI, 90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_sh_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, csp_grid)


## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.grid.Grid(RI, lat, lon)
plt_sh_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d_sh, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen, csp_sh_evaluator)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = -a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) / i2d.state.sinI * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_u(-u_north_int, u_east_int)
i2d.state.set_FAC(jparallel, csp_sh_evaluator)


#### MODEL OBJECT DONE




B_kwargs   = {'cmap':plt.cm.bwr, 'levels':np.linspace(-50, 50, 22) * 1e-9, 'extend':'both'}
eqJ_kwargs = {'colors':'black', 'levels':np.r_[-210:220:20] * 1e-3}
FAC_kwargs = {'cmap':plt.cm.bwr, 'levels':np.linspace(-.95, .95, 22) * 1e-6, 'extend':'both'}


## SET UP PLOTTING GRID AND BASIS EVALUATOR
NLA, NLO = 50, 90
lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
lat, lon = map(np.ravel, np.meshgrid(lat, lon))
plt_grid = pynamit.grid.Grid(i2d.state.RI, lat, lon)
plt_sh_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.sh, plt_grid)

## MAP PROJECTION:
global_projection = ccrs.PlateCarree(central_longitude = noon_longitude)

fig = plt.figure(figsize = (15, 13))

paxn_B = Polarplot(plt.subplot2grid((4, 4), (0, 0)))
paxs_B = Polarplot(plt.subplot2grid((4, 4), (0, 1)))
paxn_j = Polarplot(plt.subplot2grid((4, 4), (0, 2)))
paxs_j = Polarplot(plt.subplot2grid((4, 4), (0, 3)))
gax_B = plt.subplot2grid((4, 2), (1, 0), projection = global_projection, rowspan = 2)
gax_j = plt.subplot2grid((4, 2), (1, 1), projection = global_projection, rowspan = 2)
ax_1 = plt.subplot2grid((4, 3), (3, 0))
ax_2 = plt.subplot2grid((4, 3), (3, 1))
ax_3 = plt.subplot2grid((4, 3), (3, 2))

for ax in [gax_B, gax_j]: ax.coastlines(zorder = 2, color = 'grey')

## CALCULATE VALUES TO PLOT
Br  = plt_sh_evaluator.scaled_G(i2d.state.sh.n / i2d.state.num_grid.RI).dot(i2d.state.VB.coeffs)

sinI   = i2d.state.mainfield.get_sinI(plt_grid.RI, plt_grid.theta, plt_grid.lon)
FAC    = -(plt_sh_evaluator.scaled_G(1 / sinI.reshape((-1, 1)))).dot(i2d.state.TJr.coeffs)
jr_mod =   csp_sh_evaluator.G.dot(i2d.state.TJr.coeffs)
eq_current_function = plt_sh_evaluator.G.dot(i2d.state.VJ.coeffs)

## GLOBAL PLOTS
gax_B.contourf(lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), Br.reshape((NLO, NLA)), transform = ccrs.PlateCarree(), **B_kwargs)
gax_j.contour( lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), eq_current_function.reshape((NLO, NLA)), transform = ccrs.PlateCarree(), **eqJ_kwargs)
gax_j.contourf(lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), FAC.reshape((NLO, NLA)), **FAC_kwargs)


## POLAR PLOTS
mlt = (lon - noon_longitude + 180) / 15 # rotate so that noon is up

# north:
iii = lat >  50
paxn_B.contourf(lat[iii], mlt[iii], Br[iii], **B_kwargs)
paxn_j.contour( lat[iii], mlt[iii], eq_current_function[iii], **eqJ_kwargs)
paxn_j.contourf(lat[iii], mlt[iii], FAC[iii], **FAC_kwargs)

# south:
iii = lat < -50
paxs_B.contourf(lat[iii], mlt[iii], Br[iii], **B_kwargs)
paxs_j.contour( lat[iii], mlt[iii], eq_current_function[iii], **eqJ_kwargs)
paxs_j.contourf(lat[iii], mlt[iii], FAC[iii], **FAC_kwargs)


# scatter plot high latitude FACs
iii = np.abs(i2d.state.num_grid.lat) > i2d.state.latitude_boundary
jrmax = np.max(np.abs(i2d.state.jr))
ax_1.scatter(i2d.state.jr, jr_mod[iii])
ax_1.plot([-jrmax, jrmax], [-jrmax, jrmax], 'k-')
ax_1.set_xlabel('Input ')

# scatter plot FACs at conjugate points
j_par_ll = i2d.state.G_par_ll.dot(i2d.state.TB.coeffs)
j_par_ll_conj = i2d.state.G_par_ll_conj.dot(i2d.state.TB.coeffs)
j_par_max = np.max(np.abs(j_par_ll))
ax_2.scatter(j_par_ll, j_par_ll_conj)
ax_2.plot([-j_par_max, j_par_max], [-j_par_max, j_par_max], 'k-')
ax_2.set_xlabel('$j_\parallel$ [A/m$^2$] at |latitude| $< {}^\circ$'.format(i2d.state.latitude_boundary))
ax_2.set_ylabel('$j_\parallel$ [A/m$^2$] at conjugate points')

# scatter plot of ve1 and ve2 vs corresponding values at conjugate points
cu_ll = (i2d.state.u_phi_ll_conj * i2d.state.c_u_phi_conj + i2d.state.u_theta_ll_conj * i2d.state.c_u_theta_conj).flatten()
cu_co = (i2d.state.u_phi_ll * i2d.state.c_u_phi + i2d.state.u_theta_ll * i2d.state.c_u_theta).flatten()
AT_ll = i2d.state.etaP_ll * i2d.state.A_eP_T + i2d.state.etaH_ll * i2d.state.A_eH_T
AT_co = i2d.state.etaP_ll_conj * i2d.state.A_eP_conj_T + i2d.state.etaH_ll_conj * i2d.state.A_eH_conj_T 
AV_ll = -(i2d.state.etaP_ll * i2d.state.A_eP_V + i2d.state.etaH_ll * i2d.state.A_eH_V)
AV_co = -(i2d.state.etaP_ll_conj * i2d.state.A_eP_conj_V + i2d.state.etaH_ll_conj * i2d.state.A_eH_conj_V)

c_ll = cu_ll + AV_ll.dot(i2d.state.VB.coeffs)
c_co = cu_co + AV_co.dot(i2d.state.VB.coeffs)
ve1_ll, ve2_ll = np.split(c_ll + AT_ll.dot(i2d.state.TB.coeffs), 2)
ve1_co, ve2_co = np.split(c_co + AT_co.dot(i2d.state.TB.coeffs), 2)
ax_3.scatter(ve1_ll, ve1_co, label = '$v_{e_1}$')
ax_3.scatter(ve2_ll, ve2_co, label = '$v_{e_2}$')
ax_3.set_xlabel('$v_{e_i}$')
ax_3.set_ylabel('$v_{e_i}$ at conjugate points')
ax_3.legend(frameon = False)



plt.subplots_adjust(top=0.89, bottom=0.095, left=0.025, right=0.95, hspace=0.0, wspace=0.185)

plt.show()




