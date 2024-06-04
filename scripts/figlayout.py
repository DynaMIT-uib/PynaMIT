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
latitude_boundary = 35

WIND_FACTOR = 1 # scale wind by this factor

Nmax, Mmax, Ncs = 14, 14, 30
rk = RI / np.cos(np.deg2rad(np.r_[0: 80: int(80 / Nmax)])) ** 2
print(len(rk))
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

csp_grid = pynamit.Grid(90 - i2d_csp.arr_theta, i2d_csp.arr_phi)
csp_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, csp_grid)
csp_b_geometry = pynamit.BGeometry(i2d.state.mainfield, csp_grid, RI)


## SET UP PLOTTING GRID
lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
lat, lon = np.meshgrid(lat, lon)
plt_grid = pynamit.Grid(lat, lon)
plt_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, plt_grid)

## CONDUCTANCE AND FAC INPUT:
hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jparallel = a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) / csp_b_geometry.br * 1e-6
jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

i2d.state.set_u(-u_north_int * WIND_FACTOR, u_east_int * WIND_FACTOR)
i2d.state.set_FAC(jparallel, csp_i2d_evaluator)


#### MODEL OBJECT DONE
def debugplot(i2d, title = None, filename = None, noon_longitude = 0):

    ## SET UP PLOTTING GRID
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    plt_grid = pynamit.Grid(lat, lon)
    plt_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, plt_grid)


    B_kwargs   = {'cmap':plt.cm.bwr, 'levels':np.linspace(-50, 50, 22) * 1e-9, 'extend':'both'}
    eqJ_kwargs = {'colors':'black', 'levels':np.r_[-210:220:20] * 1e3}
    FAC_kwargs = {'cmap':plt.cm.bwr, 'levels':np.linspace(-.95, .95, 22) * 1e-6, 'extend':'both'}


    ## SET UP PLOTTING GRID AND BASIS EVALUATOR
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = pynamit.Grid(lat, lon)
    plt_i2d_evaluator = pynamit.BasisEvaluator(i2d.state.basis, plt_grid)
    plt_b_geometry = pynamit.BGeometry(i2d.state.mainfield, plt_grid, i2d.state.RI)

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

    for ax in [gax_B, gax_j]: 
        ax.coastlines(zorder = 2, color = 'grey')

    ## CALCULATE VALUES TO PLOT
    Br  = i2d.state.get_Br(plt_i2d_evaluator)

    FAC = (plt_i2d_evaluator.scaled_G(1 / plt_b_geometry.br.reshape((-1, 1)))).dot(i2d.state.TB_imp.coeffs * i2d.state.TB_imp_to_Jr)
    jr_mod = plt_i2d_evaluator.G.dot(i2d.state.TB_imp.coeffs * i2d.state.TB_imp_to_Jr)
    eq_current_function = i2d.state.get_Jeq(plt_i2d_evaluator)

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
    j_par_ll = i2d.state.G_par_ll.dot(i2d.state.TB_imp.coeffs)
    j_par_cp = i2d.state.G_par_cp.dot(i2d.state.TB_imp.coeffs)
    j_par_max = np.max(np.abs(j_par_ll))
    ax_2.scatter(j_par_ll, j_par_cp)
    ax_2.plot([-j_par_max, j_par_max], [-j_par_max, j_par_max], 'k-')
    ax_2.set_xlabel(r'$j_\parallel$ [A/m$^2$] at |latitude| $< {}^\circ$'.format(i2d.state.latitude_boundary))
    ax_2.set_ylabel(r'$j_\parallel$ [A/m$^2$] at conjugate points')

    # scatter plot of Ed1 and Ed2 vs corresponding values at conjugate points
    cu_ll = i2d.state.u_phi_cp * i2d.state.aup_cp   + i2d.state.u_theta_cp * i2d.state.aut_cp
    cu_cp = i2d.state.u_phi_ll * i2d.state.aup_ll   + i2d.state.u_theta_ll * i2d.state.aut_ll
    AT_ll = i2d.state.etaP_ll  * i2d.state.aeP_T_ll + i2d.state.etaH_ll    * i2d.state.aeH_T_ll
    AT_cp = i2d.state.etaP_cp  * i2d.state.aeP_T_cp + i2d.state.etaH_cp    * i2d.state.aeH_T_cp
    AV_ll = i2d.state.etaP_ll  * i2d.state.aeP_V_ll + i2d.state.etaH_ll    * i2d.state.aeH_V_ll
    AV_cp = i2d.state.etaP_cp  * i2d.state.aeP_V_cp + i2d.state.etaH_cp    * i2d.state.aeH_V_cp

    c_ll = cu_ll + AV_ll.dot(i2d.state.VB_ind.coeffs)
    c_cp = cu_cp + AV_cp.dot(i2d.state.VB_ind.coeffs)
    Ed1_ll, Ed2_ll = np.split(c_ll + AT_ll.dot(i2d.state.TB_imp.coeffs), 2)
    Ed1_cp, Ed2_cp = np.split(c_cp + AT_cp.dot(i2d.state.TB_imp.coeffs), 2)
    ax_3.scatter(Ed1_ll, Ed1_cp, label = '$E_{d_1}$')
    ax_3.scatter(Ed2_ll, Ed2_cp, label = '$E_{d_2}$')
    ax_3.set_xlabel('$E_{d_i}$')
    ax_3.set_ylabel('$E_{d_i}$ at conjugate points')
    ax_3.legend(frameon = False)

    if title is not None:
        gax_j.set_title(title)

    plt.subplots_adjust(top=0.89, bottom=0.095, left=0.025, right=0.95, hspace=0.0, wspace=0.185)
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()


debugplot(i2d, title = 'hoi!', filename = None, noon_longitude = noon_longitude)



