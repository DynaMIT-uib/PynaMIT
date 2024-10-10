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

dataset_filename_prefix = 'figlayout'

Nmax, Mmax, Ncs = 14, 14, 30
rk = RI / np.cos(np.deg2rad(np.r_[0: 80: int(80 / Nmax)])) ** 2
print(len(rk))
date = datetime.datetime(2001, 5, 12, 21, 45)
Kp   = 5
d = dipole.Dipole(date.year)
noon_longitude = d.mlt2mlon(12, date) # noon longitude

## SET UP SIMULATION OBJECT
dynamics = pynamit.Dynamics(dataset_filename_prefix = dataset_filename_prefix,
                            Nmax = Nmax,
                            Mmax = Mmax,
                            Ncs = Ncs,
                            RI = RI,
                            mainfield_kind = 'dipole',
                            FAC_integration_steps = rk,
                            ignore_PFAC = False,
                            connect_hemispheres = True,
                            latitude_boundary = latitude_boundary)

## CONDUCTANCE INPUT
conductance_lat = dynamics.state_grid.lat
conductance_lon = dynamics.state_grid.lon
hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = True)
dynamics.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

## jr INPUT
jr_lat = dynamics.state_grid.lat
jr_lon = dynamics.state_grid.lon
a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
jr = a.get_upward_current(mlat = jr_lat, mlt = d.mlon2mlt(jr_lon, date)) * 1e-6
jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr
dynamics.set_jr(jr, lat = jr_lat, lon = jr_lon)

## WIND INPUT
hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3., 
                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour, day = date.timetuple().tm_yday)
u_theta, u_phi = (-hwm14Obj.Vwind.flatten() * WIND_FACTOR, hwm14Obj.Uwind.flatten() * WIND_FACTOR)
u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')
dynamics.set_u(u_theta = u_theta, u_phi = u_phi, lat = u_lat, lon = u_lon, weights = np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)))

dynamics.update_conductance()
dynamics.update_u()
dynamics.update_jr()
dynamics.state.impose_constraints()

#### MODEL OBJECT DONE
def debugplot(dynamics, title = None, filename = None, noon_longitude = 0):

    B_kwargs   = {'cmap':plt.cm.bwr, 'levels':np.linspace(-50, 50, 22) * 1e-9, 'extend':'both'}
    eqJ_kwargs = {'colors':'black', 'levels':np.r_[-210:220:20] * 1e3}
    FAC_kwargs = {'cmap':plt.cm.bwr, 'levels':np.linspace(-.95, .95, 22) * 1e-6, 'extend':'both'}

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

    ## SET UP PLOTTING GRID AND EVALUATORS
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = pynamit.Grid(lat = lat, lon = lon)
    plt_state_evaluator = pynamit.BasisEvaluator(dynamics.state_basis, plt_grid)
    plt_b_evaluator = pynamit.FieldEvaluator(dynamics.state.mainfield, plt_grid, dynamics.state.RI)

    ## CALCULATE VALUES TO PLOT
    Br  = dynamics.state.get_Br(plt_state_evaluator)
    FAC = (plt_state_evaluator.scaled_G(1 / plt_b_evaluator.br.reshape((-1, 1)))).dot(dynamics.state.m_imp.coeffs * dynamics.state.m_imp_to_jr)
    jr_mod = plt_state_evaluator.G.dot(dynamics.state.m_imp.coeffs * dynamics.state.m_imp_to_jr)
    eq_current_function = dynamics.state.get_Jeq(plt_state_evaluator)

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


    # scatter plot high latitude jr
    iii = np.abs(dynamics.state_grid.lat) > dynamics.state.latitude_boundary
    jrmax = np.max(np.abs(dynamics.state.jr))
    ax_1.scatter(dynamics.state.jr, jr_mod[iii])
    ax_1.plot([-jrmax, jrmax], [-jrmax, jrmax], 'k-')
    ax_1.set_xlabel('Input ')

    # scatter plot FACs at conjugate points
    j_par_ll = dynamics.state.G_par_ll.dot(dynamics.state.m_imp.coeffs)
    j_par_cp = dynamics.state.G_par_cp.dot(dynamics.state.m_imp.coeffs)
    j_par_max = np.max(np.abs(j_par_ll))
    ax_2.scatter(j_par_ll, j_par_cp)
    ax_2.plot([-j_par_max, j_par_max], [-j_par_max, j_par_max], 'k-')
    ax_2.set_xlabel(r'$j_\parallel$ [A/m$^2$] at |latitude| $< {}^\circ$'.format(dynamics.state.latitude_boundary))
    ax_2.set_ylabel(r'$j_\parallel$ [A/m$^2$] at conjugate points')

    # scatter plot of Ed1 and Ed2 vs corresponding values at conjugate points
    cu_ll    = dynamics.state.u_phi_cp * dynamics.state.aup_cp     + dynamics.state.u_theta_cp * dynamics.state.aut_cp
    cu_cp    = dynamics.state.u_phi_ll * dynamics.state.aup_ll     + dynamics.state.u_theta_ll * dynamics.state.aut_ll
    A_imp_ll = dynamics.state.etaP_ll  * dynamics.state.aeP_imp_ll + dynamics.state.etaH_ll    * dynamics.state.aeH_imp_ll
    A_imp_cp = dynamics.state.etaP_cp  * dynamics.state.aeP_imp_cp + dynamics.state.etaH_cp    * dynamics.state.aeH_imp_cp
    A_ind_ll = dynamics.state.etaP_ll  * dynamics.state.aeP_ind_ll + dynamics.state.etaH_ll    * dynamics.state.aeH_ind_ll
    A_ind_cp = dynamics.state.etaP_cp  * dynamics.state.aeP_ind_cp + dynamics.state.etaH_cp    * dynamics.state.aeH_ind_cp

    c_ll = cu_ll + A_ind_ll.dot(dynamics.state.m_ind.coeffs)
    c_cp = cu_cp + A_ind_cp.dot(dynamics.state.m_ind.coeffs)
    Ed1_ll, Ed2_ll = np.split(c_ll + A_imp_ll.dot(dynamics.state.m_imp.coeffs), 2)
    Ed1_cp, Ed2_cp = np.split(c_cp + A_imp_cp.dot(dynamics.state.m_imp.coeffs), 2)
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


debugplot(dynamics, title = 'hoi!', filename = None, noon_longitude = noon_longitude)



