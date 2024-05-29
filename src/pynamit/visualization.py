import numpy as np
import matplotlib.pyplot as plt
import polplot
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from polplot import Polarplot
import pynamit

def cs_interpolate(projection, inlat, inlon, values, outlat, outlon, **kwargs):
    """ Interpolate from cubed sphere grid to new points ``lon``, ``lat``.

    Parameters
    ----------
    projection: CSProjection
        Cubed sphere projection object.
    inlat: array
        Latitudes of input.
    inlon: array
        Longitudes of input.
    """

    inlat, inlon, values = map(np.ravel, np.broadcast_arrays(inlat, inlon, values))
    in_r = np.vstack((np.cos(np.deg2rad(inlat)) * np.cos(np.deg2rad(inlon)),
                      np.cos(np.deg2rad(inlat)) * np.sin(np.deg2rad(inlon)),
                      np.sin(np.deg2rad(inlat))  
                    ))

    outlon, outlat = np.broadcast_arrays(outlon, outlat)
    shape = outlon.shape # get the shape so we can reshape the result in the end
    outlon, outlat = outlon.flatten(), outlat.flatten()

    result = np.zeros_like(outlon) -1 

    xi_o, eta_o, block_o = projection.geo2cube(outlon, outlat)

    # go through each block:
    for i in range(6):
        jjj = (block_o == i) # these are the points we want to specify

        # find the points that are on the right side:
        _, th0, ph0 = projection.cube2spherical(0, 0, i)
        r0 = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)])
        iii = np.sum(r0.reshape((-1, 1)) * in_r, axis = 0) > 0
        xi_i, eta_i, _ = projection.geo2cube(inlon[iii], inlat[iii], block = i)
        result[jjj] = griddata(np.vstack((xi_i, eta_i)).T, values[iii], np.vstack((xi_o[jjj], eta_o[jjj])).T, **kwargs)

    return(result.reshape(shape))



def globalplot(lon, lat, data, noon_longitude = 0, scatter = False, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    
    if 'title' in kwargs.keys():
        title = kwargs.pop('title')
    else:
        title = None
    if 'save' in kwargs.keys():
        save = kwargs.pop('save')
    else:
        save = None

    if 'returnplot' in kwargs.keys():
        returnplot = kwargs.pop('returnplot')
    else:
        returnplot = False



    # global plot:
    global_projection = ccrs.PlateCarree(central_longitude = noon_longitude)
    ax = fig.add_subplot(2, 1, 2, projection = global_projection)    
    ax.coastlines(zorder = 2, color = 'grey')
    if scatter:
        ax.scatter(lon, lat, c = data, transform = ccrs.PlateCarree(), **kwargs)
    else:
        ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), **kwargs)
    
    if title is not None:
        ax.set_title(title)

    pax1 = polplot.Polarplot(fig.add_subplot(2, 2, 1), minlat = 50)
    pax2 = polplot.Polarplot(fig.add_subplot(2, 2, 2), minlat = 50)

    lon = lon - noon_longitude + 180 # rotate so that noon is up

    iii = lat > 50
    if scatter:
        pax1.scatter(lat[iii],  lon[iii] / 15, c = data[iii], **kwargs)
    else:
        pax1.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax1.ax.set_title('North')

    iii = lat < -50
    if scatter:
        pax2.scatter(lat[iii], lon[iii] / 15, c = data[iii], **kwargs)
    else:
        pax2.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax2.ax.set_title('South')


    plt.tight_layout()

    if returnplot:
        return(fig, pax1, pax2, ax)


    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


def debugplot(i2d, title = None, filename = None, noon_longitude = 0):

    ## SET UP PLOTTING GRID
    lat, lon = np.linspace(-89.9, 89.9, 60), np.linspace(-180, 180, 100)
    lat, lon = np.meshgrid(lat, lon)
    plt_grid = pynamit.grid.Grid(lat, lon)
    plt_i2d_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, plt_grid)
    plt_b_geometry = pynamit.b_field.BGeometry(i2d.state.mainfield, plt_grid, i2d.state.RI)

    csp_i2d_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, i2d.state.num_grid)

    B_kwargs   = {'cmap':plt.cm.bwr, 'levels':np.linspace(-100, 100, 22) * 1e-9, 'extend':'both'}
    eqJ_kwargs = {'colors':'black', 'levels':np.r_[-210:220:20] * 1e3}
    FAC_kwargs = {'cmap':plt.cm.bwr, 'levels':np.linspace(-.95, .95, 22)/2 * 1e-6, 'extend':'both'}


    ## SET UP PLOTTING GRID AND BASIS EVALUATOR
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = pynamit.grid.Grid(lat, lon)
    plt_i2d_evaluator = pynamit.basis_evaluator.BasisEvaluator(i2d.state.basis, plt_grid)

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

    FAC    = -plt_i2d_evaluator.G.dot(i2d.state.TB.coeffs * i2d.state.TB_to_Jr) / plt_b_geometry.sinI
    jr_mod =  csp_i2d_evaluator.G.dot(i2d.state.TB.coeffs * i2d.state.TB_to_Jr)
    eq_current_function = i2d.state.get_Jeq(plt_i2d_evaluator)

    ## GLOBAL PLOTS
    gax_B.contourf(lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), Br.reshape((NLO, NLA)), transform = ccrs.PlateCarree(), **B_kwargs)
    gax_j.contour( lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), eq_current_function.reshape((NLO, NLA)), transform = ccrs.PlateCarree(), **eqJ_kwargs)
    gax_j.contourf(lon.reshape((NLO, NLA)), lat.reshape((NLO, NLA)), FAC.reshape((NLO, NLA)), transform = ccrs.PlateCarree(), **FAC_kwargs)


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
    j_par_cp = i2d.state.G_par_cp.dot(i2d.state.TB.coeffs)
    j_par_max = np.max(np.abs(j_par_ll))
    ax_2.scatter(j_par_ll, j_par_cp)
    ax_2.plot([-j_par_max, j_par_max], [-j_par_max, j_par_max], 'k-')
    ax_2.set_xlabel(r'$j_\parallel$ [A/m$^2$] at |latitude| $< {}^\circ$'.format(i2d.state.latitude_boundary))
    ax_2.set_ylabel(r'$j_\parallel$ [A/m$^2$] at conjugate points')

    # scatter plot of Ed1 and Ed2 vs corresponding values at conjugate points
    cu_cp = i2d.state.u_phi_cp * i2d.state.aup_cp   + i2d.state.u_theta_cp * i2d.state.aut_cp
    cu_ll = i2d.state.u_phi_ll * i2d.state.aup_ll   + i2d.state.u_theta_ll * i2d.state.aut_ll
    AT_ll = i2d.state.etaP_ll  * i2d.state.aeP_T_ll + i2d.state.etaH_ll    * i2d.state.aeH_T_ll
    AT_cp = i2d.state.etaP_cp  * i2d.state.aeP_T_cp + i2d.state.etaH_cp    * i2d.state.aeH_T_cp
    AV_ll = i2d.state.etaP_ll  * i2d.state.aeP_V_ll + i2d.state.etaH_ll    * i2d.state.aeH_V_ll
    AV_cp = i2d.state.etaP_cp  * i2d.state.aeP_V_cp + i2d.state.etaH_cp    * i2d.state.aeH_V_cp

    c_ll = cu_ll + AV_ll.dot(i2d.state.VB.coeffs)
    c_cp = cu_cp + AV_cp.dot(i2d.state.VB.coeffs)
    Ed1_ll, Ed2_ll = np.split(c_ll + AT_ll.dot(i2d.state.TB.coeffs), 2)
    Ed1_cp, Ed2_cp = np.split(c_cp + AT_cp.dot(i2d.state.TB.coeffs), 2)
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


if __name__ == "__main__":

    # import cubedsphere submodule
    import os
    import sys
    cs_path = os.path.join(os.path.dirname(__file__), 'cubedsphere')
    sys.path.insert(0, cs_path)
    import cubedsphere
    csp = cubedsphere.CSProjection() # cubed sphere projection object

    Ncs = 30
    k, i, j = csp.get_gridpoints(Ncs)
    xi, eta = csp.xi(i, Ncs), csp.eta(j, Ncs)
    _, theta, phi = csp.cube2spherical(xi, eta, k, deg = True)

    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)

    from lompe import conductance
    import dipole
    import datetime

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp   = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    hall, pedersen = conductance.hardy_EUV(phi, 90 - theta, Kp, date, starlight = 1, dipole = True)

    hall_plt = cs_interpolate(csp, 90 - theta, phi, hall, lat, lon)
    pede_plt = cs_interpolate(csp, 90 - theta, phi, pedersen, lat, lon)

    globalplot(lon, lat, hall_plt, noon_longitude = lon0, levels = np.linspace(0, 20, 100))

