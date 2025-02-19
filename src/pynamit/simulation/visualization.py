"""Visualization utilities for simulation results.

This module provides plotting functions for visualizing ionospheric simulation results, including global maps, diagnostic plots, and time series visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import polplot
from scipy.interpolate import griddata
from polplot import Polarplot
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator


def cs_interpolate(projection, inlat, inlon, values, outlat, outlon, **kwargs):
    """Interpolate data from cubed sphere to regular grid.

    Parameters
    ----------
    projection : CSBasis
        Cubed sphere projection object
    inlat : array-like
        Latitude coordinates of input data
    inlon : array-like
        Longitude coordinates of input data
    values : array-like
        Field values to interpolate
    outlat : array-like
        Latitude coordinates of output grid
    outlon : array-like
        Longitude coordinates of output grid
    kwargs : dict
        Additional arguments for griddata interpolation
    """
    inlat, inlon, values = map(np.ravel, np.broadcast_arrays(inlat, inlon, values))
    in_r = np.vstack(
        (
            np.cos(np.deg2rad(inlat)) * np.cos(np.deg2rad(inlon)),
            np.cos(np.deg2rad(inlat)) * np.sin(np.deg2rad(inlon)),
            np.sin(np.deg2rad(inlat)),
        )
    )

    outlon, outlat = np.broadcast_arrays(outlon, outlat)
    shape = outlon.shape  # get the shape so we can reshape the result in the end
    outlon, outlat = outlon.flatten(), outlat.flatten()

    result = np.zeros_like(outlon) - 1

    xi_o, eta_o, block_o = projection.geo2cube(outlon, outlat)

    # go through each block:
    for i in range(6):
        jjj = block_o == i  # these are the points we want to specify

        # find the points that are on the right side:
        _, th0, ph0 = projection.cube2spherical(0, 0, i)
        r0 = np.array(
            [np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)]
        )
        iii = np.sum(r0.reshape((-1, 1)) * in_r, axis=0) > 0
        xi_i, eta_i, _ = projection.geo2cube(inlon[iii], inlat[iii], block=i)
        result[jjj] = griddata(
            np.vstack((xi_i, eta_i)).T,
            values[iii],
            np.vstack((xi_o[jjj], eta_o[jjj])).T,
            **kwargs
        )

    return result.reshape(shape)


def globalplot(lon, lat, data, noon_longitude=0, scatter=False, **kwargs):
    """Create global map visualization of field data.

    Parameters
    ----------
    lon : array-like
        Longitude coordinates in degrees
    lat : array-like
        Latitude coordinates in degrees
    data : array-like
        Field values to plot, must broadcast with lon/lat
    title : str, optional
        Plot title
    returnplot : bool, optional
        If True, return figure and axes objects, by default False
    levels : array-like, optional
        Contour level boundaries, by default None for automatic
    cmap : str, optional
        Matplotlib colormap name, by default 'viridis'
    noon_longitude : float, optional
        Longitude of local noon meridian, by default 0
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to extend colormap at ends, by default 'both'

    Returns
    -------
    tuple, optional
        (figure, axes) if returnplot=True
    """
    fig = plt.figure(figsize=(10, 10))

    if "title" in kwargs.keys():
        title = kwargs.pop("title")
    else:
        title = None
    if "save" in kwargs.keys():
        save = kwargs.pop("save")
    else:
        save = None

    if "returnplot" in kwargs.keys():
        returnplot = kwargs.pop("returnplot")
    else:
        returnplot = False

    # global plot:
    global_projection = ccrs.PlateCarree(central_longitude=noon_longitude)
    ax = fig.add_subplot(2, 1, 2, projection=global_projection)
    ax.coastlines(zorder=2, color="grey")
    if scatter:
        ax.scatter(lon, lat, c=data, transform=ccrs.PlateCarree(), **kwargs)
    else:
        ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), **kwargs)

    if title is not None:
        ax.set_title(title)

    pax1 = polplot.Polarplot(fig.add_subplot(2, 2, 1), minlat=50)
    pax2 = polplot.Polarplot(fig.add_subplot(2, 2, 2), minlat=50)

    lon = lon - noon_longitude + 180  # rotate so that noon is up

    iii = lat > 50
    if scatter:
        pax1.scatter(lat[iii], lon[iii] / 15, c=data[iii], **kwargs)
    else:
        pax1.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax1.ax.set_title("North")

    iii = lat < -50
    if scatter:
        pax2.scatter(lat[iii], lon[iii] / 15, c=data[iii], **kwargs)
    else:
        pax2.contourf(lat[iii], lon[iii] / 15, data[iii], **kwargs)
    pax2.ax.set_title("South")

    plt.tight_layout()

    if returnplot:
        return (fig, pax1, pax2, ax)

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


def debugplot(dynamics, title=None, filename=None, noon_longitude=0):
    """Generate diagnostic plots of simulation state.

    Creates visualizations of radial magnetic field, field-aligned currents,
    and equivalent current function for debugging.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object containing current state
    title : str, optional
        Plot title
    filename : str, optional
        If provided, save plot to this file
    noon_longitude : float, optional
        Longitude of local noon meridian, by default 0

    Notes
    -----
    Generates plots on a 50x90 lat-lon grid interpolated from simulation grid.
    Shows:
    - Radial magnetic field (Br)
    - Field-aligned currents normalized by radial field
    - Equivalent current function
    """
    B_kwargs = {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-100, 100, 22) * 1e-9,
        "extend": "both",
    }
    eqJ_kwargs = {"colors": "black", "levels": np.r_[-210:220:20] * 1e3}
    FAC_kwargs = {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-0.95, 0.95, 22) / 2 * 1e-6,
        "extend": "both",
    }

    # MAP PROJECTION:
    global_projection = ccrs.PlateCarree(central_longitude=noon_longitude)

    fig = plt.figure(figsize=(15, 13))

    paxn_B = Polarplot(plt.subplot2grid((4, 4), (0, 0)))
    paxs_B = Polarplot(plt.subplot2grid((4, 4), (0, 1)))
    paxn_j = Polarplot(plt.subplot2grid((4, 4), (0, 2)))
    paxs_j = Polarplot(plt.subplot2grid((4, 4), (0, 3)))
    gax_B = plt.subplot2grid((4, 2), (1, 0), projection=global_projection, rowspan=2)
    gax_j = plt.subplot2grid((4, 2), (1, 1), projection=global_projection, rowspan=2)
    ax_1 = plt.subplot2grid((4, 3), (3, 0))
    ax_2 = plt.subplot2grid((4, 3), (3, 1))
    ax_3 = plt.subplot2grid((4, 3), (3, 2))

    for ax in [gax_B, gax_j]:
        ax.coastlines(zorder=2, color="grey")

    # SET UP PLOTTING GRID AND EVALUATORS
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = Grid(lat=lat, lon=lon)
    plt_state_evaluator = BasisEvaluator(dynamics.state.basis, plt_grid)
    plt_b_evaluator = FieldEvaluator(
        dynamics.state.mainfield, plt_grid, dynamics.state.RI
    )

    # CALCULATE VALUES TO PLOT
    Br = dynamics.state.get_Br(plt_state_evaluator)
    FAC = (
        plt_state_evaluator.G.dot(
            dynamics.state.m_imp.coeffs * dynamics.state.m_imp_to_jr
        )
        / plt_b_evaluator.br
    )
    eq_current_function = dynamics.state.get_Jeq(plt_state_evaluator)

    jr_mod = dynamics.state_basis_evaluator.G.dot(
        dynamics.state.m_imp.coeffs * dynamics.state.m_imp_to_jr
    )

    # GLOBAL PLOTS
    gax_B.contourf(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        Br.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **B_kwargs
    )
    gax_j.contour(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        eq_current_function.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **eqJ_kwargs
    )
    gax_j.contourf(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        FAC.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **FAC_kwargs
    )

    # POLAR PLOTS
    mlt = (lon - noon_longitude + 180) / 15  # rotate so that noon is up

    # north:
    iii = lat > 50
    paxn_B.contourf(lat[iii], mlt[iii], Br[iii], **B_kwargs)
    paxn_j.contour(lat[iii], mlt[iii], eq_current_function[iii], **eqJ_kwargs)
    paxn_j.contourf(lat[iii], mlt[iii], FAC[iii], **FAC_kwargs)

    # south:
    iii = lat < -50
    paxs_B.contourf(lat[iii], mlt[iii], Br[iii], **B_kwargs)
    paxs_j.contour(lat[iii], mlt[iii], eq_current_function[iii], **eqJ_kwargs)
    paxs_j.contourf(lat[iii], mlt[iii], FAC[iii], **FAC_kwargs)

    # scatter plot high latitude jr
    iii = np.abs(dynamics.state_grid.lat) > dynamics.state.latitude_boundary
    jrmax = np.max(np.abs(dynamics.state.jr))
    ax_1.scatter(dynamics.state.jr, jr_mod[iii])
    ax_1.plot([-jrmax, jrmax], [-jrmax, jrmax], "k-")
    ax_1.set_xlabel("Input ")

    # scatter plot FACs at conjugate points
    j_par_ll = dynamics.state.G_par_ll.dot(dynamics.state.m_imp.coeffs)
    j_par_cp = dynamics.state.G_par_cp.dot(dynamics.state.m_imp.coeffs)
    j_par_max = np.max(np.abs(j_par_ll))
    ax_2.scatter(j_par_ll, j_par_cp)
    ax_2.plot([-j_par_max, j_par_max], [-j_par_max, j_par_max], "k-")
    ax_2.set_xlabel(
        r"$j_\parallel$ [A/m$^2$] at |latitude| $< {}^\circ$".format(
            dynamics.state.latitude_boundary
        )
    )
    ax_2.set_ylabel(r"$j_\parallel$ [A/m$^2$] at conjugate points")

    # scatter plot of Ed1 and Ed2 vs corresponding values at conjugate points
    cu_cp = (
        dynamics.state.u_phi_cp * dynamics.state.aup_cp
        + dynamics.state.u_theta_cp * dynamics.state.aut_cp
    )
    cu_ll = (
        dynamics.state.u_phi_ll * dynamics.state.aup_ll
        + dynamics.state.u_theta_ll * dynamics.state.aut_ll
    )
    A_imp_ll = (
        dynamics.state.etaP_ll * dynamics.state.aeP_imp_ll
        + dynamics.state.etaH_ll * dynamics.state.aeH_imp_ll
    )
    A_imp_cp = (
        dynamics.state.etaP_cp * dynamics.state.aeP_imp_cp
        + dynamics.state.etaH_cp * dynamics.state.aeH_imp_cp
    )
    A_ind_ll = (
        dynamics.state.etaP_ll * dynamics.state.aeP_ind_ll
        + dynamics.state.etaH_ll * dynamics.state.aeH_ind_ll
    )
    A_ind_cp = (
        dynamics.state.etaP_cp * dynamics.state.aeP_ind_cp
        + dynamics.state.etaH_cp * dynamics.state.aeH_ind_cp
    )

    c_ll = cu_ll + A_ind_ll.dot(dynamics.state.m_ind.coeffs)
    c_cp = cu_cp + A_ind_cp.dot(dynamics.state.m_ind.coeffs)
    Ed1_ll, Ed2_ll = np.split(c_ll + A_imp_ll.dot(dynamics.state.m_imp.coeffs), 2)
    Ed1_cp, Ed2_cp = np.split(c_cp + A_imp_cp.dot(dynamics.state.m_imp.coeffs), 2)
    ax_3.scatter(Ed1_ll, Ed1_cp, label="$E_{d_1}$")
    ax_3.scatter(Ed2_ll, Ed2_cp, label="$E_{d_2}$")
    ax_3.set_xlabel("$E_{d_i}$")
    ax_3.set_ylabel("$E_{d_i}$ at conjugate points")
    ax_3.legend(frameon=False)

    if title is not None:
        gax_j.set_title(title)

    plt.subplots_adjust(
        top=0.89,
        bottom=0.095,
        left=0.025,
        right=0.95,
        hspace=0.0,
        wspace=0.185,
    )
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    # import cubedsphere submodule
    import os
    import sys

    cs_path = os.path.join(os.path.dirname(__file__), "cubedsphere")
    sys.path.insert(0, cs_path)
    import cubed_sphere

    Ncs = 30
    cs_basis = cubed_sphere.CSBasis(Ncs)  # cubed sphere projection object
    k, i, j = cs_basis.get_gridpoints(Ncs)
    xi, eta = cs_basis.xi(i, Ncs), cs_basis.eta(j, Ncs)
    _, theta, phi = cs_basis.cube2spherical(xi, eta, k, deg=True)

    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)

    from lompe import conductance
    import dipole
    import datetime

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    hall, pedersen = conductance.hardy_EUV(
        phi, 90 - theta, Kp, date, starlight=1, dipole=True
    )

    hall_plt = cs_interpolate(cs_basis, 90 - theta, phi, hall, lat, lon)
    pede_plt = cs_interpolate(cs_basis, 90 - theta, phi, pedersen, lat, lon)

    globalplot(lon, lat, hall_plt, noon_longitude=lon0, levels=np.linspace(0, 20, 100))


def compare_AMPS_jr_and_CF_currents(dynamics, a, d, date, lon0):
    """
    Compare AMPS jr and curl-free currents.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object.
    a : pyAMPS
        pyAMPS object.
    d : Date
        Date object.
    date : datetime
        Date.
    lon0 : float
        Noon longitude.
    """
    # compare jr and curl-free currents:
    _, axes = plt.subplots(ncols=2, nrows=2)
    SCALE = 1e3
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for jr muA/m^2

    # Define grid used for plotting
    Ncs = 30
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape

    paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]

    ju_amps = a.get_upward_current()
    je_amps, jn_amps = a.get_curl_free_current()

    mlat, mlt = a.scalargrid
    mlatv, mltv = a.vectorgrid
    mlatn, mltn = np.split(mlat, 2)[0], np.split(mlt, 2)[0]
    mlatnv, mltnv = np.split(mlatv, 2)[0], np.split(mltv, 2)[0]

    lon = d.mlt2mlon(mlt, date)
    lonv = d.mlt2mlon(mltv, date)

    mn_grid = Grid(lat=mlatn, lon=mltn)
    mnv_grid = Grid(lat=mlatnv, lon=mltnv)

    paxes[0].contourf(
        mn_grid.lat,
        mn_grid.lon,
        np.split(ju_amps, 2)[0],
        levels=levels,
        cmap=plt.cm.bwr,
    )
    paxes[0].quiver(
        mnv_grid.lat,
        mnv_grid.lon,
        np.split(jn_amps, 2)[0],
        np.split(je_amps, 2)[0],
        scale=SCALE,
        color="black",
    )
    paxes[1].contourf(
        mn_grid.lat,
        mn_grid.lon,
        np.split(ju_amps, 2)[1],
        levels=levels,
        cmap=plt.cm.bwr,
    )
    paxes[1].quiver(
        mnv_grid.lat,
        mnv_grid.lon,
        -np.split(jn_amps, 2)[1],
        np.split(je_amps, 2)[1],
        scale=SCALE,
        color="black",
    )

    m_state_evaluator = BasisEvaluator(dynamics.state_basis, Grid(lat=mlat, lon=lon))
    jr = dynamics.get_jr(m_state_evaluator) * 1e6

    mv_state_evaluator = BasisEvaluator(dynamics.state_basis, Grid(lat=mlatv, lon=lonv))
    js, je = dynamics.state.get_JS(mv_state_evaluator) * 1e3
    jn = -js

    jrn, jrs = np.split(jr, 2)
    paxes[2].contourf(mn_grid.lat, mn_grid.lon, jrn, levels=levels, cmap=plt.cm.bwr)
    paxes[2].quiver(
        mnv_grid.lat,
        mnv_grid.lon,
        np.split(jn, 2)[0],
        np.split(je, 2)[0],
        scale=SCALE,
        color="black",
    )
    paxes[3].contourf(mn_grid.lat, mn_grid.lon, jrs, levels=levels, cmap=plt.cm.bwr)
    paxes[3].quiver(
        mnv_grid.lat,
        mnv_grid.lon,
        -np.split(jn, 2)[1],
        np.split(je, 2)[1],
        scale=SCALE,
        color="black",
    )

    plt.show()
    plt.close()

    plt_grid = Grid(lat=lat, lon=lon)
    plt_state_evaluator = BasisEvaluator(dynamics.state_basis, plt_grid)
    jr = dynamics.get_jr(plt_state_evaluator)

    globalplot(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        jr.reshape(pltshape) * 1e6,
        noon_longitude=lon0,
        cmap=plt.cm.bwr,
        levels=levels,
    )


def plot_AMPS_Br(a):
    """
    Plot AMPS Br.

    Parameters
    ----------
    a : pyAMPS
        pyAMPS object.
    """
    Blevels = np.linspace(-300, 300, 22) * 1e-9  # color levels for Br
    _, axes = plt.subplots(ncols=2, figsize=(10, 5))
    paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]

    if not compare_AMPS_jr_and_CF_currents:
        mlat, mlt = a.scalargrid
        mlatn, mltn = np.split(mlat, 2)[0], np.split(mlt, 2)[0]
        mn_grid = Grid(lat=mlatn, lon=mltn)

    Bu = a.get_ground_Buqd(height=a.height)
    paxes[0].contourf(
        mn_grid.lat,
        mn_grid.lon,
        np.split(Bu, 2)[0],
        levels=Blevels * 1e9,
        cmap=plt.cm.bwr,
    )
    paxes[1].contourf(
        mn_grid.lat,
        mn_grid.lon,
        np.split(Bu, 2)[1],
        levels=Blevels * 1e9,
        cmap=plt.cm.bwr,
    )

    plt.show()
    plt.close()


def show_jr_and_conductance(dynamics, conductance_grid, hall, pedersen, lon0):
    """
    Show jr and conductance.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object.
    conductance_grid : Grid
        Conductance grid.
    hall : array-like
        Hall conductance.
    pedersen : array-like
        Pedersen conductance.
    lon0 : float
        Noon longitude.
    """
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for jr muA/m^2
    c_levels = np.linspace(0, 20, 100)  # color levels for conductance

    # Define grid used for plotting
    Ncs = 30
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape

    plt_grid = Grid(lat=lat, lon=lon)
    hall_plt = cs_interpolate(
        cs_basis,
        conductance_grid.lat,
        conductance_grid.lon,
        hall,
        plt_grid.lat,
        plt_grid.lon,
    )
    pede_plt = cs_interpolate(
        cs_basis,
        conductance_grid.lat,
        conductance_grid.lon,
        pedersen,
        plt_grid.lat,
        plt_grid.lon,
    )

    globalplot(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        hall_plt.reshape(pltshape),
        noon_longitude=lon0,
        levels=c_levels,
        save="hall.png",
    )
    globalplot(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        pede_plt.reshape(pltshape),
        noon_longitude=lon0,
        levels=c_levels,
        save="pede.png",
    )

    plt_state_evaluator = BasisEvaluator(dynamics.state_basis, plt_grid)
    jr = dynamics.state.get_jr(plt_state_evaluator)
    globalplot(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        jr.reshape(pltshape),
        noon_longitude=lon0,
        levels=levels * 1e-6,
        save="jr.png",
        cmap=plt.cm.bwr,
    )


def make_colorbars():
    """
    Create colorbars for the plots.
    """
    levels = np.linspace(-0.9, 0.9, 22)  # color levels for jr muA/m^2
    c_levels = np.linspace(0, 20, 100)  # color levels for conductance
    Blevels = np.linspace(-300, 300, 22) * 1e-9  # color levels for Br

    # conductance:
    _, axc = plt.subplots(figsize=(1, 10))
    cz, co = np.zeros_like(c_levels), np.ones_like(c_levels)
    axc.contourf(
        np.vstack((cz, co)).T,
        np.vstack((c_levels, c_levels)).T,
        np.vstack((c_levels, c_levels)).T,
        levels=c_levels,
    )
    axc.set_ylabel("mho", size=16)
    axc.set_xticks([])
    plt.subplots_adjust(left=0.7)
    plt.savefig("conductance_colorbar.png")
    plt.close()

    # jr and Br:
    _, axf = plt.subplots(figsize=(2, 10))
    fz, fo = np.zeros_like(levels), np.ones_like(levels)
    axf.contourf(
        np.vstack((fz, fo)).T,
        np.vstack((levels, levels)).T,
        np.vstack((levels, levels)).T,
        levels=levels,
        cmap=plt.cm.bwr,
    )
    axf.set_ylabel(r"$\mu$A/m$^2$", size=16)
    axf.set_xticks([])

    axB = axf.twinx()
    Bz, Bo = np.zeros_like(Blevels), np.ones_like(Blevels)
    axB.contourf(
        np.vstack((Bz, Bo)).T,
        np.vstack((Blevels, Blevels)).T * 1e9,
        np.vstack((Blevels, Blevels)).T,
        levels=Blevels,
        cmap=plt.cm.bwr,
    )
    axB.set_ylabel(r"nT", size=16)
    axB.set_xticks([])

    plt.subplots_adjust(left=0.45, right=0.6)
    plt.savefig("mag_colorbar.png")
    plt.close()


def time_dependent_plot(
    dynamics,
    fig_directory,
    filecount,
    lon0,
    plt_grid,
    pltshape,
    plt_state_evaluator,
):
    """Create time series visualization frame.

    Generates and saves a single frame for time-dependent visualization of
    simulation evolution.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object with current state
    fig_directory : str
        Directory to save output frames
    filecount : int
        Frame number for filename
    lon0 : float
        Reference longitude for local time
    plt_grid : Grid
        Grid for visualization interpolation
    pltshape : tuple
        Shape of plotting grid (nlat, nlon)
    plt_state_evaluator : BasisEvaluator
        Evaluator for computing fields on plot grid

    Notes
    -----
    Saves frame as PNG with radial field colored contours and
    electric potential contour lines in both hemispheres.
    """
    import os

    Blevels = np.linspace(-300, 300, 22) * 1e-9  # color levels for Br
    Philevels = np.r_[-212.5:212.5:5]  # color levels for Phi

    fn = os.path.join(fig_directory, "new_" + str(filecount).zfill(3) + ".png")
    title = "t = {:.3} s".format(dynamics.current_time)

    Br = dynamics.state.get_Br(plt_state_evaluator)

    _, paxn, paxs, _ = globalplot(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        Br.reshape(pltshape),
        title=title,
        returnplot=True,
        levels=Blevels,
        cmap="bwr",
        noon_longitude=lon0,
        extend="both",
    )

    Phi = dynamics.state.get_Phi(plt_state_evaluator) * 1e-3

    # W = dynamics.state.get_W(plt_state_evaluator) * 1e-3
    nnn = plt_grid.lat.flatten() > 50
    sss = plt_grid.lat.flatten() < -50
    # paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
    # paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
    paxn.contour(
        plt_grid.lat[nnn],
        (plt_grid.lon - lon0)[nnn] / 15,
        Phi[nnn],
        colors="black",
        levels=Philevels,
        linewidths=0.5,
    )
    paxs.contour(
        plt_grid.lat[sss],
        (plt_grid.lon - lon0)[sss] / 15,
        Phi[sss],
        colors="black",
        levels=Philevels,
        linewidths=0.5,
    )
    plt.savefig(fn)
    plt.close()
