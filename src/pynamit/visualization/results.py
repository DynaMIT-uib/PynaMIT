"""Visualization utilities for simulation results.

This module contains plotting functions for visualizing ionospheric
simulation results, including global maps, diagnostic plots, and time
series visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import polplot
from scipy.interpolate import griddata
from polplot import Polarplot
from pynamit.sphere import Grid
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.plot_helpers import style_global_axis
from pynamit.visualization.state_fields import (
    evaluate_Br,
    evaluate_Phi,
    evaluate_equivalent_current_function,
    evaluate_jr,
    evaluate_sheet_current,
)


def cs_interpolate(projection, inlat, inlon, values, outlat, outlon, **kwargs):
    """Interpolate data from cubed sphere to regular grid.

    Parameters
    ----------
    projection : CSBasis
        Cubed sphere projection object.
    inlat : array-like
        Latitude coordinates of input data.
    inlon : array-like
        Longitude coordinates of input data.
    values : array-like
        Field values to interpolate.
    outlat : array-like
        Latitude coordinates of output grid.
    outlon : array-like
        Longitude coordinates of output grid.
    kwargs : dict
        Additional arguments for griddata interpolation.
    """
    inlat, inlon, values = map(np.ravel, np.broadcast_arrays(inlat, inlon, values))
    input_radius_vectors = np.vstack(
        (
            np.cos(np.deg2rad(inlat)) * np.cos(np.deg2rad(inlon)),
            np.cos(np.deg2rad(inlat)) * np.sin(np.deg2rad(inlon)),
            np.sin(np.deg2rad(inlat)),
        )
    )

    outlon, outlat = np.broadcast_arrays(outlon, outlat)
    output_shape = outlon.shape
    outlon, outlat = outlon.reshape(-1), outlat.reshape(-1)

    interpolated = np.zeros_like(outlon) - 1

    output_xi, output_eta, output_block = projection.geo2cube(outlon, outlat)

    for block in range(6):
        target_mask = output_block == block
        _, theta0, phi0 = projection.cube2spherical(0, 0, block)
        block_center_vector = np.array(
            [np.sin(theta0) * np.cos(phi0), np.sin(theta0) * np.sin(phi0), np.cos(theta0)]
        )
        source_visible_mask = (
            np.sum(block_center_vector.reshape((-1, 1)) * input_radius_vectors, axis=0) > 0
        )
        source_xi, source_eta, _ = projection.geo2cube(
            inlon[source_visible_mask], inlat[source_visible_mask], block=block
        )
        interpolated[target_mask] = griddata(
            np.vstack((source_xi, source_eta)).T,
            values[source_visible_mask],
            np.vstack((output_xi[target_mask], output_eta[target_mask])).T,
            **kwargs,
        )

    return interpolated.reshape(output_shape)


def plot_global_polar_map(lon, lat, data, noon_longitude=0, scatter=False, **kwargs):
    """Create global and polar map panels for field data.

    Parameters
    ----------
    lon : array-like
        Longitude coordinates in degrees.
    lat : array-like
        Latitude coordinates in degrees.
    data : array-like
        Field values to plot, must broadcast with lon/lat.
    noon_longitude : float, optional
        Longitude of local noon meridian.
    scatter : bool, optional
        If True, plot data as scatter points.
    **kwargs : dict
        Additional arguments, such as title, save, returnplot, and
        arguments passed to scatter or contourf.

    Returns
    -------
    tuple, optional
        (figure, axes) if `returnplot` is ``True``.
    """
    fig = plt.figure(figsize=(10, 10))

    title = kwargs.pop("title", None)
    save = kwargs.pop("save", None)
    returnplot = kwargs.pop("returnplot", False)
    coordinate_context = kwargs.pop("coordinate_context", None)
    if coordinate_context is None:
        coordinate_context = MapCoordinateContext.from_noon_longitude(noon_longitude)

    global_projection = coordinate_context.projection()
    global_axis = fig.add_subplot(2, 1, 2, projection=global_projection)
    style_global_axis(global_axis, coordinate_context=coordinate_context, coastline_color="grey")
    if scatter:
        global_axis.scatter(lon, lat, c=data, transform=ccrs.PlateCarree(), **kwargs)
    else:
        global_axis.contourf(lon, lat, data, transform=ccrs.PlateCarree(), **kwargs)

    if title is not None:
        global_axis.set_title(title)

    north_axis = polplot.Polarplot(fig.add_subplot(2, 2, 1), minlat=50)
    south_axis = polplot.Polarplot(fig.add_subplot(2, 2, 2), minlat=50)

    mlt = coordinate_context.longitude_to_local_time(lon, wrap=False)

    north_mask = lat > 50
    if scatter:
        north_axis.scatter(lat[north_mask], mlt[north_mask], c=data[north_mask], **kwargs)
    else:
        north_axis.contourf(lat[north_mask], mlt[north_mask], data[north_mask], **kwargs)
    north_axis.ax.set_title("North")

    south_mask = lat < -50
    if scatter:
        south_axis.scatter(lat[south_mask], mlt[south_mask], c=data[south_mask], **kwargs)
    else:
        south_axis.contourf(lat[south_mask], mlt[south_mask], data[south_mask], **kwargs)
    south_axis.ax.set_title("South")

    plt.tight_layout()

    if returnplot:
        return (fig, north_axis, south_axis, global_axis)

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


def plot_state_diagnostics(
    dynamics, title=None, filename=None, noon_longitude=0, coordinate_context=None
):
    """Generate diagnostic plots of simulation state.

    Creates visualizations of radial magnetic field, field-aligned
    currents, and equivalent current function for debugging.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object containing time series data.
    title : str, optional
        Plot title.
    filename : str, optional
        If provided, save plot to this file.
    noon_longitude : float, optional
        Longitude of local noon meridian.
    coordinate_context : MapCoordinateContext, optional
        Map coordinate context for projection and local-time axes.

    Notes
    -----
    Generates plots on a 50x90 lat-lon grid interpolated from
    simulation grid.

    Shows:
    - Radial magnetic field (Br).
    - Field-aligned currents normalized by radial field.
    - Equivalent current function.
    """
    br_kwargs = {"cmap": plt.cm.bwr, "levels": np.linspace(-100, 100, 22) * 1e-9, "extend": "both"}
    equivalent_current_kwargs = {"colors": "black", "levels": np.r_[-210:220:20] * 1e3}
    fac_kwargs = {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-0.95, 0.95, 22) / 2 * 1e-6,
        "extend": "both",
    }
    if coordinate_context is None:
        coordinate_context = MapCoordinateContext.from_noon_longitude(noon_longitude)

    global_projection = coordinate_context.projection()

    fig = plt.figure(figsize=(15, 10))

    north_br_axis = Polarplot(plt.subplot2grid((3, 4), (0, 0)))
    south_br_axis = Polarplot(plt.subplot2grid((3, 4), (0, 1)))
    north_current_axis = Polarplot(plt.subplot2grid((3, 4), (0, 2)))
    south_current_axis = Polarplot(plt.subplot2grid((3, 4), (0, 3)))
    global_br_axis = plt.subplot2grid((3, 3), (1, 0), projection=global_projection, rowspan=2)
    global_current_axis = plt.subplot2grid((3, 3), (1, 1), projection=global_projection, rowspan=2)
    global_equivalent_current_axis = plt.subplot2grid(
        (3, 3), (1, 2), projection=global_projection, rowspan=2
    )

    for ax in [global_br_axis, global_current_axis, global_equivalent_current_axis]:
        style_global_axis(ax, coordinate_context=coordinate_context, coastline_color="grey")

    # Set up plotting grid and evaluators.
    NLA, NLO = 50, 90
    lat, lon = np.linspace(-89.9, 89.9, NLA), np.linspace(-180, 180, NLO)
    lat, lon = map(np.ravel, np.meshgrid(lat, lon))
    plt_grid = Grid(lat=lat, lon=lon)
    state_transform = SphericalTransform(dynamics.state.basis, plt_grid)
    mainfield_evaluator = FieldEvaluator(dynamics.mainfield, plt_grid, dynamics.state.RI)

    # Calculate values to plot.
    br_values = evaluate_Br(dynamics, state_transform)
    fac_values = evaluate_jr(dynamics, state_transform) / mainfield_evaluator.br
    eq_current_function = evaluate_equivalent_current_function(dynamics, state_transform)

    # Make global plots.
    global_br_axis.contourf(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        br_values.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **br_kwargs,
    )
    global_current_axis.contour(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        eq_current_function.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **equivalent_current_kwargs,
    )
    global_current_axis.contourf(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        fac_values.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **fac_kwargs,
    )
    global_equivalent_current_axis.contour(
        lon.reshape((NLO, NLA)),
        lat.reshape((NLO, NLA)),
        eq_current_function.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **equivalent_current_kwargs,
    )

    # Make polar plots.
    mlt = coordinate_context.longitude_to_local_time(lon, wrap=False)

    north_mask = lat > 50
    north_br_axis.contourf(lat[north_mask], mlt[north_mask], br_values[north_mask], **br_kwargs)
    north_current_axis.contour(
        lat[north_mask],
        mlt[north_mask],
        eq_current_function[north_mask],
        **equivalent_current_kwargs,
    )
    north_current_axis.contourf(
        lat[north_mask], mlt[north_mask], fac_values[north_mask], **fac_kwargs
    )

    south_mask = lat < -50
    south_br_axis.contourf(lat[south_mask], mlt[south_mask], br_values[south_mask], **br_kwargs)
    south_current_axis.contour(
        lat[south_mask],
        mlt[south_mask],
        eq_current_function[south_mask],
        **equivalent_current_kwargs,
    )
    south_current_axis.contourf(
        lat[south_mask], mlt[south_mask], fac_values[south_mask], **fac_kwargs
    )

    if title is not None:
        global_current_axis.set_title(title)

    plt.subplots_adjust(top=0.89, bottom=0.095, left=0.025, right=0.95, hspace=0.0, wspace=0.185)
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()


def compare_AMPS_jr_and_CF_currents(dynamics, a, d, date, lon0):
    """Compare AMPS jr and curl-free currents.

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
    # Compare jr and curl-free currents.
    _, axes = plt.subplots(ncols=2, nrows=2)
    SCALE = 1e3
    levels = np.linspace(-0.9, 0.9, 22)  # Color levels for jr (muA/m^2)

    # Define grid used for plotting.
    Ncs = 30
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape

    paxes = [polplot.Polarplot(ax) for ax in axes.reshape(-1)]

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
        mn_grid.lat, mn_grid.lon, np.split(ju_amps, 2)[0], levels=levels, cmap=plt.cm.bwr
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
        mn_grid.lat, mn_grid.lon, np.split(ju_amps, 2)[1], levels=levels, cmap=plt.cm.bwr
    )
    paxes[1].quiver(
        mnv_grid.lat,
        mnv_grid.lon,
        -np.split(jn_amps, 2)[1],
        np.split(je_amps, 2)[1],
        scale=SCALE,
        color="black",
    )

    m_state_evaluator = SphericalTransform(dynamics.horizontal_basis, Grid(lat=mlat, lon=lon))
    jr = evaluate_jr(dynamics, m_state_evaluator) * 1e6

    mv_state_evaluator = SphericalTransform(dynamics.horizontal_basis, Grid(lat=mlatv, lon=lonv))
    js, je = evaluate_sheet_current(dynamics, mv_state_evaluator) * 1e3
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
    plt_state_evaluator = SphericalTransform(dynamics.horizontal_basis, plt_grid)
    jr = evaluate_jr(dynamics, plt_state_evaluator)

    plot_global_polar_map(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        jr.reshape(pltshape) * 1e6,
        noon_longitude=lon0,
        cmap=plt.cm.bwr,
        levels=levels,
    )


def plot_AMPS_Br(a):
    """Plot AMPS Br.

    Parameters
    ----------
    a : pyAMPS
        pyAMPS object.
    """
    Blevels = np.linspace(-300, 300, 22) * 1e-9  # Color levels for Br
    _, axes = plt.subplots(ncols=2, figsize=(10, 5))
    paxes = [polplot.Polarplot(ax) for ax in axes.reshape(-1)]

    mlat, mlt = a.scalargrid
    mlatn, mltn = np.split(mlat, 2)[0], np.split(mlt, 2)[0]
    mn_grid = Grid(lat=mlatn, lon=mltn)

    Bu = a.get_ground_Buqd(height=a.height)
    paxes[0].contourf(
        mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[0], levels=Blevels * 1e9, cmap=plt.cm.bwr
    )
    paxes[1].contourf(
        mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[1], levels=Blevels * 1e9, cmap=plt.cm.bwr
    )

    plt.show()
    plt.close()


def show_jr_and_conductance(dynamics, conductance_grid, hall, pedersen, lon0):
    """Show jr and conductance.

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
    levels = np.linspace(-0.9, 0.9, 22)  # Color levels for jr (muA/m^2)
    c_levels = np.linspace(0, 20, 100)  # Color levels for conductance

    # Define grid used for plotting.
    Ncs = 30
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape

    plt_grid = Grid(lat=lat, lon=lon)
    hall_plt = cs_interpolate(
        dynamics.cs_basis,
        conductance_grid.lat,
        conductance_grid.lon,
        hall,
        plt_grid.lat,
        plt_grid.lon,
    )
    pede_plt = cs_interpolate(
        dynamics.cs_basis,
        conductance_grid.lat,
        conductance_grid.lon,
        pedersen,
        plt_grid.lat,
        plt_grid.lon,
    )

    plot_global_polar_map(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        hall_plt.reshape(pltshape),
        noon_longitude=lon0,
        levels=c_levels,
        save="hall.png",
    )
    plot_global_polar_map(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        pede_plt.reshape(pltshape),
        noon_longitude=lon0,
        levels=c_levels,
        save="pede.png",
    )

    plt_state_evaluator = SphericalTransform(dynamics.horizontal_basis, plt_grid)
    jr = evaluate_jr(dynamics, plt_state_evaluator)
    plot_global_polar_map(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        jr.reshape(pltshape),
        noon_longitude=lon0,
        levels=levels * 1e-6,
        save="jr.png",
        cmap=plt.cm.bwr,
    )


def make_colorbars():
    """Create colorbars for the plots."""
    levels = np.linspace(-0.9, 0.9, 22)  # Color levels for jr (muA/m^2)
    c_levels = np.linspace(0, 20, 100)  # Color levels for conductance
    Blevels = np.linspace(-300, 300, 22) * 1e-9  # Color levels for Br

    # Make conductance colorbar.
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

    # Make jr colorbar.
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

    # Make Br colorbar.
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
    dynamics, fig_directory, filecount, lon0, plt_grid, pltshape, plt_state_evaluator
):
    """Create time series visualization frame.

    Generates and saves a single frame for time-dependent visualization
    of simulation results.

    Parameters
    ----------
    dynamics : Dynamics
        Simulation dynamics object with current state.
    fig_directory : str
        Directory for saving output frames.
    filecount : int
        Frame number for filename.
    lon0 : float
        Reference longitude for local time.
    plt_grid : Grid
        Grid for visualization interpolation.
    pltshape : tuple
        Shape of plotting grid (nlat, nlon).
    plt_state_evaluator : SphericalTransform
        Evaluator for computing fields on plot grid.

    Notes
    -----
    Saves frame as PNG with radial field colored contours and electric
    potential contour lines in both hemispheres.
    """
    import os

    br_levels = np.linspace(-300, 300, 22) * 1e-9
    phi_levels = np.r_[-212.5:212.5:5]

    fn = os.path.join(fig_directory, "new_" + str(filecount).zfill(3) + ".png")
    title = "t = {:.3} s".format(dynamics.current_time)

    br_values = evaluate_Br(dynamics, plt_state_evaluator)

    _, paxn, paxs, _ = plot_global_polar_map(
        plt_grid.lon.reshape(pltshape),
        plt_grid.lat.reshape(pltshape),
        br_values.reshape(pltshape),
        title=title,
        returnplot=True,
        levels=br_levels,
        cmap="bwr",
        noon_longitude=lon0,
        extend="both",
    )

    phi_values = evaluate_Phi(dynamics, plt_state_evaluator) * 1e-3

    north_mask = plt_grid.lat.reshape(-1) > 50
    south_mask = plt_grid.lat.reshape(-1) < -50
    paxn.contour(
        plt_grid.lat[north_mask],
        (plt_grid.lon - lon0)[north_mask] / 15,
        phi_values[north_mask],
        colors="black",
        levels=phi_levels,
        linewidths=0.5,
    )
    paxs.contour(
        plt_grid.lat[south_mask],
        (plt_grid.lon - lon0)[south_mask] / 15,
        phi_values[south_mask],
        colors="black",
        levels=phi_levels,
        linewidths=0.5,
    )
    plt.savefig(fn)
    plt.close()
