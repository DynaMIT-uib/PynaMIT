"""Visualization utilities for simulation results.

This module contains plotting functions for global field maps and
current-state diagnostics.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.sphere import Grid
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.visualization.hemisphere import (
    DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE,
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.plot_helpers import style_global_axis
from pynamit.visualization.state_fields import (
    evaluate_Br,
    evaluate_equivalent_current_function,
    evaluate_jr,
)


def plot_global_polar_map(
    lon,
    lat,
    data,
    noon_longitude=0,
    scatter=False,
    hemisphere_min_abs_latitude=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE,
    **kwargs,
):
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

    north_axis = make_hemisphere_polarplot(fig.add_subplot(2, 2, 1), hemisphere_min_abs_latitude)
    south_axis = make_hemisphere_polarplot(fig.add_subplot(2, 2, 2), hemisphere_min_abs_latitude)

    mlt = coordinate_context.longitude_to_local_time(lon, wrap=False)

    north_mask, south_mask = hemisphere_masks_for_latitude(
        lat, min_abs_latitude=hemisphere_min_abs_latitude
    )
    if scatter:
        north_axis.scatter(lat[north_mask], mlt[north_mask], c=data[north_mask], **kwargs)
    else:
        north_axis.contourf(lat[north_mask], mlt[north_mask], data[north_mask], **kwargs)
    north_axis.ax.set_title("North")

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
    simulation,
    title=None,
    filename=None,
    noon_longitude=0,
    coordinate_context=None,
    hemisphere_min_abs_latitude=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE,
):
    """Generate diagnostic plots of simulation state.

    Creates visualizations of radial magnetic field, field-aligned
    currents, and equivalent current function for debugging.

    Parameters
    ----------
    simulation : Simulation
        Simulation simulation object containing time series data.
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

    north_br_axis = make_hemisphere_polarplot(
        plt.subplot2grid((3, 4), (0, 0)), hemisphere_min_abs_latitude
    )
    south_br_axis = make_hemisphere_polarplot(
        plt.subplot2grid((3, 4), (0, 1)), hemisphere_min_abs_latitude
    )
    north_current_axis = make_hemisphere_polarplot(
        plt.subplot2grid((3, 4), (0, 2)), hemisphere_min_abs_latitude
    )
    south_current_axis = make_hemisphere_polarplot(
        plt.subplot2grid((3, 4), (0, 3)), hemisphere_min_abs_latitude
    )
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
    state_transform = SphericalTransform(simulation.geometry.horizontal_basis, plt_grid)
    main_field_evaluation = MagneticFieldEvaluation(
        simulation.geometry.main_field, plt_grid, simulation.config.RI
    )

    # Calculate values to plot.
    br_values = evaluate_Br(simulation, state_transform)
    fac_values = evaluate_jr(simulation, state_transform) / main_field_evaluation.unit_br
    eq_current_function = evaluate_equivalent_current_function(simulation, state_transform)

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

    north_mask, south_mask = hemisphere_masks_for_latitude(
        lat, min_abs_latitude=hemisphere_min_abs_latitude
    )
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
