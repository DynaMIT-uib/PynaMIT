"""Diagnostic plotting utilities for simulation results.

This module contains plotting functions for global field maps and
current-output diagnostics.
"""

import datetime as dt

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from kompe import SphericalGrid
from kompe.spherical_transform import SphericalTransform

from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.plotting.hemisphere import (
    DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE,
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.plotting.map_coordinates import MapCoordinateContext
from pynamit.plotting.plot_helpers import style_global_axis
from pynamit.results.grid_evaluation import model_grid_for_geographic_display
from pynamit.results.output_fields import (
    evaluate_boundary_jr,
    evaluate_equivalent_current_function,
    evaluate_induced_Br,
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


def plot_output_diagnostics(
    simulation,
    title=None,
    filename=None,
    noon_longitude=None,
    coordinate_context=None,
    hemisphere_min_abs_latitude=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE,
):
    """Generate diagnostic plots of current simulation output.

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
        Override for the model-coordinate local-noon meridian.
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
    coordinate_reference_time = dt.datetime.fromisoformat(simulation.config.t0)
    plot_time = coordinate_reference_time + dt.timedelta(seconds=float(simulation.current_time))
    main_field = simulation.geometry.main_field
    magnetic_coordinates_available = main_field.kind != "radial"
    model_context = coordinate_context
    if model_context is None:
        if noon_longitude is None:
            noon_longitude = (
                main_field.magnetic_noon_longitude(plot_time)
                if magnetic_coordinates_available
                else main_field.local_noon_longitude(plot_time)
            )
        model_context = MapCoordinateContext.from_noon_longitude(
            noon_longitude,
            longitude_kind="magnetic" if magnetic_coordinates_available else "geographic",
            local_time_kind="magnetic" if magnetic_coordinates_available else "solar",
            reference_time=plot_time,
        )
    global_context = MapCoordinateContext.geographic(plot_time)
    global_projection = global_context.projection()

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
        style_global_axis(ax, coordinate_context=global_context, coastline_color="grey")

    # Display global maps in GEO; evaluate them in model coordinates.
    NLA, NLO = 50, 90
    latitude = np.linspace(-89.9, 89.9, NLA)
    longitude = np.linspace(-180.0, 180.0, NLO)
    geographic_lat, geographic_lon = map(np.ravel, np.meshgrid(latitude, longitude))
    global_grid = model_grid_for_geographic_display(
        simulation.geometry.main_field, geographic_lat, geographic_lon, event_time=plot_time
    )
    global_transform = SphericalTransform(simulation.geometry.horizontal_basis, global_grid)
    global_field_evaluation = MagneticFieldEvaluation(
        simulation.geometry.main_field, global_grid, simulation.config.RI
    )

    global_br = evaluate_induced_Br(simulation, global_transform)
    global_fac = (
        evaluate_boundary_jr(simulation, global_transform) / global_field_evaluation.unit_br
    )
    global_eq_current = evaluate_equivalent_current_function(simulation, global_transform)

    # Evaluate hemisphere fields on the model grid, then express the
    # sample positions in magnetic coordinates for polar display.
    model_lat, model_lon = map(np.ravel, np.meshgrid(latitude, longitude))
    model_grid = SphericalGrid(lat=model_lat, lon=model_lon)
    model_transform = SphericalTransform(simulation.geometry.horizontal_basis, model_grid)
    model_field_evaluation = MagneticFieldEvaluation(
        simulation.geometry.main_field, model_grid, simulation.config.RI
    )
    model_br = evaluate_induced_Br(simulation, model_transform)
    model_fac = evaluate_boundary_jr(simulation, model_transform) / model_field_evaluation.unit_br
    model_eq_current = evaluate_equivalent_current_function(simulation, model_transform)
    if magnetic_coordinates_available:
        geographic_model_lat, geographic_model_lon = main_field.model_to_geo_coordinates(
            model_lat, model_lon
        )
        polar_lat, polar_lon = main_field.geographic_to_magnetic_coordinates(
            geographic_model_lat, geographic_model_lon
        )
    else:
        polar_lat, polar_lon = model_lat, model_lon

    # Make global plots.
    global_br_axis.contourf(
        geographic_lon.reshape((NLO, NLA)),
        geographic_lat.reshape((NLO, NLA)),
        global_br.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **br_kwargs,
    )
    global_current_axis.contour(
        geographic_lon.reshape((NLO, NLA)),
        geographic_lat.reshape((NLO, NLA)),
        global_eq_current.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **equivalent_current_kwargs,
    )
    global_current_axis.contourf(
        geographic_lon.reshape((NLO, NLA)),
        geographic_lat.reshape((NLO, NLA)),
        global_fac.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **fac_kwargs,
    )
    global_equivalent_current_axis.contour(
        geographic_lon.reshape((NLO, NLA)),
        geographic_lat.reshape((NLO, NLA)),
        global_eq_current.reshape((NLO, NLA)),
        transform=ccrs.PlateCarree(),
        **equivalent_current_kwargs,
    )

    # Make polar plots.
    mlt = model_context.longitude_to_local_time(polar_lon, wrap=False)

    north_mask, south_mask = hemisphere_masks_for_latitude(
        polar_lat, min_abs_latitude=hemisphere_min_abs_latitude
    )
    north_br_axis.contourf(
        polar_lat[north_mask], mlt[north_mask], model_br[north_mask], **br_kwargs
    )
    north_current_axis.contour(
        polar_lat[north_mask],
        mlt[north_mask],
        model_eq_current[north_mask],
        **equivalent_current_kwargs,
    )
    north_current_axis.contourf(
        polar_lat[north_mask], mlt[north_mask], model_fac[north_mask], **fac_kwargs
    )

    south_br_axis.contourf(
        polar_lat[south_mask], mlt[south_mask], model_br[south_mask], **br_kwargs
    )
    south_current_axis.contour(
        polar_lat[south_mask],
        mlt[south_mask],
        model_eq_current[south_mask],
        **equivalent_current_kwargs,
    )
    south_current_axis.contourf(
        polar_lat[south_mask], mlt[south_mask], model_fac[south_mask], **fac_kwargs
    )

    if title is not None:
        global_current_axis.set_title(title)

    plt.subplots_adjust(top=0.89, bottom=0.095, left=0.025, right=0.95, hspace=0.0, wspace=0.185)
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()
