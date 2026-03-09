"""Stateless plotting helpers for map-based visualizations."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from polplot import Polarplot


def geocentric_to_plate_carree_vector_components(
    east: np.ndarray,
    north: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert east/north vectors to PlateCarree-compatible components."""
    magnitude = np.sqrt(east**2 + north**2)
    east_pc = east / np.cos(latitude * np.pi / 180)
    magnitude_pc = np.sqrt(east_pc**2 + north**2)
    east_pc = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc
    return east_pc, north_pc


def make_global_projection(noon_longitude: float) -> ccrs.PlateCarree:
    """Return a global PlateCarree projection centered on local noon."""
    return ccrs.PlateCarree(central_longitude=float(noon_longitude))


def decorate_global_axes(
    ax: Any,
    *,
    mainfield: Any,
    latitude_boundary: float,
    draw_labels: bool = True,
    draw_coastlines: bool = True,
) -> None:
    """Add coastlines, gridlines, and dip-equator/boundary overlays."""
    if draw_coastlines:
        ax.coastlines(zorder=2, color="grey")

    gridlines = ax.gridlines(draw_labels=draw_labels)
    gridlines.right_labels = False
    gridlines.top_labels = False

    ll = np.linspace(-180, 180, 200)
    dip_lat = 90 - mainfield.dip_equator(ll)
    lbn = 90 - mainfield.dip_equator(ll, theta=90 - latitude_boundary)
    lbs = 90 - mainfield.dip_equator(ll, theta=90 + latitude_boundary)

    ax.plot(ll, dip_lat, color="blue", linestyle="--", linewidth=1, transform=ccrs.PlateCarree())
    ax.plot(ll, lbn, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())
    ax.plot(ll, lbs, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())


def plot_scalar_map_on_ax(
    ax: Any,
    lon_coords_2d: np.ndarray,
    lat_coords_2d: np.ndarray,
    data_2d_arr: np.ndarray,
    *,
    title: str = "",
    cmap: str = "viridis",
    norm: Any = None,
) -> Any:
    """Plot one scalar map on a Cartopy axis."""
    if norm is None:
        raise ValueError("Norm object must be provided to plot_scalar_map_on_ax.")
    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
    data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)
    image = ax.pcolormesh(
        lon_coords_2d,
        lat_coords_2d,
        data_to_plot_masked,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )
    ax.set_title(title, fontsize=9)
    return image


def get_ticks_from_levels(plot_kwargs: dict[str, Any]) -> Optional[np.ndarray]:
    """Return midpoint ticks for contour levels when available."""
    levels = plot_kwargs.get("levels")
    if levels is not None and len(levels) > 1:
        return (levels[:-1] + levels[1:]) / 2
    return None


def remove_artists(artist_list: list[Any]) -> None:
    """Remove previously drawn matplotlib artists in-place."""
    for artist in artist_list:
        if artist:
            artist.remove()
    artist_list.clear()


def create_comparison_figure_axes(
    plot_mode: str,
    *,
    global_projection: Optional[ccrs.Projection] = None,
    existing_fig: Optional[plt.Figure] = None,
    polar_minlat: float = 50.0,
) -> dict[str, Any]:
    """Create grouped axes for notebook-style inductive/steady/difference plots."""
    if existing_fig is not None:
        existing_fig.clear()
        fig = existing_fig
        fig.set_constrained_layout(True)
    else:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)

    axes_groups: list[list[Any]] = []
    cbar_axes: list[Any] = []

    if plot_mode == "hemispheres":
        fig.set_figheight(8)
        gs = gridspec.GridSpec(
            2, 5, figure=fig, width_ratios=[1, 1, 1, 0.05, 0.05], wspace=0.2, hspace=0.15
        )
        paxn_s = Polarplot(fig.add_subplot(gs[0, 0]), minlat=polar_minlat)
        paxn_ss = Polarplot(fig.add_subplot(gs[0, 1]), minlat=polar_minlat)
        paxn_d = Polarplot(fig.add_subplot(gs[0, 2]), minlat=polar_minlat)
        paxs_s = Polarplot(fig.add_subplot(gs[1, 0]), minlat=polar_minlat)
        paxs_ss = Polarplot(fig.add_subplot(gs[1, 1]), minlat=polar_minlat)
        paxs_d = Polarplot(fig.add_subplot(gs[1, 2]), minlat=polar_minlat)
        for pax in (paxn_s, paxn_ss, paxn_d, paxs_s, paxs_ss, paxs_d):
            pax.ax.set_aspect("equal", adjustable="box")
        cax1, cax2 = fig.add_subplot(gs[:, 3]), fig.add_subplot(gs[:, 4])
        paxn_s.ax.set_title("Inductive", fontsize=18)
        paxn_ss.ax.set_title("Magnetostatic", fontsize=18)
        paxn_d.ax.set_title("Difference", fontsize=18)
        paxn_s.ax.text(
            -0.4, 0.5, "NORTH", transform=paxn_s.ax.transAxes, ha="center", va="center", rotation=90, fontsize=18
        )
        paxs_s.ax.text(
            -0.4, 0.5, "SOUTH", transform=paxs_s.ax.transAxes, ha="center", va="center", rotation=90, fontsize=18
        )
        axes_groups = [[paxn_s, paxn_ss, paxn_d], [paxs_s, paxs_ss, paxs_d]]
        cbar_axes = [cax1, cax2]
    elif plot_mode == "global":
        projection = global_projection if global_projection is not None else make_global_projection(0.0)
        gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 0.05, 0.05], wspace=0.1)
        ax_s = fig.add_subplot(gs[0, 0], projection=projection)
        ax_ss = fig.add_subplot(gs[0, 1], projection=projection)
        ax_d = fig.add_subplot(gs[0, 2], projection=projection)
        cax1, cax2 = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])
        ax_s.set_title("Inductive", fontsize=14)
        ax_ss.set_title("Magnetostatic", fontsize=14)
        ax_d.set_title("Difference", fontsize=14)
        for ax in (ax_s, ax_ss, ax_d):
            ax.coastlines()
        axes_groups = [[ax_s, ax_ss, ax_d]]
        cbar_axes = [cax1, cax2]
    else:
        raise ValueError("plot_mode must be either 'hemispheres' or 'global'")

    return {"fig": fig, "axes_groups": axes_groups, "cbar_axes": cbar_axes}


def draw_comparison_colorbars(
    fig: plt.Figure,
    cbar_axes: list[Any],
    main_mappable: Any,
    diff_mappable: Any,
    *,
    main_specs: list[dict[str, Any]],
    diff_specs: list[dict[str, Any]],
) -> tuple[Any, Any]:
    """Draw colorbars or text labels for notebook comparison figures."""
    cbar1, cbar2 = None, None
    cax1, cax2 = cbar_axes

    if main_mappable:
        main_spec = main_specs[0]
        label_text = f"{main_spec.get('symbol', '')} ({main_spec.get('units', '')})"
        cbar1 = fig.colorbar(main_mappable, cax=cax1, ticks=get_ticks_from_levels(main_spec))
        cbar1.ax.yaxis.get_offset_text().set_fontsize(12)
        cbar1.set_label(label_text, size=14)
        cbar1.ax.tick_params(labelsize=12)
    else:
        cax1.cla()
        cax1.axis("off")
        y_pos = np.linspace(0.75, 0.25, len(main_specs)) if len(main_specs) > 1 else [0.5]
        for i, spec in enumerate(main_specs):
            interval = spec["levels"][1] - spec["levels"][0]
            label = f"{spec.get('symbol', '')}, interval: {interval:.2e} {spec.get('units', '')}"
            cax1.text(
                0.5,
                y_pos[i],
                label,
                ha="center",
                va="center",
                rotation="vertical",
                color=spec.get("colors", "black"),
                fontsize=14,
            )

    if diff_mappable:
        diff_spec = diff_specs[0]
        label_text = f"{diff_spec.get('symbol', '')} ({diff_spec.get('units', '')})"
        cbar2 = fig.colorbar(diff_mappable, cax=cax2, ticks=get_ticks_from_levels(diff_spec))
        cbar2.ax.yaxis.get_offset_text().set_fontsize(12)
        cbar2.set_label(label_text, size=14)
        cbar2.ax.tick_params(labelsize=12)
    else:
        cax2.cla()
        cax2.axis("off")
        y_pos = np.linspace(0.75, 0.25, len(diff_specs)) if len(diff_specs) > 1 else [0.5]
        for i, spec in enumerate(diff_specs):
            interval = spec["levels"][1] - spec["levels"][0]
            label = f"{spec.get('symbol', '')}, interval: {interval:.2e} {spec.get('units', '')}"
            cax2.text(
                0.5,
                y_pos[i],
                label,
                ha="center",
                va="center",
                rotation="vertical",
                color=spec.get("colors", "black"),
                fontsize=14,
            )

    return cbar1, cbar2


def draw_comparison_field_sets(
    axes_groups: list[list[Any]],
    *,
    variables: list[str],
    fields_dict: dict[str, np.ndarray],
    plot_specs: dict[str, dict[str, Any]],
    diff_specs: dict[str, dict[str, Any]],
    global_lat: np.ndarray,
    global_lon: np.ndarray,
    north_mask: np.ndarray,
    south_mask: np.ndarray,
    time: Any,
    dipole: Any = None,
) -> tuple[list[Any], Any, Any]:
    """Draw inductive/steady/difference field sets on grouped map axes."""
    new_artists: list[Any] = []
    main_mappable = None
    diff_mappable = None

    if dipole is not None:
        polar_lat, polar_lon = dipole.geo2mag(global_lat, global_lon)
        polar_lon = dipole.mlon2mlt(polar_lon, time)
    else:
        polar_lat = global_lat
        polar_lon = (global_lon + 180.0) % 360.0 / 15.0

    for variable in variables:
        state_field = fields_dict[f"{variable}_state"]
        steady_field = fields_dict[f"{variable}_steady"]
        diff_field = state_field - steady_field
        all_specs = [plot_specs[variable], plot_specs[variable], diff_specs[variable]]
        all_fields = [state_field, steady_field, diff_field]

        for group_index, group in enumerate(axes_groups):
            is_polar = isinstance(group[0], Polarplot)
            mask = north_mask if group_index == 0 else south_mask
            for plot_index, (ax, field, spec) in enumerate(zip(group, all_fields, all_specs)):
                artist_kwargs = {k: v for k, v in spec.items() if k not in {"symbol", "units"}}
                plot_func = ax.contourf if "cmap" in artist_kwargs else ax.contour

                if is_polar:
                    artist = plot_func(polar_lat[mask], polar_lon[mask], field[mask], **artist_kwargs)
                else:
                    gridlines = ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=1,
                        color="gray",
                        alpha=0.5,
                        linestyle="--",
                    )
                    gridlines.top_labels = False
                    gridlines.right_labels = False
                    if plot_index > 0:
                        gridlines.left_labels = False
                    artist = plot_func(
                        global_lon,
                        global_lat,
                        field,
                        transform=ccrs.PlateCarree(),
                        **artist_kwargs,
                    )

                new_artists.append(artist)
                if "cmap" in artist_kwargs:
                    if plot_index < 2:
                        main_mappable = artist
                    elif plot_index == 2:
                        diff_mappable = artist

    return new_artists, main_mappable, diff_mappable


def _get_polar_xy(
    ax: Polarplot,
    polar_lat: np.ndarray,
    polar_lon: np.ndarray,
    *,
    dipole: Any,
    time: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return polar plotting coordinates on a ``Polarplot`` axis."""
    mlt = dipole.mlon2mlt(polar_lon, time)
    return ax._latlt2xy(polar_lat, mlt)


def plot_region_contour(
    ax: Any,
    values: np.ndarray,
    *,
    region: str,
    global_lon: np.ndarray,
    global_lat: np.ndarray,
    polar_lat: np.ndarray,
    polar_lon: np.ndarray,
    dipole: Any,
    time: Any,
    projection: ccrs.Projection,
    **kwargs: Any,
) -> Any:
    """Plot contour lines on either global or polar axes."""
    if region in {"south", "north"}:
        if not isinstance(ax, Polarplot):
            raise TypeError("Polar regions require a Polarplot axis.")
        xx, yy = _get_polar_xy(ax, polar_lat, polar_lon, dipole=dipole, time=time)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="No contour levels were found within the data range."
            )
            return ax.ax.contour(xx, yy, np.asarray(values).reshape(polar_lat.shape), **kwargs)

    if region != "global":
        raise ValueError("region must be one of: global, north, south")
    if not ax.projection.equals(projection):
        raise ValueError("Global contour axis does not match the requested projection.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No contour levels were found within the data range."
        )
        return ax.contour(
            global_lon,
            global_lat,
            np.asarray(values).reshape(global_lon.shape),
            transform=ccrs.PlateCarree(),
            **kwargs,
        )


def plot_region_filled_contour(
    ax: Any,
    values: np.ndarray,
    *,
    region: str,
    global_lon: np.ndarray,
    global_lat: np.ndarray,
    polar_lat: np.ndarray,
    polar_lon: np.ndarray,
    dipole: Any,
    time: Any,
    projection: ccrs.Projection,
    **kwargs: Any,
) -> Any:
    """Plot filled contours on either global or polar axes."""
    if region in {"south", "north"}:
        if not isinstance(ax, Polarplot):
            raise TypeError("Polar regions require a Polarplot axis.")
        xx, yy = _get_polar_xy(ax, polar_lat, polar_lon, dipole=dipole, time=time)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="No contour levels were found within the data range."
            )
            return ax.ax.contourf(xx, yy, np.asarray(values).reshape(polar_lat.shape), **kwargs)

    if region != "global":
        raise ValueError("region must be one of: global, north, south")
    if not ax.projection.equals(projection):
        raise ValueError("Global contour axis does not match the requested projection.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No contour levels were found within the data range."
        )
        return ax.contourf(
            global_lon,
            global_lat,
            np.asarray(values).reshape(global_lon.shape),
            transform=ccrs.PlateCarree(),
            **kwargs,
        )


def plot_region_quiver(
    ax: Any,
    east: np.ndarray,
    north: np.ndarray,
    *,
    region: str,
    global_lon: np.ndarray,
    global_lat: np.ndarray,
    projection: ccrs.Projection,
    **kwargs: Any,
) -> Any:
    """Plot a global vector field on a Cartopy axis."""
    if region in {"south", "north"}:
        warnings.warn(
            "Vector plotting on polar grids is not implemented; returning None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if region != "global":
        raise ValueError("region must be one of: global, north, south")
    if not ax.projection.equals(projection):
        raise ValueError("Global quiver axis does not match the requested projection.")
    return ax.quiver(
        global_lon,
        global_lat,
        east,
        north,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )


def plot_polar_filled_contour(
    ax: Polarplot,
    lat: np.ndarray,
    mlt: np.ndarray,
    values: np.ndarray,
    **kwargs: Any,
) -> Any:
    """Plot filled contours on a ``Polarplot`` using latitude/MLT coordinates."""
    return ax.contourf(np.asarray(lat), np.asarray(mlt), np.asarray(values), **kwargs)


def plot_polar_contour(
    ax: Polarplot,
    lat: np.ndarray,
    mlt: np.ndarray,
    values: np.ndarray,
    **kwargs: Any,
) -> Any:
    """Plot contour lines on a ``Polarplot`` using latitude/MLT coordinates."""
    return ax.contour(np.asarray(lat), np.asarray(mlt), np.asarray(values), **kwargs)


def build_even_global_curve_sites(
    *,
    min_lat: float = -75.0,
    max_lat: float = 75.0,
    lat_step: float = 10.0,
    equatorial_spacing_deg: float = 18.0,
    min_sites_per_row: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return approximately even lon/lat anchor locations for curve-map plots."""
    latitudes = np.arange(float(min_lat), float(max_lat) + 0.5 * float(lat_step), float(lat_step))
    equatorial_count = max(int(round(360.0 / max(float(equatorial_spacing_deg), 1.0))), 1)

    lon_sites: list[np.ndarray] = []
    lat_sites: list[np.ndarray] = []
    for row_index, latitude in enumerate(latitudes):
        cos_lat = float(np.cos(np.deg2rad(latitude)))
        row_count = max(min_sites_per_row, int(round(equatorial_count * max(cos_lat, 0.2))))
        if row_count <= 0:
            continue
        row_spacing = 360.0 / float(row_count)
        row_offset = 0.5 * row_spacing if row_index % 2 else 0.0
        row_lons = np.linspace(-180.0, 180.0, row_count, endpoint=False) + row_offset
        row_lons = ((row_lons + 180.0) % 360.0) - 180.0
        lon_sites.append(row_lons)
        lat_sites.append(np.full(row_lons.shape, latitude, dtype=float))

    if not lon_sites:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(lon_sites), np.concatenate(lat_sites)


def _wrap_longitudes(lon_values: np.ndarray, *, central_longitude: float) -> np.ndarray:
    """Wrap longitudes into the visible PlateCarree interval."""
    center = float(central_longitude)
    return ((np.asarray(lon_values, dtype=float) - center + 180.0) % 360.0) + center - 180.0


def _split_wrapped_curve(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    *,
    central_longitude: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a local curve into dateline-safe segments."""
    lon_wrapped = _wrap_longitudes(lon_values, central_longitude=central_longitude)
    lat_arr = np.asarray(lat_values, dtype=float)
    finite_mask = np.isfinite(lon_wrapped) & np.isfinite(lat_arr)
    if not np.any(finite_mask):
        return []

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start_idx = 0
    n_points = lon_wrapped.size
    while start_idx < n_points:
        while start_idx < n_points and not finite_mask[start_idx]:
            start_idx += 1
        if start_idx >= n_points:
            break
        end_idx = start_idx + 1
        while end_idx < n_points and finite_mask[end_idx]:
            end_idx += 1

        lon_slice = lon_wrapped[start_idx:end_idx]
        lat_slice = lat_arr[start_idx:end_idx]
        if lon_slice.size >= 2:
            jump_indices = np.where(np.abs(np.diff(lon_slice)) > 180.0)[0] + 1
            split_points = np.r_[0, jump_indices, lon_slice.size]
            for begin, end in zip(split_points[:-1], split_points[1:]):
                if end - begin >= 2:
                    segments.append((lon_slice[begin:end], lat_slice[begin:end]))
        start_idx = end_idx
    return segments


def draw_timeseries_curve_map(
    ax: Any,
    *,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    normalized_time: np.ndarray,
    layers: list[dict[str, Any]],
    curve_width_deg: float = 10.0,
    curve_height_deg: float = 3.0,
    value_scale: Optional[float] = None,
    central_longitude: float = 0.0,
    show_anchor_points: bool = False,
    anchor_point_kwargs: Optional[dict[str, Any]] = None,
    add_legend: bool = True,
    legend_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Draw small time-series curves anchored at global map locations."""
    lon_sites = np.asarray(site_lon, dtype=float).reshape(-1)
    lat_sites = np.asarray(site_lat, dtype=float).reshape(-1)
    time_arr = np.asarray(normalized_time, dtype=float).reshape(-1)
    if lon_sites.size != lat_sites.size:
        raise ValueError("site_lon and site_lat must have the same length.")
    if time_arr.size == 0:
        raise ValueError("normalized_time must not be empty.")

    all_values: list[np.ndarray] = []
    for layer in layers:
        values = np.asarray(layer["values"], dtype=float)
        if values.shape != (lon_sites.size, time_arr.size):
            raise ValueError(
                "Layer values must have shape "
                f"({lon_sites.size}, {time_arr.size}), got {values.shape}."
            )
        all_values.append(values.reshape(-1))

    if value_scale is None:
        finite_values = np.concatenate([vals[np.isfinite(vals)] for vals in all_values if np.any(np.isfinite(vals))])
        if finite_values.size == 0:
            scale = 1.0
        else:
            scale = float(np.nanmax(np.abs(finite_values)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
    else:
        scale = max(float(value_scale), np.finfo(float).tiny)

    curve_x = float(curve_width_deg) * (time_arr - 0.5)
    artists: list[Any] = []
    legend_handles: list[Any] = []

    for layer in layers:
        style = {
            "color": layer.get("color", "black"),
            "linewidth": layer.get("linewidth", 1.0),
            "linestyle": layer.get("linestyle", "-"),
            "alpha": layer.get("alpha", 1.0),
            "marker": layer.get("marker", None),
            "markersize": layer.get("markersize", 2.0),
            "zorder": layer.get("zorder", 3),
        }
        values = np.asarray(layer["values"], dtype=float)
        if add_legend:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    label=layer.get("label", ""),
                )
            )

        for site_index in range(lon_sites.size):
            local_lon = lon_sites[site_index] + curve_x
            local_lat = lat_sites[site_index] + float(curve_height_deg) * (values[site_index] / scale)
            for lon_segment, lat_segment in _split_wrapped_curve(
                local_lon,
                local_lat,
                central_longitude=central_longitude,
            ):
                artists.extend(
                    ax.plot(
                        lon_segment,
                        lat_segment,
                        transform=ccrs.PlateCarree(),
                        color=style["color"],
                        linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        alpha=style["alpha"],
                        marker=style["marker"],
                        markersize=style["markersize"],
                        zorder=style["zorder"],
                    )
                )

    anchor_scatter = None
    if show_anchor_points:
        point_kwargs = {"marker": "x", "s": 10, "color": "black", "linewidths": 0.6, "zorder": 2}
        if anchor_point_kwargs:
            point_kwargs.update(anchor_point_kwargs)
        anchor_scatter = ax.scatter(
            lon_sites,
            lat_sites,
            transform=ccrs.PlateCarree(),
            **point_kwargs,
        )
        artists.append(anchor_scatter)

    legend = None
    if add_legend and legend_handles:
        default_legend_kwargs = {"loc": "lower left", "framealpha": 0.95, "fontsize": 9}
        if legend_kwargs:
            default_legend_kwargs.update(legend_kwargs)
        legend = ax.legend(handles=legend_handles, **default_legend_kwargs)

    return {
        "artists": artists,
        "legend": legend,
        "value_scale": scale,
        "anchor_scatter": anchor_scatter,
    }
