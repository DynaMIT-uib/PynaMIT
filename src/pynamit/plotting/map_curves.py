"""Map-curve geometry helpers for plotting workflows."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from cartopy import crs as ccrs
from matplotlib.lines import Line2D

from pynamit.coordinates import (
    datetime_to_utc_hours,
    local_time_hours_to_longitude,
    longitude_to_local_time_hours,
    wrap_longitude_180,
)

CURVE_GROUP_ZORDER_BASE = 4.0
CURVE_GROUP_ZORDER_STEP = 0.025
CURVE_GROUP_REFERENCE_ZOFFSET = 0.004
CURVE_GROUP_MEASURED_ZOFFSET = 0.008
CURVE_GROUP_SECONDARY_ZOFFSET = 0.012
CURVE_GROUP_PRIMARY_ZOFFSET = 0.016
CURVE_LEGEND_ZORDER = 1000.0


def build_even_global_sites(
    *,
    min_lat=-75.0,
    max_lat=75.0,
    lat_step=10.0,
    equatorial_spacing_deg=18.0,
    min_sites_per_row=6,
    lat_count=None,
    equatorial_count=None,
    reference_time=None,
    visually_even=False,
):
    """Build globally distributed longitude/latitude sample sites.

    By default the number of longitude samples scales with ``cos(lat)``,
    which gives approximately even geographic spacing. ``visually_even``
    keeps the same longitude count on every latitude row. That is useful
    for stylized comparison figures.
    """
    min_lat = float(min_lat)
    max_lat = float(max_lat)
    if not np.isfinite(min_lat) or not np.isfinite(max_lat) or min_lat > max_lat:
        raise ValueError("min_lat and max_lat must be finite and ordered.")
    if min_lat < -90.0 or max_lat > 90.0:
        raise ValueError("min_lat and max_lat must lie between -90 and 90 degrees.")
    if lat_count is None:
        lat_step = float(lat_step)
        if not np.isfinite(lat_step) or lat_step <= 0.0:
            raise ValueError("lat_step must be finite and positive.")
        latitudes = np.arange(min_lat, max_lat + 0.5 * lat_step, lat_step)
    else:
        if isinstance(lat_count, (bool, np.bool_)):
            raise ValueError("lat_count must be a positive integer.")
        integer_lat_count = int(lat_count)
        if integer_lat_count != lat_count or integer_lat_count < 1:
            raise ValueError("lat_count must be a positive integer.")
        lat_count = integer_lat_count
        if lat_count == 1:
            center_lat = 0.0 if min_lat <= 0.0 <= max_lat else 0.5 * (min_lat + max_lat)
            latitudes = np.array([center_lat], dtype=float)
        else:
            latitudes = np.linspace(min_lat, max_lat, lat_count)

    if equatorial_count is None:
        equatorial_spacing_deg = float(equatorial_spacing_deg)
        if not np.isfinite(equatorial_spacing_deg) or equatorial_spacing_deg <= 0.0:
            raise ValueError("equatorial_spacing_deg must be finite and positive.")
        equatorial_count = max(int(round(360.0 / equatorial_spacing_deg)), 1)
    else:
        if isinstance(equatorial_count, (bool, np.bool_)):
            raise ValueError("equatorial_count must be a positive integer.")
        integer_equatorial_count = int(equatorial_count)
        if integer_equatorial_count != equatorial_count or integer_equatorial_count < 1:
            raise ValueError("equatorial_count must be a positive integer.")
        equatorial_count = integer_equatorial_count

    if isinstance(min_sites_per_row, (bool, np.bool_)):
        raise ValueError("min_sites_per_row must be a positive integer.")
    integer_min_sites = int(min_sites_per_row)
    if integer_min_sites != min_sites_per_row or integer_min_sites < 1:
        raise ValueError("min_sites_per_row must be a positive integer.")
    min_sites_per_row = integer_min_sites

    lon_sites, lat_sites = [], []
    for row_index, latitude in enumerate(latitudes):
        if visually_even:
            row_count = int(equatorial_count)
        else:
            cos_lat = float(np.cos(np.deg2rad(latitude)))
            row_count = max(min_sites_per_row, int(round(equatorial_count * max(cos_lat, 0.2))))

        if reference_time is None:
            row_spacing = 360.0 / float(row_count)
            row_offset = 0.5 * row_spacing if row_index % 2 else 0.0
            row_lons = np.linspace(-180.0, 180.0, row_count, endpoint=False) + row_offset
        else:
            utc_hours = datetime_to_utc_hours(reference_time)
            row_spacing_hours = 24.0 / float(row_count)
            row_offset_hours = 0.5 * row_spacing_hours if row_index % 2 else 0.0
            row_local_times = (
                12.0
                + (np.arange(row_count, dtype=float) - row_count // 2) * row_spacing_hours
                + row_offset_hours
            ) % 24.0
            row_lons = wrap_longitude_180((row_local_times - utc_hours) * 15.0)

        row_lons = wrap_longitude_180(row_lons)
        lon_sites.append(np.asarray(row_lons, dtype=float))
        lat_sites.append(np.full(row_lons.shape, latitude, dtype=float))

    if not lon_sites:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return np.concatenate(lon_sites), np.concatenate(lat_sites)


def wrap_longitudes(lon_values, central_longitude=0.0):
    """Wrap longitudes around ``central_longitude``."""
    center = float(central_longitude)
    return ((np.asarray(lon_values, dtype=float) - center + 180.0) % 360.0) + center - 180.0


def split_wrapped_curve(lon_values, lat_values, central_longitude=0.0):
    """Split a curve into finite segments without dateline jumps."""
    lon_wrapped = wrap_longitudes(lon_values, central_longitude=central_longitude).reshape(-1)
    lat_arr = np.asarray(lat_values, dtype=float).reshape(-1)
    if lon_wrapped.size != lat_arr.size:
        raise ValueError("lon_values and lat_values must have the same size.")

    finite_mask = np.isfinite(lon_wrapped) & np.isfinite(lat_arr)
    if not np.any(finite_mask):
        return []

    segments = []
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
            for begin, end in pairwise(split_points):
                if end - begin >= 2:
                    segments.append((lon_slice[begin:end], lat_slice[begin:end]))
        start_idx = end_idx
    return segments


def curve_site_group_zorders(
    lon_values,
    *,
    central_longitude=0.0,
    base=CURVE_GROUP_ZORDER_BASE,
    step=CURVE_GROUP_ZORDER_STEP,
):
    """Return stable per-site z-orders in plotted coordinates."""
    lon_arr = np.asarray(lon_values, dtype=float).reshape(-1)
    if lon_arr.size == 0:
        return np.array([], dtype=float)
    plot_lon = ((lon_arr - float(central_longitude) + 180.0) % 360.0) - 180.0
    finite_plot_lon = np.where(np.isfinite(plot_lon), plot_lon, -np.inf)
    order = np.argsort(finite_plot_lon, kind="mergesort")
    ranks = np.empty(lon_arr.size, dtype=float)
    ranks[order] = np.arange(lon_arr.size, dtype=float)
    return float(base) + float(step) * ranks


def curve_layer_zoffset(
    layer,
    *,
    measured_zoffset=CURVE_GROUP_MEASURED_ZOFFSET,
    secondary_zoffset=CURVE_GROUP_SECONDARY_ZOFFSET,
    primary_zoffset=CURVE_GROUP_PRIMARY_ZOFFSET,
):
    """Return a small z-order offset for overplotted map curves."""
    series_key = str(layer.get("series_key", "")).lower()
    label = str(layer.get("label", "")).lower()
    if series_key == "measured" or (not series_key and "measured" in label):
        return float(measured_zoffset)
    if series_key == "equilibrium" or "equilibrium" in label or "non-inductive" in label:
        return float(secondary_zoffset)
    if series_key == "dynamic" or "dynamic" in label:
        return float(primary_zoffset)
    return float(secondary_zoffset)


def build_timeseries_curve_layers(layer_specs, *, visible_series=None):
    """Return validated time-series map curve layer dictionaries."""
    visible = None
    if visible_series is not None:
        visible = {str(series_key) for series_key in visible_series}

    layers = []
    for spec in layer_specs:
        series_key = str(spec.get("series_key", ""))
        if visible is not None and series_key not in visible:
            continue
        if "values" not in spec:
            raise ValueError("Each curve layer spec must include values.")
        layer = dict(spec)
        layer["series_key"] = series_key
        layer["values"] = np.asarray(layer["values"], dtype=float)
        layers.append(layer)
    return layers


def interpolate_curve_value_at_normalized_position(time_values, values, position):
    """Interpolate one curve at a normalized time position."""
    time_arr = np.asarray(time_values, dtype=float).reshape(-1)
    value_arr = np.asarray(values, dtype=float).reshape(-1)
    if time_arr.size != value_arr.size:
        raise ValueError("time_values and values must have the same length.")
    if time_arr.size == 0:
        return np.nan
    position = float(position)
    if not np.isfinite(position):
        raise ValueError("position must be finite.")
    finite = np.isfinite(time_arr) & np.isfinite(value_arr)
    if not np.any(finite):
        return np.nan
    finite_time = time_arr[finite]
    finite_values = value_arr[finite]
    sort_order = np.argsort(finite_time)
    finite_time = finite_time[sort_order]
    finite_values = finite_values[sort_order]
    unique_time, unique_indices = np.unique(finite_time, return_index=True)
    unique_values = finite_values[unique_indices]
    if unique_time.size == 1:
        return float(unique_values[0]) if np.isclose(position, unique_time[0]) else np.nan
    if position < unique_time[0] or position > unique_time[-1]:
        return np.nan
    return float(np.interp(position, unique_time, unique_values))


def _reference_center_values(reference_line, layers, n_sites, n_times):
    center_values = reference_line.get("center_values") if reference_line is not None else None
    if center_values is not None:
        center_values = np.asarray(center_values, dtype=float)
        expected_shape = (n_sites, n_times)
        if center_values.shape != expected_shape:
            raise ValueError(
                f"reference center_values must have shape {expected_shape}, "
                f"got {center_values.shape}."
            )
        return center_values
    fallback_values = None
    dynamic_values = None
    nondynamic_values = None
    for layer in layers:
        series_key = str(layer.get("series_key", "")).lower()
        label = str(layer.get("label", "")).lower()
        values = np.asarray(layer.get("values", []), dtype=float)
        if values.shape != (n_sites, n_times):
            raise ValueError(
                f"Layer values must have shape {(n_sites, n_times)}, got {values.shape}."
            )
        if series_key == "measured" or "measured" in label:
            return values
        if series_key == "dynamic" or ("dynamic" in label and "non-inductive" not in label):
            dynamic_values = values
        if series_key == "equilibrium" or "equilibrium" in label or "non-inductive" in label:
            nondynamic_values = values
        if fallback_values is None:
            fallback_values = values
    if dynamic_values is not None and nondynamic_values is not None:
        stacked = np.stack([dynamic_values, nondynamic_values], axis=0)
        finite = np.isfinite(stacked)
        count = finite.sum(axis=0)
        total = np.where(finite, stacked, 0.0).sum(axis=0)
        center = np.full(total.shape, np.nan, dtype=float)
        np.divide(total, count, out=center, where=count > 0)
        return center
    return fallback_values


def reference_aligned_curve_centers(
    site_lon,
    site_lat,
    normalized_time,
    layers,
    *,
    curve_width_deg,
    curve_height_deg,
    value_scale,
    site_curve_scale=None,
    reference_line=None,
):
    """Return reference-aligned curve centers.

    Curve maps normally center the full time axis on the site.  With a
    reference time, shift each curve so the reference sample intersects
    the site position.  Only curve centers move; the layer values are
    left unchanged.
    """
    lon_sites = np.asarray(site_lon, dtype=float).reshape(-1)
    lat_sites = np.asarray(site_lat, dtype=float).reshape(-1)
    time_arr = np.asarray(normalized_time, dtype=float).reshape(-1)
    curve_lon = lon_sites.copy()
    curve_lat = lat_sites.copy()
    if lon_sites.size != lat_sites.size:
        raise ValueError("site_lon and site_lat must have the same length.")
    if reference_line is None:
        return curve_lon, curve_lat
    if time_arr.size == 0:
        raise ValueError("normalized_time must not be empty when aligning a reference line.")

    reference_position = float(reference_line.get("position", np.nan))
    if not np.isfinite(reference_position):
        raise ValueError("reference_line position must be finite.")

    if site_curve_scale is None:
        site_scale = np.ones(lon_sites.size, dtype=float)
    else:
        site_scale = np.asarray(site_curve_scale, dtype=float).reshape(-1)
        if site_scale.size != lon_sites.size:
            raise ValueError("site_curve_scale must match the number of sites.")
        if not np.all(np.isfinite(site_scale)) or np.any(site_scale <= 0.0):
            raise ValueError("site_curve_scale values must be finite and positive.")

    center_values = _reference_center_values(reference_line, layers, lon_sites.size, time_arr.size)
    if center_values is None:
        return curve_lon, curve_lat

    scale = float(value_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("value_scale must be finite and positive.")
    for site_index in range(lon_sites.size):
        center_value = interpolate_curve_value_at_normalized_position(
            time_arr, center_values[site_index], reference_position
        )
        if not np.isfinite(center_value):
            continue
        curve_lon[site_index] = lon_sites[site_index] - float(curve_width_deg) * (
            reference_position - 0.5
        )
        curve_lat[site_index] = lat_sites[site_index] - float(curve_height_deg) * site_scale[
            site_index
        ] * (center_value / scale)
    return curve_lon, curve_lat


def _validated_curve_map_inputs(site_lon, site_lat, normalized_time, layers):
    """Return normalized sites, times, and validated layer arrays."""
    lon_sites = np.asarray(site_lon, dtype=float).reshape(-1)
    lat_sites = np.asarray(site_lat, dtype=float).reshape(-1)
    time_values = np.asarray(normalized_time, dtype=float).reshape(-1)
    if lon_sites.size != lat_sites.size:
        raise ValueError("site_lon and site_lat must have the same length.")
    if time_values.size == 0:
        raise ValueError("normalized_time must not be empty.")

    layer_values = []
    expected_shape = (lon_sites.size, time_values.size)
    for layer in layers:
        values = np.asarray(layer["values"], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(f"Layer values must have shape {expected_shape}, got {values.shape}.")
        layer_values.append(values)
    return lon_sites, lat_sites, time_values, layer_values


def _curve_map_value_scale(layer_values, requested_scale):
    """Return a finite positive scale for all curve layers."""
    if requested_scale is not None:
        scale = float(requested_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("value_scale must be finite and positive.")
        return scale
    finite_chunks = [values[np.isfinite(values)] for values in layer_values]
    finite_chunks = [values for values in finite_chunks if values.size]
    if not finite_chunks:
        return 1.0
    scale = float(np.nanmax(np.abs(np.concatenate(finite_chunks))))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _curve_map_site_scale(site_curve_scale, n_sites):
    """Return one finite positive vertical scale per site."""
    if site_curve_scale is None:
        return np.ones(n_sites, dtype=float)
    scales = np.asarray(site_curve_scale, dtype=float).reshape(-1)
    if scales.size != n_sites:
        raise ValueError("site_curve_scale must match the number of sites.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("site_curve_scale values must be finite and positive.")
    return scales


def _curve_map_centers(values, site_values, *, name):
    """Return validated curve-center coordinates for one axis."""
    if values is None:
        return site_values
    centers = np.asarray(values, dtype=float).reshape(-1)
    if centers.size != site_values.size:
        raise ValueError(f"{name} must match the number of sites.")
    return centers


def _curve_layer_style(layer, default_linewidth):
    """Return normalized Matplotlib line style for one curve layer."""
    return {
        "color": layer.get("color", "black"),
        "linewidth": layer.get("linewidth", default_linewidth),
        "linestyle": layer.get("linestyle", "-"),
        "alpha": layer.get("alpha", 1.0),
        "marker": layer.get("marker", None),
        "markersize": layer.get("markersize", 2.0),
        "markeredgewidth": layer.get("markeredgewidth", 1.0),
    }


def _draw_curve_reference_lines(
    ax,
    reference_line,
    layers,
    time_values,
    curve_lon_sites,
    curve_lat_sites,
    site_scale,
    site_group_zorders,
    *,
    curve_width_deg,
    curve_height_deg,
    value_scale,
    central_longitude,
    reference_zoffset,
    default_color,
    default_linewidth,
    default_linestyle,
):
    """Draw the reference-time marker at every valid curve site."""
    if reference_line is None:
        return [], None
    reference_position = float(reference_line.get("position", np.nan))
    if not np.isfinite(reference_position):
        return [], None

    style = {
        "color": reference_line.get("color", default_color),
        "linewidth": reference_line.get("linewidth", default_linewidth),
        "linestyle": reference_line.get("linestyle", default_linestyle),
        "alpha": reference_line.get("alpha", 0.95),
    }
    center_values = _reference_center_values(
        reference_line, layers, curve_lon_sites.size, time_values.size
    )
    if center_values is None:
        return [], None
    value_span = float(reference_line.get("value_span", np.nan))
    if not np.isfinite(value_span) or value_span <= 0.0:
        value_span = 2.0 * value_scale
    longitude_offset = float(curve_width_deg) * (reference_position - 0.5)
    half_latitude_span = 0.5 * float(curve_height_deg) * (value_span / value_scale)

    artists = []
    for site_index in range(curve_lon_sites.size):
        center_value = interpolate_curve_value_at_normalized_position(
            time_values, center_values[site_index], reference_position
        )
        if not np.isfinite(center_value):
            continue
        center_latitude = curve_lat_sites[site_index] + float(curve_height_deg) * site_scale[
            site_index
        ] * (center_value / value_scale)
        local_lon = np.full(2, curve_lon_sites[site_index] + longitude_offset, dtype=float)
        local_lat = np.array(
            [center_latitude - half_latitude_span, center_latitude + half_latitude_span]
        )
        zorder = site_group_zorders[site_index] + float(reference_zoffset)
        for lon_segment, lat_segment in split_wrapped_curve(
            local_lon, local_lat, central_longitude=central_longitude
        ):
            artists.extend(
                ax.plot(
                    lon_segment, lat_segment, transform=ccrs.PlateCarree(), zorder=zorder, **style
                )
            )

    if not artists:
        return artists, None
    legend_handle = Line2D([0], [0], **style, label=reference_line.get("label", "Reference time"))
    return artists, legend_handle


def draw_timeseries_curve_map(
    ax,
    site_lon,
    site_lat,
    normalized_time,
    layers,
    *,
    curve_width_deg=10.0,
    curve_height_deg=3.0,
    value_scale=None,
    central_longitude=0.0,
    show_anchor_points=False,
    anchor_point_kwargs=None,
    add_legend=True,
    legend_kwargs=None,
    site_curve_scale=None,
    reference_line=None,
    curve_center_lon=None,
    curve_center_lat=None,
    extra_legend_handles=None,
    default_linewidth=1.0,
    reference_color="#0072B2",
    reference_linewidth=1.5,
    reference_linestyle=(0, (1, 1)),
    reference_zoffset=CURVE_GROUP_REFERENCE_ZOFFSET,
    legend_zorder=CURVE_LEGEND_ZORDER,
):
    """Draw compact time-series curves centered on geographic map sites.

    ``layers`` is a sequence of dictionaries containing at least
    ``values`` with shape ``(n_sites, n_times)``. Styling keys match
    common Matplotlib line keywords.
    """
    lon_sites, lat_sites, time_arr, layer_values = _validated_curve_map_inputs(
        site_lon, site_lat, normalized_time, layers
    )
    scale = _curve_map_value_scale(layer_values, value_scale)
    site_scale = _curve_map_site_scale(site_curve_scale, lon_sites.size)
    curve_lon_sites = _curve_map_centers(curve_center_lon, lon_sites, name="curve_center_lon")
    curve_lat_sites = _curve_map_centers(curve_center_lat, lat_sites, name="curve_center_lat")

    curve_x = float(curve_width_deg) * (time_arr - 0.5)
    site_group_zorders = curve_site_group_zorders(lon_sites, central_longitude=central_longitude)
    artists = []
    legend_handles = []

    for layer, values in zip(layers, layer_values, strict=True):
        style = _curve_layer_style(layer, default_linewidth)
        layer_zoffset = curve_layer_zoffset(layer)
        if add_legend:
            legend_handles.append(Line2D([0], [0], **style, label=layer.get("label", "")))

        for site_index in range(lon_sites.size):
            local_lon = curve_lon_sites[site_index] + curve_x
            local_lat = curve_lat_sites[site_index] + float(curve_height_deg) * site_scale[
                site_index
            ] * (values[site_index] / scale)
            for lon_segment, lat_segment in split_wrapped_curve(
                local_lon, local_lat, central_longitude=central_longitude
            ):
                artists.extend(
                    ax.plot(
                        lon_segment,
                        lat_segment,
                        transform=ccrs.PlateCarree(),
                        zorder=site_group_zorders[site_index] + layer_zoffset,
                        **style,
                    )
                )

    reference_artists, reference_legend_handle = _draw_curve_reference_lines(
        ax,
        reference_line,
        layers,
        time_arr,
        curve_lon_sites,
        curve_lat_sites,
        site_scale,
        site_group_zorders,
        curve_width_deg=curve_width_deg,
        curve_height_deg=curve_height_deg,
        value_scale=scale,
        central_longitude=central_longitude,
        reference_zoffset=reference_zoffset,
        default_color=reference_color,
        default_linewidth=reference_linewidth,
        default_linestyle=reference_linestyle,
    )
    artists.extend(reference_artists)
    if add_legend and reference_legend_handle is not None:
        legend_handles.append(reference_legend_handle)

    anchor_scatter = None
    if show_anchor_points:
        point_kwargs = {"marker": "x", "s": 10, "color": "black", "linewidths": 0.6, "zorder": 2}
        if anchor_point_kwargs:
            point_kwargs.update(anchor_point_kwargs)
        anchor_scatter = ax.scatter(
            lon_sites, lat_sites, transform=ccrs.PlateCarree(), **point_kwargs
        )
        artists.append(anchor_scatter)

    if add_legend and extra_legend_handles:
        legend_handles.extend(extra_legend_handles)

    legend = None
    if add_legend and legend_handles:
        default_legend_kwargs = {"loc": "lower left", "framealpha": 0.95, "fontsize": 9}
        if legend_kwargs:
            default_legend_kwargs.update(legend_kwargs)
        legend = ax.legend(handles=legend_handles, **default_legend_kwargs)
        legend.set_zorder(float(legend_zorder))
        legend.get_frame().set_zorder(float(legend_zorder))

    return {
        "artists": artists,
        "legend": legend,
        "value_scale": scale,
        "anchor_scatter": anchor_scatter,
    }


def draw_curve_scale_inset(
    ax,
    *,
    curve_width_deg,
    curve_height_deg,
    value_scale,
    scale_display_value,
    scale_annotation,
    duration_annotation,
    map_extent=None,
    low_lat_scale_annotation="",
    color="0.2",
):
    """Draw a compact map-curve time/value scale inset."""
    curve_width_deg = float(curve_width_deg)
    curve_height_deg = float(curve_height_deg)
    value_scale = float(value_scale)
    scale_display_value = float(scale_display_value)
    if not np.isfinite(curve_width_deg) or curve_width_deg <= 0.0:
        raise ValueError("curve_width_deg must be finite and positive.")
    if not np.isfinite(curve_height_deg) or curve_height_deg <= 0.0:
        raise ValueError("curve_height_deg must be finite and positive.")
    if not np.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError("value_scale must be finite and positive.")
    if not np.isfinite(scale_display_value) or scale_display_value <= 0.0:
        raise ValueError("scale_display_value must be finite and positive.")

    if map_extent is None:
        map_width_deg, map_height_deg = 360.0, 180.0
    else:
        extent = np.asarray(map_extent, dtype=float).reshape(-1)
        if extent.size != 4 or not np.all(np.isfinite(extent)):
            raise ValueError("map_extent must contain four finite values.")
        map_width_deg = abs(float(extent[1]) - float(extent[0]))
        map_height_deg = abs(float(extent[3]) - float(extent[2]))
        if map_width_deg == 0.0 or map_height_deg == 0.0:
            raise ValueError("map_extent must have non-zero width and height.")

    curve_width_ax = max(curve_width_deg / map_width_deg, 0.02)
    trace_height_ax = max(2.0 * curve_height_deg / map_height_deg, 0.025)
    scale_ratio = scale_display_value / (2.0 * value_scale)
    bar_height_ax = trace_height_ax * scale_ratio

    left_margin_ax = 0.070
    right_margin_ax = 0.006
    bottom_margin_ax = 0.012
    top_margin_ax = 0.012
    inset_width_ax = left_margin_ax + curve_width_ax + right_margin_ax
    inset_height_ax = bottom_margin_ax + bar_height_ax + top_margin_ax
    x_origin = left_margin_ax / inset_width_ax
    x_end = (left_margin_ax + curve_width_ax) / inset_width_ax
    y_bottom = bottom_margin_ax / inset_height_ax
    y_top = (bottom_margin_ax + bar_height_ax) / inset_height_ax
    y_center = 0.5 * (y_bottom + y_top)

    target_scale_axis_x = 0.044
    min_scale_text_x = 0.010
    scale_text_x_offset = inset_width_ax * (x_origin - 0.30)
    x0 = max(target_scale_axis_x - left_margin_ax, min_scale_text_x - scale_text_x_offset)

    scale_ax = ax.inset_axes(
        [x0, 0.036, inset_width_ax, inset_height_ax], transform=ax.transAxes, zorder=11
    )
    scale_ax.set_facecolor("none")
    scale_ax.set_xlim(0.0, 1.0)
    scale_ax.set_ylim(0.0, 1.0)
    scale_ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=True,
        left=True,
        top=False,
        right=False,
        direction="out",
        length=4.0,
        width=1.0,
        colors=color,
        pad=1.0,
    )
    scale_ax.set_xticks([x_origin, x_end])
    scale_ax.set_yticks([y_bottom, y_top])
    scale_ax.patch.set_alpha(0.0)

    for spine_name, spine in scale_ax.spines.items():
        spine.set_visible(spine_name in {"left", "bottom"})
        spine.set_color(color)
        spine.set_linewidth(1.2)

    scale_ax.spines["left"].set_position(("axes", x_origin))
    scale_ax.spines["left"].set_bounds(y_bottom, y_top)
    scale_ax.spines["bottom"].set_position(("axes", y_center))
    scale_ax.spines["bottom"].set_bounds(x_origin, x_end)

    duration_text = scale_ax.text(
        0.5 * (x_origin + x_end),
        y_bottom - 0.10,
        str(duration_annotation),
        ha="center",
        va="top",
        fontsize=9,
        color=color,
        transform=scale_ax.transAxes,
        clip_on=False,
    )
    scale_text = scale_ax.text(
        x_origin - 0.30,
        0.5 * (y_bottom + y_top),
        str(scale_annotation),
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color=color,
        transform=scale_ax.transAxes,
        clip_on=False,
    )
    artists = [scale_ax, duration_text, scale_text]
    if low_lat_scale_annotation:
        artists.append(
            scale_ax.text(
                x_origin - 0.16,
                0.5 * (y_bottom + y_top),
                str(low_lat_scale_annotation),
                ha="center",
                va="center",
                rotation=90,
                fontsize=8.5,
                color=color,
                transform=scale_ax.transAxes,
                clip_on=False,
            )
        )
    return artists


def local_time_window_is_full(lt_min, lt_max):
    """Return whether a local-time window covers the full day."""
    return float(lt_min) <= 0.0 and float(lt_max) >= 24.0


def _coordinate_window(values, *, name, lower, upper, ordered):
    """Return a finite two-value coordinate window."""
    window = np.asarray(values, dtype=float)
    if window.shape != (2,) or not np.all(np.isfinite(window)):
        raise ValueError(f"{name} must contain two finite values.")
    if np.any(window < lower) or np.any(window > upper):
        raise ValueError(f"{name} must lie between {lower:g} and {upper:g}.")
    if ordered and window[0] > window[1]:
        raise ValueError(f"{name} must be ordered from minimum to maximum.")
    return float(window[0]), float(window[1])


def geographic_local_time_mask(
    lat_values,
    lon_values,
    *,
    lat_window=(-90.0, 90.0),
    local_time_window=(0.0, 24.0),
    reference_time=None,
):
    """Return a finite-site mask for latitude/local-time windows."""
    lat_arr = np.asarray(lat_values, dtype=float).reshape(-1)
    lon_arr = np.asarray(lon_values, dtype=float).reshape(-1)
    if lat_arr.size != lon_arr.size:
        raise ValueError("lat_values and lon_values must have the same size.")

    lat_min, lat_max = _coordinate_window(
        lat_window, name="lat_window", lower=-90.0, upper=90.0, ordered=True
    )
    mask = (
        np.isfinite(lat_arr) & np.isfinite(lon_arr) & (lat_arr >= lat_min) & (lat_arr <= lat_max)
    )

    lt_min, lt_max = _coordinate_window(
        local_time_window, name="local_time_window", lower=0.0, upper=24.0, ordered=False
    )
    if local_time_window_is_full(lt_min, lt_max):
        return mask
    if reference_time is None:
        raise ValueError("reference_time is required when local_time_window is not full.")

    local_time = longitude_to_local_time_hours(lon_arr, reference_time)
    if lt_min <= lt_max:
        lt_mask = (local_time >= lt_min) & (local_time <= lt_max)
    else:
        lt_mask = (local_time >= lt_min) | (local_time <= lt_max)
    return mask & lt_mask


def local_time_window_extent(
    *,
    lat_window=(-90.0, 90.0),
    local_time_window=(0.0, 24.0),
    reference_time,
    central_longitude=0.0,
):
    """Return a Cartopy extent for latitude/local-time map windows."""
    lat_min, lat_max = _coordinate_window(
        lat_window, name="lat_window", lower=-90.0, upper=90.0, ordered=True
    )
    lt_min, lt_max = _coordinate_window(
        local_time_window, name="local_time_window", lower=0.0, upper=24.0, ordered=False
    )
    full_lat = lat_min <= -90.0 and lat_max >= 90.0
    full_lt = local_time_window_is_full(lt_min, lt_max)
    if full_lat and full_lt:
        return None

    lat_min = max(float(lat_min), -89.9)
    lat_max = min(float(lat_max), 89.9)
    if full_lt:
        lon_min, lon_max = -180.0, 180.0
    else:
        width_hours = (float(lt_max) - float(lt_min)) % 24.0
        if width_hours <= 0.0:
            width_hours = 24.0
        lon_min = wrap_longitudes(
            local_time_hours_to_longitude(float(lt_min), reference_time),
            central_longitude=central_longitude,
        )
        lon_min = float(np.asarray(lon_min))
        lon_max = lon_min + 15.0 * width_hours
        if lon_max - lon_min >= 359.9:
            lon_min, lon_max = -180.0, 180.0
    return [float(lon_min), float(lon_max), float(lat_min), float(lat_max)]


__all__ = [
    "build_even_global_sites",
    "build_timeseries_curve_layers",
    "curve_layer_zoffset",
    "curve_site_group_zorders",
    "draw_curve_scale_inset",
    "draw_timeseries_curve_map",
    "geographic_local_time_mask",
    "interpolate_curve_value_at_normalized_position",
    "local_time_window_extent",
    "local_time_window_is_full",
    "reference_aligned_curve_centers",
    "split_wrapped_curve",
    "wrap_longitudes",
]
