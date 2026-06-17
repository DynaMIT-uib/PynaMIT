"""Map-curve geometry helpers for visualization workflows."""

from __future__ import annotations

import numpy as np

from pynamit.coordinates import (
    datetime_to_utc_hours,
    local_time_hours_to_longitude,
    longitude_to_local_time_hours,
    wrap_longitude_180,
)


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
    min_lat, max_lat = sorted((float(min_lat), float(max_lat)))
    if lat_count is None:
        latitudes = np.arange(min_lat, max_lat + 0.5 * float(lat_step), float(lat_step))
    else:
        lat_count = max(1, int(lat_count))
        if lat_count == 1:
            center_lat = 0.0 if min_lat <= 0.0 <= max_lat else 0.5 * (min_lat + max_lat)
            latitudes = np.array([center_lat], dtype=float)
        else:
            latitudes = np.linspace(min_lat, max_lat, lat_count)

    if equatorial_count is None:
        equatorial_count = max(int(round(360.0 / max(float(equatorial_spacing_deg), 1.0))), 1)
    else:
        equatorial_count = max(1, int(equatorial_count))

    lon_sites, lat_sites = [], []
    for row_index, latitude in enumerate(latitudes):
        if visually_even:
            row_count = int(equatorial_count)
        else:
            cos_lat = float(np.cos(np.deg2rad(latitude)))
            row_count = max(
                int(min_sites_per_row), int(round(equatorial_count * max(cos_lat, 0.2)))
            )
        if row_count <= 0:
            continue

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
            for begin, end in zip(split_points[:-1], split_points[1:]):
                if end - begin >= 2:
                    segments.append((lon_slice[begin:end], lat_slice[begin:end]))
        start_idx = end_idx
    return segments


def local_time_window_is_full(lt_min, lt_max):
    """Return whether a local-time window covers the full day."""
    return float(lt_min) <= 0.0 and float(lt_max) >= 24.0


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

    lat_min, lat_max = sorted(np.clip(np.asarray(lat_window, dtype=float), -90.0, 90.0))
    mask = (
        np.isfinite(lat_arr) & np.isfinite(lon_arr) & (lat_arr >= lat_min) & (lat_arr <= lat_max)
    )

    lt_min, lt_max = np.clip(np.asarray(local_time_window, dtype=float), 0.0, 24.0)
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
    lat_min, lat_max = sorted(np.clip(np.asarray(lat_window, dtype=float), -90.0, 90.0))
    lt_min, lt_max = np.clip(np.asarray(local_time_window, dtype=float), 0.0, 24.0)
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
    "geographic_local_time_mask",
    "local_time_window_extent",
    "local_time_window_is_full",
    "split_wrapped_curve",
    "wrap_longitudes",
]
