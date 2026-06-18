"""Map panel drawing primitives."""

from __future__ import annotations

import cartopy.crs as ccrs

from pynamit.visualization.hemisphere import hemisphere_masks_for_latitude
from pynamit.visualization.plot_helpers import (
    contour_kwargs_for_display,
    set_contour_edges_to_face,
)


def _axes_from_group(group):
    return group.get("axes", []) if isinstance(group, dict) else group


def _hemisphere_from_group(group, index):
    if isinstance(group, dict):
        return group.get("hemisphere", "global")
    return "north" if index == 0 else "south"


def _is_polarplot_axis(axis):
    return axis.__class__.__name__ == "Polarplot"


def draw_field_comparison_artists(
    axes_groups,
    filled_key,
    overlay_keys,
    fields_dict,
    lat,
    lon,
    current_time,
    *,
    plot_kwargs,
    diff_kwargs,
    dipole_obj=None,
    hemisphere_min_abs_latitude=50.0,
):
    """Draw state, steady-state, and difference fields."""
    new_artists, main_mappable, diff_mappable = [], None, None
    filled_key = None if str(filled_key) == "none" else str(filled_key)
    overlay_keys = list(overlay_keys)

    if dipole_obj:
        magnetic_latitude, magnetic_longitude = dipole_obj.geo2mag(lat, lon)
        polar_x = magnetic_latitude
        polar_y = dipole_obj.mlon2mlt(magnetic_longitude, current_time)
    else:
        polar_x = lat
        polar_y = (lon + 180.0) % 360.0 / 15.0
    polar_north_mask, polar_south_mask = hemisphere_masks_for_latitude(
        polar_x, hemisphere_min_abs_latitude
    )

    if filled_key is not None:
        state_field = fields_dict[f"{filled_key}_state"]
        steady_field = fields_dict[f"{filled_key}_steady"]
        fill_kwargs = [plot_kwargs[filled_key], plot_kwargs[filled_key], diff_kwargs[filled_key]]
        fill_fields = [state_field, steady_field, state_field - steady_field]
    else:
        fill_kwargs, fill_fields = [], []

    overlay_specs = []
    for overlay_key in overlay_keys:
        state_field = fields_dict[f"{overlay_key}_state"]
        steady_field = fields_dict[f"{overlay_key}_steady"]
        overlay_specs.append(
            (
                [plot_kwargs[overlay_key], plot_kwargs[overlay_key], diff_kwargs[overlay_key]],
                [state_field, steady_field, state_field - steady_field],
            )
        )

    for group_index, group in enumerate(axes_groups):
        axes = _axes_from_group(group)
        hemisphere = _hemisphere_from_group(group, group_index)
        is_polar = bool(axes) and _is_polarplot_axis(axes[0])
        if hemisphere == "north":
            mask = polar_north_mask
        elif hemisphere == "south":
            mask = polar_south_mask
        else:
            mask = None

        for panel_index, axis in enumerate(axes):
            if filled_key is not None:
                display_kwargs = contour_kwargs_for_display(fill_kwargs[panel_index])
                if is_polar:
                    plot_args = (polar_x[mask], polar_y[mask], fill_fields[panel_index][mask])
                    transform_args = {}
                else:
                    plot_args = (lon, lat, fill_fields[panel_index])
                    transform_args = {"transform": ccrs.PlateCarree()}
                artist = axis.contourf(*plot_args, **transform_args, **display_kwargs)
                set_contour_edges_to_face(artist)
                new_artists.append(artist)
                if panel_index < 2:
                    main_mappable = artist
                if panel_index == 2:
                    diff_mappable = artist

            for overlay_kwargs, overlay_fields in overlay_specs:
                display_kwargs = contour_kwargs_for_display(overlay_kwargs[panel_index])
                if is_polar:
                    plot_args = (polar_x[mask], polar_y[mask], overlay_fields[panel_index][mask])
                    transform_args = {}
                else:
                    plot_args = (lon, lat, overlay_fields[panel_index])
                    transform_args = {"transform": ccrs.PlateCarree()}
                new_artists.append(axis.contour(*plot_args, **transform_args, **display_kwargs))

    return new_artists, main_mappable, diff_mappable


__all__ = ["draw_field_comparison_artists"]
