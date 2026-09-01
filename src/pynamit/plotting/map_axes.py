"""Shared presentation for global map axes and figure panels."""

import cartopy.crs as ccrs

from pynamit.plotting.map_coordinates import MapCoordinateContext


def style_global_axis(
    ax,
    *,
    coordinate_context=None,
    local_time_reference=None,
    draw_labels=True,
    draw_coastlines=True,
    set_global=True,
    left_labels=True,
    bottom_labels=True,
    coastline_color="0.45",
    coastline_linewidth=0.7,
    grid_color="0.72",
    grid_linewidth=0.7,
    grid_alpha=0.75,
    label_size=8,
):
    """Style a global Cartopy axis for PynaMIT map plots."""
    if set_global:
        ax.set_global()
    coordinates_are_geographic = (
        coordinate_context is None or coordinate_context.longitude_kind == "geographic"
    )
    if draw_coastlines and coordinates_are_geographic:
        ax.coastlines(color=coastline_color, linewidth=coastline_linewidth, zorder=2)
    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=draw_labels,
        linewidth=grid_linewidth,
        color=grid_color,
        alpha=grid_alpha,
        linestyle="--",
        zorder=1,
    )
    gridliner.top_labels = False
    gridliner.right_labels = False
    gridliner.left_labels = bool(draw_labels and left_labels)
    gridliner.bottom_labels = bool(draw_labels and bottom_labels)
    if coordinate_context is not None:
        coordinate_context.apply_grid_labels(gridliner)
    elif local_time_reference is not None:
        MapCoordinateContext.geographic(local_time_reference).apply_grid_labels(gridliner)
    gridliner.xlabel_style = {"size": label_size}
    gridliner.ylabel_style = {"size": label_size}
    return gridliner


__all__ = ["style_global_axis"]
