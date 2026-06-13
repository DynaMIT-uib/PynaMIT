"""Small plotting helpers shared by visualization workflows."""

import numpy as np
import cartopy.crs as ccrs

from pynamit.visualization.local_time import apply_local_time_grid_labels


def symmetric_contour_levels_without_zero(max_abs, interval):
    """Return symmetric contour levels whose centers avoid zero."""
    max_abs = float(max_abs)
    interval = float(interval)
    edge_level = max_abs - 0.5 * interval
    return np.arange(-edge_level, edge_level + 0.5 * interval, interval)


def get_ticks_from_levels(plot_kwargs):
    """Return colorbar ticks centered between contour levels."""
    levels = plot_kwargs.get("levels")
    if levels is not None and len(levels) > 1:
        return (np.asarray(levels[:-1]) + np.asarray(levels[1:])) / 2
    return None


def format_contour_interval(interval):
    """Format a contour interval for a compact label."""
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        return str(interval)
    if not np.isfinite(interval):
        return str(interval)
    abs_interval = abs(interval)
    if 1e-2 <= abs_interval < 1e4:
        return f"{interval:g}"
    return f"{interval:.2e}"


def contour_kwargs_for_display(plot_kwargs):
    """Drop metadata keys before forwarding kwargs to Matplotlib."""
    return {
        key: value
        for key, value in plot_kwargs.items()
        if key not in {"symbol", "units", "scale"}
    }


def set_contour_edges_to_face(contour):
    """Avoid hairline gaps in filled contour artists."""
    try:
        contour.set_edgecolor("face")
        return contour
    except Exception:
        pass
    for collection in getattr(contour, "collections", []):
        try:
            collection.set_edgecolor("face")
        except Exception:
            pass
    return contour


def stabilize_polarplot(pax):
    """Set stable aspect/anchor properties on a polplot axis."""
    try:
        pax.ax.set_aspect("equal", adjustable="box")
        pax.ax.set_anchor("C")
    except Exception:
        pass
    return pax


def remove_artists(artist_list):
    """Remove artists in-place and empty the list."""
    for artist in artist_list:
        if artist:
            artist.remove()
    artist_list.clear()


def style_global_axis(
    ax,
    *,
    coordinate_context=None,
    local_time_reference=None,
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
    ax.set_global()
    ax.coastlines(
        color=coastline_color,
        linewidth=coastline_linewidth,
        zorder=2,
    )
    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=grid_linewidth,
        color=grid_color,
        alpha=grid_alpha,
        linestyle="--",
        zorder=1,
    )
    gridliner.top_labels = False
    gridliner.right_labels = False
    gridliner.left_labels = bool(left_labels)
    gridliner.bottom_labels = bool(bottom_labels)
    if coordinate_context is not None:
        coordinate_context.apply_grid_labels(gridliner)
    elif local_time_reference is not None:
        apply_local_time_grid_labels(
            gridliner,
            reference_time=local_time_reference,
        )
    try:
        gridliner.xlabel_style = {"size": label_size}
        gridliner.ylabel_style = {"size": label_size}
    except Exception:
        pass
    return gridliner


def style_global_input_axis(
    ax,
    *,
    coordinate_context=None,
    local_time_reference=None,
    left_labels=True,
    bottom_labels=True,
):
    """Style a global axis for input-driver comparison plots."""
    return style_global_axis(
        ax,
        coordinate_context=coordinate_context,
        local_time_reference=local_time_reference,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
        coastline_color="0.45",
        coastline_linewidth=0.7,
        grid_color="0.72",
        grid_linewidth=0.7,
        grid_alpha=0.75,
        label_size=8,
    )


def style_global_comparison_axis(
    ax,
    *,
    coordinate_context=None,
    local_time_reference=None,
    left_labels=True,
    bottom_labels=True,
):
    """Style a global axis for state-vs-baseline comparisons."""
    return style_global_axis(
        ax,
        coordinate_context=coordinate_context,
        local_time_reference=local_time_reference,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
        coastline_color="black",
        coastline_linewidth=1.0,
        grid_color="gray",
        grid_linewidth=1.0,
        grid_alpha=0.5,
        label_size=8,
    )


def add_panel_label(ax, label):
    """Add a compact panel label in the upper-left axis corner."""
    return ax.text(
        0.015,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.0,
        },
        zorder=10,
    )


__all__ = [
    "add_panel_label",
    "contour_kwargs_for_display",
    "format_contour_interval",
    "get_ticks_from_levels",
    "remove_artists",
    "set_contour_edges_to_face",
    "stabilize_polarplot",
    "style_global_axis",
    "style_global_comparison_axis",
    "style_global_input_axis",
    "symmetric_contour_levels_without_zero",
]
