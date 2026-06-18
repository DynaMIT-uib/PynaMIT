"""Small plotting helpers shared by visualization workflows."""

from contextlib import contextmanager

import numpy as np
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

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


def draw_line_contour_legend(ax, overlay_keys, kwargs_source, title="Line contours"):
    """Draw a compact vertical legend for line-contour overlays."""
    ax.cla()
    ax.axis("off")
    overlay_keys = list(overlay_keys)
    if not overlay_keys:
        return
    y_pos = np.linspace(0.78, 0.22, len(overlay_keys)) if len(overlay_keys) > 1 else [0.5]
    for y, var_key in zip(y_pos, overlay_keys):
        kwargs = kwargs_source[var_key]
        interval = kwargs["levels"][1] - kwargs["levels"][0]
        label = (
            f"{kwargs.get('symbol', '')}, interval: "
            f"{format_contour_interval(interval)} {kwargs.get('units', '')}"
        )
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            rotation="vertical",
            color=kwargs.get("colors", "black"),
            fontsize=12,
        )
    if title:
        ax.set_title(title, fontsize=9, pad=6)


def contour_kwargs_for_display(plot_kwargs):
    """Drop metadata keys before forwarding kwargs to Matplotlib."""
    return {
        key: value for key, value in plot_kwargs.items() if key not in {"symbol", "units", "scale"}
    }


def _finite_concatenated_values(data_arrays):
    flattened = []
    for values in data_arrays:
        array = np.asarray(values)
        if array.size > 0:
            flattened.append(array.reshape(-1))
    if not flattened:
        return np.array([])
    values = np.concatenate(flattened)
    return values[np.isfinite(values)]


def build_percentile_color_scale(
    data_arrays,
    *,
    strictly_positive=False,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
    scale_type="linear",
    cmap=None,
    minimum_positive=1e-12,
    label="data",
):
    """Build a Matplotlib color scale from finite data percentiles."""
    if scale_type not in {"linear", "log"}:
        raise ValueError("scale_type must be 'linear' or 'log'.")
    if not (0.0 <= float(vmin_percentile) <= 100.0):
        raise ValueError("vmin_percentile must be between 0 and 100.")
    if not (0.0 <= float(vmax_percentile) <= 100.0):
        raise ValueError("vmax_percentile must be between 0 and 100.")
    if float(vmin_percentile) > float(vmax_percentile):
        raise ValueError("vmin_percentile cannot exceed vmax_percentile.")
    if scale_type == "log" and not strictly_positive:
        raise ValueError("Log color scales require strictly_positive=True.")

    finite_values = _finite_concatenated_values(data_arrays)
    if finite_values.size == 0:
        raise ValueError(f"No finite data available for '{label}' color scale.")

    if strictly_positive:
        if np.any(finite_values < -1e-9):
            raise ValueError(
                f"Data for '{label}' is marked strictly positive but contains negative values."
            )
        percentile_values = finite_values[finite_values >= 0.0]
        if scale_type == "log":
            percentile_values = percentile_values[percentile_values > float(minimum_positive)]
            if percentile_values.size == 0:
                raise ValueError(
                    f"No data above {minimum_positive:g} for '{label}' log color scale."
                )
    else:
        percentile_values = finite_values

    if percentile_values.size == 0:
        raise ValueError(f"No valid data available for '{label}' color scale.")

    if strictly_positive:
        vmin = float(np.percentile(percentile_values, vmin_percentile))
        vmax = float(np.percentile(percentile_values, vmax_percentile))
        if scale_type == "linear":
            vmin = 0.0
    else:
        abs_max = float(np.percentile(np.abs(percentile_values), vmax_percentile))
        vmin, vmax = -abs_max, abs_max

    if scale_type == "log":
        if not vmax > vmin:
            center = max(float(vmax), float(minimum_positive) * 10.0)
            vmin = max(center / np.sqrt(10.0), float(minimum_positive))
            vmax = center * np.sqrt(10.0)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        if abs(vmax - vmin) < 1e-12:
            epsilon = abs(vmax) * 0.05 if abs(vmax) > 1e-9 else 0.05
            if vmin == vmax == 0.0:
                vmax = epsilon
                if not strictly_positive:
                    vmin = -epsilon
            else:
                vmin -= epsilon
                vmax += epsilon
            if strictly_positive and vmin < 0.0:
                vmin = 0.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    return {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap or ("viridis" if strictly_positive else "bwr"),
        "norm": norm,
        "scale_type": scale_type,
        "strictly_positive": bool(strictly_positive),
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


@contextmanager
def suppress_empty_contour_warnings():
    """Suppress Matplotlib's no-contour-levels warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No contour levels were found within the data range."
        )
        yield


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
    if draw_coastlines:
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
        apply_local_time_grid_labels(gridliner, reference_time=local_time_reference)
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
    draw_labels=True,
    draw_coastlines=True,
    set_global=True,
    left_labels=True,
    bottom_labels=True,
):
    """Style a global axis for input-driver comparison plots."""
    return style_global_axis(
        ax,
        coordinate_context=coordinate_context,
        local_time_reference=local_time_reference,
        draw_labels=draw_labels,
        draw_coastlines=draw_coastlines,
        set_global=set_global,
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
    draw_labels=True,
    draw_coastlines=True,
    set_global=True,
    left_labels=True,
    bottom_labels=True,
):
    """Style a global axis for state-vs-baseline comparisons."""
    return style_global_axis(
        ax,
        coordinate_context=coordinate_context,
        local_time_reference=local_time_reference,
        draw_labels=draw_labels,
        draw_coastlines=draw_coastlines,
        set_global=set_global,
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
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        zorder=10,
    )


__all__ = [
    "add_panel_label",
    "build_percentile_color_scale",
    "contour_kwargs_for_display",
    "format_contour_interval",
    "get_ticks_from_levels",
    "remove_artists",
    "set_contour_edges_to_face",
    "stabilize_polarplot",
    "style_global_axis",
    "style_global_comparison_axis",
    "style_global_input_axis",
    "suppress_empty_contour_warnings",
    "symmetric_contour_levels_without_zero",
]
