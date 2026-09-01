"""Contour levels, color scales, and Matplotlib contour presentation."""

import matplotlib.colors as mcolors
import numpy as np


def symmetric_contour_levels(first_abs_level, interval, levels_per_sign):
    """Return zero-free levels from a positive sequence."""
    first_abs_level = float(first_abs_level)
    interval = float(interval)
    integer_levels = int(levels_per_sign)
    if not np.isfinite(first_abs_level) or first_abs_level <= 0.0:
        raise ValueError("first_abs_level must be finite and positive.")
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("interval must be finite and positive.")
    if (
        isinstance(levels_per_sign, (bool, np.bool_))
        or integer_levels != levels_per_sign
        or integer_levels < 1
    ):
        raise ValueError("levels_per_sign must be an integer of at least one.")
    levels_per_sign = integer_levels
    positive = first_abs_level + interval * np.arange(levels_per_sign, dtype=float)
    return np.concatenate((-positive[::-1], positive))


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


def format_contour_interval(interval, units=None):
    """Format a contour interval for a compact label."""
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        text = str(interval)
        return f"{text} {units}" if units else text
    if not np.isfinite(interval):
        text = str(interval)
        return f"{text} {units}" if units else text
    abs_interval = abs(interval)
    if 1e-2 <= abs_interval < 1e4:
        text = f"{interval:g}"
    else:
        text = f"{interval:.2e}"
    return f"{text} {units}" if units else text


def draw_line_contour_legend(ax, overlay_keys, kwargs_source, title="Line contours"):
    """Draw a compact vertical legend for line-contour overlays."""
    ax.cla()
    ax.axis("off")
    overlay_keys = list(overlay_keys)
    if not overlay_keys:
        return
    y_pos = np.linspace(0.78, 0.22, len(overlay_keys)) if len(overlay_keys) > 1 else [0.5]
    for y, var_key in zip(y_pos, overlay_keys, strict=True):
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


def finite_values(data_arrays):
    """Return finite flattened values from one or more arrays."""
    flattened = []
    for values in data_arrays:
        array = np.asarray(values)
        if array.size > 0:
            flattened.append(array.reshape(-1))
    if not flattened:
        return np.array([])
    values = np.concatenate(flattened)
    return values[np.isfinite(values)]


def _color_scale_values(data_arrays, *, strictly_positive, scale_type, minimum_positive, label):
    """Return finite values eligible for percentile color limits."""
    finite = finite_values(data_arrays)
    if finite.size == 0:
        raise ValueError(f"No finite data available for '{label}' color scale.")
    if not strictly_positive:
        return finite
    if np.any(finite < -1e-9):
        raise ValueError(
            f"Data for '{label}' is marked strictly positive but contains negative values."
        )

    values = finite[finite >= 0.0]
    if scale_type == "log":
        values = values[values > float(minimum_positive)]
        if values.size == 0:
            raise ValueError(f"No data above {minimum_positive:g} for '{label}' log color scale.")
    if values.size == 0:
        raise ValueError(f"No valid data available for '{label}' color scale.")
    return values


def _color_normalization(vmin, vmax, *, strictly_positive, scale_type, minimum_positive):
    """Return nondegenerate limits and normalization."""
    if scale_type == "log":
        if not vmax > vmin:
            center = max(float(vmax), float(minimum_positive) * 10.0)
            vmin = max(center / np.sqrt(10.0), float(minimum_positive))
            vmax = center * np.sqrt(10.0)
        return vmin, vmax, mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

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
    return vmin, vmax, mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)


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

    percentile_values = _color_scale_values(
        data_arrays,
        strictly_positive=strictly_positive,
        scale_type=scale_type,
        minimum_positive=minimum_positive,
        label=label,
    )
    if strictly_positive:
        vmin = float(np.percentile(percentile_values, vmin_percentile))
        vmax = float(np.percentile(percentile_values, vmax_percentile))
        if scale_type == "linear":
            vmin = 0.0
    else:
        abs_max = float(np.percentile(np.abs(percentile_values), vmax_percentile))
        vmin, vmax = -abs_max, abs_max
    vmin, vmax, norm = _color_normalization(
        vmin,
        vmax,
        strictly_positive=strictly_positive,
        scale_type=scale_type,
        minimum_positive=minimum_positive,
    )

    return {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap or ("viridis" if strictly_positive else "bwr"),
        "norm": norm,
        "scale_type": scale_type,
        "strictly_positive": bool(strictly_positive),
    }


def percentile_contour_levels(
    data_arrays, fallback_levels, *, percentile=99.8, strictly_positive=False
):
    """Build contour levels from a robust data percentile."""
    finite = finite_values(data_arrays)
    if finite.size == 0:
        return fallback_levels
    n_levels = max(len(fallback_levels), 3)
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    if strictly_positive:
        finite = finite[finite >= 0.0]
        if finite.size == 0:
            return fallback_levels
        vmax = float(np.percentile(finite, percentile))
        if not np.isfinite(vmax) or vmax <= 0.0:
            return fallback_levels
        return np.linspace(0.0, vmax, n_levels)
    vmax = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return fallback_levels
    return np.linspace(-vmax, vmax, n_levels)


def set_contour_edges_to_face(contour):
    """Avoid hairline gaps in filled contour artists."""
    contour.set_edgecolor("face")
    return contour


__all__ = [
    "build_percentile_color_scale",
    "contour_kwargs_for_display",
    "draw_line_contour_legend",
    "finite_values",
    "format_contour_interval",
    "get_ticks_from_levels",
    "percentile_contour_levels",
    "set_contour_edges_to_face",
    "symmetric_contour_levels",
    "symmetric_contour_levels_without_zero",
]
