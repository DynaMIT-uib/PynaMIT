"""Shared style tables and small helpers for PynaMIT figures."""

import matplotlib.pyplot as plt
import numpy as np

from pynamit.visualization.plot_helpers import (
    symmetric_contour_levels,
    symmetric_contour_levels_without_zero,
)

FIELD_PLOT_KWARGS = {
    "Br": {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-8.5, 8.5, 18) * 1e-8,
        "extend": "both",
        "symbol": "$B_r$",
        "units": "T",
    },
    "jr": {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-8.5, 8.5, 18) * 1e-7,
        "extend": "both",
        "symbol": "$j_r$",
        "units": "A/m$^2$",
    },
    "joule": {
        "cmap": plt.cm.inferno,
        "levels": np.linspace(0.0, 8.5, 18) * 1e-3,
        "extend": "max",
        "symbol": "Joule heating",
        "units": "W/m$^2$",
    },
    "Jeq": {
        "colors": "black",
        "levels": np.linspace(-4, 4, 50) * 1e5,
        "symbol": "$J_{eq}$",
        "units": "A",
    },
    "Phi": {
        "colors": "black",
        "levels": symmetric_contour_levels_without_zero(168.0, 8.0),
        "symbol": "$\\Phi$",
        "units": "kV",
    },
    "W": {
        "colors": "green",
        "levels": symmetric_contour_levels_without_zero(40.0, 8.0),
        "symbol": "$W$",
        "units": "kV",
    },
}

FIELD_DIFF_KWARGS = {
    "Br": {
        **FIELD_PLOT_KWARGS["Br"],
        "levels": FIELD_PLOT_KWARGS["Br"]["levels"] * 0.5,
        "symbol": "$\\Delta B_r$",
    },
    "jr": {
        **FIELD_PLOT_KWARGS["jr"],
        "levels": FIELD_PLOT_KWARGS["jr"]["levels"] * 0.5,
        "symbol": "$\\Delta j_r$",
    },
    "joule": {
        **FIELD_PLOT_KWARGS["joule"],
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-4.25, 4.25, 18) * 1e-3,
        "symbol": "$\\Delta$ Joule heating",
        "extend": "both",
    },
    "Jeq": {
        **FIELD_PLOT_KWARGS["Jeq"],
        "levels": FIELD_PLOT_KWARGS["Jeq"]["levels"] * 0.5,
        "symbol": "$\\Delta J_{eq}$",
    },
    "Phi": {
        **FIELD_PLOT_KWARGS["Phi"],
        "levels": symmetric_contour_levels_without_zero(36.0, 4.0),
        "symbol": "$\\Delta \\Phi$",
    },
    "W": {
        **FIELD_PLOT_KWARGS["W"],
        "levels": symmetric_contour_levels_without_zero(40.0, 4.0),
        "symbol": "$\\Delta W$",
    },
}

INPUT_SUMMARY_KWARGS = {
    "jr": {
        "cmap": plt.cm.bwr,
        "levels": symmetric_contour_levels_without_zero(0.9, 0.1),
        "extend": "both",
        "symbol": r"$j_r$",
        "units": r"$\mu$A/m$^2$",
        "scale": 1e6,
    },
    "Br": {
        "cmap": plt.cm.bwr,
        "levels": symmetric_contour_levels_without_zero(16.0, 2.0),
        "extend": "both",
        "symbol": r"$B_r(r=R_M)$",
        "units": "nT",
        "scale": 1e9,
    },
    "conductance": {
        "cmap": plt.cm.viridis,
        "levels": np.linspace(0.0, 40.0, 21),
        "extend": "max",
        "symbol": r"$\Sigma$",
        "units": "S",
        "scale": 1.0,
    },
    "wind": {
        "cmap": plt.cm.coolwarm,
        "levels": np.linspace(-500.0, 500.0, 21),
        "extend": "both",
        "symbol": r"$u$",
        "units": "m/s",
        "scale": 1.0,
    },
}

MANUAL_COLOR_CONTROL_UNITS = {"Br": ("nT", 1e9), "jr": ("µA/m²", 1e6), "joule": ("mW/m²", 1e3)}


def map_line_keys(value):
    """Return contour-line field keys for one UI value."""
    value = str(value)
    if value == "none":
        return []
    if value == "Phi_W":
        return ["Phi", "W"]
    return [value]


def manual_color_limits(field_key):
    """Return the preset manual color limits for one filled field."""
    levels = np.asarray(FIELD_PLOT_KWARGS[field_key]["levels"], dtype=float)
    return float(levels[0]), float(levels[-1])


def manual_color_control_units(field_key):
    """Return human-readable units and the SI-to-display scale."""
    return MANUAL_COLOR_CONTROL_UNITS[field_key]


def manual_color_display_value(field_key, value):
    """Convert an SI color limit to a compact GUI value."""
    _, scale = manual_color_control_units(field_key)
    return float(f"{float(value) * scale:.12g}")


def manual_color_levels(field_key, minimum, maximum):
    """Return manual color levels with the preset band count."""
    preset = np.asarray(FIELD_PLOT_KWARGS[field_key]["levels"], dtype=float)
    return np.linspace(float(minimum), float(maximum), preset.size)


def manual_line_parameters(field_key):
    """Return the three parameters defining a line preset."""
    levels = np.asarray(FIELD_PLOT_KWARGS[field_key]["levels"], dtype=float)
    positive = np.sort(levels[levels > 0.0])
    if positive.size == 0:
        raise ValueError(f"Line field {field_key!r} has no positive preset levels.")
    interval = positive[1] - positive[0] if positive.size > 1 else positive[0]
    return float(positive[0]), float(interval), int(positive.size)


def manual_line_levels(first_abs_level, interval, levels_per_sign):
    """Return a user-configured zero-free symmetric line sequence."""
    return symmetric_contour_levels(first_abs_level, interval, levels_per_sign)


__all__ = [
    "FIELD_DIFF_KWARGS",
    "FIELD_PLOT_KWARGS",
    "INPUT_SUMMARY_KWARGS",
    "manual_color_control_units",
    "manual_color_display_value",
    "manual_color_levels",
    "manual_color_limits",
    "manual_line_levels",
    "manual_line_parameters",
    "map_line_keys",
]
