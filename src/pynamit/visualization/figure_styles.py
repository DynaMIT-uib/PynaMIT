"""Shared style tables and small helpers for PynaMIT figures."""

import numpy as np
import matplotlib.pyplot as plt

from pynamit.visualization.plot_helpers import symmetric_contour_levels_without_zero


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


def map_line_keys(value):
    """Return contour-line field keys for one UI value."""
    value = str(value)
    if value == "none":
        return []
    if value == "Phi_W":
        return ["Phi", "W"]
    return [value]


__all__ = ["FIELD_DIFF_KWARGS", "FIELD_PLOT_KWARGS", "INPUT_SUMMARY_KWARGS", "map_line_keys"]
