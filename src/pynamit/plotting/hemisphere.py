"""Hemisphere helpers for polar plots."""

from __future__ import annotations

import numpy as np

DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE = 50.0


def coerce_hemisphere_min_abs_latitude(value, *, max_abs_latitude=89.9):
    """Return a validated polar latitude cutoff."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= max_abs_latitude:
        raise ValueError(
            f"hemisphere_min_abs_latitude must be between 0 and {max_abs_latitude} degrees."
        )
    return value


def hemisphere_masks_for_latitude(latitudes, min_abs_latitude=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE):
    """Return north/south masks outside a low-latitude band."""
    cutoff = coerce_hemisphere_min_abs_latitude(min_abs_latitude)
    latitudes = np.asarray(latitudes, dtype=float)
    return latitudes > cutoff, latitudes < -cutoff


def make_hemisphere_polarplot(
    ax, min_abs_latitude=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE, *, polarplot_cls=None
):
    """Create a stabilized polar plot with a shared cutoff."""
    if polarplot_cls is None:
        from polplot import Polarplot

        polarplot_cls = Polarplot
    polarplot = polarplot_cls(ax, minlat=coerce_hemisphere_min_abs_latitude(min_abs_latitude))
    polarplot.ax.set_aspect("equal", adjustable="box")
    polarplot.ax.set_anchor("C")
    return polarplot


__all__ = [
    "DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE",
    "coerce_hemisphere_min_abs_latitude",
    "hemisphere_masks_for_latitude",
    "make_hemisphere_polarplot",
]
