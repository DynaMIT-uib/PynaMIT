"""Hemisphere helpers for polar visualization."""

from __future__ import annotations

import numpy as np

from pynamit.visualization.plot_helpers import stabilize_polarplot

DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE = 50.0


def coerce_hemisphere_min_abs_latitude(
    value, *, default=DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE, max_abs_latitude=89.9
):
    """Return a finite polar latitude cutoff."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        value = float(default)
    return float(np.clip(value, 0.0, float(max_abs_latitude)))


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
    return stabilize_polarplot(
        polarplot_cls(ax, minlat=coerce_hemisphere_min_abs_latitude(min_abs_latitude))
    )


__all__ = [
    "DEFAULT_HEMISPHERE_MIN_ABS_LATITUDE",
    "coerce_hemisphere_min_abs_latitude",
    "hemisphere_masks_for_latitude",
    "make_hemisphere_polarplot",
]
