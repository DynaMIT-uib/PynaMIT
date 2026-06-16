"""Local-time longitude helpers for map visualization."""

import numpy as np

from pynamit.coordinates import (
    DEFAULT_LOCAL_TIME_GRID_HOURS,
    datetime_to_utc_hours,
    local_noon_longitude,
    local_time_hours_to_longitude,
    local_time_longitude_to_geographic,
    longitude_to_local_time_from_noon_longitude,
    longitude_to_local_time_hours,
    wrap_longitude_180,
)


def local_time_grid_longitudes(reference_time, hours=DEFAULT_LOCAL_TIME_GRID_HOURS):
    """Return geographic longitudes for selected local-time ticks."""
    return local_time_hours_to_longitude(np.asarray(hours, dtype=float), reference_time)


def format_local_time_longitude_label(lon, pos=None, *, reference_time):
    """Format a longitude tick as a local-time label."""
    del pos
    hour = int(np.round(longitude_to_local_time_hours(lon, reference_time))) % 24
    return f"{hour} LT"


def make_local_time_longitude_formatter(reference_time):
    """Create a Matplotlib formatter for local-time longitude ticks."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(
        lambda lon, pos: format_local_time_longitude_label(lon, pos, reference_time=reference_time)
    )


def apply_local_time_grid_labels(
    gridliner, *, reference_time, hours=DEFAULT_LOCAL_TIME_GRID_HOURS
):
    """Apply local-time longitudes and labels to a Cartopy gridliner."""
    from matplotlib.ticker import FixedLocator

    gridliner.xlocator = FixedLocator(local_time_grid_longitudes(reference_time, hours))
    gridliner.xformatter = make_local_time_longitude_formatter(reference_time)
    return gridliner


__all__ = [
    "DEFAULT_LOCAL_TIME_GRID_HOURS",
    "apply_local_time_grid_labels",
    "datetime_to_utc_hours",
    "format_local_time_longitude_label",
    "local_noon_longitude",
    "local_time_grid_longitudes",
    "local_time_hours_to_longitude",
    "local_time_longitude_to_geographic",
    "longitude_to_local_time_from_noon_longitude",
    "longitude_to_local_time_hours",
    "make_local_time_longitude_formatter",
    "wrap_longitude_180",
]
