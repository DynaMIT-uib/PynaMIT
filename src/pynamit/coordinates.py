"""Coordinate conversion helpers."""

import datetime as dt

import numpy as np

DEFAULT_LOCAL_TIME_GRID_HOURS = (3, 9, 15, 21)


def wrap_longitude_180(lon):
    """Wrap longitude values to the ``[-180, 180)`` interval."""
    lon_array = np.asarray(lon, dtype=float)
    wrapped = ((lon_array + 180.0) % 360.0) - 180.0
    if np.isscalar(lon):
        return float(wrapped)
    return wrapped


def datetime_to_utc_hours(time_value):
    """Return fractional UTC hours from a datetime-like value."""
    if hasattr(time_value, "to_pydatetime"):
        time_value = time_value.to_pydatetime()
    if isinstance(time_value, dt.datetime):
        if time_value.tzinfo is not None:
            time_value = time_value.astimezone(dt.timezone.utc)
        time_value = time_value.time()
    if not isinstance(time_value, dt.time):
        raise TypeError("time_value must be a datetime, time, or Timestamp.")
    return (
        time_value.hour
        + time_value.minute / 60.0
        + time_value.second / 3600.0
        + time_value.microsecond / 3.6e9
    )


def decimal_year_to_datetime(epoch):
    """Convert decimal year while preserving day boundaries."""
    epoch = float(epoch)
    year = int(np.floor(epoch))
    year_start = dt.datetime(year, 1, 1)
    next_year_start = dt.datetime(year + 1, 1, 1)
    year_seconds = (next_year_start - year_start).total_seconds()
    elapsed_seconds = (epoch - year) * year_seconds

    # At decimal-year magnitudes, a datetime round trip can lose a few
    # microseconds. Preserve exact day boundaries when the discrepancy
    # is only floating-point roundoff.
    day_seconds = 86400.0
    nearest_day = round(elapsed_seconds / day_seconds) * day_seconds
    roundoff_tolerance = max(1e-6, 4.0 * abs(np.spacing(epoch)) * year_seconds)
    if abs(elapsed_seconds - nearest_day) <= roundoff_tolerance:
        elapsed_seconds = nearest_day
    return year_start + dt.timedelta(seconds=elapsed_seconds)


def local_noon_longitude(reference_time):
    """Return the mean-solar local-noon geographic longitude."""
    utc_hours = datetime_to_utc_hours(reference_time)
    return wrap_longitude_180((12.0 - utc_hours) * 15.0)


def longitude_to_local_time_hours(lon, reference_time):
    """Convert geographic longitude to local-time hours."""
    utc_hours = datetime_to_utc_hours(reference_time)
    return (utc_hours + np.asarray(lon, dtype=float) / 15.0) % 24.0


def longitude_to_local_time_from_noon_longitude(lon, noon_longitude, *, wrap=True):
    """Convert longitude to local-time hours from a noon meridian.

    Parameters
    ----------
    lon : array-like
        Longitudes in the plotted coordinate system.
    noon_longitude : float
        Longitude of local noon in the same coordinate system.
    wrap : bool, optional
        Wrap output to ``[0, 24)``. Set false when a continuous
        unwrapped coordinate is preferable for polar contour plots.
    """
    local_time = 12.0 + (np.asarray(lon, dtype=float) - float(noon_longitude)) / 15.0
    if wrap:
        local_time = local_time % 24.0
    if np.isscalar(lon):
        return float(local_time)
    return local_time


def local_time_hours_to_longitude(local_time_hours, reference_time):
    """Convert local-time hours to geographic longitude."""
    utc_hours = datetime_to_utc_hours(reference_time)
    return wrap_longitude_180((np.asarray(local_time_hours, dtype=float) - utc_hours) * 15.0)


def local_time_longitude_to_geographic(lon, *, noon_longitude, local_noon_longitude=0.0):
    """Convert local-time-like longitude to geographic longitude.

    Parameters
    ----------
    lon : array-like
        Source longitudes where ``local_noon_longitude`` is noon.
    noon_longitude : float
        Geographic longitude of local noon.
    local_noon_longitude : float, optional
        Source-grid longitude corresponding to local noon.
    """
    return wrap_longitude_180(
        np.asarray(lon, dtype=float) - float(local_noon_longitude) + float(noon_longitude)
    )


__all__ = [
    "DEFAULT_LOCAL_TIME_GRID_HOURS",
    "decimal_year_to_datetime",
    "datetime_to_utc_hours",
    "local_noon_longitude",
    "local_time_hours_to_longitude",
    "local_time_longitude_to_geographic",
    "longitude_to_local_time_from_noon_longitude",
    "longitude_to_local_time_hours",
    "wrap_longitude_180",
]
