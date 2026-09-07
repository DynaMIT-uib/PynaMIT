"""Ground-field and magnetometer signals on requested time grids."""

import numpy as np

from pynamit.results.time_series import (
    centered_difference_at_times,
    resample_to_times,
    vector_magnitude,
)


def _component_selection(component):
    """Interpret a geographic component or its absolute value."""
    absolute = component.startswith("Abs")
    base = component[3:] if absolute else component
    if base not in {"North", "East", "Down", "Magnitude"} or (absolute and base == "Magnitude"):
        raise ValueError(f"Unsupported magnetic component: {component!r}")
    return base, absolute


def station_signal_at_times(
    measured,
    target_times,
    *,
    component="North",
    quantity="b",
    half_window_points=1,
    cadence_seconds=None,
):
    """Sample geographic magnetometer components in nT or nT/s.

    The DataFrame's North/East/Down columns are in nT. Components are
    resampled or differentiated before computing vector magnitude.
    """
    base, absolute = _component_selection(component)
    columns = ["North", "East", "Down"] if base == "Magnitude" else [base]
    values = measured[columns].to_numpy(dtype=float).T
    if quantity == "dbdt":
        sampled = centered_difference_at_times(
            measured.index,
            values,
            target_times,
            half_window_points=half_window_points,
            cadence_seconds=cadence_seconds,
        )
    elif quantity == "b":
        sampled = resample_to_times(measured.index, values, target_times)
    else:
        raise ValueError("quantity must be 'b' or 'dbdt'.")
    signal = vector_magnitude(sampled) if base == "Magnitude" else sampled[0]
    return np.abs(signal) if absolute else signal


def ground_signal_at_times(
    component,
    radial,
    tangential,
    source_times,
    target_times,
    *,
    quantity="b",
    half_window_points=1,
    cadence_seconds=None,
):
    """Sample geographic model ground fields in nT or nT/s.

    Inputs are in tesla, with tangential order (theta, phi) and time on
    the last axis. For B, the selected component/magnitude is evaluated
    at source times before interpolation. For dB/dt, differentiate
    components before taking magnitude: |dB/dt| is not d|B|/dt.
    """
    base, absolute = _component_selection(component)
    components = (-np.asarray(tangential[0]), np.asarray(tangential[1]), -np.asarray(radial))
    values = (
        np.stack(components)
        if base == "Magnitude"
        else components[("North", "East", "Down").index(base)]
    ) * 1e9
    if quantity == "dbdt":
        signal = centered_difference_at_times(
            source_times,
            values,
            target_times,
            half_window_points=half_window_points,
            cadence_seconds=cadence_seconds,
        )
        if base == "Magnitude":
            signal = vector_magnitude(signal)
        return np.abs(signal) if absolute else signal
    if quantity != "b":
        raise ValueError("quantity must be 'b' or 'dbdt'.")
    if base == "Magnitude":
        values = vector_magnitude(values)
    if absolute:
        values = np.abs(values)
    return resample_to_times(source_times, values, target_times)
