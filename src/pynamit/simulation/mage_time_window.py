"""Helpers for selecting time windows from MAGE HDF5 timestamps."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MageTimeWindow:
    """Resolved subset of one MAGE time axis."""

    indices: np.ndarray
    timestamps: tuple[dt.datetime, ...]
    relative_seconds: np.ndarray
    requested_start: dt.datetime | None
    requested_end: dt.datetime | None

    @property
    def start(self) -> dt.datetime:
        """Return the first selected timestamp."""
        return self.timestamps[0]

    @property
    def end(self) -> dt.datetime:
        """Return the last selected timestamp."""
        return self.timestamps[-1]


def _parse_mage_timestamp(value: object) -> dt.datetime:
    """Convert one raw MAGE HDF5 timestamp value to ``datetime``."""
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)
    return dt.datetime.fromisoformat(text)


def _normalize_window_bound(
    value: dt.datetime | dt.time | str | None, *, reference_date: dt.date
) -> dt.datetime | None:
    """Normalize one user-provided window bound against one reference date."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.time):
        return dt.datetime.combine(reference_date, value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        pass

    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return dt.datetime.combine(reference_date, dt.datetime.strptime(text, fmt).time())
        except ValueError:
            continue

    raise ValueError(
        f"Invalid MAGE time window bound {value!r}. Use HH:MM, HH:MM:SS, or ISO datetime."
    )


def select_mage_time_window(
    raw_timestamps: Iterable[object],
    *,
    start: dt.datetime | dt.time | str | None = None,
    end: dt.datetime | dt.time | str | None = None,
) -> MageTimeWindow:
    """Return the subset of MAGE timestamps that lies within one requested window."""
    timestamps = tuple(_parse_mage_timestamp(value) for value in raw_timestamps)
    if not timestamps:
        raise ValueError("MAGE time axis is empty.")

    requested_start = _normalize_window_bound(start, reference_date=timestamps[0].date())
    requested_end = _normalize_window_bound(end, reference_date=timestamps[0].date())

    if (
        requested_start is not None
        and requested_end is not None
        and requested_end < requested_start
    ):
        raise ValueError(
            f"Requested MAGE end time {requested_end.isoformat(sep=' ')} is before start time "
            f"{requested_start.isoformat(sep=' ')}."
        )

    mask = np.ones(len(timestamps), dtype=bool)
    if requested_start is not None:
        mask &= np.array([timestamp >= requested_start for timestamp in timestamps], dtype=bool)
    if requested_end is not None:
        mask &= np.array([timestamp <= requested_end for timestamp in timestamps], dtype=bool)

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        requested_range = (
            f"{requested_start.isoformat(sep=' ') if requested_start is not None else '-inf'} to "
            f"{requested_end.isoformat(sep=' ') if requested_end is not None else '+inf'}"
        )
        raise ValueError(
            "Requested MAGE time window "
            f"{requested_range} does not overlap available data "
            f"{timestamps[0].isoformat(sep=' ')} to {timestamps[-1].isoformat(sep=' ')}."
        )

    selected_timestamps = tuple(timestamps[index] for index in indices)
    t0 = selected_timestamps[0]
    relative_seconds = np.array(
        [(timestamp - t0).total_seconds() for timestamp in selected_timestamps], dtype=float
    )

    return MageTimeWindow(
        indices=indices,
        timestamps=selected_timestamps,
        relative_seconds=relative_seconds,
        requested_start=requested_start,
        requested_end=requested_end,
    )
