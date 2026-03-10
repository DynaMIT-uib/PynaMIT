"""Tests for MAGE input time-window selection."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from pynamit.simulation.mage_time_window import select_mage_time_window


def _build_raw_timestamps(n_steps: int = 360) -> np.ndarray:
    """Build one synthetic MAGE time axis with 10-second cadence."""
    start = dt.datetime(2011, 10, 24, 18, 0, 10)
    return np.array(
        [
            (start + dt.timedelta(seconds=10 * step)).isoformat().encode("utf-8")
            for step in range(n_steps)
        ],
        dtype="S19",
    )


def test_select_mage_time_window_supports_hhmm_bounds() -> None:
    raw_timestamps = _build_raw_timestamps()

    window = select_mage_time_window(raw_timestamps, start="18:20", end="18:45")

    assert window.start == dt.datetime(2011, 10, 24, 18, 20, 0)
    assert window.end == dt.datetime(2011, 10, 24, 18, 45, 0)
    assert int(window.indices[0]) == 119
    assert int(window.indices[-1]) == 269
    assert int(window.relative_seconds[0]) == 0
    assert int(window.relative_seconds[-1]) == 1500


def test_select_mage_time_window_supports_iso_start_and_open_end() -> None:
    raw_timestamps = _build_raw_timestamps()

    window = select_mage_time_window(raw_timestamps, start="2011-10-24T18:20:00")

    assert window.start == dt.datetime(2011, 10, 24, 18, 20, 0)
    assert window.end == dt.datetime(2011, 10, 24, 19, 0, 0)
    assert int(window.indices[0]) == 119
    assert int(window.indices[-1]) == 359


def test_select_mage_time_window_rejects_reversed_bounds() -> None:
    raw_timestamps = _build_raw_timestamps()

    with pytest.raises(ValueError, match="before start time"):
        select_mage_time_window(raw_timestamps, start="18:45", end="18:20")


def test_select_mage_time_window_rejects_non_overlapping_window() -> None:
    raw_timestamps = _build_raw_timestamps()

    with pytest.raises(ValueError, match="does not overlap available data"):
        select_mage_time_window(raw_timestamps, start="19:05")
