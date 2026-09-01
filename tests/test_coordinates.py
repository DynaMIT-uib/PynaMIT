"""Geographic and epoch coordinate conventions."""

import datetime as dt

import pytest

from pynamit.coordinates import datetime_to_utc_hours, decimal_year, decimal_year_to_datetime


def test_decimal_year_roundtrip_preserves_every_supported_kaiju_day():
    """Midnight roundoff must never select the preceding Geopack day."""
    for year in range(1965, 2026):
        value = dt.datetime(year, 1, 1)
        while value.year == year:
            epoch = decimal_year(value)
            assert decimal_year_to_datetime(epoch).date() == value.date()
            value += dt.timedelta(days=1)


def test_epoch_conversions_use_utc_and_retain_fractional_seconds():
    """UTC and offset datetimes represent the same physical instant."""
    utc = dt.datetime(2020, 7, 2, 6, 15, 20, 250000, tzinfo=dt.timezone.utc)
    offset = utc.astimezone(dt.timezone(dt.timedelta(hours=2)))
    assert datetime_to_utc_hours(offset) == pytest.approx(6 + 15 / 60 + 20.25 / 3600)
    assert decimal_year(offset) == decimal_year(utc)
    assert decimal_year(2020.5) == 2020.5
    restored = decimal_year_to_datetime(decimal_year(offset))
    assert abs((restored - utc.replace(tzinfo=None)).total_seconds()) < 1e-5
