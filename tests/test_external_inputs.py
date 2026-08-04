"""Tests for external input handling with fallback datasets."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import pynamit.external_inputs as external_inputs_module

from pynamit.external_inputs import (
    _expand_time_series,
    _load_fallback,
    _select_fallback_entry,
    get_conductance_inputs,
    get_input_source,
    get_jr_inputs,
    get_wind_inputs,
    save_fallback_dataset,
    set_input_source,
)


def _utc_now():
    """Return a timezone-aware UTC datetime for input-source calls."""
    return datetime.datetime.now(datetime.UTC)


def test_native_conductance_uses_geographic_coordinates(monkeypatch):
    """Hardy receives GEO positions and performs its own conversion."""
    captured = {}

    class FakeConductance:
        @staticmethod
        def hardy_EUV(lon, lat, kp, date, *, starlight, dipole):
            captured.update(
                lon=np.asarray(lon).copy(),
                lat=np.asarray(lat).copy(),
                kp=kp,
                date=date,
                starlight=starlight,
                dipole=dipole,
            )
            values = np.ones(np.asarray(lat).shape)
            return values, 2.0 * values

    monkeypatch.setattr(external_inputs_module, "get_input_source", lambda: "native")
    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: FakeConductance,
    )

    date = datetime.datetime(2001, 5, 12, 21, 45)
    geo_lat = np.array([20.0, 60.0])
    geo_lon = np.array([-30.0, 80.0])
    hall, pedersen, out_lat, out_lon = get_conductance_inputs(
        date,
        geo_lat,
        geo_lon,
        None,
    )

    assert captured["dipole"] is False
    assert captured["kp"] == 5
    np.testing.assert_allclose(captured["lat"], geo_lat)
    np.testing.assert_allclose(captured["lon"], geo_lon)
    np.testing.assert_allclose(hall, 1.0)
    np.testing.assert_allclose(pedersen, 2.0)
    np.testing.assert_allclose(out_lat, geo_lat)
    np.testing.assert_allclose(out_lon, geo_lon)


def test_native_jr_converts_geo_to_modified_apex_and_pyamps_mlt(monkeypatch):
    """AMPS receives modified-apex latitude and its own MLT convention."""
    captured = {}

    class FakeApex:
        def __init__(self, *, date, refh):
            captured["apex_date"] = date
            captured["apex_refh"] = refh

        def geo2apex(self, lat, lon, height):
            captured["geo"] = (
                np.asarray(lat).copy(),
                np.asarray(lon).copy(),
                height,
            )
            return np.asarray(lat) + 5.0, np.asarray(lon) + 10.0

    class FakeAMPS:
        def __init__(self, *args, **kwargs):
            captured["amps_init"] = (args, kwargs)

        def get_upward_current(self, *, mlat, mlt):
            captured["amps_query"] = (
                np.asarray(mlat).copy(),
                np.asarray(mlt).copy(),
            )
            return np.ones(np.asarray(mlat).shape)

    def fake_mlon_to_mlt(mlon, date, epoch):
        captured["mlt_conversion"] = (
            np.asarray(mlon).copy(),
            date,
            epoch,
        )
        return np.asarray(mlon) / 15.0

    fake_pyamps = SimpleNamespace(
        __file__="/tmp/pyamps/__init__.py",
        AMPS=FakeAMPS,
        mlon_to_mlt=fake_mlon_to_mlt,
    )

    monkeypatch.setattr(external_inputs_module, "get_input_source", lambda: "native")
    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: fake_pyamps,
    )
    monkeypatch.setattr(external_inputs_module.apexpy, "Apex", FakeApex)

    date = datetime.datetime(2001, 5, 12, 21, 45)
    geo_lat = np.array([-70.0, 20.0])
    geo_lon = np.array([-30.0, 80.0])
    jr, out_lat, out_lon = get_jr_inputs(date, geo_lat, geo_lon, None)

    expected_mlat = geo_lat + 5.0
    expected_mlon = geo_lon + 10.0
    np.testing.assert_allclose(captured["geo"][0], geo_lat)
    np.testing.assert_allclose(captured["geo"][1], geo_lon)
    assert captured["geo"][2] == 110.0
    np.testing.assert_allclose(captured["mlt_conversion"][0], expected_mlon)
    assert captured["mlt_conversion"][1] == date
    assert captured["mlt_conversion"][2] == external_inputs_module.decimal_year(date)
    np.testing.assert_allclose(captured["amps_query"][0], expected_mlat)
    np.testing.assert_allclose(captured["amps_query"][1], expected_mlon / 15.0)
    np.testing.assert_allclose(jr, [1e-6, 0.0])
    np.testing.assert_allclose(out_lat, geo_lat)
    np.testing.assert_allclose(out_lon, geo_lon)


@pytest.fixture
def force_fallback():
    """Fixture to force the use of fallback external inputs."""
    previous = get_input_source()
    set_input_source("fallback")
    yield
    set_input_source(previous)


def test_fallback_conductance(force_fallback):
    """Test that cond. inputs are correctly loaded from fallback."""
    fallback = _load_fallback()
    key, entry = next(iter(fallback["conductance"].items()))
    lat = entry["lat"]
    lon = entry["lon"]
    time = None
    hall, pedersen, out_lat, out_lon = get_conductance_inputs(_utc_now(), lat, lon, time)
    assert hall.shape == (entry["hall"].size,)
    assert pedersen.shape == (entry["pedersen"].size,)
    np.testing.assert_allclose(out_lat, entry["lat"])
    np.testing.assert_allclose(out_lon, entry["lon"])


def test_bundled_fallback_loads_return_independent_arrays():
    """Cached parsing must not expose shared mutable input arrays."""
    first = _load_fallback()
    second = _load_fallback()

    first["wind"]["lat"][0] += 1.0

    assert not np.shares_memory(first["wind"]["lat"], second["wind"]["lat"])
    assert first["wind"]["lat"][0] != second["wind"]["lat"][0]


def test_fallback_grid_selection_uses_grid_hash(monkeypatch):
    """Fallback grid matching uses coordinate hashes."""
    lat = np.array([60.0, 61.0, 62.0])
    lon = np.array([10.0, 11.0, 12.0])
    entry = {"lat": lat + 1e-10, "lon": lon - 1e-10}

    def fail_allclose(*args, **kwargs):
        raise AssertionError("Fallback grid matching should use coordinate hashes")

    monkeypatch.setattr("pynamit.external_inputs.np.allclose", fail_allclose)

    assert _select_fallback_entry({"grid": entry}, lat, lon, "test") is entry


def test_fallback_multi_time_scaling(force_fallback):
    """Test that time-dependent scaling is applied correctly."""
    time = np.array([0.0, 10.0, 20.0])
    fallback = _load_fallback()
    key, entry = next(iter(fallback["conductance"].items()))
    hall, pedersen, *_ = get_conductance_inputs(_utc_now(), entry["lat"], entry["lon"], time)
    assert hall.shape == (time.size, entry["hall"].size)
    assert pedersen.shape == (time.size, entry["pedersen"].size)


def test_fallback_currents(force_fallback):
    """Test that jr inputs are correctly loaded from fallback."""
    fallback = _load_fallback()
    key, entry = next(iter(fallback["jr"].items()))
    jr, lat, lon = get_jr_inputs(_utc_now(), entry["lat"], entry["lon"], None)
    np.testing.assert_allclose(lat, entry["lat"])
    np.testing.assert_allclose(lon, entry["lon"])
    assert jr.shape == (entry["jr"].size,)


def test_fallback_wind(force_fallback):
    """Test that wind inputs are correctly loaded from fallback."""
    result = get_wind_inputs(_utc_now(), use_wind=True, time=None)
    assert result is not None
    u_theta, u_phi, lat, lon, weights = result
    fallback = _load_fallback()
    wind = fallback["wind"]
    assert u_theta.shape == (wind["u_theta"].size,)
    np.testing.assert_allclose(lat, wind["lat"])
    np.testing.assert_allclose(lon, wind["lon"])
    assert weights.shape[0] == 2


def test_wind_disabled(force_fallback):
    """Test that wind inputs are disabled when requested."""
    assert get_wind_inputs(_utc_now(), use_wind=False, time=None) is None


def test_fallback_roundtrip(tmp_path):
    """Test saving and loading a custom fallback dataset."""
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 90.0])
    grid_shape = (lat.size, lon.size)
    base = np.arange(np.prod(grid_shape), dtype=float).reshape(grid_shape)
    hall = base.copy()
    pedersen = base + 0.5
    jr = base - 0.25
    u_theta = -np.ones(grid_shape)
    u_phi = 2.0 * np.ones(grid_shape)
    time = np.array([0.0, 600.0])

    destination = tmp_path / "custom_fallback.json"
    saved_path = save_fallback_dataset(
        destination,
        lat=lat,
        lon=lon,
        hall=hall,
        pedersen=pedersen,
        jr=jr,
        u_theta=u_theta,
        u_phi=u_phi,
        time=time,
        indent=None,
    )

    assert saved_path == destination

    payload = _load_fallback(saved_path)
    assert payload["version"] == 3
    assert payload["coordinate_system"] == "GEO"
    np.testing.assert_allclose(payload["time"], time)

    wind = payload["wind"]
    np.testing.assert_allclose(wind["lat"], lat.reshape(-1))
    np.testing.assert_allclose(wind["lon"], lon.reshape(-1))
    np.testing.assert_allclose(wind["u_theta"], u_theta.reshape(-1))
    np.testing.assert_allclose(wind["u_phi"], u_phi.reshape(-1))

    conductance = payload["conductance"]["default"]
    np.testing.assert_allclose(conductance["lat"], lat.reshape(-1))
    np.testing.assert_allclose(conductance["lon"], lon.reshape(-1))
    np.testing.assert_allclose(conductance["hall"], hall.reshape(-1))
    np.testing.assert_allclose(conductance["pedersen"], pedersen.reshape(-1))

    jr_entry = payload["jr"]["default"]
    np.testing.assert_allclose(jr_entry["lat"], lat.reshape(-1))
    np.testing.assert_allclose(jr_entry["lon"], lon.reshape(-1))
    np.testing.assert_allclose(jr_entry["jr"], jr.reshape(-1))


def test_expand_time_series_repeats_values():
    """Test that _expand_time_series correctly repeats values."""
    data = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0, 2.0])
    expanded = _expand_time_series(data, time)
    assert expanded.shape == (time.size, data.size)
    scaling = np.linspace(1.0, 2.0, time.size)
    for idx, scale in enumerate(scaling):
        np.testing.assert_allclose(expanded[idx], data * scale)
