"""Tests for external input handling with fallback datasets."""

import datetime

import numpy as np
import pytest

from pynamit.external_inputs import (
    _load_fallback,
    _expand_time_series,
    get_conductance_inputs,
    get_jr_inputs,
    get_wind_inputs,
    get_input_source,
    set_input_source,
    save_fallback_dataset,
)


pytestmark = pytest.mark.requires_native_inputs


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
    hall, pedersen, out_lat, out_lon = get_conductance_inputs(
        datetime.datetime.utcnow(), lat, lon, time
    )
    assert hall.shape == (entry["hall"].size,)
    assert pedersen.shape == (entry["pedersen"].size,)
    np.testing.assert_allclose(out_lat, entry["lat"])
    np.testing.assert_allclose(out_lon, entry["lon"])


def test_fallback_multi_time_scaling(force_fallback):
    """Test that time-dependent scaling is applied correctly."""
    time = np.array([0.0, 10.0, 20.0])
    fallback = _load_fallback()
    key, entry = next(iter(fallback["conductance"].items()))
    hall, pedersen, *_ = get_conductance_inputs(
        datetime.datetime.utcnow(), entry["lat"], entry["lon"], time
    )
    assert hall.shape == (time.size, entry["hall"].size)
    assert pedersen.shape == (time.size, entry["pedersen"].size)


def test_fallback_currents(force_fallback):
    """Test that jr inputs are correctly loaded from fallback."""
    fallback = _load_fallback()
    key, entry = next(iter(fallback["jr"].items()))
    jr, lat, lon = get_jr_inputs(datetime.datetime.utcnow(), entry["lat"], entry["lon"], None)
    np.testing.assert_allclose(lat, entry["lat"])
    np.testing.assert_allclose(lon, entry["lon"])
    assert jr.shape == (entry["jr"].size,)


def test_fallback_wind(force_fallback):
    """Test that wind inputs are correctly loaded from fallback."""
    result = get_wind_inputs(datetime.datetime.utcnow(), use_wind=True, time=None)
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
    assert get_wind_inputs(datetime.datetime.utcnow(), use_wind=False, time=None) is None


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
    assert payload["version"] == 2
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
