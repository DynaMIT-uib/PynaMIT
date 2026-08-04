"""Tests for native and fallback external-input adapters."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import pynamit.external_inputs as external_inputs_module
from pynamit.external_input_contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    ExternalInputRequest,
    LIBRARY_GEOGRAPHIC_110KM,
)
from pynamit.external_inputs import (
    _expand_time_series,
    _library_horizontal_wind_to_spherical,
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
    """Return a timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.UTC)


def _request(grid_id="test-source"):
    return ExternalInputRequest.from_geocentric_geo(
        np.array([-70.0, 0.0, 45.0]),
        np.array([-30.0, 10.0, 80.0]),
        grid_id=grid_id,
    )


def test_native_conductance_uses_shared_library_request_grid(monkeypatch):
    """Hardy receives the shared identity-mapped library request grid."""
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

    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: FakeConductance,
    )
    request = _request()
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    source = request.source_grid
    date = datetime.datetime(2001, 5, 12, 21, 45)

    hall, pedersen, out_lat, out_lon = get_conductance_inputs(
        date,
        None,
        None,
        None,
        request=request,
    )

    assert captured["dipole"] is False
    assert captured["kp"] == 5
    np.testing.assert_allclose(captured["lat"], provider_grid.lat)
    np.testing.assert_allclose(captured["lon"], provider_grid.lon)
    assert provider_grid.coordinate_contract is LIBRARY_GEOGRAPHIC_110KM
    np.testing.assert_allclose(hall, 1.0)
    np.testing.assert_allclose(pedersen, 2.0)
    np.testing.assert_allclose(out_lat, source.lat)
    np.testing.assert_allclose(out_lon, source.lon)
    np.testing.assert_array_equal(provider_grid.lat, source.lat)
    np.testing.assert_array_equal(provider_grid.lon, source.lon)


def test_native_jr_uses_same_library_request_grid(monkeypatch):
    """AMPS receives the same identity-mapped view before Apex conversion."""
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
    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: fake_pyamps,
    )
    monkeypatch.setattr(external_inputs_module.apexpy, "Apex", FakeApex)

    request = _request()
    provider_grid = request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
    date = datetime.datetime(2001, 5, 12, 21, 45)
    jr, out_lat, out_lon = get_jr_inputs(
        date,
        None,
        None,
        None,
        request=request,
    )

    expected_mlat = provider_grid.lat + 5.0
    expected_mlon = provider_grid.lon + 10.0
    np.testing.assert_allclose(captured["geo"][0], provider_grid.lat)
    np.testing.assert_allclose(captured["geo"][1], provider_grid.lon)
    assert captured["geo"][2] == 110.0
    np.testing.assert_allclose(captured["mlt_conversion"][0], expected_mlon)
    np.testing.assert_allclose(captured["amps_query"][0], expected_mlat)
    np.testing.assert_allclose(
        captured["amps_query"][1],
        expected_mlon / 15.0,
    )
    expected_jr = np.full(request.source_grid.size, 1e-6)
    expected_jr[np.abs(expected_mlat) < 50.0] = 0.0
    np.testing.assert_allclose(jr, expected_jr)
    np.testing.assert_allclose(out_lat, request.source_grid.lat)
    np.testing.assert_allclose(out_lon, request.source_grid.lon)


def test_native_wind_uses_requested_positions_and_correct_date(monkeypatch):
    """HWM vectorized receives the shared library grid and full date/time."""
    captured = {}

    class FakePyHWM:
        @staticmethod
        def hwm14_vectorized(**kwargs):
            captured.update(kwargs)
            size = np.asarray(kwargs["glat_deg"]).size
            return np.full(size, 12.0), np.full(size, 30.0)

    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: FakePyHWM,
    )

    request = _request()
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    date = datetime.datetime(2001, 5, 12, 21, 45, 30, 500000)
    result = get_wind_inputs(
        date,
        use_wind=True,
        time=None,
        request=request,
    )
    assert result is not None
    u_theta, u_phi, lat, lon, weights = result

    np.testing.assert_allclose(captured["glat_deg"], provider_grid.lat)
    np.testing.assert_allclose(captured["glon_deg"], provider_grid.lon)
    np.testing.assert_allclose(captured["alt_km"], 110.0)
    np.testing.assert_allclose(
        captured["utc_hours"],
        21 + 45 / 60 + 30.5 / 3600,
    )
    assert captured["iyd"] == 1132
    assert captured["ap"] == [-1, 35]

    np.testing.assert_array_equal(provider_grid.lat, request.source_grid.lat)
    np.testing.assert_array_equal(provider_grid.lon, request.source_grid.lon)
    np.testing.assert_allclose(u_phi, 12.0)
    np.testing.assert_allclose(u_theta, -30.0)
    np.testing.assert_allclose(lat, request.source_grid.lat)
    np.testing.assert_allclose(lon, request.source_grid.lon)
    assert weights is None


def test_library_wind_mapping_is_spherical_component_identity():
    """Library east/north map directly to spherical phi/minus-theta."""
    request = _request()
    u_theta, u_phi = _library_horizontal_wind_to_spherical(
        request,
        np.full(request.source_grid.size, 5.0),
        np.full(request.source_grid.size, 20.0),
    )
    np.testing.assert_allclose(u_phi, 5.0)
    np.testing.assert_allclose(u_theta, -20.0)


@pytest.fixture
def force_fallback():
    """Force bundled fallback adapters."""
    previous = get_input_source()
    set_input_source("fallback")
    yield
    set_input_source(previous)


def test_loaded_collection_shares_both_grid_views():
    """All providers share source and identity-mapped request-grid objects."""
    fallback = _load_fallback()
    assert fallback.version == 4
    for source_grid_id in fallback.datasets["conductance"]:
        hardy = fallback.datasets["conductance"][source_grid_id]
        amps = fallback.datasets["boundary_jr"][source_grid_id]
        hwm = fallback.datasets["neutral_wind"][source_grid_id]
        assert hardy.source_grid is amps.source_grid is hwm.source_grid
        assert hardy.request_grid is amps.request_grid is hwm.request_grid
        assert (
            hardy.spec.request_coordinate_contract
            is amps.spec.request_coordinate_contract
            is hwm.spec.request_coordinate_contract
            is LIBRARY_GEOGRAPHIC_110KM
        )


def test_fallback_all_providers_match_exact_source_grid(force_fallback):
    """All fallback providers select the same exact source grid."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    source = fallback.datasets["conductance"][source_grid_id].source_grid
    request = ExternalInputRequest(source)

    hall, pedersen, hall_lat, hall_lon = get_conductance_inputs(
        _utc_now(),
        None,
        None,
        None,
        request=request,
    )
    jr, jr_lat, jr_lon = get_jr_inputs(
        _utc_now(),
        None,
        None,
        None,
        request=request,
    )
    wind = get_wind_inputs(
        _utc_now(),
        use_wind=True,
        time=None,
        request=request,
    )
    assert wind is not None
    u_theta, u_phi, wind_lat, wind_lon, weights = wind

    assert hall.shape == pedersen.shape == jr.shape == u_theta.shape == u_phi.shape
    np.testing.assert_allclose(hall_lat, source.lat)
    np.testing.assert_allclose(hall_lon, source.lon)
    np.testing.assert_allclose(jr_lat, source.lat)
    np.testing.assert_allclose(jr_lon, source.lon)
    np.testing.assert_allclose(wind_lat, source.lat)
    np.testing.assert_allclose(wind_lon, source.lon)
    assert weights is None


def test_fallback_selection_is_provider_specific():
    """A dataset cannot be selected through another provider specification."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    dataset = fallback.datasets["conductance"][source_grid_id]
    request = ExternalInputRequest(dataset.source_grid)
    selected = _select_fallback_entry(
        fallback.datasets["conductance"],
        request,
        "conductance",
        spec=CONDUCTANCE_PROVIDER_SPEC,
    )
    assert selected is dataset
    with pytest.raises(ValueError, match="provider specification"):
        _select_fallback_entry(
            fallback.datasets["conductance"],
            request,
            "boundary_jr",
            spec=BOUNDARY_JR_PROVIDER_SPEC,
        )


def test_fallback_error_lists_compatible_grid_geometry():
    """Missing-grid diagnostics describe available source grids."""
    fallback = _load_fallback()
    request = ExternalInputRequest.from_geocentric_geo(
        np.array([0.0]),
        np.array([0.0]),
        grid_id="missing",
    )
    with pytest.raises(ValueError) as error:
        _select_fallback_entry(
            fallback.datasets["conductance"],
            request,
            "conductance",
            spec=fallback.providers["conductance"],
        )
    message = str(error.value)
    assert "Available compatible grids:" in message
    assert "geographic-ncs-18 (geographic, Ncs=18" in message


def test_synthetic_multi_time_scaling_is_preserved(force_fallback):
    """Constructed changes exercise multi-step input functionality."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["neutral_wind"]))
    source = fallback.datasets["neutral_wind"][source_grid_id].source_grid
    request = ExternalInputRequest(source)
    time = np.array([0.0, 10.0, 20.0])

    wind = get_wind_inputs(
        _utc_now(),
        use_wind=True,
        time=time,
        request=request,
    )
    assert wind is not None
    u_theta, u_phi, *_ = wind
    assert u_theta.shape == (time.size, source.size)
    assert u_phi.shape == (time.size, source.size)
    np.testing.assert_allclose(u_theta[-1], 2.0 * u_theta[0])
    np.testing.assert_allclose(u_phi[-1], 2.0 * u_phi[0])


def test_wind_disabled_requires_no_grid():
    """Disabled wind does not require source coordinates."""
    assert get_wind_inputs(_utc_now(), use_wind=False, time=None) is None


def test_fallback_roundtrip_defaults_to_one_shared_source_grid(tmp_path):
    """The convenience writer shares source/request grids where possible."""
    lat_axis = np.array([-60.0, 0.0, 60.0])
    lon_axis = np.array([0.0, 90.0])
    shape = (lat_axis.size, lon_axis.size)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    destination = tmp_path / "fallback.json"

    save_fallback_dataset(
        destination,
        lat=lat_axis,
        lon=lon_axis,
        hall=base,
        pedersen=base + 0.5,
        jr=base - 0.25,
        u_theta=-np.ones(shape),
        u_phi=2.0 * np.ones(shape),
        time=np.array([0.0, 600.0]),
        indent=None,
    )

    collection = _load_fallback(destination)
    hardy = collection.datasets["conductance"]["default"]
    amps = collection.datasets["boundary_jr"]["default"]
    hwm = collection.datasets["neutral_wind"]["default"]
    assert hardy.source_grid is amps.source_grid is hwm.source_grid
    assert hardy.request_grid is amps.request_grid is hwm.request_grid
    np.testing.assert_allclose(hardy.values["hall"], base.reshape(-1))
    np.testing.assert_allclose(
        amps.values["jr"],
        (base - 0.25).reshape(-1),
    )


def test_expand_time_series_returns_caller_owned_values():
    """Caller mutation cannot change immutable cached provider values."""
    source = np.array([1.0, 2.0, 3.0])
    result = _expand_time_series(source, None)
    result[0] = 99.0
    assert source[0] == pytest.approx(1.0)



def test_native_conductance_accepts_explicit_kp(monkeypatch):
    """Provider physics parameters are separate from coordinate contracts."""
    captured = {}

    class FakeConductance:
        @staticmethod
        def hardy_EUV(lon, lat, kp, date, *, starlight, dipole):
            captured["kp"] = kp
            values = np.ones(np.asarray(lat).shape)
            return values, values

    request = ExternalInputRequest.from_geocentric_geo(
        np.array([60.0]),
        np.array([10.0]),
    )
    monkeypatch.setattr(
        external_inputs_module,
        "_load_optional_module",
        lambda _name, _package: FakeConductance,
    )
    get_conductance_inputs(
        _utc_now(),
        request.source_grid.lat,
        request.source_grid.lon,
        None,
        request=request,
        kp=4.0,
    )
    assert captured["kp"] == pytest.approx(4.0)
