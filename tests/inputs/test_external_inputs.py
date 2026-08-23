"""Tests for native and fallback external-input adapters."""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from tests.example_scenario import EVENT_TIME

from pynamit.external_inputs import providers as external_inputs_module
from pynamit.external_inputs.contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    LIBRARY_GEOGRAPHIC_110KM,
    NEUTRAL_WIND_PROVIDER_SPEC,
    ExternalInputRequest,
)
from pynamit.external_inputs.providers import (
    _library_horizontal_wind_to_spherical,
    _load_fallback,
    _select_fallback_entry,
    get_boundary_jr_inputs,
    get_conductance_inputs,
    get_input_source,
    get_wind_inputs,
    save_fallback_dataset,
    set_input_source,
)


def _utc_now():
    """Return a timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


def _request(grid_id="test-source"):
    return ExternalInputRequest.from_geocentric_geo(
        np.array([-70.0, 0.0, 45.0]), np.array([-30.0, 10.0, 80.0]), grid_id=grid_id
    )


def _centered_dipole_request(grid_id="test-dipole-source"):
    model_epoch = 2001.3613869863013
    model_lat = np.array([-75.0, 10.0, 65.0])
    model_lon = np.array([-90.0, 20.0, 130.0])
    geographic_lat, geographic_lon = external_inputs_module.dipole.Dipole(model_epoch).mag2geo(
        model_lat, model_lon
    )
    return ExternalInputRequest.from_model_coordinates(
        model_lat,
        model_lon,
        geographic_lat=geographic_lat,
        geographic_lon=geographic_lon,
        coordinate_system="centered_dipole",
        model_epoch=model_epoch,
        grid_id=grid_id,
    )


def test_native_geographic_conductance_uses_shared_library_request_grid(monkeypatch):
    """Let Hardy derive modified-Apex coordinates for a GEO request."""
    captured = {}

    class FakeSunlight:
        @staticmethod
        def sza(lat, lon, date):
            captured["sza"] = (np.asarray(lat), np.asarray(lon), date)
            return np.full(np.asarray(lat).shape, 60.0)

    class FakeConductance:
        @staticmethod
        def EUV_conductance(sza, f107, components, *, calibration):
            captured["euv"] = (np.asarray(sza), f107, components, calibration)
            values = np.ones(np.asarray(sza).shape)
            return 3.0 * values, 4.0 * values

        @staticmethod
        def hardy(lat, mlt, kp, components):
            captured["hardy"] = (np.asarray(lat), np.asarray(mlt), kp, components)
            values = np.ones(np.asarray(lat).shape)
            return values, 2.0 * values

        sunlight = FakeSunlight

    class FakeApex:
        def __init__(self, *, date, refh):
            captured["apex"] = (date, refh)

        def geo2apex(self, lat, lon, height):
            captured["geo"] = (np.asarray(lat), np.asarray(lon), height)
            return np.asarray(lat) + 5.0, np.asarray(lon) + 10.0

    class FakeDipole:
        def __init__(self, epoch):
            captured["epoch"] = epoch

        def mlon2mlt(self, lon, date):
            captured["mlt_date"] = date
            return np.asarray(lon) / 15.0

    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: FakeConductance
    )
    monkeypatch.setattr(external_inputs_module.apexpy, "Apex", FakeApex)
    monkeypatch.setattr(external_inputs_module.dipole, "Dipole", FakeDipole)
    request = _request()
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    source = request.source_grid
    date = datetime.datetime(
        2001, 5, 13, 0, 45, tzinfo=datetime.timezone(datetime.timedelta(hours=3))
    )

    pedersen, hall, out_lat, out_lon = get_conductance_inputs(
        date, request=request, kp=5, starlight=1.0
    )

    provider_date = datetime.datetime(2001, 5, 12, 21, 45)
    assert captured["apex"] == (provider_date, 110.0)
    assert captured["epoch"] == pytest.approx(external_inputs_module.decimal_year(provider_date))
    np.testing.assert_allclose(captured["geo"][0], provider_grid.lat)
    np.testing.assert_allclose(captured["geo"][1], provider_grid.lon)
    assert captured["geo"][2] == 110.0
    np.testing.assert_allclose(captured["hardy"][0], provider_grid.lat + 5.0)
    np.testing.assert_allclose(captured["hardy"][1], (provider_grid.lon + 10.0) / 15.0)
    assert captured["hardy"][2:] == (5, "hp")
    np.testing.assert_allclose(captured["sza"][0], source.lat)
    np.testing.assert_allclose(captured["sza"][1], source.lon)
    assert captured["sza"][2] == provider_date
    assert captured["euv"][1:] == (100, "hp", "MoenBrekke1993")
    assert provider_grid.coordinate_convention is LIBRARY_GEOGRAPHIC_110KM
    np.testing.assert_allclose(hall, np.sqrt(11.0))
    np.testing.assert_allclose(pedersen, np.sqrt(21.0))
    np.testing.assert_allclose(out_lat, source.lat)
    np.testing.assert_allclose(out_lon, source.lon)
    np.testing.assert_array_equal(provider_grid.lat, source.lat)
    np.testing.assert_array_equal(provider_grid.lon, source.lon)


def test_native_dipole_conductance_uses_explicit_model_and_geo_views(monkeypatch):
    """Evaluate auroral and solar terms in their declared views."""
    captured = {}

    class FakeSunlight:
        @staticmethod
        def sza(lat, lon, date):
            captured["sza"] = (np.asarray(lat), np.asarray(lon), date)
            return np.full(np.asarray(lat).shape, 60.0)

    class FakeConductance:
        @staticmethod
        def EUV_conductance(sza, f107, components, *, calibration):
            captured["euv"] = (np.asarray(sza), f107, components, calibration)
            values = np.ones(np.asarray(sza).shape)
            return 3.0 * values, 4.0 * values

        @staticmethod
        def hardy(lat, mlt, kp, components):
            captured["hardy"] = (np.asarray(lat), np.asarray(mlt), kp, components)
            values = np.ones(np.asarray(lat).shape)
            return values, 2.0 * values

        sunlight = FakeSunlight

    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: FakeConductance
    )
    request = _centered_dipole_request()
    real_dipole = external_inputs_module.dipole.Dipole(request.model_epoch)

    class FakeDipole:
        def __init__(self, epoch):
            captured["epoch"] = epoch

        def mag2geo(self, lat, lon):
            return real_dipole.mag2geo(lat, lon)

        def mlon2mlt(self, lon, date):
            captured["mlt_date"] = date
            return np.asarray(lon) / 15.0

    monkeypatch.setattr(external_inputs_module.dipole, "Dipole", FakeDipole)

    date = _utc_now()
    pedersen, hall, out_lat, out_lon = get_conductance_inputs(
        date, request=request, kp=5, starlight=1.0
    )

    assert captured["epoch"] == pytest.approx(request.model_epoch)
    np.testing.assert_allclose(captured["hardy"][0], request.model_grid.lat)
    np.testing.assert_allclose(captured["hardy"][1], request.model_grid.lon / 15.0)
    assert captured["hardy"][2:] == (5, "hp")
    np.testing.assert_allclose(captured["sza"][0], request.source_grid.lat)
    np.testing.assert_allclose(captured["sza"][1], request.source_grid.lon)
    assert captured["euv"][1:] == (100, "hp", "MoenBrekke1993")
    np.testing.assert_allclose(hall, np.sqrt(11.0))
    np.testing.assert_allclose(pedersen, np.sqrt(21.0))
    np.testing.assert_allclose(out_lat, request.source_grid.lat)
    np.testing.assert_allclose(out_lon, request.source_grid.lon)


def test_native_geographic_jr_uses_shared_library_request_grid(monkeypatch):
    """Let ApexPy derive real-Earth AMPS coordinates for GEO."""
    captured = {}

    class FakeApex:
        def __init__(self, *, date, refh):
            captured["apex_date"] = date
            captured["apex_refh"] = refh

        def geo2apex(self, lat, lon, height):
            captured["geo"] = (np.asarray(lat).copy(), np.asarray(lon).copy(), height)
            return np.asarray(lat) + 5.0, np.asarray(lon) + 10.0

    class FakeAMPS:
        def __init__(self, *args, **kwargs):
            captured["amps_init"] = (args, kwargs)

        def get_upward_current(self, *, mlat, mlt):
            captured["amps_query"] = (np.asarray(mlat).copy(), np.asarray(mlt).copy())
            return np.ones(np.asarray(mlat).shape)

    def fake_mlon_to_mlt(mlon, date, epoch):
        captured["mlt_conversion"] = (np.asarray(mlon).copy(), date, epoch)
        return np.asarray(mlon) / 15.0

    fake_pyamps = SimpleNamespace(
        __file__="/tmp/pyamps/__init__.py", AMPS=FakeAMPS, mlon_to_mlt=fake_mlon_to_mlt
    )
    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: fake_pyamps
    )
    monkeypatch.setattr(external_inputs_module.apexpy, "Apex", FakeApex)

    request = _request()
    provider_grid = request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
    date = datetime.datetime(
        2001, 5, 13, 0, 45, tzinfo=datetime.timezone(datetime.timedelta(hours=3))
    )
    jr, out_lat, out_lon = get_boundary_jr_inputs(
        date, request=request, v=300.0, By=0.0, Bz=-4.0, tilt=20.0, f107=100.0, minlat=50.0
    )

    expected_mlat = provider_grid.lat + 5.0
    expected_mlon = provider_grid.lon + 10.0
    expected_date = datetime.datetime(2001, 5, 12, 21, 45)
    assert captured["apex_date"] == expected_date
    np.testing.assert_allclose(captured["geo"][0], provider_grid.lat)
    np.testing.assert_allclose(captured["geo"][1], provider_grid.lon)
    assert captured["geo"][2] == 110.0
    np.testing.assert_allclose(captured["mlt_conversion"][0], expected_mlon)
    assert captured["mlt_conversion"][1] == expected_date
    assert captured["mlt_conversion"][2] == external_inputs_module.decimal_year(expected_date)
    np.testing.assert_allclose(captured["amps_query"][0], expected_mlat)
    np.testing.assert_allclose(captured["amps_query"][1], expected_mlon / 15.0)
    expected_jr = np.full(request.source_grid.size, 1e-6)
    expected_jr[np.abs(expected_mlat) < 50.0] = 0.0
    np.testing.assert_allclose(jr, expected_jr)
    np.testing.assert_allclose(out_lat, request.source_grid.lat)
    np.testing.assert_allclose(out_lon, request.source_grid.lon)


def test_native_dipole_jr_uses_explicit_model_view_and_epoch(monkeypatch):
    """Evaluate AMPS directly in the simulation's ideal-dipole frame."""
    captured = {}
    request = _centered_dipole_request()
    real_dipole = external_inputs_module.dipole.Dipole(request.model_epoch)

    class FakeDipole:
        def __init__(self, epoch):
            captured["epoch"] = epoch

        def mag2geo(self, lat, lon):
            return real_dipole.mag2geo(lat, lon)

        def mlon2mlt(self, lon, date):
            captured["mlt_conversion"] = (np.asarray(lon).copy(), date)
            return np.asarray(lon) / 15.0

    class FakeApex:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("Dipole AMPS input must not use ApexPy.")

    class FakeAMPS:
        def __init__(self, *args, **kwargs):
            captured["amps_init"] = (args, kwargs)

        def get_upward_current(self, *, mlat, mlt):
            captured["amps_query"] = (np.asarray(mlat).copy(), np.asarray(mlt).copy())
            return np.ones(np.asarray(mlat).shape)

    def fake_mlon_to_mlt(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Dipole AMPS input must use the simulation dipole for MLT.")

    fake_pyamps = SimpleNamespace(
        __file__="/tmp/pyamps/__init__.py", AMPS=FakeAMPS, mlon_to_mlt=fake_mlon_to_mlt
    )
    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: fake_pyamps
    )
    monkeypatch.setattr(external_inputs_module.dipole, "Dipole", FakeDipole)
    monkeypatch.setattr(external_inputs_module.apexpy, "Apex", FakeApex)
    date = datetime.datetime(
        2001, 5, 13, 0, 45, tzinfo=datetime.timezone(datetime.timedelta(hours=3))
    )

    jr, out_lat, out_lon = get_boundary_jr_inputs(
        date, request=request, v=300.0, By=0.0, Bz=-4.0, tilt=20.0, f107=100.0, minlat=50.0
    )

    expected_date = datetime.datetime(2001, 5, 12, 21, 45)
    assert captured["epoch"] == pytest.approx(request.model_epoch)
    np.testing.assert_array_equal(captured["mlt_conversion"][0], request.model_grid.lon)
    assert captured["mlt_conversion"][1] == expected_date
    np.testing.assert_array_equal(captured["amps_query"][0], request.model_grid.lat)
    np.testing.assert_allclose(captured["amps_query"][1], request.model_grid.lon / 15.0)
    expected_jr = np.full(request.source_grid.size, 1e-6)
    expected_jr[np.abs(request.model_grid.lat) < 50.0] = 0.0
    np.testing.assert_allclose(jr, expected_jr)
    np.testing.assert_array_equal(out_lat, request.source_grid.lat)
    np.testing.assert_array_equal(out_lon, request.source_grid.lon)


def test_native_wind_uses_requested_positions_and_correct_date(monkeypatch):
    """HWM receives the shared grid and full date/time."""
    captured = {}

    class FakePyHWM:
        @staticmethod
        def hwm14_vectorized(**kwargs):
            captured.update(kwargs)
            size = np.asarray(kwargs["glat_deg"]).size
            return np.full(size, 12.0), np.full(size, 30.0)

    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: FakePyHWM
    )

    request = _request()
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    date = datetime.datetime(
        2001, 5, 13, 0, 45, 30, 500000, tzinfo=datetime.timezone(datetime.timedelta(hours=3))
    )
    u_theta, u_phi, lat, lon, weights = get_wind_inputs(date, request=request, ap=(-1, 35))

    np.testing.assert_allclose(captured["glat_deg"], provider_grid.lat)
    np.testing.assert_allclose(captured["glon_deg"], provider_grid.lon)
    np.testing.assert_allclose(captured["alt_km"], 110.0)
    np.testing.assert_allclose(captured["utc_hours"], 21 + 45 / 60 + 30.5 / 3600)
    assert captured["iyd"] == 1132
    assert captured["ap"] == [-1, 35]

    np.testing.assert_array_equal(provider_grid.lat, request.source_grid.lat)
    np.testing.assert_array_equal(provider_grid.lon, request.source_grid.lon)
    np.testing.assert_allclose(u_phi, 12.0)
    np.testing.assert_allclose(u_theta, -30.0)
    np.testing.assert_allclose(lat, request.source_grid.lat)
    np.testing.assert_allclose(lon, request.source_grid.lon)
    assert weights is None


def test_native_conductance_rejects_wrong_sample_count(monkeypatch):
    """Provider results must remain aligned with the requested grid."""

    class FakeSunlight:
        @staticmethod
        def sza(lat, lon, date):
            del lon, date
            return np.ones(np.asarray(lat).shape)

    class FakeConductance:
        @staticmethod
        def EUV_conductance(sza, f107, components, *, calibration):
            del f107, components, calibration
            values = np.ones(np.asarray(sza).size - 1)
            return values, values

        @staticmethod
        def hardy(lat, mlt, kp, components):
            del mlt, kp, components
            values = np.ones(np.asarray(lat).size - 1)
            return values, values

        sunlight = FakeSunlight

    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: FakeConductance
    )

    with pytest.raises(ValueError, match="2 'hall' values for a 3-point"):
        get_conductance_inputs(_utc_now(), request=_request(), kp=5, starlight=1.0)


def test_library_wind_mapping_is_spherical_component_identity():
    """Library east/north map directly to spherical phi/minus-theta."""
    request = _request()
    u_theta, u_phi = _library_horizontal_wind_to_spherical(
        request, np.full(request.source_grid.size, 5.0), np.full(request.source_grid.size, 20.0)
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


def test_fallback_rejects_inconsistent_dipole_and_geo_views(force_fallback):
    """Fallback enforces the native paired-frame contract."""
    valid = _centered_dipole_request()
    inconsistent = ExternalInputRequest.from_model_coordinates(
        valid.model_grid.lat,
        valid.model_grid.lon,
        geographic_lat=valid.source_grid.lat + 0.1,
        geographic_lon=valid.source_grid.lon,
        coordinate_system="centered_dipole",
        model_epoch=valid.model_epoch,
    )

    with pytest.raises(ValueError, match="physical GEO samples disagree"):
        get_conductance_inputs(_utc_now(), request=inconsistent, kp=5, starlight=1.0)


def test_loaded_collection_shares_both_grid_views():
    """Providers share source and request-grid objects."""
    fallback = _load_fallback()
    assert fallback.version == 7
    assert fallback.event_time == EVENT_TIME.isoformat()
    assert fallback.conditions == {
        "conductance": {"kp": 5, "starlight": 1.0},
        "boundary_jr": {
            "v": 300.0,
            "By": 0.0,
            "Bz": -4.0,
            "tilt": 20.0,
            "f107": 100.0,
            "minlat": 50.0,
        },
        "neutral_wind": {"ap": (-1, 35)},
    }
    for source_grid_id in fallback.datasets["conductance"]:
        hardy = fallback.datasets["conductance"][source_grid_id]
        amps = fallback.datasets["boundary_jr"][source_grid_id]
        hwm = fallback.datasets["neutral_wind"][source_grid_id]
        assert hardy.source_grid is amps.source_grid is hwm.source_grid
        assert hardy.request_grid is amps.request_grid is hwm.request_grid
        assert (
            hardy.spec.request_coordinate_convention
            == amps.spec.request_coordinate_convention
            == hwm.spec.request_coordinate_convention
            == LIBRARY_GEOGRAPHIC_110KM
        )


def test_fallback_all_providers_match_exact_source_grid(force_fallback):
    """All fallback providers select the same exact source grid."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    source = fallback.datasets["conductance"][source_grid_id].source_grid
    request = ExternalInputRequest(source)

    pedersen, hall, conductance_lat, conductance_lon = get_conductance_inputs(
        EVENT_TIME, request=request, kp=5, starlight=1.0
    )
    jr, jr_lat, jr_lon = get_boundary_jr_inputs(
        EVENT_TIME, request=request, v=300.0, By=0.0, Bz=-4.0, tilt=20.0, f107=100.0, minlat=50.0
    )
    u_theta, u_phi, wind_lat, wind_lon, weights = get_wind_inputs(
        EVENT_TIME, request=request, ap=(-1, 35)
    )

    assert hall.shape == pedersen.shape == jr.shape == u_theta.shape == u_phi.shape
    np.testing.assert_allclose(conductance_lat, source.lat)
    np.testing.assert_allclose(conductance_lon, source.lon)
    np.testing.assert_allclose(jr_lat, source.lat)
    np.testing.assert_allclose(jr_lon, source.lon)
    np.testing.assert_allclose(wind_lat, source.lat)
    np.testing.assert_allclose(wind_lon, source.lon)
    assert weights is None


def test_fallback_rejects_another_event_time(force_fallback):
    """A bundled snapshot cannot silently stand in for another event."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    source = fallback.datasets["conductance"][source_grid_id].source_grid

    with pytest.raises(ValueError, match="describe only 2001-05-12 21:45:00 UTC"):
        get_conductance_inputs(
            datetime.datetime(2001, 5, 12, 21, 46),
            request=ExternalInputRequest(source),
            kp=5,
            starlight=1.0,
        )


def test_fallback_selection_is_provider_specific():
    """A dataset cannot use another provider specification."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    dataset = fallback.datasets["conductance"][source_grid_id]
    request = ExternalInputRequest(dataset.source_grid)
    selected = _select_fallback_entry(
        fallback.datasets["conductance"], request, "conductance", spec=CONDUCTANCE_PROVIDER_SPEC
    )
    assert selected is dataset
    with pytest.raises(ValueError, match="provider specification"):
        _select_fallback_entry(
            fallback.datasets["conductance"],
            request,
            "boundary_jr",
            spec=BOUNDARY_JR_PROVIDER_SPEC,
        )


def test_fallback_rejects_conditions_other_than_its_cached_event(force_fallback):
    """Cached fields cannot represent different physical drivers."""
    fallback = _load_fallback()
    source_grid_id = next(iter(fallback.datasets["conductance"]))
    request = ExternalInputRequest(fallback.datasets["conductance"][source_grid_id].source_grid)

    with pytest.raises(ValueError, match="conductance fallback data use"):
        get_conductance_inputs(EVENT_TIME, request=request, kp=4, starlight=1.0)
    with pytest.raises(ValueError, match="AMPS fallback data use"):
        get_boundary_jr_inputs(
            EVENT_TIME,
            request=request,
            v=301.0,
            By=0.0,
            Bz=-4.0,
            tilt=20.0,
            f107=100.0,
            minlat=50.0,
        )
    with pytest.raises(ValueError, match="HWM fallback data use"):
        get_wind_inputs(EVENT_TIME, request=request, ap=(-1, 34))


def test_fallback_error_lists_compatible_grid_geometry():
    """Missing-grid diagnostics describe available source grids."""
    fallback = _load_fallback()
    request = ExternalInputRequest.from_geocentric_geo(
        np.array([0.0]), np.array([0.0]), grid_id="missing"
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
    assert "geographic-ncs-18 (geocentric_geographic, Ncs=18" in message


def test_fallback_roundtrip_defaults_to_one_shared_source_grid(tmp_path):
    """The convenience writer shares equivalent grids."""
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
    np.testing.assert_allclose(amps.values["jr"], (base - 0.25).reshape(-1))


def test_native_conductance_accepts_explicit_kp(monkeypatch):
    """Provider physics parameters are separate from contracts."""
    captured = {}

    class FakeSunlight:
        @staticmethod
        def sza(lat, lon, date):
            del lon, date
            return np.ones(np.asarray(lat).shape)

    class FakeConductance:
        @staticmethod
        def EUV_conductance(sza, f107, components, *, calibration):
            del f107, components, calibration
            values = np.zeros(np.asarray(sza).shape)
            return values, values

        @staticmethod
        def hardy(lat, mlt, kp, components):
            del mlt, components
            captured["kp"] = kp
            values = np.ones(np.asarray(lat).shape)
            return values, values

        sunlight = FakeSunlight

    request = ExternalInputRequest.from_geocentric_geo(np.array([60.0]), np.array([10.0]))
    monkeypatch.setattr(
        external_inputs_module, "_load_optional_module", lambda _name, _package: FakeConductance
    )
    get_conductance_inputs(
        _utc_now(),
        request.source_grid.lat,
        request.source_grid.lon,
        request=request,
        kp=4.0,
        starlight=1.0,
    )
    assert captured["kp"] == pytest.approx(4.0)
