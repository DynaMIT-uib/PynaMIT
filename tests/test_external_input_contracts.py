"""Tests for external-input contracts and shared provider grids."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pynamit.external_input_contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    LIBRARY_GEOGRAPHIC_110KM,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    PYNAMIT_CENTERED_DIPOLE_110KM,
    PYNAMIT_SPHERICAL_GEO_110KM,
    CoordinateContract,
    ExternalInputRequest,
    FallbackCollection,
    ProviderDataset,
    ReferenceSurface,
    SampleGrid,
)


def _request(grid_id="source"):
    return ExternalInputRequest.from_geocentric_geo(
        np.array([-75.0, -20.0, 0.0, 45.0, 80.0]),
        np.array([-180.0, -30.0, 0.0, 90.0, 179.0]),
        grid_id=grid_id,
    )


def test_provider_specs_are_independent_but_share_request_contract():
    """Each library owns a spec while equal contracts are interned."""
    assert CONDUCTANCE_PROVIDER_SPEC is not BOUNDARY_JR_PROVIDER_SPEC
    assert BOUNDARY_JR_PROVIDER_SPEC is not NEUTRAL_WIND_PROVIDER_SPEC
    assert (
        CONDUCTANCE_PROVIDER_SPEC.request_coordinate_contract
        is BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_contract
        is NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_contract
        is LIBRARY_GEOGRAPHIC_110KM
    )
    assert (
        CONDUCTANCE_PROVIDER_SPEC.output_coordinate_contract
        is BOUNDARY_JR_PROVIDER_SPEC.output_coordinate_contract
        is NEUTRAL_WIND_PROVIDER_SPEC.output_coordinate_contract
        is PYNAMIT_SPHERICAL_GEO_110KM
    )
    assert CONDUCTANCE_PROVIDER_SPEC.fields == ("hall", "pedersen")
    assert CONDUCTANCE_PROVIDER_SPEC.request_coordinate_views == {
        "model": PYNAMIT_CENTERED_DIPOLE_110KM
    }
    assert not BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_views
    assert not NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_views
    assert BOUNDARY_JR_PROVIDER_SPEC.fields == ("jr",)
    assert NEUTRAL_WIND_PROVIDER_SPEC.fields == ("u_theta", "u_phi")


def test_shared_request_reuses_one_converted_grid_object():
    """Equal provider contracts reuse one identity-mapped grid."""
    request = _request()
    hardy_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    amps_grid = request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
    hwm_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    assert hardy_grid is amps_grid is hwm_grid
    assert hardy_grid.coordinate_contract is LIBRARY_GEOGRAPHIC_110KM
    np.testing.assert_array_equal(hardy_grid.lat, request.source_grid.lat)
    np.testing.assert_array_equal(hardy_grid.lon, request.source_grid.lon)


def test_model_request_retains_centered_dipole_and_geographic_views():
    """Expose both required views of one ordered sample set."""
    model_lat = np.array([75.0, -60.0])
    model_lon = np.array([-20.0, 130.0])
    geo_lat = np.array([68.0, -52.0])
    geo_lon = np.array([70.0, -140.0])

    request = ExternalInputRequest.from_model_coordinates(
        model_lat,
        model_lon,
        geographic_lat=geo_lat,
        geographic_lon=geo_lon,
        coordinate_system="centered_dipole",
        model_epoch=2001.5,
    )

    assert request.model_grid.coordinate_contract is PYNAMIT_CENTERED_DIPOLE_110KM
    np.testing.assert_array_equal(request.model_grid.lat, model_lat)
    np.testing.assert_array_equal(request.model_grid.lon, model_lon)
    np.testing.assert_array_equal(request.source_grid.lat, geo_lat)
    np.testing.assert_array_equal(request.source_grid.lon, geo_lon)
    assert request.model_epoch == pytest.approx(2001.5)
    assert request.grid_for(PYNAMIT_CENTERED_DIPOLE_110KM) is request.model_grid


def test_centered_dipole_model_request_requires_epoch():
    """A magnetic coordinate view requires its axis epoch."""
    with pytest.raises(ValueError, match="require model_epoch"):
        ExternalInputRequest.from_model_coordinates(
            np.array([60.0]),
            np.array([10.0]),
            geographic_lat=np.array([50.0]),
            geographic_lon=np.array([20.0]),
            coordinate_system="centered_dipole",
        )


def test_geographic_model_request_reuses_source_view():
    """A GEO model frame does not create a redundant coordinate grid."""
    lat = np.array([60.0, -40.0])
    lon = np.array([10.0, 120.0])
    request = ExternalInputRequest.from_model_coordinates(
        lat, lon, geographic_lat=lat, geographic_lon=lon, coordinate_system="geocentric_geographic"
    )
    assert request.model_grid is request.source_grid


def test_geographic_model_request_rejects_inconsistent_views():
    """Reject inconsistent GEO model and physical coordinates."""
    with pytest.raises(ValueError, match="must match geographic samples"):
        ExternalInputRequest.from_model_coordinates(
            np.array([60.0]),
            np.array([10.0]),
            geographic_lat=np.array([61.0]),
            geographic_lon=np.array([10.0]),
            coordinate_system="geocentric_geographic",
        )


def test_changing_one_provider_contract_does_not_change_the_others():
    """A provider can independently adopt another interface contract."""
    another_contract = CoordinateContract(
        coordinate_system="example_provider_coordinates",
        angular_units="degrees",
        latitude_definition="example",
        longitude_definition="east_positive",
        longitude_wrap="[-180,180)",
        reference_surface=ReferenceSurface(kind="sphere", radius_m=6_500_000.0),
    )
    changed = replace(BOUNDARY_JR_PROVIDER_SPEC, request_coordinate_contract=another_contract)
    assert changed.request_coordinate_contract is another_contract
    assert CONDUCTANCE_PROVIDER_SPEC.request_coordinate_contract is LIBRARY_GEOGRAPHIC_110KM
    assert NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_contract is LIBRARY_GEOGRAPHIC_110KM


def test_library_request_mapping_is_numeric_identity():
    """The common request preserves spherical coordinate labels."""
    request = _request()
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    np.testing.assert_array_equal(provider_grid.lat, request.source_grid.lat)
    np.testing.assert_array_equal(provider_grid.lon, request.source_grid.lon)
    assert provider_grid is not request.source_grid
    assert provider_grid.coordinate_contract is LIBRARY_GEOGRAPHIC_110KM


def test_coordinate_identity_normalizes_longitude_and_preserves_order():
    """Equivalent longitudes match while reordered samples do not."""
    contract = PYNAMIT_SPHERICAL_GEO_110KM
    first = contract.coordinate_identity(np.array([10.0, 20.0]), np.array([180.0, 350.0]))
    equivalent = contract.coordinate_identity(np.array([10.0, 20.0]), np.array([-180.0, -10.0]))
    reordered = contract.coordinate_identity(np.array([20.0, 10.0]), np.array([-10.0, -180.0]))
    assert first == equivalent
    assert first != reordered


def test_equal_arrays_under_different_contracts_are_different_grids():
    """Coordinate semantics are part of ordered-grid identity."""
    lat = np.array([10.0, 20.0])
    lon = np.array([0.0, 30.0])
    assert PYNAMIT_SPHERICAL_GEO_110KM.coordinate_identity(
        lat, lon
    ) != LIBRARY_GEOGRAPHIC_110KM.coordinate_identity(lat, lon)


def test_sample_grid_is_immutable_and_owns_arrays():
    """External mutation cannot alter a registered coordinate grid."""
    lat = np.array([10.0, 20.0])
    geometry = {"type": "sample_points"}
    grid = SampleGrid(
        grid_id="grid",
        coordinate_contract=PYNAMIT_SPHERICAL_GEO_110KM,
        lat=lat,
        lon=np.array([0.0, 30.0]),
        sampling_geometry=geometry,
    )
    lat[0] = -80.0
    geometry["type"] = "changed"
    assert grid.lat[0] == pytest.approx(10.0)
    assert grid.sampling_geometry["type"] == "sample_points"
    with pytest.raises(ValueError):
        grid.lat[0] = 0.0
    with pytest.raises(TypeError):
        grid.sampling_geometry["type"] = "changed"


def test_provider_dataset_requires_both_grid_contracts():
    """Provider datasets validate both coordinate contracts."""
    request = _request()
    source = request.source_grid
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    with pytest.raises(ValueError, match="request-grid contracts differ"):
        ProviderDataset(
            spec=CONDUCTANCE_PROVIDER_SPEC,
            source_grid=source,
            request_grid=source,
            values={"hall": np.ones(source.size), "pedersen": np.ones(source.size)},
        )
    dataset = ProviderDataset(
        spec=CONDUCTANCE_PROVIDER_SPEC,
        source_grid=source,
        request_grid=provider_grid,
        values={"hall": np.ones(source.size), "pedersen": np.ones(source.size)},
    )
    assert dataset.source_grid is source
    assert dataset.request_grid is provider_grid


def test_collection_roundtrip_shares_source_and_request_grids():
    """Equal provider views remain shared after JSON."""
    request = _request()
    source = request.source_grid
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    datasets = {
        "conductance": {
            source.grid_id: ProviderDataset(
                CONDUCTANCE_PROVIDER_SPEC,
                source,
                provider_grid,
                {"hall": np.ones(source.size), "pedersen": 2 * np.ones(source.size)},
            )
        },
        "boundary_jr": {
            source.grid_id: ProviderDataset(
                BOUNDARY_JR_PROVIDER_SPEC,
                source,
                request.grid_for(BOUNDARY_JR_PROVIDER_SPEC),
                {"jr": np.zeros(source.size)},
            )
        },
        "neutral_wind": {
            source.grid_id: ProviderDataset(
                NEUTRAL_WIND_PROVIDER_SPEC,
                source,
                request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC),
                {"u_theta": np.zeros(source.size), "u_phi": np.ones(source.size)},
            )
        },
    }
    collection = FallbackCollection(
        version=4,
        event_time="2001-05-12T21:45:00",
        time=np.array([0.0]),
        grids={source.grid_id: source, provider_grid.grid_id: provider_grid},
        providers=PROVIDER_SPECS,
        datasets=datasets,
    )
    loaded = FallbackCollection.from_payload(collection.to_payload(), expected_version=4)
    hardy = loaded.datasets["conductance"][source.grid_id]
    amps = loaded.datasets["boundary_jr"][source.grid_id]
    hwm = loaded.datasets["neutral_wind"][source.grid_id]
    assert hardy.source_grid is amps.source_grid is hwm.source_grid
    assert hardy.request_grid is amps.request_grid is hwm.request_grid
    assert (
        hardy.spec.request_coordinate_contract
        is amps.spec.request_coordinate_contract
        is hwm.spec.request_coordinate_contract
    )
