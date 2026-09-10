"""Tests for bundled external-input snapshots and shared grids."""

import numpy as np
import pytest

from pynamit.external_inputs.coordinates import ExternalInputCoordinates
from pynamit.external_inputs.fallback_data import FallbackCollection, ProviderSnapshot
from pynamit.external_inputs.provider_definitions import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
)


def _coordinates(grid_id="geographic"):
    return ExternalInputCoordinates.from_geocentric_geo(
        np.array([-75.0, -20.0, 0.0, 45.0, 80.0]),
        np.array([-180.0, -30.0, 0.0, 90.0, 179.0]),
        grid_id=grid_id,
    )


def test_provider_snapshot_requires_both_grid_conventions():
    """Provider datasets validate both coordinate conventions."""
    coordinates = _coordinates()
    geographic_grid = coordinates.geographic_grid
    provider_grid = coordinates.sample_grid(
        CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
    )
    with pytest.raises(ValueError, match="request-grid conventions differ"):
        ProviderSnapshot(
            spec=CONDUCTANCE_PROVIDER_SPEC,
            geographic_grid=geographic_grid,
            request_grid=geographic_grid,
            values={
                "hall": np.ones(geographic_grid.size),
                "pedersen": np.ones(geographic_grid.size),
            },
        )
    dataset = ProviderSnapshot(
        spec=CONDUCTANCE_PROVIDER_SPEC,
        geographic_grid=geographic_grid,
        request_grid=provider_grid,
        values={"hall": np.ones(geographic_grid.size), "pedersen": np.ones(geographic_grid.size)},
    )
    assert dataset.geographic_grid is geographic_grid
    assert dataset.request_grid is provider_grid


def test_collection_roundtrip_shares_geographic_and_request_grids():
    """Equal provider views remain shared after JSON."""
    coordinates = _coordinates()
    geographic_grid = coordinates.geographic_grid
    provider_grid = coordinates.sample_grid(
        CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
    )
    datasets = {
        "conductance": {
            geographic_grid.grid_id: ProviderSnapshot(
                CONDUCTANCE_PROVIDER_SPEC,
                geographic_grid,
                provider_grid,
                {
                    "hall": np.ones(geographic_grid.size),
                    "pedersen": 2 * np.ones(geographic_grid.size),
                },
            )
        },
        "boundary_jr": {
            geographic_grid.grid_id: ProviderSnapshot(
                BOUNDARY_JR_PROVIDER_SPEC,
                geographic_grid,
                coordinates.sample_grid(BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention),
                {"jr": np.zeros(geographic_grid.size)},
            )
        },
        "neutral_wind": {
            geographic_grid.grid_id: ProviderSnapshot(
                NEUTRAL_WIND_PROVIDER_SPEC,
                geographic_grid,
                coordinates.sample_grid(NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention),
                {
                    "u_theta": np.zeros(geographic_grid.size),
                    "u_phi": np.ones(geographic_grid.size),
                },
            )
        },
    }
    collection = FallbackCollection(
        version=4,
        event_time="2001-05-12T21:45:00",
        time=np.array([0.0]),
        grids={geographic_grid.grid_id: geographic_grid, provider_grid.grid_id: provider_grid},
        providers=PROVIDER_SPECS,
        datasets=datasets,
    )
    loaded = FallbackCollection.from_payload(collection.to_payload(), expected_version=4)
    hardy = loaded.datasets["conductance"][geographic_grid.grid_id]
    amps = loaded.datasets["boundary_jr"][geographic_grid.grid_id]
    hwm = loaded.datasets["neutral_wind"][geographic_grid.grid_id]
    assert hardy.geographic_grid is amps.geographic_grid is hwm.geographic_grid
    assert hardy.request_grid is amps.request_grid is hwm.request_grid
    assert (
        hardy.spec.request_coordinate_convention
        == amps.spec.request_coordinate_convention
        == hwm.spec.request_coordinate_convention
    )
