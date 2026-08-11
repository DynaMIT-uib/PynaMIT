#!/usr/bin/env python3
"""Regenerate fallback inputs from native empirical providers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.external_input_contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    ExternalInputRequest,
    FallbackCollection,
    ProviderDataset,
)
from pynamit.external_inputs import (
    FALLBACK_SCHEMA_VERSION,
    _read_fallback,
    get_conductance_inputs,
    get_jr_inputs,
    get_wind_inputs,
    native_inputs_available,
    set_input_source,
)
from pynamit.geomagnetism import decimal_year
from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.prepared_inputs import _DEFAULT_INPUT_TIME

OUTPUT = Path("src/pynamit/data/fallback_inputs.json")


@dataclass(frozen=True)
class GridSpec:
    """An exact grid cached for requested-position providers."""

    grid_id: str
    ncs: int
    main_field_kind: str
    main_field_epoch: float


EVENT_TIME = _DEFAULT_INPUT_TIME
EVENT_EPOCH = float(decimal_year(EVENT_TIME))

GRID_SPECS = (
    GridSpec("centered-dipole-event-ncs-08", 8, "dipole", EVENT_EPOCH),
    GridSpec("centered-dipole-event-ncs-12", 12, "dipole", EVENT_EPOCH),
    GridSpec("centered-dipole-event-ncs-18", 18, "dipole", EVENT_EPOCH),
    GridSpec("centered-dipole-event-ncs-20", 20, "dipole", EVENT_EPOCH),
    GridSpec("centered-dipole-event-ncs-22", 22, "dipole", EVENT_EPOCH),
    GridSpec("geographic-ncs-18", 18, "igrf", EVENT_EPOCH),
    GridSpec("geographic-ncs-20", 20, "igrf", EVENT_EPOCH),
)


def _require_source_grid(
    provider_name: str,
    request: ExternalInputRequest,
    returned_lat: np.ndarray,
    returned_lon: np.ndarray,
) -> None:
    """Require an adapter to preserve its source grid."""
    returned = request.source_grid.coordinate_contract.coordinate_identity(
        returned_lat, returned_lon
    )
    if returned != request.source_grid.coordinate_identity:
        raise RuntimeError(f"{provider_name} did not return the requested source grid.")


def _register_request_grids(grids: dict, request: ExternalInputRequest) -> None:
    """Register source and provider-interface grid views."""
    source = request.source_grid
    grids[source.grid_id] = source
    for spec in PROVIDER_SPECS.values():
        provider_grid = request.grid_for(spec)
        existing = grids.get(provider_grid.grid_id)
        if (
            existing is not None
            and existing.coordinate_identity != provider_grid.coordinate_identity
        ):
            raise RuntimeError(f"Grid ID collision for {provider_grid.grid_id!r}.")
        grids[provider_grid.grid_id] = provider_grid


def main() -> None:
    """Generate, validate, and atomically install fallback inputs."""
    if not native_inputs_available():
        raise SystemExit("Install native inputs first with: pip install -e '.[inputs]'")

    set_input_source("native")
    grids = {}
    datasets: dict[str, dict[str, ProviderDataset]] = {
        provider_key: {} for provider_key in PROVIDER_SPECS
    }

    with tempfile.TemporaryDirectory(prefix="pynamit-fallback-generation-") as root:
        root = Path(root)
        for spec in GRID_SPECS:
            simulation = Simulation(
                run_directory=root / spec.grid_id,
                Nmax=4,
                Mmax=4,
                Ncs=spec.ncs,
                RI=EARTH_RADIUS_M + 110.0e3,
                main_field_kind=spec.main_field_kind,
                main_field_epoch=spec.main_field_epoch,
                t0=EVENT_TIME.isoformat(sep=" "),
                enable_pfac_coupling=False,
                backend="numpy",
            )
            model_lat = np.asarray(simulation.model_grid.lat)
            model_lon = np.asarray(simulation.model_grid.lon)
            geo_lat, geo_lon = simulation.geometry.main_field.model_to_geo_coordinates(
                model_lat, model_lon, event_time=EVENT_TIME
            )
            request = ExternalInputRequest.from_model_coordinates(
                model_lat,
                model_lon,
                geographic_lat=geo_lat,
                geographic_lon=geo_lon,
                coordinate_system=simulation.geometry.main_field.horizontal_coordinate_system,
                model_epoch=simulation.geometry.main_field.epoch,
                grid_id=spec.grid_id,
                sampling_geometry={"type": "cubed_sphere", "ncs": spec.ncs},
                provenance={
                    "originating_model_frame": {
                        "horizontal_coordinate_system": (
                            simulation.geometry.main_field.horizontal_coordinate_system
                        ),
                        "main_field_kind": spec.main_field_kind,
                        "epoch": spec.main_field_epoch,
                    }
                },
            )
            _register_request_grids(grids, request)

            hall, pedersen, hall_lat, hall_lon = get_conductance_inputs(
                EVENT_TIME, request=request
            )
            jr, jr_lat, jr_lon = get_jr_inputs(EVENT_TIME, request=request)
            wind = get_wind_inputs(EVENT_TIME, request=request)
            if wind is None:
                raise RuntimeError("Native HWM14 returned no wind data.")
            u_theta, u_phi, wind_lat, wind_lon, weights = wind
            if weights is not None:
                raise RuntimeError("Requested-position HWM should not supply source-grid weights.")

            _require_source_grid("Hardy/EUV", request, hall_lat, hall_lon)
            _require_source_grid("AMPS", request, jr_lat, jr_lon)
            _require_source_grid("HWM14", request, wind_lat, wind_lon)

            source = request.source_grid
            datasets[CONDUCTANCE_PROVIDER_SPEC.key][source.grid_id] = ProviderDataset(
                spec=CONDUCTANCE_PROVIDER_SPEC,
                source_grid=source,
                request_grid=request.grid_for(CONDUCTANCE_PROVIDER_SPEC),
                values={"hall": hall, "pedersen": pedersen},
            )
            datasets[BOUNDARY_JR_PROVIDER_SPEC.key][source.grid_id] = ProviderDataset(
                spec=BOUNDARY_JR_PROVIDER_SPEC,
                source_grid=source,
                request_grid=request.grid_for(BOUNDARY_JR_PROVIDER_SPEC),
                values={"jr": jr},
            )
            datasets[NEUTRAL_WIND_PROVIDER_SPEC.key][source.grid_id] = ProviderDataset(
                spec=NEUTRAL_WIND_PROVIDER_SPEC,
                source_grid=source,
                request_grid=request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC),
                values={"u_theta": u_theta, "u_phi": u_phi},
            )
            print(f"Generated {source.grid_id}: {source.size} source positions.", flush=True)

    collection = FallbackCollection(
        version=FALLBACK_SCHEMA_VERSION,
        event_time=EVENT_TIME.isoformat(),
        time=np.array([0.0]),
        grids=grids,
        providers=PROVIDER_SPECS,
        datasets=datasets,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT.with_suffix(".json.tmp")
    backup = OUTPUT.with_suffix(".json.bak")
    collection.write(temporary_output)

    loaded = _read_fallback(temporary_output)
    for source_grid_id in datasets[CONDUCTANCE_PROVIDER_SPEC.key]:
        conductance = loaded.datasets[CONDUCTANCE_PROVIDER_SPEC.key][source_grid_id]
        boundary_jr = loaded.datasets[BOUNDARY_JR_PROVIDER_SPEC.key][source_grid_id]
        wind = loaded.datasets[NEUTRAL_WIND_PROVIDER_SPEC.key][source_grid_id]
        if not (conductance.source_grid is boundary_jr.source_grid is wind.source_grid):
            raise RuntimeError(f"Source grid {source_grid_id!r} was not structurally shared.")
        if not (conductance.request_grid is boundary_jr.request_grid is wind.request_grid):
            raise RuntimeError(f"Provider request grid for {source_grid_id!r} was not shared.")

    if OUTPUT.exists():
        backup.write_bytes(OUTPUT.read_bytes())
    temporary_output.replace(OUTPUT)

    print(f"Wrote:  {OUTPUT}")
    if backup.exists():
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
