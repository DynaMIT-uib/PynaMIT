#!/usr/bin/env python3
"""Regenerate fallback inputs from native empirical providers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kompe.constants import EARTH_RADIUS_M
from tests.example_scenario import (
    AMPS_MIN_LATITUDE_DEG,
    DIPOLE_TILT_DEG,
    EVENT_TIME,
    F107_SFU,
    HWM_AP,
    IMF_BY_NT,
    IMF_BZ_NT,
    KP,
    SOLAR_WIND_SPEED_KM_S,
    STARLIGHT_CONDUCTANCE_S,
)

from pynamit import Simulation
from pynamit.external_inputs import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    ExternalInputCoordinates,
    FallbackCollection,
    ProviderSnapshot,
    get_boundary_jr_inputs,
    get_conductance_inputs,
    get_wind_inputs,
    native_inputs_available,
    set_input_source,
)
from pynamit.external_inputs.fallback_data import FALLBACK_SCHEMA_VERSION
from pynamit.external_inputs.providers import _read_fallback
from pynamit.geomagnetism import decimal_year

OUTPUT = Path("src/pynamit/data/fallback_inputs.json")


@dataclass(frozen=True)
class GridSpec:
    """An exact grid cached for requested-position providers."""

    grid_id: str
    ncs: int
    main_field_kind: str
    main_field_epoch: float


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


def _require_geographic_grid(
    provider_name: str,
    coordinates: ExternalInputCoordinates,
    returned_lat: np.ndarray,
    returned_lon: np.ndarray,
) -> None:
    """Require an adapter to preserve its geographic grid."""
    returned = coordinates.geographic_grid.coordinate_convention.coordinate_identity(
        returned_lat, returned_lon
    )
    if returned != coordinates.geographic_grid.coordinate_identity:
        raise RuntimeError(f"{provider_name} did not return the requested geographic grid.")


def _register_coordinate_grids(grids: dict, coordinates: ExternalInputCoordinates) -> None:
    """Register geographic and provider-interface grid views."""
    geographic_grid = coordinates.geographic_grid
    grids[geographic_grid.grid_id] = geographic_grid
    for spec in PROVIDER_SPECS.values():
        provider_grid = coordinates.sample_grid(spec.request_coordinate_convention)
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
        raise SystemExit(
            "Install Lompe and PyAMPS from requirements/pip-input-models.txt, then "
            "install pyHWM2014 with --no-deps as described in README.md."
        )

    set_input_source("native")
    grids = {}
    datasets: dict[str, dict[str, ProviderSnapshot]] = {
        provider_key: {} for provider_key in PROVIDER_SPECS
    }

    with tempfile.TemporaryDirectory(prefix="pynamit-fallback-generation-") as root:
        root = Path(root)
        for spec in GRID_SPECS:
            simulation = Simulation(
                simulation_directory=root / spec.grid_id,
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
            coordinates = ExternalInputCoordinates.from_model_coordinates(
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
            _register_coordinate_grids(grids, coordinates)

            pedersen, hall, conductance_lat, conductance_lon = get_conductance_inputs(
                EVENT_TIME, coordinates=coordinates, kp=KP, starlight=STARLIGHT_CONDUCTANCE_S
            )
            jr, jr_lat, jr_lon = get_boundary_jr_inputs(
                EVENT_TIME,
                coordinates=coordinates,
                v=SOLAR_WIND_SPEED_KM_S,
                By=IMF_BY_NT,
                Bz=IMF_BZ_NT,
                tilt=DIPOLE_TILT_DEG,
                f107=F107_SFU,
                minlat=AMPS_MIN_LATITUDE_DEG,
            )
            wind = get_wind_inputs(EVENT_TIME, coordinates=coordinates, ap=HWM_AP)
            u_theta, u_phi, wind_lat, wind_lon, weights = wind
            if weights is not None:
                raise RuntimeError(
                    "Requested-position HWM should not supply geographic-grid weights."
                )

            _require_geographic_grid("Hardy/EUV", coordinates, conductance_lat, conductance_lon)
            _require_geographic_grid("AMPS", coordinates, jr_lat, jr_lon)
            _require_geographic_grid("HWM14", coordinates, wind_lat, wind_lon)

            geographic_grid = coordinates.geographic_grid
            datasets[CONDUCTANCE_PROVIDER_SPEC.key][geographic_grid.grid_id] = ProviderSnapshot(
                spec=CONDUCTANCE_PROVIDER_SPEC,
                geographic_grid=geographic_grid,
                request_grid=coordinates.sample_grid(
                    CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
                ),
                values={"hall": hall, "pedersen": pedersen},
            )
            datasets[BOUNDARY_JR_PROVIDER_SPEC.key][geographic_grid.grid_id] = ProviderSnapshot(
                spec=BOUNDARY_JR_PROVIDER_SPEC,
                geographic_grid=geographic_grid,
                request_grid=coordinates.sample_grid(
                    BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention
                ),
                values={"jr": jr},
            )
            datasets[NEUTRAL_WIND_PROVIDER_SPEC.key][geographic_grid.grid_id] = ProviderSnapshot(
                spec=NEUTRAL_WIND_PROVIDER_SPEC,
                geographic_grid=geographic_grid,
                request_grid=coordinates.sample_grid(
                    NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention
                ),
                values={"u_theta": u_theta, "u_phi": u_phi},
            )
            print(
                f"Generated {geographic_grid.grid_id}: "
                f"{geographic_grid.size} geographic positions.",
                flush=True,
            )

    collection = FallbackCollection(
        version=FALLBACK_SCHEMA_VERSION,
        event_time=EVENT_TIME.isoformat(),
        conditions={
            CONDUCTANCE_PROVIDER_SPEC.key: {"kp": KP, "starlight": STARLIGHT_CONDUCTANCE_S},
            BOUNDARY_JR_PROVIDER_SPEC.key: {
                "v": SOLAR_WIND_SPEED_KM_S,
                "By": IMF_BY_NT,
                "Bz": IMF_BZ_NT,
                "tilt": DIPOLE_TILT_DEG,
                "f107": F107_SFU,
                "minlat": AMPS_MIN_LATITUDE_DEG,
            },
            NEUTRAL_WIND_PROVIDER_SPEC.key: {"ap": HWM_AP},
        },
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
    for geographic_grid_id in datasets[CONDUCTANCE_PROVIDER_SPEC.key]:
        conductance = loaded.datasets[CONDUCTANCE_PROVIDER_SPEC.key][geographic_grid_id]
        boundary_jr = loaded.datasets[BOUNDARY_JR_PROVIDER_SPEC.key][geographic_grid_id]
        wind = loaded.datasets[NEUTRAL_WIND_PROVIDER_SPEC.key][geographic_grid_id]
        if not (
            conductance.geographic_grid is boundary_jr.geographic_grid is wind.geographic_grid
        ):
            raise RuntimeError(
                f"Geographic grid {geographic_grid_id!r} was not structurally shared."
            )
        if not (conductance.request_grid is boundary_jr.request_grid is wind.request_grid):
            raise RuntimeError(f"Provider request grid for {geographic_grid_id!r} was not shared.")

    if OUTPUT.exists():
        backup.write_bytes(OUTPUT.read_bytes())
    temporary_output.replace(OUTPUT)

    print(f"Wrote:  {OUTPUT}")
    if backup.exists():
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
