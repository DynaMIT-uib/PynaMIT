#!/usr/bin/env python3
"""Regenerate fallback inputs for the grids used by the test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.external_inputs import (
    get_conductance_inputs,
    get_jr_inputs,
    get_wind_inputs,
    native_inputs_available,
    set_input_source,
)
from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.prepared_inputs import (
    _DEFAULT_INPUT_TIME,
)

OUTPUT = Path("src/pynamit/data/fallback_inputs.json")


def flattened(array):
    return np.asarray(array).reshape(-1).tolist()


def main():
    if not native_inputs_available():
        raise SystemExit("Install native inputs first with: pip install -e '.[inputs]'")

    # These keys are the Ncs resolutions currently required by tests.
    existing = json.loads(OUTPUT.read_text())
    test_resolutions = sorted(
        set(existing["conductance"]) | set(existing["jr"]),
        key=int,
    )

    set_input_source("native")
    event_time = _DEFAULT_INPUT_TIME

    payload = {
        "version": 3,
        "coordinate_system": "GEO",
        "time": [0.0],
        "conductance": {},
        "jr": {},
    }

    with tempfile.TemporaryDirectory() as temporary_root:
        temporary_root = Path(temporary_root)

        for grid_id in test_resolutions:
            ncs = int(grid_id)

            # Only the current model grid is needed. Keeping the SH
            # truncation small avoids constructing unnecessary operators.
            simulation = Simulation(
                run_directory=temporary_root / f"ncs-{ncs}",
                Nmax=4,
                Mmax=4,
                Ncs=ncs,
                RI=EARTH_RADIUS_M + 110.0e3,
                main_field_kind="dipole",
                main_field_epoch=2020,
                t0=event_time.isoformat(sep=" "),
                enable_pfac_coupling=False,
                backend="numpy",
            )

            model_lat = np.asarray(simulation.geometry.model_grid.lat)
            model_lon = np.asarray(simulation.geometry.model_grid.lon)

            # Provider adapters receive geographic positions.
            geo_lat, geo_lon = simulation.geometry.main_field.model_to_geo_coordinates(
                model_lat,
                model_lon,
                event_time=event_time,
            )

            hall, pedersen, _, _ = get_conductance_inputs(
                event_time,
                geo_lat,
                geo_lon,
                time=None,
            )
            jr, _, _ = get_jr_inputs(
                event_time,
                geo_lat,
                geo_lon,
                time=None,
            )

            payload["conductance"][grid_id] = {
                "lat": flattened(geo_lat),
                "lon": flattened(geo_lon),
                "hall": flattened(hall),
                "pedersen": flattened(pedersen),
            }
            payload["jr"][grid_id] = {
                "lat": flattened(geo_lat),
                "lon": flattened(geo_lon),
                "jr": flattened(jr),
            }

            print(f"Generated test fallback for Ncs={ncs}")

    wind = get_wind_inputs(event_time, use_wind=True, time=None)
    if wind is None:
        raise RuntimeError("Native wind model returned no data.")

    u_theta, u_phi, wind_lat, wind_lon, _ = wind
    payload["wind"] = {
        "lat": flattened(wind_lat),
        "lon": flattened(wind_lon),
        "u_theta": flattened(u_theta),
        "u_phi": flattened(u_phi),
    }

    temporary_output = OUTPUT.with_suffix(".json.tmp")
    backup = OUTPUT.with_suffix(".json.bak")

    backup.write_bytes(OUTPUT.read_bytes())
    temporary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    # Basic structural validation before replacing the fixture.
    regenerated = json.loads(temporary_output.read_text())
    assert set(regenerated["conductance"]) == set(test_resolutions)
    assert set(regenerated["jr"]) == set(test_resolutions)
    assert regenerated["wind"]["lat"]

    temporary_output.replace(OUTPUT)

    print(f"Wrote:  {OUTPUT}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()