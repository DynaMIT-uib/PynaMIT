"""Prepare inputs for PynaMIT's fixed empirical example."""

from __future__ import annotations

import datetime as _datetime
from pathlib import Path

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.external_input_contracts import ExternalInputRequest
from pynamit.external_inputs import (
    get_conductance_inputs,
    get_input_source,
    get_jr_inputs,
    get_wind_inputs,
)
from pynamit.simulation.api import InputPreparation
from pynamit.simulation.input_manifest import clear_prepared_input_package
from pynamit.storage import ArtifactStore

_EXAMPLE_EVENT_TIME = _datetime.datetime(2001, 5, 12, 21, 45)


def _wind_to_model_coordinates(main_field, u_theta, u_phi, lat, lon, *, event_time=None):
    """Rotate spherical-GEO wind samples into model coordinates."""
    u_theta, u_phi = np.broadcast_arrays(np.asarray(u_theta), np.asarray(u_phi))
    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    if lat.size != lon.size or u_theta.shape[-1] != lat.size:
        raise ValueError("Wind coordinates must match the final wind-sample dimension.")

    model_lat, model_lon = main_field.geo_to_model_coordinates(lat, lon, event_time=event_time)
    vector_lat = np.broadcast_to(lat, u_theta.shape)
    vector_lon = np.broadcast_to(lon, u_theta.shape)
    _, _, model_east, model_north = main_field.geo_to_model_coordinates(
        vector_lat, vector_lon, east=u_phi, north=-u_theta, event_time=event_time
    )
    return -model_north, model_east, model_lat, model_lon


def _require_source_grid(provider_name, request, returned_lat, returned_lon):
    """Require an adapter to preserve the source-grid identity."""
    returned_identity = request.source_grid.coordinate_contract.coordinate_identity(
        returned_lat, returned_lon
    )
    if returned_identity != request.source_grid.coordinate_identity:
        returned_size = np.asarray(returned_lat).size
        raise ValueError(
            f"{provider_name} must return values on the shared "
            f"{request.source_grid.coordinate_contract.coordinate_system} "
            f"source grid; expected {request.source_grid.size} ordered "
            f"points but received a different {returned_size}-point grid."
        )


def prepare_example_inputs(
    input_directory=None,
    *,
    final_time=100,
    Nmax=20,
    Mmax=20,
    Ncs=30,
    main_field_kind="dipole",
    main_field_epoch=None,
    main_field_B0=None,
    use_wind=False,
    use_Q_eff=False,
    use_boundary_jr=True,
    boundary_jr_projection_basis=None,
    boundary_Br_projection_basis=None,
    conductance_projection_basis=None,
    u_projection_basis=None,
    Q_eff_projection_basis=None,
    boundary_jr_lambda=None,
    conductance_lambda=None,
    u_lambda=None,
    Q_eff_lambda=None,
    multi_data=False,
    artifact_storage="auto",
    horizontal_basis_kind="SH",
    area_weighted_least_squares=False,
):
    """Prepare the package's fixed empirical example without running it.

    The example uses the 12 May 2001 event and the configured external
    input providers. It writes projected datasets and their manifest to
    ``input_directory`` and returns their :class:`InputPreparation`.
    """
    if use_Q_eff and not use_wind:
        raise ValueError("use_Q_eff=True requires use_wind=True in prepare_example_inputs.")

    event_time = _EXAMPLE_EVENT_TIME
    input_directory = (
        ArtifactStore.create_temporary_directory("simulation/inputs")
        if input_directory is None
        else str(Path(input_directory).resolve())
    )
    clear_prepared_input_package(input_directory, artifact_storage=artifact_storage)
    preparation = InputPreparation(
        input_directory=input_directory,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=EARTH_RADIUS_M + 110.0e3,
        main_field_kind=main_field_kind,
        main_field_epoch=main_field_epoch,
        main_field_B0=main_field_B0,
        t0=event_time.isoformat(sep=" "),
        boundary_jr_projection_basis=boundary_jr_projection_basis,
        boundary_Br_projection_basis=boundary_Br_projection_basis,
        conductance_projection_basis=conductance_projection_basis,
        u_projection_basis=u_projection_basis,
        Q_eff_projection_basis=Q_eff_projection_basis,
        horizontal_basis_kind=horizontal_basis_kind,
        area_weighted_least_squares=area_weighted_least_squares,
        artifact_storage=artifact_storage,
    )

    time = np.linspace(0, final_time, 4) if multi_data else None
    model_lat = preparation.model_grid.lat
    model_lon = preparation.model_grid.lon
    geo_lat, geo_lon = preparation.geometry.main_field.model_to_geo_coordinates(
        model_lat, model_lon, event_time=event_time
    )
    external_request = ExternalInputRequest.from_model_coordinates(
        model_lat,
        model_lon,
        geographic_lat=geo_lat,
        geographic_lon=geo_lon,
        coordinate_system=preparation.geometry.main_field.horizontal_coordinate_system,
        model_epoch=preparation.geometry.main_field.epoch,
        grid_id="prepared-input-source",
        sampling_geometry={"type": "simulation_model_grid"},
        provenance={
            "originating_model_frame": {
                "horizontal_coordinate_system": (
                    preparation.geometry.main_field.horizontal_coordinate_system
                ),
                "main_field_kind": preparation.geometry.main_field.kind,
                "epoch": preparation.geometry.main_field.epoch,
            }
        },
    )

    hall, pedersen, hall_lat, hall_lon = get_conductance_inputs(
        event_time, time=time, request=external_request
    )
    _require_source_grid("Conductance adapter", external_request, hall_lat, hall_lon)
    preparation.set_conductance(
        hall, pedersen, lat=model_lat, lon=model_lon, reg_lambda=conductance_lambda, time=time
    )

    if use_boundary_jr:
        boundary_jr, jr_lat, jr_lon = get_jr_inputs(
            event_time, time=time, request=external_request
        )
        _require_source_grid("AMPS boundary-jr adapter", external_request, jr_lat, jr_lon)
        preparation.set_boundary_jr(
            boundary_jr, lat=model_lat, lon=model_lon, reg_lambda=boundary_jr_lambda, time=time
        )

    wind_inputs = get_wind_inputs(
        event_time, use_wind=use_wind, time=time, request=external_request
    )
    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs
        _require_source_grid("HWM neutral-wind adapter", external_request, u_lat, u_lon)
        u_theta, u_phi, _, _ = _wind_to_model_coordinates(
            preparation.geometry.main_field, u_theta, u_phi, u_lat, u_lon, event_time=event_time
        )
        if use_Q_eff:
            preparation.set_Q_eff_from_neutral_wind(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=model_lat,
                lon=model_lon,
                sqrt_weights=weights,
                wind_reg_lambda=u_lambda,
                Q_eff_reg_lambda=Q_eff_lambda,
                time=time,
            )
        else:
            preparation.set_neutral_wind(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=model_lat,
                lon=model_lon,
                sqrt_weights=weights,
                reg_lambda=u_lambda,
                time=time,
            )

    notes = []
    if use_Q_eff:
        notes.append(
            "Q_eff was derived from neutral wind through the current model operators; "
            "prefer E_neutral_wind for externally prepared conductivity-weighted winds."
        )
    preparation.write_manifest(
        source="pynamit.default_external_inputs",
        notes=notes,
        metadata={
            "external_input_source": get_input_source(),
            "multi_data": bool(multi_data),
            "projection_regularization": {
                "boundary_jr_lambda": boundary_jr_lambda,
                "conductance_lambda": conductance_lambda,
                "u_lambda": u_lambda,
                "Q_eff_lambda": Q_eff_lambda,
            },
        },
    )
    return preparation


__all__ = ["prepare_example_inputs"]
