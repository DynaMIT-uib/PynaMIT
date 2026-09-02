"""Shared HDF5 contract for prepared MAGE forcing."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.coordinates import parse_utc_datetime

# Prepared-forcing identity and coordinate conventions
IONOSPHERE_RADIUS_M = 6.5e6
MAGE_FORCING_KIND = "pynamit_mage_forcing"
MAGE_FORCING_VERSION = 13
MAGE_TIME_AXIS = "tiegcm_mtime_nominal"
MAGE_SOURCE_TIME_TOLERANCE_SECONDS = 0.1

# Physical choices recorded in every prepared file
CONDUCTANCE_FLOOR_MODEL = "global_hard_minimum"
PEDERSEN_CONDUCTANCE_FLOOR_S = 2.0
HALL_CONDUCTANCE_FLOOR_S = 1.0
TIEGCM_DYNAMO_BOTTOM_ILEV = -8.5
TIEGCM_DYNAMO_REFERENCE_HEIGHT_M = 90_000.0
TIEGCM_PEDERSEN_LOWER_SCALE_M = 5_000.0
TIEGCM_HALL_LOWER_SCALE_M = 3_000.0

IONOSPHERE_DATASETS = ("jr", "SH", "SP", "u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi")
BOUNDARY_DATASETS = ("delta_Br",)
STATIC_DATASETS = (
    "time",
    "gamera_source_time",
    "remix_source_time",
    "gamera_time_offset_seconds",
    "remix_time_offset_seconds",
    "ionosphere_lat",
    "ionosphere_lon",
    "boundary_lat",
    "boundary_lon",
    "boundary_radius",
    "boundary_solid_angle",
)
REQUIRED_ATTRIBUTES = (
    "time_axis",
    "source_time_tolerance_seconds",
    "fac_convention",
    "radial_current_convention",
    "remix_fac_interpolation",
    "gamera_boundary_interpolation",
    "gamera_source_coordinate_system",
    "gamera_sm_transform_time_convention",
    "coordinate_system",
    "longitude_convention",
    "gamera_mag_m0_nT",
    "main_field_B0_T",
    "main_field_B0_reference_radius_m",
    "gamera_internal_dipole_moment_axis",
    "gamera_internal_magnetic_north_axis",
    "gamera_background_reference",
    "tiegcm_source_coordinate_system",
    "tiegcm_conductance_integration",
    "tiegcm_dynamo_bottom_ilev",
    "tiegcm_dynamo_reference_height_m",
    "tiegcm_pedersen_lower_scale_m",
    "tiegcm_hall_lower_scale_m",
    "conductance_floor_model",
    "pedersen_conductance_floor_S",
    "hall_conductance_floor_S",
    "remix_grid_equatorward_sm_latitude_deg",
    "ionosphere_radius_m",
)
DATASET_UNITS = {
    "gamera_time_offset_seconds": "s",
    "remix_time_offset_seconds": "s",
    "jr": "uA m-2",
    "SH": "S",
    "SP": "S",
    "u_p_theta": "m s-1",
    "u_p_phi": "m s-1",
    "u_h_theta": "m s-1",
    "u_h_phi": "m s-1",
    "delta_Br": "nT",
    "ionosphere_lat": "degree",
    "ionosphere_lon": "degree",
    "boundary_lat": "degree",
    "boundary_lon": "degree",
    "boundary_radius": "m",
    "boundary_solid_angle": "sr",
}


def forcing_times(raw_times: Any) -> tuple[list[dt.datetime], np.ndarray]:
    """Return UTC timestamps and seconds relative to the first entry."""
    timestamps = [parse_utc_datetime(value) for value in raw_times]
    if not timestamps:
        raise ValueError("Prepared forcing time dataset is empty.")
    relative_seconds = np.array(
        [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps], dtype=float
    )
    if np.any(~np.isfinite(relative_seconds)):
        raise ValueError("Prepared forcing time dataset produced non-finite relative seconds.")
    if np.any(np.diff(relative_seconds) <= 0.0):
        raise ValueError("Prepared forcing time dataset must be strictly increasing.")
    return timestamps, relative_seconds


def _matching_grid_shape(h5_file: Any, names: tuple[str, ...], label: str) -> tuple[int, ...]:
    shape = h5_file[names[0]].shape
    if not shape or any(h5_file[name].shape != shape for name in names[1:]):
        joined_names = "/".join(names)
        raise RuntimeError(
            f"Prepared forcing {joined_names} {label} grids must have the same non-empty shape."
        )
    return shape


def _validate_time_series_shapes(
    h5_file: Any, names: tuple[str, ...], n_steps: int, spatial_shape: tuple[int, ...]
) -> None:
    expected = (n_steps, *spatial_shape)
    for name in names:
        if h5_file[name].shape != expected:
            raise RuntimeError(
                f"Prepared forcing dataset {name!r} has shape {h5_file[name].shape}; "
                f"expected {expected}."
            )


def _validate_time_axis(h5_file: Any, n_steps: int) -> None:
    """Validate the nominal clock and exact source-time provenance."""
    timestamp_names = ("time", "gamera_source_time", "remix_source_time")
    for name in timestamp_names:
        if h5_file[name].shape != (n_steps,):
            raise RuntimeError(
                f"Prepared forcing dataset {name!r} has shape {h5_file[name].shape}; "
                f"expected {(n_steps,)}."
            )

    try:
        nominal_times, nominal_seconds = forcing_times(h5_file["time"][:])
        gamera_times, _ = forcing_times(h5_file["gamera_source_time"][:])
        remix_times, _ = forcing_times(h5_file["remix_source_time"][:])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Prepared forcing time metadata is invalid: {exc}") from exc

    nominal_intervals = np.diff(nominal_seconds)
    if nominal_intervals.size > 1 and not np.allclose(
        nominal_intervals, nominal_intervals[0], rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("Prepared forcing nominal time axis must have a uniform cadence.")

    for source, times in {"gamera": gamera_times, "remix": remix_times}.items():
        offsets = np.array(
            [
                (source_time - nominal_time).total_seconds()
                for source_time, nominal_time in zip(times, nominal_times, strict=True)
            ],
            dtype=float,
        )
        dataset_name = f"{source}_time_offset_seconds"
        stored_offsets = np.asarray(h5_file[dataset_name][:], dtype=float)
        if stored_offsets.shape != (n_steps,) or np.any(~np.isfinite(stored_offsets)):
            raise RuntimeError(
                f"Prepared forcing dataset {dataset_name!r} must contain one finite value "
                "per forcing step."
            )
        if not np.allclose(stored_offsets, offsets, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"Prepared forcing dataset {dataset_name!r} does not match its timestamps."
            )
        if np.any(np.abs(offsets) > MAGE_SOURCE_TIME_TOLERANCE_SECONDS):
            raise RuntimeError(
                f"Prepared forcing {source.upper()} source times exceed the allowed "
                f"{MAGE_SOURCE_TIME_TOLERANCE_SECONDS:g} s offset from the nominal axis."
            )


def _validate_conductance_floor(h5_file: Any, n_steps: int) -> None:
    tolerance = 8.0 * np.finfo(np.float32).eps
    for step in range(n_steps):
        pedersen = np.asarray(h5_file["SP"][step], dtype=float)
        hall = np.asarray(h5_file["SH"][step], dtype=float)
        if np.any(pedersen < PEDERSEN_CONDUCTANCE_FLOOR_S - tolerance):
            raise RuntimeError(
                f"Prepared Pedersen conductance violates the global hard floor at step {step}."
            )
        if np.any(hall < HALL_CONDUCTANCE_FLOOR_S - tolerance):
            raise RuntimeError(
                f"Prepared Hall conductance violates the global hard floor at step {step}."
            )


def validate_prepared_forcing(h5_file: Any) -> None:
    """Validate a prepared file before projection or diagnostics."""
    if h5_file.attrs.get("kind") != MAGE_FORCING_KIND:
        raise RuntimeError(
            "Prepared forcing has no supported MAGE forcing contract. "
            "Regenerate it with scripts/simulation/mage_prepare.py."
        )
    if h5_file.attrs.get("version") != MAGE_FORCING_VERSION:
        raise RuntimeError(
            f"Prepared forcing version {h5_file.attrs.get('version')!r} is unsupported; "
            f"expected {MAGE_FORCING_VERSION}."
        )
    complete = h5_file.attrs.get("complete", False)
    if not isinstance(complete, (bool, np.bool_)) or not bool(complete):
        raise RuntimeError("Prepared forcing is incomplete; regenerate it before projection.")

    required_datasets = (*STATIC_DATASETS, *IONOSPHERE_DATASETS, *BOUNDARY_DATASETS)
    missing_datasets = [name for name in required_datasets if name not in h5_file]
    missing_attributes = [name for name in REQUIRED_ATTRIBUTES if name not in h5_file.attrs]
    if missing_datasets or missing_attributes:
        details = []
        if missing_datasets:
            details.append(f"datasets {missing_datasets}")
        if missing_attributes:
            details.append(f"attributes {missing_attributes}")
        raise RuntimeError("Prepared forcing is missing required " + " and ".join(details) + ".")

    if h5_file.attrs["time_axis"] != MAGE_TIME_AXIS:
        raise RuntimeError("MAGE projection requires the nominal TIEGCM mtime forcing axis.")
    source_time_tolerance = float(h5_file.attrs["source_time_tolerance_seconds"])
    if not np.isfinite(source_time_tolerance) or not np.isclose(
        source_time_tolerance, MAGE_SOURCE_TIME_TOLERANCE_SECONDS, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "Prepared forcing uses an incompatible source-time alignment tolerance."
        )
    if h5_file.attrs["gamera_source_coordinate_system"] != "SM":
        raise RuntimeError("MAGE preparation requires GAMERA source coordinates in SM.")
    if h5_file.attrs["gamera_sm_transform_time_convention"] != "kaiju_mjdrecalc_nearest_second":
        raise RuntimeError(
            "MAGE projection requires Kaiju's nearest-second SM transform convention."
        )
    if h5_file.attrs["coordinate_system"] != "GEO":
        raise RuntimeError("MAGE projection requires Earth-fixed geographic coordinates.")
    if h5_file.attrs["longitude_convention"] != "east_positive_degrees":
        raise RuntimeError("MAGE projection requires east-positive geographic longitudes.")
    if h5_file.attrs["fac_convention"] != "upward":
        raise RuntimeError("MAGE projection requires an upward-positive REMIX FAC source.")
    if h5_file.attrs["radial_current_convention"] != "outward":
        raise RuntimeError("MAGE projection requires outward-positive prepared radial current.")
    if h5_file.attrs["tiegcm_source_coordinate_system"] != "geographic":
        raise RuntimeError("MAGE projection requires geographic TIEGCM forcing.")
    if (
        h5_file.attrs["tiegcm_conductance_integration"]
        != "radial_geometric_height_with_lower_dynamo_extension"
    ):
        raise RuntimeError(
            "MAGE projection requires radial TIEGCM conductance with its lower dynamo extension."
        )

    expected_dynamo_parameters = {
        "tiegcm_dynamo_bottom_ilev": TIEGCM_DYNAMO_BOTTOM_ILEV,
        "tiegcm_dynamo_reference_height_m": TIEGCM_DYNAMO_REFERENCE_HEIGHT_M,
        "tiegcm_pedersen_lower_scale_m": TIEGCM_PEDERSEN_LOWER_SCALE_M,
        "tiegcm_hall_lower_scale_m": TIEGCM_HALL_LOWER_SCALE_M,
    }
    invalid_dynamo_parameters = [
        name
        for name, expected in expected_dynamo_parameters.items()
        if not np.isclose(float(h5_file.attrs[name]), expected, rtol=0.0, atol=1e-12)
    ]
    if invalid_dynamo_parameters:
        raise RuntimeError(
            "Prepared forcing uses incompatible TIEGCM lower-dynamo parameters: "
            f"{invalid_dynamo_parameters}."
        )
    if h5_file.attrs["conductance_floor_model"] != CONDUCTANCE_FLOOR_MODEL:
        raise RuntimeError("Prepared forcing does not use the required global conductance floor.")
    expected_floors = {
        "pedersen_conductance_floor_S": PEDERSEN_CONDUCTANCE_FLOOR_S,
        "hall_conductance_floor_S": HALL_CONDUCTANCE_FLOOR_S,
    }
    invalid_floors = [
        name
        for name, expected in expected_floors.items()
        if not np.isclose(float(h5_file.attrs[name]), expected, rtol=0.0, atol=1e-12)
    ]
    if invalid_floors:
        raise RuntimeError(
            f"Prepared forcing uses incompatible conductance floors: {invalid_floors}."
        )

    equatorward_latitude = float(h5_file.attrs["remix_grid_equatorward_sm_latitude_deg"])
    if not np.isfinite(equatorward_latitude) or not 0.0 < equatorward_latitude < 90.0:
        raise RuntimeError(
            "Prepared forcing ReMIX grid boundary must be between 0 and 90 degrees SM latitude."
        )
    if h5_file.attrs["gamera_background_reference"] != "cell_volume_average_split_B0":
        raise RuntimeError(
            "MAGE projection requires GAMERA total B minus its matching volume-averaged B0."
        )
    if h5_file.attrs["remix_fac_interpolation"] != "kaiju_native_periodic":
        raise RuntimeError("MAGE projection requires Kaiju-native ReMIX FAC interpolation.")
    if (
        h5_file.attrs["gamera_boundary_interpolation"]
        != "gamera_native_periodic_bilinear_with_polar_mean"
    ):
        raise RuntimeError(
            "MAGE projection requires GAMERA-native bilinear boundary interpolation."
        )
    reference_radius = float(h5_file.attrs["main_field_B0_reference_radius_m"])
    if not np.isfinite(reference_radius) or not np.isclose(
        reference_radius, EARTH_RADIUS_M, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            "Prepared MAGE main_field_B0_T must use PynaMIT's dipole reference radius."
        )
    ionosphere_radius = float(h5_file.attrs["ionosphere_radius_m"])
    if not np.isfinite(ionosphere_radius) or not np.isclose(
        ionosphere_radius, IONOSPHERE_RADIUS_M, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            "Prepared MAGE forcing must use the 6500 km Kaiju/ReMIX ionosphere radius."
        )

    invalid_units = {
        name: h5_file[name].attrs.get("units")
        for name, expected in DATASET_UNITS.items()
        if h5_file[name].attrs.get("units") != expected
    }
    if invalid_units:
        details = ", ".join(
            f"{name}={actual!r} (expected {DATASET_UNITS[name]!r})"
            for name, actual in invalid_units.items()
        )
        raise RuntimeError(f"Prepared forcing has incompatible dataset units: {details}.")

    time_shape = h5_file["time"].shape
    if len(time_shape) != 1 or time_shape[0] == 0:
        raise RuntimeError("Prepared forcing time must be a non-empty one-dimensional dataset.")
    n_steps = time_shape[0]
    _validate_time_axis(h5_file, n_steps)
    ionosphere_shape = _matching_grid_shape(
        h5_file, ("ionosphere_lat", "ionosphere_lon"), "ionosphere"
    )
    boundary_shape = _matching_grid_shape(
        h5_file,
        ("boundary_lat", "boundary_lon", "boundary_radius", "boundary_solid_angle"),
        "boundary",
    )
    _validate_time_series_shapes(h5_file, IONOSPHERE_DATASETS, n_steps, ionosphere_shape)
    _validate_time_series_shapes(h5_file, BOUNDARY_DATASETS, n_steps, boundary_shape)
    _validate_conductance_floor(h5_file, n_steps)
