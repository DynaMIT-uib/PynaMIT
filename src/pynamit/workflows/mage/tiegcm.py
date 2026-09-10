"""Read and vertically integrate TIEGCM forcing for MAGE."""

from __future__ import annotations

import calendar as calendar_module
import datetime as dt
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.workflows.mage.prepared_forcing import (
    HALL_CONDUCTANCE_FLOOR_S,
    PEDERSEN_CONDUCTANCE_FLOOR_S,
    TIEGCM_DYNAMO_BOTTOM_ILEV,
    TIEGCM_DYNAMO_REFERENCE_HEIGHT_M,
    TIEGCM_HALL_LOWER_SCALE_M,
    TIEGCM_PEDERSEN_LOWER_SCALE_M,
)

TIEGCM_FILL_THRESHOLD = 1e30


def _resolve_tiegcm_path(gamera_directory: Path, explicit_path: str | Path | None) -> Path:
    """Resolve the TIEGCM NetCDF path."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TIEGCM NetCDF does not exist: {path}")
        return path

    matches = sorted(gamera_directory.glob("*sech_tie*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a '*sech_tie*.nc' TIEGCM file in {gamera_directory}"
        )
    if len(matches) > 1:
        formatted = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Found multiple TIEGCM files; set tiegcm_path explicitly:\n  {formatted}"
        )
    return matches[0]


def _read_tiegcm_step(dataset: Any, name: str, step: int) -> np.ndarray:
    """Read a TIEGCM slice with missing values normalized to NaN."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="WARNING: missing_value not used since it.*", category=UserWarning
        )
        values = dataset.variables[name][step]
    if np.ma.isMaskedArray(values):
        values = np.ma.asarray(values, dtype=float).filled(np.nan)
    array = np.asarray(values, dtype=float)
    array[np.abs(array) > TIEGCM_FILL_THRESHOLD] = np.nan
    return array


def _column_conductance_and_winds(
    layer_conductance: np.ndarray,
    lower_conductance: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate conductance and wind moments over a radial column.

    The returned winds are the compact representation
    ``integral(sigma * u * dr) / integral(sigma * dr)``. Pedersen and
    Hall columns are handled separately because their altitude profiles
    differ. Multiplying each returned wind by its conductance therefore
    reconstructs the corresponding Appendix-A wind moment without
    storing four additional current-like arrays.
    """
    conductance = np.sum(layer_conductance, axis=0) + lower_conductance
    # TIEGCM holds the lowest resolved wind constant through its lower
    # extension. Its moment is exact without six additional wind arrays.
    east_num = np.sum(layer_conductance * wind_east, axis=0) + (lower_conductance * wind_east[0])
    north_num = np.sum(layer_conductance * wind_north, axis=0) + (
        lower_conductance * wind_north[0]
    )
    east = np.divide(east_num, conductance, out=np.zeros_like(east_num), where=conductance > 0.0)
    north = np.divide(
        north_num, conductance, out=np.zeros_like(north_num), where=conductance > 0.0
    )
    return conductance, east.astype(np.float32), north.astype(np.float32)


def _lower_dynamo_layer_count(interface_levels: np.ndarray) -> int:
    """Count TIEGCM layers missing below the saved history."""
    interface_levels = np.asarray(interface_levels, dtype=float)
    if (
        interface_levels.ndim != 1
        or interface_levels.size < 2
        or np.any(~np.isfinite(interface_levels))
    ):
        raise RuntimeError("TIEGCM ilev must be a finite one-dimensional grid.")
    spacing = np.diff(interface_levels)
    if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise RuntimeError("TIEGCM ilev must be strictly increasing and uniform.")

    missing_layers = (interface_levels[0] - TIEGCM_DYNAMO_BOTTOM_ILEV) / spacing[0]
    rounded_layers = int(round(float(missing_layers)))
    if rounded_layers < 1 or not np.isclose(missing_layers, rounded_layers, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "TIEGCM's saved lower interface must lie an integer number of layers "
            f"above the dynamo ilev {TIEGCM_DYNAMO_BOTTOM_ILEV:g}."
        )
    return rounded_layers


def _lower_dynamo_conductances(
    interface_levels: np.ndarray,
    geopotential_height_m: np.ndarray,
    geometric_height_m: np.ndarray,
    bottom_pedersen_conductivity: np.ndarray,
    bottom_hall_conductivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce TIEGCM's lower dynamo conductivity continuation."""
    n_lower = _lower_dynamo_layer_count(interface_levels)
    horizontal_shape = geopotential_height_m.shape[1:]
    if (
        geopotential_height_m.shape != geometric_height_m.shape
        or geopotential_height_m.ndim < 1
        or geopotential_height_m.shape[0] < 2
        or bottom_pedersen_conductivity.shape != horizontal_shape
        or bottom_hall_conductivity.shape != horizontal_shape
    ):
        raise RuntimeError(
            "TIEGCM Z/ZG histories and bottom conductivities use incompatible shapes."
        )

    bottom_geopotential = geopotential_height_m[0]
    bottom_geometric = geometric_height_m[0]
    if np.any(bottom_geopotential <= TIEGCM_DYNAMO_REFERENCE_HEIGHT_M) or np.any(
        bottom_geometric <= TIEGCM_DYNAMO_REFERENCE_HEIGHT_M
    ):
        raise RuntimeError("TIEGCM's saved lower interface must lie above 90 km.")

    fraction_shape = (n_lower + 1, *((1,) * (geopotential_height_m.ndim - 1)))
    fractions = np.linspace(0.0, 1.0, n_lower + 1).reshape(fraction_shape)
    lower_geopotential_interfaces = TIEGCM_DYNAMO_REFERENCE_HEIGHT_M + fractions * (
        bottom_geopotential - TIEGCM_DYNAMO_REFERENCE_HEIGHT_M
    )
    lower_geometric_interfaces = TIEGCM_DYNAMO_REFERENCE_HEIGHT_M + fractions * (
        bottom_geometric - TIEGCM_DYNAMO_REFERENCE_HEIGHT_M
    )
    lower_midpoint_height = 0.5 * (
        lower_geopotential_interfaces[:-1] + lower_geopotential_interfaces[1:]
    )
    first_saved_midpoint_height = 0.5 * (geopotential_height_m[0] + geopotential_height_m[1])
    lower_dz = np.diff(lower_geometric_interfaces, axis=0)

    pedersen_extension = bottom_pedersen_conductivity * np.exp(
        (lower_midpoint_height - first_saved_midpoint_height) / TIEGCM_PEDERSEN_LOWER_SCALE_M
    )
    hall_extension = bottom_hall_conductivity * np.exp(
        (lower_midpoint_height - first_saved_midpoint_height) / TIEGCM_HALL_LOWER_SCALE_M
    )
    return (
        np.sum(pedersen_extension * lower_dz, axis=0),
        np.sum(hall_extension * lower_dz, axis=0),
    )


def _integrate_tiegcm_step(dataset: Any, step: int) -> dict[str, np.ndarray]:
    """Height-integrate conductivity and neutral-wind moments."""
    pedersen_conductivity = _read_tiegcm_step(dataset, "SIGMA_PED", step)
    hall_conductivity = _read_tiegcm_step(dataset, "SIGMA_HAL", step)
    geopotential_height_m = _read_tiegcm_step(dataset, "Z", step) / 100.0
    geometric_height_m = _read_tiegcm_step(dataset, "ZG", step) / 100.0
    wind_east = _read_tiegcm_step(dataset, "UN", step) * 1e-2
    wind_north = _read_tiegcm_step(dataset, "VN", step) * 1e-2
    interface_levels = np.asarray(dataset.variables["ilev"][:], dtype=float)

    field_shapes = {
        pedersen_conductivity.shape,
        hall_conductivity.shape,
        geopotential_height_m.shape,
        geometric_height_m.shape,
        wind_east.shape,
        wind_north.shape,
    }
    if (
        len(field_shapes) != 1
        or pedersen_conductivity.ndim < 1
        or pedersen_conductivity.shape[0] < 2
    ):
        raise RuntimeError(
            "TIEGCM conductivity, Z/ZG, and wind histories must have matching "
            "shapes with at least two vertical levels."
        )

    # TIEGCM's first n-1 ``lev`` entries are centered between the n
    # ``ilev`` heights. Its final lev entry has no upper interface and
    # is a fill-only history in the MAGE file, so it is deliberately
    # excluded from both conductance and wind moments.
    dz = np.diff(geometric_height_m, axis=0)
    if np.any(~np.isfinite(geopotential_height_m)) or np.any(~np.isfinite(geometric_height_m)):
        raise RuntimeError("TIEGCM Z/ZG contains missing or non-finite interface heights.")
    if np.any(dz <= 0.0):
        raise RuntimeError("TIEGCM geometric height must increase with vertical level.")
    pedersen_layers = pedersen_conductivity[:-1]
    hall_layers = hall_conductivity[:-1]
    wind_east = wind_east[:-1]
    wind_north = wind_north[:-1]
    layer_fields = {
        "SIGMA_PED": pedersen_layers,
        "SIGMA_HAL": hall_layers,
        "UN": wind_east,
        "VN": wind_north,
    }
    invalid = [name for name, values in layer_fields.items() if np.any(~np.isfinite(values))]
    if invalid:
        raise RuntimeError(
            f"TIEGCM contains missing or non-finite values in integrated layers: {invalid}."
        )
    if np.any(pedersen_layers < 0.0) or np.any(hall_layers < 0.0):
        raise RuntimeError("TIEGCM Pedersen/Hall conductivity must be non-negative.")

    lower_pedersen, lower_hall = _lower_dynamo_conductances(
        interface_levels,
        geopotential_height_m,
        geometric_height_m,
        pedersen_layers[0],
        hall_layers[0],
    )

    pedersen_conductance, u_p_east, u_p_north = _column_conductance_and_winds(
        pedersen_layers * dz, lower_pedersen, wind_east, wind_north
    )
    hall_conductance, u_h_east, u_h_north = _column_conductance_and_winds(
        hall_layers * dz, lower_hall, wind_east, wind_north
    )

    return {
        "SP": pedersen_conductance.astype(np.float32),
        "SH": hall_conductance.astype(np.float32),
        "u_p_theta": -u_p_north,
        "u_p_phi": u_p_east,
        "u_h_theta": -u_h_north,
        "u_h_phi": u_h_east,
    }


def _tiegcm_times(dataset: Any, reference_times: list[dt.datetime]) -> list[dt.datetime]:
    """Return standard TIEGCM mtime histories as datetimes."""
    if not reference_times:
        raise ValueError("At least one GAMERA reference time is required.")

    mtime_variable = dataset.variables.get("mtime")
    if mtime_variable is None:
        raise RuntimeError("TIEGCM file must provide standard mtime values.")
    raw_mtime = np.asarray(mtime_variable[:], dtype=int)
    dimensions = tuple(getattr(mtime_variable, "dimensions", ()))
    if raw_mtime.ndim != 2 or dimensions.count("mtimedim") != 1:
        raise RuntimeError(
            "TIEGCM mtime must be two-dimensional with one 'mtimedim' axis; "
            f"got dimensions {dimensions} and shape {raw_mtime.shape}."
        )
    raw_mtime = np.moveaxis(raw_mtime, dimensions.index("mtimedim"), -1)
    if raw_mtime.shape[1] not in (3, 4):
        raise RuntimeError(
            "TIEGCM mtimedim must contain day, hour, minute[, second]; "
            f"got {raw_mtime.shape[1]} components."
        )
    if raw_mtime.shape[0] < len(reference_times):
        raise RuntimeError(
            f"TIEGCM provides {raw_mtime.shape[0]} histories but GAMERA requires "
            f"{len(reference_times)}."
        )

    year_variable = dataset.variables.get("year")
    if year_variable is None:
        raise RuntimeError("TIEGCM file must provide calendar year values.")
    years = np.asarray(year_variable[:], dtype=int).reshape(-1)
    if years.size < len(reference_times):
        raise RuntimeError(
            f"TIEGCM provides {years.size} year values but GAMERA requires {len(reference_times)}."
        )

    times = []
    for components, year in zip(
        raw_mtime[: len(reference_times)], years[: len(reference_times)], strict=True
    ):
        day_of_year, hour, minute = components[:3]
        second = 0 if components.size == 3 else components[3]
        if (
            not 1 <= year <= 9999
            or not 1 <= day_of_year <= 365 + calendar_module.isleap(year)
            or not 0 <= hour < 24
            or not 0 <= minute < 60
            or not 0 <= second < 60
        ):
            raise RuntimeError(
                "TIEGCM year/mtime contains an invalid value: "
                f"year={year}, mtime={tuple(components)}."
            )
        times.append(
            dt.datetime(year, 1, 1)
            + dt.timedelta(
                days=int(day_of_year) - 1,
                hours=int(hour),
                minutes=int(minute),
                seconds=int(second),
            )
        )

    return times


def _validate_tiegcm_variables(dataset: Any, n_steps: int) -> None:
    """Require the geographic grid and vertical-layer contract."""
    required = [
        "lon",
        "lat",
        "lev",
        "ilev",
        "mtime",
        "year",
        "SIGMA_PED",
        "SIGMA_HAL",
        "Z",
        "ZG",
        "UN",
        "VN",
    ]
    missing = [name for name in required if name not in dataset.variables]
    if missing:
        raise RuntimeError(f"TIEGCM file is missing required variables {missing}.")
    too_short = [
        name
        for name in ("SIGMA_PED", "SIGMA_HAL", "Z", "ZG", "UN", "VN")
        if dataset.variables[name].shape[0] < n_steps
    ]
    if too_short:
        raise RuntimeError(
            f"TIEGCM variables {too_short} contain fewer than the required {n_steps} histories."
        )

    expected_dimensions = {
        "lon": ("lon",),
        "lat": ("lat",),
        "lev": ("lev",),
        "ilev": ("ilev",),
        "SIGMA_PED": ("time", "lev", "lat", "lon"),
        "SIGMA_HAL": ("time", "lev", "lat", "lon"),
        "Z": ("time", "ilev", "lat", "lon"),
        "ZG": ("time", "ilev", "lat", "lon"),
        "UN": ("time", "lev", "lat", "lon"),
        "VN": ("time", "lev", "lat", "lon"),
    }
    wrong_dimensions = {
        name: tuple(dataset.variables[name].dimensions)
        for name, expected in expected_dimensions.items()
        if tuple(dataset.variables[name].dimensions) != expected
    }
    if wrong_dimensions:
        raise RuntimeError(f"TIEGCM variables use incompatible dimensions: {wrong_dimensions}.")

    expected_units = {
        "lon": "degrees_east",
        "lat": "degrees_north",
        "SIGMA_PED": "S/m",
        "SIGMA_HAL": "S/m",
        "Z": "cm",
        "ZG": "cm",
        "UN": "cm/s",
        "VN": "cm/s",
    }
    wrong_units = {
        name: getattr(dataset.variables[name], "units", None)
        for name, expected in expected_units.items()
        if getattr(dataset.variables[name], "units", None) != expected
    }
    if wrong_units:
        raise RuntimeError(f"TIEGCM variables use incompatible units: {wrong_units}.")

    longitude = np.asarray(dataset.variables["lon"][:], dtype=float)
    latitude = np.asarray(dataset.variables["lat"][:], dtype=float)
    for name, values, full_span in (
        ("longitude", longitude, 360.0),
        ("latitude", latitude, 180.0),
    ):
        if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
            raise RuntimeError(f"TIEGCM {name} must be a finite one-dimensional grid.")
        spacing = np.diff(values)
        if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-10):
            raise RuntimeError(f"TIEGCM {name} must be strictly increasing and uniform.")
        if not np.isclose(values[-1] - values[0] + spacing[0], full_span, atol=1e-10):
            raise RuntimeError(f"TIEGCM {name} must cover one global cell-centred span.")
    if latitude[0] <= -90.0 or latitude[-1] >= 90.0:
        raise RuntimeError("TIEGCM latitude must contain cell centres strictly between the poles.")

    lev = np.asarray(dataset.variables["lev"][:], dtype=float)
    ilev = np.asarray(dataset.variables["ilev"][:], dtype=float)
    if (
        lev.ndim != 1
        or ilev.ndim != 1
        or lev.size != ilev.size
        or lev.size < 2
        or np.any(~np.isfinite(lev))
        or np.any(~np.isfinite(ilev))
        or np.any(np.diff(ilev) <= 0.0)
        or not np.allclose(lev[:-1], 0.5 * (ilev[:-1] + ilev[1:]), rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("TIEGCM lev[:-1] must be centered between consecutive ilev interfaces.")
    _lower_dynamo_layer_count(ilev)


def _apply_conductance_floor(
    pedersen_conductance: np.ndarray, hall_conductance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the global background minima used by the PynaMIT sheet."""
    pedersen, hall = np.broadcast_arrays(
        np.asarray(pedersen_conductance, dtype=float), np.asarray(hall_conductance, dtype=float)
    )
    if np.any(~np.isfinite(pedersen)) or np.any(pedersen < 0.0):
        raise ValueError("Pedersen conductance must be finite and non-negative.")
    if np.any(~np.isfinite(hall)) or np.any(hall < 0.0):
        raise ValueError("Hall conductance must be finite and non-negative.")

    floored_pedersen = np.maximum(pedersen, PEDERSEN_CONDUCTANCE_FLOOR_S)
    floored_hall = np.maximum(hall, HALL_CONDUCTANCE_FLOOR_S)
    return floored_pedersen.astype(np.float32), floored_hall.astype(np.float32)
