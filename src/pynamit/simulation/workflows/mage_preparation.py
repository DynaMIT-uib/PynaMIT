"""Build reusable MAGE/GAMERA/TIEGCM forcing.

The expensive height integration and source-coordinate transformations
are done here once. The output HDF5 contains the fields used by the
projection step on fixed, Earth-attached geographic grids:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S, with global
  2 S / 1 S background minima for the global PynaMIT sheet.
- ``u_p_theta``/``u_p_phi``: Pedersen-weighted model-basis wind in m/s.
- ``u_h_theta``/``u_h_phi``: Hall-weighted model-basis wind in m/s.
- radial current derived from REMIX FAC and the GAMERA inner-boundary
  radial magnetic perturbation,
  remapped from their timestamped SM source coordinates.

The wind integration intentionally stores conductivity-weighted winds,
not a height-resolved ``u x B`` source. The projection step uses
the PynaMIT sheet-radius main field and sheet resistance, matching the
thin-sheet ``JS -> E_S`` closure.

The prepared file is the minimal projection contract, not a diagnostic
archive. It is written atomically so a failed preparation cannot replace
the last complete forcing file.
"""

from __future__ import annotations

import calendar as calendar_module
import datetime as dt
import operator
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm
from pynamit.math.constants import RE

# Prepared-forcing contract
IONOSPHERE_RADIUS_M = 6.5e6
MAGE_FORCING_KIND = "pynamit_mage_forcing"
MAGE_FORCING_VERSION = 13
MAGE_TIME_AXIS = "tiegcm_mtime_nominal"
MAGE_SOURCE_TIME_TOLERANCE_SECONDS = 0.1

# Source-model conventions
CONDUCTANCE_FLOOR_MODEL = "global_hard_minimum"
PEDERSEN_CONDUCTANCE_FLOOR_S = 2.0
HALL_CONDUCTANCE_FLOOR_S = 1.0
TIEGCM_DYNAMO_BOTTOM_ILEV = -8.5
TIEGCM_DYNAMO_REFERENCE_HEIGHT_M = 90_000.0
TIEGCM_PEDERSEN_LOWER_SCALE_M = 5_000.0
TIEGCM_HALL_LOWER_SCALE_M = 3_000.0
GAMERA_EARTH_SPEED_SCALE_M_S = 1.0e5
REMIX_TIME_TOLERANCE_SECONDS = 1.0e-3
TIEGCM_FILL_THRESHOLD = 1e30
MJD_EPOCH = dt.datetime(1858, 11, 17)


@dataclass(frozen=True)
class PreparationSettings:
    """Inputs and output policy for one MAGE forcing preparation."""

    gamera_directory: Path
    output_path: Path
    tag: str = "msphere"
    inner_index: int = 0
    tiegcm_path: Path | None = None
    compression: str = "lzf"
    max_steps: int | None = None


# Time and coordinate conventions


def _datetime_from_mjd(value: float) -> dt.datetime:
    """Convert one finite MJD value to a naive UTC datetime."""
    mjd = float(value)
    if not np.isfinite(mjd):
        raise RuntimeError("Source MJD time must be finite.")
    return MJD_EPOCH + dt.timedelta(days=mjd)


def _kaiju_sm_transform_time(event_time: dt.datetime) -> dt.datetime:
    """Return the whole-second time used by Kaiju's ``mjdRECALC``."""
    if not isinstance(event_time, dt.datetime):
        raise TypeError("Kaiju SM transform time must be a datetime.")
    if event_time.tzinfo is not None:
        event_time = event_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
    # Fortran NINT rounds the non-negative second-of-minute value to the
    # nearest integer, including carry into the next minute.
    return (event_time + dt.timedelta(microseconds=500_000)).replace(microsecond=0)


def _gamera_internal_dipole_axes(mag_m0_nT: float) -> dict[str, np.ndarray]:
    """Return GAMERA dipole-moment and magnetic-pole axes."""
    if not np.isfinite(mag_m0_nT) or mag_m0_nT == 0.0:
        raise ValueError("GAMERA MagM0 must be finite and nonzero.")
    sign = float(np.sign(mag_m0_nT))
    moment_axis = np.array([0.0, 0.0, sign])
    north_axis = -moment_axis
    moment_axis[np.isclose(moment_axis, 0.0)] = 0.0
    north_axis[np.isclose(north_axis, 0.0)] = 0.0
    return {"moment_axis": moment_axis, "north_axis": north_axis}


def _pynamit_dipole_B0_T(mag_m0_nT: float, length_scale_m: float) -> float:
    """Convert GAMERA MagM0 to PynaMIT's reference-radius B0."""
    if not np.isfinite(mag_m0_nT) or mag_m0_nT == 0.0:
        raise ValueError("GAMERA MagM0 must be finite and nonzero.")
    if not np.isfinite(length_scale_m) or length_scale_m <= 0.0:
        raise ValueError("GAMERA length scale must be finite and positive.")
    return abs(float(mag_m0_nT)) * 1e-9 * (float(length_scale_m) / RE) ** 3


def _centered_dipole_alignment_attrs(event_time: dt.datetime, mag_m0_nT: float) -> dict[str, Any]:
    """Return coordinate alignment for prepared GAMERA forcing."""
    transform_time = _kaiju_sm_transform_time(event_time)
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(transform_time))
    alignment = main_field.alignment_metadata(transform_time)
    internal = _gamera_internal_dipole_axes(mag_m0_nT)
    return {
        "gamera_source_coordinate_system": "SM",
        "gamera_internal_magnetic_north_axis": internal["north_axis"],
        "gamera_internal_dipole_moment_axis": internal["moment_axis"],
        **alignment,
    }


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


def _h5_text(value: Any) -> str:
    """Return one HDF5 text attribute as a stripped string."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return str(value).strip()


def _gamera_length_scale_m(gsph: Any) -> float:
    """Return the EARTH-normalized GAMERA coordinate scale in metres."""
    with h5py.File(gsph.f0, "r") as file:
        units_id = _h5_text(file.attrs.get("UnitsID", ""))
        if units_id.upper() != "EARTH":
            raise RuntimeError(
                "MAGE preparation requires an EARTH-normalized GAMERA file; "
                f"got UnitsID={units_id!r}."
            )
        if "tScl" not in file.attrs:
            raise RuntimeError("EARTH-normalized GAMERA metadata is missing tScl.")
        time_scale_seconds = float(file.attrs["tScl"])
        if not np.isfinite(time_scale_seconds) or time_scale_seconds <= 0.0:
            raise RuntimeError("GAMERA tScl must be finite and positive.")
        if _h5_text(file.attrs.get("timeID", "")) != "s":
            raise RuntimeError("GAMERA tScl must be expressed in seconds.")
    # Kaiju's EARTH normalization fixes v0=100 km/s and tScl=Rp/v0.
    return time_scale_seconds * GAMERA_EARTH_SPEED_SCALE_M_S


def _gamera_dipole_strength_nT(gsph: Any) -> float:
    """Return GAMERA's required signed dipole strength in nT."""
    with h5py.File(gsph.f0, "r") as file:
        if "MagM0" not in file.attrs:
            raise RuntimeError(
                "GAMERA root metadata is missing the signed dipole strength MagM0. "
                "It is required to align and scale the prepared forcing."
            )
        strength = float(file.attrs["MagM0"])
    if not np.isfinite(strength) or strength == 0.0:
        raise RuntimeError("GAMERA MagM0 must be finite and nonzero.")
    return strength


def _gamera_background_field(
    gsph: Any, inner_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the volume-averaged inner-boundary GAMERA split field."""
    names = ("Bx0", "By0", "Bz0")
    with h5py.File(gsph.f0, "r") as root_file:
        missing = [name for name in names if name not in root_file]
        wrong_units = [
            name
            for name in names
            if name in root_file and _h5_text(root_file[name].attrs.get("Units", "")) != "nT"
        ]
    if missing:
        raise RuntimeError(
            "This preparation path expects Kaiju background-field output. "
            f"Missing root datasets: {missing}. For MAGE/GAMERA Earth runs, "
            "Kaiju writes total Bx/By/Bz and root Bx0/By0/Bz0."
        )
    if wrong_units:
        raise RuntimeError(f"GAMERA split-background datasets must use nT: {wrong_units}.")
    return tuple(np.asarray(gsph.GetVar(name)[inner_index]) for name in names)


def _validate_gamera_dynamic_field_units(gsph: Any, step: int) -> None:
    """Require physical nT units for saved GAMERA magnetic histories."""
    group_name = f"Step#{step}"
    names = ("Bx", "By", "Bz")
    with h5py.File(gsph.f0, "r") as file:
        if group_name not in file:
            raise RuntimeError(f"GAMERA file is missing {group_name!r}.")
        group = file[group_name]
        missing = [name for name in names if name not in group]
        wrong_units = [
            name
            for name in names
            if name in group and _h5_text(group[name].attrs.get("Units", "")) != "nT"
        ]
    if missing:
        raise RuntimeError(f"GAMERA history {group_name!r} is missing {missing}.")
    if wrong_units:
        raise RuntimeError(f"GAMERA magnetic histories must use nT: {wrong_units}.")


# TIEGCM conductance and wind preparation


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


def _time_offsets_seconds(
    source_times: list[dt.datetime], nominal_times: list[dt.datetime]
) -> np.ndarray:
    """Return signed source-minus-nominal time offsets."""
    return np.array(
        [
            (source_time - nominal_time).total_seconds()
            for source_time, nominal_time in zip(source_times, nominal_times, strict=True)
        ],
        dtype=float,
    )


def _validate_forcing_time_axis(
    nominal_times: list[dt.datetime],
    gamera_times: list[dt.datetime],
    remix_times: list[dt.datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the nominal schedule and exact source times."""
    n_steps = len(nominal_times)
    if n_steps == 0:
        raise RuntimeError("No forcing steps are available.")
    source_lengths = {"GAMERA": len(gamera_times), "ReMIX": len(remix_times)}
    mismatched = {name: size for name, size in source_lengths.items() if size != n_steps}
    if mismatched:
        raise RuntimeError(
            f"The nominal time axis has {n_steps} histories but source counts are {mismatched}."
        )

    time_axes = {"nominal TIEGCM": nominal_times, "GAMERA": gamera_times, "ReMIX": remix_times}
    for name, times in time_axes.items():
        intervals = np.array(
            [
                (next_time - time).total_seconds()
                for time, next_time in zip(times[:-1], times[1:], strict=True)
            ],
            dtype=float,
        )
        if np.any(intervals <= 0.0):
            raise RuntimeError(f"The {name} time axis must be strictly increasing.")
        if (
            name == "nominal TIEGCM"
            and intervals.size > 1
            and not np.allclose(intervals, intervals[0], rtol=0.0, atol=1e-9)
        ):
            raise RuntimeError("The nominal TIEGCM time axis must have a uniform cadence.")

    gamera_offsets = _time_offsets_seconds(gamera_times, nominal_times)
    remix_offsets = _time_offsets_seconds(remix_times, nominal_times)
    for source, offsets in {"GAMERA": gamera_offsets, "ReMIX": remix_offsets}.items():
        mismatch = np.flatnonzero(np.abs(offsets) > MAGE_SOURCE_TIME_TOLERANCE_SECONDS)
        if mismatch.size:
            index = int(mismatch[0])
            raise RuntimeError(
                f"{source} is not aligned with the nominal forcing time at source step "
                f"{index}: offset={offsets[index]:g} s; allowed absolute offset is "
                f"{MAGE_SOURCE_TIME_TOLERANCE_SECONDS:g} s."
            )

    remix_gamera_offsets = _time_offsets_seconds(remix_times, gamera_times)
    mismatch = np.flatnonzero(np.abs(remix_gamera_offsets) > REMIX_TIME_TOLERANCE_SECONDS)
    if mismatch.size:
        index = int(mismatch[0])
        raise RuntimeError(
            "ReMIX is not aligned with GAMERA at source step "
            f"{index}: offset={remix_gamera_offsets[index]:g} s; allowed absolute offset is "
            f"{REMIX_TIME_TOLERANCE_SECONDS:g} s."
        )
    return gamera_offsets, remix_offsets


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


# GAMERA boundary geometry and interpolation


@dataclass(frozen=True)
class _GameraBoundaryGeometry:
    """Geometry of one GAMERA magnetic-field shell."""

    sm_latitude: np.ndarray
    sm_longitude: np.ndarray
    radius_m: np.ndarray
    radial_unit_x: np.ndarray
    radial_unit_y: np.ndarray
    radial_unit_z: np.ndarray
    solid_angle: np.ndarray

    def radial_component(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
        """Return a Cartesian field's radial component on this shell."""
        bx, by, bz = (np.asarray(component) for component in (bx, by, bz))
        if bx.shape != self.radius_m.shape or by.shape != bx.shape or bz.shape != bx.shape:
            raise ValueError("GAMERA boundary-field components must match the boundary geometry.")
        return bx * self.radial_unit_x + by * self.radial_unit_y + bz * self.radial_unit_z


def _gamera_inner_boundary_geometry(
    gsph: Any, inner_index: int, length_scale_m: float
) -> _GameraBoundaryGeometry:
    """Return Kaiju cell centers corresponding to ``B[inner_index]``.

    GAMERA stores ``X/Y/Z`` at cell vertices and magnetic fields at cell
    centers. The selected magnetic shell therefore lies between vertex
    shells ``inner_index`` and ``inner_index + 1``. Kaiju defines the
    location as the volume barycenter of the trilinear cell.
    """
    vertices = _gamera_inner_boundary_vertices(gsph, inner_index)
    x, y, z = np.moveaxis(_trilinear_hexahedron_volume_centers(vertices), -1, 0)

    r_re = np.sqrt(x**2 + y**2 + z**2)
    radial_unit_x = x / r_re
    radial_unit_y = y / r_re
    radial_unit_z = z / r_re
    return _GameraBoundaryGeometry(
        sm_latitude=np.degrees(np.arcsin(np.clip(radial_unit_z, -1.0, 1.0))),
        sm_longitude=np.degrees(np.arctan2(y, x)),
        radius_m=r_re * length_scale_m,
        radial_unit_x=radial_unit_x,
        radial_unit_y=radial_unit_y,
        radial_unit_z=radial_unit_z,
        solid_angle=_gamera_boundary_solid_angle(vertices),
    )


def _gamera_inner_boundary_vertices(gsph: Any, inner_index: int) -> np.ndarray:
    """Return boundary hexahedron vertices in Kaiju's corner order."""
    x, y, z = (
        np.asarray(coordinate[inner_index : inner_index + 2], dtype=float)
        for coordinate in (gsph.X, gsph.Y, gsph.Z)
    )
    positions = np.stack((x, y, z), axis=-1)
    vertices = np.stack(
        (
            positions[0, :-1, :-1],
            positions[1, :-1, :-1],
            positions[1, 1:, :-1],
            positions[0, 1:, :-1],
            positions[0, :-1, 1:],
            positions[1, :-1, 1:],
            positions[1, 1:, 1:],
            positions[0, 1:, 1:],
        ),
        axis=-2,
    )
    if np.any(~np.isfinite(vertices)):
        raise RuntimeError("GAMERA inner-boundary vertices must be finite.")
    return vertices


def _trilinear_hexahedron_volume_centers(vertices: np.ndarray) -> np.ndarray:
    """Return volume barycenters using Kaiju's Gaussian quadrature."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[-2:] != (8, 3):
        raise ValueError("Hexahedron vertices must have final shape (8, 3).")
    corner_signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    points, weights = np.polynomial.legendre.leggauss(12)
    volume = np.zeros(vertices.shape[:-2], dtype=float)
    first_moment = np.zeros(vertices.shape[:-2] + (3,), dtype=float)
    for i, xi in enumerate(points):
        for j, eta in enumerate(points):
            for k, zeta in enumerate(points):
                factors = (
                    (1.0 + corner_signs[:, 0] * xi)
                    * (1.0 + corner_signs[:, 1] * eta)
                    * (1.0 + corner_signs[:, 2] * zeta)
                    / 8.0
                )
                position = np.einsum("...vc,v->...c", vertices, factors, optimize=True)
                derivatives = (
                    np.stack(
                        (
                            corner_signs[:, 0]
                            * (1.0 + corner_signs[:, 1] * eta)
                            * (1.0 + corner_signs[:, 2] * zeta),
                            (1.0 + corner_signs[:, 0] * xi)
                            * corner_signs[:, 1]
                            * (1.0 + corner_signs[:, 2] * zeta),
                            (1.0 + corner_signs[:, 0] * xi)
                            * (1.0 + corner_signs[:, 1] * eta)
                            * corner_signs[:, 2],
                        ),
                        axis=-1,
                    )
                    / 8.0
                )
                jacobian = np.einsum("...vc,vq->...cq", vertices, derivatives, optimize=True)
                weighted_volume = (
                    weights[i] * weights[j] * weights[k] * np.abs(np.linalg.det(jacobian))
                )
                volume += weighted_volume
                first_moment += weighted_volume[..., None] * position
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise RuntimeError("GAMERA inner-boundary cells must have finite positive volumes.")
    return first_moment / volume[..., None]


def _spherical_triangle_solid_angle(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> np.ndarray:
    """Return the unsigned solid angle of unit-vector triangles."""
    numerator = np.abs(np.einsum("...i,...i->...", first, np.cross(second, third)))
    denominator = (
        1.0
        + np.einsum("...i,...i->...", first, second)
        + np.einsum("...i,...i->...", second, third)
        + np.einsum("...i,...i->...", third, first)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def _gamera_boundary_solid_angle(vertices: np.ndarray) -> np.ndarray:
    """Return each boundary cell's solid angle from its vertices."""
    mid_shell = np.stack(
        (
            0.5 * (vertices[..., 0, :] + vertices[..., 1, :]),
            0.5 * (vertices[..., 3, :] + vertices[..., 2, :]),
            0.5 * (vertices[..., 7, :] + vertices[..., 6, :]),
            0.5 * (vertices[..., 4, :] + vertices[..., 5, :]),
        ),
        axis=-2,
    )
    norms = np.linalg.norm(mid_shell, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("GAMERA inner-boundary vertices must have finite nonzero radii.")
    unit = mid_shell / norms
    lower_left, upper_left, upper_right, lower_right = np.moveaxis(unit, -2, 0)
    solid_angle = _spherical_triangle_solid_angle(
        lower_left, upper_left, upper_right
    ) + _spherical_triangle_solid_angle(lower_left, upper_right, lower_right)
    if np.any(~np.isfinite(solid_angle)) or np.any(solid_angle <= 0.0):
        raise RuntimeError("GAMERA inner-boundary cells must have finite positive solid angles.")
    return solid_angle


def _gamera_native_angles(
    sm_latitude: np.ndarray, sm_longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return GAMERA-native colatitude and azimuth in radians.

    GAMERA's spherical grid uses the SM +x axis as its polar axis and
    measures azimuth from +y toward +z.
    """
    sm_latitude, sm_longitude = np.broadcast_arrays(
        np.asarray(sm_latitude, dtype=float), np.asarray(sm_longitude, dtype=float)
    )
    latitude = np.deg2rad(sm_latitude)
    longitude = np.deg2rad(sm_longitude)
    cos_latitude = np.cos(latitude)
    x = cos_latitude * np.cos(longitude)
    y = cos_latitude * np.sin(longitude)
    z = np.sin(latitude)
    colatitude = np.arccos(np.clip(x, -1.0, 1.0))
    azimuth = np.mod(np.arctan2(z, y), 2.0 * np.pi)
    return colatitude, azimuth


class _GameraBoundaryInterpolator:
    """Apply Kaiju-style four-point bilinear interpolation on GAMERA.

    The full inner boundary is a periodic tensor grid in GAMERA's native
    angular coordinates even though it is folded in ordinary SM
    latitude/longitude. Cell transforms are built once and reused for
    every magnetic history. Values at the omitted +x and -x poles are
    reconstructed from the means of their adjacent cell-center rings.
    """

    def __init__(self, source_sm_lat: np.ndarray, source_sm_lon: np.ndarray) -> None:
        source_sm_lat, source_sm_lon = np.broadcast_arrays(
            np.asarray(source_sm_lat, dtype=float), np.asarray(source_sm_lon, dtype=float)
        )
        if source_sm_lat.ndim != 2 or min(source_sm_lat.shape) < 2:
            raise ValueError(
                "A GAMERA boundary grid must be two-dimensional with at least two cells per axis."
            )
        if np.any(~np.isfinite(source_sm_lat)) or np.any(~np.isfinite(source_sm_lon)):
            raise ValueError("GAMERA boundary coordinates must be finite.")

        colatitude, azimuth = _gamera_native_angles(source_sm_lat, source_sm_lon)
        azimuth = np.unwrap(azimuth, axis=1)
        azimuth += 2.0 * np.pi * np.round((azimuth[0, 0] - azimuth[:, [0]]) / (2.0 * np.pi))
        colatitude_axis = np.mean(colatitude, axis=1)
        azimuth_axis = np.mean(azimuth, axis=0)

        self._colatitude_order = np.arange(source_sm_lat.shape[0])
        self._azimuth_order = np.arange(source_sm_lat.shape[1])
        if np.all(np.diff(colatitude_axis) < 0.0):
            self._colatitude_order = self._colatitude_order[::-1]
            colatitude = colatitude[::-1]
            colatitude_axis = colatitude_axis[::-1]
            azimuth = azimuth[::-1]
        if np.all(np.diff(azimuth_axis) < 0.0):
            self._azimuth_order = self._azimuth_order[::-1]
            colatitude = colatitude[:, ::-1]
            azimuth = azimuth[:, ::-1]
            azimuth_axis = azimuth_axis[::-1]
        if np.any(np.diff(colatitude_axis) <= 0.0) or np.any(np.diff(azimuth_axis) <= 0.0):
            raise ValueError("GAMERA native angular coordinates must be monotonic.")
        if colatitude_axis[0] <= 0.0 or colatitude_axis[-1] >= np.pi:
            raise ValueError("GAMERA cell-center colatitudes must lie strictly between its poles.")

        colatitude_step = np.min(np.diff(colatitude_axis))
        azimuth_step = np.min(np.diff(azimuth_axis))
        if np.max(np.abs(colatitude - colatitude_axis[:, None])) >= 0.25 * colatitude_step:
            raise ValueError("GAMERA colatitudes do not form a searchable logical grid.")
        if np.max(np.abs(azimuth - azimuth_axis[None, :])) >= 0.25 * azimuth_step:
            raise ValueError("GAMERA azimuths do not form a searchable logical grid.")

        self._source_shape = source_sm_lat.shape
        self._colatitude_axis = np.concatenate(([0.0], colatitude_axis, [np.pi]))
        self._azimuth_axis = azimuth_axis
        self._periodic_azimuth_axis = np.concatenate(
            (azimuth_axis, [azimuth_axis[0] + 2.0 * np.pi])
        )
        polar_azimuth = np.broadcast_to(azimuth_axis, (1, azimuth_axis.size))
        colatitude = np.vstack(
            (np.zeros_like(polar_azimuth), colatitude, np.full_like(polar_azimuth, np.pi))
        )
        azimuth = np.vstack((polar_azimuth, azimuth, polar_azimuth))
        self._cell_inverse = self._build_cell_inverse(colatitude, azimuth)

    @staticmethod
    def _build_cell_inverse(colatitude: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        """Return Kaiju's four-corner bilinear transforms."""
        lower_azimuth = azimuth[:-1]
        upper_azimuth = azimuth[1:]
        lower_right_azimuth = np.roll(lower_azimuth, -1, axis=1)
        upper_right_azimuth = np.roll(upper_azimuth, -1, axis=1)
        lower_right_azimuth[:, -1] += 2.0 * np.pi
        upper_right_azimuth[:, -1] += 2.0 * np.pi
        vertex_azimuth = np.stack(
            (lower_azimuth, lower_right_azimuth, upper_azimuth, upper_right_azimuth), axis=-1
        )
        vertex_colatitude = np.stack(
            (
                colatitude[:-1],
                np.roll(colatitude[:-1], -1, axis=1),
                colatitude[1:],
                np.roll(colatitude[1:], -1, axis=1),
            ),
            axis=-1,
        )
        basis = np.stack(
            (
                np.ones_like(vertex_azimuth),
                vertex_azimuth,
                vertex_colatitude,
                vertex_azimuth * vertex_colatitude,
            ),
            axis=-2,
        )
        try:
            inverse = np.linalg.inv(basis)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "GAMERA boundary cells must have invertible angular geometry."
            ) from exc
        if np.any(~np.isfinite(inverse)):
            raise ValueError("GAMERA boundary interpolation geometry must be finite.")
        return inverse

    def interpolate(
        self, values: np.ndarray, *, target_sm_lat: np.ndarray, target_sm_lon: np.ndarray
    ) -> np.ndarray:
        """Interpolate one boundary field at SM target positions."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._source_shape:
            raise ValueError(
                f"GAMERA boundary field shape {values.shape} does not match {self._source_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("GAMERA boundary interpolation requires finite source values.")
        values = values[np.ix_(self._colatitude_order, self._azimuth_order)]
        values = np.vstack(
            (
                np.full((1, values.shape[1]), np.mean(values[0])),
                values,
                np.full((1, values.shape[1]), np.mean(values[-1])),
            )
        )

        target_sm_lat, target_sm_lon = np.broadcast_arrays(
            np.asarray(target_sm_lat, dtype=float), np.asarray(target_sm_lon, dtype=float)
        )
        target_shape = target_sm_lat.shape
        colatitude, azimuth = _gamera_native_angles(target_sm_lat, target_sm_lon)
        colatitude = colatitude.reshape(-1)
        azimuth = azimuth.reshape(-1)
        if np.any(~np.isfinite(colatitude)) or np.any(~np.isfinite(azimuth)):
            raise ValueError("GAMERA boundary target coordinates must be finite.")
        azimuth = np.mod(azimuth - self._azimuth_axis[0], 2.0 * np.pi) + self._azimuth_axis[0]

        colatitude_index = np.searchsorted(self._colatitude_axis, colatitude, side="right") - 1
        colatitude_index = np.clip(colatitude_index, 0, self._colatitude_axis.size - 2)
        azimuth_index = np.searchsorted(self._periodic_azimuth_axis, azimuth, side="right") - 1
        azimuth_index = np.clip(azimuth_index, 0, self._azimuth_axis.size - 1)
        next_azimuth_index = (azimuth_index + 1) % self._azimuth_axis.size

        target_basis = np.column_stack(
            (np.ones_like(azimuth), azimuth, colatitude, azimuth * colatitude)
        )
        weights = np.einsum(
            "nij,nj->ni", self._cell_inverse[colatitude_index, azimuth_index], target_basis
        )
        weights = np.clip(weights, 0.0, 1.0)
        corners = np.column_stack(
            (
                values[colatitude_index, azimuth_index],
                values[colatitude_index, next_azimuth_index],
                values[colatitude_index + 1, azimuth_index],
                values[colatitude_index + 1, next_azimuth_index],
            )
        )
        return np.sum(weights * corners, axis=1).reshape(target_shape)


# ReMIX radial-current preparation


class _RemixGridInterpolator:
    """Interpolate one saved ReMIX hemisphere on its native tensor grid.

    Kaiju's ReMIX coupling uses a four-point interpolant in colatitude
    and longitude, with a three-vertex rule in the cell touching the
    pole. ReMIX writes fields without that degenerate pole and stores a
    staggered X/Y grid whose cell centers locate the remaining field
    nodes. This class reconstructs the pole and applies the same mapping
    geometry.
    """

    def __init__(self, source_lat: np.ndarray, source_lon: np.ndarray) -> None:
        source_lat, source_lon = np.broadcast_arrays(
            np.asarray(source_lat, dtype=float), np.asarray(source_lon, dtype=float)
        )
        if source_lat.ndim != 2 or min(source_lat.shape) < 2:
            raise ValueError(
                "A ReMIX grid must be two-dimensional with at least two cells per axis."
            )
        if np.any(~np.isfinite(source_lat)) or np.any(~np.isfinite(source_lon)):
            raise ValueError("ReMIX grid coordinates must be finite.")

        latitude = source_lat[:, 0]
        longitude = np.mod(source_lon[0], 360.0)
        longitude_residual = wrap_longitude_180(source_lon - source_lon[[0]])
        if not np.allclose(source_lat, latitude[:, None], rtol=0.0, atol=1e-12) or not np.allclose(
            longitude_residual, 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "Saved ReMIX coordinates must form a rectilinear latitude/longitude grid."
            )
        if not (np.all(latitude > 0.0) or np.all(latitude < 0.0)):
            raise ValueError("A saved ReMIX grid must contain exactly one magnetic hemisphere.")

        self._source_shape = source_lat.shape
        self._latitude_order = np.argsort(latitude)
        self._longitude_order = np.argsort(longitude)
        self._latitude = latitude[self._latitude_order]
        self._longitude = longitude[self._longitude_order]
        if np.any(np.diff(self._latitude) <= 0.0) or np.any(np.diff(self._longitude) <= 0.0):
            raise ValueError("ReMIX latitude and longitude coordinates must be unique.")

    def interpolate(
        self, values: np.ndarray, target_lon: np.ndarray, target_lat: np.ndarray
    ) -> np.ndarray:
        """Interpolate periodically within the source hemisphere."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._source_shape:
            raise ValueError(
                f"ReMIX field shape {values.shape} does not match {self._source_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("ReMIX interpolation requires finite source values.")
        values = values[np.ix_(self._latitude_order, self._longitude_order)]

        # ReMIX omits the degenerate pole when writing fields and
        # restores it as the mean of the poleward ring when reading.
        latitude = self._latitude
        if latitude[0] > 0.0:
            poleward_ring = values[-1]
            latitude = np.concatenate((latitude, [90.0]))
            pole_value = np.mean(poleward_ring)
            values = np.vstack((values, np.full((1, values.shape[1]), pole_value)))
            north = True
        else:
            poleward_ring = values[0]
            latitude = np.concatenate(([-90.0], latitude))
            pole_value = np.mean(poleward_ring)
            values = np.vstack((np.full((1, values.shape[1]), pole_value), values))
            north = False

        target_lon, target_lat = np.broadcast_arrays(
            np.asarray(target_lon, dtype=float), np.asarray(target_lat, dtype=float)
        )
        target_shape = target_lat.shape
        query_latitude = target_lat.reshape(-1)
        query_longitude = (
            np.mod(target_lon.reshape(-1) - self._longitude[0], 360.0) + self._longitude[0]
        )
        finite = np.isfinite(query_latitude) & np.isfinite(query_longitude)
        latitude_tolerance = max(
            1e-12, 16.0 * np.finfo(float).eps * float(np.max(np.abs(latitude)))
        )
        covered = (
            finite
            & (query_latitude >= latitude[0] - latitude_tolerance)
            & (query_latitude <= latitude[-1] + latitude_tolerance)
        )
        result = np.full(query_latitude.size, np.nan)
        if not np.any(covered):
            return result.reshape(target_shape)

        covered_indices = np.flatnonzero(covered)
        query_latitude = np.clip(query_latitude[covered], latitude[0], latitude[-1])
        query_latitude = np.where(
            np.abs(query_latitude - self._latitude[0]) <= latitude_tolerance,
            self._latitude[0],
            query_latitude,
        )
        query_latitude = np.where(
            np.abs(query_latitude - self._latitude[-1]) <= latitude_tolerance,
            self._latitude[-1],
            query_latitude,
        )
        query_longitude = query_longitude[covered]
        latitude_index = np.searchsorted(latitude, query_latitude, side="right") - 1
        latitude_index = np.clip(latitude_index, 0, latitude.size - 2)

        periodic_longitude = np.concatenate((self._longitude, [self._longitude[0] + 360.0]))
        longitude_index = np.searchsorted(periodic_longitude, query_longitude, side="right") - 1
        longitude_index = np.clip(longitude_index, 0, self._longitude.size - 1)
        next_longitude_index = (longitude_index + 1) % self._longitude.size

        latitude_fraction = (query_latitude - latitude[latitude_index]) / (
            latitude[latitude_index + 1] - latitude[latitude_index]
        )
        next_longitude = periodic_longitude[longitude_index + 1]
        longitude_fraction = (query_longitude - periodic_longitude[longitude_index]) / (
            next_longitude - periodic_longitude[longitude_index]
        )

        lower_left = values[latitude_index, longitude_index]
        lower_right = values[latitude_index, next_longitude_index]
        upper_left = values[latitude_index + 1, longitude_index]
        upper_right = values[latitude_index + 1, next_longitude_index]
        result[covered_indices] = (1.0 - latitude_fraction) * (
            (1.0 - longitude_fraction) * lower_left + longitude_fraction * lower_right
        ) + latitude_fraction * (
            (1.0 - longitude_fraction) * upper_left + longitude_fraction * upper_right
        )

        # Kaiju treats the polar quadrilateral as a triangle because all
        # longitude vertices at the pole are one physical point. Its map
        # therefore averages the reconstructed pole and the two adjacent
        # values on the poleward ring, independently of polar distance.
        polar_cap = (
            query_latitude > self._latitude[-1] if north else query_latitude < self._latitude[0]
        )
        if np.any(polar_cap):
            result[covered_indices[polar_cap]] = (
                pole_value
                + poleward_ring[longitude_index[polar_cap]]
                + poleward_ring[next_longitude_index[polar_cap]]
            ) / 3.0
        return result.reshape(target_shape)


def _geographic_grid_in_sm(
    latitude: np.ndarray, longitude: np.ndarray, event_time: dt.datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Return Kaiju SM coordinates of a fixed GEO grid."""
    return kaiju_geopack_sm(_kaiju_sm_transform_time(event_time)).geo2sm(latitude, longitude)


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


def _combine_remix_hemispheres(south: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Merge hemispheres and set the uncovered low latitudes to zero."""
    output = np.array(south, copy=True)
    mask = np.isnan(output)
    output[mask] = north[mask]
    output[np.isnan(output)] = 0.0
    if np.any(~np.isfinite(output)):
        raise ValueError("REMIX FAC interpolation produced non-finite values.")
    return output


def _dipole_radial_direction_cosine(magnetic_latitude: np.ndarray) -> np.ndarray:
    """Return a centered dipole's absolute radial direction cosine."""
    sin_latitude = np.sin(np.deg2rad(np.asarray(magnetic_latitude, dtype=float)))
    return np.abs(2.0 * sin_latitude / np.sqrt(1.0 + 3.0 * sin_latitude**2))


def _upward_fac_to_radial_current(
    upward_fac: np.ndarray, magnetic_latitude: np.ndarray
) -> np.ndarray:
    """Convert upward-positive dipole FAC to outward current."""
    return np.asarray(upward_fac, dtype=float) * _dipole_radial_direction_cosine(magnetic_latitude)


def _remix_upward_fac_source(
    hemisphere: str,
    fac: np.ndarray,
    unsigned_magnetic_latitude: np.ndarray,
    grid_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical SM positions and upward-positive ReMIX FAC.

    ReMIX stores both hemispheres on the same unsigned polar grid. Kaiju
    interprets southern positions with latitude ``-latitude`` and
    longitude ``-longitude``. Its saved FAC is parallel-positive, so it
    is negated in the north and retained in the south to obtain one
    outward/upward-positive convention.
    """
    hemisphere = str(hemisphere).upper()
    if hemisphere not in {"NORTH", "SOUTH"}:
        raise ValueError("ReMIX hemisphere must be 'NORTH' or 'SOUTH'.")

    latitude, longitude = np.broadcast_arrays(
        np.asarray(unsigned_magnetic_latitude, dtype=float),
        np.asarray(grid_longitude, dtype=float),
    )
    fac = np.asarray(fac, dtype=float)
    if fac.shape != latitude.shape:
        raise RuntimeError(
            f"ReMIX {hemisphere} FAC shape {fac.shape} does not match "
            f"the cell-center grid {latitude.shape}."
        )
    if np.any(~np.isfinite(fac)):
        raise RuntimeError(f"ReMIX {hemisphere} FAC must be finite.")

    if hemisphere == "NORTH":
        return latitude, wrap_longitude_180(longitude), -fac
    return -latitude, wrap_longitude_180(-longitude), fac


def _remix_cell_center_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return saved ReMIX field-node latitude and longitude."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 2 or min(x.shape) < 3 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise RuntimeError("ReMIX X/Y must be finite two-dimensional corner grids.")
    x_center = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y_center = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
    polar_radius = np.hypot(x_center, y_center)
    tolerance = 32.0 * np.finfo(float).eps
    if np.any(polar_radius > 1.0 + tolerance):
        raise RuntimeError("ReMIX X/Y cell centres must lie inside the unit polar disk.")
    colatitude = np.arcsin(np.clip(polar_radius, 0.0, 1.0))
    longitude = np.arctan2(y_center, x_center)
    return 90.0 - np.degrees(colatitude), wrap_longitude_180(np.degrees(longitude))


class _RemixRadialCurrentReader:
    """Read only the ReMIX FAC needed for outward current forcing."""

    def __init__(self, remix_file: Path) -> None:
        self._remix_file = Path(remix_file)
        self._file: h5py.File | None = None
        self._unsigned_latitude: np.ndarray | None = None
        self._grid_longitude: np.ndarray | None = None
        self._interpolators: dict[str, _RemixGridInterpolator] = {}

    def __enter__(self):
        """Open and validate the reusable ReMIX grid."""
        self._file = h5py.File(self._remix_file, "r")
        try:
            if _h5_text(self._file.attrs.get("UnitsID", "")) != "ReMIX":
                raise RuntimeError("ReMIX forcing must declare UnitsID='ReMIX'.")
            missing = [name for name in ("X", "Y") if name not in self._file]
            if missing:
                raise RuntimeError(f"ReMIX forcing is missing grid datasets {missing}.")
            wrong_units = [
                name
                for name in ("X", "Y")
                if _h5_text(self._file[name].attrs.get("Units", "")) != "Ri"
            ]
            if wrong_units:
                raise RuntimeError(f"ReMIX grid datasets must use Ri units: {wrong_units}.")
            unsigned_latitude, grid_longitude = _remix_cell_center_coordinates(
                self._file["X"][:], self._file["Y"][:]
            )
            self._unsigned_latitude = unsigned_latitude
            self._grid_longitude = grid_longitude
            shape = unsigned_latitude.shape
            zeros = np.zeros(shape)
            for hemisphere in ("NORTH", "SOUTH"):
                source_lat, source_lon, _ = _remix_upward_fac_source(
                    hemisphere, zeros, unsigned_latitude, grid_longitude
                )
                self._interpolators[hemisphere] = _RemixGridInterpolator(source_lat, source_lon)
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the ReMIX source file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _history(self, step: int) -> tuple[h5py.Group, dt.datetime]:
        """Return one ReMIX history and its exact source time."""
        if self._file is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        group_name = f"Step#{step}"
        if group_name not in self._file:
            raise RuntimeError(f"ReMIX forcing is missing {group_name!r}.")
        history = self._file[group_name]
        if "MJD" not in history.attrs:
            raise RuntimeError(f"ReMIX history {group_name!r} is missing MJD time metadata.")
        source_time = _datetime_from_mjd(history.attrs["MJD"])
        return history, source_time

    def source_time(self, step: int) -> dt.datetime:
        """Return the exact timestamp of one ReMIX history."""
        return self._history(step)[1]

    @property
    def equatorward_sm_latitude(self) -> float:
        """Return the saved ReMIX grid's equatorward SM latitude."""
        if self._unsigned_latitude is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        return float(np.min(np.abs(self._unsigned_latitude)))

    @staticmethod
    def _fac(history: h5py.Group, hemisphere: str) -> np.ndarray:
        """Read one parallel-positive ReMIX FAC field."""
        dataset_name = f"Field-aligned current {hemisphere}"
        if dataset_name not in history:
            raise RuntimeError(f"ReMIX history is missing {dataset_name!r}.")
        dataset = history[dataset_name]
        if _h5_text(dataset.attrs.get("Units", "")) != "muA/m**2":
            raise RuntimeError(f"ReMIX {dataset_name!r} must use muA/m**2 units.")
        return np.asarray(dataset, dtype=float)

    def _hemisphere(
        self,
        hemisphere: str,
        fac: np.ndarray,
        target_sm_lon: np.ndarray,
        target_sm_lat: np.ndarray,
    ) -> np.ndarray:
        """Sample one FAC hemisphere at target SM positions."""
        if self._unsigned_latitude is None or self._grid_longitude is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        _, _, upward_fac = _remix_upward_fac_source(
            hemisphere, fac, self._unsigned_latitude, self._grid_longitude
        )
        return self._interpolators[hemisphere].interpolate(
            upward_fac, target_sm_lon, target_sm_lat
        )

    def read(
        self,
        step: int,
        target_longitude: np.ndarray,
        target_latitude: np.ndarray,
        gamera_time: dt.datetime,
    ) -> np.ndarray:
        """Return outward current on the fixed geographic grid."""
        history, source_time = self._history(step)
        offset_seconds = abs((source_time - gamera_time).total_seconds())
        if offset_seconds > REMIX_TIME_TOLERANCE_SECONDS:
            raise RuntimeError(
                f"ReMIX Step#{step} is not aligned with GAMERA: "
                f"ReMIX={source_time.isoformat()}, GAMERA={gamera_time.isoformat()}, "
                f"offset={offset_seconds:g} s."
            )
        target_sm_lat, target_sm_lon = _geographic_grid_in_sm(
            target_latitude, target_longitude, gamera_time
        )
        north = self._hemisphere(
            "NORTH", self._fac(history, "NORTH"), target_sm_lon, target_sm_lat
        )
        south = self._hemisphere(
            "SOUTH", self._fac(history, "SOUTH"), target_sm_lon, target_sm_lat
        )
        upward_fac = _combine_remix_hemispheres(south, north)
        return _upward_fac_to_radial_current(upward_fac, target_sm_lat)


# Prepared-forcing output


def _h5_dataset_kwargs(compression: str) -> dict[str, Any]:
    """Return h5py dataset creation options."""
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


@contextmanager
def _atomic_prepared_output(output_path: Path):
    """Publish a temporary HDF5 file atomically when complete."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}-", suffix=".tmp.h5", dir=output_path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with h5py.File(temporary_path, "w") as output:
            yield output
            output.attrs["complete"] = True
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _create_output_datasets(
    output: h5py.File,
    n_steps: int,
    ion_shape: tuple[int, int],
    inner_shape: tuple[int, int],
    compression: str,
) -> None:
    """Create all time-dependent output datasets."""
    kwargs = _h5_dataset_kwargs(compression)
    for name in ("jr", "SH", "SP", "u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        output.create_dataset(name, shape=(n_steps, *ion_shape), dtype="f4", **kwargs)
    output.create_dataset("delta_Br", shape=(n_steps, *inner_shape), dtype="f4", **kwargs)
    output["jr"].attrs["units"] = "uA m-2"
    output["jr"].attrs["description"] = (
        "outward radial current from upward-positive REMIX FAC times abs(source unit_br); "
        "zero outside REMIX coverage"
    )
    output["SP"].attrs["units"] = "S"
    output["SP"].attrs["description"] = (
        "radially integrated TIEGCM Pedersen conductance with a global hard 2 S "
        "background minimum for the PynaMIT sheet"
    )
    output["SH"].attrs["units"] = "S"
    output["SH"].attrs["description"] = (
        "radially integrated TIEGCM Hall conductance with a global hard 1 S "
        "background minimum for the PynaMIT sheet"
    )
    for name in ("u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        output[name].attrs["units"] = "m s-1"
    output["delta_Br"].attrs["units"] = "nT"
    output["delta_Br"].attrs["description"] = (
        "radial perturbation from cell-volume-average total B minus the matching "
        "cell-volume-average GAMERA split background B0"
    )


def _write_time_axis(
    output: h5py.File,
    nominal_times: list[dt.datetime],
    gamera_times: list[dt.datetime],
    remix_times: list[dt.datetime],
) -> None:
    """Write the nominal clock and exact coupled-source times."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    timestamp_datasets = {
        "time": nominal_times,
        "gamera_source_time": gamera_times,
        "remix_source_time": remix_times,
    }
    for name, times in timestamp_datasets.items():
        values = np.asarray([value.isoformat() for value in times], dtype=string_dtype)
        output.create_dataset(name, data=values, dtype=string_dtype)

    output["time"].attrs["description"] = (
        "nominal forcing application time from the uniform TIEGCM mtime schedule"
    )
    output["gamera_source_time"].attrs["description"] = (
        "exact GAMERA history time retained as provenance; Kaiju SM transformations "
        "round it to the nearest whole second"
    )
    output["remix_source_time"].attrs["description"] = "exact coupled ReMIX history time"

    for source, times in {"gamera": gamera_times, "remix": remix_times}.items():
        name = f"{source}_time_offset_seconds"
        output.create_dataset(name, data=_time_offsets_seconds(times, nominal_times))
        output[name].attrs["units"] = "s"
        output[name].attrs["description"] = f"{source.upper()} source time minus nominal time"

    output.attrs["time_axis"] = MAGE_TIME_AXIS
    output.attrs["source_time_tolerance_seconds"] = MAGE_SOURCE_TIME_TOLERANCE_SECONDS


def _write_static_datasets(
    output: h5py.File,
    gamera_reference_time: dt.datetime,
    ionosphere_lat: np.ndarray,
    ionosphere_lon: np.ndarray,
    inner_lat: np.ndarray,
    inner_lon: np.ndarray,
    inner_r: np.ndarray,
    inner_solid_angle: np.ndarray,
    settings: PreparationSettings,
    gamera_run_dir: Path,
    length_scale_m: float,
    mag_m0_nT: float,
    tiegcm_path: Path,
    remix_equatorward_sm_latitude: float,
) -> None:
    """Write static datasets and metadata."""
    output.attrs["kind"] = MAGE_FORCING_KIND
    output.attrs["version"] = MAGE_FORCING_VERSION
    output.attrs["complete"] = False
    output.create_dataset("ionosphere_lat", data=ionosphere_lat)
    output.create_dataset("ionosphere_lon", data=ionosphere_lon)
    output.create_dataset("boundary_lat", data=inner_lat)
    output.create_dataset("boundary_lon", data=inner_lon)
    output.create_dataset("boundary_radius", data=inner_r)
    output.create_dataset("boundary_solid_angle", data=inner_solid_angle)
    for name in ("ionosphere_lat", "ionosphere_lon", "boundary_lat", "boundary_lon"):
        output[name].attrs["units"] = "degree"
    output["boundary_radius"].attrs["units"] = "m"
    output["boundary_radius"].attrs["description"] = (
        "radius of the Kaiju volume-barycentric GAMERA boundary cell center"
    )
    output["boundary_solid_angle"].attrs["units"] = "sr"
    output["boundary_solid_angle"].attrs["description"] = (
        "cell solid angle from the true GAMERA inner-boundary vertices"
    )
    output.attrs["gamera_run_dir"] = str(gamera_run_dir)
    output.attrs["tiegcm_nc"] = str(tiegcm_path)
    output.attrs["tiegcm_conductance_integration"] = (
        "radial_geometric_height_with_lower_dynamo_extension"
    )
    output.attrs["tiegcm_dynamo_bottom_ilev"] = TIEGCM_DYNAMO_BOTTOM_ILEV
    output.attrs["tiegcm_dynamo_reference_height_m"] = TIEGCM_DYNAMO_REFERENCE_HEIGHT_M
    output.attrs["tiegcm_pedersen_lower_scale_m"] = TIEGCM_PEDERSEN_LOWER_SCALE_M
    output.attrs["tiegcm_hall_lower_scale_m"] = TIEGCM_HALL_LOWER_SCALE_M
    output.attrs["conductance_floor_model"] = CONDUCTANCE_FLOOR_MODEL
    output.attrs["pedersen_conductance_floor_S"] = PEDERSEN_CONDUCTANCE_FLOOR_S
    output.attrs["hall_conductance_floor_S"] = HALL_CONDUCTANCE_FLOOR_S
    output.attrs["remix_grid_equatorward_sm_latitude_deg"] = float(remix_equatorward_sm_latitude)
    output.attrs["tiegcm_vertical_grid"] = (
        "SIGMA_PED/SIGMA_HAL and UN/VN at lev[:-1], with dz=diff(ZG at ilev); "
        "terminal fill-only lev omitted; below the first saved interface, conductivity is "
        "continued against Z to ilev=-8.5 at 90 km using TIEGCM pdynamo scale lengths, "
        "radial thickness uses the corresponding ZG intervals, and the lowest winds are "
        "held constant"
    )
    output.attrs["coordinate_system"] = "GEO"
    output.attrs["longitude_convention"] = "east_positive_degrees"
    output.attrs["tiegcm_source_coordinate_system"] = "geographic"
    output.attrs["ionosphere_radius_m"] = IONOSPHERE_RADIUS_M
    output.attrs["wind_weighting"] = (
        "u_p = integral(sigma_P*u*dr)/SP and u_h = integral(sigma_H*u*dr)/SH; "
        "components are geographic south/east on the native TIEGCM grid. Where a "
        "global background minimum raises a conductance, its unresolved conductivity "
        "is assumed to share the corresponding TIEGCM conductivity-weighted mean wind"
    )
    output.attrs["remix_tag"] = settings.tag
    output.attrs["fac_convention"] = "upward"
    output.attrs["fac_source"] = (
        "Kaiju ReMIX Field-aligned current NORTH/SOUTH, converted from "
        "parallel-positive to upward-positive"
    )
    output.attrs["radial_current_convention"] = "outward"
    output.attrs["fac_to_radial_current"] = "jr = FAC_upward * abs(source unit_br)"
    output.attrs["remix_fac_interpolation"] = "kaiju_native_periodic"
    output.attrs["gamera_boundary_interpolation"] = (
        "gamera_native_periodic_bilinear_with_polar_mean"
    )
    output.attrs["gamera_sm_transform_time_convention"] = "kaiju_mjdrecalc_nearest_second"
    output.attrs["gamera_inner_index"] = int(settings.inner_index)
    output.attrs["gamera_length_scale_m"] = float(length_scale_m)
    output.attrs["gamera_background_reference"] = "cell_volume_average_split_B0"
    output.attrs["gamera_B_output"] = (
        "Kaiju cell-volume-average total Bx/By/Bz; delta_Br removes the matching "
        "cell-volume-average split B0, not point-sampled BxD/ByD/BzD"
    )
    for name, value in _centered_dipole_alignment_attrs(gamera_reference_time, mag_m0_nT).items():
        output.attrs[name] = value
    output.attrs["gamera_mag_m0_nT"] = float(mag_m0_nT)
    output.attrs["main_field_B0_T"] = _pynamit_dipole_B0_T(mag_m0_nT, length_scale_m)
    output.attrs["main_field_B0_reference_radius_m"] = RE


def _validate_settings(settings: PreparationSettings) -> None:
    """Validate MAGE preparation settings."""
    if settings.compression not in ("lzf", "gzip", "none"):
        raise ValueError(
            f"compression must be 'lzf', 'gzip', or 'none'; got {settings.compression!r}."
        )
    if isinstance(settings.inner_index, (bool, np.bool_)):
        raise ValueError("inner_index must be a non-negative integer.")
    try:
        inner_index = operator.index(settings.inner_index)
    except TypeError as exc:
        raise ValueError("inner_index must be a non-negative integer.") from exc
    if inner_index < 0:
        raise ValueError(f"inner_index must be non-negative; got {settings.inner_index}.")
    if settings.max_steps is not None:
        if isinstance(settings.max_steps, (bool, np.bool_)):
            raise ValueError("max_steps must be a positive integer.")
        try:
            max_steps = operator.index(settings.max_steps)
        except TypeError as exc:
            raise ValueError("max_steps must be a positive integer.") from exc
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {settings.max_steps}.")


def prepare_forcing(settings: PreparationSettings) -> Path:
    """Prepare the HDF5 forcing file."""
    _validate_settings(settings)
    from netCDF4 import Dataset

    try:
        import kaipy.gamera.magsphere as msph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MAGE forcing preparation needs kaipy to read GAMERA files. "
            f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
            "where kaipy and its dependencies are installed."
        ) from exc

    gamera_run_dir = Path(settings.gamera_directory).expanduser()
    if not gamera_run_dir.is_dir():
        raise FileNotFoundError(f"GAMERA directory does not exist: {gamera_run_dir}")
    tiegcm_path = _resolve_tiegcm_path(gamera_run_dir, settings.tiegcm_path)
    remix_file = gamera_run_dir / f"{settings.tag}.mix.h5"
    if not remix_file.is_file():
        raise FileNotFoundError(f"REMIX file does not exist: {remix_file}")
    output_path = Path(settings.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using GAMERA directory: {gamera_run_dir}", flush=True)
    print(f"Using TIEGCM file: {tiegcm_path}", flush=True)
    print(f"Using REMIX file: {remix_file}", flush=True)
    print(f"Writing prepared forcing: {output_path}", flush=True)

    gsph = msph.GamsphPipe(str(gamera_run_dir), settings.tag, doFast=False)
    if settings.inner_index >= gsph.X.shape[0] - 1:
        raise ValueError(
            f"inner_index must be between 0 and {gsph.X.shape[0] - 2}; got {settings.inner_index}."
        )
    length_scale_m = _gamera_length_scale_m(gsph)
    mag_m0_nT = _gamera_dipole_strength_nT(gsph)
    bx0, by0, bz0 = _gamera_background_field(gsph, settings.inner_index)
    print(f"Using GAMERA length scale: {length_scale_m:.6g} m", flush=True)
    axes = _gamera_internal_dipole_axes(mag_m0_nT)
    print(f"Using GAMERA dipole MagM0: {mag_m0_nT:.6g} nT", flush=True)
    print(
        "GAMERA internal moment axis: "
        f"{axes['moment_axis'][0]:.3g}, {axes['moment_axis'][1]:.3g}, "
        f"{axes['moment_axis'][2]:.3g}; magnetic north axis: "
        f"{axes['north_axis'][0]:.3g}, {axes['north_axis'][1]:.3g}, "
        f"{axes['north_axis'][2]:.3g}",
        flush=True,
    )
    print(f"Using GAMERA inner index: {settings.inner_index}", flush=True)
    if not getattr(gsph, "hasMJD", False):
        raise RuntimeError("GAMERA forcing must provide MJD time metadata.")
    n_steps = len(gsph.MJDs) - 1
    if settings.max_steps is not None:
        n_steps = min(n_steps, int(settings.max_steps))
    if n_steps <= 0:
        raise RuntimeError("GAMERA contains no forcing steps after its initial state.")
    _validate_gamera_dynamic_field_units(gsph, gsph.s0 + 1)

    with Dataset(tiegcm_path, mode="r") as tiegcm:
        _validate_tiegcm_variables(tiegcm, n_steps)
        gamera_times = [_datetime_from_mjd(value) for value in gsph.MJDs[1 : n_steps + 1]]
        nominal_times = _tiegcm_times(tiegcm, gamera_times)
        source_lon = np.asarray(tiegcm.variables["lon"][:], dtype=float)
        source_lat = np.asarray(tiegcm.variables["lat"][:], dtype=float)
        ionosphere_lon, ionosphere_lat = np.meshgrid(wrap_longitude_180(source_lon), source_lat)

        boundary = _gamera_inner_boundary_geometry(gsph, settings.inner_index, length_scale_m)
        boundary_lat, boundary_lon = kaiju_geopack_sm(
            _kaiju_sm_transform_time(gamera_times[0])
        ).sm2geo(boundary.sm_latitude, boundary.sm_longitude)
        boundary_interpolator = _GameraBoundaryInterpolator(
            boundary.sm_latitude, boundary.sm_longitude
        )
        # Kaiju gioH5 writes Bx/By/Bz as total field when
        # Model%doBackground is true, and root Bx0/By0/Bz0 as Gr%B0.
        # PynaMIT needs the perturbation.
        with _RemixRadialCurrentReader(remix_file) as radial_current_reader:
            remix_equatorward_sm_latitude = radial_current_reader.equatorward_sm_latitude
            print(
                "Applying global PynaMIT sheet-conductance floors: "
                f"Pedersen {PEDERSEN_CONDUCTANCE_FLOOR_S:g} S, "
                f"Hall {HALL_CONDUCTANCE_FLOOR_S:g} S",
                flush=True,
            )
            gamera_steps = [gsph.s0 + out_step + 1 for out_step in range(n_steps)]
            remix_times = [
                radial_current_reader.source_time(gamera_step) for gamera_step in gamera_steps
            ]
            gamera_offsets, remix_offsets = _validate_forcing_time_axis(
                nominal_times, gamera_times, remix_times
            )
            print(
                "Canonical forcing clock: TIEGCM mtime; "
                f"GAMERA offsets {gamera_offsets.min():.6g} to {gamera_offsets.max():.6g} s; "
                f"ReMIX offsets {remix_offsets.min():.6g} to {remix_offsets.max():.6g} s",
                flush=True,
            )

            with _atomic_prepared_output(output_path) as output:
                _write_time_axis(output, nominal_times, gamera_times, remix_times)
                _write_static_datasets(
                    output,
                    gamera_times[0],
                    ionosphere_lat,
                    ionosphere_lon,
                    boundary_lat,
                    boundary_lon,
                    boundary.radius_m,
                    boundary.solid_angle,
                    settings,
                    gamera_run_dir,
                    length_scale_m,
                    mag_m0_nT,
                    tiegcm_path,
                    remix_equatorward_sm_latitude,
                )
                _create_output_datasets(
                    output, n_steps, ionosphere_lat.shape, boundary_lat.shape, settings.compression
                )

                for out_step, (gamera_step, gamera_time) in enumerate(
                    zip(gamera_steps, gamera_times, strict=True)
                ):
                    print(
                        f"Preparing step {out_step + 1} of {n_steps}: "
                        f"nominal {nominal_times[out_step].isoformat()}, "
                        f"GAMERA {gamera_time.isoformat()}",
                        flush=True,
                    )

                    integrated = _integrate_tiegcm_step(tiegcm, out_step)
                    integrated["SP"], integrated["SH"] = _apply_conductance_floor(
                        integrated["SP"], integrated["SH"]
                    )
                    for key, values in integrated.items():
                        output[key][out_step] = values

                    output["jr"][out_step] = radial_current_reader.read(
                        gamera_step, ionosphere_lon, ionosphere_lat, gamera_time
                    ).astype(np.float32)

                    bx = gsph.GetVar("Bx", gamera_step)[settings.inner_index] - bx0
                    by = gsph.GetVar("By", gamera_step)[settings.inner_index] - by0
                    bz = gsph.GetVar("Bz", gamera_step)[settings.inner_index] - bz0
                    delta_br_sm = boundary.radial_component(bx, by, bz)
                    boundary_sm_lat, boundary_sm_lon = _geographic_grid_in_sm(
                        boundary_lat, boundary_lon, gamera_time
                    )
                    output["delta_Br"][out_step] = boundary_interpolator.interpolate(
                        delta_br_sm, target_sm_lat=boundary_sm_lat, target_sm_lon=boundary_sm_lon
                    ).astype(np.float32)

    return output_path


__all__ = [
    "IONOSPHERE_RADIUS_M",
    "MAGE_FORCING_KIND",
    "MAGE_FORCING_VERSION",
    "MAGE_SOURCE_TIME_TOLERANCE_SECONDS",
    "MAGE_TIME_AXIS",
    "PreparationSettings",
    "CONDUCTANCE_FLOOR_MODEL",
    "HALL_CONDUCTANCE_FLOOR_S",
    "PEDERSEN_CONDUCTANCE_FLOOR_S",
    "TIEGCM_DYNAMO_BOTTOM_ILEV",
    "TIEGCM_DYNAMO_REFERENCE_HEIGHT_M",
    "TIEGCM_HALL_LOWER_SCALE_M",
    "TIEGCM_PEDERSEN_LOWER_SCALE_M",
    "prepare_forcing",
]
