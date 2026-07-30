"""Project prepared MAGE forcing into reusable PynaMIT inputs.

The workflow owns validation, file lifetime, and provenance. Its
private projector reuses grids, transforms, and fit operators across
forcing times.
"""

from __future__ import annotations

import datetime as dt
import operator
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import pynamit
from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MagneticFieldEvaluation, MainField, decimal_year
from pynamit.math.constants import RE
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    electric_field_from_weighted_winds,
    resistance_from_log_conductance_coordinates,
)
from pynamit.simulation.workflows.mage_preparation import (
    CONDUCTANCE_FLOOR_MODEL,
    HALL_CONDUCTANCE_FLOOR_S,
    IONOSPHERE_RADIUS_M,
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    MAGE_SOURCE_TIME_TOLERANCE_SECONDS,
    MAGE_TIME_AXIS,
    PEDERSEN_CONDUCTANCE_FLOOR_S,
    TIEGCM_DYNAMO_BOTTOM_ILEV,
    TIEGCM_DYNAMO_REFERENCE_HEIGHT_M,
    TIEGCM_HALL_LOWER_SCALE_M,
    TIEGCM_PEDERSEN_LOWER_SCALE_M,
)
from pynamit.simulation.workflows.prepared_inputs import (
    clear_prepared_input_package,
    write_input_manifest,
)
from pynamit.sphere.spherical_transform import grid_sqrt_area_weights

MAGE_MAIN_FIELD_KIND = "kaiju_dipole"

_MAGE_IONOSPHERE_DATASETS = ("jr", "SH", "SP", "u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi")
_MAGE_BOUNDARY_DATASETS = ("delta_Br",)
_MAGE_STATIC_DATASETS = (
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
_MAGE_REQUIRED_ATTRIBUTES = (
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
_MAGE_DATASET_UNITS = {
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


def _matching_grid_shape(h5_file: Any, names: tuple[str, ...], label: str) -> tuple[int, ...]:
    """Return the common non-empty shape of related grid datasets."""
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
    """Require each named dataset on one grid for every forcing time."""
    expected = (n_steps, *spatial_shape)
    for name in names:
        if h5_file[name].shape != expected:
            raise RuntimeError(
                f"Prepared forcing dataset {name!r} has shape {h5_file[name].shape}; "
                f"expected {expected}."
            )


def _validate_prepared_forcing(h5_file: Any) -> None:
    """Validate the prepared-forcing schema before replacing outputs."""
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

    required_datasets = (
        *_MAGE_STATIC_DATASETS,
        *_MAGE_IONOSPHERE_DATASETS,
        *_MAGE_BOUNDARY_DATASETS,
    )
    missing_datasets = [name for name in required_datasets if name not in h5_file]
    missing_attributes = [name for name in _MAGE_REQUIRED_ATTRIBUTES if name not in h5_file.attrs]
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
    equatorward_sm_latitude = float(h5_file.attrs["remix_grid_equatorward_sm_latitude_deg"])
    if not np.isfinite(equatorward_sm_latitude) or not 0.0 < equatorward_sm_latitude < 90.0:
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
        reference_radius, RE, rtol=0.0, atol=1e-6
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
        for name, expected in _MAGE_DATASET_UNITS.items()
        if h5_file[name].attrs.get("units") != expected
    }
    if invalid_units:
        details = ", ".join(
            f"{name}={actual!r} (expected {_MAGE_DATASET_UNITS[name]!r})"
            for name, actual in invalid_units.items()
        )
        raise RuntimeError(f"Prepared forcing has incompatible dataset units: {details}.")

    time_shape = h5_file["time"].shape
    if len(time_shape) != 1 or time_shape[0] == 0:
        raise RuntimeError("Prepared forcing time must be a non-empty one-dimensional dataset.")
    n_steps = time_shape[0]
    _validate_prepared_time_axis(h5_file, n_steps)
    ionosphere_shape = _matching_grid_shape(
        h5_file, ("ionosphere_lat", "ionosphere_lon"), "ionosphere"
    )
    boundary_shape = _matching_grid_shape(
        h5_file,
        ("boundary_lat", "boundary_lon", "boundary_radius", "boundary_solid_angle"),
        "boundary",
    )
    _validate_time_series_shapes(h5_file, _MAGE_IONOSPHERE_DATASETS, n_steps, ionosphere_shape)
    _validate_time_series_shapes(h5_file, _MAGE_BOUNDARY_DATASETS, n_steps, boundary_shape)
    _validate_conductance_floor(h5_file, n_steps)


def _validate_conductance_floor(h5_file: Any, n_steps: int) -> None:
    """Require every global sheet sample to satisfy its hard floor."""
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


def _parse_h5_time(value: Any) -> dt.datetime:
    """Parse an ISO timestamp stored as HDF5 bytes or text."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    timestamp = dt.datetime.fromisoformat(str(value))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return timestamp


def _h5_time_vector_seconds(raw_times: Any) -> tuple[list[dt.datetime], np.ndarray]:
    """Return forcing times and seconds relative to the first entry."""
    parsed_times = [_parse_h5_time(value) for value in raw_times]
    if not parsed_times:
        raise ValueError("Forcing HDF5 time dataset is empty.")
    event_time = parsed_times[0]
    relative_seconds = np.array(
        [(time_value - event_time).total_seconds() for time_value in parsed_times], dtype=float
    )
    if np.any(~np.isfinite(relative_seconds)):
        raise ValueError("Forcing HDF5 time dataset produced non-finite relative seconds.")
    if np.any(np.diff(relative_seconds) <= 0.0):
        raise ValueError("Forcing HDF5 time dataset must be strictly increasing.")
    return parsed_times, relative_seconds


def _validate_prepared_time_axis(h5_file: Any, n_steps: int) -> None:
    """Validate the nominal clock and exact source-time provenance."""
    timestamp_names = ("time", "gamera_source_time", "remix_source_time")
    for name in timestamp_names:
        if h5_file[name].shape != (n_steps,):
            raise RuntimeError(
                f"Prepared forcing dataset {name!r} has shape {h5_file[name].shape}; "
                f"expected {(n_steps,)}."
            )

    try:
        nominal_times, nominal_seconds = _h5_time_vector_seconds(h5_file["time"][:])
        gamera_times, _ = _h5_time_vector_seconds(h5_file["gamera_source_time"][:])
        remix_times, _ = _h5_time_vector_seconds(h5_file["remix_source_time"][:])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Prepared forcing time metadata is invalid: {exc}") from exc

    nominal_intervals = np.diff(nominal_seconds)
    if nominal_intervals.size > 1 and not np.allclose(
        nominal_intervals, nominal_intervals[0], rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("Prepared forcing nominal time axis must have a uniform cadence.")

    source_times = {"gamera": gamera_times, "remix": remix_times}
    for source, times in source_times.items():
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


def _input_cadence(relative_seconds: np.ndarray) -> dict[str, float | None]:
    """Return compact cadence metadata for projected input times."""
    if relative_seconds.size < 2:
        return {"input_dt_median_s": None, "input_dt_min_s": None, "input_dt_max_s": None}
    intervals = np.diff(relative_seconds)
    return {
        "input_dt_median_s": float(np.median(intervals)),
        "input_dt_min_s": float(np.min(intervals)),
        "input_dt_max_s": float(np.max(intervals)),
    }


def _boundary_radius(h5_file: Any, explicit_radius: float | None) -> float:
    """Return the area-weighted radius used for Br fitting."""
    if explicit_radius is not None:
        mean_radius = float(explicit_radius)
        if not np.isfinite(mean_radius) or mean_radius <= 0.0:
            raise ValueError("The explicit boundary radius must be finite and positive.")
        return mean_radius
    if "boundary_radius" not in h5_file:
        raise RuntimeError(
            "Prepared MAGE forcing is missing the 'boundary_radius' dataset. "
            "Regenerate it with scripts/simulation/mage_prepare.py or set "
            "SETTINGS.boundary_radius explicitly in mage_project.py."
        )
    radius = np.asarray(h5_file["boundary_radius"][:], dtype=float)
    if np.any(~np.isfinite(radius)):
        raise RuntimeError(
            "Prepared MAGE forcing dataset 'boundary_radius' contains non-finite values. "
            "Regenerate the forcing or set SETTINGS.boundary_radius explicitly."
        )
    if np.any(radius <= 0.0):
        raise RuntimeError("Prepared MAGE forcing boundary radii must be positive.")
    if "boundary_solid_angle" not in h5_file:
        raise RuntimeError("Prepared MAGE forcing is missing boundary_solid_angle.")
    solid_angle = np.asarray(h5_file["boundary_solid_angle"][:], dtype=float)
    if solid_angle.shape != radius.shape:
        raise RuntimeError("Prepared boundary radii and solid angles must have matching shapes.")
    if np.any(~np.isfinite(solid_angle)) or np.any(solid_angle <= 0.0):
        raise RuntimeError("Prepared MAGE boundary solid angles must be finite and positive.")
    mean_radius = float(np.average(radius, weights=solid_angle))
    relative_spread = float((np.max(radius) - np.min(radius)) / mean_radius)
    if relative_spread > 1e-3:
        print(
            "Warning: Br grid radius varies by "
            f"{relative_spread:.3%}; using solid-angle-weighted RM={mean_radius:.6g} m.",
            flush=True,
        )
    return mean_radius


def _dipole_B0(h5_file: Any, explicit_B0: float | None) -> float:
    """Return the centered-dipole equatorial field in tesla."""
    if explicit_B0 is not None:
        field_strength = float(explicit_B0)
    elif "main_field_B0_T" in h5_file.attrs:
        field_strength = float(h5_file.attrs["main_field_B0_T"])
    else:
        raise RuntimeError(
            "Prepared MAGE forcing is missing dipole strength metadata "
            "'main_field_B0_T'. Regenerate it with "
            "scripts/simulation/mage_prepare.py or set SETTINGS.dipole_B0 "
            "explicitly in mage_project.py."
        )
    if not np.isfinite(field_strength) or field_strength <= 0.0:
        raise ValueError("The centered-dipole field strength must be finite and positive.")
    return field_strength


def _gamera_dipole_metadata(h5_file: Any) -> dict[str, np.ndarray | float]:
    """Return signed GAMERA dipole metadata from prepared forcing."""
    required = (
        "gamera_mag_m0_nT",
        "gamera_internal_dipole_moment_axis",
        "gamera_internal_magnetic_north_axis",
    )
    missing = [name for name in required if name not in h5_file.attrs]
    if missing:
        raise RuntimeError(
            "Prepared MAGE forcing is missing GAMERA dipole metadata "
            f"{missing}. Regenerate it with scripts/simulation/mage_prepare.py."
        )

    def normalized_axis(name: str) -> np.ndarray:
        axis = np.asarray(h5_file.attrs[name], dtype=float)
        norm = np.linalg.norm(axis)
        if axis.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                f"Prepared MAGE forcing metadata {name!r} must be a finite 3-vector."
            )
        axis = axis / norm
        axis[np.isclose(axis, 0.0)] = 0.0
        return axis

    mag_m0_nT = float(h5_file.attrs["gamera_mag_m0_nT"])
    if not np.isfinite(mag_m0_nT) or mag_m0_nT == 0.0:
        raise RuntimeError("Prepared GAMERA MagM0 must be finite and nonzero.")
    moment_axis = normalized_axis("gamera_internal_dipole_moment_axis")
    north_axis = normalized_axis("gamera_internal_magnetic_north_axis")
    if not np.allclose(north_axis, -moment_axis, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Prepared GAMERA magnetic-north and dipole-moment axes must be antiparallel."
        )
    expected_moment = np.array([0.0, 0.0, np.sign(mag_m0_nT)])
    if not np.allclose(moment_axis, expected_moment, rtol=0.0, atol=1e-12):
        raise RuntimeError("Prepared GAMERA MagM0 and dipole-moment axis are inconsistent.")
    if not np.allclose(north_axis, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        raise RuntimeError("MAGE projection currently requires GAMERA SM magnetic north along +Z.")
    return {"mag_m0_nT": mag_m0_nT, "moment_axis": moment_axis, "north_axis": north_axis}


def _load_weighted_winds(
    h5_file: Any, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load model-basis weighted winds for one forcing step."""
    required = ("u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi")
    missing = [name for name in required if name not in h5_file]
    if missing:
        raise RuntimeError(
            "Prepared MAGE forcing is missing required weighted-wind dataset(s) "
            f"{missing}. Regenerate it with scripts/simulation/mage_prepare.py; "
            "the projection step cannot reconstruct them."
        )
    return tuple(np.asarray(h5_file[name][step], dtype=float) for name in required)


def _source_file_metadata(path: str | Path) -> dict[str, int | str]:
    """Return cheap provenance metadata for a source file."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _print_field_stats(label: str, values: np.ndarray) -> None:
    """Print compact finite-value diagnostics."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        print(f"{label}: no finite values", flush=True)
        return
    sample = values[finite]
    print(
        f"{label}: min={sample.min():.6g}, rms={np.sqrt(np.mean(sample**2)):.6g}, "
        f"max={sample.max():.6g}",
        flush=True,
    )


def _clear_existing_input_package(directory: Path, artifact_storage: str) -> None:
    """Clear generated PynaMIT artifacts before reprojection."""
    artifact_names = clear_prepared_input_package(directory, artifact_storage=artifact_storage)
    (directory / "mage_input_metadata.json").unlink(missing_ok=True)
    if artifact_names:
        names = ", ".join(artifact_names)
        print(f"Replacing existing PynaMIT artifacts: {names}", flush=True)


@contextmanager
def _staged_input_package(directory: Path, artifact_storage: str):
    """Publish a projected package only after complete construction."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-projecting-", dir=directory.parent
    ) as temporary_directory:
        staged_directory = Path(temporary_directory)
        yield staged_directory

        directory.mkdir(parents=True, exist_ok=True)
        _clear_existing_input_package(directory, artifact_storage)
        for staged_path in staged_directory.iterdir():
            staged_path.replace(directory / staged_path.name)


class _MageInputProjector:
    """Project MAGE fields through one fixed Earth-attached geometry.

    Geometry, field evaluation, and least-squares weights do not vary
    with forcing time. Keeping them together makes their lifecycle clear
    and avoids rebuilding operators for every input step. The
    surrounding workflow retains ownership of files and provenance.
    """

    def __init__(
        self,
        *,
        simulation: pynamit.Simulation,
        ionosphere_grid: pynamit.Grid,
        magnetosphere_grid: pynamit.Grid,
        boundary_Br_lambda: float,
        conductance_lambda: float,
        boundary_jr_lambda: float,
        e_neutral_wind_lambda: float,
    ) -> None:
        self._simulation = simulation
        self._ionosphere_grid = ionosphere_grid
        self._magnetosphere_grid = magnetosphere_grid
        self._boundary_Br_lambda = boundary_Br_lambda
        self._conductance_lambda = conductance_lambda
        self._boundary_jr_lambda = boundary_jr_lambda
        self._e_neutral_wind_lambda = e_neutral_wind_lambda

        self._ionosphere_field = MagneticFieldEvaluation(
            simulation.geometry.main_field, ionosphere_grid, IONOSPHERE_RADIUS_M
        )
        self._magnetosphere_sqrt_weights = grid_sqrt_area_weights(magnetosphere_grid)
        self._ionosphere_sqrt_weights = grid_sqrt_area_weights(ionosphere_grid)
        self._ionosphere_tangential_sqrt_weights = np.tile(self._ionosphere_sqrt_weights, (2, 1))
        conductance_space = simulation.run_data.schema.input_field_spaces["conductance"]
        self._conductance_evaluator = (
            conductance_space.representation.get_scalar_evaluation_operator(ionosphere_grid)
        )

    def project_step(self, h5_file: Any, step: int, input_time: float) -> None:
        """Project every forcing field for one source time step."""
        self._project_boundary_br(h5_file, step, input_time)
        self._project_radial_current(h5_file, step, input_time)
        sigma_p, sigma_h = self._project_conductance(h5_file, step, input_time)
        self._project_wind_source(h5_file, step, input_time, sigma_p, sigma_h)

    def _project_boundary_br(self, h5_file: Any, step: int, input_time: float) -> None:
        """Project magnetospheric inner-boundary radial field."""
        delta_br = np.asarray(h5_file["delta_Br"][step], dtype=float).reshape(-1) * 1e-9
        if np.any(~np.isfinite(delta_br)):
            raise ValueError("Br input contains non-finite values.")
        _print_field_stats("  Delta Br [T]", delta_br)
        self._simulation.set_boundary_Br(
            delta_br,
            lat=self._magnetosphere_grid.lat,
            lon=self._magnetosphere_grid.lon,
            time=input_time,
            sqrt_weights=self._magnetosphere_sqrt_weights,
            reg_lambda=self._boundary_Br_lambda,
        )

    def _project_radial_current(self, h5_file: Any, step: int, input_time: float) -> None:
        """Project prepared outward radial current density."""
        boundary_jr = np.asarray(h5_file["jr"][step], dtype=float).reshape(-1) * 1e-6
        if np.any(~np.isfinite(boundary_jr)):
            raise ValueError("Prepared radial current contains non-finite values.")
        _print_field_stats("  boundary jr [A/m^2]", boundary_jr)
        self._simulation.set_boundary_jr(
            boundary_jr,
            lat=self._ionosphere_grid.lat,
            lon=self._ionosphere_grid.lon,
            time=input_time,
            sqrt_weights=self._ionosphere_sqrt_weights,
            reg_lambda=self._boundary_jr_lambda,
        )

    def _project_conductance(
        self, h5_file: Any, step: int, input_time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project and return sampled Pedersen/Hall conductance."""
        sigma_h = np.asarray(h5_file["SH"][step], dtype=float).reshape(-1)
        sigma_p = np.asarray(h5_file["SP"][step], dtype=float).reshape(-1)
        if np.any(~np.isfinite(sigma_h)) or np.any(sigma_h < 0.0):
            raise ValueError("Hall conductance contains non-finite or negative values.")
        if np.any(~np.isfinite(sigma_p)) or np.any(sigma_p < 0.0):
            raise ValueError("Pedersen conductance contains non-finite or negative values.")
        if np.any(sigma_p**2 + sigma_h**2 <= np.finfo(float).tiny):
            raise ValueError("Pedersen and Hall conductance cannot both be zero.")
        _print_field_stats("  Hall conductance [S]", sigma_h)
        _print_field_stats("  Pedersen conductance [S]", sigma_p)
        self._simulation.set_conductance(
            sigma_h,
            sigma_p,
            lat=self._ionosphere_grid.lat,
            lon=self._ionosphere_grid.lon,
            time=input_time,
            sqrt_weights=self._ionosphere_sqrt_weights,
            reg_lambda=self._conductance_lambda,
        )
        return sigma_p, sigma_h

    def _projected_resistance(self, input_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct fitted sheet resistance on the forcing grid."""
        input_series = self._simulation.run_data.input_series
        conductance_entry = input_series.get_entry("conductance", input_time)
        if conductance_entry is None:
            raise RuntimeError("Conductance must be set before computing wind-driven E.")
        log_magnitude = np.asarray(
            self._conductance_evaluator.matvec(conductance_entry["log_conductance_magnitude"])
        ).reshape(-1)
        log_ratio = np.asarray(
            self._conductance_evaluator.matvec(conductance_entry["log_hall_to_pedersen_ratio"])
        ).reshape(-1)
        return resistance_from_log_conductance_coordinates(log_magnitude, log_ratio)

    def _project_wind_source(
        self, h5_file: Any, step: int, input_time: float, sigma_p: np.ndarray, sigma_h: np.ndarray
    ) -> None:
        """Project equator-safe E from the integrated wind current."""
        u_p_theta, u_p_phi, u_h_theta, u_h_phi = _load_weighted_winds(h5_file, step)
        u_p_theta = np.asarray(u_p_theta, dtype=float).reshape(-1)
        u_p_phi = np.asarray(u_p_phi, dtype=float).reshape(-1)
        _print_field_stats("  Pedersen-weighted wind speed [m/s]", np.hypot(u_p_theta, u_p_phi))
        u_h_theta = np.asarray(u_h_theta, dtype=float).reshape(-1)
        u_h_phi = np.asarray(u_h_phi, dtype=float).reshape(-1)
        _print_field_stats("  Hall-weighted wind speed [m/s]", np.hypot(u_h_theta, u_h_phi))

        eta_p, eta_h = self._projected_resistance(input_time)
        wind_driven_e_theta, wind_driven_e_phi = electric_field_from_weighted_winds(
            sigma_p=sigma_p,
            sigma_h=sigma_h,
            u_p_theta=u_p_theta,
            u_p_phi=u_p_phi,
            u_h_theta=u_h_theta,
            u_h_phi=u_h_phi,
            field=self._ionosphere_field,
            eta_p=eta_p,
            eta_h=eta_h,
        )
        _print_field_stats(
            "  Wind-driven E [V/m]", np.hypot(wind_driven_e_theta, wind_driven_e_phi)
        )
        self._simulation.set_E_neutral_wind(
            E_neutral_wind_theta=wind_driven_e_theta,
            E_neutral_wind_phi=wind_driven_e_phi,
            lat=self._ionosphere_grid.lat,
            lon=self._ionosphere_grid.lon,
            time=input_time,
            sqrt_weights=self._ionosphere_tangential_sqrt_weights,
            reg_lambda=self._e_neutral_wind_lambda,
        )


def project_inputs(
    *,
    forcing_path: str | Path,
    projection_directory: str | Path,
    dipole_B0_override: float | None,
    boundary_radius_override: float | None,
    nmax: int,
    mmax: int,
    ncs: int,
    max_steps: int | None,
    boundary_Br_lambda: float,
    conductance_lambda: float,
    boundary_jr_lambda: float,
    e_neutral_wind_lambda: float,
    artifact_storage: str,
    operator_cache_directory: str | Path | None = None,
) -> Path:
    """Project one prepared MAGE forcing file into PynaMIT inputs."""
    if max_steps is not None:
        if isinstance(max_steps, (bool, np.bool_)):
            raise ValueError("max_steps must be a positive integer.")
        try:
            max_steps = operator.index(max_steps)
        except TypeError as exc:
            raise ValueError("max_steps must be a positive integer.") from exc
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}.")

    forcing_path = Path(forcing_path).expanduser()
    if not forcing_path.is_file():
        raise FileNotFoundError(
            f"Prepared forcing does not exist: {forcing_path}. "
            "Run scripts/simulation/mage_prepare.py or provide the correct path."
        )

    import h5py

    projection_directory = Path(projection_directory).expanduser()

    with _staged_input_package(projection_directory, artifact_storage) as staged_directory:
        with h5py.File(forcing_path, "r") as file:
            _validate_prepared_forcing(file)
            nominal_times, input_times = _h5_time_vector_seconds(file["time"][:])
            gamera_source_times = [
                _parse_h5_time(value) for value in file["gamera_source_time"][:]
            ]
            remix_source_times = [_parse_h5_time(value) for value in file["remix_source_time"][:]]
            gamera_time_offsets = np.asarray(file["gamera_time_offset_seconds"][:], dtype=float)
            remix_time_offsets = np.asarray(file["remix_time_offset_seconds"][:], dtype=float)
            event_time = nominal_times[0]
            dipole_epoch = decimal_year(event_time)
            boundary_radius = _boundary_radius(file, boundary_radius_override)
            dipole_B0 = _dipole_B0(file, dipole_B0_override)
            main_field = MainField(kind=MAGE_MAIN_FIELD_KIND, epoch=dipole_epoch, B0=dipole_B0)
            gamera_dipole = _gamera_dipole_metadata(file)
            alignment = main_field.alignment_metadata(event_time)
            ionosphere_lat = np.asarray(file["ionosphere_lat"][:], dtype=float)
            ionosphere_lon = wrap_longitude_180(file["ionosphere_lon"][:])
            ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)

            magnetosphere_lat = np.asarray(file["boundary_lat"][:], dtype=float)
            magnetosphere_lon = wrap_longitude_180(file["boundary_lon"][:])
            boundary_solid_angle = np.asarray(file["boundary_solid_angle"][:], dtype=float)
            if np.any(~np.isfinite(boundary_solid_angle)) or np.any(boundary_solid_angle <= 0.0):
                raise RuntimeError(
                    "Prepared MAGE boundary solid angles must be finite and positive."
                )
            if not np.isclose(np.sum(boundary_solid_angle), 4.0 * np.pi, rtol=1e-6, atol=1e-10):
                raise RuntimeError(
                    "Prepared MAGE boundary solid angles must cover the complete sphere."
                )
            magnetosphere_grid = pynamit.Grid(
                lat=magnetosphere_lat, lon=magnetosphere_lon, area_weights=boundary_solid_angle
            )

            print(f"Using forcing file: {forcing_path}", flush=True)
            print(f"Writing projected input package: {projection_directory}", flush=True)
            print(f"Nominal event time: {event_time.isoformat()}", flush=True)
            print(
                "Nominal forcing time span: "
                f"{nominal_times[0].isoformat()} to {nominal_times[-1].isoformat()} "
                f"({len(nominal_times)} step(s))",
                flush=True,
            )
            print(
                "Exact source offsets from nominal time: "
                f"GAMERA {gamera_time_offsets.min():.6g} to "
                f"{gamera_time_offsets.max():.6g} s; "
                f"ReMIX {remix_time_offsets.min():.6g} to "
                f"{remix_time_offsets.max():.6g} s",
                flush=True,
            )
            print(f"Main field used for projection: {MAGE_MAIN_FIELD_KIND}", flush=True)
            print(f"Dipole alignment model: {alignment['dipole_alignment_model']}", flush=True)
            print(f"Main-field epoch: {dipole_epoch:.9f}", flush=True)
            print(
                "Geopack coefficient epoch: "
                f"{alignment['dipole_alignment_epoch']:.9f} "
                "(Kaiju day-of-year interpolation)",
                flush=True,
            )
            print(
                f"PynaMIT-reference dipole B0: {dipole_B0:.6g} T ({dipole_B0 * 1e9:.6g} nT)",
                flush=True,
            )
            print(f"GAMERA signed MagM0: {gamera_dipole['mag_m0_nT']:.6g} nT", flush=True)
            print(
                "GAMERA source coordinates: SM using Kaiju's nearest-second MJDRecalc time",
                flush=True,
            )
            print("PynaMIT model and prepared forcing coordinates: Earth-fixed GEO", flush=True)
            print(f"RM: {boundary_radius:.6g} m", flush=True)
            print(
                "Neutral-wind forcing: equivalent E from Pedersen/Hall-weighted winds", flush=True
            )
            print(
                "Global sheet-conductance floors: "
                f"Pedersen {PEDERSEN_CONDUCTANCE_FLOOR_S:g} S, "
                f"Hall {HALL_CONDUCTANCE_FLOOR_S:g} S",
                flush=True,
            )
            print("REMIX FAC source convention: upward positive", flush=True)
            print("Prepared jr convention: outward positive", flush=True)

            simulation = pynamit.Simulation(
                run_directory=staged_directory,
                Nmax=nmax,
                Mmax=mmax,
                Ncs=ncs,
                RI=IONOSPHERE_RADIUS_M,
                RM=boundary_radius,
                main_field_kind=MAGE_MAIN_FIELD_KIND,
                main_field_epoch=dipole_epoch,
                main_field_B0=dipole_B0,
                enable_pfac_coupling=False,
                t0=str(event_time),
                artifact_storage=artifact_storage,
                operator_cache_directory=operator_cache_directory,
            )
            projector = _MageInputProjector(
                simulation=simulation,
                ionosphere_grid=ionosphere_grid,
                magnetosphere_grid=magnetosphere_grid,
                boundary_Br_lambda=boundary_Br_lambda,
                conductance_lambda=conductance_lambda,
                boundary_jr_lambda=boundary_jr_lambda,
                e_neutral_wind_lambda=e_neutral_wind_lambda,
            )

            if max_steps is not None:
                input_times = input_times[: int(max_steps)]
                nominal_times = nominal_times[: int(max_steps)]
                gamera_source_times = gamera_source_times[: int(max_steps)]
                remix_source_times = remix_source_times[: int(max_steps)]
                gamera_time_offsets = gamera_time_offsets[: int(max_steps)]
                remix_time_offsets = remix_time_offsets[: int(max_steps)]
            n_steps = input_times.size
            if n_steps == 0:
                raise ValueError("No forcing time steps selected for projection.")

            for step in range(n_steps):
                input_time = float(input_times[step])
                print(
                    f"Projecting input step {step + 1} of {n_steps} "
                    f"at t={input_time:g} s ({nominal_times[step].isoformat()})",
                    flush=True,
                )
                projector.project_step(file, step, input_time)

            projected_datasets = [
                key
                for key in simulation.run_data.schema.input_variables
                if key in simulation.run_data.input_series.datasets
            ]
            source_tiegcm = file.attrs.get("tiegcm_nc", None)
            if isinstance(source_tiegcm, bytes):
                source_tiegcm = source_tiegcm.decode("utf-8", errors="replace")
            sources = {
                "prepared_forcing": _source_file_metadata(forcing_path),
                "tiegcm": None if source_tiegcm is None else str(source_tiegcm),
            }
            input_time_metadata = {
                "time_axis": str(file.attrs["time_axis"]),
                "nominal_time_first": nominal_times[0].isoformat(),
                "nominal_time_last": nominal_times[-1].isoformat(),
                "gamera_source_time_first": gamera_source_times[0].isoformat(),
                "gamera_source_time_last": gamera_source_times[-1].isoformat(),
                "remix_source_time_first": remix_source_times[0].isoformat(),
                "remix_source_time_last": remix_source_times[-1].isoformat(),
                "gamera_time_offset_min_s": float(np.min(gamera_time_offsets)),
                "gamera_time_offset_max_s": float(np.max(gamera_time_offsets)),
                "remix_time_offset_min_s": float(np.min(remix_time_offsets)),
                "remix_time_offset_max_s": float(np.max(remix_time_offsets)),
                "input_time_first_s": float(input_times[0]),
                "input_time_last_s": float(input_times[-1]),
                **_input_cadence(input_times),
            }
            write_input_manifest(
                staged_directory,
                simulation.run_data.config,
                input_datasets=projected_datasets,
                source="pynamit.simulation.workflows.mage_projection",
                notes=(
                    "The stored E_neutral_wind input is the equivalent electric-field "
                    "contribution derived from Pedersen/Hall-weighted neutral winds "
                    "using the sheet-radius main field and projected sheet resistance; "
                    "it is not total model E.",
                    "This E formulation is algebraically equivalent to the Appendix-A "
                    "Q_eff formulation away from the dip equator but remains finite at "
                    "the equator because it never divides by the radial field component.",
                    "All input fits used explicit square-root surface-area weights.",
                ),
                metadata={
                    "input_kind": "mage_gamera_tiegcm",
                    "event_time": event_time.isoformat(),
                    "coordinate_system": "GEO",
                    "source_coordinate_systems": {
                        "GAMERA_and_REMIX": "SM",
                        "TIEGCM": "geographic",
                    },
                    "gamera_sm_transform_time_convention": str(
                        file.attrs["gamera_sm_transform_time_convention"]
                    ),
                    "fac_convention": "upward",
                    "fac_to_radial_current": "jr = FAC_upward * abs(source unit_br)",
                    "least_squares_weighting": "surface_area",
                    "conductance_floor": {
                        "model": str(file.attrs["conductance_floor_model"]),
                        "pedersen_S": float(file.attrs["pedersen_conductance_floor_S"]),
                        "hall_S": float(file.attrs["hall_conductance_floor_S"]),
                        "domain": "global",
                    },
                    "remix_fac_equatorward_sm_latitude_deg": float(
                        file.attrs["remix_grid_equatorward_sm_latitude_deg"]
                    ),
                    "projection_regularization": {
                        "boundary_Br_lambda": boundary_Br_lambda,
                        "conductance_lambda": conductance_lambda,
                        "boundary_jr_lambda": boundary_jr_lambda,
                        "E_neutral_wind_lambda": e_neutral_wind_lambda,
                    },
                    "n_projected_steps": n_steps,
                    "sources": sources,
                    "gamera_dipole": gamera_dipole,
                    "alignment": alignment,
                    **input_time_metadata,
                },
            )

    print(f"Projected input package written to {projection_directory}", flush=True)
    return projection_directory


__all__ = ["MAGE_MAIN_FIELD_KIND", "project_inputs"]
