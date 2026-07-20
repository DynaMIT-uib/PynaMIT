"""Prepare reusable MAGE/GAMERA/TIEGCM forcing.

The expensive TIEGCM height integration is done here once.  The output
HDF5 contains the fields used by the projection step:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S.
- ``We``/``Wn``: Pedersen-weighted eastward/northward wind in m/s.
- ``WeH``/``WnH``: Hall-weighted eastward/northward neutral wind in m/s.
- REMIX FAC and GAMERA inner-boundary radial magnetic perturbation.

The wind integration intentionally stores conductivity-weighted winds,
not a height-resolved ``u x B`` source. The projection step uses
the PynaMIT sheet-radius main field and sheet resistance, matching the
thin-sheet ``JS -> E_S`` closure.

The prepared file is the minimal projection contract, not a diagnostic
archive. It is written atomically so a failed preparation cannot replace
the last complete forcing file.

Typical use on the MAGE machine:

    python scripts/simulation/mage_prepare.py

Edit ``SETTINGS`` below to change paths or run parameters. By default,
the GAMERA directory is ``/disk/Gamera_Dong``. Output is written under
``scripts/simulation/mage_prepared``.
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
from scipy.interpolate import griddata

from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.simulation.workflows.mage_projection import MAGE_FORCING_KIND, MAGE_FORCING_VERSION

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GAMERA_DIRECTORY = Path("/disk/Gamera_Dong")
DEFAULT_OUTPUT_DIRECTORY = SCRIPT_DIR / "mage_prepared"
DEFAULT_OUTPUT_NAME = "mage_prepared_forcing.h5"
DEFAULT_TAG = "msphere"

FALLBACK_EARTH_RADIUS_M = 6371.0e3
FILL_THRESHOLD = 1e30


@dataclass(frozen=True)
class PreparationSettings:
    """Defaults intended to be edited for preparation runs."""

    gamera_directory: Path = DEFAULT_GAMERA_DIRECTORY
    tag: str = DEFAULT_TAG
    inner_index: int = 0
    tiegcm_path: Path | None = None
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    output_name: str = DEFAULT_OUTPUT_NAME
    conductance_source: str = "computed"
    compression: str = "lzf"
    max_steps: int | None = None


SETTINGS = PreparationSettings()


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


def _centered_dipole_alignment_attrs(event_time: dt.datetime, mag_m0_nT: float) -> dict[str, Any]:
    """Return coordinate alignment for prepared GAMERA forcing."""
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))
    alignment = main_field.alignment_metadata(event_time)
    internal = _gamera_internal_dipole_axes(mag_m0_nT)
    return {
        "gamera_coordinate_system": "SM",
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
            f"Found multiple TIEGCM files; set SETTINGS.tiegcm_path explicitly:\n  {formatted}"
        )
    return matches[0]


def _gamera_length_scale_m(gsph: Any) -> float:
    """Return the GAMERA-to-meter length scale."""
    with h5py.File(gsph.f0, "r") as file:
        units_id = file.attrs.get("UnitsID", b"")
        if isinstance(units_id, bytes):
            units_id = units_id.decode("ascii", errors="ignore")
        if str(units_id).upper().startswith("EARTH") and "tScl" in file.attrs:
            return float(file.attrs["tScl"]) * 1.0e5

    warnings.warn(
        f"GAMERA file has no EARTH tScl metadata; using {FALLBACK_EARTH_RADIUS_M:g} m.",
        RuntimeWarning,
        stacklevel=2,
    )
    return FALLBACK_EARTH_RADIUS_M


def _gamera_dipole_strength_nT(gsph: Any) -> float | None:
    """Return GAMERA's signed dipole strength in nT if available."""
    with h5py.File(gsph.f0, "r") as file:
        if "MagM0" in file.attrs:
            return float(file.attrs["MagM0"])
    return None


def _gamera_background_field(
    gsph: Any, inner_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the inner-boundary GAMERA background field."""
    names = ("Bx0", "By0", "Bz0")
    with h5py.File(gsph.f0, "r") as root_file:
        missing = [name for name in names if name not in root_file]
    if missing:
        raise RuntimeError(
            "This preparation path expects Kaiju background-field output. "
            f"Missing root datasets: {missing}. For MAGE/GAMERA Earth runs, "
            "Kaiju writes total Bx/By/Bz and root Bx0/By0/Bz0."
        )
    return tuple(np.asarray(gsph.GetVar(name)[inner_index]) for name in names)


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
    array[np.abs(array) > FILL_THRESHOLD] = np.nan
    return array


def _conductance_normalized_wind(
    numerator_sigma: np.ndarray,
    denominator_conductance: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return winds preserving conductivity-wind current moments."""
    east_num = np.nansum(numerator_sigma * wind_east * dz, axis=0)
    north_num = np.nansum(numerator_sigma * wind_north * dz, axis=0)
    east = np.divide(
        east_num,
        denominator_conductance,
        out=np.zeros_like(east_num),
        where=denominator_conductance > 0.0,
    )
    north = np.divide(
        north_num,
        denominator_conductance,
        out=np.zeros_like(north_num),
        where=denominator_conductance > 0.0,
    )
    return east.astype(np.float32), north.astype(np.float32)


def _integrate_tiegcm_step(
    dataset: Any, step: int, conductance_source: str
) -> dict[str, np.ndarray]:
    """Height-integrate conductivities and weighted winds."""
    sigma_p = _read_tiegcm_step(dataset, "SIGMA_PED", step)
    sigma_h = _read_tiegcm_step(dataset, "SIGMA_HAL", step)
    height_m = _read_tiegcm_step(dataset, "ZG", step) / 100.0
    wind_east = _read_tiegcm_step(dataset, "UN", step) * 1e-2
    wind_north = _read_tiegcm_step(dataset, "VN", step) * 1e-2

    field_shapes = {
        sigma_p.shape,
        sigma_h.shape,
        height_m.shape,
        wind_east.shape,
        wind_north.shape,
    }
    if len(field_shapes) != 1 or sigma_p.ndim < 1 or sigma_p.shape[0] < 2:
        raise RuntimeError(
            "TIEGCM conductivity, height, and wind histories must have matching "
            "shapes with at least two vertical levels."
        )

    dz = np.diff(height_m, axis=0)
    if np.any(np.isfinite(dz) & (dz <= 0.0)):
        raise RuntimeError("TIEGCM geometric height must increase with vertical level.")
    sigma_p_layer = sigma_p[:-1]
    sigma_h_layer = sigma_h[:-1]
    wind_east = wind_east[:-1]
    wind_north = wind_north[:-1]
    sigma_p_int = np.nansum(sigma_p_layer * dz, axis=0)
    sigma_h_int = np.nansum(sigma_h_layer * dz, axis=0)

    if conductance_source == "native":
        sigma_p_out = _read_tiegcm_step(dataset, "gzigm1", step)
        sigma_h_out = _read_tiegcm_step(dataset, "gzigm2", step)
    else:
        sigma_p_out = sigma_p_int
        sigma_h_out = sigma_h_int

    horizontal_shape = sigma_p.shape[1:]
    if sigma_p_out.shape != horizontal_shape or sigma_h_out.shape != horizontal_shape:
        raise RuntimeError(
            "TIEGCM conductance grids must match the horizontal conductivity grid; "
            f"expected {horizontal_shape}, got {sigma_p_out.shape} and {sigma_h_out.shape}."
        )

    u_p_east, u_p_north = _conductance_normalized_wind(
        sigma_p_layer, sigma_p_out, wind_east, wind_north, dz
    )
    u_h_east, u_h_north = _conductance_normalized_wind(
        sigma_h_layer, sigma_h_out, wind_east, wind_north, dz
    )

    return {
        "SP": sigma_p_out.astype(np.float32),
        "SH": sigma_h_out.astype(np.float32),
        "We": u_p_east,
        "Wn": u_p_north,
        "WeH": u_h_east,
        "WnH": u_h_north,
    }


def _tiegcm_times(
    dataset: Any, reference_times: list[dt.datetime]
) -> tuple[list[dt.datetime], float]:
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

    return times, 1.0 if raw_mtime.shape[1] == 4 else 60.0


def _validate_source_times(
    gamera_times: list[dt.datetime], tiegcm_times: list[dt.datetime], *, tolerance_seconds: float
) -> None:
    """Require one corresponding TIEGCM history per GAMERA step."""
    if not gamera_times:
        raise RuntimeError("No GAMERA forcing steps are available.")
    if len(tiegcm_times) < len(gamera_times):
        raise RuntimeError(
            f"TIEGCM provides {len(tiegcm_times)} histories but GAMERA requires "
            f"{len(gamera_times)}."
        )
    offsets = np.array(
        [
            abs((tiegcm_time - gamera_time).total_seconds())
            for gamera_time, tiegcm_time in zip(gamera_times, tiegcm_times, strict=False)
        ]
    )
    mismatch = np.flatnonzero(offsets > tolerance_seconds)
    if mismatch.size:
        index = int(mismatch[0])
        raise RuntimeError(
            "GAMERA and TIEGCM histories are not time-aligned at source step "
            f"{index}: GAMERA={gamera_times[index].isoformat()}, "
            f"TIEGCM={tiegcm_times[index].isoformat()}, "
            f"offset={offsets[index]:g} s."
        )


def _validate_tiegcm_variables(dataset: Any, n_steps: int, conductance_source: str) -> None:
    """Require every TIEGCM input variable and selected time range."""
    required = ["lon", "lat", "SIGMA_PED", "SIGMA_HAL", "ZG", "UN", "VN"]
    if conductance_source == "native":
        required.extend(("gzigm1", "gzigm2"))
    missing = [name for name in required if name not in dataset.variables]
    if missing:
        raise RuntimeError(f"TIEGCM file is missing required variables {missing}.")
    too_short = [
        name
        for name in required
        if name not in {"lon", "lat"} and dataset.variables[name].shape[0] < n_steps
    ]
    if too_short:
        raise RuntimeError(
            f"TIEGCM variables {too_short} contain fewer than the required {n_steps} histories."
        )


def _gamera_inner_boundary_geometry(
    gsph: Any, inner_index: int, length_scale_m: float
) -> tuple[np.ndarray, ...]:
    """Return centered inner-boundary grid and helper arrays."""
    x = gsph.X[inner_index]
    y = gsph.Y[inner_index]
    z = gsph.Z[inner_index]

    x = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
    z = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])

    r_re = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r_re)
    phi = np.arctan2(y, x)

    glat = 90.0 - np.degrees(theta)
    glon = np.degrees(phi)
    r_m = r_re * length_scale_m

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return glat, glon, r_m, sin_theta, cos_theta, sin_phi, cos_phi


def _radial_component(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
) -> np.ndarray:
    """Return the radial component of a Cartesian vector field."""
    return bx * sin_theta * cos_phi + by * sin_theta * sin_phi + bz * cos_theta


def _interpolate_to_tiegcm_grid(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """Interpolate a periodic REMIX field onto the TIEGCM grid."""
    source_lat = np.asarray(source_lat, dtype=float).reshape(-1)
    source_lon = wrap_longitude_180(source_lon).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if source_lat.size != source_lon.size or source_lat.size != values.size:
        raise ValueError("REMIX coordinates and values must have matching sizes.")
    valid = np.isfinite(source_lat) & np.isfinite(source_lon) & np.isfinite(values)
    if np.count_nonzero(valid) < 3:
        raise ValueError("REMIX interpolation requires at least three finite samples.")
    source_lat = source_lat[valid]
    source_lon = source_lon[valid]
    values = values[valid]

    # Longitude is periodic, while scipy.griddata operates on a plane.
    # Shifted copies keep the date-line seam from becoming a boundary.
    periodic_lon = np.concatenate((source_lon - 360.0, source_lon, source_lon + 360.0))
    periodic_lat = np.tile(source_lat, 3)
    periodic_values = np.tile(values, 3)
    return griddata(
        (periodic_lon, periodic_lat),
        periodic_values,
        (wrap_longitude_180(target_lon), target_lat),
        method="linear",
    )


def _combine_remix_hemispheres(south: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Fill NaNs in southern interpolation with northern values."""
    output = np.array(south, copy=True)
    mask = np.isnan(output)
    output[mask] = north[mask]
    return output


def _remix_fac_for_hemisphere(
    ion: Any,
    hemisphere: str,
    coordinate_field: MainField,
    mlat: np.ndarray,
    sm_lon: np.ndarray,
    tiegcm_lon: np.ndarray,
    tiegcm_lat: np.ndarray,
    event_time: dt.datetime,
) -> np.ndarray:
    """Return one REMIX FAC hemisphere on the TIEGCM grid."""
    ion.init_vars(hemisphere)
    sign = -1.0 if hemisphere == "SOUTH" else 1.0
    lat_mag = sign * mlat

    fac = ion.variables["current"]["data"]

    scalar_lat, scalar_lon = coordinate_field.model_to_geo_coordinates(
        lat_mag, sm_lon, event_time=event_time
    )
    return _interpolate_to_tiegcm_grid(scalar_lat, scalar_lon, fac, tiegcm_lon, tiegcm_lat)


def _remix_fac_for_step(
    remix_file: Path,
    step: int,
    event_time: dt.datetime,
    tiegcm_lon: np.ndarray,
    tiegcm_lat: np.ndarray,
) -> np.ndarray:
    """Read and combine north/south REMIX FAC for one step."""
    try:
        import kaipy.remix.remix as remix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mage_prepare.py needs kaipy.remix to read REMIX files. "
            f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
            "where kaipy and its dependencies are installed."
        ) from exc

    coordinate_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))
    ion = remix.remix(str(remix_file), step)
    _, _, theta, phi = ion.cartesianCellCenters()
    mlat = 90.0 - theta / np.pi * 180.0
    sm_lon = wrap_longitude_180(phi / np.pi * 180.0)

    north = _remix_fac_for_hemisphere(
        ion, "NORTH", coordinate_field, mlat, sm_lon, tiegcm_lon, tiegcm_lat, event_time
    )
    south = _remix_fac_for_hemisphere(
        ion, "SOUTH", coordinate_field, mlat, sm_lon, tiegcm_lon, tiegcm_lat, event_time
    )
    return _combine_remix_hemispheres(south, north)


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
    for name in ("FAC", "SH", "SP", "We", "Wn", "WeH", "WnH"):
        output.create_dataset(name, shape=(n_steps, *ion_shape), dtype="f4", **kwargs)
    output.create_dataset("Bu", shape=(n_steps, *inner_shape), dtype="f4", **kwargs)
    output["FAC"].attrs["units"] = "uA m-2"
    for name in ("SH", "SP"):
        output[name].attrs["units"] = "S"
    for name in ("We", "Wn", "WeH", "WnH"):
        output[name].attrs["units"] = "m s-1"
    output["Bu"].attrs["units"] = "nT"
    output["Bu"].attrs["description"] = "radial perturbation from total B minus background B0"


def _write_static_datasets(
    output: h5py.File,
    time_values: np.ndarray,
    event_time: dt.datetime,
    tiegcm_lat: np.ndarray,
    tiegcm_lon: np.ndarray,
    inner_lat: np.ndarray,
    inner_lon: np.ndarray,
    inner_r: np.ndarray,
    settings: PreparationSettings,
    gamera_run_dir: Path,
    length_scale_m: float,
    mag_m0_nT: float,
    tiegcm_path: Path,
) -> None:
    """Write static datasets and metadata."""
    output.attrs["kind"] = MAGE_FORCING_KIND
    output.attrs["version"] = MAGE_FORCING_VERSION
    output.attrs["complete"] = False
    string_dtype = h5py.string_dtype(encoding="utf-8")
    output.create_dataset(
        "time", data=np.asarray(time_values, dtype=string_dtype), dtype=string_dtype
    )
    output.create_dataset("glat", data=tiegcm_lat)
    output.create_dataset("glon", data=tiegcm_lon)
    output.create_dataset("Blat", data=inner_lat)
    output.create_dataset("Blon", data=inner_lon)
    output.create_dataset("r", data=inner_r)
    for name in ("glat", "glon", "Blat", "Blon"):
        output[name].attrs["units"] = "degree"
    output["r"].attrs["units"] = "m"
    output.attrs["gamera_run_dir"] = str(gamera_run_dir)
    output.attrs["tiegcm_nc"] = str(tiegcm_path)
    output.attrs["conductance_source"] = settings.conductance_source
    output.attrs["wind_weighting"] = (
        "Pedersen datasets We/Wn; Hall datasets WeH/WnH; projection uses "
        "sheet-radius B and b for the electrodynamic source"
    )
    output.attrs["remix_tag"] = settings.tag
    output.attrs["fac_convention"] = "upward"
    output.attrs["fac_source"] = "kaipy.remix.init_vars"
    output.attrs["gamera_inner_index"] = int(settings.inner_index)
    output.attrs["gamera_length_scale_m"] = float(length_scale_m)
    output.attrs["gamera_B_output"] = "Kaiju Bx/By/Bz total field, with B0 active"
    for name, value in _centered_dipole_alignment_attrs(event_time, mag_m0_nT).items():
        output.attrs[name] = value
    output.attrs["gamera_mag_m0_nT"] = float(mag_m0_nT)
    output.attrs["gamera_dipole_B0_T"] = abs(float(mag_m0_nT)) * 1e-9


def _validate_settings(settings: PreparationSettings) -> None:
    """Validate in-script preparation settings."""
    if settings.conductance_source not in ("computed", "native"):
        raise ValueError(
            "conductance_source must be 'computed' or 'native'; "
            f"got {settings.conductance_source!r}."
        )
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


def prepare_forcing(settings: PreparationSettings = SETTINGS) -> Path:
    """Prepare the HDF5 forcing file."""
    _validate_settings(settings)
    from netCDF4 import Dataset

    try:
        import kaipy.gamera.magsphere as msph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mage_prepare.py needs kaipy to read GAMERA/REMIX files. "
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
    output_dir = Path(settings.output_directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.output_name

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
    if mag_m0_nT is None:
        raise RuntimeError(
            "GAMERA root metadata is missing the signed dipole strength MagM0. "
            "It is required to align and scale the prepared forcing."
        )
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
    n_available = len(gsph.UT) - 1
    if settings.max_steps is not None:
        n_available = min(n_available, int(settings.max_steps))
    if n_available <= 0:
        raise RuntimeError("GAMERA contains no forcing steps after its initial state.")

    with Dataset(tiegcm_path, mode="r") as tiegcm:
        n_steps = n_available
        _validate_tiegcm_variables(tiegcm, n_steps, settings.conductance_source)
        gamera_times = [value.replace(tzinfo=None) for value in gsph.UT[1 : n_steps + 1]]
        tiegcm_times, time_tolerance_seconds = _tiegcm_times(tiegcm, gamera_times)
        _validate_source_times(
            gamera_times, tiegcm_times, tolerance_seconds=time_tolerance_seconds
        )
        lon = np.asarray(tiegcm.variables["lon"][:], dtype=float)
        lon[lon < 0.0] += 360.0
        lat = np.asarray(tiegcm.variables["lat"][:], dtype=float)
        tiegcm_lon, tiegcm_lat = np.meshgrid(lon, lat)

        # Keep the source timestamps exact. MAGE histories can carry a
        # fractional-second offset even when the nominal cadence is
        # integral.
        time_values = np.array([value.isoformat() for value in gamera_times], dtype=object)

        inner_lat, inner_lon, inner_r, sin_theta, cos_theta, sin_phi, cos_phi = (
            _gamera_inner_boundary_geometry(gsph, settings.inner_index, length_scale_m)
        )
        # Kaiju gioH5 writes Bx/By/Bz as total field when
        # Model%doBackground is true, and root Bx0/By0/Bz0 as Gr%B0.
        # PynaMIT needs the perturbation.
        with _atomic_prepared_output(output_path) as output:
            _write_static_datasets(
                output,
                time_values,
                gamera_times[0],
                tiegcm_lat,
                tiegcm_lon,
                inner_lat,
                inner_lon,
                inner_r,
                settings,
                gamera_run_dir,
                length_scale_m,
                mag_m0_nT,
                tiegcm_path,
            )
            _create_output_datasets(
                output, n_steps, tiegcm_lat.shape, inner_lat.shape, settings.compression
            )

            for out_step, event_time in enumerate(gamera_times):
                gamera_step = gsph.s0 + out_step + 1
                print(
                    f"Preparing step {out_step + 1} of {n_steps}: {event_time.isoformat()}",
                    flush=True,
                )

                integrated = _integrate_tiegcm_step(tiegcm, out_step, settings.conductance_source)
                for key, values in integrated.items():
                    output[key][out_step] = values

                output["FAC"][out_step] = _remix_fac_for_step(
                    remix_file, gamera_step, event_time, tiegcm_lon, tiegcm_lat
                ).astype(np.float32)

                bx = gsph.GetVar("Bx", gamera_step)[settings.inner_index] - bx0
                by = gsph.GetVar("By", gamera_step)[settings.inner_index] - by0
                bz = gsph.GetVar("Bz", gamera_step)[settings.inner_index] - bz0
                output["Bu"][out_step] = _radial_component(
                    bx, by, bz, sin_theta, cos_theta, sin_phi, cos_phi
                ).astype(np.float32)

    return output_path


def main(settings: PreparationSettings = SETTINGS) -> None:
    """Prepare forcing from in-script settings."""
    output_path = prepare_forcing(settings)
    print(f"Prepared forcing written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
