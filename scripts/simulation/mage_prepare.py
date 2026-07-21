"""Prepare reusable MAGE/GAMERA/TIEGCM forcing.

The expensive height integration and source-coordinate transformations
are done here once. The output HDF5 contains the fields used by the
projection step on fixed, Earth-attached geographic grids:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S.
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

from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm
from pynamit.math.constants import RE
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


def _naive_utc_datetime(value: dt.datetime) -> dt.datetime:
    """Normalize a source time to the naive-UTC convention."""
    if value.tzinfo is None:
        return value
    return value.astimezone(dt.timezone.utc).replace(tzinfo=None)


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
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))
    alignment = main_field.alignment_metadata(event_time)
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
    """Return Kaiju cell centers corresponding to ``B[inner_index]``.

    GAMERA stores ``X/Y/Z`` at cell vertices and magnetic fields at cell
    centers. The selected magnetic shell therefore lies between vertex
    shells ``inner_index`` and ``inner_index + 1``. Kaiju defines the
    location as the volume barycenter of the trilinear cell.
    """
    vertices = _gamera_inner_boundary_vertices(gsph, inner_index)
    x, y, z = np.moveaxis(_trilinear_hexahedron_volume_centers(vertices), -1, 0)

    r_re = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r_re)
    phi = np.arctan2(y, x)

    latitude = 90.0 - np.degrees(theta)
    longitude = np.degrees(phi)
    r_m = r_re * length_scale_m

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return latitude, longitude, r_m, sin_theta, cos_theta, sin_phi, cos_phi


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
                jacobian = np.einsum(
                    "...vc,vq->...cq", vertices, derivatives, optimize=True
                )
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


def _gamera_inner_boundary_solid_angle(gsph: Any, inner_index: int) -> np.ndarray:
    """Return each GAMERA boundary cell's solid angle in steradians."""
    vertices = _gamera_inner_boundary_vertices(gsph, inner_index)
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
        azimuth += 2.0 * np.pi * np.round(
            (azimuth[0, 0] - azimuth[:, [0]]) / (2.0 * np.pi)
        )
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
            (
                np.zeros_like(polar_azimuth),
                colatitude,
                np.full_like(polar_azimuth, np.pi),
            )
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
            (lower_azimuth, lower_right_azimuth, upper_azimuth, upper_right_azimuth),
            axis=-1,
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
        self,
        values: np.ndarray,
        *,
        target_sm_lat: np.ndarray,
        target_sm_lon: np.ndarray,
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

        colatitude_index = np.searchsorted(
            self._colatitude_axis, colatitude, side="right"
        ) - 1
        colatitude_index = np.clip(colatitude_index, 0, self._colatitude_axis.size - 2)
        azimuth_index = np.searchsorted(
            self._periodic_azimuth_axis, azimuth, side="right"
        ) - 1
        azimuth_index = np.clip(azimuth_index, 0, self._azimuth_axis.size - 1)
        next_azimuth_index = (azimuth_index + 1) % self._azimuth_axis.size

        target_basis = np.column_stack(
            (np.ones_like(azimuth), azimuth, colatitude, azimuth * colatitude)
        )
        weights = np.einsum(
            "nij,nj->ni",
            self._cell_inverse[colatitude_index, azimuth_index],
            target_basis,
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
        self._source_lat = np.array(source_lat, copy=True)
        self._source_lon = wrap_longitude_180(source_lon).copy()
        self._latitude_order = np.argsort(latitude)
        self._longitude_order = np.argsort(longitude)
        self._latitude = latitude[self._latitude_order]
        self._longitude = longitude[self._longitude_order]
        if np.any(np.diff(self._latitude) <= 0.0) or np.any(np.diff(self._longitude) <= 0.0):
            raise ValueError("ReMIX latitude and longitude coordinates must be unique.")

    def matches(self, source_lat: np.ndarray, source_lon: np.ndarray) -> bool:
        """Return whether coordinates match this saved ReMIX grid."""
        source_lat, source_lon = np.broadcast_arrays(
            np.asarray(source_lat, dtype=float), np.asarray(source_lon, dtype=float)
        )
        return (
            source_lat.shape == self._source_shape
            and np.allclose(source_lat, self._source_lat, rtol=0.0, atol=1e-12)
            and np.allclose(
                wrap_longitude_180(source_lon - self._source_lon), 0.0, rtol=0.0, atol=1e-12
            )
        )

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

        periodic_longitude = np.concatenate(
            (self._longitude, [self._longitude[0] + 360.0])
        )
        longitude_index = np.searchsorted(
            periodic_longitude, query_longitude, side="right"
        ) - 1
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
        result[covered_indices] = (
            (1.0 - latitude_fraction)
            * ((1.0 - longitude_fraction) * lower_left + longitude_fraction * lower_right)
            + latitude_fraction
            * ((1.0 - longitude_fraction) * upper_left + longitude_fraction * upper_right)
        )

        # Kaiju treats the polar quadrilateral as a triangle because all
        # longitude vertices at the pole are one physical point. Its map
        # therefore averages the reconstructed pole and the two adjacent
        # values on the poleward ring, independently of polar distance.
        polar_cap = (
            query_latitude > self._latitude[-1]
            if north
            else query_latitude < self._latitude[0]
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
    """Return timestamped Kaiju SM coordinates of a fixed GEO grid."""
    return kaiju_geopack_sm(event_time).geo2sm(latitude, longitude)


def _tiegcm_step_in_geographic_coordinates(
    integrated: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Express one native GEO TIEGCM history in spherical components."""
    return {
        "SP": integrated["SP"].astype(np.float32),
        "SH": integrated["SH"].astype(np.float32),
        "u_p_theta": (-integrated["Wn"]).astype(np.float32),
        "u_p_phi": integrated["We"].astype(np.float32),
        "u_h_theta": (-integrated["WnH"]).astype(np.float32),
        "u_h_phi": integrated["WeH"].astype(np.float32),
    }


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
    ion: Any, hemisphere: str, unsigned_magnetic_latitude: np.ndarray, grid_longitude: np.ndarray
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
    dataset_name = f"Field-aligned current {hemisphere}"
    if dataset_name not in ion.ion:
        raise RuntimeError(f"ReMIX history is missing {dataset_name!r}.")
    fac = np.asarray(ion.ion[dataset_name], dtype=float)
    if fac.shape != latitude.shape:
        raise RuntimeError(
            f"ReMIX {dataset_name!r} shape {fac.shape} does not match "
            f"the cell-center grid {latitude.shape}."
        )

    if hemisphere == "NORTH":
        return latitude, wrap_longitude_180(longitude), -fac
    return -latitude, wrap_longitude_180(-longitude), fac


class _RemixRadialCurrentReader:
    """Return outward current from REMIX FAC on a fixed GEO grid."""

    def __init__(self, remix_file: Path) -> None:
        try:
            import kaipy.remix.remix as remix
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "mage_prepare.py needs kaipy.remix to read REMIX files. "
                f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
                "where kaipy and its dependencies are installed."
            ) from exc
        self._remix = remix
        self._remix_file = Path(remix_file)
        self._interpolators: dict[str, _RemixGridInterpolator] = {}

    def _hemisphere(
        self,
        ion: Any,
        hemisphere: str,
        mlat: np.ndarray,
        sm_lon: np.ndarray,
        target_sm_lon: np.ndarray,
        target_sm_lat: np.ndarray,
    ) -> np.ndarray:
        """Sample one FAC hemisphere at target SM positions."""
        source_lat, source_lon, upward_fac = _remix_upward_fac_source(
            ion, hemisphere, mlat, sm_lon
        )
        interpolator = self._interpolators.get(hemisphere)
        if interpolator is None:
            interpolator = _RemixGridInterpolator(source_lat, source_lon)
            self._interpolators[hemisphere] = interpolator
        elif not interpolator.matches(source_lat, source_lon):
            raise RuntimeError("REMIX grid coordinates changed between forcing histories.")
        return interpolator.interpolate(upward_fac, target_sm_lon, target_sm_lat)

    def read(
        self,
        step: int,
        target_longitude: np.ndarray,
        target_latitude: np.ndarray,
        event_time: dt.datetime,
    ) -> np.ndarray:
        """Return outward current on the fixed geographic grid."""
        ion = self._remix.remix(str(self._remix_file), step)
        _, _, theta, phi = ion.cartesianCellCenters()
        mlat = 90.0 - theta / np.pi * 180.0
        sm_lon = wrap_longitude_180(phi / np.pi * 180.0)
        target_sm_lat, target_sm_lon = _geographic_grid_in_sm(
            target_latitude, target_longitude, event_time
        )
        north = self._hemisphere(ion, "NORTH", mlat, sm_lon, target_sm_lon, target_sm_lat)
        south = self._hemisphere(ion, "SOUTH", mlat, sm_lon, target_sm_lon, target_sm_lat)
        upward_fac = _combine_remix_hemispheres(south, north)
        return _upward_fac_to_radial_current(upward_fac, target_sm_lat)


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
    for name in ("SH", "SP"):
        output[name].attrs["units"] = "S"
    for name in ("u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        output[name].attrs["units"] = "m s-1"
    output["delta_Br"].attrs["units"] = "nT"
    output["delta_Br"].attrs["description"] = (
        "radial perturbation from total B minus GAMERA split background B0"
    )


def _write_static_datasets(
    output: h5py.File,
    time_values: np.ndarray,
    event_time: dt.datetime,
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
) -> None:
    """Write static datasets and metadata."""
    output.attrs["kind"] = MAGE_FORCING_KIND
    output.attrs["version"] = MAGE_FORCING_VERSION
    output.attrs["complete"] = False
    string_dtype = h5py.string_dtype(encoding="utf-8")
    output.create_dataset(
        "time", data=np.asarray(time_values, dtype=string_dtype), dtype=string_dtype
    )
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
    output.attrs["conductance_source"] = settings.conductance_source
    output.attrs["coordinate_system"] = "GEO"
    output.attrs["longitude_convention"] = "east_positive_degrees"
    output.attrs["tiegcm_source_coordinate_system"] = "geographic"
    output.attrs["wind_weighting"] = (
        "Pedersen datasets u_p_theta/u_p_phi; Hall datasets u_h_theta/u_h_phi; "
        "components are geographic south/east on the native TIEGCM grid"
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
    output.attrs["gamera_inner_index"] = int(settings.inner_index)
    output.attrs["gamera_length_scale_m"] = float(length_scale_m)
    output.attrs["gamera_B_output"] = (
        "Kaiju Bx/By/Bz total field; delta_Br removes the split background B0"
    )
    for name, value in _centered_dipole_alignment_attrs(event_time, mag_m0_nT).items():
        output.attrs[name] = value
    output.attrs["gamera_mag_m0_nT"] = float(mag_m0_nT)
    output.attrs["main_field_B0_T"] = _pynamit_dipole_B0_T(mag_m0_nT, length_scale_m)
    output.attrs["main_field_B0_reference_radius_m"] = RE


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
        gamera_times = [_naive_utc_datetime(value) for value in gsph.UT[1 : n_steps + 1]]
        tiegcm_times, time_tolerance_seconds = _tiegcm_times(tiegcm, gamera_times)
        _validate_source_times(
            gamera_times, tiegcm_times, tolerance_seconds=time_tolerance_seconds
        )
        source_lon = np.asarray(tiegcm.variables["lon"][:], dtype=float)
        source_lat = np.asarray(tiegcm.variables["lat"][:], dtype=float)
        ionosphere_lon, ionosphere_lat = np.meshgrid(wrap_longitude_180(source_lon), source_lat)

        # Keep the source timestamps exact. MAGE histories can carry a
        # fractional-second offset even when the nominal cadence is
        # integral.
        time_values = np.array([value.isoformat() for value in gamera_times], dtype=object)

        radial_current_reader = _RemixRadialCurrentReader(remix_file)

        inner_sm_lat, inner_sm_lon, inner_r, sin_theta, cos_theta, sin_phi, cos_phi = (
            _gamera_inner_boundary_geometry(gsph, settings.inner_index, length_scale_m)
        )
        inner_solid_angle = _gamera_inner_boundary_solid_angle(gsph, settings.inner_index)
        boundary_lat, boundary_lon = kaiju_geopack_sm(gamera_times[0]).sm2geo(
            inner_sm_lat, inner_sm_lon
        )
        boundary_interpolator = _GameraBoundaryInterpolator(inner_sm_lat, inner_sm_lon)
        # Kaiju gioH5 writes Bx/By/Bz as total field when
        # Model%doBackground is true, and root Bx0/By0/Bz0 as Gr%B0.
        # PynaMIT needs the perturbation.
        with _atomic_prepared_output(output_path) as output:
            _write_static_datasets(
                output,
                time_values,
                gamera_times[0],
                ionosphere_lat,
                ionosphere_lon,
                boundary_lat,
                boundary_lon,
                inner_r,
                inner_solid_angle,
                settings,
                gamera_run_dir,
                length_scale_m,
                mag_m0_nT,
                tiegcm_path,
            )
            _create_output_datasets(
                output, n_steps, ionosphere_lat.shape, boundary_lat.shape, settings.compression
            )

            for out_step, event_time in enumerate(gamera_times):
                gamera_step = gsph.s0 + out_step + 1
                print(
                    f"Preparing step {out_step + 1} of {n_steps}: {event_time.isoformat()}",
                    flush=True,
                )

                integrated = _integrate_tiegcm_step(tiegcm, out_step, settings.conductance_source)
                model_inputs = _tiegcm_step_in_geographic_coordinates(integrated)
                for key, values in model_inputs.items():
                    output[key][out_step] = values

                output["jr"][out_step] = radial_current_reader.read(
                    gamera_step, ionosphere_lon, ionosphere_lat, event_time
                ).astype(np.float32)

                bx = gsph.GetVar("Bx", gamera_step)[settings.inner_index] - bx0
                by = gsph.GetVar("By", gamera_step)[settings.inner_index] - by0
                bz = gsph.GetVar("Bz", gamera_step)[settings.inner_index] - bz0
                delta_br_sm = _radial_component(bx, by, bz, sin_theta, cos_theta, sin_phi, cos_phi)
                boundary_sm_lat, boundary_sm_lon = _geographic_grid_in_sm(
                    boundary_lat, boundary_lon, event_time
                )
                output["delta_Br"][out_step] = boundary_interpolator.interpolate(
                    delta_br_sm,
                    target_sm_lat=boundary_sm_lat,
                    target_sm_lon=boundary_sm_lon,
                ).astype(np.float32)

    return output_path


def main(settings: PreparationSettings = SETTINGS) -> None:
    """Prepare forcing from in-script settings."""
    output_path = prepare_forcing(settings)
    print(f"Prepared forcing written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
