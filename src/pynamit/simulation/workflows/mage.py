"""Reusable helpers for MAGE/GAMERA PynaMIT workflows."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.geomagnetism import MainField, decimal_year
from pynamit.geomagnetism.kaiju_geopack import axis_lat_lon

DEFAULT_NMAX = 50
DEFAULT_MMAX = 50
DEFAULT_NCS = 50


def gamera_internal_dipole_axes(mag_m0_nT: float | None) -> dict[str, np.ndarray]:
    """Return GAMERA dipole-moment and magnetic-pole axes."""
    sign = -1.0
    if mag_m0_nT is not None and np.isfinite(mag_m0_nT) and mag_m0_nT != 0.0:
        sign = float(np.sign(mag_m0_nT))
    moment_axis = np.array([0.0, 0.0, sign])
    north_axis = -moment_axis
    moment_axis[np.isclose(moment_axis, 0.0)] = 0.0
    north_axis[np.isclose(north_axis, 0.0)] = 0.0
    return {"moment_axis": moment_axis, "north_axis": north_axis, "south_axis": -north_axis}


def centered_dipole_alignment_attrs(
    event_time: dt.datetime, mag_m0_nT: float | None
) -> dict[str, Any]:
    """Return model-alignment metadata for GAMERA forcing."""
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))
    alignment = main_field.alignment_metadata(event_time)
    internal = gamera_internal_dipole_axes(mag_m0_nT)
    return {
        "gamera_coordinate_system": "SM",
        "gamera_internal_dipole_axis": internal["north_axis"],
        "gamera_internal_magnetic_north_axis": internal["north_axis"],
        "gamera_internal_magnetic_south_axis": internal["south_axis"],
        "gamera_internal_dipole_moment_axis": internal["moment_axis"],
        "gamera_internal_north_pole_lat_lon": axis_lat_lon(internal["north_axis"]),
        "gamera_internal_south_pole_lat_lon": axis_lat_lon(internal["south_axis"]),
        "remix_local_noon_longitude_deg": 0.0,
        "pynamit_run_coordinate_system": "SM",
        "dipole_noon_mlon_deg_at_start": 0.0,
        "dipole_mag_noon_mlon_deg_at_start": alignment["dipole_mag_noon_mlon_deg"],
        **alignment,
    }


def _resolution_tag(nmax: int, mmax: int, ncs: int) -> str:
    """Return the directory tag for one projected resolution."""
    return f"N{int(nmax)}_M{int(mmax)}_Ncs{int(ncs)}"


def projection_directory_for_resolution(nmax: int, mmax: int, ncs: int, root: Path) -> Path:
    """Return the projected-input directory for a resolution."""
    return Path(root) / "projections" / _resolution_tag(nmax, mmax, ncs)


def result_directory_for_resolution(nmax: int, mmax: int, ncs: int, root: Path) -> Path:
    """Return the run-output directory for a resolution."""
    return Path(root) / "results" / _resolution_tag(nmax, mmax, ncs)


def file_fingerprint(path: str | Path | None) -> dict[str, int | str] | None:
    """Return cheap provenance metadata for a source file."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def dipole_radial_sampling(r_min: float, r_max: float, n_steps: int) -> np.ndarray:
    """Return radial samples along a dipole field line."""
    ratio = r_min / r_max
    max_angle = np.rad2deg(np.arccos(np.sqrt(ratio)))
    angles = np.linspace(0.0, max_angle, int(n_steps))
    return r_min / np.cos(np.deg2rad(angles)) ** 2


def _parse_h5_time(value: Any) -> dt.datetime:
    """Parse an HDF5 ISO timestamp stored as bytes or str."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return dt.datetime.fromisoformat(str(value))


def h5_time_vector_seconds(raw_times: Any) -> tuple[list[dt.datetime], np.ndarray]:
    """Return parsed HDF5 datetimes and seconds from the first entry."""
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


def summarize_input_cadence(relative_seconds: np.ndarray) -> dict[str, float | None]:
    """Return compact cadence metadata for projected input times."""
    relative_seconds = np.asarray(relative_seconds, dtype=float)
    if relative_seconds.size < 2:
        return {"input_dt_median_s": None, "input_dt_min_s": None, "input_dt_max_s": None}
    diffs = np.diff(relative_seconds)
    return {
        "input_dt_median_s": float(np.median(diffs)),
        "input_dt_min_s": float(np.min(diffs)),
        "input_dt_max_s": float(np.max(diffs)),
    }


def area_sqrt_weights(lat: np.ndarray) -> np.ndarray:
    """Latitude-only surface-area square-root weights."""
    theta = np.deg2rad(90.0 - np.asarray(lat, dtype=float).reshape(-1))
    return np.sqrt(np.clip(np.sin(theta), 0.0, None))


def tangential_sqrt_weights(lat: np.ndarray) -> np.ndarray:
    """Two-component area square-root weights for tangential fits."""
    return np.tile(area_sqrt_weights(lat), (2, 1))


def boundary_radius_from_h5(h5_file: Any, explicit_rm: float | None) -> float:
    """Return the magnetospheric boundary radius used for Br fitting."""
    if explicit_rm is not None:
        return float(explicit_rm)
    if "r" not in h5_file:
        raise RuntimeError(
            "Prepared MAGE forcing is missing the inner-boundary radius dataset 'r'. "
            "Regenerate it with scripts/simulation/mage_prepare_forcing.py or set "
            "SETTINGS.RM explicitly in mage_project_inputs.py."
        )
    radius = np.asarray(h5_file["r"][:], dtype=float)
    finite = radius[np.isfinite(radius)]
    if finite.size == 0:
        raise RuntimeError(
            "Prepared MAGE forcing dataset 'r' contains no finite values. "
            "Regenerate the forcing or set SETTINGS.RM explicitly."
        )
    rm = float(np.mean(finite))
    rel_spread = float((np.max(finite) - np.min(finite)) / rm)
    if rel_spread > 1e-3:
        print(
            f"Warning: Br grid radius varies by {rel_spread:.3%}; using mean RM={rm:.6g} m.",
            flush=True,
        )
    return rm


def dipole_B0_from_h5(h5_file: Any, explicit_B0: float | None) -> float:
    """Return the centered-dipole equatorial field in tesla."""
    if explicit_B0 is not None:
        return float(explicit_B0)
    if "gamera_dipole_B0_T" in h5_file.attrs:
        return float(h5_file.attrs["gamera_dipole_B0_T"])
    if "gamera_mag_m0_nT" in h5_file.attrs:
        return abs(float(h5_file.attrs["gamera_mag_m0_nT"])) * 1e-9
    raise RuntimeError(
        "Prepared MAGE forcing is missing dipole strength metadata "
        "'gamera_dipole_B0_T'/'gamera_mag_m0_nT'. Regenerate it with "
        "scripts/simulation/mage_prepare_forcing.py or set SETTINGS.dipole_B0 "
        "explicitly in mage_project_inputs.py."
    )


def gamera_internal_dipole_details(h5_file: Any) -> dict[str, np.ndarray | float | None]:
    """Return signed GAMERA dipole metadata from prepared HDF5."""
    required = (
        "gamera_mag_m0_nT",
        "gamera_internal_dipole_moment_axis",
        "gamera_internal_magnetic_north_axis",
    )
    missing = [name for name in required if name not in h5_file.attrs]
    if missing:
        raise RuntimeError(
            "Prepared MAGE forcing is missing GAMERA dipole metadata "
            f"{missing}. Regenerate it with scripts/simulation/mage_prepare_forcing.py."
        )
    mag_m0_nT = float(h5_file.attrs["gamera_mag_m0_nT"])

    def axis_attr(name: str) -> np.ndarray:
        axis = np.asarray(h5_file.attrs[name], dtype=float)
        norm = np.linalg.norm(axis)
        if axis.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                f"Prepared MAGE forcing metadata {name!r} must be a finite 3-vector."
            )
        axis = axis / norm
        axis[np.isclose(axis, 0.0)] = 0.0
        return axis

    return {
        "mag_m0_nT": mag_m0_nT,
        "moment_axis": axis_attr("gamera_internal_dipole_moment_axis"),
        "north_axis": axis_attr("gamera_internal_magnetic_north_axis"),
    }


def load_weighted_winds(
    h5_file: Any, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Pedersen- and Hall-weighted winds for one prepared step."""
    required = ("We", "Wn", "WeH", "WnH")
    missing = [name for name in required if name not in h5_file]
    if missing:
        raise RuntimeError(
            "Prepared MAGE forcing is missing required weighted-wind dataset(s) "
            f"{missing}. Regenerate the forcing with scripts/simulation/mage_prepare_forcing.py; "
            "the projection step does not reconstruct missing Hall/Pedersen wind products."
        )
    u_p_east = np.asarray(h5_file["We"][step], dtype=float)
    u_p_north = np.asarray(h5_file["Wn"][step], dtype=float)
    return (
        u_p_east,
        u_p_north,
        np.asarray(h5_file["WeH"][step], dtype=float),
        np.asarray(h5_file["WnH"][step], dtype=float),
    )


def cross_spherical(
    a_r: np.ndarray,
    a_theta: np.ndarray,
    a_phi: np.ndarray,
    b_r: np.ndarray,
    b_theta: np.ndarray,
    b_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross product in the local spherical basis."""
    return (
        a_theta * b_phi - a_phi * b_theta,
        a_phi * b_r - a_r * b_phi,
        a_r * b_theta - a_theta * b_r,
    )


def weighted_wind_current_source(
    *,
    sigma_p: np.ndarray,
    sigma_h: np.ndarray,
    u_p_theta: np.ndarray,
    u_p_phi: np.ndarray,
    u_h_theta: np.ndarray,
    u_h_phi: np.ndarray,
    field: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 3D Pedersen/Hall weighted wind-current source."""
    sigma_p = np.asarray(sigma_p, dtype=float).reshape(-1)
    sigma_h = np.asarray(sigma_h, dtype=float).reshape(-1)
    u_p_theta = np.asarray(u_p_theta, dtype=float).reshape(-1)
    u_p_phi = np.asarray(u_p_phi, dtype=float).reshape(-1)
    u_h_theta = np.asarray(u_h_theta, dtype=float).reshape(-1)
    u_h_phi = np.asarray(u_h_phi, dtype=float).reshape(-1)

    b_r = np.asarray(field.unit_br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.unit_btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.unit_bphi, dtype=float).reshape(-1)
    B_r = np.asarray(field.Br, dtype=float).reshape(-1)
    B_theta = np.asarray(field.Btheta, dtype=float).reshape(-1)
    B_phi = np.asarray(field.Bphi, dtype=float).reshape(-1)

    zero = np.zeros_like(u_p_theta)
    u_p_cross_B = cross_spherical(zero, u_p_theta, u_p_phi, B_r, B_theta, B_phi)
    u_h_cross_B = cross_spherical(zero, u_h_theta, u_h_phi, B_r, B_theta, B_phi)
    q_h = cross_spherical(b_r, b_theta, b_phi, *u_h_cross_B)
    return (
        sigma_p * u_p_cross_B[0] + sigma_h * q_h[0],
        sigma_p * u_p_cross_B[1] + sigma_h * q_h[1],
        sigma_p * u_p_cross_B[2] + sigma_h * q_h[2],
    )


def direct_E_source_for_pynamit(
    *,
    sigma_p: np.ndarray,
    sigma_h: np.ndarray,
    u_p_theta: np.ndarray,
    u_p_phi: np.ndarray,
    u_h_theta: np.ndarray,
    u_h_phi: np.ndarray,
    field: Any,
    eta_p: np.ndarray,
    eta_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the direct weighted-wind electric-field source in V/m."""
    q_r, q_theta, q_phi = weighted_wind_current_source(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
    )
    eta_p = np.asarray(eta_p, dtype=float).reshape(-1)
    eta_h = np.asarray(eta_h, dtype=float).reshape(-1)
    b_r = np.asarray(field.unit_br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.unit_btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.unit_bphi, dtype=float).reshape(-1)

    q_dot_b = q_r * b_r + q_theta * b_theta + q_phi * b_phi
    q_perp_theta = q_theta - q_dot_b * b_theta
    q_perp_phi = q_phi - q_dot_b * b_phi
    q_cross_b = cross_spherical(q_r, q_theta, q_phi, b_r, b_theta, b_phi)
    return (
        -(eta_p * q_perp_theta + eta_h * q_cross_b[1]),
        -(eta_p * q_perp_phi + eta_h * q_cross_b[2]),
    )


__all__ = [
    "DEFAULT_MMAX",
    "DEFAULT_NCS",
    "DEFAULT_NMAX",
    "area_sqrt_weights",
    "boundary_radius_from_h5",
    "centered_dipole_alignment_attrs",
    "cross_spherical",
    "dipole_B0_from_h5",
    "dipole_radial_sampling",
    "direct_E_source_for_pynamit",
    "file_fingerprint",
    "gamera_internal_dipole_axes",
    "gamera_internal_dipole_details",
    "h5_time_vector_seconds",
    "load_weighted_winds",
    "projection_directory_for_resolution",
    "result_directory_for_resolution",
    "summarize_input_cadence",
    "tangential_sqrt_weights",
    "weighted_wind_current_source",
]
