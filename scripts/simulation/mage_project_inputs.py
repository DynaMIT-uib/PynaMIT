"""Project prepared MAGE/GAMERA/TIEGCM forcing into PynaMIT inputs.

Run ``mage_prepare_forcing.py`` first if the height-integrated HDF5 file
does not exist.  This script does the PynaMIT-specific projection step:
it reads the prepared HDF5 file, projects Br, jr, conductance, and the
direct Pedersen/Hall weighted wind electric-field source, then writes a
reusable input package under
``scripts/simulation/mage_runs/mage_2011_kaiju_direct_e/projections``.

The time evolution is intentionally not done here.  Use
``mage_forcing_final.py`` to run PynaMIT from the projected input
package.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pynamit

from pynamit.coordinates import wrap_longitude_180
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.simulation.mainfield import Mainfield, decimal_year
from pynamit.simulation.prepared_inputs import INPUT_DATASET_KEYS, write_input_manifest

SCRIPT_DIR = Path(__file__).resolve().parent
RE = 6381e3
RI = 6.5e6
LATITUDE_BOUNDARY = 35.0
DEFAULT_NMAX = 50
DEFAULT_MMAX = 50
DEFAULT_NCS = 50

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3.0
JR_LAMBDA = 0.1
E_SOURCE_LAMBDA = 0.1

MAGE_BR_LOCAL_NOON_LONGITUDE = 0.0
MAGE_DIPOLE_B0_T = 29617.369174957275e-9
CENTERED_DIPOLE_MODELS = ("kaiju_dipole", "dipole")

DEFAULT_FORCING_CANDIDATES = (
    SCRIPT_DIR / "mage_prepared" / "mage_prepared_forcing.h5",
    SCRIPT_DIR / "mage_prepared" / "data_H_int_qeff.h5",
    Path("~/Gamera_Dong/prep_Pynamit/mage_prepared_forcing.h5"),
    Path("~/Gamera_Dong/prep_Pynamit/data_H_int_qeff.h5"),
    Path("mage_2011/data_H_int.h5"),
    Path("~/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
    Path("/Users/andreasskeidsvoll/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
)
DEFAULT_TIEGCM_CANDIDATES = (
    Path("~/Gamera_Dong/11OcA_sech_tie_2011-10-24T18-00-10_2011-10-24T19-00-00.nc"),
    Path(
        "/Users/andreasskeidsvoll/Gamera_Dong/"
        "11OcA_sech_tie_2011-10-24T18-00-10_2011-10-24T19-00-00.nc"
    ),
)
DEFAULT_MAGE_RUN_ROOT = SCRIPT_DIR / "mage_runs" / "mage_2011_kaiju_direct_e"
MAGE_INPUT_METADATA_FILENAME = "mage_input_metadata.json"


def resolution_tag(nmax: int, mmax: int, ncs: int) -> str:
    """Return the directory tag for one projected resolution."""
    return f"N{int(nmax)}_M{int(mmax)}_Ncs{int(ncs)}"


def projection_directory_for_resolution(
    nmax: int, mmax: int, ncs: int, root: Path = DEFAULT_MAGE_RUN_ROOT
) -> Path:
    """Return the default projected-input directory for a resolution."""
    return Path(root) / "projections" / resolution_tag(nmax, mmax, ncs)


def result_directory_for_resolution(
    nmax: int, mmax: int, ncs: int, root: Path = DEFAULT_MAGE_RUN_ROOT
) -> Path:
    """Return the default run-output directory for a resolution."""
    return Path(root) / "results" / resolution_tag(nmax, mmax, ncs)


DEFAULT_INPUT_DIRECTORY = projection_directory_for_resolution(
    DEFAULT_NMAX, DEFAULT_MMAX, DEFAULT_NCS
)
DEFAULT_RESULT_DIRECTORY = result_directory_for_resolution(DEFAULT_NMAX, DEFAULT_MMAX, DEFAULT_NCS)


@dataclass(frozen=True)
class MageInputProjectionSettings:
    """Defaults intended to be edited for the MAGE projection step."""

    forcing_h5: Path | None = None
    tiegcm_nc: Path | None = None
    input_directory: Path | None = None
    mainfield_kind: str = "kaiju_dipole"
    dipole_B0: float | None = None
    fac_convention: str = "upward"
    RM: float | None = None
    nmax: int = DEFAULT_NMAX
    mmax: int = DEFAULT_MMAX
    ncs: int = DEFAULT_NCS
    max_steps: int | None = None
    br_lambda: float = BR_LAMBDA
    conductance_lambda: float = CONDUCTANCE_LAMBDA
    jr_lambda: float = JR_LAMBDA
    e_source_lambda: float = E_SOURCE_LAMBDA
    artifact_storage: str = "auto"


SETTINGS = MageInputProjectionSettings()


def dipole_radial_sampling(r_min: float, r_max: float, n_steps: int) -> np.ndarray:
    """Return radial samples along a dipole field line."""
    ratio = r_min / r_max
    max_angle = np.rad2deg(np.arccos(np.sqrt(ratio)))
    angles = np.linspace(0.0, max_angle, n_steps)
    return r_min / np.cos(np.deg2rad(angles)) ** 2


def resolve_existing_path(
    path: str | Path | None, candidates: tuple[Path, ...], label: str
) -> Path:
    """Resolve an explicit path or the first existing candidate."""
    if path:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        return resolved

    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked: {formatted}")


def parse_h5_time(value: Any):
    """Parse an HDF5 ISO timestamp stored as bytes or str."""
    import datetime as dt

    if isinstance(value, bytes):
        value = value.decode("ascii")
    return dt.datetime.fromisoformat(str(value))


def h5_time_vector_seconds(raw_times: Any) -> tuple[list[Any], np.ndarray]:
    """Return parsed HDF5 datetimes and seconds from the first entry."""
    parsed_times = [parse_h5_time(value) for value in raw_times]
    if not parsed_times:
        raise ValueError("Forcing HDF5 time dataset is empty.")
    event_time = parsed_times[0]
    relative_seconds = np.array(
        [(time_value - event_time).total_seconds() for time_value in parsed_times],
        dtype=float,
    )
    if np.any(~np.isfinite(relative_seconds)):
        raise ValueError("Forcing HDF5 time dataset produced non-finite relative seconds.")
    if np.any(np.diff(relative_seconds) <= 0.0):
        raise ValueError("Forcing HDF5 time dataset must be strictly increasing.")
    return parsed_times, relative_seconds


def summarize_input_cadence(relative_seconds: np.ndarray) -> dict[str, float | None]:
    """Return compact cadence metadata for projected input times."""
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
        return 1.5 * RI
    radius = np.asarray(h5_file["r"][:], dtype=float)
    finite = radius[np.isfinite(radius)]
    if finite.size == 0:
        return 1.5 * RI
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
    return MAGE_DIPOLE_B0_T


def gamera_internal_dipole_details(h5_file: Any) -> dict[str, np.ndarray | float | None]:
    """Return signed GAMERA dipole metadata from prepared HDF5."""
    mag_m0_nT = (
        float(h5_file.attrs["gamera_mag_m0_nT"]) if "gamera_mag_m0_nT" in h5_file.attrs else None
    )
    sign = -1.0
    if mag_m0_nT is not None and np.isfinite(mag_m0_nT) and mag_m0_nT != 0.0:
        sign = float(np.sign(mag_m0_nT))
    moment_fallback = np.array([0.0, 0.0, sign])
    north_fallback = -moment_fallback

    def axis_attr(name: str, fallback: np.ndarray) -> np.ndarray:
        if name in h5_file.attrs:
            axis = np.asarray(h5_file.attrs[name], dtype=float)
            if axis.shape == (3,) and np.linalg.norm(axis) > 0.0:
                axis = axis / np.linalg.norm(axis)
                axis[np.isclose(axis, 0.0)] = 0.0
                return axis
        fallback = fallback / np.linalg.norm(fallback)
        fallback[np.isclose(fallback, 0.0)] = 0.0
        return fallback

    return {
        "mag_m0_nT": mag_m0_nT,
        "moment_axis": axis_attr("gamera_internal_dipole_moment_axis", moment_fallback),
        "north_axis": axis_attr("gamera_internal_magnetic_north_axis", north_fallback),
    }


def conductivity_weighted_winds_from_tiegcm_step(dataset: Any, step: int) -> dict[str, np.ndarray]:
    """Compute Hall/Pedersen winds when HDF5 lacks WeH/WnH."""

    def read(name: str) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="WARNING: missing_value not used since it.*",
                category=UserWarning,
            )
            values = np.asarray(dataset.variables[name][step], dtype=float)
        values[values > 1e30] = np.nan
        return values

    sigma_p = read("SIGMA_PED")
    sigma_h = read("SIGMA_HAL")
    height_m = read("ZG") / 100.0
    u_east = read("UN") * 1e-2
    u_north = read("VN") * 1e-2

    dz = np.diff(height_m, axis=0)
    sigma_p_layer = sigma_p[:-1]
    sigma_h_layer = sigma_h[:-1]
    east_layer = u_east[:-1]
    north_layer = u_north[:-1]
    sigma_p_int = np.nansum(sigma_p_layer * dz, axis=0)
    sigma_h_int = np.nansum(sigma_h_layer * dz, axis=0)

    def weighted(sigma_layer: np.ndarray, sigma_int: np.ndarray):
        east_num = np.nansum(sigma_layer * east_layer * dz, axis=0)
        north_num = np.nansum(sigma_layer * north_layer * dz, axis=0)
        east = np.divide(east_num, sigma_int, out=np.zeros_like(east_num), where=sigma_int > 0.0)
        north = np.divide(
            north_num, sigma_int, out=np.zeros_like(north_num), where=sigma_int > 0.0
        )
        return east, north

    pedersen_east, pedersen_north = weighted(sigma_p_layer, sigma_p_int)
    hall_east, hall_north = weighted(sigma_h_layer, sigma_h_int)
    return {
        "pedersen_east": pedersen_east,
        "pedersen_north": pedersen_north,
        "hall_east": hall_east,
        "hall_north": hall_north,
    }


def load_weighted_winds(
    h5_file: Any, step: int, *, tiegcm_dataset: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Pedersen- and Hall-weighted winds for one prepared step."""
    u_p_east = np.asarray(h5_file["We"][step], dtype=float)
    u_p_north = np.asarray(h5_file["Wn"][step], dtype=float)
    if "WeH" in h5_file and "WnH" in h5_file:
        return (
            u_p_east,
            u_p_north,
            np.asarray(h5_file["WeH"][step], dtype=float),
            np.asarray(h5_file["WnH"][step], dtype=float),
        )
    if tiegcm_dataset is not None:
        weighted = conductivity_weighted_winds_from_tiegcm_step(tiegcm_dataset, step)
        return (
            weighted["pedersen_east"],
            weighted["pedersen_north"],
            weighted["hall_east"],
            weighted["hall_north"],
        )
    raise RuntimeError(
        "Weighted-wind forcing requires Hall-weighted wind. Provide WeH/WnH in the HDF5 "
        "or set SETTINGS.tiegcm_nc so U_H can be computed from SIGMA_HAL."
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
    field: FieldEvaluator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted wind-current source at sheet radius."""
    sigma_p = np.asarray(sigma_p, dtype=float).reshape(-1)
    sigma_h = np.asarray(sigma_h, dtype=float).reshape(-1)
    u_p_theta = np.asarray(u_p_theta, dtype=float).reshape(-1)
    u_p_phi = np.asarray(u_p_phi, dtype=float).reshape(-1)
    u_h_theta = np.asarray(u_h_theta, dtype=float).reshape(-1)
    u_h_phi = np.asarray(u_h_phi, dtype=float).reshape(-1)

    b_r = np.asarray(field.br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.bphi, dtype=float).reshape(-1)
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


def projected_resistance_values(
    dynamics: pynamit.Dynamics, grid: pynamit.Grid, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate projected sheet resistance coefficients on ``grid``."""
    conductance_entry = dynamics.input_timeseries.get_entry("conductance", time)
    if conductance_entry is None:
        raise RuntimeError("Conductance must be set before computing direct wind E_source.")
    field_space = dynamics.input_field_spaces["conductance"]
    evaluator = field_space.representation.get_scalar_evaluation_operator(grid)
    return (
        np.asarray(evaluator.matvec(conductance_entry["etaP"])).reshape(-1),
        np.asarray(evaluator.matvec(conductance_entry["etaH"])).reshape(-1),
    )


def direct_E_source_for_pynamit(
    *,
    sigma_p: np.ndarray,
    sigma_h: np.ndarray,
    u_p_theta: np.ndarray,
    u_p_phi: np.ndarray,
    u_h_theta: np.ndarray,
    u_h_phi: np.ndarray,
    field: FieldEvaluator,
    eta_p: np.ndarray,
    eta_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direct wind electric-field source samples in V/m."""
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
    b_r = np.asarray(field.br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.bphi, dtype=float).reshape(-1)

    q_dot_b = q_r * b_r + q_theta * b_theta + q_phi * b_phi
    q_perp_theta = q_theta - q_dot_b * b_theta
    q_perp_phi = q_phi - q_dot_b * b_phi
    q_cross_b = cross_spherical(q_r, q_theta, q_phi, b_r, b_theta, b_phi)
    return (
        -(eta_p * q_perp_theta + eta_h * q_cross_b[1]),
        -(eta_p * q_perp_phi + eta_h * q_cross_b[2]),
    )


def print_field_stats(label: str, values: np.ndarray) -> None:
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


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable metadata value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_mage_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write MAGE-specific input-package metadata."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8"
    )


def _prepared_input_datasets(dynamics: pynamit.Dynamics) -> list[str]:
    """Return projected input artifacts present in ``dynamics``."""
    artifacts = dynamics.io.scan_run_artifacts()
    return [key for key in INPUT_DATASET_KEYS if key in artifacts]


def project_mage_inputs(settings: MageInputProjectionSettings = SETTINGS) -> Path:
    """Project configured MAGE inputs into a PynaMIT input package."""
    if settings.mainfield_kind not in CENTERED_DIPOLE_MODELS:
        raise ValueError(
            f"Unsupported mainfield_kind {settings.mainfield_kind!r}; "
            f"expected one of {CENTERED_DIPOLE_MODELS}."
        )
    if settings.fac_convention not in ("upward", "field_aligned"):
        raise ValueError(
            "fac_convention must be either 'upward' or 'field_aligned'; "
            f"got {settings.fac_convention!r}."
        )

    h5_path = resolve_existing_path(
        settings.forcing_h5, DEFAULT_FORCING_CANDIDATES, "forcing HDF5"
    )
    tiegcm_path = None
    if settings.tiegcm_nc is not None:
        tiegcm_path = resolve_existing_path(settings.tiegcm_nc, (), "TIEGCM NetCDF")
    else:
        for candidate in DEFAULT_TIEGCM_CANDIDATES:
            expanded_candidate = candidate.expanduser()
            if expanded_candidate.exists():
                tiegcm_path = expanded_candidate
                break

    import h5py

    tiegcm_dataset = None
    if tiegcm_path is not None:
        from netCDF4 import Dataset

        tiegcm_dataset = Dataset(tiegcm_path, mode="r")

    input_directory = Path(
        settings.input_directory
        or projection_directory_for_resolution(settings.nmax, settings.mmax, settings.ncs)
    ).expanduser()
    input_directory.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(h5_path, "r") as file:
            forcing_times, input_times = h5_time_vector_seconds(file["time"][:])
            event_time = forcing_times[0]
            coordinate_time = event_time
            dipole_epoch = decimal_year(event_time)
            RM = boundary_radius_from_h5(file, settings.RM)
            dipole_B0 = dipole_B0_from_h5(file, settings.dipole_B0)
            mainfield = Mainfield(kind=settings.mainfield_kind, epoch=dipole_epoch, B0=dipole_B0)
            gamera_dipole = gamera_internal_dipole_details(file)
            alignment = mainfield.alignment_metadata(event_time)
            rk = dipole_radial_sampling(RI, RM, n_steps=40)

            ionosphere_lat_geo = np.asarray(file["glat"][:], dtype=float)
            ionosphere_lon_geo = wrap_longitude_180(file["glon"][:])
            ionosphere_lat, ionosphere_lon = mainfield.geo_to_model_coordinates(
                ionosphere_lat_geo, ionosphere_lon_geo, event_time=coordinate_time
            )
            ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)

            magnetosphere_lat_raw = np.asarray(file["Blat"][:], dtype=float)
            magnetosphere_lon_raw = np.asarray(file["Blon"][:], dtype=float)

            print(f"Using forcing file: {h5_path}", flush=True)
            if tiegcm_path is not None:
                print(f"Using TIEGCM file for weighted winds: {tiegcm_path}", flush=True)
            print(f"Writing projected input package: {input_directory}", flush=True)
            print(f"Event time: {event_time.isoformat()}", flush=True)
            print(f"Coordinate time: {coordinate_time.isoformat()}", flush=True)
            print(
                "Forcing time span: "
                f"{forcing_times[0].isoformat()} to {forcing_times[-1].isoformat()} "
                f"({len(forcing_times)} step(s))",
                flush=True,
            )
            print(f"Main field used for projection: {settings.mainfield_kind}", flush=True)
            print(f"Dipole alignment model: {alignment['dipole_alignment_model']}", flush=True)
            print(f"Dipole epoch: {dipole_epoch:.9f}", flush=True)
            print(f"Dipole B0: {dipole_B0:.6g} T ({dipole_B0 * 1e9:.6g} nT)", flush=True)
            if gamera_dipole["mag_m0_nT"] is not None:
                print(f"GAMERA signed MagM0: {gamera_dipole['mag_m0_nT']:.6g} nT", flush=True)
            else:
                print(
                    "GAMERA signed MagM0: not present in forcing file; "
                    "assuming Earth-like negative moment for internal-axis metadata",
                    flush=True,
                )
            print("GAMERA coordinates: SM; REMIX longitude 0 = noon", flush=True)
            print(f"RM: {RM:.6g} m", flush=True)
            print("Wind forcing: direct E_source from Pedersen/Hall weighted winds", flush=True)
            print(f"FAC convention: {settings.fac_convention}", flush=True)

            dynamics = pynamit.Dynamics(
                run_directory=input_directory,
                Nmax=settings.nmax,
                Mmax=settings.mmax,
                Ncs=settings.ncs,
                RI=RI,
                RM=RM,
                RM_shielding=False,
                mainfield_kind=settings.mainfield_kind,
                mainfield_epoch=dipole_epoch,
                mainfield_B0=dipole_B0,
                FAC_integration_steps=rk,
                ignore_PFAC=True,
                connect_hemispheres=False,
                latitude_boundary=LATITUDE_BOUNDARY,
                ih_constraint_scaling=1e-5,
                t0=str(event_time),
                save_steady_states=False,
                integrator="exponential",
                artifact_storage=settings.artifact_storage,
            )

            FAC_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)
            wind_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)

            if settings.max_steps is not None:
                input_times = input_times[: int(settings.max_steps)]
                forcing_times = forcing_times[: int(settings.max_steps)]
            n_steps = input_times.size
            if n_steps == 0:
                raise ValueError("No forcing time steps selected for projection.")

            for step in range(n_steps):
                input_time = float(input_times[step])
                print(
                    f"Projecting input step {step + 1} of {n_steps} "
                    f"at t={input_time:g} s ({forcing_times[step].isoformat()})",
                    flush=True,
                )

                delta_Br = np.asarray(file["Bu"][step], dtype=float).reshape(-1) * 1e-9
                if np.any(~np.isfinite(delta_Br)):
                    raise ValueError("Br input contains non-finite values.")
                print_field_stats("  Delta Br [T]", delta_Br)
                magnetosphere_lon = mainfield.local_time_longitude_to_model_longitude(
                    magnetosphere_lon_raw,
                    coordinate_time,
                    local_noon_longitude=MAGE_BR_LOCAL_NOON_LONGITUDE,
                )
                magnetosphere_grid = pynamit.Grid(lat=magnetosphere_lat_raw, lon=magnetosphere_lon)
                dynamics.set_Br(
                    delta_Br,
                    lat=magnetosphere_grid.lat,
                    lon=magnetosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=area_sqrt_weights(magnetosphere_grid.lat),
                    reg_lambda=settings.br_lambda,
                )

                FAC = np.asarray(file["FAC"][step], dtype=float) * 1e-6
                if np.any(~np.isfinite(FAC)):
                    print("  FAC contains non-finite values; setting them to 0.", flush=True)
                    FAC[~np.isfinite(FAC)] = 0.0
                if settings.fac_convention == "field_aligned":
                    jr = FAC.reshape(-1) * FAC_b_evaluator.br
                else:
                    jr = FAC.reshape(-1)
                print_field_stats("  jr [A/m^2]", jr)
                dynamics.set_jr(
                    jr,
                    lat=ionosphere_grid.lat,
                    lon=ionosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=area_sqrt_weights(ionosphere_grid.lat),
                    reg_lambda=settings.jr_lambda,
                )

                sigma_h = np.asarray(file["SH"][step], dtype=float).reshape(-1)
                sigma_p = np.asarray(file["SP"][step], dtype=float).reshape(-1)
                if np.any(~np.isfinite(sigma_h)) or np.any(sigma_h <= 0.0):
                    raise ValueError(
                        "Hall conductance contains non-finite or non-positive values."
                    )
                if np.any(~np.isfinite(sigma_p)) or np.any(sigma_p <= 0.0):
                    raise ValueError(
                        "Pedersen conductance contains non-finite or non-positive values."
                    )
                print_field_stats("  Hall conductance [S]", sigma_h)
                print_field_stats("  Pedersen conductance [S]", sigma_p)
                dynamics.set_conductance(
                    sigma_h,
                    sigma_p,
                    lat=ionosphere_grid.lat,
                    lon=ionosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=area_sqrt_weights(ionosphere_grid.lat),
                    reg_lambda=settings.conductance_lambda,
                )

                u_p_east, u_p_north, u_h_east, u_h_north = load_weighted_winds(
                    file, step, tiegcm_dataset=tiegcm_dataset
                )
                _, _, u_p_east, u_p_north = mainfield.geo_to_model_coordinates(
                    ionosphere_lat_geo,
                    ionosphere_lon_geo,
                    u_p_east,
                    u_p_north,
                    event_time=coordinate_time,
                )
                u_p_theta = -np.asarray(u_p_north, dtype=float).reshape(-1)
                u_p_phi = np.asarray(u_p_east, dtype=float).reshape(-1)
                print_field_stats(
                    "  Pedersen-weighted wind speed [m/s]", np.hypot(u_p_theta, u_p_phi)
                )

                _, _, u_h_east, u_h_north = mainfield.geo_to_model_coordinates(
                    ionosphere_lat_geo,
                    ionosphere_lon_geo,
                    u_h_east,
                    u_h_north,
                    event_time=coordinate_time,
                )
                u_h_theta = -np.asarray(u_h_north, dtype=float).reshape(-1)
                u_h_phi = np.asarray(u_h_east, dtype=float).reshape(-1)
                print_field_stats("  Hall-weighted wind speed [m/s]", np.hypot(u_h_theta, u_h_phi))

                eta_p, eta_h = projected_resistance_values(dynamics, ionosphere_grid, input_time)
                e_source_theta, e_source_phi = direct_E_source_for_pynamit(
                    sigma_p=sigma_p,
                    sigma_h=sigma_h,
                    u_p_theta=u_p_theta,
                    u_p_phi=u_p_phi,
                    u_h_theta=u_h_theta,
                    u_h_phi=u_h_phi,
                    field=wind_b_evaluator,
                    eta_p=eta_p,
                    eta_h=eta_h,
                )
                print_field_stats(
                    "  Direct wind E_source [V/m]", np.hypot(e_source_theta, e_source_phi)
                )
                dynamics.set_E_source(
                    E_source_theta=e_source_theta,
                    E_source_phi=e_source_phi,
                    lat=ionosphere_grid.lat,
                    lon=ionosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=tangential_sqrt_weights(ionosphere_grid.lat),
                    reg_lambda=settings.e_source_lambda,
                )

            projected_datasets = _prepared_input_datasets(dynamics)
            input_time_metadata = {
                "source_time_first": forcing_times[0].isoformat(),
                "source_time_last": forcing_times[-1].isoformat(),
                "input_time_first_s": float(input_times[0]),
                "input_time_last_s": float(input_times[-1]),
                **summarize_input_cadence(input_times),
            }
            write_input_manifest(
                input_directory,
                dynamics.settings,
                input_datasets=projected_datasets,
                source="mage_project_inputs.py",
                notes=(
                    "MAGE direct E_source was computed from Pedersen/Hall weighted winds "
                    "using the sheet-radius main field and projected sheet resistance.",
                ),
                metadata={
                    "input_kind": "mage_gamera_tiegcm",
                    "forcing_h5": str(h5_path),
                    "tiegcm_nc": None if tiegcm_path is None else str(tiegcm_path),
                    "event_time": event_time.isoformat(),
                    "coordinate_time": coordinate_time.isoformat(),
                    "fac_convention": settings.fac_convention,
                    **input_time_metadata,
                },
            )
            _write_mage_metadata(
                input_directory / MAGE_INPUT_METADATA_FILENAME,
                {
                    "forcing_h5": h5_path,
                    "tiegcm_nc": tiegcm_path,
                    "event_time": event_time.isoformat(),
                    "coordinate_time": coordinate_time.isoformat(),
                    "mainfield_kind": settings.mainfield_kind,
                    "mainfield_epoch": dipole_epoch,
                    "dipole_B0_T": dipole_B0,
                    "RM_m": RM,
                    "fac_convention": settings.fac_convention,
                    "n_projected_steps": n_steps,
                    **input_time_metadata,
                    "projected_datasets": projected_datasets,
                    "gamera_dipole": gamera_dipole,
                    "alignment": alignment,
                },
            )

    finally:
        if tiegcm_dataset is not None:
            tiegcm_dataset.close()

    print(f"Projected input package written to {input_directory}", flush=True)
    return input_directory


def main(settings: MageInputProjectionSettings = SETTINGS) -> None:
    """Project MAGE inputs from in-script settings."""
    project_mage_inputs(settings)


if __name__ == "__main__":
    main()
