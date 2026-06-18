"""Run PynaMIT with MAGE/GAMERA/TIEGCM forcing.

This is a cleaned-up version of the MAGE forcing workflow.  The default
configuration assumes that the prepared HDF5 file contains TIEGCM
ionospheric inputs on a geographic grid and a MAGE/REMIX inner-boundary
magnetic grid whose longitude is local-time-like. The default PynaMIT
main field is ``kaiju_dipole``: a centered dipole with Kaiju/Geopack
alignment, using SM coordinates as the model horizontal coordinates.
The TIEGCM grid is converted through ``Mainfield`` helpers, and wind
vectors are rotated into the model east/north basis before setting the
inputs.

Neutral-wind forcing is applied as a direct electric-field source from
the Pedersen- and Hall-weighted wind-current terms, using the same
projected sheet resistance as PynaMIT.

If the HDF5 file does not contain Hall-weighted winds named ``WeH`` and
``WnH``, wind forcing can derive them from the original TIEGCM NetCDF
file.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pynamit

from pynamit.coordinates import wrap_longitude_180
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.simulation.mainfield import Mainfield, decimal_year


SCRIPT_DIR = Path(__file__).resolve().parent
RE = 6381e3
RI = 6.5e6
LATITUDE_BOUNDARY = 35.0

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3.0
JR_LAMBDA = 0.1
E_SOURCE_LAMBDA = 0.1

# Kaipy/REMIX polar plots place raw longitude 0 at noon. In
# kaiju_dipole mode this is already the SM longitude origin used by
# the run. Legacy dipole mode keeps the old MLT -> centered-dipole
# magnetic-longitude conversion.
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


@dataclass(frozen=True)
class MageForcingSettings:
    """Defaults intended to be edited in this script for normal runs."""

    forcing_h5: Path | None = None
    tiegcm_nc: Path | None = None
    run_directory: Path = SCRIPT_DIR / "mage_runs" / "results_mage_2011_kaiju_direct_e"
    mainfield_kind: str = "kaiju_dipole"
    dipole_B0: float | None = None
    fac_convention: str = "upward"
    RM: float | None = None
    RM_shielding: bool = False
    nmax: int = 80
    mmax: int = 80
    ncs: int = 60
    dt: float = 10.0
    final_time: float = 3600.0
    max_steps: int | None = None
    sampling_step_interval: int = 1
    saving_sample_interval: int = 6
    br_lambda: float = BR_LAMBDA
    conductance_lambda: float = CONDUCTANCE_LAMBDA
    jr_lambda: float = JR_LAMBDA
    e_source_lambda: float = E_SOURCE_LAMBDA
    steady_state_initialization: bool = False
    save_steady_states: bool = False


SETTINGS = MageForcingSettings()


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
        if candidate.exists():
            return candidate

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked: {formatted}")


def parse_h5_time(value: Any) -> dt.datetime:
    """Parse an HDF5 ISO timestamp stored as bytes or str."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return dt.datetime.fromisoformat(str(value))


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
    """Return the centered-dipole equatorial field magnitude.

    The return value is in tesla.
    """
    if explicit_B0 is not None:
        return float(explicit_B0)
    if "gamera_dipole_B0_T" in h5_file.attrs:
        return float(h5_file.attrs["gamera_dipole_B0_T"])
    if "gamera_mag_m0_nT" in h5_file.attrs:
        return abs(float(h5_file.attrs["gamera_mag_m0_nT"])) * 1e-9
    return MAGE_DIPOLE_B0_T


def gamera_mag_m0_from_h5(h5_file: Any) -> float | None:
    """Return signed GAMERA MagM0 in nT if available."""
    if "gamera_mag_m0_nT" in h5_file.attrs:
        return float(h5_file.attrs["gamera_mag_m0_nT"])
    return None


def gamera_internal_axis_from_h5(h5_file: Any, name: str, fallback: np.ndarray) -> np.ndarray:
    """Return a stored GAMERA internal axis or a fallback axis."""
    if name in h5_file.attrs:
        axis = np.asarray(h5_file.attrs[name], dtype=float)
        if axis.shape == (3,) and np.linalg.norm(axis) > 0.0:
            unit = axis / np.linalg.norm(axis)
            unit[np.isclose(unit, 0.0)] = 0.0
            return unit
    fallback = np.asarray(fallback, dtype=float)
    unit = fallback / np.linalg.norm(fallback)
    unit[np.isclose(unit, 0.0)] = 0.0
    return unit


def gamera_internal_dipole_details(h5_file: Any) -> dict[str, np.ndarray | float | None]:
    """Return signed GAMERA dipole details from prepared metadata."""
    mag_m0_nT = gamera_mag_m0_from_h5(h5_file)
    sign = -1.0
    if mag_m0_nT is not None and np.isfinite(mag_m0_nT) and mag_m0_nT != 0.0:
        sign = float(np.sign(mag_m0_nT))
    moment_fallback = np.array([0.0, 0.0, sign])
    north_fallback = -moment_fallback
    return {
        "mag_m0_nT": mag_m0_nT,
        "moment_axis": gamera_internal_axis_from_h5(
            h5_file, "gamera_internal_dipole_moment_axis", moment_fallback
        ),
        "north_axis": gamera_internal_axis_from_h5(
            h5_file, "gamera_internal_magnetic_north_axis", north_fallback
        ),
    }


def replace_fill_values(values: np.ndarray, fill_threshold: float = 1e30) -> np.ndarray:
    """Return float data with TIEGCM fill values replaced by NaN."""
    array = np.asarray(values, dtype=float)
    array[array > fill_threshold] = np.nan
    return array


def read_tiegcm_step_variable(dataset: Any, name: str, step: int) -> np.ndarray:
    """Read one TIEGCM slice while silencing fill-value warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="WARNING: missing_value not used since it.*", category=UserWarning
        )
        return np.asarray(dataset.variables[name][step])


def conductivity_weighted_winds_from_tiegcm_step(dataset: Any, step: int) -> dict[str, np.ndarray]:
    """Compute height-integrated conductances and winds.

    The returned winds are conductivity-weighted averages:

        U_c = int sigma_c u dz / int sigma_c dz,

    with c in {P, H}.  TIEGCM wind inputs are cm/s and heights are cm.
    """
    sigma_p = replace_fill_values(read_tiegcm_step_variable(dataset, "SIGMA_PED", step))
    sigma_h = replace_fill_values(read_tiegcm_step_variable(dataset, "SIGMA_HAL", step))
    height_m = replace_fill_values(read_tiegcm_step_variable(dataset, "ZG", step)) / 100.0
    u_east = replace_fill_values(read_tiegcm_step_variable(dataset, "UN", step)) * 1e-2
    u_north = replace_fill_values(read_tiegcm_step_variable(dataset, "VN", step)) * 1e-2

    dz = np.diff(height_m, axis=0)
    sigma_p_layer = sigma_p[:-1]
    sigma_h_layer = sigma_h[:-1]
    east_layer = u_east[:-1]
    north_layer = u_north[:-1]

    sigma_p_int = np.nansum(sigma_p_layer * dz, axis=0)
    sigma_h_int = np.nansum(sigma_h_layer * dz, axis=0)

    def weighted_wind(
        sigma_layer: np.ndarray, sigma_int: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        east_num = np.nansum(sigma_layer * east_layer * dz, axis=0)
        north_num = np.nansum(sigma_layer * north_layer * dz, axis=0)
        east = np.divide(east_num, sigma_int, out=np.zeros_like(east_num), where=sigma_int > 0.0)
        north = np.divide(
            north_num, sigma_int, out=np.zeros_like(north_num), where=sigma_int > 0.0
        )
        return east, north

    pedersen_east, pedersen_north = weighted_wind(sigma_p_layer, sigma_p_int)
    hall_east, hall_north = weighted_wind(sigma_h_layer, sigma_h_int)

    return {
        "sigma_p": sigma_p_int,
        "sigma_h": sigma_h_int,
        "pedersen_east": pedersen_east,
        "pedersen_north": pedersen_north,
        "hall_east": hall_east,
        "hall_north": hall_north,
    }


def load_weighted_winds(
    h5_file: Any, step: int, *, tiegcm_dataset: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load weighted winds for one step."""
    u_p_east = np.asarray(h5_file["We"][step], dtype=float)
    u_p_north = np.asarray(h5_file["Wn"][step], dtype=float)

    if "WeH" in h5_file and "WnH" in h5_file:
        u_h_east = np.asarray(h5_file["WeH"][step], dtype=float)
        u_h_north = np.asarray(h5_file["WnH"][step], dtype=float)
        return u_p_east, u_p_north, u_h_east, u_h_north

    if tiegcm_dataset is not None:
        weighted = conductivity_weighted_winds_from_tiegcm_step(tiegcm_dataset, step)
        return (
            weighted["pedersen_east"],
            weighted["pedersen_north"],
            weighted["hall_east"],
            weighted["hall_north"],
        )

    raise RuntimeError(
        "Weighted-wind forcing requires Hall-weighted wind. Provide HDF5 datasets WeH/WnH "
        "or set SETTINGS.tiegcm_nc so the script can compute U_H from SIGMA_HAL."
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
    """Return the height-integrated Pedersen+Hall wind-current source.

    The source follows the height-integrated Ohm's-law expansion
    ``Q = SigmaP (U_P x B) + SigmaH b x (U_H x B)``. The Hall rotation
    sign is the one required by PynaMIT's ``j x b`` resistance tensor
    and by the height-independent limit: when ``U_P = U_H = u``, the
    sheet resistance gives the ordinary ``-(u x B)`` electric field.

    ``field`` is evaluated at the PynaMIT sheet radius. This uses the
    same sheet-constant main-field geometry as the ``JS -> E_S`` closure
    instead of a height-varying ``B(z)`` integral.
    """
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
    q_p_r = sigma_p * u_p_cross_B[0]
    q_p_theta = sigma_p * u_p_cross_B[1]
    q_p_phi = sigma_p * u_p_cross_B[2]

    u_h_cross_B = cross_spherical(zero, u_h_theta, u_h_phi, B_r, B_theta, B_phi)
    q_h = cross_spherical(b_r, b_theta, b_phi, u_h_cross_B[0], u_h_cross_B[1], u_h_cross_B[2])
    q_h_r = sigma_h * q_h[0]
    q_h_theta = sigma_h * q_h[1]
    q_h_phi = sigma_h * q_h[2]

    return q_p_r + q_h_r, q_p_theta + q_h_theta, q_p_phi + q_h_phi


def projected_resistance_values(
    dynamics: pynamit.Dynamics, grid: pynamit.Grid, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate projected sheet resistance coefficients on ``grid``."""
    conductance_entry = dynamics.input_timeseries.get_entry("conductance", time)
    if conductance_entry is None:
        raise RuntimeError("Conductance must be set before computing direct wind E_source.")

    field_space = dynamics.input_field_spaces["conductance"]
    evaluator = field_space.representation.get_scalar_evaluation_operator(grid)
    eta_p = np.asarray(evaluator.matvec(conductance_entry["etaP"])).reshape(-1)
    eta_h = np.asarray(evaluator.matvec(conductance_entry["etaH"])).reshape(-1)
    return eta_p, eta_h


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
    """Compute direct wind electric-field source samples in V/m.

    This evaluates ``-P_S R_3D Q`` in the ``eta_parallel = 0`` model
    without constructing the Eq. (A8) current-equivalent ``Q_eff``.
    """
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

    e_theta = -(eta_p * q_perp_theta + eta_h * q_cross_b[1])
    e_phi = -(eta_p * q_perp_phi + eta_h * q_cross_b[2])
    return e_theta, e_phi


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


def main(settings: MageForcingSettings = SETTINGS) -> None:
    """Run the configured MAGE/PynaMIT simulation."""
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
    explicit_tiegcm = settings.tiegcm_nc
    if explicit_tiegcm is not None:
        tiegcm_path = resolve_existing_path(explicit_tiegcm, (), "TIEGCM NetCDF")
    else:
        for candidate in DEFAULT_TIEGCM_CANDIDATES:
            if candidate.exists():
                tiegcm_path = candidate
                break

    import h5py

    tiegcm_dataset = None
    if tiegcm_path is not None:
        from netCDF4 import Dataset

        tiegcm_dataset = Dataset(tiegcm_path, mode="r")

    try:
        with h5py.File(h5_path, "r") as file:
            event_time = parse_h5_time(file["time"][0])
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
            # PynaMIT's main-field geometry and PFAC matrix are built
            # once, so all inputs are expressed in one frozen SM frame.
            ionosphere_lat, ionosphere_lon = mainfield.geo_to_model_coordinates(
                ionosphere_lat_geo, ionosphere_lon_geo, event_time=coordinate_time
            )

            magnetosphere_lat_raw = np.asarray(file["Blat"][:], dtype=float)
            magnetosphere_lon_raw = np.asarray(file["Blon"][:], dtype=float)
            ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)

            print(f"Using forcing file: {h5_path}", flush=True)
            if tiegcm_path is not None:
                print(f"Using TIEGCM file for weighted winds: {tiegcm_path}", flush=True)
            print(f"Event time: {event_time.isoformat()}", flush=True)
            print(f"Coordinate frame time: {coordinate_time.isoformat()}", flush=True)
            print(f"Main field: {settings.mainfield_kind}", flush=True)
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
            print(
                "GAMERA internal moment axis: "
                f"{gamera_dipole['moment_axis'][0]:.3g}, "
                f"{gamera_dipole['moment_axis'][1]:.3g}, "
                f"{gamera_dipole['moment_axis'][2]:.3g}; "
                "magnetic north axis: "
                f"{gamera_dipole['north_axis'][0]:.3g}, "
                f"{gamera_dipole['north_axis'][1]:.3g}, "
                f"{gamera_dipole['north_axis'][2]:.3g}",
                flush=True,
            )
            print(
                "Dipole north pole GEO lat/lon: "
                f"{alignment['north_pole_geo_lat_lon'][0]:.6f}, "
                f"{alignment['north_pole_geo_lat_lon'][1]:.6f}",
                flush=True,
            )
            print(
                "Dipole axis GEO Cartesian: "
                f"{alignment['axis_geo_cartesian'][0]:.8f}, "
                f"{alignment['axis_geo_cartesian'][1]:.8f}, "
                f"{alignment['axis_geo_cartesian'][2]:.8f}",
                flush=True,
            )
            print(
                f"Noon longitude in run coordinates: {alignment['noon_mlon_deg']:.6f} deg",
                flush=True,
            )
            print(f"RM: {RM:.6g} m", flush=True)
            print(f"Induced RM shielding: {settings.RM_shielding}", flush=True)
            print("Wind forcing: direct E_source from Pedersen/Hall weighted winds", flush=True)
            print(f"FAC convention: {settings.fac_convention}", flush=True)
            print(
                f"Steady-state initialization: {settings.steady_state_initialization}", flush=True
            )

            dynamics = pynamit.Dynamics(
                run_directory=settings.run_directory,
                Nmax=settings.nmax,
                Mmax=settings.mmax,
                Ncs=settings.ncs,
                RI=RI,
                RM=RM,
                RM_shielding=settings.RM_shielding,
                mainfield_kind=settings.mainfield_kind,
                mainfield_epoch=dipole_epoch,
                mainfield_B0=dipole_B0,
                FAC_integration_steps=rk,
                ignore_PFAC=False,
                connect_hemispheres=True,
                latitude_boundary=LATITUDE_BOUNDARY,
                ih_constraint_scaling=1e-5,
                t0=str(event_time),
                save_steady_states=settings.save_steady_states,
                integrator="exponential",
            )

            FAC_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)
            wind_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)

            n_steps = file["time"].shape[0]
            if settings.max_steps is not None:
                n_steps = min(n_steps, int(settings.max_steps))

            for step in range(n_steps):
                input_time = settings.dt * step
                print(f"Processing input step {step + 1} of {n_steps}", flush=True)

                delta_Br = np.asarray(file["Bu"][step], dtype=float).reshape(-1) * 1e-9
                if np.any(~np.isfinite(delta_Br)):
                    raise ValueError("Br input contains non-finite values.")
                print_field_stats("  Delta Br [T]", delta_Br)
                magnetosphere_lat = np.asarray(magnetosphere_lat_raw, dtype=float)
                magnetosphere_lon = mainfield.local_time_longitude_to_model_longitude(
                    magnetosphere_lon_raw,
                    coordinate_time,
                    local_noon_longitude=MAGE_BR_LOCAL_NOON_LONGITUDE,
                )
                magnetosphere_grid = pynamit.Grid(lat=magnetosphere_lat, lon=magnetosphere_lon)
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

            print(
                "Time evolution: "
                f"final_time={settings.final_time:g} s, dt={settings.dt:g} s, "
                f"sample every {settings.sampling_step_interval} step(s), "
                f"save every {settings.saving_sample_interval} sample(s)",
                flush=True,
            )
            dynamics.evolve_to_time(
                settings.final_time,
                dt=settings.dt,
                sampling_step_interval=settings.sampling_step_interval,
                saving_sample_interval=settings.saving_sample_interval,
                steady_state_initialization=settings.steady_state_initialization,
                run_steady_state=settings.save_steady_states,
            )
            print("Time evolution complete", flush=True)
    finally:
        if tiegcm_dataset is not None:
            tiegcm_dataset.close()


if __name__ == "__main__":
    main()
