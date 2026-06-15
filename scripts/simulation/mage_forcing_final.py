"""Run PynaMIT with MAGE/GAMERA/TIEGCM forcing.

This is a cleaned-up version of the MAGE forcing workflow.  The default
configuration assumes that the prepared HDF5 file contains geographic
TIEGCM ionospheric inputs and a MAGE inner-boundary magnetic grid whose
longitude is local-time-like.  For neutral-wind forcing, the recommended
default is ``--wind-mode q_eff``: compute the effective sheet-current input
from Appendix A of Laundal et al. (2025), using both Pedersen- and
Hall-weighted winds.

If the HDF5 file does not contain Hall-weighted winds named ``WeH`` and
``WnH``, q_eff mode can derive them from the original TIEGCM NetCDF file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path
from typing import Any

import dipole
import numpy as np
import pynamit

from pynamit.coordinates import wrap_longitude_180
from pynamit.primitives.field_evaluator import FieldEvaluator


SCRIPT_DIR = Path(__file__).resolve().parent
RE = 6381e3
RI = 6.5e6
LATITUDE_BOUNDARY = 35.0

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3.0
JR_LAMBDA = 0.1
U_LAMBDA = 0.1
Q_EFF_LAMBDA = 0.1

# Kaipy/REMIX polar plots place raw longitude 0 at noon.  Treat Blon as a
# local-time longitude and rotate it through MLT -> magnetic longitude ->
# geographic longitude before fitting Br in the IGRF/geographic simulation.
MAGE_BR_LOCAL_NOON_LONGITUDE = 0.0

DEFAULT_FORCING_CANDIDATES = (
    SCRIPT_DIR / "mage_prepared" / "data_H_int_qeff.h5",
    Path("/disk/Gamera_Dong/prep_Pynamit/data_H_int_qeff.h5"),
    Path("mage_2011/data_H_int.h5"),
    Path("/disk/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
    Path("/Users/andreasskeidsvoll/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
)
DEFAULT_TIEGCM_CANDIDATES = (
    Path(
        "/disk/Gamera_Dong/"
        "11OcA_sech_tie_2011-10-24T18-00-10_2011-10-24T19-00-00.nc"
    ),
    Path(
        "/Users/andreasskeidsvoll/Gamera_Dong/"
        "11OcA_sech_tie_2011-10-24T18-00-10_2011-10-24T19-00-00.nc"
    ),
)


def dipole_radial_sampling(r_min: float, r_max: float, n_steps: int) -> np.ndarray:
    """Return radial samples along a dipole field line."""
    ratio = r_min / r_max
    max_angle = np.rad2deg(np.arccos(np.sqrt(ratio)))
    angles = np.linspace(0.0, max_angle, n_steps)
    return r_min / np.cos(np.deg2rad(angles)) ** 2


def resolve_existing_path(
    path: str | None, candidates: tuple[Path, ...], label: str
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


def mage_br_grid_to_geographic(
    magnetic_lat: np.ndarray,
    local_time_lon: np.ndarray,
    event_time: dt.datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MAGE Br grid coordinates to geographic coordinates."""
    dpl = dipole.Dipole(event_time.year)
    mlt = (
        (np.asarray(local_time_lon, dtype=float) - MAGE_BR_LOCAL_NOON_LONGITUDE)
        / 15.0
        + 12.0
    ) % 24.0
    magnetic_lon = dpl.mlt2mlon(mlt, event_time)
    geographic_lat, geographic_lon = dpl.mag2geo(magnetic_lat, magnetic_lon)
    return np.asarray(geographic_lat), wrap_longitude_180(geographic_lon)


def replace_fill_values(values: np.ndarray, fill_threshold: float = 1e30) -> np.ndarray:
    """Return float data with TIEGCM fill values replaced by NaN."""
    array = np.asarray(values, dtype=float)
    array[array > fill_threshold] = np.nan
    return array


def read_tiegcm_step_variable(dataset: Any, name: str, step: int) -> np.ndarray:
    """Read one TIEGCM variable slice while silencing known fill-value warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="WARNING: missing_value not used since it.*",
            category=UserWarning,
        )
        return np.asarray(dataset.variables[name][step])


def conductivity_weighted_winds_from_tiegcm_step(
    dataset: Any, step: int
) -> dict[str, np.ndarray]:
    """Compute height-integrated conductances and weighted winds for one TIEGCM step.

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
        east = np.divide(
            east_num, sigma_int, out=np.zeros_like(east_num), where=sigma_int > 0.0
        )
        north = np.divide(
            north_num,
            sigma_int,
            out=np.zeros_like(north_num),
            where=sigma_int > 0.0,
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
    h5_file: Any,
    step: int,
    *,
    tiegcm_dataset: Any | None,
    require_hall_weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load Pedersen- and optionally Hall-weighted winds for one step."""
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

    if require_hall_weighted:
        raise RuntimeError(
            "Q_eff mode requires Hall-weighted wind. Provide HDF5 datasets WeH/WnH "
            "or pass --tiegcm-nc so the script can compute U_H from SIGMA_HAL."
        )

    return u_p_east, u_p_north, None, None


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


def paper_q_eff_for_pynamit(
    *,
    sigma_p: np.ndarray,
    sigma_h: np.ndarray,
    u_p_theta: np.ndarray,
    u_p_phi: np.ndarray,
    u_h_theta: np.ndarray,
    u_h_phi: np.ndarray,
    field: FieldEvaluator,
    parallel_conductance: float = np.inf,
    br_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PynaMIT Q_eff samples from Appendix A of Laundal et al. (2025).

    Eq. (A3)-(A4) define Pedersen- and Hall-weighted wind-current terms.
    Eq. (A8) projects the full height-integrated wind term into the
    effective tangential sheet-current ``Q_eff``.  PynaMIT's ``set_Q_eff``
    adds the supplied current proxy through ``+ A_res Q_eff`` whereas the
    generalized Ohm's law in Eq. (A11) has ``A_res(J_S - Q_eff)``.  The
    returned samples therefore use the sign convention expected by PynaMIT.

    The Hall wind term below is the term that must enter Eq. (A2) with
    PynaMIT's resistance-tensor convention.  With identical Pedersen- and
    Hall-weighted winds it satisfies ``A_res Q_eff = u x B`` before the
    final PynaMIT sign flip, so q_eff mode reduces to direct neutral-wind
    forcing in the height-independent limit.
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
    q_h = cross_spherical(
        b_r,
        b_theta,
        b_phi,
        u_h_cross_B[0],
        u_h_cross_B[1],
        u_h_cross_B[2],
    )
    q_h_r = sigma_h * q_h[0]
    q_h_theta = sigma_h * q_h[1]
    q_h_phi = sigma_h * q_h[2]

    q_r = q_p_r + q_h_r
    q_theta = q_p_theta + q_h_theta
    q_phi = q_p_phi + q_h_phi

    if np.isinf(parallel_conductance):
        valid = np.abs(b_r) > br_floor
        correction_theta = np.divide(
            b_theta * q_r,
            b_r,
            out=np.zeros_like(q_r),
            where=valid,
        )
        correction_phi = np.divide(
            b_phi * q_r,
            b_r,
            out=np.zeros_like(q_r),
            where=valid,
        )
    else:
        sigma_parallel = float(parallel_conductance)
        denominator = sigma_p * (b_theta**2 + b_phi**2) + sigma_parallel * b_r**2
        r_cross_b_theta = -b_phi
        r_cross_b_phi = b_theta
        numerator_theta = (
            (sigma_parallel - sigma_p) * b_r * b_theta
            - sigma_h * r_cross_b_theta
        ) * q_r
        numerator_phi = (
            (sigma_parallel - sigma_p) * b_r * b_phi
            - sigma_h * r_cross_b_phi
        ) * q_r
        correction_theta = np.divide(
            numerator_theta,
            denominator,
            out=np.zeros_like(q_r),
            where=denominator > 0.0,
        )
        correction_phi = np.divide(
            numerator_phi,
            denominator,
            out=np.zeros_like(q_r),
            where=denominator > 0.0,
        )

    q_eff_theta_physical = q_theta - correction_theta
    q_eff_phi_physical = q_phi - correction_phi

    return -q_eff_theta_physical, -q_eff_phi_physical


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forcing-h5", default=None, help="Prepared MAGE/TIEGCM HDF5 file."
    )
    parser.add_argument(
        "--tiegcm-nc",
        default=None,
        help="Original TIEGCM NetCDF file, needed for q_eff if WeH/WnH are absent.",
    )
    parser.add_argument(
        "--run-directory",
        default=str(SCRIPT_DIR / "mage_runs" / "results_mage_2011_qeff"),
        help="PynaMIT output directory.",
    )
    parser.add_argument(
        "--wind-mode", choices=("q_eff", "neutral_wind", "none"), default="q_eff"
    )
    parser.add_argument("--mainfield-kind", choices=("igrf",), default="igrf")
    parser.add_argument("--nmax", type=int, default=80)
    parser.add_argument("--mmax", type=int, default=80)
    parser.add_argument("--ncs", type=int, default=60)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--final-time", type=float, default=3600.0)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Limit input steps for a short test run."
    )
    parser.add_argument("--br-lambda", type=float, default=BR_LAMBDA)
    parser.add_argument("--conductance-lambda", type=float, default=CONDUCTANCE_LAMBDA)
    parser.add_argument("--jr-lambda", type=float, default=JR_LAMBDA)
    parser.add_argument("--u-lambda", type=float, default=U_LAMBDA)
    parser.add_argument("--q-eff-lambda", type=float, default=Q_EFF_LAMBDA)
    parser.add_argument("--br-floor", type=float, default=1e-3)
    parser.add_argument("--parallel-conductance", type=float, default=np.inf)
    return parser


def main() -> None:
    """Run the configured MAGE/PynaMIT simulation."""
    args = build_arg_parser().parse_args()

    h5_path = resolve_existing_path(
        args.forcing_h5, DEFAULT_FORCING_CANDIDATES, "forcing HDF5"
    )
    tiegcm_path = None
    if args.wind_mode == "q_eff":
        explicit_tiegcm = args.tiegcm_nc
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
            rk = dipole_radial_sampling(RI, 1.5 * RI, n_steps=40)

            ionosphere_lat = np.asarray(file["glat"][:], dtype=float)
            ionosphere_lon = wrap_longitude_180(file["glon"][:])

            magnetosphere_lat, magnetosphere_lon = mage_br_grid_to_geographic(
                np.asarray(file["Blat"][:], dtype=float),
                np.asarray(file["Blon"][:], dtype=float),
                event_time,
            )

            ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)
            magnetosphere_grid = pynamit.Grid(
                lat=magnetosphere_lat, lon=magnetosphere_lon
            )

            print(f"Using forcing file: {h5_path}", flush=True)
            if tiegcm_path is not None:
                print(f"Using TIEGCM file for weighted winds: {tiegcm_path}", flush=True)
            print(f"Event time: {event_time.isoformat()}", flush=True)
            print(f"Wind mode: {args.wind_mode}", flush=True)

            dynamics = pynamit.Dynamics(
                run_directory=args.run_directory,
                Nmax=args.nmax,
                Mmax=args.mmax,
                Ncs=args.ncs,
                RI=RI,
                RM=1.5 * RI,
                mainfield_kind=args.mainfield_kind,
                mainfield_epoch=event_time.year,
                FAC_integration_steps=rk,
                ignore_PFAC=False,
                connect_hemispheres=True,
                latitude_boundary=LATITUDE_BOUNDARY,
                ih_constraint_scaling=1e-5,
                t0=str(event_time),
                integrator="exponential",
            )

            FAC_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)
            q_eff_b_evaluator = FieldEvaluator(dynamics.mainfield, ionosphere_grid, RI)

            n_steps = file["time"].shape[0]
            if args.max_steps is not None:
                n_steps = min(n_steps, int(args.max_steps))

            for step in range(n_steps):
                input_time = args.dt * step
                print(f"Processing input step {step + 1} of {n_steps}", flush=True)

                delta_Br = np.asarray(file["Bu"][step], dtype=float).reshape(-1) * 1e-9
                if np.any(~np.isfinite(delta_Br)):
                    raise ValueError("Br input contains non-finite values.")
                print_field_stats("  Delta Br [T]", delta_Br)
                dynamics.set_Br(
                    delta_Br,
                    lat=magnetosphere_grid.lat,
                    lon=magnetosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=area_sqrt_weights(magnetosphere_grid.lat),
                    reg_lambda=args.br_lambda,
                )

                FAC = np.asarray(file["FAC"][step], dtype=float) * 1e-6
                if np.any(~np.isfinite(FAC)):
                    print("  FAC contains non-finite values; setting them to 0.", flush=True)
                    FAC[~np.isfinite(FAC)] = 0.0
                jr = FAC.reshape(-1) * FAC_b_evaluator.br

                # Flip sign in North (different convention in MAGE).
                jr[ionosphere_lat.reshape(-1) > 0] *= -1

                print_field_stats("  jr [A/m^2]", jr)
                dynamics.set_jr(
                    jr,
                    lat=ionosphere_grid.lat,
                    lon=ionosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=area_sqrt_weights(ionosphere_grid.lat),
                    reg_lambda=args.jr_lambda,
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
                    reg_lambda=args.conductance_lambda,
                )

                if args.wind_mode == "none":
                    continue

                u_p_east, u_p_north, u_h_east, u_h_north = load_weighted_winds(
                    file,
                    step,
                    tiegcm_dataset=tiegcm_dataset,
                    require_hall_weighted=args.wind_mode == "q_eff",
                )
                u_p_theta = -np.asarray(u_p_north, dtype=float).reshape(-1)
                u_p_phi = np.asarray(u_p_east, dtype=float).reshape(-1)
                print_field_stats(
                    "  Pedersen-weighted wind speed [m/s]",
                    np.hypot(u_p_theta, u_p_phi),
                )

                if args.wind_mode == "neutral_wind":
                    dynamics.set_neutral_wind(
                        u_theta=u_p_theta,
                        u_phi=u_p_phi,
                        lat=ionosphere_grid.lat,
                        lon=ionosphere_grid.lon,
                        time=input_time,
                        sqrt_weights=tangential_sqrt_weights(ionosphere_grid.lat),
                        reg_lambda=args.u_lambda,
                    )
                    continue

                if u_h_east is None or u_h_north is None:
                    raise RuntimeError(
                        "Internal error: q_eff mode did not load Hall-weighted winds."
                    )

                u_h_theta = -np.asarray(u_h_north, dtype=float).reshape(-1)
                u_h_phi = np.asarray(u_h_east, dtype=float).reshape(-1)
                print_field_stats(
                    "  Hall-weighted wind speed [m/s]", np.hypot(u_h_theta, u_h_phi)
                )

                q_eff_theta, q_eff_phi = paper_q_eff_for_pynamit(
                    sigma_p=sigma_p,
                    sigma_h=sigma_h,
                    u_p_theta=u_p_theta,
                    u_p_phi=u_p_phi,
                    u_h_theta=u_h_theta,
                    u_h_phi=u_h_phi,
                    field=q_eff_b_evaluator,
                    parallel_conductance=args.parallel_conductance,
                    br_floor=args.br_floor,
                )
                print_field_stats(
                    "  Q_eff magnitude [A/m]", np.hypot(q_eff_theta, q_eff_phi)
                )
                dynamics.set_Q_eff(
                    Q_eff_theta=q_eff_theta,
                    Q_eff_phi=q_eff_phi,
                    lat=ionosphere_grid.lat,
                    lon=ionosphere_grid.lon,
                    time=input_time,
                    sqrt_weights=tangential_sqrt_weights(ionosphere_grid.lat),
                    reg_lambda=args.q_eff_lambda,
                )

            print("Time evolution", flush=True)
            dynamics.evolve_to_time(
                args.final_time,
                dt=args.dt,
                sampling_step_interval=1,
                saving_sample_interval=1,
            )
    finally:
        if tiegcm_dataset is not None:
            tiegcm_dataset.close()


if __name__ == "__main__":
    main()
