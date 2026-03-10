"""Run the 2011 MAGE forcing case with optional delayed-start windowing."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py as h5
import numpy as np

import pynamit
from pynamit.simulation.input import compute_spherical_input_sqrt_weights
from pynamit.simulation.mage_time_window import MageTimeWindow, select_mage_time_window
from pynamit.simulation.settings import (
    DynamicsMode,
    ExponentialSolverKind,
    IntegratorKind,
    MainfieldKind,
)

RI = 6.5e6
LATITUDE_BOUNDARY = 35

PLOT = True

# Regularization parameters
BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3
JR_LAMBDA = 0.1
U_LAMBDA = 0.1

# Input-fit weighting policies.
# Regular ionosphere-grid inputs support unit/sin_theta/mw.
IONOSPHERE_WEIGHTING = "mw"
# Magnetosphere Br grid is curvilinear in geographic coordinates.
BR_WEIGHTING = "geom_area"

DEFAULT_RUN_DIRECTORY = "results_mage_2011_full_induction"
DEFAULT_PLOT_FILENAME = "input_vs_fitted_comparison_full_induction.png"
DEFAULT_H5_PATH = Path(__file__).resolve().parent / "mage_2011" / "data_H_int.h5"
NMAX, MMAX, NCS = 50, 30, 40
DT = 10.0


def dipole_radial_sampling(
    r_min: float, r_max: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate radial sampling points for the dipole model."""
    ratio = r_min / r_max
    max_angle = np.rad2deg(np.arccos(np.sqrt(ratio)))
    angles = np.linspace(0, max_angle, n_steps)
    rk = r_min / np.cos(np.deg2rad(angles)) ** 2
    return rk, angles


def _window_suffix(window: MageTimeWindow) -> str:
    """Return one filename-safe suffix for one selected time window."""
    return f"{window.start:%H%M%S}_{window.end:%H%M%S}"


def _resolve_run_directory(requested_run_directory: str | None, *, window: MageTimeWindow) -> str:
    """Return the run directory for this simulation."""
    if requested_run_directory:
        return requested_run_directory
    if window.requested_start is None and window.requested_end is None:
        return DEFAULT_RUN_DIRECTORY
    return f"{DEFAULT_RUN_DIRECTORY}_{_window_suffix(window)}"


def _resolve_plot_filename(window: MageTimeWindow) -> str:
    """Return the plot output filename for this simulation."""
    if window.requested_start is None and window.requested_end is None:
        return DEFAULT_PLOT_FILENAME
    stem = Path(DEFAULT_PLOT_FILENAME).stem
    suffix = Path(DEFAULT_PLOT_FILENAME).suffix
    return f"{stem}_{_window_suffix(window)}{suffix}"


def _default_plot_timesteps(num_steps: int, count: int = 5) -> list[int]:
    """Return evenly distributed local timesteps for quick-look plots."""
    if num_steps <= 0:
        return []
    if num_steps <= count:
        return list(range(num_steps))
    values = np.linspace(0, num_steps - 1, num=count)
    return sorted({int(round(value)) for value in values})


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the MAGE forcing script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", help="Start of the simulated MAGE window. Use HH:MM, HH:MM:SS, or ISO datetime."
    )
    parser.add_argument(
        "--end", help="End of the simulated MAGE window. Use HH:MM, HH:MM:SS, or ISO datetime."
    )
    parser.add_argument(
        "--run-directory",
        help="Output run directory. Defaults to a window-specific directory name.",
    )
    parser.add_argument(
        "--h5-file",
        default=str(DEFAULT_H5_PATH),
        help="Path to the MAGE HDF5 file. Defaults to the bundled 2011 dataset.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable the input-vs-interpolated diagnostic plot."
    )
    return parser


def run_simulation(
    *,
    h5_filepath: str | Path = DEFAULT_H5_PATH,
    start: str | None = None,
    end: str | None = None,
    run_directory: str | None = None,
    plot: bool = PLOT,
) -> pynamit.Dynamics:
    """Run the MAGE forcing simulation for one selected input window."""
    rk, _ = dipole_radial_sampling(RI, 1.5 * RI, n_steps=40)
    h5_path = Path(h5_filepath).expanduser().resolve()

    with h5.File(h5_path, "r") as file:
        window = select_mage_time_window(file["time"][:], start=start, end=end)
        resolved_run_directory = _resolve_run_directory(run_directory, window=window)

        print(
            "Selected MAGE window:",
            f"{window.start.isoformat(sep=' ')} -> {window.end.isoformat(sep=' ')}",
            f"({window.indices.size} samples)",
        )
        print("Run directory:", resolved_run_directory)

        ionosphere_lat = file["glat"][:]
        ionosphere_lon = file["glon"][:]
        magnetosphere_lat = file["Blat"][:]
        magnetosphere_lon = file["Blon"][:]

        sqrt_weights_iono_scalar = compute_spherical_input_sqrt_weights(
            ionosphere_lat, ionosphere_lon, weighting=IONOSPHERE_WEIGHTING, nmax=NMAX
        )
        sqrt_weights_iono_vector = compute_spherical_input_sqrt_weights(
            ionosphere_lat, ionosphere_lon, weighting=IONOSPHERE_WEIGHTING, nmax=NMAX, vector=True
        )
        sqrt_weights_mag_geom = compute_spherical_input_sqrt_weights(
            magnetosphere_lat, magnetosphere_lon, weighting=BR_WEIGHTING, periodic_lon=True
        )

        print("Setting up simulation object")
        dynamics = pynamit.Dynamics(
            run_directory=resolved_run_directory,
            Nmax=NMAX,
            Mmax=MMAX,
            Ncs=NCS,
            RI=RI,
            RM=1.5 * RI,
            mainfield_kind=MainfieldKind.DIPOLE,
            FAC_integration_steps=rk,
            ignore_PFAC=False,
            connect_hemispheres=True,
            latitude_boundary=LATITUDE_BOUNDARY,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            magnetospheric_toroidal_lock=True,
            magnetospheric_poloidal_lock=False,
            least_squares_solver="normal_eq",
            t0=window.start.strftime("%Y-%m-%d %H:%M:%S"),
            integrator=IntegratorKind.EXPONENTIAL,
            exponential_solver=ExponentialSolverKind.EXPM_MULTIPLY,
        )

        for local_step, source_step in enumerate(window.indices):
            input_time = float(window.relative_seconds[local_step])
            source_timestamp = window.timestamps[local_step]
            print(
                "Processing input step",
                local_step + 1,
                "of",
                window.indices.size,
                f"(source index {source_step}, {source_timestamp.isoformat(sep=' ')})",
            )

            delta_Br = file["Bu"][source_step, :, :].flatten() * 1e-9
            if np.any(np.isnan(delta_Br)):
                raise ValueError("Br input contains NaN values.")

            print("Setting Delta Br with (abs. min, RMS, abs. max):")
            print(
                f"\t({np.min(np.abs(delta_Br))}, "
                f"{np.sqrt(np.mean(delta_Br**2))}, "
                f"{np.max(np.abs(delta_Br))})"
            )

            dynamics.set_Br(
                delta_Br,
                lat=magnetosphere_lat,
                lon=magnetosphere_lon,
                time=input_time,
                sqrt_weights=sqrt_weights_mag_geom,
                reg_lambda=BR_LAMBDA,
            )

            fac = file["FAC"][source_step, :, :] * 1e-6
            if np.any(np.isnan(fac)):
                print("FAC input contains NaN values. Setting to 0.")
                fac[np.isnan(fac)] = 0
            fac[ionosphere_lat > 0] *= -1

            print("Setting FAC with (abs. min, RMS, abs. max):")
            print(f"\t({np.min(np.abs(fac))}, {np.sqrt(np.mean(fac**2))}, {np.max(np.abs(fac))})")

            dynamics.set_FAC(
                fac,
                lat=ionosphere_lat,
                lon=ionosphere_lon,
                time=input_time,
                sqrt_weights=sqrt_weights_iono_scalar,
                reg_lambda=JR_LAMBDA,
            )

            conductance_hall = file["SH"][source_step, :, :].flatten()
            conductance_pedersen = file["SP"][source_step, :, :].flatten()

            if np.any(np.isnan(conductance_hall)):
                raise ValueError("Hall conductance input contains NaN values.")
            if np.any(np.isnan(conductance_pedersen)):
                raise ValueError("Pedersen conductance input contains NaN values.")
            if np.any(conductance_hall <= 0):
                raise ValueError("Hall conductance input contains non-positive values.")
            if np.any(conductance_pedersen <= 0):
                raise ValueError("Pedersen conductance input contains non-positive values.")

            print("Setting Hall conductance with (min, RMS, max):")
            print(
                f"\t({np.min(np.abs(conductance_hall))}, "
                f"{np.sqrt(np.mean(conductance_hall**2))}, "
                f"{np.max(np.abs(conductance_hall))})"
            )
            print("Setting Pedersen conductance with (min, RMS, max):")
            print(
                f"\t({np.min(np.abs(conductance_pedersen))}, "
                f"{np.sqrt(np.mean(conductance_pedersen**2))}, "
                f"{np.max(np.abs(conductance_pedersen))})"
            )

            dynamics.set_conductance(
                conductance_hall,
                conductance_pedersen,
                lat=ionosphere_lat,
                lon=ionosphere_lon,
                time=input_time,
                sqrt_weights=sqrt_weights_iono_scalar,
                reg_lambda=CONDUCTANCE_LAMBDA,
            )

            u_east = file["We"][source_step, :, :]
            u_north = file["Wn"][source_step, :, :]
            u_theta, u_phi = (-u_north.flatten(), u_east.flatten())

            if np.any(np.isnan(u_theta)):
                raise ValueError("Wind input contains NaN values.")
            if np.any(np.isnan(u_phi)):
                raise ValueError("Wind input contains NaN values.")

            print("Setting wind with (abs. min, RMS, abs. max):")
            print(
                f"\t({np.min(np.sqrt(u_theta**2 + u_phi**2))}, "
                f"{np.sqrt(np.mean(u_theta**2 + u_phi**2))}, "
                f"{np.max(np.sqrt(u_theta**2 + u_phi**2))})"
            )

            dynamics.set_u(
                u_theta=u_theta,
                u_phi=u_phi,
                lat=ionosphere_lat,
                lon=ionosphere_lon,
                time=input_time,
                sqrt_weights=sqrt_weights_iono_vector,
                reg_lambda=U_LAMBDA,
            )

    if plot:
        print("Plotting input data")
        timesteps_for_figure = _default_plot_timesteps(window.indices.size)
        data_types_for_figure = ["Br", "jr", "u_mag", "SP", "SH"]

        pynamit.visualization.plot_input_vs_interpolated(
            h5_filepath=str(h5_path),
            interpolated_run_directory=resolved_run_directory,
            timesteps_to_plot=timesteps_for_figure,
            data_types_to_plot=data_types_for_figure,
            input_dt=DT,
            noon_longitude=0,
            h5_timestep_offset=int(window.indices[0]),
            vmin_percentile=0,
            vmax_percentile=95,
            output_filename=_resolve_plot_filename(window),
        )

    print("Time evolution")
    final_time = float(window.relative_seconds[-1])
    if final_time <= 0:
        print("Selected window contains a single input sample; skipping time evolution.")
        return dynamics

    dynamics.evolve_to_time(final_time, dt=DT, sampling_step_interval=1, saving_sample_interval=1)
    return dynamics


def main() -> None:
    """Parse CLI arguments and run the selected MAGE case."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    run_simulation(
        h5_filepath=args.h5_file,
        start=args.start,
        end=args.end,
        run_directory=args.run_directory,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
