"""Run the 2011 MAGE forcing case with precompute and simulation stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3
JR_LAMBDA = 0.1
U_LAMBDA = 0.1

IONOSPHERE_WEIGHTING = "mw"
BR_WEIGHTING = "geom_area"

DEFAULT_RUN_DIRECTORY = "results_mage_2011_full_induction"
DEFAULT_PLOT_FILENAME = "input_vs_fitted_comparison_full_induction.png"
DEFAULT_H5_PATH = Path(__file__).resolve().parent / "mage_2011" / "data_H_int.h5"
NMAX, MMAX, NCS = 50, 30, 40
DT = 10.0


@dataclass(frozen=True)
class MageProjectionContext:
    """Static MAGE grid metadata and projection weights."""

    ionosphere_lat: np.ndarray
    ionosphere_lon: np.ndarray
    magnetosphere_lat: np.ndarray
    magnetosphere_lon: np.ndarray
    sqrt_weights_iono_scalar: np.ndarray
    sqrt_weights_iono_vector: np.ndarray
    sqrt_weights_mag_geom: np.ndarray


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


def _resolve_projection_batch_size(num_steps: int, projection_batch_size: int | None) -> int:
    """Return the effective projection batch size."""
    if projection_batch_size is None:
        return num_steps
    if projection_batch_size <= 0:
        raise ValueError(
            f"projection_batch_size must be positive when provided, got {projection_batch_size!r}."
        )
    return min(int(projection_batch_size), max(num_steps, 1))


def _iter_batch_bounds(num_steps: int, projection_batch_size: int | None):
    """Yield half-open batch bounds over one selected time window."""
    batch_size = _resolve_projection_batch_size(num_steps, projection_batch_size)
    for start in range(0, num_steps, batch_size):
        yield start, min(start + batch_size, num_steps)


def _trim_persisted_input_history(dynamics: pynamit.Dynamics, key: str) -> None:
    """Drop older in-memory samples after persisting one input datatype."""
    dynamics.input_timeseries.trim_in_memory(key, keep_last=1)


def _print_toroidal_driver_balance_report(dynamics: pynamit.Dynamics, *, time: float) -> None:
    """Print one compact LL-conflict summary for the current toroidal forcing channels."""
    dynamics.state.update(dynamics.input_manager, time, interpolation=True)
    report = dynamics.state.get_toroidal_driver_balance_report()
    print(
        "Toroidal driver balance report "
        f"(t={time:.1f} s, LL rows={report['constraint_rows']['ll']})"
    )
    ordered_names = (
        "wind",
        "Br",
        "magnetic_imposed",
        "magnetic_driver_raw",
        "magnetic_driver",
        "driver_feedback_rhs",
        "residual_after_driver_subtraction",
        "total_external",
    )
    components = report["components"]
    for name in ordered_names:
        component = components.get(name)
        if component is None:
            continue
        print(
            f"  {name:>31s}: "
            f"||dt_alpha||={component['dt_alpha_norm']:.3e}, "
            f"||C_ll dt_alpha||={component['ll_conflict_norm']:.3e}, "
            f"ratio={component['ll_conflict_ratio']:.3e}"
        )


def _build_projection_context(file: h5.File) -> MageProjectionContext:
    """Load static MAGE grids and reusable projection weights."""
    ionosphere_lat = file["glat"][:]
    ionosphere_lon = file["glon"][:]
    magnetosphere_lat = file["Blat"][:]
    magnetosphere_lon = file["Blon"][:]

    return MageProjectionContext(
        ionosphere_lat=ionosphere_lat,
        ionosphere_lon=ionosphere_lon,
        magnetosphere_lat=magnetosphere_lat,
        magnetosphere_lon=magnetosphere_lon,
        sqrt_weights_iono_scalar=compute_spherical_input_sqrt_weights(
            ionosphere_lat, ionosphere_lon, weighting=IONOSPHERE_WEIGHTING, nmax=NMAX
        ),
        sqrt_weights_iono_vector=compute_spherical_input_sqrt_weights(
            ionosphere_lat, ionosphere_lon, weighting=IONOSPHERE_WEIGHTING, nmax=NMAX, vector=True
        ),
        sqrt_weights_mag_geom=compute_spherical_input_sqrt_weights(
            magnetosphere_lat, magnetosphere_lon, weighting=BR_WEIGHTING, periodic_lon=True
        ),
    )


def _build_dynamics(run_directory: str, *, t0: str) -> pynamit.Dynamics:
    """Create one Dynamics object for this MAGE case."""
    rk, _ = dipole_radial_sampling(RI, 1.5 * RI, n_steps=40)
    return pynamit.Dynamics(
        run_directory=run_directory,
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
        magnetospheric_shielding=False,
        least_squares_solver="normal_eq",
        t0=t0,
        integrator=IntegratorKind.EXPONENTIAL,
        exponential_solver=ExponentialSolverKind.EXPM_MULTIPLY,
        enable_fast_input_path=True,
    )


def _log_batch(label: str, *, batch_start: int, batch_end: int, num_steps: int) -> None:
    """Print one concise batch progress line."""
    print(f"Precomputing {label}: samples {batch_start + 1}-{batch_end} of {num_steps}")


def _precompute_br(
    dynamics: pynamit.Dynamics,
    file: h5.File,
    *,
    window: MageTimeWindow,
    context: MageProjectionContext,
    projection_batch_size: int | None,
) -> None:
    """Project and persist Delta Br inputs for the selected window."""
    num_steps = window.indices.size
    for batch_start, batch_end in _iter_batch_bounds(num_steps, projection_batch_size):
        _log_batch("Br", batch_start=batch_start, batch_end=batch_end, num_steps=num_steps)
        source_indices = window.indices[batch_start:batch_end]
        batch_times = window.relative_seconds[batch_start:batch_end]
        delta_br = np.asarray(file["Bu"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )
        delta_br *= 1e-9

        if np.any(np.isnan(delta_br)):
            raise ValueError("Br input contains NaN values.")

        dynamics.set_Br(
            delta_br,
            lat=context.magnetosphere_lat,
            lon=context.magnetosphere_lon,
            time=batch_times,
            sqrt_weights=context.sqrt_weights_mag_geom,
            reg_lambda=BR_LAMBDA,
        )
        _trim_persisted_input_history(dynamics, "Br")


def _precompute_fac(
    dynamics: pynamit.Dynamics,
    file: h5.File,
    *,
    window: MageTimeWindow,
    context: MageProjectionContext,
    projection_batch_size: int | None,
) -> None:
    """Project and persist FAC inputs for the selected window."""
    num_steps = window.indices.size
    northern_mask = context.ionosphere_lat.reshape(-1) > 0
    for batch_start, batch_end in _iter_batch_bounds(num_steps, projection_batch_size):
        _log_batch("FAC", batch_start=batch_start, batch_end=batch_end, num_steps=num_steps)
        source_indices = window.indices[batch_start:batch_end]
        batch_times = window.relative_seconds[batch_start:batch_end]
        fac = np.asarray(file["FAC"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )
        fac *= 1e-6

        if np.any(np.isnan(fac)):
            print("FAC input contains NaN values. Setting to 0.")
            fac = fac.copy()
            fac[np.isnan(fac)] = 0

        fac[:, northern_mask] *= -1

        dynamics.set_FAC(
            fac,
            lat=context.ionosphere_lat,
            lon=context.ionosphere_lon,
            time=batch_times,
            sqrt_weights=context.sqrt_weights_iono_scalar,
            reg_lambda=JR_LAMBDA,
        )
        _trim_persisted_input_history(dynamics, "jr")


def _precompute_conductance(
    dynamics: pynamit.Dynamics,
    file: h5.File,
    *,
    window: MageTimeWindow,
    context: MageProjectionContext,
    projection_batch_size: int | None,
) -> None:
    """Project and persist conductance inputs for the selected window."""
    num_steps = window.indices.size
    for batch_start, batch_end in _iter_batch_bounds(num_steps, projection_batch_size):
        _log_batch(
            "conductance", batch_start=batch_start, batch_end=batch_end, num_steps=num_steps
        )
        source_indices = window.indices[batch_start:batch_end]
        batch_times = window.relative_seconds[batch_start:batch_end]
        conductance_hall = np.asarray(file["SH"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )
        conductance_pedersen = np.asarray(file["SP"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )

        if np.any(np.isnan(conductance_hall)):
            raise ValueError("Hall conductance input contains NaN values.")
        if np.any(np.isnan(conductance_pedersen)):
            raise ValueError("Pedersen conductance input contains NaN values.")
        if np.any(conductance_hall <= 0):
            raise ValueError("Hall conductance input contains non-positive values.")
        if np.any(conductance_pedersen <= 0):
            raise ValueError("Pedersen conductance input contains non-positive values.")

        dynamics.set_conductance(
            conductance_hall,
            conductance_pedersen,
            lat=context.ionosphere_lat,
            lon=context.ionosphere_lon,
            time=batch_times,
            sqrt_weights=context.sqrt_weights_iono_scalar,
            reg_lambda=CONDUCTANCE_LAMBDA,
        )
        _trim_persisted_input_history(dynamics, "conductance")


def _precompute_wind(
    dynamics: pynamit.Dynamics,
    file: h5.File,
    *,
    window: MageTimeWindow,
    context: MageProjectionContext,
    projection_batch_size: int | None,
) -> None:
    """Project and persist wind inputs for the selected window."""
    num_steps = window.indices.size
    for batch_start, batch_end in _iter_batch_bounds(num_steps, projection_batch_size):
        _log_batch("wind", batch_start=batch_start, batch_end=batch_end, num_steps=num_steps)
        source_indices = window.indices[batch_start:batch_end]
        batch_times = window.relative_seconds[batch_start:batch_end]
        u_east = np.asarray(file["We"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )
        u_north = np.asarray(file["Wn"][source_indices, :, :], dtype=float).reshape(
            batch_end - batch_start, -1
        )
        u_theta = -u_north
        u_phi = u_east

        if np.any(np.isnan(u_theta)):
            raise ValueError("Wind input contains NaN values.")
        if np.any(np.isnan(u_phi)):
            raise ValueError("Wind input contains NaN values.")

        dynamics.set_u(
            u_theta=u_theta,
            u_phi=u_phi,
            lat=context.ionosphere_lat,
            lon=context.ionosphere_lon,
            time=batch_times,
            sqrt_weights=context.sqrt_weights_iono_vector,
            reg_lambda=U_LAMBDA,
        )
        _trim_persisted_input_history(dynamics, "u")


def precompute_inputs(
    *,
    h5_filepath: str | Path,
    window: MageTimeWindow,
    run_directory: str,
    projection_batch_size: int | None,
) -> str:
    """Project all selected MAGE inputs and persist them before simulation."""
    h5_path = Path(h5_filepath).expanduser().resolve()
    print("Precompute stage")
    print("Run directory:", run_directory)
    print(
        "Selected MAGE window:",
        f"{window.start.isoformat(sep=' ')} -> {window.end.isoformat(sep=' ')}",
        f"({window.indices.size} samples)",
    )

    with h5.File(h5_path, "r") as file:
        context = _build_projection_context(file)
        dynamics = _build_dynamics(run_directory, t0=window.start.strftime("%Y-%m-%d %H:%M:%S"))

        _precompute_br(
            dynamics,
            file,
            window=window,
            context=context,
            projection_batch_size=projection_batch_size,
        )
        _precompute_fac(
            dynamics,
            file,
            window=window,
            context=context,
            projection_batch_size=projection_batch_size,
        )
        _precompute_conductance(
            dynamics,
            file,
            window=window,
            context=context,
            projection_batch_size=projection_batch_size,
        )
        _precompute_wind(
            dynamics,
            file,
            window=window,
            context=context,
            projection_batch_size=projection_batch_size,
        )
        _print_toroidal_driver_balance_report(
            dynamics,
            time=float(window.relative_seconds[-1]) if window.relative_seconds.size > 0 else 0.0,
        )

    return run_directory


def _latest_precomputed_input_time(dynamics: pynamit.Dynamics) -> float:
    """Return the latest saved time across all precomputed input datasets."""
    latest_times: list[float] = []
    for key, dataset in dynamics.input_timeseries.datasets.items():
        if int(dataset.sizes.get("time", 0)) > 0:
            latest_times.append(float(np.max(dataset.time.values)))
    if not latest_times:
        raise ValueError(
            f"No precomputed input datasets were found in run_directory={dynamics.run_directory!r}."
        )
    return float(max(latest_times))


def simulate_from_precomputed(*, run_directory: str) -> pynamit.Dynamics:
    """Restart from one precomputed run directory and evolve the simulation."""
    print("Simulation stage")
    dynamics = pynamit.Dynamics.from_directory(run_directory)
    final_time = _latest_precomputed_input_time(dynamics)

    if final_time <= 0:
        print("Selected window contains a single input sample; skipping time evolution.")
        return dynamics

    print(f"Evolving from precomputed inputs to t = {final_time:.1f} s")
    dynamics.evolve_to_time(final_time, dt=DT, sampling_step_interval=1, saving_sample_interval=1)
    return dynamics


def _plot_precomputed_inputs(
    *, h5_filepath: str | Path, run_directory: str, window: MageTimeWindow
) -> None:
    """Plot saved projected inputs against the selected raw MAGE slices."""
    print("Plotting input data")
    timesteps_for_figure = _default_plot_timesteps(window.indices.size)
    data_types_for_figure = ["Br", "jr", "u_mag", "SP", "SH"]

    pynamit.visualization.plot_input_vs_interpolated(
        h5_filepath=str(Path(h5_filepath).expanduser().resolve()),
        interpolated_run_directory=run_directory,
        timesteps_to_plot=timesteps_for_figure,
        data_types_to_plot=data_types_for_figure,
        input_dt=DT,
        noon_longitude=0,
        h5_timestep_offset=int(window.indices[0]),
        vmin_percentile=0,
        vmax_percentile=95,
        output_filename=_resolve_plot_filename(window),
    )


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
        "--projection-batch-size",
        type=int,
        default=None,
        help="Number of timesteps to project at once during the precompute stage. "
        "Default: all selected timesteps.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "precompute", "simulate"),
        default="all",
        help="Which stage to run: project inputs, simulate from precomputed inputs, or both.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the input-vs-interpolated diagnostic plot after precompute.",
    )
    return parser


def run_simulation(
    *,
    h5_filepath: str | Path = DEFAULT_H5_PATH,
    start: str | None = None,
    end: str | None = None,
    run_directory: str | None = None,
    projection_batch_size: int | None = None,
    stage: str = "all",
    plot: bool = PLOT,
) -> pynamit.Dynamics | None:
    """Run the selected MAGE workflow."""
    h5_path = Path(h5_filepath).expanduser().resolve()

    with h5.File(h5_path, "r") as file:
        window = select_mage_time_window(file["time"][:], start=start, end=end)

    resolved_run_directory = _resolve_run_directory(run_directory, window=window)

    if stage in {"all", "precompute"}:
        precompute_inputs(
            h5_filepath=h5_path,
            window=window,
            run_directory=resolved_run_directory,
            projection_batch_size=projection_batch_size,
        )
        if plot:
            _plot_precomputed_inputs(
                h5_filepath=h5_path, run_directory=resolved_run_directory, window=window
            )

    if stage == "precompute":
        return None

    return simulate_from_precomputed(run_directory=resolved_run_directory)


def main() -> None:
    """Parse CLI arguments and run the selected MAGE case."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    run_simulation(
        h5_filepath=args.h5_file,
        start=args.start,
        end=args.end,
        run_directory=args.run_directory,
        projection_batch_size=args.projection_batch_size,
        stage=args.stage,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
