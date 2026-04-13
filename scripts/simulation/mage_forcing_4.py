"""Run the 2011 MAGE forcing case with homogeneous outer magnetic update.

This script keeps the existing MAGE upstream forcing assembly

    ``(u, Br, jr, Sigma) -> E_{S,I}^{known}``

and then runs the toroidal solve with an explicit homogeneous-outer
column-solve radial-shell response model. The outer magnetic update is held
homogeneous, so ``Br`` is treated only as an upstream contributor to the shell
electric forcing state, not as a separate live ``R_M`` boundary forcing
channel. It is an experimental/diagnostic branch, so it runs under
``benchmark_mode=True`` rather than the supported operational
full-induction runtime mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py as h5
import numpy as np

import pynamit
from pynamit.simulation.induction import (
    CurrentContinuityExteriorToroidalUpdateModel,
    HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel,
    NonlocalShellElectricRadialResponseModel,
    QTraceKnownSourceRadialResponseModel,
)
from pynamit.simulation.mage_time_window import select_mage_time_window
from pynamit.simulation.settings import (
    ConductanceInterpolationMode,
    DynamicsMode,
    ExponentialSolverKind,
    IntegratorKind,
    MainfieldKind,
)

from mage_forcing_3 import (
    BR_LAMBDA,
    CONDUCTANCE_LAMBDA,
    DEFAULT_H5_PATH,
    DT,
    JR_LAMBDA,
    LATITUDE_BOUNDARY,
    NCS,
    NMAX,
    MMAX,
    PLOT,
    RI,
    U_LAMBDA,
    MageProjectionContext,
    _build_projection_context,
    _iter_batch_bounds,
    _latest_precomputed_input_time,
    _log_batch,
    _plot_precomputed_inputs,
    _resolve_run_directory,
    _trim_persisted_input_history,
    dipole_radial_sampling,
)


def _build_dynamics(run_directory: str, *, t0: str) -> pynamit.Dynamics:
    """Create one Dynamics object for the homogeneous-outer-update branch."""
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
        # Keep the upstream MAGE assembly comparable to mage_forcing_3.
        magnetospheric_shielding=False,
        least_squares_solver="normal_eq",
        t0=t0,
        integrator=IntegratorKind.EXPONENTIAL,
        exponential_solver=ExponentialSolverKind.EXPM_MULTIPLY,
        enable_fast_input_path=True,
        conductance_interpolation_mode=ConductanceInterpolationMode.SIGMA_LOG,
        radial_shell_response_model=NonlocalShellElectricRadialResponseModel(
            shell_response_model=QTraceKnownSourceRadialResponseModel(
                q_trace_model=HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel(
                    outer_boundary_mode="closed"
                )
            ),
            exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
        ),
        benchmark_mode=True,
    )


def _print_homogeneous_outer_connector_report(
    dynamics: pynamit.Dynamics, *, time: float, outer_boundary_mode: str = "shielded"
) -> None:
    """Print one compact connector summary for the current input state."""
    dynamics.state.update(dynamics.input_manager, time, interpolation=True)
    state = dynamics.state
    connector = state.get_pragmatic_homogeneous_rm_connector_report(
        outer_boundary_mode=outer_boundary_mode
    )

    print(
        "Homogeneous-outer connector report "
        f"(t={time:.1f} s, outer_mode={outer_boundary_mode})"
    )
    for name in ("wind", "Br", "magnetic_imposed", "total_external"):
        component = connector["component_report"][name]
        print(
            f"  {name:>16s}: "
            f"||q||={component['q_norm']:.3e}, "
            f"||chi||={component['chi_norm']:.3e}, "
            f"||dtjr_known||={component['dtjr_known_norm']:.3e}"
        )
    print(
        "  outer update: "
        "homogeneous (no separate Br(R_M) or psi(R_M) boundary forcing channel)"
    )
    if connector["driver"] is not None:
        print(
            "  FAC driver: "
            f"||dt_alpha_driver||={connector['driver']['dt_alpha_driver_norm']:.3e}, "
            f"||chi_driver||={connector['driver']['chi_driver_norm']:.3e}"
        )
    if connector["live_difference_norm"] is not None:
        print(
            "  live comparison: "
            f"||chi_pragmatic - dpsi_dt_live||={connector['live_difference_norm']:.3e}, "
            f"||q_known - R_I dpsi_dt_live||={connector['known_q_difference_norm']:.3e}"
        )


def _precompute_br(
    dynamics: pynamit.Dynamics,
    file: h5.File,
    *,
    window,
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
    window,
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
    window,
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
    window,
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

        if np.any(np.isnan(u_theta)) or np.any(np.isnan(u_phi)):
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
    window,
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
        _print_homogeneous_outer_connector_report(
            dynamics,
            time=float(window.relative_seconds[-1]) if window.relative_seconds.size > 0 else 0.0,
        )

    return run_directory


def simulate_from_precomputed(*, run_directory: str) -> pynamit.Dynamics:
    """Restart from one precomputed run directory and evolve the simulation."""
    print("Simulation stage")
    dynamics = pynamit.Dynamics.from_directory(
        run_directory,
        radial_shell_response_model=NonlocalShellElectricRadialResponseModel(
            shell_response_model=QTraceKnownSourceRadialResponseModel(
                q_trace_model=HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel(
                    outer_boundary_mode="closed"
                )
            ),
            exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
        ),
    )
    final_time = _latest_precomputed_input_time(dynamics)

    _print_homogeneous_outer_connector_report(dynamics, time=final_time)

    if final_time <= 0:
        print("Selected window contains a single input sample; skipping time evolution.")
        return dynamics

    print(f"Evolving from precomputed inputs to t = {final_time:.1f} s")
    dynamics.evolve_to_time(final_time, dt=DT, sampling_step_interval=1, saving_sample_interval=1)
    return dynamics


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the pragmatic homogeneous-``R_M`` MAGE script."""
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
    """Run the selected MAGE workflow with the pragmatic homogeneous-``R_M`` connector."""
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
    """Parse CLI arguments and run the selected pragmatic MAGE case."""
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
