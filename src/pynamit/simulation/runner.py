"""Default run script for PynaMIT.

This module contains the function run_pynamit() which sets up and runs a
default PynaMIT simulation. It is primarily used for testing purposes
and as a starting point for simulation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def run_pynamit(
    final_time: float = 100.0,
    plotsteps: int = 200,
    dt: float = 5e-4,
    Nmax: int = 20,
    Mmax: int = 20,
    Ncs: int = 30,
    RI: Optional[float] = None,
    RM: Optional[float] = None,
    mainfield_kind: str = "dipole",
    ignore_PFAC: bool = True,
    connect_hemispheres: bool = False,
    northern_hemisphere_apex_constraints: bool = False,
    latitude_boundary: float = 50.0,
    wind: bool = False,
    steady_state_initialization: bool = True,
    vector_jr: bool = True,
    vector_Br: bool = True,
    vector_conductance: bool = True,
    vector_u: bool = True,
    integrator: str = "euler",
    jr_lambda: Optional[float] = None,
    conductance_lambda: Optional[float] = None,
    u_lambda: Optional[float] = None,
    multi_data: bool = False,
    solution_basis_kind: str = "SH",
    simulation_mode: Optional[str] = None,
    least_squares_solver: str = "lsmr",
    m_imp_regularization_lambda: float = 0.0,
    mainfield_B0: Optional[float] = None,
    dynamics_mode: str = "legacy",
    mainfield_epoch: int = 2020,
    input_weighting: Optional[str] = None,
    run_directory: Optional[str | Path] = None,
    artifact_storage: str = "auto",
    use_jr: bool = True,
    apply_psi_gauge: bool = True,
    apply_m_ind_gauge: bool = True,
    apply_m_imp_gauge: bool = True,
    magnetospheric_toroidal_lock: bool = False,
    magnetospheric_poloidal_lock: bool = True,
    toroidal_weighting: str = "none",
    poloidal_weighting: str = "none",
    least_squares_preconditioner: Optional[str] = "pinv",
    conductance_interpolation_mode: str = "legacy_eta_linear",
    conductance_interpolation_floor: float = 1e-3,
    toroidal_regularization_lambda: float = 0.0,
    dense_full_operators: bool = False,
    enable_fast_input_path: bool = False,
    exponential_solver: str = "expm",
    benchmark_mode: bool = False,
) -> Any:
    """Run a default PynaMIT simulation with the given parameters.

    Parameters
    ----------
    ...
    run_directory : str or Path, optional
        Directory for one persisted run. If omitted, the run persists to a
        unique timestamped run directory under ``simulation/`` and artifacts are
        written using the requested ``artifact_storage`` policy.
    artifact_storage : {"auto", "netcdf", "zarr"}, optional
        Default storage format for new persisted artifacts. ``"auto"`` prefers
        Zarr when available and otherwise falls back to NetCDF.
    use_jr : bool, optional
        Whether to drive with field aligned currents (default True).
    """
    import datetime
    import numpy as np

    from pynamit.math.constants import RE
    from pynamit.primitives.io import IO
    from pynamit.simulation.dynamics import Dynamics
    from pynamit.simulation.settings import DynamicsSettings, SimulationMode
    from pynamit.simulation.input import compute_spherical_input_sqrt_weights
    from pynamit.data import get_conductance_inputs, get_jr_inputs, get_wind_inputs

    def _infer_structured_shape(
        lat_arr: np.ndarray, lon_arr: np.ndarray
    ) -> Optional[tuple[int, int]]:
        """Infer a structured 2D shape from flattened lat/lon arrays."""
        lat_flat = np.asarray(lat_arr)
        lon_flat = np.asarray(lon_arr)
        if lat_flat.shape != lon_flat.shape:
            return None
        if lat_flat.ndim == 2:
            return lat_flat.shape
        if lat_flat.ndim != 1 or lat_flat.size == 0:
            return None
        unique_lats = np.unique(lat_flat)
        n_lat = unique_lats.size
        if n_lat == 0 or lat_flat.size % n_lat != 0:
            return None
        n_lon = lat_flat.size // n_lat
        return (n_lat, n_lon)

    def _get_sqrt_weights(
        lat, lon, *, weighting: Optional[str], nmax: int, vector: bool = False, strict: bool = True
    ):
        if weighting in (None, "unit"):
            return None

        lat_arr = np.asarray(lat)
        lon_arr = np.asarray(lon)
        shape_2d = _infer_structured_shape(lat_arr, lon_arr)
        if shape_2d is None:
            if strict:
                raise ValueError("Could not infer structured lat/lon shape for weighting.")
            return None

        try:
            lat_2d = lat_arr.reshape(shape_2d)
            lon_2d = lon_arr.reshape(shape_2d)
            return compute_spherical_input_sqrt_weights(
                lat_2d, lon_2d, weighting=weighting, nmax=nmax, vector=vector
            )
        except ValueError:
            if strict:
                raise
            return None

    # Initialize the 2D ionosphere object.
    if RI is None:
        RI = RE + 110.0e3

    resolved_run_directory = (
        IO.build_run_directory(run_directory)
        if run_directory is not None
        else IO.build_temporary_run_directory_in_directory("simulation")
    )

    settings_kwargs = dict(
        run_directory=resolved_run_directory,
        artifact_storage=artifact_storage,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RI,
        RM=RM,
        mainfield_kind=mainfield_kind,
        mainfield_B0=mainfield_B0,
        ignore_PFAC=ignore_PFAC,
        connect_hemispheres=connect_hemispheres,
        northern_hemisphere_apex_constraints=northern_hemisphere_apex_constraints,
        latitude_boundary=latitude_boundary,
        vector_jr=vector_jr,
        vector_Br=vector_Br,
        vector_conductance=vector_conductance,
        vector_u=vector_u,
        integrator=integrator,
        solution_basis_kind=solution_basis_kind,
        least_squares_solver=least_squares_solver,
        m_imp_regularization_lambda=m_imp_regularization_lambda,
        dynamics_mode=dynamics_mode,
        mainfield_epoch=mainfield_epoch,
        apply_psi_gauge=apply_psi_gauge,
        apply_m_ind_gauge=apply_m_ind_gauge,
        apply_m_imp_gauge=apply_m_imp_gauge,
        magnetospheric_toroidal_lock=magnetospheric_toroidal_lock,
        magnetospheric_poloidal_lock=magnetospheric_poloidal_lock,
        toroidal_weighting=toroidal_weighting,
        poloidal_weighting=poloidal_weighting,
        least_squares_preconditioner=least_squares_preconditioner,
        conductance_interpolation_mode=conductance_interpolation_mode,
        conductance_interpolation_floor=conductance_interpolation_floor,
        toroidal_regularization_lambda=toroidal_regularization_lambda,
        dense_full_operators=dense_full_operators,
        enable_fast_input_path=enable_fast_input_path,
        exponential_solver=exponential_solver,
    )
    if simulation_mode is not None:
        settings_kwargs["simulation_mode"] = SimulationMode(simulation_mode)

    settings = DynamicsSettings(**settings_kwargs)
    dynamics = Dynamics(settings, benchmark_mode=benchmark_mode)

    date = datetime.datetime(2001, 5, 12, 21, 45)
    time = np.linspace(0, final_time, 4) if multi_data else None

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon

    # Weighting policy for regular ionosphere-grid inputs (jr, conductance, u).
    ionosphere_input_weighting = input_weighting

    hall, pedersen, conductance_lat, conductance_lon = get_conductance_inputs(
        date, conductance_lat, conductance_lon, time
    )

    w_cond = _get_sqrt_weights(
        conductance_lat,
        conductance_lon,
        weighting=ionosphere_input_weighting,
        nmax=Nmax,
        strict=(input_weighting is not None),
    )

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    jr, jr_lat, jr_lon = get_jr_inputs(date, jr_lat, jr_lon, time)
    if not use_jr:
        jr = np.zeros_like(jr)

    w_jr = _get_sqrt_weights(
        jr_lat,
        jr_lon,
        weighting=ionosphere_input_weighting,
        nmax=Nmax,
        strict=(input_weighting is not None),
    )

    wind_inputs = get_wind_inputs(date, wind=wind, time=time)

    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs
        if ionosphere_input_weighting is not None:
            w_u = _get_sqrt_weights(
                u_lat,
                u_lon,
                weighting=ionosphere_input_weighting,
                nmax=Nmax,
                vector=True,
                strict=(input_weighting is not None),
            )
            if w_u is not None:
                weights = w_u

    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=conductance_lambda,
        sqrt_weights=w_cond,
        time=time,
    )

    if use_jr:
        dynamics.set_jr(
            jr, lat=jr_lat, lon=jr_lon, reg_lambda=jr_lambda, sqrt_weights=w_jr, time=time
        )

    if wind_inputs is not None:
        dynamics.set_u(
            u_theta=u_theta,
            u_phi=u_phi,
            lat=u_lat,
            lon=u_lon,
            sqrt_weights=weights,
            reg_lambda=u_lambda,
            time=time,
        )

    dynamics.evolve_to_time(
        t=final_time,
        dt=dt,
        sampling_step_interval=1,
        saving_sample_interval=plotsteps,
        steady_state_initialization=steady_state_initialization,
    )

    return dynamics
