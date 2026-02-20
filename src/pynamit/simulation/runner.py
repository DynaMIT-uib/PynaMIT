"""Default run script for PynaMIT.

This module contains the function run_pynamit() which sets up and runs a
default PynaMIT simulation. It is primarily used for testing purposes
and as a starting point for simulation scripts.
"""

from __future__ import annotations

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
    fig_directory: str = "./figs",
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
    least_squares_solver: str = "cg",
    m_imp_regularization_lambda: float = 0.0,
    mainfield_B0: Optional[float] = None,
    dynamics_mode: str = "legacy",
    mainfield_epoch: int = 2020,
    use_exact_weights: bool = False,
    filename_prefix: Optional[str] = None,
    use_jr: bool = True,
    apply_psi_gauge: bool = True,
    apply_m_ind_gauge: bool = True,
    magnetospheric_toroidal_lock: bool = False,
    magnetospheric_poloidal_lock: bool = True,
    toroidal_weighting: str = "none",
    poloidal_weighting: str = "none",
    least_squares_preconditioner: Optional[str] = "pinv",
    toroidal_regularization_lambda: float = 0.0,
    dense_full_operators: bool = False,
    benchmark_mode: bool = False,
) -> Any:
    """Run a default PynaMIT simulation with the given parameters.
    
    Parameters
    ----------
    ...
    filename_prefix : str, optional
        Prefix for output files.
    use_jr : bool, optional
        Whether to drive with field aligned currents (default True).
    """
    import datetime
    import numpy as np

    from pynamit.math.constants import RE
    from pynamit.simulation.dynamics import Dynamics, SimulationMode
    from pynamit.data import get_conductance_inputs, get_jr_inputs, get_wind_inputs
    from pynamit.spherical_harmonics.sh_basis import SHBasis

    # Helper for weight generation
    def _get_weights(lat, lon, Nmax, use_exact):
        if not use_exact:
            return None
            
        # Check for regularity (Iso-Latitude)
        # 1D Unique Latitudes
        unique_lats = np.unique(lat)
        n_lat = len(unique_lats)
        if n_lat < 2: return None # Robustness
        
        # Check if grid size matches N_lat * N_lon roughly
        # Or simply: can we construct a 2D mesh?
        # Assuming lat/lon are flattened.
        if lat.size % n_lat != 0:
            # Not a simple product grid?
            # Fallback
            return None
            
        n_lon = lat.size // n_lat
        
        # Compute exact weights for unique lats
        # Sort descending (90 to -90) or however they appear?
        # compute_exact_weights takes theta in radians.
        # Ensure unique_lats are sorted as they appear in the grid?
        # If grid is (lat, lon) meshgrid, lats are block constants.
        
        # We assume standard meshgrid order or consistent blocks.
        # Let's verify separability more strictly if needed, but for now specific to regular grids.
        
        theta_1d = np.deg2rad(90 - unique_lats)
        # Note: compute_exact_weights requires theta to be sorted? 
        # Integration depends on points. Order implies output order.
        # If unique_lats is sorted (np.unique does), theta_1d is sorted (0 to pi).
        
        weights_1d = SHBasis.compute_exact_weights(theta_1d, Nmax)
        
        # We need to map these 1D weights back to the full grid (lat, lon).
        # We can use a lookup or broadcast if we know the shape.
        # Safe way: interp? Or lookup.
        
        # Create a map lat -> weight
        w_map = {l: w for l, w in zip(unique_lats, weights_1d)}
        
        # Map to full grid
        # Only works if lat contains EXACTLY the unique values (it does by definition).
        weights_full = np.array([w_map[l] for l in lat.flatten()])
        
        return np.sqrt(weights_full) # Return sqrt weights for solver

    # Initialize the 2D ionosphere object.
    if RI is None:
        RI = RE + 110.0e3

    dynamics = Dynamics(
        filename_prefix=filename_prefix,
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
        simulation_mode=None if simulation_mode is None else SimulationMode(simulation_mode),
        least_squares_solver=least_squares_solver,
        m_imp_regularization_lambda=m_imp_regularization_lambda,
        dynamics_mode=dynamics_mode,
        mainfield_epoch=mainfield_epoch,
        apply_psi_gauge=apply_psi_gauge,
        apply_m_ind_gauge=apply_m_ind_gauge,
        magnetospheric_toroidal_lock=magnetospheric_toroidal_lock,
        magnetospheric_poloidal_lock=magnetospheric_poloidal_lock,
        toroidal_weighting=toroidal_weighting,
        poloidal_weighting=poloidal_weighting,
        least_squares_preconditioner=least_squares_preconditioner,
        toroidal_regularization_lambda=toroidal_regularization_lambda,
        dense_full_operators=dense_full_operators,
        benchmark_mode=benchmark_mode,
    )


    date = datetime.datetime(2001, 5, 12, 21, 45)
    time = np.linspace(0, final_time, 4) if multi_data else None


    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon

    hall, pedersen, conductance_lat, conductance_lon = get_conductance_inputs(
        date, conductance_lat, conductance_lon, time
    )
    
    w_cond = _get_weights(conductance_lat, conductance_lon, Nmax, use_exact_weights)

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    jr, jr_lat, jr_lon = get_jr_inputs(date, jr_lat, jr_lon, time)
    if not use_jr:
        jr = np.zeros_like(jr)
    
    w_jr = _get_weights(jr_lat, jr_lon, Nmax, use_exact_weights)

    wind_inputs = get_wind_inputs(date, wind=wind, time=time)

    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs
        if use_exact_weights:
             w_u = _get_weights(u_lat, u_lon, Nmax, True)
             if w_u is not None:
                 weights = np.tile(w_u, (2, 1)).flatten().reshape(2, -1) # Vector weighting?
                 # Or just pass flat? set_u expects...
                 # set_u signature: sqrt_weights.
                 # Usually expects 1D or matched to input.
                 # The default `weights` from input is None?
                 pass

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
            jr, 
            lat=jr_lat, 
            lon=jr_lon, 
            reg_lambda=jr_lambda, 
            sqrt_weights=w_jr,
            time=time
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
