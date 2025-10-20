"""Default run script for PynaMIT.

This module contains the function run_pynamit() which sets up and runs a
default PynaMIT simulation. It is primarily used for testing purposes
and as a starting point for simulation scripts.
"""


def run_pynamit(
    final_time=100,
    plotsteps=200,
    dt=5e-4,
    Nmax=20,
    Mmax=20,
    Ncs=30,
    RM=None,
    mainfield_kind="dipole",
    fig_directory="./figs",
    ignore_PFAC=True,
    connect_hemispheres=False,
    latitude_boundary=50,
    wind=False,
    steady_state_initialization=True,
    vector_jr=True,
    vector_Br=True,
    vector_conductance=True,
    vector_u=True,
    integrator="euler",
    jr_lambda=None,
    conductance_lambda=None,
    u_lambda=None,
    multi_data=False,
):
    """Run a default PynaMIT simulation with the given parameters.

    Parameters
    ----------
    final_time : float, optional
        The final time of the simulation in seconds.
    plotsteps : int, optional
        The number of steps between each plot.
    dt : float, optional
        The time step for the simulation.
    Nmax : int, optional
        The maximum degree of the spherical harmonics.
    Mmax : int, optional
        The maximum order of the spherical harmonics.
    Ncs : int, optional
        The number of grid points in the cubed sphere grid.
    mainfield_kind : str, optional
        The type of main field model.
    fig_directory : str, optional
        The directory to save the figures.
    ignore_PFAC : bool, optional
        Whether to ignore the poloidal field-aligned currents.
    connect_hemispheres : bool, optional
        Whether to connect the hemispheres.
    latitude_boundary : float, optional
        The latitude boundary for the simulation.
    wind : bool, optional
        Whether to include wind in the simulation.
    steady_state : bool, optional
        Whether to impose a steady state.
    vector_jr : bool, optional
        Whether to use vector representation for radial current.
    vector_Br : bool, optional
        Whether to use vector representation for magnetic field.
    vector_conductance : bool, optional
        Whether to use vector representation for conductance.
    vector_u : bool, optional
        Whether to use vector representation for wind.
    integrator : {'euler', 'exponential'}, optional
        Integrator type for time evolution.
    jr_lambda : float, optional
        Regularization parameter for the radial current.
    conductance_lambda : float, optional
        Regularization parameter for the conductance.
    u_lambda : float, optional
        Regularization parameter for the wind.

    Returns
    -------
    dynamics : Dynamics
        The dynamics object for performing the simulation and handling
        the simulation results.
    """
    import datetime
    import numpy as np

    from pynamit.math.constants import RE
    from pynamit.simulation.dynamics import Dynamics
    from pynamit.external_inputs import get_conductance_inputs, get_jr_inputs, get_wind_inputs

    # Initialize the 2D ionosphere object at 110 km altitude.
    RI = RE + 110.0e3
    dynamics = Dynamics(
        filename_prefix=None,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RI,
        RM=RM,
        mainfield_kind=mainfield_kind,
        ignore_PFAC=ignore_PFAC,
        connect_hemispheres=connect_hemispheres,
        latitude_boundary=latitude_boundary,
        vector_jr=vector_jr,
        vector_Br=vector_Br,
        vector_conductance=vector_conductance,
        vector_u=vector_u,
        integrator=integrator,
    )

    date = datetime.datetime(2001, 5, 12, 21, 45)
    time = np.linspace(0, final_time, 4) if multi_data else None

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon

    hall, pedersen, conductance_lat, conductance_lon = get_conductance_inputs(
        date, conductance_lat, conductance_lon, time
    )

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    jr, jr_lat, jr_lon = get_jr_inputs(date, jr_lat, jr_lon, time)

    wind_inputs = get_wind_inputs(date, wind=wind, time=time)

    if wind_inputs is not None:
        u_theta, u_phi, u_lat, u_lon, weights = wind_inputs

    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        reg_lambda=conductance_lambda,
        time=time,
    )

    dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=jr_lambda, time=time)

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
