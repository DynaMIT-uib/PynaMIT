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
    mainfield_kind="dipole",
    fig_directory="./figs",
    ignore_PFAC=True,
    connect_hemispheres=False,
    latitude_boundary=50,
    wind=False,
    steady_state=False,
    vector_jr=True,
    vector_conductance=True,
    vector_u=True,
    integrator="euler",
    jr_lambda=None,
    conductance_lambda=None,
    u_lambda=None,
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
    import os

    import datetime
    import numpy as np

    from lompe import conductance
    import dipole
    import pyamps
    import pyhwm2014  # https://github.com/rilma/pyHWM14

    from pynamit.simulation.dynamics import Dynamics
    from pynamit.math.constants import RE

    # Initialize the 2D ionosphere object at 110 km altitude.
    RI = RE + 110.0e3
    dynamics = Dynamics(
        filename_prefix=None,
        Nmax=Nmax,
        Mmax=Mmax,
        Ncs=Ncs,
        RI=RI,
        mainfield_kind=mainfield_kind,
        ignore_PFAC=ignore_PFAC,
        connect_hemispheres=connect_hemispheres,
        latitude_boundary=latitude_boundary,
        vector_jr=vector_jr,
        vector_conductance=vector_conductance,
        vector_u=vector_u,
        integrator=integrator,
    )

    date = datetime.datetime(2001, 5, 12, 21, 45)

    # Get and set conductance input.
    conductance_lat = dynamics.state_grid.lat
    conductance_lon = dynamics.state_grid.lon
    Kp = 5
    hall, pedersen = conductance.hardy_EUV(
        conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=True
    )
    dynamics.set_conductance(
        hall, pedersen, lat=conductance_lat, lon=conductance_lon, reg_lambda=conductance_lambda
    )

    # Get and set jr input.
    jr_lat = dynamics.state_grid.lat
    jr_lon = dynamics.state_grid.lon
    d = dipole.Dipole(date.year)
    a = pyamps.AMPS(
        300,
        0,
        -4,
        20,
        100,
        minlat=50,
        coeff_fn=os.path.join(
            os.path.dirname(pyamps.__file__),
            "coefficients",
            "SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt",
        ),
    )
    jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
    # Filter low latitude jr.
    jr[np.abs(jr_lat) < 50] = 0
    dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon, reg_lambda=jr_lambda)

    # Get and set wind input.
    if wind:
        hwm14Obj = pyhwm2014.HWM142D(
            alt=110.0,
            ap=[35, 35],
            glatlim=[-88.5, 88.5],
            glatstp=1.5,
            glonlim=[-180.0, 180.0],
            glonstp=3.0,
            option=6,
            verbose=False,
            ut=date.hour + date.minute / 60,
            day=date.timetuple().tm_yday,
        )
        u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
        u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing="ij")

        dynamics.set_u(
            u_theta=u_theta,
            u_phi=u_phi,
            lat=u_lat,
            lon=u_lon,
            weights=np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)),
            reg_lambda=u_lambda,
        )

    if steady_state:
        dynamics.impose_steady_state()

    dynamics.evolve_to_time(
        t=final_time, dt=dt, sampling_step_interval=1, saving_sample_interval=plotsteps
    )

    return dynamics
