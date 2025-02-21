"""Default Run Script for PynaMIT.

This module provides the function run_pynamit() which sets up and runs a
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
):
    """Run a default PynaMIT simulation with the given parameters.

    Parameters
    ----------
    final_time : float, optional
        The final time of the simulation in seconds. Default is 100.
    plotsteps : int, optional
        The number of steps between each plot. Default is 200.
    dt : float, optional
        The time step for the simulation. Default is 5e-4.
    Nmax : int, optional
        The maximum degree of the spherical harmonics. Default is 20.
    Mmax : int, optional
        The maximum order of the spherical harmonics. Default is 20.
    Ncs : int, optional
        The number of grid points in the cubed sphere grid. Default is
        30.
    mainfield_kind : str, optional
        The type of main field model. Default is 'dipole'.
    fig_directory : str, optional
        The directory to save the figures. Default is './figs'.
    ignore_PFAC : bool, optional
        Whether to ignore the poloidal field-aligned currents. Default
        is True.
    connect_hemispheres : bool, optional
        Whether to connect the hemispheres. Default is False.
    latitude_boundary : float, optional
        The latitude boundary for the simulation. Default is 50.
    wind : bool, optional
        Whether to include wind in the simulation. Default is False.
    steady_state : bool, optional
        Whether to impose a steady state. Default is False.
    vector_jr : bool, optional
        Whether to use vector representation for radial current. Default
        is True.
    vector_conductance : bool, optional
        Whether to use vector representation for conductance. Default is
        True.
    vector_u : bool, optional
        Whether to use vector representation for wind. Default is True.

    Returns
    -------
    dynamics : Dynamics
        The dynamics object for performing the simulation and handling
        the simulation results.
    """
    import datetime
    import numpy as np

    from lompe import conductance
    import dipole
    import pyamps
    import pyhwm2014  # https://github.com/rilma/pyHWM14

    from pynamit.simulation.dynamics import Dynamics
    from pynamit.math.constants import RE

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.0e3
    dynamics = Dynamics(
        dataset_filename_prefix=None,
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
    )

    date = datetime.datetime(2001, 5, 12, 21, 45)

    # CONDUCTANCE INPUT
    conductance_lat = dynamics.state_grid.lat
    conductance_lon = dynamics.state_grid.lon
    Kp = 5
    hall, pedersen = conductance.hardy_EUV(
        conductance_lon, conductance_lat, Kp, date, starlight=1, dipole=True
    )
    dynamics.set_conductance(hall, pedersen, lat=conductance_lat, lon=conductance_lon)

    # jr INPUT
    jr_lat = dynamics.state_grid.lat
    jr_lon = dynamics.state_grid.lon
    d = dipole.Dipole(date.year)
    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50)
    jr = a.get_upward_current(mlat=jr_lat, mlt=d.mlon2mlt(jr_lon, date)) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0  # filter low latitude jr
    dynamics.set_jr(jr, lat=jr_lat, lon=jr_lon)

    # WIND INPUT
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
        )

    if steady_state:
        dynamics.impose_steady_state()

    dynamics.evolve_to_time(
        t=final_time, dt=dt, sampling_step_interval=1, saving_sample_interval=plotsteps
    )

    return dynamics
