def run_pynamit(final_time = 100, plotsteps = 200, dt = 5e-4, Nmax = 20, Mmax = 20, Ncs = 30, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PFAC = True, connect_hemispheres = False, latitude_boundary = 50, wind = False, steady_state = False, vector_jr = True, vector_conductance = True, vector_u = True):
    """ Run the pynamit simulation with the given parameters. """

    import datetime
    import numpy as np

    from lompe import conductance
    import dipole
    import pyamps
    import pyhwm2014 # https://github.com/rilma/pyHWM14

    from pynamit.simulation.dynamics import Dynamics
    from pynamit.math.constants import RE

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    dynamics = Dynamics(dataset_filename_prefix = None,
                        Nmax = Nmax,
                        Mmax = Mmax,
                        Ncs = Ncs,
                        RI = RI,
                        mainfield_kind = mainfield_kind,
                        ignore_PFAC = ignore_PFAC,
                        connect_hemispheres = connect_hemispheres,
                        latitude_boundary = latitude_boundary,
                        vector_jr = vector_jr,
                        vector_conductance = vector_conductance,
                        vector_u = vector_u)

    date = datetime.datetime(2001, 5, 12, 21, 45)

    ## CONDUCTANCE INPUT
    conductance_lat = dynamics.state_grid.lat
    conductance_lon = dynamics.state_grid.lon
    Kp = 5
    hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = True)
    dynamics.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

    ## jr INPUT
    jr_lat = dynamics.state_grid.lat
    jr_lon = dynamics.state_grid.lon
    d = dipole.Dipole(date.year)
    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    jr = a.get_upward_current(mlat = jr_lat, mlt = d.mlon2mlt(jr_lon, date)) * 1e-6
    jr[np.abs(jr_lat) < 50] = 0 # filter low latitude jr
    dynamics.set_jr(jr, lat = jr_lat, lon = jr_lon)

    ## WIND INPUT
    if wind:
        hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-88.5, 88.5], glatstp = 1.5,
                                     glonlim=[-180., 180.], glonstp = 3., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)
        u_theta, u_phi = (-hwm14Obj.Vwind.flatten(), hwm14Obj.Uwind.flatten())
        u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

        dynamics.set_u(u_theta = u_theta, u_phi = u_phi, lat = u_lat, lon = u_lon, weights = np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())),  (2, 1)))


    if steady_state:
        dynamics.set_jr(jr = jr, lat = jr_lat, lon = jr_lon)

        timeseries_keys = list(dynamics.timeseries.keys())
        if 'state' in timeseries_keys:
            timeseries_keys.remove('state')
        if timeseries_keys is not None:
            for key in timeseries_keys:
                dynamics.select_timeseries_data(key, interpolation = False)

        mv = dynamics.state.steady_state_m_ind()
        dynamics.state.set_coeffs(m_ind = mv)

        dynamics.state.impose_constraints()

    dynamics.evolve_to_time(t = final_time, dt = dt, sampling_step_interval = 1, saving_sample_interval = plotsteps)

    return dynamics