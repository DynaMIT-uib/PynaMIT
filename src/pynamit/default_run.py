def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 20, Mmax = 20, Ncs = 30, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PFAC = True, connect_hemispheres = False, latitude_boundary = 50, zero_jr_at_dip_equator = False, wind_directory = None, vector_FAC = True, vector_conductance = True, vector_u = True):
    """ Run the pynamit simulation with the given parameters. """

    import os
    import datetime
    import numpy as np

    from lompe import conductance
    import dipole
    import pyamps
    #import pyhwm2014 # https://github.com/rilma/pyHWM14

    from pynamit.cubed_sphere import cubed_sphere
    from pynamit.simulation.ionosphere_evolution import I2D
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.field_evaluator import FieldEvaluator
    from pynamit.various.constants import RE

    WIND_FACTOR = 1

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    i2d = I2D(result_filename_prefix = None,
              Nmax = Nmax,
              Mmax = Mmax,
              Ncs = Ncs,
              RI = RI,
              mainfield_kind = mainfield_kind,
              ignore_PFAC = ignore_PFAC,
              connect_hemispheres = connect_hemispheres,
              latitude_boundary = latitude_boundary,
              zero_jr_at_dip_equator = zero_jr_at_dip_equator,
              vector_FAC = vector_FAC,
              vector_conductance = vector_conductance,
              vector_u = vector_u)

    date = datetime.datetime(2001, 5, 12, 21, 45)

    # Define CS grid used for conductance and FAC input
    csp = cubed_sphere.CSProjection(Ncs) # cubed sphere projection object

    ## CONDUCTANCE INPUT
    conductance_lat = 90 - csp.arr_theta
    conductance_lon = csp.arr_phi
    Kp = 5
    hall, pedersen = conductance.hardy_EUV(conductance_lon, conductance_lat, Kp, date, starlight = 1, dipole = True)
    i2d.set_conductance(hall, pedersen, lat = conductance_lat, lon = conductance_lon)

    ## FAC INPUT
    FAC_lat = 90 - csp.arr_theta
    FAC_lon = csp.arr_phi
    d = dipole.Dipole(date.year)
    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    csp_b_evaluator = FieldEvaluator(i2d.state.mainfield, Grid(lat = FAC_lat, lon = FAC_lon), RI)
    jparallel = a.get_upward_current(mlat = FAC_lat, mlt = d.mlon2mlt(FAC_lon, date)) / csp_b_evaluator.br * 1e-6
    jparallel[np.abs(FAC_lat) < 50] = 0 # filter low latitude FACs
    i2d.set_FAC(jparallel, lat = conductance_lat, lon = conductance_lon)

    ## WIND INPUT
    if (wind_directory is not None) and os.path.exists(wind_directory):
        #hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3.,
        #                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)
        #u_phi   =  hwm14Obj.Uwind
        #u_theta = -hwm14Obj.Vwind
        #u = (-hwm14Obj.Vwind.flatten() * WIND_FACTOR, hwm14Obj.Uwind.flatten() * WIND_FACTOR)
        #u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

        u_lat, u_lon, u_phi, u_theta = np.load(os.path.join(wind_directory, 'ulat.npy')), np.load(os.path.join(wind_directory, 'ulon.npy')), np.load(os.path.join(wind_directory, 'uphi.npy')), np.load(os.path.join(wind_directory, 'utheta.npy'))
        u = (u_theta.flatten() * WIND_FACTOR, u_phi.flatten() * WIND_FACTOR)
        u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')

        i2d.set_u(u, lat = u_lat, lon = u_lon)

    i2d.evolve_to_time(t = (totalsteps + 2) * dt, dt = dt, history_update_interval = 1, history_save_interval = plotsteps)

    return i2d