from pynamit.cubed_sphere import cubed_sphere
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.simulation.ionosphere_evolution import I2D
from pynamit.primitives.grid import Grid
from pynamit.various.constants import RE
import numpy as np
import os

def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 20, Mmax = 20, Ncs = 30, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PFAC = True, connect_hemispheres = False, latitude_boundary = 50, zero_jr_at_dip_equator = False, wind_directory = None, vector_FAC = True, vector_conductance = True, vector_u = True):

    # Set up the spherical harmonic basis object
    i2d_sh = SHBasis(Nmax, Mmax)

    # Define CS grid used for SH analysis and gradient calculations
    # Each cube block with have ``(Ncs-1)*(Ncs-1)`` cells.
    csp = cubed_sphere.CSProjection(Ncs) # cubed sphere projection object

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    i2d = I2D(result_filename_prefix = None, sh = i2d_sh, csp = csp, RI = RI, mainfield_kind = mainfield_kind, ignore_PFAC = ignore_PFAC, connect_hemispheres = connect_hemispheres, latitude_boundary = latitude_boundary, zero_jr_at_dip_equator = zero_jr_at_dip_equator, vector_FAC = vector_FAC, vector_conductance = vector_conductance, vector_u = vector_u)

    import pyamps
    from pynamit.simulation.visualization import globalplot, cs_interpolate
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime
    import polplot
    from pynamit.primitives.basis_evaluator import BasisEvaluator
    from pynamit.primitives.field_evaluator import FieldEvaluator
    from pynamit.simulation.visualization import time_dependent_plot
    #import pyhwm2014 # https://github.com/rilma/pyHWM14

    SIMULATE = True
    WIND_FACTOR = 1

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp   = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    # Define cubed sphere grid
    csp_grid = Grid(90 - csp.arr_theta, csp.arr_phi)
    csp_b_evaluator = FieldEvaluator(i2d.state.mainfield, csp_grid, RI)

    hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
    i2d.set_conductance(hall, pedersen, csp_grid)

    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    jparallel = a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) / csp_b_evaluator.br * 1e-6
    jparallel[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

    if (wind_directory is not None) and os.path.exists(wind_directory):
        #hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3.,
        #                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)
        #u_phi   =  hwm14Obj.Uwind
        #u_theta = -hwm14Obj.Vwind
        #u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

        u_lat, u_lon, u_phi, u_theta = np.load(os.path.join(wind_directory, 'ulat.npy')), np.load(os.path.join(wind_directory, 'ulon.npy')), np.load(os.path.join(wind_directory, 'uphi.npy')), np.load(os.path.join(wind_directory, 'utheta.npy'))
        u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')
        u_grid = Grid(u_lat, u_lon)

        i2d.set_u(u_theta.flatten() * WIND_FACTOR, u_phi.flatten() * WIND_FACTOR, u_grid)

    i2d.set_FAC(jparallel, csp_grid)

    fig_directory_writeable = os.access(fig_directory, os.W_OK)

    if not fig_directory_writeable:
        print('Figure directory {} is not writeable, proceeding without figure generation. For figures, rerun after ensuring that the directory exists and is writeable.'.format(fig_directory))
    # Define grid used for plotting
    Ncs = 30
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape
    plt_grid = Grid(lat, lon)
    plt_i2d_evaluator = BasisEvaluator(i2d.basis, plt_grid)

    if SIMULATE:

        coeffs = []
        count = 0
        filecount = 1
        while True:

            i2d.state.evolve_Br(dt)
            i2d.latest_time += i2d.latest_time + dt
            coeffs.append(i2d.state.m_ind.coeffs)
            count += 1

            if (count % plotsteps == 0) and fig_directory_writeable:
                print(count, i2d.latest_time, (i2d.state.m_ind.coeffs * i2d.state.m_ind_to_Br)[:3])
                time_dependent_plot(i2d, fig_directory, filecount, lon0, plt_grid, pltshape, plt_i2d_evaluator)
                filecount += 1


            if count > totalsteps:
                break
    return coeffs