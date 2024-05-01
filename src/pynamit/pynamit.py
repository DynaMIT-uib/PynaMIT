import numpy as np
from pynamit.mainfield import Mainfield
from pynamit.sha.sha import sha
import os
from pynamit.cubedsphere import cubedsphere
#from pynamit.cs_equations import cs_equations
from pynamit.grid import grid
from pynamit.state import state

RE = 6371.2e3
mu0 = 4 * np.pi * 1e-7


class I2D(object):
    """ 2D ionosphere. """

    def __init__(self, sha, csp,
                       RI = RE + 110.e3, mainfield_kind = 'dipole', 
                       B0_parameters = {'epoch':2020}, 
                       FAC_integration_parameters = {'steps':np.logspace(np.log10(RE + 110.e3), np.log10(4 * RE), 11)},
                       ignore_PNAF = False,
                       connect_hemispheres = False,
                       latitude_boundary = 50):
        """

        Parameters
        ----------
        sha: sha object
            Spherical harmonic analysis object.
        csp: cubedsphere.CSprojection object
            Cubed sphere projection object.
        RI: float, optional, default = RE + 110.e3
            Radius of the ionosphere in m.
        mainfield_kind: string, {'dipole', 'radial', 'igrf'}, default = 'dipole'
            Set to the main field model you want. For 'dipole' and
            'igrf', you can specify epoch via `B0_parameters`.
        FAC_integration_parameters: dict
            Use this to specify parameters in the integration required to
            find the poloidal part of the magnetic field of FACs. Not
            relevant for radial main field.

        """
        self.latitude_boundary = latitude_boundary

        self.sha = sha

        self.mainfield = Mainfield(kind = mainfield_kind, **B0_parameters)

        self.num_grid = grid(RI, 90 - csp.arr_theta, csp.arr_phi)

        # Construct matrices for transforming from surface spherical harmonic coefficients to cubed sphere grid
        self.num_grid.construct_G(sha)
        self.num_grid.construct_dG(sha)
        self.num_grid.construct_GTG()
        self.num_grid.construct_vector_to_shc_cf_df()

        #self.cs_equations = cs_equations(csp, RI)

        # Initialize the state of the ionosphere
        self.state = state(sha, self.mainfield, self.num_grid, RI, ignore_PNAF, FAC_integration_parameters, connect_hemispheres, latitude_boundary)




def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 45, Mmax = 3, Ncs = 60, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PNAF = True):

    # Set up the spherical harmonic analysis object
    i2d_sha = sha(Nmax, Mmax)

    # Define CS grid used for SH analysis and gradient calculations
    # Each cube block with have ``(Ncs-1)*(Ncs-1)`` cells.
    csp = cubedsphere.CSprojection(Ncs) # cubed sphere projection object

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    i2d = I2D(i2d_sha, csp, RI, mainfield_kind, ignore_PNAF = ignore_PNAF)

    import pyamps
    from pynamit.visualization import globalplot, cs_interpolate
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime
    import polplot

    compare_AMPS_FAC_and_CF_currents = False # set to True for debugging
    SIMULATE = True
    show_FAC_and_conductance = False
    make_colorbars = False
    plot_AMPS_Br = False

    Blevels = np.linspace(-300, 300, 22) * 1e-9 # color levels for Br
    levels = np.linspace(-.9, .9, 22) # color levels for FAC muA/m^2
    c_levels = np.linspace(0, 20, 100) # color levels for conductance
    #Wlevels = np.r_[-512.5:512.5:5]
    Philevels = np.r_[-212.5:212.5:5]

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp   = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    # Define grid used for plotting
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    plt_grid = grid(RI, lat, lon)

    # Construct matrix for transforming from surface spherical harmonic coefficients to plotting grid
    plt_grid.construct_G(i2d_sha)

    hall, pedersen = conductance.hardy_EUV(i2d.num_grid.lon, i2d.num_grid.lat, Kp, date, starlight = 1, dipole = True)
    i2d.state.set_conductance(hall, pedersen)

    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    ju = a.get_upward_current(mlat = i2d.num_grid.lat, mlt = d.mlon2mlt(i2d.num_grid.lon, date)) * 1e-6
    ju[np.abs(i2d.num_grid.lat) < 50] = 0 # filter low latitude FACs

    ju[i2d.num_grid.theta < 90] = -ju[i2d.num_grid.theta < 90] # we need the current to refer to magnetic field direction, so changing sign in the north since the field there points down 

    i2d.state.set_FAC(ju)

    if compare_AMPS_FAC_and_CF_currents:
        # compare FACs and curl-free currents:
        fig, axes = plt.subplots(ncols = 2, nrows = 2)
        SCALE = 1e3


        paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]

        ju_amps = a.get_upward_current()
        je_amps, jn_amps = a.get_curl_free_current()

        mlat  , mlt   = a.scalargrid
        mlatv , mltv  = a.vectorgrid
        mlatn , mltn  = np.split(mlat , 2)[0], np.split(mlt , 2)[0]
        mlatnv, mltnv = np.split(mlatv, 2)[0], np.split(mltv, 2)[0]

        lon  = d.mlt2mlon(mlt , date)
        lonv = d.mlt2mlon(mltv, date)

        m_grid = grid(i2d.RI, mlat, lon)
        mv_grid = grid(i2d.RI, mlatv, lonv)
        mn_grid = grid(i2d.RI, mlatn, mltn)
        mnv_grid = grid(i2d.RI, mlatnv, mltnv)

        paxes[0].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[0], levels = levels, cmap = plt.cm.bwr)
        paxes[0].quiver(  mnv_grid.lat, mnv_grid.lon,  np.split(jn_amps, 2)[0], np.split(je_amps, 2)[0], scale = SCALE, color = 'black')
        paxes[1].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[1], levels = levels, cmap = plt.cm.bwr)
        paxes[1].quiver(  mnv_grid.lat, mnv_grid.lon, -np.split(jn_amps, 2)[1], np.split(je_amps, 2)[1], scale = SCALE, color = 'black')

        G   = i2d_sha.get_G(m_grid) * 1e6
        Gph = i2d_sha.get_G(mv_grid, derivative = 'phi'  ) * 1e3
        Gth = i2d_sha.get_G(mv_grid, derivative = 'theta') * 1e3
        jr = G.dot(i2d.state.shc_TJr)

        je = -Gph.dot(i2d.state.shc_TJ)
        jn =  Gth.dot(i2d.state.shc_TJ)

        jrn, jrs = np.split(jr, 2) 
        paxes[2].contourf(mn_grid.lat,  mn_grid.lon,   jrn, levels = levels, cmap = plt.cm.bwr)
        paxes[2].quiver(  mnv_grid.lat, mnv_grid.lon,  np.split(jn, 2)[0], np.split(je, 2)[0], scale = SCALE, color = 'black')
        paxes[3].contourf(mn_grid.lat,  mn_grid.lon,   jrs, levels = levels, cmap = plt.cm.bwr)
        paxes[3].quiver(  mnv_grid.lat, mnv_grid.lon,  -np.split(jn, 2)[1], np.split(je, 2)[1], scale = SCALE, color = 'black')

        jr = i2d.get_Jr(plt_grid)

        globalplot(plt_grid.lon, plt_grid.lat, jr.reshape(plt_grid.lon.shape) * 1e6, noon_longitude = lon0, cmap = plt.cm.bwr, levels = levels)

        plt.show()


    if plot_AMPS_Br:

        fig, axes = plt.subplots(ncols = 2, figsize = (10, 5))
        paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]

        if not compare_AMPS_FAC_and_CF_currents:
            mlat  , mlt   = a.scalargrid
            mlatn , mltn  = np.split(mlat , 2)[0], np.split(mlt , 2)[0]
            mn_grid = grid(i2d.RI, mlatn, mltn)

        Bu = a.get_ground_Buqd(height = a.height)
        paxes[0].contourf(mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[0], levels = Blevels * 1e9, cmap = plt.cm.bwr)
        paxes[1].contourf(mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[1], levels = Blevels * 1e9, cmap = plt.cm.bwr)

        plt.show()


    if show_FAC_and_conductance:

        hall_plt = cs_interpolate(csp, i2d.num_grid.lat, i2d.num_grid.lon, hall, plt_grid.lat, plt_grid.lon)
        pede_plt = cs_interpolate(csp, i2d.num_grid.lat, i2d.num_grid.lon, pedersen, plt_grid.lat, plt_grid.lon)

        globalplot(plt_grid.lon, plt_grid.lat, hall_plt, noon_longitude = lon0, levels = c_levels, save = 'hall.png')
        globalplot(plt_grid.lon, plt_grid.lat, pede_plt, noon_longitude = lon0, levels = c_levels, save = 'pede.png')

        jr = i2d.state.get_Jr(plt_grid)
        globalplot(plt_grid.lon, plt_grid.lat, jr.reshape(plt_grid.lon.shape), noon_longitude = lon0, levels = levels * 1e-6, save = 'jr.png', cmap = plt.cm.bwr)

    if make_colorbars:
        # conductance:
        fig, axc = plt.subplots(figsize = (1, 10))
        cz, co = np.zeros_like(c_levels), np.ones_like(c_levels)
        axc.contourf(np.vstack((cz, co)).T, np.vstack((c_levels, c_levels)).T, np.vstack((c_levels, c_levels)).T, levels = c_levels)
        axc.set_ylabel('mho', size = 16)
        axc.set_xticks([])
        plt.subplots_adjust(left = .7)
        plt.savefig('conductance_colorbar.png')

        # FAC and Br:
        fig, axf = plt.subplots(figsize = (2, 10))
        fz, fo = np.zeros_like(levels), np.ones_like(levels)
        axf.contourf(np.vstack((fz, fo)).T, np.vstack((levels, levels)).T, np.vstack((levels, levels)).T, levels = levels, cmap = plt.cm.bwr)
        axf.set_ylabel(r'$\mu$A/m$^2$', size = 16)
        axf.set_xticks([])

        axB = axf.twinx()
        Bz, Bo = np.zeros_like(Blevels), np.ones_like(Blevels)
        axB.contourf(np.vstack((Bz, Bo)).T, np.vstack((Blevels, Blevels)).T * 1e9, np.vstack((Blevels, Blevels)).T, levels = Blevels, cmap = plt.cm.bwr)
        axB.set_ylabel(r'nT', size = 16)
        axB.set_xticks([])

        plt.subplots_adjust(left = .45, right = .6)
        plt.savefig('mag_colorbar.png')



    print('bug in cartopy makes it impossible to not center levels at zero... replace when cartopy has been improved')
    #globalplot(plt_grid.lon, plt_grid.lat, jr.reshape(plt_grid.lat.shape), 
    #           levels = levels, cmap = 'bwr', central_longitude = lon0)

    #globalplot(i2d.num_grid.lon, i2d.num_grid.lat, i2d.SH, vmin = 0, vmax = 20, cmap = 'viridis', scatter = True, central_longitude = lon0)

    fig_directory_writeable = os.access(fig_directory, os.W_OK)

    if not fig_directory_writeable:
        print('Figure directory {} is not writeable, proceeding without figure generation. For figures, rerun after ensuring that the directory exists and is writeable.'.format(fig_directory))

    if SIMULATE:

        coeffs = []
        count = 0
        filecount = 1
        time = 0
        while True:

            i2d.state.evolve_Br(dt)
            time = time + dt
            coeffs.append(i2d.state.shc_VB)
            count += 1
            #print(count, time, i2d.state.shc_Br[:3])

            if (count % plotsteps == 0) and fig_directory_writeable:
                print(count, time, i2d.state.shc_Br[:3])
                fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
                filecount +=1
                title = 't = {:.3} s'.format(time)
                Br = i2d.state.get_Br(plt_grid)
                fig, paxn, paxs, axg =  globalplot(plt_grid.lon, plt_grid.lat, Br.reshape(plt_grid.lat.shape) , title = title, returnplot = True, 
                                                   levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
                #W = i2d.get_W(plt_grid)

                i2d.state.update_shc_Phi(i2d.num_grid)
                Phi = i2d.state.get_Phi(plt_grid)


                nnn = plt_grid.lat.flatten() >  50
                sss = plt_grid.lat.flatten() < -50
                #paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
                #paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
                paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
                paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
                plt.savefig(fn)

            if count > totalsteps:
                break
    return coeffs