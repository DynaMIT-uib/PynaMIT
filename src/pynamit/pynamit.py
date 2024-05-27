import numpy as np
import xarray as xr
from pynamit.mainfield import Mainfield
from pynamit.sha.sh_basis import SHBasis
import os
from pynamit.cubedsphere import cubedsphere
#from pynamit.cs_equations import CSEquations
from pynamit.grid import Grid
from pynamit.state import State
from pynamit.constants import RE
import pynamit

class I2D(object):
    """ 2D ionosphere. """

    def __init__(self, fn = 'tmp.ncdf',
                       sh = None, csp = None,
                       RI = RE + 110.e3, mainfield_kind = 'dipole',
                       B0_parameters = {'epoch':2020, 'B0':None},
                       FAC_integration_steps = np.logspace(np.log10(RE + 110.e3), np.log10(4 * RE), 11),
                       ignore_PFAC = False,
                       connect_hemispheres = False,
                       latitude_boundary = 50,
                       zero_jr_at_dip_equator = False,
                       ih_constraint_scaling = 1e-5,
                       PFAC_matrix = None):
        """

        Parameters
        ----------
        sh: sha.SHBasis object
            Spherical harmonic basis object.
        csp: cubedsphere.CSProjection object
            Cubed sphere projection object.
        RI: float, optional, default = RE + 110.e3
            Radius of the ionosphere in m.
        mainfield_kind: string, {'dipole', 'radial', 'igrf'}, default = 'dipole'
            Set to the main field model you want. For 'dipole' and
            'igrf', you can specify epoch via `B0_parameters`.
        FAC_integration_steps: array-like
            Use this to specify the radii used in the integral to calculate
            the poloidal field of FACs

        """
        self.filename = fn
        self.FAC_integration_steps  = FAC_integration_steps
        self.zero_jr_at_dip_equator = zero_jr_at_dip_equator
        self.connect_hemispheres    = connect_hemispheres
        self.ignore_PFAC            = ignore_PFAC
        self.latitude_boundary      = latitude_boundary
        self.ih_constraint_scaling  = ih_constraint_scaling
        self.RI                     = RI
        self.mainfield_kind         = mainfield_kind
        self.mainfield_epoch        = B0_parameters['epoch']
        self.mainfield_B0           = B0_parameters['B0']

        if os.path.exists(self.filename): # override input and load parameters from file:
            dataset = xr.load_dataset(self.filename) 

            self.FAC_integration_steps  = dataset.FAC_integration_steps
            self.zero_jr_at_dip_equator = dataset.zero_jr_at_dip_equator
            self.connect_hemispheres    = dataset.connect_hemispheres
            self.ignore_PFAC            = dataset.ignore_PFAC
            self.latitude_boundary      = dataset.latitude_boundary
            self.ih_constraint_scaling  = dataset.ih_constraint_scaling
            self.RI                     = dataset.RI
            self.mainfield_kind         = dataset.mainfield_kind
            self.mainfield_epoch        = dataset.mainfield_epoch
            self.mainfield_B0           = dataset.mainfield_B0

            shape = (dataset.i.size, dataset.i.size)
            PFAC_matrix                 = dataset.PFAC_matrix.reshape( shape )

            sh  = pynamit.SHBasis(dataset.N, dataset.M)
            csp = pynamit.CSProjection(dataset.Ncs)

            B0_parameters = {'epoch':self.mainfield_epoch, 'B0':self.mainfield_B0}
            self.latest_time = dataset.time.values[-1]
        else:
            self.latest_time = 0


        B0_parameters['hI'] = (self.RI - RE) * 1e-3 # add ionosphere height in km
        mainfield = Mainfield(kind = self.mainfield_kind, **B0_parameters)
        num_grid = Grid(self.RI, 90 - csp.arr_theta, csp.arr_phi, mainfield)


        # Initialize the state of the ionosphere
        self.state = State(sh, mainfield, num_grid, 
                           RI = self.RI, 
                           ignore_PFAC = self.ignore_PFAC, 
                           FAC_integration_steps = self.FAC_integration_steps, 
                           connect_hemispheres = self.connect_hemispheres, 
                           latitude_boundary = self.latitude_boundary, 
                           zero_jr_at_dip_equator = self.zero_jr_at_dip_equator, 
                           ih_constraint_scaling = self.ih_constraint_scaling,
                           PFAC_matrix = PFAC_matrix
                           )


    def save_state(self, time):
        """ save state to file """

        if time == 0: # the file will be initialized

            # resolution parameters:
            resolution_params = {}
            resolution_params['Ncs'] = int(np.sqrt(self.state.num_grid.size / 6) + 1)
            resolution_params['N']   = self.state.sh.Nmax
            resolution_params['M']   = self.state.sh.Mmax
            resolution_params['FAC_integration_steps'] = self.FAC_integration_steps

            # model settings:
            model_settings = {}
            model_settings['RI']                     = self.RI
            model_settings['ih_constraint_scaling']  = self.ih_constraint_scaling
            model_settings['latitude_boundary']      = self.latitude_boundary
            model_settings['zero_jr_at_dip_equator'] = int(self.zero_jr_at_dip_equator)
            model_settings['connect_hemispheres']    = int(self.connect_hemispheres   )
            model_settings['ignore_PFAC']            = int(self.ignore_PFAC           )
            model_settings['mainfield_kind']         = self.mainfield_kind
            model_settings['mainfield_epoch']        = self.mainfield_epoch
            model_settings['mainfield_B0']           = 0 if self.mainfield_B0 is None else self.mainfield_B0

            PFAC_matrix = self.state.TB_to_VB_PFAC
            size = self.state.sh.num_coeffs

            dataset = xr.Dataset()

            dataset['SH_coefficients_imposed'] = xr.DataArray(self.state.TB.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            dataset['SH_coefficients_induced'] = xr.DataArray(self.state.VB.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])

            dataset.attrs.update(resolution_params)
            dataset.attrs.update(model_settings)
            dataset.attrs.update({'PFAC_matrix':PFAC_matrix.flatten()})
            dataset['n'] = xr.DataArray(self.state.sh.n, coords = {'i': range(size)}, dims = ['i'], name = 'n')
            dataset['m'] = xr.DataArray(self.state.sh.m, coords = {'i': range(size)}, dims = ['i'], name = 'm')

            dataset.to_netcdf(self.filename)
            print('Created {}'.format(self.filename))


        else: # adding new coefficients to existing file:

            dataset = xr.load_dataset(self.filename)

            size = self.state.sh.num_coeffs
            imposed_coeffs = xr.DataArray(self.state.TB.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            induced_coeffs = xr.DataArray(self.state.VB.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])

            # make a copy of the dataset but with new coefficients:            
            new_dataset = dataset.copy(deep = True).drop(["SH_coefficients_imposed", "SH_coefficients_induced", 'time'])
            new_dataset['SH_coefficients_induced'] = induced_coeffs
            new_dataset['SH_coefficients_imposed'] = imposed_coeffs

            # merge the copy with the old_
            dataset = xr.concat([dataset, new_dataset], dim = 'time', data_vars = 'minimal', combine_attrs = 'identical')

            dataset.to_netcdf(self.filename)



    def evolve_to_time(self, t, dt = 5e-4, save_steps = 200, quiet = False):
        """ Evolve to given time

        """

        if self.latest_time == 0:
            self.save_state(0) # initialize save file

        time = self.latest_time
        count = 0
        while self.latest_time <= t:

            self.state.evolve_Br(dt)

            time  += dt
            count += 1

            if count % save_steps == 0:
                self.save_state(time)
                self.latest_time = time
                if quiet:
                    pass
                else:
                    print('Saved output at t = {:.2f} s'.format(time), end = '\r')








def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 45, Mmax = 3, Ncs = 60, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PFAC = True, connect_hemispheres = False, latitude_boundary = 50, zero_jr_at_dip_equator = False, wind_directory = None):

    # Set up the spherical harmonic basis object
    i2d_sh = SHBasis(Nmax, Mmax)

    # Define CS grid used for SH analysis and gradient calculations
    # Each cube block with have ``(Ncs-1)*(Ncs-1)`` cells.
    csp = cubedsphere.CSProjection(Ncs) # cubed sphere projection object

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    i2d = I2D(sh = i2d_sh, csp = csp, RI = RI, mainfield_kind = mainfield_kind, ignore_PFAC = ignore_PFAC, connect_hemispheres = connect_hemispheres, latitude_boundary = latitude_boundary, zero_jr_at_dip_equator = zero_jr_at_dip_equator)

    import pyamps
    from pynamit.visualization import globalplot, cs_interpolate
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime
    import polplot
    from pynamit.basis_evaluator import BasisEvaluator
    #import pyhwm2014 # https://github.com/rilma/pyHWM14

    compare_AMPS_FAC_and_CF_currents = False # set to True for debugging
    SIMULATE = True
    show_FAC_and_conductance = False
    make_colorbars = False
    plot_AMPS_Br = False

    WIND_FACTOR = 1

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

    # Define cubed sphere grid
    csp_grid = Grid(RI, 90 - csp.arr_theta, csp.arr_phi)
    csp_i2d_evaluator = BasisEvaluator(i2d.state.basis, csp_grid)

    # Define grid used for plotting
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape
    plt_grid = Grid(RI, lat, lon)
    plt_i2d_evaluator = BasisEvaluator(i2d.state.basis, plt_grid)

    hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
    i2d.state.set_conductance(hall, pedersen, csp_i2d_evaluator)

    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    ju = a.get_upward_current(mlat = csp_grid.lat, mlt = d.mlon2mlt(csp_grid.lon, date)) * 1e-6
    ju[np.abs(csp_grid.lat) < 50] = 0 # filter low latitude FACs

    ju[csp_grid.theta < 90] = -ju[csp_grid.theta < 90] # we need the current to refer to magnetic field direction, so changing sign in the north since the field there points down 

    if (wind_directory is not None) and os.path.exists(wind_directory):
        #hwm14Obj = pyhwm2014.HWM142D(alt=110., ap=[35, 35], glatlim=[-89., 88.], glatstp = 3.,
        #                             glonlim=[-180., 180.], glonstp = 8., option = 6, verbose = False, ut = date.hour + date.minute/60, day = date.timetuple().tm_yday)
        #u_phi   =  hwm14Obj.Uwind
        #u_theta = -hwm14Obj.Vwind
        #u_lat, u_lon = np.meshgrid(hwm14Obj.glatbins, hwm14Obj.glonbins, indexing = 'ij')

        u_lat, u_lon, u_phi, u_theta = np.load(os.path.join(wind_directory, 'ulat.npy')), np.load(os.path.join(wind_directory, 'ulon.npy')), np.load(os.path.join(wind_directory, 'uphi.npy')), np.load(os.path.join(wind_directory, 'utheta.npy'))
        u_lat, u_lon = np.meshgrid(u_lat, u_lon, indexing = 'ij')

        u_int = csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, csp.arr_theta, csp.arr_phi)
        print("u_int norm", np.linalg.norm(u_int))
        exit()
        u_east_int, u_north_int, u_r_int = u_int
        i2d.state.set_u(-u_north_int * WIND_FACTOR, u_east_int * WIND_FACTOR)

    i2d.state.set_FAC(ju, csp_i2d_evaluator)

    if compare_AMPS_FAC_and_CF_currents:
        # compare FACs and curl-free currents:
        _, axes = plt.subplots(ncols = 2, nrows = 2)
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

        mn_grid = Grid(RI, mlatn, mltn)
        mnv_grid = Grid(RI, mlatnv, mltnv)

        paxes[0].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[0], levels = levels, cmap = plt.cm.bwr)
        paxes[0].quiver(  mnv_grid.lat, mnv_grid.lon,  np.split(jn_amps, 2)[0], np.split(je_amps, 2)[0], scale = SCALE, color = 'black')
        paxes[1].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[1], levels = levels, cmap = plt.cm.bwr)
        paxes[1].quiver(  mnv_grid.lat, mnv_grid.lon, -np.split(jn_amps, 2)[1], np.split(je_amps, 2)[1], scale = SCALE, color = 'black')

        m_i2d_evaluator = BasisEvaluator(i2d.state.basis, Grid(RI, mlat, lon))
        jr = i2d.get_Jr(m_i2d_evaluator) * 1e6

        mv_i2d_evaluator = BasisEvaluator(i2d.state.basis, Grid(RI, mlatv, lonv))
        js, je = i2d.state.get_JS(mv_i2d_evaluator) * 1e3
        jn = -js

        jrn, jrs = np.split(jr, 2) 
        paxes[2].contourf(mn_grid.lat,  mn_grid.lon,   jrn, levels = levels, cmap = plt.cm.bwr)
        paxes[2].quiver(  mnv_grid.lat, mnv_grid.lon,  np.split(jn, 2)[0], np.split(je, 2)[0], scale = SCALE, color = 'black')
        paxes[3].contourf(mn_grid.lat,  mn_grid.lon,   jrs, levels = levels, cmap = plt.cm.bwr)
        paxes[3].quiver(  mnv_grid.lat, mnv_grid.lon,  -np.split(jn, 2)[1], np.split(je, 2)[1], scale = SCALE, color = 'black')

        plt.show()
        plt.close()

        jr = i2d.get_Jr(plt_i2d_evaluator)

        globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), jr.reshape(pltshape) * 1e6, noon_longitude = lon0, cmap = plt.cm.bwr, levels = levels)


    if plot_AMPS_Br:

        _, axes = plt.subplots(ncols = 2, figsize = (10, 5))
        paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]

        if not compare_AMPS_FAC_and_CF_currents:
            mlat  , mlt   = a.scalargrid
            mlatn , mltn  = np.split(mlat , 2)[0], np.split(mlt , 2)[0]
            mn_grid = Grid(RI, mlatn, mltn)

        Bu = a.get_ground_Buqd(height = a.height)
        paxes[0].contourf(mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[0], levels = Blevels * 1e9, cmap = plt.cm.bwr)
        paxes[1].contourf(mn_grid.lat, mn_grid.lon, np.split(Bu, 2)[1], levels = Blevels * 1e9, cmap = plt.cm.bwr)

        plt.show()
        plt.close()


    if show_FAC_and_conductance:

        hall_plt = cs_interpolate(csp, csp_grid.lat, csp_grid.lon, hall, plt_grid.lat, plt_grid.lon)
        pede_plt = cs_interpolate(csp, csp_grid.lat, csp_grid.lon, pedersen, plt_grid.lat, plt_grid.lon)

        globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), hall_plt.reshape(pltshape), noon_longitude = lon0, levels = c_levels, save = 'hall.png')
        globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), pede_plt.reshape(pltshape), noon_longitude = lon0, levels = c_levels, save = 'pede.png')

        jr = i2d.state.get_Jr(plt_i2d_evaluator)
        globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), jr.reshape(pltshape), noon_longitude = lon0, levels = levels * 1e-6, save = 'jr.png', cmap = plt.cm.bwr)

    if make_colorbars:
        # conductance:
        _, axc = plt.subplots(figsize = (1, 10))
        cz, co = np.zeros_like(c_levels), np.ones_like(c_levels)
        axc.contourf(np.vstack((cz, co)).T, np.vstack((c_levels, c_levels)).T, np.vstack((c_levels, c_levels)).T, levels = c_levels)
        axc.set_ylabel('mho', size = 16)
        axc.set_xticks([])
        plt.subplots_adjust(left = .7)
        plt.savefig('conductance_colorbar.png')
        plt.close()

        # FAC and Br:
        _, axf = plt.subplots(figsize = (2, 10))
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
        plt.close()



    print('bug in cartopy makes it impossible to not center levels at zero... replace when cartopy has been improved')
    #globalplot(plt_grid.lon, plt_grid.lat, jr.reshape(plt_grid.lat.shape), 
    #           levels = levels, cmap = 'bwr', central_longitude = lon0)

    #globalplot(csp_grid.lon, csp_grid.lat, i2d.SH, vmin = 0, vmax = 20, cmap = 'viridis', scatter = True, central_longitude = lon0)

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
            coeffs.append(i2d.state.VB.coeffs)
            count += 1
            #print(count, time, (i2d.state.VB.coeffs * i2d.state.VB_to_Br)[:3])

            if (count % plotsteps == 0) and fig_directory_writeable:
                print(count, time, (i2d.state.VB.coeffs * i2d.state.VB_to_Br)[:3])
                fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
                filecount +=1
                title = 't = {:.3} s'.format(time)
                Br = i2d.state.get_Br(plt_i2d_evaluator)
                _, paxn, paxs, _ =  globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), Br.reshape(pltshape) , title = title, returnplot = True,
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
                #W = i2d.state.get_W(plt_i2d_evaluator) * 1e-3

                i2d.state.update_Phi()
                Phi = i2d.state.get_Phi(plt_i2d_evaluator) * 1e-3


                nnn = plt_grid.lat.flatten() >  50
                sss = plt_grid.lat.flatten() < -50
                #paxn.contour(plt_grid.lat.flatten()[nnn], (plt_grid.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
                #paxs.contour(plt_grid.lat.flatten()[sss], (plt_grid.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
                paxn.contour(plt_grid.lat[nnn], (plt_grid.lon - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
                paxs.contour(plt_grid.lat[sss], (plt_grid.lon - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
                plt.savefig(fn)
                plt.close()

            if count > totalsteps:
                break
    return coeffs