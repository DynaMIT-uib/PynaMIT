import numpy as np
import xarray as xr
from pynamit.mainfield import Mainfield
from pynamit.sha.sh_basis import SHBasis
import os
from pynamit.cubedsphere import cubedsphere
#from pynamit.cs_equations import CSEquations
from pynamit.primitives.grid import Grid
from pynamit.state import State
from pynamit.constants import RE
import pynamit
import scipy.sparse as sp

class I2D(object):
    """ 2D ionosphere. """

    def __init__(self, result_filename = 'tmp.ncdf',
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
        self.result_filename        = result_filename
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
        self.csp                    = csp

        if (self.result_filename is not None) and os.path.exists(self.result_filename): # override input and load parameters from file:
            dataset = xr.load_dataset(self.result_filename)

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
            self.csp = pynamit.CSProjection(dataset.Ncs)

            B0_parameters = {'epoch':self.mainfield_epoch, 'B0':self.mainfield_B0}
            self.latest_time = dataset.time.values[-1]
        else:
            self.latest_time = np.float64(0)
            if (self.result_filename is None):
                self.result_filename = 'tmp.ncdf'


        B0_parameters['hI'] = (self.RI - RE) * 1e-3 # add ionosphere height in km
        mainfield = Mainfield(kind = self.mainfield_kind, **B0_parameters)
        self.num_grid = Grid(90 - self.csp.arr_theta, self.csp.arr_phi)


        # Initialize the state of the ionosphere
        self.state = State(sh, mainfield, self.num_grid, 
                           RI = self.RI, 
                           ignore_PFAC = self.ignore_PFAC, 
                           FAC_integration_steps = self.FAC_integration_steps, 
                           connect_hemispheres = self.connect_hemispheres, 
                           latitude_boundary = self.latitude_boundary, 
                           zero_jr_at_dip_equator = self.zero_jr_at_dip_equator, 
                           ih_constraint_scaling = self.ih_constraint_scaling,
                           PFAC_matrix = PFAC_matrix
                           )


        self.updated_FAC = False
        self.updated_u = False
        self.updated_conductance = False


    def save_state(self, time):
        """ save state to file """

        time = np.float64(time)
        self.state.update_Phi_and_EW()

        if self.updated_FAC:
            # Add FAC file writing here
            self.updated_FAC = False

        if self.updated_u:
            # Add u file writing here
            self.updated_u = False

        if self.updated_conductance:
            # Add conductance file writing here
            self.updated_conductance = False

        if (time == 0) and (not os.path.exists(self.result_filename)): # the file will be initialized

            # resolution parameters:
            resolution_params = {}
            resolution_params['Ncs'] = int(np.sqrt(self.state.num_grid.size / 6))
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

            PFAC_matrix = self.state.m_imp_to_B_pol
            size = self.state.sh.num_coeffs

            dataset = xr.Dataset()

            dataset['SH_coefficients_imposed'] = xr.DataArray(self.state.m_imp.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            dataset['SH_coefficients_induced'] = xr.DataArray(self.state.m_ind.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            dataset['SH_Phi']                  = xr.DataArray(self.state.Phi.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            dataset['SH_W'  ]                  = xr.DataArray(self.state.EW.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])


            dataset.attrs.update(resolution_params)
            dataset.attrs.update(model_settings)
            dataset.attrs.update({'PFAC_matrix':PFAC_matrix.flatten()})
            dataset['n'] = xr.DataArray(self.state.sh.n, coords = {'i': range(size)}, dims = ['i'], name = 'n')
            dataset['m'] = xr.DataArray(self.state.sh.m, coords = {'i': range(size)}, dims = ['i'], name = 'm')

            dataset.to_netcdf(self.result_filename)
            print('Created {}'.format(self.result_filename))


        else: # adding new coefficients to existing file:

            dataset = xr.load_dataset(self.result_filename)

            size = self.state.sh.num_coeffs
            imposed_coeffs = xr.DataArray(self.state.m_imp.coeffs. reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            induced_coeffs = xr.DataArray(self.state.m_ind.coeffs. reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            Phi_coeffs     = xr.DataArray(self.state.Phi.coeffs.reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])
            EW_coeffs      = xr.DataArray(self.state.EW.coeffs. reshape((1, size)), coords = {'time':[time], 'i': range(size)}, dims = ['time', 'i'])


            # make a copy of the dataset but with new coefficients:            
            new_dataset = dataset.copy(deep = True).drop(['SH_coefficients_imposed', 'SH_coefficients_induced', 'SH_Phi', 'SH_W', 'time'])
            new_dataset['SH_coefficients_induced'] = induced_coeffs
            new_dataset['SH_coefficients_imposed'] = imposed_coeffs
            new_dataset['SH_Phi'                 ] = Phi_coeffs
            new_dataset['SH_W'                   ] = EW_coeffs 

            # merge the copy with the old_
            dataset = xr.concat([dataset, new_dataset], dim = 'time', data_vars = 'minimal', combine_attrs = 'identical')

            dataset.to_netcdf(self.result_filename)


    def evolve_to_time(self, t, dt = 5e-4, save_steps = 200, quiet = False):
        """ Evolve to given time

        """

        if self.latest_time == 0:
            self.save_state(0) # initialize save file

        time = self.latest_time
        count = 0
        while self.latest_time < t:

            self.update_FAC()
            self.update_u()
            self.update_conductance()

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


    def set_FAC(self, FAC, _basis_evaluator, time = None):
        """
        Specify field-aligned current at ``self.num_grid.theta``,
        ``self.num_grid.lon``.

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at
                ``self.num_grid.theta`` and ``self.num_grid.lon``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        self.FAC = np.atleast_2d(FAC)
        self.FAC_basis_evaluator = _basis_evaluator

        if time is None:
            if self.FAC.shape[0] > 1:
                raise ValueError('Time has to be specified if FAC is given for multiple times')
            time = self.latest_time

        self.FAC_time = np.atleast_1d(time)

        self.next_FAC = 0
        self.update_FAC()


    def set_u(self, u_theta, u_phi, time = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        self.u_theta = np.atleast_2d(u_theta)
        self.u_phi = np.atleast_2d(u_phi)

        if time is None:
            if self.u_theta.shape[0] > 1 or self.u_phi.shape[0] > 1:
                raise ValueError('Time has to be specified if u is given for multiple times')
            time = self.latest_time

        self.u_time = np.atleast_1d(time)

        self.next_u = 0
        self.update_u()


    def set_conductance(self, Hall, Pedersen, _basis_evaluator, time = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.lon``.

        """

        self.Hall = np.atleast_2d(Hall)
        self.Pedersen = np.atleast_2d(Pedersen)
        self.conductance_basis_evaluator = _basis_evaluator

        if time is None:
            if self.Hall.shape[0] > 1 or self.Pedersen.shape[0] > 1:
                raise ValueError('Time has to be specified if conductance is given for multiple times')
            time = self.latest_time

        self.conductance_time = np.atleast_1d(time)

        self.next_conductance = 0
        self.update_conductance()


    def update_FAC(self):
        """ Update FAC """

        if self.next_FAC < self.FAC_time.size:
            if self.latest_time >= self.FAC_time[self.next_FAC]:
                self.state.set_FAC(self.FAC[self.next_FAC], self.FAC_basis_evaluator)
                self.next_FAC += 1
                self.updated_FAC = True


    def update_u(self):
        """ Update neutral wind """

        if self.next_u < self.u_time.size:
            if self.latest_time >= self.u_time[self.next_u]:
                self.state.set_u(self.u_theta[self.next_u], self.u_phi[self.next_u])
                self.next_u += 1
                self.updated_u = True


    def update_conductance(self):
        """ Update conductance """

        if self.next_conductance < self.conductance_time.size:
            if self.latest_time >= self.conductance_time[self.next_conductance]:
                self.state.set_conductance(self.Hall[self.next_conductance], self.Pedersen[self.next_conductance], self.conductance_basis_evaluator)
                self.next_conductance += 1
                self.updated_conductance = True

    @property
    def fd_curl_matrix(self, stencil_size = 1, interpolation_points = 4):
        """ Calculate matrix that returns the radial curl, using finite differences 
            when operated on a column vector of (theta, phi) vector components. 
            The function also returns the pseudo-inverse of the matrix. 
        """

        if not hasattr(self, '_fd_curl_matrix'):
            
            Dxi, Deta = self.csp.get_Diff(self.csp.N, coordinate = 'both', Ns = stencil_size, Ni = interpolation_points, order = 1)
            sqrtg = np.sqrt(self.csp.detg)
            g11_scaled = sp.diags(self.csp.g[:, 0, 0] / sqrtg)
            g12_scaled = sp.diags(self.csp.g[:, 0, 1] / sqrtg)
            g22_scaled = sp.diags(self.csp.g[:, 1, 1] / sqrtg)

            # matrix that operates on column vector of u1, u2 and produces radial curl
            D_curlr_u1u2 = sp.hstack(((Dxi.dot(g12_scaled) - Deta.dot(g11_scaled)),
                                      (Dxi.dot(g22_scaled) - Deta.dot(g12_scaled))))

            # matrix that transforms theta, phi to u1, u2:
            Ps_dense = self.csp.get_Ps(self.csp.arr_xi, self.csp.arr_eta, block = self.csp.arr_block) # N x 3 x 3
            # extract relevant elements, rearrange so that the matrix operates on (theta, phi) and not (east, north), 
            # and insert in sparse diagonal matrices. Also include the normalization from the Q matrix in Yin et al.:
            rr, rrcosl = self.RI, self.RI * np.cos(np.deg2rad(self.num_grid.lat)) # normalization factors
            Ps00 = sp.diags(-Ps_dense[:, 0, 1] / rr    ) 
            Ps01 = sp.diags( Ps_dense[:, 0, 0] / rrcosl) 
            Ps10 = sp.diags(-Ps_dense[:, 1, 1] / rr    ) 
            Ps11 = sp.diags( Ps_dense[:, 1, 0] / rrcosl)
            # stack:
            Ps = sp.vstack((sp.hstack((Ps00, Ps01)), sp.hstack((Ps10, Ps11))))

            # combine:
            self._fd_curl_matrix = D_curlr_u1u2.dot(Ps)

        return(self._fd_curl_matrix)

    def steady_state_m_ind(self, m_imp = None):
        """ Calculate coefficients for induced field in steady state 
            
            Parameters:
            -----------
            m_imp: array, optional
                the coefficient vector for the imposed magnetic field. If None, the
                vector for the current state will be used

            Returns:
            --------
            m_ind_ss: array
                array of coefficients for the induced magnetic field in steady state

        """
        
        GVJ = self.state.G_m_ind_to_JS
        GTJ = self.state.G_m_imp_to_JS

        br, bt, bp = self.state.b_evaluator.br, self.state.b_evaluator.btheta, self.state.b_evaluator.bphi
        eP, eH = self.state.etaP, self.state.etaH
        C00 = sp.diags(eP * (bp**2 + br**2))
        C01 = sp.diags(eP * (-bt * bp) + eH * br)
        C10 = sp.diags(eP * (-bt * bp) - eH * br)
        C11 = sp.diags(eP * (bt**2 + br**2))
        C = sp.vstack((sp.hstack((C00, C01)), sp.hstack((C10, C11))))

        uxb = np.hstack((self.state.uxB_theta, self.state.uxB_phi))

        GcCGVJ = self.fd_curl_matrix.dot(C).dot(GVJ)
        GcCGTJ = self.fd_curl_matrix.dot(C).dot(GTJ)

        if m_imp is None:
            m_imp = self.state.m_imp.coeffs

        m_ind_ss = np.linalg.pinv(GcCGVJ, rcond = 0).dot(self.fd_curl_matrix.dot(uxb) - GcCGTJ.dot(m_imp))

        return(m_ind_ss)





def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 45, Mmax = 3, Ncs = 60, mainfield_kind = 'dipole', fig_directory = './figs', ignore_PFAC = True, connect_hemispheres = False, latitude_boundary = 50, zero_jr_at_dip_equator = False, wind_directory = None):

    # Set up the spherical harmonic basis object
    i2d_sh = SHBasis(Nmax, Mmax)

    # Define CS grid used for SH analysis and gradient calculations
    # Each cube block with have ``(Ncs-1)*(Ncs-1)`` cells.
    csp = cubedsphere.CSProjection(Ncs) # cubed sphere projection object

    # Initialize the 2D ionosphere object at 110 km altitude
    RI = RE + 110.e3
    i2d = I2D(result_filename = None, sh = i2d_sh, csp = csp, RI = RI, mainfield_kind = mainfield_kind, ignore_PFAC = ignore_PFAC, connect_hemispheres = connect_hemispheres, latitude_boundary = latitude_boundary, zero_jr_at_dip_equator = zero_jr_at_dip_equator)

    import pyamps
    from pynamit.visualization import globalplot, cs_interpolate
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime
    import polplot
    from pynamit.primitives.basis_evaluator import BasisEvaluator
    from pynamit.primitives.field_evaluator import FieldEvaluator
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
    csp_grid = Grid(90 - csp.arr_theta, csp.arr_phi)
    csp_i2d_evaluator = BasisEvaluator(i2d_sh, csp_grid)
    csp_b_evaluator = FieldEvaluator(i2d.state.mainfield, csp_grid, RI)

    # Define grid used for plotting
    lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
    lat, lon = np.meshgrid(lat, lon)
    pltshape = lat.shape
    plt_grid = Grid(lat, lon)
    plt_i2d_evaluator = BasisEvaluator(i2d_sh, plt_grid)

    hall, pedersen = conductance.hardy_EUV(csp_grid.lon, csp_grid.lat, Kp, date, starlight = 1, dipole = True)
    i2d.set_conductance(hall, pedersen, csp_i2d_evaluator)

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

        u_int = csp.interpolate_vector_components(u_phi, -u_theta, np.zeros_like(u_phi), 90 - u_lat, u_lon, csp.arr_theta, csp.arr_phi)
        u_east_int, u_north_int, u_r_int = u_int
        i2d.set_u(-u_north_int * WIND_FACTOR, u_east_int * WIND_FACTOR)

    i2d.set_FAC(jparallel, csp_i2d_evaluator)

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

        mn_grid = Grid(mlatn, mltn)
        mnv_grid = Grid(mlatnv, mltnv)

        paxes[0].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[0], levels = levels, cmap = plt.cm.bwr)
        paxes[0].quiver(  mnv_grid.lat, mnv_grid.lon,  np.split(jn_amps, 2)[0], np.split(je_amps, 2)[0], scale = SCALE, color = 'black')
        paxes[1].contourf(mn_grid.lat , mn_grid.lon ,  np.split(ju_amps, 2)[1], levels = levels, cmap = plt.cm.bwr)
        paxes[1].quiver(  mnv_grid.lat, mnv_grid.lon, -np.split(jn_amps, 2)[1], np.split(je_amps, 2)[1], scale = SCALE, color = 'black')

        m_i2d_evaluator = BasisEvaluator(i2d_sh, Grid(mlat, lon))
        jr = i2d.get_Jr(m_i2d_evaluator) * 1e6

        mv_i2d_evaluator = BasisEvaluator(i2d_sh, Grid(mlatv, lonv))
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
            mn_grid = Grid(mlatn, mltn)

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
            coeffs.append(i2d.state.m_ind.coeffs)
            count += 1
            #print(count, time, (i2d.state.m_ind.coeffs * i2d.state.m_ind_to_Br)[:3])

            if (count % plotsteps == 0) and fig_directory_writeable:
                print(count, time, (i2d.state.m_ind.coeffs * i2d.state.m_ind_to_Br)[:3])
                fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
                filecount +=1
                title = 't = {:.3} s'.format(time)
                Br = i2d.state.get_Br(plt_i2d_evaluator)
                _, paxn, paxs, _ =  globalplot(plt_grid.lon.reshape(pltshape), plt_grid.lat.reshape(pltshape), Br.reshape(pltshape) , title = title, returnplot = True,
                                               levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
                #W = i2d.state.get_W(plt_i2d_evaluator) * 1e-3

                i2d.state.update_Phi_and_EW()
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