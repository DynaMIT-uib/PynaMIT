import numpy as np
from pynamit.decorators import default_2Dcoords, default_3Dcoords
from pynamit.mainfield import Mainfield
from sh_utils.sh_utils import get_G
import os
from pynamit.cubedsphere import cubedsphere

RE = 6371.2e3
mu0 = 4 * np.pi * 1e-7


class I2D(object):
    """ 2D ionosphere. """

    def __init__(self, Nmax, Mmax, Ncs = 20, 
                       RI = RE + 110.e3, B0 = 'dipole', 
                       B0_parameters = {'epoch':2020}, 
                       FAC_integration_parameters = {'steps':np.logspace(np.log10(RE + 110.e3), np.log10(4 * RE), 11)},
                       ignore_PNAF = False):
        """

        Parameters
        ----------
        Nmax: int
            Maximum spherical harmonic degree.
        Mmax: int
            Maximum spherical harmonic order.
        Ncs: int, optional, default = 20
            Each cube block with have ``(Ncs-1)*(Ncs-1)`` cells.
        RI: float, optional, default = RE + 110.e3
            Radius of the ionosphere in m.
        B0: string, {'dipole', 'radial', 'igrf'}, default = 'dipole'
            Set to the main field model you want. For 'dipole' and
            'igrf', you can specify epoch via `B0_parameters`.
        FAC_integration_parameters: dict
            Use this to specify parameters in the integration required to
            find the poloidal part of the magnetic field of FACs. Not
            relevant for radial `B0`.

        """
        self.ignore_PNAF = ignore_PNAF
        self.RI = RI
        self.Nmax, self.Mmax = Nmax, Mmax

        self.B0 = Mainfield(kind = B0, **B0_parameters)

        # Define CS grid used for SH analysis and gradient calculations
        self.csp = cubedsphere.CSprojection(Ncs) # cubed sphere projection object
        self.theta, self.phi = self.csp.arr_theta, self.csp.arr_phi
        #self.Dxi, self.Deta = self.csp.get_Diff(Ncs, coordinate = 'both') # differentiation matrices in xi and eta directions
        self.g  = self.csp.g # csp.get_metric_tensor(xi, eta, 1, covariant = True) 
        self.Ps = self.csp.get_Ps(self.csp.arr_xi, self.csp.arr_eta, 1, self.csp.arr_block)                           # matrices to convert from u^east, u^north, u^up to u^1 ,u^2, u^3 (A1 in Yin)
        self.Qi = self.csp.get_Q(90 - self.theta, self.RI, inverse = True) # matrices to convert from physical north, east, radial to u^east, u^north, u^up (A1 in Yin)
        self.sqrtg = np.sqrt(self.csp.detg) #np.sqrt(cubedsphere.arrayutils.get_3D_determinants(self.g))
        self.g12 = self.g[:, 0, 1]
        self.g22 = self.g[:, 1, 1]
        self.g11 = self.g[:, 0, 0]

        # get magnetic field unit vectors at CS grid:
        B = np.vstack(self.B0.get_B(self.RI, self.theta, self.phi))
        self.br, self.btheta, self.bphi = B / np.linalg.norm(B, axis = 0)
        self.sinI = self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)
        # construct the elements in the matrix in the electric field equation
        self.b00 = self.bphi**2 + self.br**2
        self.b01 = -self.btheta * self.bphi
        self.b10 = -self.btheta * self.bphi
        self.b11 = self.btheta**2 + self.br**2

        # Define grid used for plotting 
        lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
        self.lat, self.lon = np.meshgrid(lat, lon)

        # Define matrices for surface spherical harmonics
        self.Gnum, self.n, self.m = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, return_nm  = True)
        self.Gnum_ph              = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, derivative = 'phi'  )
        self.Gnum_th              = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, derivative = 'theta')
        self.Gplt                 = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI)
        self.Gplt_ph              = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI, derivative = 'phi'  )
        self.Gplt_th              = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI, derivative = 'theta')

        self.Nshc = self.Gnum.shape[1] # number of spherical harmonic coefficients

        # Pre-calculate GTG and its inverse
        self.GTG = self.Gnum.T.dot(self.Gnum)
        self.GTG_inv = np.linalg.pinv(self.GTG)

        # Pre-calculate matrix to get coefficients for curl-free fields:
        self.Gcf = np.vstack((-self.Gnum_th, -self.Gnum_ph)) 
        self.GTGcf_inv = np.linalg.pinv(self.Gcf.T.dot(self.Gcf))
        
        self.vector_to_shc_cf = self.GTGcf_inv.dot(self.Gcf.T)

        # Pre-calculate matrix to get coefficients for divergence-free fields
        self.Gdf = np.vstack((-self.Gnum_ph, self.Gnum_th)) 
        self.GTGdf_inv = np.linalg.pinv(self.Gdf.T.dot(self.Gdf))

        self.vector_to_shc_df = self.GTGdf_inv.dot(self.Gdf.T)

        # Report condition number for GTG
        self.cond_GTG = np.linalg.cond(self.GTG)
        print('The condition number for the surface SH matrix is {:.1f}'.format(self.cond_GTG))

        # Pre-calculate the matrix that maps from TJr_shc to coefficients for the poloidal magnetic field of FACs
        if self.B0 == 'radial' or self.ignore_PNAF: # no Poloidal field so get matrix of zeros
            self.shc_TJr_to_shc_PFAC = np.zeros((self.Nshc, self.Nshc))
        else: # Use the method by Engels and Olsen 1998, Eq. 13:
            r_k_steps = FAC_integration_parameters['steps']
            Delta_k = np.diff(r_k_steps)
            r_k = np.array(r_k_steps[:-1] + 0.5 * Delta_k)

            # initialize matrix that will map from self.TJr to coefficients for poloidal field:
            shc_TJr_to_shc_PFAC = np.zeros((self.Nshc, self.Nshc))
            for i in range(r_k.size): # TODO: it would be useful to use Dask for this loop to speed things up a little
                print(f'Calculating matrix for poloidal field of FACs. Progress: {i+1}/{r_k.size}', end = '\r' if i < (r_k.size - 1) else '\n')
                # map coordinates from r_k[i] to RI:
                theta_mapped, phi_mapped = self.B0.map_coords(self.RI, r_k[i], self.theta, self.phi)

                # Calculate magnetic field at grid points at r_k[i]:
                B_rk  = np.vstack(self.B0.get_B(r_k[i], self.theta, self.phi))
                B0_rk = np.linalg.norm(B_rk, axis = 0) # magnetic field magnitude
                b_rk = B_rk / B0_rk # unit vectors

                # Calculate magnetic field at the points in the ionosphere to which the grid maps:
                B_RI  = np.vstack(self.B0.get_B(self.RI, theta_mapped, phi_mapped))
                B0_RI = np.linalg.norm(B_RI, axis = 0) # magnetic field magnitude
                sinI_RI = B_RI[0] / B0_RI

                # find matrix that gets radial current at these coordinates:
                Q_k = get_G(90 - theta_mapped, phi_mapped, self.Nmax, self.Mmax, a = self.RI)

                # we need to scale this by 1/sin(inclination) to get the FAC:
                Q_k = Q_k / sinI_RI.reshape((-1, 1)) # TODO: Handle singularity at equator (may be fine)

                # matrix that scales the FAC at RI to r_k and extracts the horizontal components:
                ratio = (B0_rk / B0_RI).reshape((1, -1))
                S_k = np.vstack((np.diag(b_rk[1]), np.diag(b_rk[2]))) * ratio

                # matrix that scales the terms by (R/r_k)**(n-1):
                A_k = np.diag((self.RI / r_k[i])**(self.n - 1))

                # put it all together (crazy)
                shc_TJr_to_shc_PFAC += Delta_k[i] * A_k.dot(self.vector_to_shc_df.dot(S_k.dot(Q_k)))

            # finally scale the matrix by the term in front of the integral
            self.shc_TJr_to_shc_PFAC = -np.diag((self.n + 1) / (2 * self.n + 1)).dot(shc_TJr_to_shc_PFAC) / self.RI

            # make matrices that translate shc_PFAC to horizontal current density (assuming divergence-free shielding current)
            self.shc_PFAC_to_Jph = -  1 / (self.n + 1) * self.Gnum_ph / mu0
            self.shc_PFAC_to_Jth =    1 / (self.n + 1) * self.Gnum_th / mu0

        # Initialize the spherical harmonic coefficients
        shc_VB, shc_TB = np.zeros(self.Gnum.shape[1]), np.zeros(self.Gnum.shape[1])
        self.set_shc(VB = shc_VB)
        self.set_shc(TB = shc_TB)




    def evolve_Br(self, dt):
        """ Evolve Br in time.

        """

        #Eth, Eph = self.get_E()
        #u1, u2, u3 = self.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        #curlEr = self.curlr(u1, u2) 
        #Br = -self.GBr.dot(self.shc_VB) - dt * curlEr

        #self.set_shc(Br = self.GTG_inv.dot(self.Gnum.T.dot(-Br)))

        #GTE = self.Gcf.T.dot(np.hstack( self.get_E()) )
        #self.shc_EW = self.GTGcf_inv.dot(GTE) # find coefficients for divergence-free / inductive E
        self.shc_EW = self.vector_to_shc_df.dot(np.hstack( self.get_E()))
        self.set_shc(Br = self.shc_Br + self.n * (self.n + 1) * self.shc_EW * dt / self.RI**2)


    def sph_to_contravariant_cs(self, Ar, Atheta, Aphi):
        """
        Convert from ``(east, north, up)`` to ``(u^1, u^2, u^3)`` (ref.
        Yin).

        The input must match the CS grid.

        """

        east = Aphi
        north = -Atheta
        up = Ar

        #print('TODO: Add checks that input matches grid etc.')

        v = np.vstack((east, north, up))
        v_components = np.einsum('nij, jn -> in', self.Qi, v)
        u1, u2, u3   = np.einsum('nij, jn -> in', self.Ps, v_components)
        
        return u1, u2, u3




    def curlr(self, u1, u2):
        """
        Construct a matrix that calculates the radial curl using B6 in
        Yin et al.

        """
        


        return( 1/self.sqrtg * ( self.Dxi.dot(self.g12 * u1 + self.g22 * u2) - 
                                 self.Deta.dot(self.g11 * u1 + self.g12 *u2) ) )


    def set_shc(self, **kwargs):
        """ Set spherical harmonic coefficients.

        Specify a set of spherical harmonic coefficients and update the
        rest so that they are consistent. 

        This function accepts one (and only one) set of spherical harmonic
        coefficients. Valid values for kwargs (only one):

        - 'VB' : Coefficients for magnetic field scalar ``V``.
        - 'TB' : Coefficients for surface current scalar ``T``.
        - 'VJ' : Coefficients for magnetic field scalar ``V``.
        - 'TJ' : Coefficients for surface current scalar ``T``.
        - 'Br' : Coefficients for magnetic field ``Br`` (at ``r = RI``).
        - 'TJr': Coefficients for radial current scalar.

        """ 
        valid_kws = ['VB', 'TB', 'VJ', 'TJ', 'Br', 'TJr']

        if len(kwargs) != 1:
            raise Exception('Expected one and only one keyword argument, you provided {}'.format(len(kwargs)))
        key = list(kwargs.keys())[0]
        if key not in valid_kws:
            raise Exception('Invalid keyword. See documentation')

        if key == 'VB':
            self.shc_VB = kwargs['VB']
            self.shc_VJ = self.RI / mu0 * (2 * self.n + 1) / (self.n + 1) * self.shc_VB
            self.shc_Br = 1 / self.n * self.shc_VB
        elif key == 'TB':
            self.shc_TB = kwargs['TB']
            self.shc_TJ = -self.RI / mu0 * self.shc_TB
            self.shc_TJr = -self.n * (self.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'VJ':
            self.shc_VJ = kwargs['VJ']
            self.shc_VB = mu0 / self.RI * (self.n + 1) / (2 * self.n + 1) * self.shc_VJ
            self.shc_Br = 1 / self.n * self.shc_VB
        elif key == 'TJ':
            self.shc_TJ = kwargs['TJ']
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            self.shc_TJr = -self.n * (self.n + 1) / self.RI**2 * self.shc_TJ 
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
        elif key == 'Br':
            self.shc_Br = kwargs['Br']
            self.shc_VB = self.shc_Br / self.n
            self.shc_VJ = -self.RI / mu0 * (2 * self.n + 1) / (self.n + 1) * self.shc_VB
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = -1 /(self.n * (self.n + 1)) * self.shc_TJr * self.RI**2
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            self.shc_PFAC = self.shc_TJr_to_shc_PFAC.dot(self.shc_TJr) 
            print('check the factor RI**2!')
        else:
            raise Exception('This should not happen')



    def set_initial_condition(self, I2D_object):
        """ Set initial conditions.

        If this is not called, initial condictions should be zero.

        """
        print('not implemented. inital conditions will be zero')


    def set_FAC(self, FAC):
        """ Specify field-aligned current at ``self.theta``, ``self.phi``.

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at ``self.theta`` and
                ``self.phi``, at ``RI``. The values in the array have to
                match the corresponding coordinates.

        """

        # Extract the radial component of the FAC:
        jr = FAC * self.sinI 
        # Get the corresponding spherical harmonic coefficients
        TJr = np.linalg.lstsq(self.GTG, self.Gnum.T.dot(jr), rcond = 1e-3)[0]
        # Propagate to the other coefficients (TB, TJ, PFAC):
        self.set_shc(TJr = TJr)

        print('Note: Check if rcond is really needed. It should not be necessary if the FAC is given sufficiently densely')
        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_conductance(self, Hall, Pedersen):
        """
        Specify Hall and Pedersen conductance at ``self.theta``,
        ``self.phi``.

        """
        if Hall.size != Pedersen.size != self.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.SH = Hall
        self.SP = Pedersen
        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)


    @default_3Dcoords
    def get_Br(self, r = None, theta = None, phi = None, deg = False):
        """ Calculate ``Br``.

        """
        return(self.Gplt.dot(self.shc_Br))


    @default_2Dcoords
    def get_JS(self, theta = None, phi = None, deg = False):
        """ Calculate ionospheric sheet current.

        """
        Je_V =  self.Gnum_th.dot(self.shc_VJ) # r cross grad(VJ) eastward component
        Js_V = -self.Gnum_ph.dot(self.shc_VJ) # r cross grad(VJ) southward component
        Je_T = -self.Gnum_ph.dot(self.shc_TJ) # -grad(VT) eastward component
        Js_T = -self.Gnum_th.dot(self.shc_TJ) # -grad(VT) southward component

        Jth, Jph = Js_V + Js_T, Je_V + Je_T

        if not self.ignore_PNAF:
            Jth = Jth + self.shc_PFAC_to_Jth.dot(self.shc_PFAC)
            Jph = Jph + self.shc_PFAC_to_Jph.dot(self.shc_PFAC)

        return(Jth, Jph)


    @default_2Dcoords
    def get_Jr(self, theta = None, phi = None, deg = False):
        """ Calculate radial current.

        """

        print('this must be fixed so that I can evaluate anywere')
        return self.Gplt.dot(self.shc_TJr)




    @default_2Dcoords
    def get_equivalent_current_function(self, theta = None, phi = None, deg = False):
        """ Calculate equivalent current function.

        """
        print('not implemented')


    @default_2Dcoords
    def get_Phi(self, theta = None, phi = None, deg = False):
        """ Calculate electric potential.

        """

        print('this must be fixed so that Phi can be evaluated anywere')
        return self.Gplt.dot(self.shc_Phi) * 1e-3


    @default_2Dcoords
    def get_W(self, theta = None, phi = None, deg = False):
        """ Calculate the induction electric field scalar.

        """

        print('this must be fixed so that W can be evaluated anywere')
        return self.Gplt.dot(self.shc_EW) * 1e-3


    @default_2Dcoords
    def get_E(self, theta = None, phi = None, deg = False):
        """ Calculate electric field.

        """


        Jth, Jph = self.get_JS(theta = theta, phi = phi)


        Eth = self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.br * Jph)
        Eph = self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.br * Jth)

        return(Eth, Eph)

def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 45, Mmax = 3, Ncs = 60, B0_type = 'dipole', fig_directory = './figs'):

    i2d = I2D(Nmax, Mmax, Ncs, B0 = B0_type, ignore_PNAF = True)

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

    hall, pedersen = conductance.hardy_EUV(i2d.phi, 90 - i2d.theta, Kp, date, starlight = 1, dipole = True)
    i2d.set_conductance(hall, pedersen)

    a = pyamps.AMPS(300, 0, -4, 20, 100, minlat = 50)
    ju = a.get_upward_current(mlat = 90 - i2d.theta, mlt = d.mlon2mlt(i2d.phi, date)) * 1e-6
    ju[np.abs(90 - i2d.theta) < 50] = 0 # filter low latitude FACs

    ju[i2d.theta < 90] = -ju[i2d.theta < 90] # we need the current to refer to magnetic field direction, so changing sign in the north since the field there points down 

    i2d.set_FAC(ju)

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
        paxes[0].contourf(mlatn , mltn ,  np.split(ju_amps, 2)[0], levels = levels, cmap = plt.cm.bwr)
        paxes[0].quiver(  mlatnv, mltnv,  np.split(jn_amps, 2)[0], np.split(je_amps, 2)[0], scale = SCALE, color = 'black')
        paxes[1].contourf(mlatn , mltn ,  np.split(ju_amps, 2)[1], levels = levels, cmap = plt.cm.bwr)
        paxes[1].quiver(  mlatnv, mltnv, -np.split(jn_amps, 2)[1], np.split(je_amps, 2)[1], scale = SCALE, color = 'black')


        lon  = d.mlt2mlon(mlt , date)
        lonv = d.mlt2mlon(mltv, date)
        G   = get_G(mlat ,  lon, i2d.Nmax, i2d.Mmax, a = i2d.RI) * 1e6
        Gph = get_G(mlatv, lonv, i2d.Nmax, i2d.Mmax, a = i2d.RI, derivative = 'phi'  ) * 1e3 
        Gth = get_G(mlatv, lonv, i2d.Nmax, i2d.Mmax, a = i2d.RI, derivative = 'theta') * 1e3
        jr = G.dot(i2d.shc_TJr)

        je = -Gph.dot(i2d.shc_TJ)
        jn =  Gth.dot(i2d.shc_TJ)

        jrn, jrs = np.split(jr, 2) 
        paxes[2].contourf(mlatn, mltn, jrn, levels = levels, cmap = plt.cm.bwr)
        paxes[2].quiver(mlatnv, mltnv,  np.split(jn, 2)[0], np.split(je, 2)[0], scale = SCALE, color = 'black')
        paxes[3].contourf(mlatn, mltn, jrs, levels = levels, cmap = plt.cm.bwr)
        paxes[3].quiver(mlatnv, mltnv,  -np.split(jn, 2)[1], np.split(je, 2)[1], scale = SCALE, color = 'black')

        jr = i2d.get_Jr()

        globalplot(i2d.lon, i2d.lat, jr.reshape(i2d.lon.shape) * 1e6, noon_longitude = lon0, cmap = plt.cm.bwr, levels = levels)




        plt.show()



    if plot_AMPS_Br:
        fig, axes = plt.subplots(ncols = 2, figsize = (10, 5))
        paxes = [polplot.Polarplot(ax) for ax in axes.flatten()]
        mlat  , mlt   = a.scalargrid
        mlatn , mltn  = np.split(mlat , 2)[0], np.split(mlt , 2)[0]
        Bu = a.get_ground_Buqd(height = a.height)
        paxes[0].contourf(mlatn, mltn, np.split(Bu, 2)[0], levels = Blevels * 1e9, cmap = plt.cm.bwr)
        paxes[1].contourf(mlatn, mltn, np.split(Bu, 2)[1], levels = Blevels * 1e9, cmap = plt.cm.bwr)

        plt.show()


    if show_FAC_and_conductance:

        hall_plt = cs_interpolate(i2d.csp, 90 - i2d.theta, i2d.phi, hall, i2d.lat, i2d.lon)
        pede_plt = cs_interpolate(i2d.csp, 90 - i2d.theta, i2d.phi, pedersen, i2d.lat, i2d.lon)

        globalplot(i2d.lon, i2d.lat, hall_plt, noon_longitude = lon0, levels = c_levels, save = 'hall.png')
        globalplot(i2d.lon, i2d.lat, pede_plt, noon_longitude = lon0, levels = c_levels, save = 'pede.png')

        jr = i2d.get_Jr()
        globalplot(i2d.lon, i2d.lat, jr.reshape(i2d.lon.shape), noon_longitude = lon0, levels = levels * 1e-6, save = 'jr.png', cmap = plt.cm.bwr)

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
    #globalplot(i2d.lon, i2d.lat, jr.reshape(i2d.lat.shape), 
    #           levels = levels, cmap = 'bwr', central_longitude = lon0)

    #globalplot(i2d.phi, 90 - i2d.theta, i2d.SH, vmin = 0, vmax = 20, cmap = 'viridis', scatter = True, central_longitude = lon0)

    fig_directory_writeable = os.access(fig_directory, os.W_OK)

    if not fig_directory_writeable:
        print('Figure directory {} is not writeable, proceeding without figure generation. For figures, rerun after ensuring that the directory exists and is writeable.'.format(fig_directory))

    if SIMULATE:

        coeffs = []
        count = 0
        filecount = 1
        time = 0
        while True:

            i2d.evolve_Br(dt)
            time = time + dt
            coeffs.append(i2d.shc_VB)
            count += 1
            #print(count, time, i2d.shc_Br[:3])

            if (count % plotsteps == 0) and fig_directory_writeable:
                print(count, time, i2d.shc_Br[:3])
                fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
                filecount +=1
                title = 't = {:.3} s'.format(time)
                Br = i2d.get_Br()
                fig, paxn, paxs, axg =  globalplot(i2d.lon, i2d.lat, Br.reshape(i2d.lat.shape) , title = title, returnplot = True, 
                                                   levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
                #W = i2d.get_W()

                i2d.shc_Phi = i2d.vector_to_shc_df.dot(np.hstack( i2d.get_E()))
                Phi = i2d.get_Phi()


                nnn = i2d.lat.flatten() >  50
                sss = i2d.lat.flatten() < -50
                #paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, W  [nnn], colors = 'black', levels = Wlevels, linewidths = .5)
                #paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, W  [sss], colors = 'black', levels = Wlevels, linewidths = .5)
                paxn.contour(i2d.lat.flatten()[nnn], (i2d.lon.flatten() - lon0)[nnn] / 15, Phi[nnn], colors = 'black', levels = Philevels, linewidths = .5)
                paxs.contour(i2d.lat.flatten()[sss], (i2d.lon.flatten() - lon0)[sss] / 15, Phi[sss], colors = 'black', levels = Philevels, linewidths = .5)
                plt.savefig(fn)

            if count > totalsteps:
                break
    return coeffs