import numpy as np
import dipole
from pynamit.decorators import default_2Dcoords, default_3Dcoords
from sh_utils.sh_utils import get_G
import sys
import os

# import cubedsphere submodule
cs_path = os.path.join(os.path.dirname(__file__), 'cubedsphere')
sys.path.insert(0, cs_path)
import cubedsphere
csp = cubedsphere.CSprojection() # cubed sphere projection object

RE = 6371.2e3
mu0 = 4 * np.pi * 1e-7

def B0_radial(r, theta, phi):
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    size = r.size
    zeros, ones = np.zeros(size), np.ones(size)
    return(np.vstack((ones, zeros, zeros)))

dd = dipole.Dipole(2020)
def B0_dipole(r, theta, phi):
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    r, theta, phi = r.flatten(), theta.flatten(), phi.flatten()
    size = r.size
    Bn, Br = dd.B(90 - theta, r * 1e-3)
    return(np.vstack((Br, -Bn, np.zeros(r.size))))


class I2D(object):
    """ 2D ionosphere """

    def __init__(self, Nmax, Mmax, Ncs = 20, B0 = None, RI = RE + 110.e3):
        """


        Parameters
        ----------
        B0: function, optional
            Should return the main magnetic field r, theta, phi components, in that order.
            The function should accept r, theta, phi as input. The coordinate system used
            by this function defines the coordinate system used in I2D. It is assumed to be
            an orthogonal spherical coordinate system. 
            The default B0 for now gives a radial field
            The unit of this function does not matter, since the vectors will be normalized.  

        """
        self.RI = RI
        self.Nmax, self.Mmax = Nmax, Mmax

        self.B0 = B0_default if B0 is None else B0

        # Define CS grid used for SH analysis and gradient calculations
        k, i, j = csp.get_gridpoints(Ncs)
        xi, eta = csp.xi(i, Ncs), csp.eta(j, Ncs)
        _, self.theta, self.phi = csp.cube2spherical(xi, eta, k, deg = True)
        self.theta, self.phi = self.theta.flatten(), self.phi.flatten()
        self.Dxi, self.Deta = csp.get_Diff(Ncs, coordinate = 'both') # differentiation matrices in xi and eta directions
        self.g  = csp.get_metric_tensor(xi, eta, 1, covariant = True) 
        self.Ps = csp.get_Ps(xi, eta, 1, k)                           # matrices to convert from u^east, u^north, u^up to u^1 ,u^2, u^3 (A1 in Yin)
        self.Qi = csp.get_Q(90 - self.theta, self.RI, inverse = True) # matrices to convert from physical north, east, radial to u^east, u^north, u^up (A1 in Yin)
        self.sqrtg = np.sqrt(cubedsphere.arrayutils.get_3D_determinants(self.g))
        self.g12 = self.g[:, 0, 1]
        self.g22 = self.g[:, 1, 1]
        self.g11 = self.g[:, 0, 0]

        # get magnetic field unit vectors at CS grid:
        B = np.vstack(self.B0(self.RI, self.theta, self.phi))
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
        print('TODO: it would be nice to have access to n without this stupid syntax. write a class?')
        self.Gnum, self.n = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, return_n   = True)
        self.Gnum_ph      = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, derivative = 'phi'  )
        self.Gnum_th      = get_G(90 - self.theta, self.phi, self.Nmax, self.Mmax, a = self.RI, derivative = 'theta')
        self.Gplt         = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI)
        self.Gplt_ph      = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI, derivative = 'phi'  )
        self.Gplt_th      = get_G(self.lat, self.lon, self.Nmax, self.Mmax, a = self.RI, derivative = 'theta')

        # Initialize the spherical harmonic coefficients
        shc_VB, shc_TB = np.zeros(self.Gnum.shape[1]), np.zeros(self.Gnum.shape[1])
        self.set_shc(VB = shc_VB)
        self.set_shc(TB = shc_TB)

        # Pre-calculate GTG and its inverse
        self.GTG = self.Gnum.T.dot(self.Gnum)
        self.GTG_inv = np.linalg.pinv(self.GTG)

        # Pre-calculate matrix to get coefficients for divergence-free fields:
        self.Gcf = np.vstack((-self.Gnum_ph, self.Gnum_th)) 
        self.GTGcf_inv = np.linalg.pinv(self.Gcf.T.dot(self.Gcf))

        # Pre-calculate matrix to get coefficients for curl-free fields
        self.Gdf = np.vstack((-self.Gnum_th, -self.Gnum_ph)) 
        self.GTGdf_inv = np.linalg.pinv(self.Gdf.T.dot(self.Gdf))

        # Pre-calculate matrix for calculating Br
        self.GBr = self.Gnum * self.n.reshape((1, -1))

        # Report condition number for GTG
        self.cond_GTG = np.linalg.cond(self.GTG)
        print('The condition number for the surface SH matrix is {:.1f}'.format(self.cond_GTG))



    def evolve_Br(self, dt):
        """ evolve Br in time """

        #Eth, Eph = self.get_E()
        #u1, u2, u3 = self.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        #curlEr = self.curlr(u1, u2) 
        #Br = -self.GBr.dot(self.shc_VB) - dt * curlEr

        #self.set_shc(Br = self.GTG_inv.dot(self.Gnum.T.dot(-Br)))


        GTE = self.Gcf.T.dot(np.hstack( self.get_E()) )
        self.shc_EW = self.GTGcf_inv.dot(GTE) # find coefficients for divergence-free / inductive E
        self.set_shc(Br = self.shc_Br + self.n * (self.n + 1) * self.shc_EW * dt / self.RI**2)


    def sph_to_contravariant_cs(self, Ar, Atheta, Aphi):
        """ convert from east, north up to u^1, u^2, u^3 (ref Yin)

            The input must match the CS grid
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
        """ construct a matrix that calculates the radial curl using B6 in Yin et al. """
        


        return( 1/self.sqrtg * ( self.Dxi.dot(self.g12 * u1 + self.g22 * u2) - 
                                 self.Deta.dot(self.g11 * u1 + self.g12 *u2) ) )


    def set_shc(self, **kwargs):
        """ Set spherical harmonic coefficients 

        Specify a set of spherical harmonic coefficients and update the rest so that they are consistent. 

        This function accepts one (and only one) set of spherical harmonic coefficients.
        Valid values for kwargs (only one):
            VB : Coefficients for magnetic field scalar V
            TB : Coefficients for surface current scalar T
            VJ : Coefficients for magnetic field scalar V
            TJ : Coefficients for surface current scalar T
            Br : Coefficients for magnetic field Br (at r=RI)
            TJr: Coefficients for radial current scalar

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
        elif key == 'VJ':
            self.shc_VJ = kwargs['VJ']
            self.shc_VB = mu0 / self.RI * (self.n + 1) / (2 * self.n + 1) * self.shc_VJ
            self.shc_Br = 1 / self.n * self.shc_VB
        elif key == 'TJ':
            self.shc_TJ = kwargs['TJ']
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            self.shc_TJr = -self.n * (self.n + 1) / self.RI**2 * self.shc_TJ 
        elif key == 'Br':
            self.shc_Br = kwargs['Br']
            self.shc_VB = self.shc_Br / self.n
            self.shc_VJ = -self.RI / mu0 * (2 * self.n + 1) / (self.n + 1) * self.shc_VB
        elif key == 'TJr':
            self.shc_TJr = kwargs['TJr']
            self.shc_TJ = -1 /(self.n * (self.n + 1)) * self.shc_TJr * self.RI**2
            self.shc_TB = -mu0 / self.RI * self.shc_TJ
            print('not implemented: calculation of poloidal... -- also, check the factor RI**2!')
        else:
            raise Excpetion('This should not happen')



    def set_initial_condition(self, I2D_object):
        """ provide a

            If this is not called, initial condictions should be zero
        """
        print('not implemented. inital conditions will be zero')


    def set_FAC(self, FAC):
        """ Specify field-aligned current at self.theta, self.phi

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at self.theta and self.phi. 
                The values in the array have to match the corresponding coordinates
        """

        jr = FAC * self.sinI # get the radial component

        # SH coefficients for an expansion of T_Jr
        self.set_shc(TJr = np.linalg.lstsq(self.GTG, self.Gnum.T.dot(jr), rcond = 1e-3)[0])

        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_conductance(self, Hall, Pedersen):
        """ Specify Hall and Pedersen conductance at self.theta, self.phi


        """
        if Hall.size != Pedersen.size != self.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.SH = Hall
        self.SP = Pedersen
        self.etaP = Pedersen / (Hall**2 + Pedersen**2)
        self.etaH = Hall     / (Hall**2 + Pedersen**2)


    @default_3Dcoords
    def get_Br(self, r = None, theta = None, phi = None, deg = False):
        """ calculate Br 

        """
        return(self.Gplt.dot(self.shc_Br))


    @default_2Dcoords
    def get_JS(self, theta = None, phi = None, deg = False):
        """ calculate ionospheric sheet current

        """
        Je_V =  self.Gnum_th.dot(self.shc_VJ) # r cross grad(VJ) eastward component
        Js_V = -self.Gnum_ph.dot(self.shc_VJ) # r cross grad(VJ) southward component
        Je_T = -self.Gnum_ph.dot(self.shc_TJ) # -grad(VT) eastward component
        Js_T = -self.Gnum_th.dot(self.shc_TJ) # -grad(VT) southward component

        #print('TODO: fix the default stuff...')

        return(Js_V + Js_T, Je_V + Je_T)


    @default_2Dcoords
    def get_Jr(self, theta = None, phi = None, deg = False):
        """ calculate radial current

        """

        print('this must be fixed so that I can evaluate anywere')
        return self.Gplt.dot(self.shc_TJr)




    @default_2Dcoords
    def get_equivalent_current_function(self, theta = None, phi = None, deg = False):
        """ calculate equivalent current function

        """
        print('not implemented')


    @default_2Dcoords
    def get_Phi(self, theta = None, phi = None, deg = False):
        """ calculate electric potential

        """
        print('not implemented')


    @default_2Dcoords
    def get_W(self, theta = None, phi = None, deg = False):
        """ calculate the induction electric field scalar

        """
        print('not implemented')


    @default_2Dcoords
    def get_E(self, theta = None, phi = None, deg = False):
        """ calculate electric field

        """


        Jth, Jph = self.get_JS(theta = theta, phi = phi)


        Eth = self.etaP * (self.b00 * Jth + self.b01 * Jph) + self.etaH * ( self.br * Jph)
        Eph = self.etaP * (self.b10 * Jth + self.b11 * Jph) + self.etaH * (-self.br * Jth)

        return(Eth, Eph)

def run_pynamit(totalsteps = 200000, plotsteps = 200, dt = 5e-4, Nmax = 45, Mmax = 3, Ncs = 60, B0_type = 'dipole', fig_directory = './figs'):

    if B0_type == 'dipole':
        B0 = B0_dipole
    elif B0_type == 'radial':
        B0 = B0_radial
    else:
        raise Exception('Invalid B0_type')

    i2d = I2D(Nmax, Mmax, Ncs, B0 = B0)

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
    Wlevels = np.r_[-512.5:512.5:5]
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

        jr = i2d.Gplt.dot(i2d.shc_TJr)

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

        hall_plt = cs_interpolate(csp, 90 - i2d.theta, i2d.phi, hall, i2d.lat, i2d.lon)
        pede_plt = cs_interpolate(csp, 90 - i2d.theta, i2d.phi, pedersen, i2d.lat, i2d.lon)

        globalplot(i2d.lon, i2d.lat, hall_plt, noon_longitude = lon0, levels = c_levels, save = 'hall.png')
        globalplot(i2d.lon, i2d.lat, pede_plt, noon_longitude = lon0, levels = c_levels, save = 'pede.png')

        jr = i2d.Gplt.dot(i2d.shc_TJr)
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

            if count % plotsteps == 0:
                print(count, time, i2d.shc_Br[:3])
                fn = os.path.join(fig_directory, 'new_' + str(filecount).zfill(3) + '.png')
                filecount +=1
                title = 't = {:.3} s'.format(time)
                Br = i2d.get_Br()
                fig, paxn, paxs, axg =  globalplot(i2d.lon, i2d.lat, Br.reshape(i2d.lat.shape) , title = title, returnplot = True, 
                                                   levels = Blevels, cmap = 'bwr', noon_longitude = lon0, extend = 'both')
                W = i2d.Gplt.dot(i2d.shc_EW) * 1e-3

                GTE  = i2d.Gdf.T.dot(np.hstack( i2d.get_E()) )
                shc_Phi = i2d.GTGdf_inv.dot(GTE) # find coefficients for electric potential
                Phi = i2d.Gplt.dot(shc_Phi) * 1e-3


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


if __name__ == '__main__':
    run_pynamit()