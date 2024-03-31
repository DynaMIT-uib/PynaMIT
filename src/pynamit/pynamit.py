import numpy as np
import dipole
from decorators import default_2Dcoords, default_3Dcoords
from sh_utils import get_G
import sys
import os

# import cubedsphere submodule
cs_path = os.path.join(os.path.dirname(__file__), 'cubedsphere')
sys.path.insert(0, cs_path)
import cubedsphere
csp = cubedsphere.CSprojection() # cubed sphere projection object

RE = 6371.2e3
mu0 = 4 * np.pi * 1e-7

def B0_default(r, theta, phi):
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    size = r.size
    zeros, ones = np.zeros(size), np.ones(size)
    return(np.vstack((ones, zeros, zeros)))




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

        if B0 is None:
            self.B0 = B0_default

        # Define CS grid used for SH analysis and gradient calculations
        k, i, j = csp.get_gridpoints(Ncs)
        xi, eta = csp.xi(i, Ncs), csp.eta(j, Ncs)
        _, self.theta, self.phi = csp.cube2spherical(xi, eta, k, deg = True)
        self.theta, self.phi = self.theta.flatten(), self.phi.flatten()
        self.Dxi, self.Deta = csp.get_Diff(Ncs, coordinate = 'both') # differentiation matrices in xi and eta directions
        self.g  = csp.get_metric_tensor(xi, eta, 1, covariant = True) 
        self.Ps = csp.get_Ps(xi, eta, 1, k)                           # matrices to convert from u^east, u^north, u^up to u^1 ,u^2, u^3 (A1 in Yin)
        self.Qi = csp.get_Q(90 - self.theta, self.RI, inverse = True) # matrices to convert from physical north, east, radial to u^east, u^north, u^up (A1 in Yin)

        # get magnetic field unit vectors at CS grid:
        B = np.vstack(self.B0(self.RI, self.theta, self.phi))
        self.br, self.btheta, self.bphi = B / np.linalg.norm(B, axis = 0)
        self.sinI = self.br / np.sqrt(self.btheta**2 + self.bphi**2 + self.br**2) # sin(inclination)

        # Define grid used for plotting 
        lat, lon = np.linspace(-89.9, 89.9, Ncs * 2), np.linspace(-180, 180, Ncs * 4)
        self.lat, self.lon = np.meshgrid(lat, lon)

        # Define matrices for surface spherical harmonics
        print('TODO: it would be nice to have access to n without this stupid syntax. write a class?')
        self.Gnum, self.n = get_G(90 - self.theta, self.phi, Nmax, Mmax, a = self.RI, return_n = True)
        self.Gnum_ph      = get_G(90 - self.theta, self.phi, Nmax, Mmax, a = self.RI, derivative = 'phi'  )
        self.Gnum_th      = get_G(90 - self.theta, self.phi, Nmax, Mmax, a = self.RI, derivative = 'theta')
        self.Gplt         = get_G(self.lat, self.lon, Nmax, Mmax, a = self.RI)
        self.Gplt_ph      = get_G(self.lat, self.lon, Nmax, Mmax, a = self.RI, derivative = 'phi'  )
        self.Gplt_th      = get_G(self.lat, self.lon, Nmax, Mmax, a = self.RI, derivative = 'theta')

        # Initialize the spherical harmonic coefficients
        self.shc_VB, self.shc_TB = np.zeros(self.Gnum.shape[1]), np.zeros(self.Gnum.shape[1])
        self.shc_B2J()
        self.shc_VB2Br()
        print('TODO: Add checks that the different coefficients are defined')

        # Pre-calculate GTG and report condition number
        self.GTG = self.Gnum.T.dot(self.Gnum)

        # Pre-calculate matrix for retrieval of VB coefficients based on Br
        self.GBr = self.Gnum * self.n.reshape((1, -1))
        self.GTG_Br = self.GBr.T.dot(self.GBr)
        self.GTG_Br_inv = np.linalg.pinv(self.GTG_Br)

        self.cond_FAC = np.linalg.cond(self.GTG)
        print('The condition number for the FAC SHA matrix is {:.1f}'.format(self.cond_FAC))

        self.cond_Br = np.linalg.cond(self.GTG_Br)
        print('The condition number for the Br SHA matrix is {:.1f}'.format(self.cond_Br))


    def evolve_Br(self, dt):
        """ evolve Br in time """

        Eth, Eph = self.get_E()
        u1, u2, u3 = self.sph_to_contravariant_cs(np.zeros_like(Eph), Eth, Eph)
        curlEr = self.curlr(u1, u2)

        Br = self.GBr.dot(self.shc_VB) - dt * curlEr
        self.shc_VB = self.GTG_Br_inv.dot(self.GBr.T.dot(Br))
        self.shc_B2J() # update current coefficients


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
        
        sqrtg = np.sqrt(cubedsphere.arrayutils.get_3D_determinants(self.g))
        g12 = self.g[:, 0, 1]
        g22 = self.g[:, 1, 1]
        g11 = self.g[:, 0, 0]

        return( 1/sqrtg * ( self.Dxi.dot(g12 * u1 + g22 * u2) - self.Deta.dot(g11 * u1 + g12 *u2) ) )


    def shc_B2J(self):
        """ Calculate the spherical harmonic coefficients for the currents
            based on the spherical harmonic coefficients of the magnetic field
        """
        self.shc_VJ =  self.RI / mu0 * (2 * self.n + 1) / (self.n + 1) * self.shc_VB
        self.shc_TJ = -self.RI / mu0 * self.shc_TB

    def shc_J2B(self):
        """ Calculate the spherical harmonic coefficients for the magnetic field
            based on the spherical harmonic coefficients of the current
        """
        self.shc_VB =  mu0 / self.RI * (self.n + 1) / (2 * self.n + 1) * self.shc_VJ
        self.shc_TB = -mu0 / self.RI * self.shc_TJ

    def shc_Jr2TJTB(self):
        """ Calculate the spherical harmonic coefficient for TJ and TB 
            based on the spherical harmonic coefficients of TJr.
        """
        self.shc_TJ = -self.n * (self.n + 1) * self.shc_TJr * self.RI
        self.shc_J2B()
        print("!!!: I've multiplied by R just because the values look more reasonable. Check math !!!")



    def shc_VB2Br(self):
        """ Calculate the spherical harmonic coefficients for Br,
            based on the spherical harmonic coefficients of the magnetic field
        """
        self.shc_Br =  1 / self.n * self.shc_VB


    def shc_Br2VB(self):
        """ Calculate the spherical harmonic coefficients for Br,
            based on the spherical harmonic coefficients of the magnetic field
        """
        self.shc_VB =  self.n * self.shc_Br


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
        self.shc_TJr = np.linalg.lstsq(self.GTG, self.Gnum.T.dot(jr), rcond = 1e-3)[0] 
        self.shc_Jr2TJTB() # update the magnetic field coefficients

        print('Note to self: Remember to write a function that compares the AMPS SH coefficient to the ones derived here')


    def set_counductance(self, Hall, Pedersen):
        """ Specify Hall and Pedersen conductance at self.theta, self.phi


        """
        if Hall.size != Pedersen.size != self.theta.size:
            raise Exception('Conductances must match phi and theta')

        self.SH = Hall
        self.SP = Pedersen
        self.etaP = Pedersen ** 2 / np.sqrt(Hall**2 + Pedersen**2)
        self.etaH = Hall     ** 2 / np.sqrt(Hall**2 + Pedersen**2)


    @default_3Dcoords
    def get_Br(self, r = None, theta = None, phi = None, deg = False):
        """ calculate Br 

        """
        return((self.Gplt * self.n.reshape((1, -1))).dot(self.shc_VB))


    @default_2Dcoords
    def get_JS(self, theta = None, phi = None, deg = False):
        """ calculate ionospheric sheet current

        """
        Je_V = self.Gnum_ph.dot(self.shc_VJ)
        Js_V = self.Gnum_th.dot(self.shc_VJ)
        Je_T = self.Gnum_ph.dot(self.shc_TJ)
        Js_T = self.Gnum_th.dot(self.shc_TJ)

        #print('TODO: fix the default stuff...')

        return(Je_V + Je_T, Js_V + Js_T)


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
        Jph, Jth = self.get_JS(theta = theta, phi = phi)
        Eph = self.etaP * Jph - self.etaH * Jth
        Eth = self.etaP * Jth + self.etaH * Jph

        return(Eph, Eth)




if __name__ == '__main__':
    i2d = I2D(45, 3, Ncs = 60)

    import pyamps
    from visualization import globalplot
    import matplotlib.pyplot as plt
    from lompe import conductance
    import dipole
    import datetime

    # specify a time and Kp (for conductance):
    date = datetime.datetime(2001, 5, 12, 21, 45)
    Kp   = 5
    d = dipole.Dipole(date.year)

    # noon longitude
    lon0 = d.mlt2mlon(12, date)

    hall, pedersen = conductance.hardy_EUV(i2d.phi, 90 - i2d.theta, Kp, date, starlight = 1, dipole = True)
    i2d.set_counductance(hall, pedersen)

    a = pyamps.AMPS(300, 0, -4, 20, 100)
    fac = a.get_upward_current(mlat = 90 - i2d.theta, mlt = d.mlon2mlt(i2d.phi, date)) * 1e-6
    fac[np.abs(90 - i2d.theta) < 50] = 0 # filter low latitude FACs

    i2d.set_FAC(fac)

    jr = i2d.get_Jr()

    print('bug in cartopy makes it impossible to not center levels at zero... replace when cartopy has been improved')
    #globalplot(i2d.lon, i2d.lat, jr.reshape(i2d.lat.shape), 
    #           levels = levels, cmap = 'bwr', central_longitude = lon0)

    #globalplot(i2d.phi, 90 - i2d.theta, i2d.SH, vmin = 0, vmax = 20, cmap = 'viridis', scatter = True, central_longitude = lon0)

    dt = 1e-4 # time step in seconds    
    coeffs = []
    count = 0
    while True:

        i2d.evolve_Br(dt)
        coeffs.append(i2d.shc_VB)
        count += 1
        print(count)

        if count > 20000:
            break


    Br = i2d.get_Br()
    levels = np.linspace(-100, 100, 21)
    globalplot(i2d.lon, i2d.lat, Br.reshape(i2d.lat.shape) , 
               levels = levels, cmap = 'bwr', central_longitude = lon0)


    #globalplot(i2d.phi, 90 - i2d.theta, curlr, vmin = -2.4e-7, vmax = 2.4e-7, cmap = 'bwr', scatter = True, central_longitude = lon0)

    i2d.get_Br()
    i2d.get_W()


