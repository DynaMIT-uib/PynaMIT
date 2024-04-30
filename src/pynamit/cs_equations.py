import numpy as np

class cs_equations(object):
    """ Unused (for now) equations related to cubed sphere coordinates.

    """

    def __init__(self, csp, RI):


        #self.Dxi, self.Deta = csp.get_Diff(Ncs, coordinate = 'both') # differentiation matrices in xi and eta directions
        self.g  = csp.g # csp.get_metric_tensor(xi, eta, 1, covariant = True)
        self.sqrtg = np.sqrt(csp.detg) #np.sqrt(cubedsphere.arrayutils.get_3D_determinants(self.g))
        self.Ps = csp.get_Ps(csp.arr_xi, csp.arr_eta, 1, csp.arr_block)                           # matrices to convert from u^east, u^north, u^up to u^1 ,u^2, u^3 (A1 in Yin)
        self.Qi = csp.get_Q(90 - csp.arr_theta, self.RI, inverse = True) # matrices to convert from physical north, east, radial to u^east, u^north, u^up (A1 in Yin)

        self.g12 = self.g[:, 0, 1]
        self.g22 = self.g[:, 1, 1]
        self.g11 = self.g[:, 0, 0]


    def curlr(self, u1, u2):
        """
        Construct a matrix that calculates the radial curl using B6 in Yin
        et al.

        """
        


        return( 1/self.sqrtg * ( self.Dxi.dot(self.g12 * u1 + self.g22 * u2) - 
                                 self.Deta.dot(self.g11 * u1 + self.g12 *u2) ) )
    

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