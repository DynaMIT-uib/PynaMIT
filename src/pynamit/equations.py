import numpy as np

class equations(object):
    """
    Class to define the equations of motion for the system.
    """

    def __init__(self, mainfield, g, sqrtg, Ps, Qi):
        """
        Initialize the equations of motion.

        Parameters
        ----------
        #system : object
        #    The system object.
        """
        #self.system = system

        self.mainfield = mainfield
        self.Ps = Ps
        self.Qi = Qi

        self.g = g
        self.sqrtg = sqrtg
        self.g12 = self.g[:, 0, 1]
        self.g22 = self.g[:, 1, 1]
        self.g11 = self.g[:, 0, 0]


    def curlr(self, u1, u2):
        """
        Construct a matrix that calculates the radial curl using B6 in
        Yin et al.

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