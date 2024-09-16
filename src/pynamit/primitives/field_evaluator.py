""" Field evaluator module. """

import numpy as np

class FieldEvaluator:
    """ Field evaluator class. """

    def __init__(self, field, grid, r):
        self.field = field
        self.grid = grid
        self.r = r

    @property
    def grid_values(self):
        """ Magnetic field vector. """

        if not hasattr(self, '_grid_values'):
            self._grid_values = np.vstack(self.field.get_B(self.r, self.grid.theta, self.grid.phi))
        return self._grid_values

    @property
    def Br(self):
        """ Radial component of the magnetic field. """

        return self.grid_values[0]

    @property
    def Btheta(self):
        """ Theta component of the magnetic field. """

        return self.grid_values[1]

    @property
    def Bphi(self):
        """ Phi component of the magnetic field. """

        return self.grid_values[2]

    @property
    def B_magnitude(self):
        """ Magnitude of the magnetic field vector. """

        if not hasattr(self, '_B_magnitude'):
            self._B_magnitude = np.linalg.norm(self.grid_values, axis = 0)
        return self._B_magnitude

    @property
    def br(self):
        """ Radial component of the magnetic field unit vector. """

        if not hasattr(self, '_br'):
            self._br = self.Br / self.B_magnitude
        return self._br

    @property
    def btheta(self):
        """ Theta component of the magnetic field unit vector. """

        if not hasattr(self, '_btheta'):
            self._btheta = self.Btheta / self.B_magnitude
        return self._btheta

    @property
    def bphi(self):
        """ Phi component of the magnetic field unit vector. """

        if not hasattr(self, '_bphi'):
            self._bphi = self.Bphi / self.B_magnitude
        return self._bphi

    @property
    def basevectors(self):
        """ Base vectors of the magnetic field. """

        if not hasattr(self, '_basevectors'):
            self._basevectors = self.field.basevectors(self.r, self.grid.theta, self.grid.phi)
        return self._basevectors

    @property
    def e1r(self):
        """ Radial component of the e1 basis vector. """

        return self.basevectors[3][0]

    @property
    def e1t(self):
        """ Theta component of the e1 basis vector. """

        return self.basevectors[3][1]

    @property
    def e1p(self):
        """ Phi component of the e1 basis vector. """

        return self.basevectors[3][2]

    @property
    def e2r(self):
        """ Radial component of the e2 basis vector. """

        return self.basevectors[4][0]

    @property
    def e2t(self):
        """ Theta component of the e2 basis vector. """

        return self.basevectors[4][1]

    @property
    def e2p(self):
        """ Phi component of the e2 basis vector. """

        return self.basevectors[4][2]


    @property
    def surface_to_spherical(self):
        """
        Create the matrix that takes the surface parts of a vector in the
        spherical coordinate system and returns the components in the
        radial direction.

        """

        if not hasattr(self, '_surface_to_spherical'):
            self._surface_to_spherical = np.array([[-self.Btheta / self.Br,   -self.Bphi / self.Br],
                                                   [np.ones(self.grid.size),  np.zeros(self.grid.size)],
                                                   [np.zeros(self.grid.size), np.ones(self.grid.size)]])

        return self._surface_to_spherical


    @property
    def spherical_to_apex(self):
        """
        Create the matrix that takes the spherical parts of a vector in
        the spherical coordinate system and returns components orthogonal
        to the magnetic field in the apex coordinate system.

        """

        if not hasattr(self, '_spherical_to_apex'):
            self._spherical_to_apex = np.array([[self.e1r, self.e1t, self.e1p],
                                                [self.e2r, self.e2t, self.e2p]])

        return self._spherical_to_apex


    @property
    def surface_to_apex(self):
        """
        Create the matrix that takes the surface parts of a vector in the
        spherical coordinate system and returns components orthogonal
        to the magnetic field in the apex coordinate system.

        """

        if not hasattr(self, '_surface_to_apex'):
            self._surface_to_apex = np.einsum('ilk,ljk->ijk', self.spherical_to_apex, self.surface_to_spherical)
            #self._surface_to_apex = np.array([[self.spherical_to_apex[0][0] * self.surface_to_spherical[0][0] + self.spherical_to_apex[0][1] * self.surface_to_spherical[1][0] + self.spherical_to_apex[0][2] * self.surface_to_spherical[2][0],
            #                                  self.spherical_to_apex[0][0] * self.surface_to_spherical[0][1] + self.spherical_to_apex[0][1] * self.surface_to_spherical[1][1] + self.spherical_to_apex[0][2] * self.surface_to_spherical[2][1]],
            #                                 [self.spherical_to_apex[1][0] * self.surface_to_spherical[0][0] + self.spherical_to_apex[1][1] * self.surface_to_spherical[1][0] + self.spherical_to_apex[1][2] * self.surface_to_spherical[2][0],
            #                                  self.spherical_to_apex[1][0] * self.surface_to_spherical[0][1] + self.spherical_to_apex[1][1] * self.surface_to_spherical[1][1] + self.spherical_to_apex[1][2] * self.surface_to_spherical[2][1]]])

        return self._surface_to_apex


    @property
    def aeP(self):
        """ a e P matrix. """

        if not hasattr(self, '_aeP'):
            alpha11_eP = self.bphi**2*self.e1t - self.bphi*self.btheta*self.e1p + self.br**2*self.e1t - self.br*self.btheta*self.e1r
            alpha12_eP = -self.bphi*self.br*self.e1r - self.bphi*self.btheta*self.e1t + self.br**2*self.e1p + self.btheta**2*self.e1p
            alpha21_eP = self.bphi**2*self.e2t - self.bphi*self.btheta*self.e2p + self.br**2*self.e2t - self.br*self.btheta*self.e2r
            alpha22_eP = -self.bphi*self.br*self.e2r - self.bphi*self.btheta*self.e2t + self.br**2*self.e2p + self.btheta**2*self.e2p
            self._aeP = np.vstack((np.hstack((np.diag(alpha11_eP), np.diag(alpha12_eP))),
                                   np.hstack((np.diag(alpha21_eP), np.diag(alpha22_eP)))))
        return self._aeP

    @property
    def aeH(self):
        """ a e H matrix. """

        if not hasattr(self, '_aeH'):
            alpha11_eH = self.bphi*self.e1r - self.br*self.e1p
            alpha12_eH = self.br*self.e1t - self.btheta*self.e1r
            alpha21_eH = self.bphi*self.e2r - self.br*self.e2p
            alpha22_eH = self.br*self.e2t - self.btheta*self.e2r
            self._aeH = np.vstack((np.hstack((np.diag(alpha11_eH), np.diag(alpha12_eH))),
                                   np.hstack((np.diag(alpha21_eH), np.diag(alpha22_eH)))))
        return self._aeH

    @property
    def aut(self):
        """ a u t matrix. """

        if not hasattr(self, '_aut'):
            alpha13_ut = -self.Br*self.bphi*self.e1r/self.br + self.Br*self.e1p
            alpha23_ut = -self.Br*self.bphi*self.e2r/self.br + self.Br*self.e2p
            self._aut = np.hstack((alpha13_ut, alpha23_ut))
        return self._aut

    @property
    def aup(self):
        """ a u p matrix. """

        if not hasattr(self, '_aup'):
            alpha13_up = -self.Br*self.e1t + self.Br*self.btheta*self.e1r/self.br
            alpha23_up = -self.Br*self.e2t + self.Br*self.btheta*self.e2r/self.br
            self._aup = np.hstack((alpha13_up, alpha23_up))
        return self._aup