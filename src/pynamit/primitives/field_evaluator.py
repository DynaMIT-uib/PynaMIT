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
            self._surface_to_apex = np.einsum('ijk,jlk->ilk', self.spherical_to_apex, self.surface_to_spherical, optimize = True)
            self._surface_to_apex = np.einsum('ijk->ikj', self._surface_to_apex, optimize = True)

        return self._surface_to_apex