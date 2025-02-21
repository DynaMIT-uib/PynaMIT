"""Field evaluator module.

This module contains the FieldEvaluator class for evaluating magnetic
field quantities on spatial grids.
"""

import numpy as np


class FieldEvaluator(object):
    """Class for evaluating magnetic field quantities on spatial grids.

    Computes magnetic field quantities and handles conversions between
    vector components in different coordinate systems and construction
    of vectors that are orthogonal to or parallel with the magnetic
    field.

    Attributes
    ----------
    field : Mainfield
        Main magnetic field model.
    grid : Grid
        Evaluation grid.
    r : float
        Evaluation radius.
    Br : ndarray
        Radial magnetic field component.
    Btheta : ndarray
        Colatitudinal magnetic field component.
    Bphi : ndarray
        Longitudinal magnetic field component.
    B_magnitude : ndarray
        Magnetic field magnitude.
    br, btheta, bphi : ndarray
        Unit vector components in spherical coordinates.
    e1r, e1theta, e1phi : ndarray
        e1 magnetic apex basis vector components.
    e2r, e2theta, e2phi : ndarray
        e2 magnetic apex basis vector components.
    e3r, e3theta, e3phi : ndarray
        e3 magnetic apex basis vector components.
    """

    def __init__(self, field, grid, r):
        """Initialize the FieldEvaluator object.

        Parameters
        ----------
        field : Mainfield
            Main magnetic field model.
        grid : Grid
            Spatial grid for evaluations.
        r : float
            Evaluation radius in meters.
        """
        self.field = field
        self.grid = grid
        self.r = r

    @property
    def grid_values(self):
        """Get magnetic field components on grid.

        Returns
        -------
        ndarray
            Magnetic field vector components (Br, Bθ, Bφ) with shape
            (3, N) where N is number of grid points.
        """
        if not hasattr(self, "_grid_values"):
            self._grid_values = np.vstack(
                self.field.get_B(self.r, self.grid.theta, self.grid.phi)
            )
        return self._grid_values

    @property
    def Br(self):
        """Radial component of the magnetic field.

        Returns
        -------
        array
            Radial component of the magnetic field.
        """
        return self.grid_values[0]

    @property
    def Btheta(self):
        """Theta component of the magnetic field.

        Returns
        -------
        array
            Theta component of the magnetic field.
        """
        return self.grid_values[1]

    @property
    def Bphi(self):
        """Phi component of the magnetic field.

        Returns
        -------
        array
            Phi component of the magnetic field.
        """
        return self.grid_values[2]

    @property
    def B_magnitude(self):
        """Magnitude of the magnetic field vector.

        Returns
        -------
        array
            Magnitude of the magnetic field vector.
        """
        if not hasattr(self, "_B_magnitude"):
            self._B_magnitude = np.linalg.norm(self.grid_values, axis=0)
        return self._B_magnitude

    @property
    def br(self):
        """Radial component of the magnetic field unit vector.

        Returns
        -------
        array
            Radial component of the magnetic field unit vector.
        """
        if not hasattr(self, "_br"):
            self._br = self.Br / self.B_magnitude
        return self._br

    @property
    def btheta(self):
        """Theta component of the magnetic field unit vector.

        Returns
        -------
        array
            Theta component of the magnetic field unit vector.
        """
        if not hasattr(self, "_btheta"):
            self._btheta = self.Btheta / self.B_magnitude
        return self._btheta

    @property
    def bphi(self):
        """Phi component of the magnetic field unit vector.

        Returns
        -------
        array
            Phi component of the magnetic field unit vector.
        """
        if not hasattr(self, "_bphi"):
            self._bphi = self.Bphi / self.B_magnitude
        return self._bphi

    @property
    def basevectors(self):
        """Base vectors of the magnetic field.

        Returns
        -------
        tuple of arrays
            Base vectors of the magnetic field.
        """
        if not hasattr(self, "_basevectors"):
            self._basevectors = self.field.basevectors(
                self.r, self.grid.theta, self.grid.phi
            )
        return self._basevectors

    @property
    def e1r(self):
        """Radial component of the e1 magnetic apex basis vector.

        Returns
        -------
        array
            Radial component of the e1 magnetic apex basis vector.
        """
        return self.basevectors[3][0]

    @property
    def e1theta(self):
        """Theta component of the e1 magnetic apex basis vector.

        Returns
        -------
        array
            Theta component of the e1 magnetic apex basis vector.
        """
        return self.basevectors[3][1]

    @property
    def e1phi(self):
        """Phi component of the e1 magnetic apex basis vector.

        Returns
        -------
        array
            Phi component of the e1 magnetic apex basis vector.
        """
        return self.basevectors[3][2]

    @property
    def e2r(self):
        """Radial component of the e2 magnetic apex basis vector.

        Returns
        -------
        array
            Radial component of the e2 magnetic apex basis vector.
        """
        return self.basevectors[4][0]

    @property
    def e2theta(self):
        """Theta component of the e2 magnetic apex basis vector.

        Returns
        -------
        array
            Theta component of the e2 magnetic apex basis vector.
        """
        return self.basevectors[4][1]

    @property
    def e2phi(self):
        """Phi component of the e2 magnetic apex basis vector.

        Returns
        -------
        array
            Phi component of the e2 magnetic apex basis vector.
        """
        return self.basevectors[4][2]

    @property
    def e3r(self):
        """Radial component of the e3 magnetic apex basis vector.

        Returns
        -------
        array
            Radial component of the e3 magnetic apex basis vector.
        """
        return self.basevectors[5][0]

    @property
    def e3theta(self):
        """Theta component of the e3 magnetic apex basis vector.

        Returns
        -------
        array
            Theta component of the e3 magnetic apex basis vector.
        """
        return self.basevectors[5][1]

    @property
    def e3phi(self):
        """Phi component of the e3 magnetic apex basis vector.

        Returns
        -------
        array
            Phi component of the e3 magnetic apex basis vector.
        """
        return self.basevectors[5][2]

    @property
    def horizontal_to_field_orthogonal(self):
        """Matrix mapping horizontal to field-orthogonal.

        Returns
        -------
        array
            Matrix that maps the two components of a horizontal vector
            field to the three spherical coordinate components of the
            corresponding vector field that is orthogonal to the
            magnetic field.
        """
        if not hasattr(self, "_horizontal_to_field_orthogonal"):
            self._horizontal_to_field_orthogonal = np.array(
                [
                    [-self.btheta / self.br, -self.bphi / self.br],
                    [np.ones(self.grid.size), np.zeros(self.grid.size)],
                    [np.zeros(self.grid.size), np.ones(self.grid.size)],
                ]
            )

        return self._horizontal_to_field_orthogonal

    @property
    def field_orthogonal_to_apex(self):
        """Matrix mapping field-orthogonal to apex.

        Returns
        -------
        array
            Matrix that maps the three spherical coordinate components
            of a vector field that is orthogonal to the magnetic field
            to the two corresponding magnetic apex coordinate
            components.
        """
        if not hasattr(self, "_field_orthogonal_to_apex"):
            self._field_orthogonal_to_apex = np.array(
                [
                    [self.e1r, self.e1theta, self.e1phi],
                    [self.e2r, self.e2theta, self.e2phi],
                ]
            )

        return self._field_orthogonal_to_apex

    @property
    def horizontal_to_apex(self):
        """Matrix mapping horizontal to apex coordinates.

        Returns
        -------
        array
            Matrix that maps the two components of a horizontal vector
            field to the two magnetic apex coordinate components of the
            corresponding vector field that is orthogonal to the
            magnetic field.
        """
        if not hasattr(self, "_horizontal_to_apex"):
            self._horizontal_to_apex = np.einsum(
                "ijk,jlk->ilk",
                self.field_orthogonal_to_apex,
                self.horizontal_to_field_orthogonal,
                optimize=True,
            )

        return self._horizontal_to_apex

    @property
    def radial_to_field_parallel(self):
        """Matrix mapping radial to field-parallel.

        Returns
        -------
        array
            Matrix that maps the component of a radial vector field to
            the three spherical coordinate components of the
            corresponding vector field that is parallel with the
            magnetic field.
        """
        if not hasattr(self, "_radial_to_field_parallel"):
            self._radial_to_field_parallel = np.array(
                [
                    [np.ones(self.grid.size)],
                    [self.btheta / self.br],
                    [self.bphi / self.br],
                ]
            )

        return self._radial_to_field_parallel

    @property
    def field_parallel_to_apex(self):
        """Matrix mapping field-parallel to apex.

        Returns
        -------
        array
            Matrix that maps the three spherical coordinate components
            of a vector field that is parallel with the magnetic field
            to the corresponding magnetic apex coordinate component of
            the field.
        """
        if not hasattr(self, "_field_parallel_to_apex"):
            self._field_parallel_to_apex = np.array(
                [[self.e3r, self.e3theta, self.e3phi]]
            )

        return self._field_parallel_to_apex

    @property
    def radial_to_apex(self):
        """Matrix mapping radial to field-parallel apex.

        Returns
        -------
        array
            Matrix that maps the component of a radial vector field to
            the magnetic apex coordinate component of the corresponding
            vector field that is parallel with the magnetic field.
        """
        if not hasattr(self, "_radial_to_apex"):
            self._radial_to_apex = np.einsum(
                "ijk,jlk->ilk",
                self.field_parallel_to_apex,
                self.radial_to_field_parallel,
                optimize=True,
            )

        return self._radial_to_apex
