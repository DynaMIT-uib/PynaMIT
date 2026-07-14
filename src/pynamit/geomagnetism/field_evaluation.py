"""Magnetic-field evaluation and local coordinate maps."""

from functools import cached_property

import numpy as np


class MagneticFieldEvaluation:
    """Evaluated magnetic field and local maps on one spherical grid.

    The evaluator caches field values, unit-vector components, apex
    basis vectors, and maps used by the simulation geometry. Individual
    apex vector components remain available through ``basis_vectors``.
    """

    def __init__(self, main_field, grid, radius):
        """Bind a main-field model to a grid and radius in meters."""
        self.main_field = main_field
        self.grid = grid
        self.radius = radius

    @cached_property
    def components(self):
        """Return ``(Br, Btheta, Bphi)`` with shape ``(3, N)``."""
        return np.stack(
            self.main_field.field_components(self.radius, self.grid.theta, self.grid.phi), axis=0
        )

    @property
    def Br(self):
        """Return the radial magnetic-field component."""
        return self.components[0]

    @property
    def Btheta(self):
        """Return the colatitudinal magnetic-field component."""
        return self.components[1]

    @property
    def Bphi(self):
        """Return the azimuthal magnetic-field component."""
        return self.components[2]

    @cached_property
    def magnitude(self):
        """Return magnetic-field magnitude on the grid."""
        return np.linalg.norm(self.components, axis=0)

    @cached_property
    def unit_br(self):
        """Return the radial component of the magnetic unit vector."""
        return self.Br / self.magnitude

    @cached_property
    def unit_btheta(self):
        """Return the colatitudinal magnetic unit-vector component."""
        return self.Btheta / self.magnitude

    @cached_property
    def unit_bphi(self):
        """Return the azimuthal magnetic unit-vector component."""
        return self.Bphi / self.magnitude

    @cached_property
    def basis_vectors(self):
        """Return the six magnetic apex basis vectors."""
        return self.main_field.basis_vectors(self.radius, self.grid.theta, self.grid.phi)

    @cached_property
    def horizontal_to_field_orthogonal(self):
        """Map horizontal components to a field-orthogonal 3-vector."""
        ones = np.ones(self.grid.size)
        zeros = np.zeros(self.grid.size)
        return np.array(
            [
                [-self.unit_btheta / self.unit_br, -self.unit_bphi / self.unit_br],
                [ones, zeros],
                [zeros, ones],
            ]
        )

    @cached_property
    def field_orthogonal_to_apex(self):
        """Map a field-orthogonal 3-vector to two apex components."""
        e1, e2 = self.basis_vectors[3:5]
        return np.stack((e1, e2))

    @cached_property
    def horizontal_to_apex(self):
        """Map horizontal components to orthogonal apex components."""
        return np.einsum(
            "ijk,jlk->ilk",
            self.field_orthogonal_to_apex,
            self.horizontal_to_field_orthogonal,
            optimize=True,
        )

    @cached_property
    def radial_to_field_parallel(self):
        """Map a radial component to a field-parallel 3-vector."""
        return np.array(
            [
                [np.ones(self.grid.size)],
                [self.unit_btheta / self.unit_br],
                [self.unit_bphi / self.unit_br],
            ]
        )

    @cached_property
    def field_parallel_to_apex(self):
        """Map a field-parallel 3-vector to its apex component."""
        return np.asarray(self.basis_vectors[2])[np.newaxis, ...]

    @cached_property
    def radial_to_apex(self):
        """Map a radial component to its parallel apex component."""
        return np.einsum(
            "ijk,jlk->ilk",
            self.field_parallel_to_apex,
            self.radial_to_field_parallel,
            optimize=True,
        )
