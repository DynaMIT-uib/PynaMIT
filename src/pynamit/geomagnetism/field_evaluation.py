"""Magnetic-field evaluation and local coordinate maps."""

from functools import cached_property

from kompe.math import get_array_module


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

    def __repr__(self):
        """Summarize the field model and evaluation surface."""
        return (
            f"MagneticFieldEvaluation(main_field={self.main_field!r}, "
            f"grid={self.grid!r}, radius={float(self.radius):g})"
        )

    @cached_property
    def components(self):
        """Return ``(Br, Btheta, Bphi)`` with shape ``(3, N)``."""
        components = self.main_field.field_components(self.radius, self.grid.theta, self.grid.phi)
        xp = get_array_module(*components)
        return xp.stack([xp.asarray(component) for component in components], axis=0)

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
        xp = get_array_module(self.components)
        return xp.linalg.norm(self.components, axis=0)

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
        vectors = self.main_field.basis_vectors(self.radius, self.grid.theta, self.grid.phi)
        xp = get_array_module(*vectors)
        return tuple(xp.asarray(vector) for vector in vectors)

    @cached_property
    def horizontal_to_field_orthogonal(self):
        """Map horizontal components to a field-orthogonal 3-vector."""
        xp = get_array_module(self.unit_br, self.unit_btheta, self.unit_bphi)
        ones = xp.ones(self.grid.size)
        zeros = xp.zeros(self.grid.size)
        return xp.stack(
            (
                xp.stack((-self.unit_btheta / self.unit_br, -self.unit_bphi / self.unit_br)),
                xp.stack((ones, zeros)),
                xp.stack((zeros, ones)),
            )
        )

    @cached_property
    def field_orthogonal_to_apex(self):
        """Map a field-orthogonal 3-vector to two apex components."""
        e1, e2 = self.basis_vectors[3:5]
        return get_array_module(e1, e2).stack((e1, e2))

    @cached_property
    def horizontal_to_apex(self):
        """Map horizontal components to orthogonal apex components."""
        xp = get_array_module(self.field_orthogonal_to_apex, self.horizontal_to_field_orthogonal)
        return xp.einsum(
            "ijk,jlk->ilk",
            self.field_orthogonal_to_apex,
            self.horizontal_to_field_orthogonal,
            optimize=True,
        )

    @cached_property
    def radial_to_field_parallel(self):
        """Map a radial component to a field-parallel 3-vector."""
        xp = get_array_module(self.unit_br, self.unit_btheta, self.unit_bphi)
        return xp.stack(
            (
                xp.ones(self.grid.size),
                self.unit_btheta / self.unit_br,
                self.unit_bphi / self.unit_br,
            )
        )[:, xp.newaxis, :]

    @cached_property
    def field_parallel_to_apex(self):
        """Map a field-parallel 3-vector to its apex component."""
        d3 = self.basis_vectors[2]
        xp = get_array_module(d3)
        return xp.asarray(d3)[xp.newaxis, ...]

    @cached_property
    def radial_to_apex(self):
        """Map a radial component to its parallel apex component."""
        xp = get_array_module(self.field_parallel_to_apex, self.radial_to_field_parallel)
        return xp.einsum(
            "ijk,jlk->ilk",
            self.field_parallel_to_apex,
            self.radial_to_field_parallel,
            optimize=True,
        )
