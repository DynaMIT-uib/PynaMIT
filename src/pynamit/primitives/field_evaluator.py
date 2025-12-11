"""Field evaluator module.

This module contains the FieldEvaluator class for evaluating magnetic
field quantities on spatial grids.
"""

from __future__ import annotations
from functools import cached_property
from typing import Any, Tuple

import numpy as np
from pynamit.simulation.mainfield import Mainfield
from pynamit.primitives.grid import Grid


class FieldEvaluator:
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

    def __init__(self, field: Mainfield, grid: Grid, r: float) -> None:
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

    @cached_property
    def grid_values(self) -> np.ndarray:
        """Get magnetic field components on grid.

        Returns
        -------
        ndarray
            Magnetic field vector components (Br, Bθ, Bφ) with shape
            (3, N) where N is number of grid points.
        """
        return np.vstack(self.field.get_B(self.r, self.grid.theta, self.grid.phi))

    @property
    def Br(self) -> np.ndarray:
        """Radial component of the magnetic field."""
        return self.grid_values[0]

    @property
    def Btheta(self) -> np.ndarray:
        """Theta component of the magnetic field."""
        return self.grid_values[1]

    @property
    def Bphi(self) -> np.ndarray:
        """Phi component of the magnetic field."""
        return self.grid_values[2]

    @cached_property
    def B_magnitude(self) -> np.ndarray:
        """Magnitude of the magnetic field vector."""
        return np.linalg.norm(self.grid_values, axis=0)

    @cached_property
    def br(self) -> np.ndarray:
        """Radial component of the magnetic field unit vector."""
        return self.Br / self.B_magnitude

    @cached_property
    def btheta(self) -> np.ndarray:
        """Theta component of the magnetic field unit vector."""
        return self.Btheta / self.B_magnitude

    @cached_property
    def bphi(self) -> np.ndarray:
        """Phi component of the magnetic field unit vector."""
        return self.Bphi / self.B_magnitude

    @cached_property
    def basis_vectors(self) -> Tuple[np.ndarray, ...]:
        """Basis vectors of the magnetic field.

        Returns
        -------
        tuple of arrays
            Basis vectors of the magnetic field.
        """
        return self.field.basis_vectors(self.r, self.grid.theta, self.grid.phi)

    @property
    def d1r(self) -> np.ndarray:
        """Radial component of the d1 magnetic apex basis vector."""
        return self.basis_vectors[0][0]

    @property
    def d1theta(self) -> np.ndarray:
        """Theta component of the d1 magnetic apex basis vector."""
        return self.basis_vectors[0][1]

    @property
    def d1phi(self) -> np.ndarray:
        """Phi component of the d1 magnetic apex basis vector."""
        return self.basis_vectors[0][2]

    @property
    def d2r(self) -> np.ndarray:
        """Radial component of the d2 magnetic apex basis vector."""
        return self.basis_vectors[1][0]

    @property
    def d2theta(self) -> np.ndarray:
        """Theta component of the d2 magnetic apex basis vector."""
        return self.basis_vectors[1][1]

    @property
    def d2phi(self) -> np.ndarray:
        """Phi component of the d2 magnetic apex basis vector."""
        return self.basis_vectors[1][2]

    @property
    def d3r(self) -> np.ndarray:
        """Radial component of the d3 magnetic apex basis vector."""
        return self.basis_vectors[2][0]

    @property
    def d3theta(self) -> np.ndarray:
        """Theta component of the d3 magnetic apex basis vector."""
        return self.basis_vectors[2][1]

    @property
    def d3phi(self) -> np.ndarray:
        """Phi component of the d3 magnetic apex basis vector."""
        return self.basis_vectors[2][2]

    @property
    def e1r(self) -> np.ndarray:
        """Radial component of the e1 magnetic apex basis vector."""
        return self.basis_vectors[3][0]

    @property
    def e1theta(self) -> np.ndarray:
        """Theta component of the e1 magnetic apex basis vector."""
        return self.basis_vectors[3][1]

    @property
    def e1phi(self) -> np.ndarray:
        """Phi component of the e1 magnetic apex basis vector."""
        return self.basis_vectors[3][2]

    @property
    def e2r(self) -> np.ndarray:
        """Radial component of the e2 magnetic apex basis vector."""
        return self.basis_vectors[4][0]

    @property
    def e2theta(self) -> np.ndarray:
        """Theta component of the e2 magnetic apex basis vector."""
        return self.basis_vectors[4][1]

    @property
    def e2phi(self) -> np.ndarray:
        """Phi component of the e2 magnetic apex basis vector."""
        return self.basis_vectors[4][2]

    @property
    def e3r(self) -> np.ndarray:
        """Radial component of the e3 magnetic apex basis vector."""
        return self.basis_vectors[5][0]

    @property
    def e3theta(self) -> np.ndarray:
        """Theta component of the e3 magnetic apex basis vector."""
        return self.basis_vectors[5][1]

    @property
    def e3phi(self) -> np.ndarray:
        """Phi component of the e3 magnetic apex basis vector."""
        return self.basis_vectors[5][2]

    @cached_property
    def horizontal_to_field_orthogonal(self) -> np.ndarray:
        """Matrix mapping horizontal to field-orthogonal values."""
        return np.array(
            [
                [-self.btheta / self.br, -self.bphi / self.br],
                [np.ones(self.grid.size), np.zeros(self.grid.size)],
                [np.zeros(self.grid.size), np.ones(self.grid.size)],
            ]
        )

    @cached_property
    def field_orthogonal_to_apex(self) -> np.ndarray:
        """Matrix mapping field-orthogonal to apex coordinates."""
        return np.array(
            [[self.e1r, self.e1theta, self.e1phi], [self.e2r, self.e2theta, self.e2phi]]
        )

    @cached_property
    def horizontal_to_apex(self) -> np.ndarray:
        """Matrix mapping horizontal to apex coordinates."""
        return np.einsum(
            "ijk,jlk->ilk",
            self.field_orthogonal_to_apex,
            self.horizontal_to_field_orthogonal,
            optimize=True,
        )

    @cached_property
    def radial_to_field_parallel(self) -> np.ndarray:
        """Matrix mapping radial to field-parallel values."""
        return np.array(
            [[np.ones(self.grid.size)], [self.btheta / self.br], [self.bphi / self.br]]
        )

    @cached_property
    def field_parallel_to_apex(self) -> np.ndarray:
        """Matrix mapping field-parallel to apex coordinates."""
        return np.array([[self.d3r, self.d3theta, self.d3phi]])

    @cached_property
    def radial_to_apex(self) -> np.ndarray:
        """Matrix mapping radial to field-parallel apex coordinates."""
        return np.einsum(
            "ijk,jlk->ilk",
            self.field_parallel_to_apex,
            self.radial_to_field_parallel,
            optimize=True,
        )
