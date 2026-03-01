"""Closure-basis helpers for toroidal operator assembly.

This module keeps closure-space plumbing (auxiliary basis assembly and
projection) separate from toroidal physics formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from pynamit.simulation.geometry_utils import to_dense

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid


@dataclass(frozen=True)
class ToroidalClosureProjector:
    """Project toroidal closure operators between state and auxiliary bases."""

    state_basis: "Basis"
    closure_basis: "Basis"
    grid: "Grid"
    build_auxiliary_assembler: Callable[["Basis"], Any]

    @property
    def uses_auxiliary_basis(self) -> bool:
        """Whether closure basis differs from state basis."""
        return self.closure_basis is not self.state_basis

    @cached_property
    def auxiliary_assembler(self) -> Any:
        """Instantiate auxiliary toroidal assembler in closure basis."""
        if not self.uses_auxiliary_basis:
            raise RuntimeError("Auxiliary assembler requested without auxiliary closure basis.")
        return self.build_auxiliary_assembler(self.closure_basis)

    @cached_property
    def state_to_closure_scalar_map(self) -> np.ndarray:
        """Map state scalar coefficients -> closure scalar coefficients."""
        if not self.uses_auxiliary_basis:
            raise RuntimeError("State->closure map requested without auxiliary closure basis.")
        G_state = np.asarray(to_dense(self.state_basis.get_evaluation_matrix(self.grid)))
        P_closure = np.asarray(
            to_dense(self.closure_basis.construct_scalar_projection_matrix(self.grid))
        )
        return np.asarray(P_closure @ G_state)

    @cached_property
    def closure_to_state_scalar_map(self) -> np.ndarray:
        """Map closure scalar coefficients -> state scalar coefficients."""
        if not self.uses_auxiliary_basis:
            raise RuntimeError("Closure->state map requested without auxiliary closure basis.")
        G_closure = np.asarray(to_dense(self.closure_basis.get_evaluation_matrix(self.grid)))
        P_state = np.asarray(
            to_dense(self.state_basis.construct_scalar_projection_matrix(self.grid))
        )
        return np.asarray(P_state @ G_closure)

    def project_square_operator_to_state(self, operator: np.ndarray) -> np.ndarray:
        """Project square operator from closure basis to state basis."""
        op = np.asarray(operator)
        if not self.uses_auxiliary_basis:
            return op
        return np.asarray(
            self.closure_to_state_scalar_map
            @ op
            @ self.state_to_closure_scalar_map
        )

    def project_vector_rhs_operator_to_state(self, operator: np.ndarray) -> np.ndarray:
        """Project vector-valued RHS map from closure basis to state basis.

        Input/output operator shapes:
        - closure: (n_closure, 2*n_closure)
        - state:   (n_state,   2*n_state)
        """
        op = np.asarray(operator)
        if not self.uses_auxiliary_basis:
            return op
        s2c = np.asarray(self.state_to_closure_scalar_map)
        c2s = np.asarray(self.closure_to_state_scalar_map)
        n_state = int(self.state_basis.index_length)
        n_closure = int(op.shape[0])
        t_in = np.zeros((2 * n_closure, 2 * n_state), dtype=op.dtype)
        t_in[:n_closure, :n_state] = s2c
        t_in[n_closure:, n_state:] = s2c
        return np.asarray(c2s @ op @ t_in)
