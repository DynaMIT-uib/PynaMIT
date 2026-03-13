"""Closure-basis helpers for toroidal operator assembly.

This module keeps closure-space plumbing (auxiliary basis assembly and
projection) separate from toroidal physics formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from pynamit.simulation.spatial.geometry_utils import to_dense

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
        return np.asarray(self.closure_to_state_scalar_map @ op @ self.state_to_closure_scalar_map)

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


@dataclass(frozen=True)
class ToroidalRMBoundaryOperators:
    """Closed-boundary shell operators for the dynamic toroidal ``R_M`` lock.

    These operators make the current shell-level toroidal lock explicit as a
    fixed-point closure in coefficient space:

        ``closure_denominator @ psi_closed = psi_open``

    with

        ``closure_denominator = I - roundtrip_gain``
        ``closure_inv = closure_denominator^{-1}``
        ``reaction = closure_inv - I``.

    The ``roundtrip_gain`` operator is the shell-level surrogate for one
    ionosphere-to-``R_M``-to-ionosphere induced toroidal roundtrip. It is kept
    for diagnostics/comparison only; the runtime toroidal channel now uses the
    open shell operator and relies on the PFAC/poloidal closure for the actual
    RM electromagnetic reaction. This surrogate is distinct from the explicit
    ``R_M`` normal-current closure-current diagnostics in ``pfac.py``.
    """

    rm_to_ri: np.ndarray
    ri_to_rm: np.ndarray
    roundtrip_gain: np.ndarray
    closure_denominator: np.ndarray
    closure_inv: np.ndarray
    reaction: np.ndarray


@dataclass(frozen=True)
class ToroidalRMBoundarySourceOperators:
    """Explicit source-side toroidal boundary operator at ``R_M``.

    This captures the physically unambiguous part of the toroidal closure:

        ``alpha(R_I) -> j_n(R_M) -> chi_M(R_M) -> psi(R_M^-)``

    with

        ``psi(R_M^-) = (mu0 / R_M) * chi_M``.

    No downward continuation back to ``R_I`` is implied here.
    """

    alpha_to_sheet_boundary_psi_rm: np.ndarray


@dataclass(frozen=True)
class ToroidalRMReactionPrototype:
    """Standalone comparison bundle for open vs closed RM toroidal response.

    This is intentionally diagnostic/prototype-only. It exposes the current
    runtime shell surrogate, the corresponding open response, and the explicit
    source-side ``R_M`` boundary toroidal operator together with the normal-current
    closure operators.
    """

    rm_to_ri: np.ndarray
    ri_to_rm: np.ndarray
    roundtrip_gain: np.ndarray
    closure_denominator: np.ndarray
    closure_inv: np.ndarray
    shell_reaction_operator: np.ndarray
    alpha_to_psi_shell_closed: np.ndarray
    radial_closure_dt_psi_shell_closed: np.ndarray
    alpha_to_sheet_boundary_psi_rm: np.ndarray
    alpha_to_psi_open: np.ndarray
    alpha_to_psi_closed: np.ndarray
    alpha_to_psi_reaction: np.ndarray
    radial_closure_dt_psi_open: np.ndarray
    radial_closure_dt_psi_closed: np.ndarray
    radial_closure_dt_psi_reaction: np.ndarray
    toroidal_feedback_dtalpha_open: np.ndarray
    toroidal_feedback_dtalpha_closed: np.ndarray
    toroidal_feedback_dtalpha_reaction: np.ndarray
    dynamic_pfac_open: np.ndarray
    dynamic_pfac_closed: np.ndarray
    dynamic_pfac_reaction: np.ndarray
    alpha_to_dynamic_pfac_reaction: np.ndarray
    alpha_to_normal_current_rm_grid: np.ndarray
    alpha_to_closure_potential_rm_coeff: np.ndarray
    alpha_to_divergent_closure_current_rm_grid: np.ndarray
