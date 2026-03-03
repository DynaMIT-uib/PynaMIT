"""Closure-basis coupling helpers for poloidal-side RM/PFAC operators.

This module centralizes projection and RM-coupling operator assembly used by
the poloidal pathways. It keeps the distinction explicit between:
- solution/state coefficient basis (where unknowns are represented), and
- closure basis (where PFAC/radial closure physics is assembled).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from pynamit.simulation.spatial.geometry_utils import to_dense

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.grid import Grid
    from pynamit.simulation.spatial.pfac import PFACIntegrator


@dataclass(frozen=True)
class RMCouplingOperators:
    """RM coupling operators represented in solution coefficient space."""

    rm_to_ri: np.ndarray
    ri_to_rm: np.ndarray
    roundtrip: np.ndarray
    roundtrip_inv: np.ndarray
    feedback: np.ndarray


@dataclass(frozen=True)
class PoloidalClosureProjector:
    """Project poloidal closure operators into solution coefficient space."""

    solution_space: "Basis"
    closure_basis: "Basis"
    grid: "Grid"
    pfac_integrator: "PFACIntegrator"

    @property
    def uses_auxiliary_basis(self) -> bool:
        """Whether closure basis differs from solution basis."""
        return self.closure_basis is not self.solution_space

    @cached_property
    def solution_to_closure_scalar_map(self) -> np.ndarray:
        """Map solution scalar coefficients -> closure scalar coefficients."""
        if not self.uses_auxiliary_basis:
            raise RuntimeError("Solution->closure map requested without auxiliary closure basis.")
        G_sol = np.asarray(to_dense(self.solution_space.get_evaluation_matrix(self.grid)))
        P_closure = np.asarray(
            to_dense(self.closure_basis.construct_scalar_projection_matrix(self.grid))
        )
        return np.asarray(P_closure @ G_sol)

    @cached_property
    def closure_to_solution_scalar_map(self) -> np.ndarray:
        """Map closure scalar coefficients -> solution scalar coefficients."""
        if not self.uses_auxiliary_basis:
            raise RuntimeError("Closure->solution map requested without auxiliary closure basis.")
        G_closure = np.asarray(to_dense(self.closure_basis.get_evaluation_matrix(self.grid)))
        P_solution = np.asarray(
            to_dense(self.solution_space.construct_scalar_projection_matrix(self.grid))
        )
        return np.asarray(P_solution @ G_closure)

    def project_scalar_operator_to_solution(self, operator: np.ndarray) -> np.ndarray:
        """Project scalar closure operator from closure basis to solution basis."""
        op = np.asarray(operator)
        if not self.uses_auxiliary_basis:
            return op
        return np.asarray(
            self.closure_to_solution_scalar_map
            @ op
            @ self.solution_to_closure_scalar_map
        )

    @cached_property
    def rm_coupling_solution_operators(self) -> Optional[RMCouplingOperators]:
        """RM coupling operators represented in solution coefficient space."""
        if self.pfac_integrator.RM is None:
            return None
        br_rm_to_ri_shift, br_ri_to_rm_shift, rm_roundtrip_denominator = (
            self.pfac_integrator.get_coupling_factors()
        )

        op_rm_to_ri = np.diag(np.asarray(br_rm_to_ri_shift))
        op_ri_to_rm = np.diag(np.asarray(br_ri_to_rm_shift))
        op_roundtrip = np.diag(np.asarray(rm_roundtrip_denominator))

        op_rm_to_ri = self.project_scalar_operator_to_solution(op_rm_to_ri)
        op_ri_to_rm = self.project_scalar_operator_to_solution(op_ri_to_rm)
        op_roundtrip = self.project_scalar_operator_to_solution(op_roundtrip)

        n = int(self.solution_space.index_length)
        if not self.uses_auxiliary_basis:
            roundtrip_vec = np.asarray(rm_roundtrip_denominator, dtype=float)
            tol = max(float(np.finfo(float).eps * max(roundtrip_vec.size, 1)), 1e-15)
            inv_roundtrip_vec = np.zeros_like(roundtrip_vec)
            keep = np.abs(roundtrip_vec) > tol
            inv_roundtrip_vec[keep] = 1.0 / roundtrip_vec[keep]
            op_roundtrip_inv = np.diag(inv_roundtrip_vec)
        else:
            rcond = max(float(np.finfo(float).eps * max(n, 1)), 1e-15)
            op_roundtrip_inv = np.linalg.pinv(op_roundtrip, rcond=rcond)
        op_feedback = op_rm_to_ri @ op_ri_to_rm @ op_roundtrip_inv

        return RMCouplingOperators(
            rm_to_ri=np.asarray(op_rm_to_ri),
            ri_to_rm=np.asarray(op_ri_to_rm),
            roundtrip=np.asarray(op_roundtrip),
            roundtrip_inv=np.asarray(op_roundtrip_inv),
            feedback=np.asarray(op_feedback),
        )

    def apply_rm_closure(self, operator: np.ndarray) -> np.ndarray:
        """Apply RM closure operator ``roundtrip_inv`` to a solution-space operator."""
        op = np.asarray(operator)
        if self.pfac_integrator.RM is None:
            return op
        maps = self.rm_coupling_solution_operators
        if maps is None:
            return op
        roundtrip_inv = np.asarray(maps.roundtrip_inv)
        if roundtrip_inv.shape[1] != op.shape[0]:
            raise ValueError(
                "RM closure operator shape mismatch: "
                f"{roundtrip_inv.shape} cannot left-multiply {op.shape}."
            )
        return np.asarray(roundtrip_inv @ op)
