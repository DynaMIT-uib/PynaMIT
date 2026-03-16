"""Shared structured least-squares subproblem definitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Optional

from pynamit.math.least_squares_problem import LeastSquaresProblem, Shape
from pynamit.math.least_squares_solver import LeastSquaresSolver


@dataclass(frozen=True)
class StructuredLeastSquaresDataTerm:
    """One weighted data-fit term in a structured least-squares subproblem."""

    operator: Any
    data_shape: Shape
    sqrt_weight: Any = None


@dataclass(frozen=True)
class StructuredLeastSquaresRegularizationTerm:
    """One regularization term in a structured least-squares subproblem."""

    operator: Any
    weight: float


@dataclass(frozen=True)
class StructuredLeastSquaresSubproblem:
    """Common constrained-quadratic wrapper over ``LeastSquaresProblem``.

    This keeps different problem statements in one shared framework:

    ``min_x  sum ||A_k x - b_k||^2 + sum lambda_l ||R_l x||^2``
    ``s.t.   C x = d`` (optional at solve time)
    """

    solution_shape: Shape
    data_terms: tuple[StructuredLeastSquaresDataTerm, ...]
    regularization_terms: tuple[StructuredLeastSquaresRegularizationTerm, ...] = ()
    matrix_free: bool = True
    column_scale: Optional[Any] = None

    @cached_property
    def problem(self) -> LeastSquaresProblem:
        """Materialize the underlying ``LeastSquaresProblem``."""
        return LeastSquaresProblem(
            A=[term.operator for term in self.data_terms],
            solution_shape=self.solution_shape,
            data_shapes=[term.data_shape for term in self.data_terms],
            sqrt_weights=[term.sqrt_weight for term in self.data_terms],
            regularization_matrices=[
                term.operator for term in self.regularization_terms if term.weight > 0.0
            ],
            regularization_weights=[
                float(term.weight) for term in self.regularization_terms if term.weight > 0.0
            ],
            matrix_free=bool(self.matrix_free),
            column_scale=self.column_scale,
        )

    @property
    def solution_size(self) -> int:
        """Return the flattened solution dimension."""
        return int(self.problem.solution_size)

    @property
    def num_data_terms(self) -> int:
        """Return the number of data-fit terms."""
        return int(self.problem.num_data_terms)

    def with_equality(
        self,
        *,
        equality_operator: Optional[Any] = None,
        equality_rhs_builder: Optional[Callable[[Any], Any]] = None,
    ) -> ConstrainedStructuredLeastSquaresSubproblem:
        """Attach optional hard equalities to this structured subproblem."""
        return ConstrainedStructuredLeastSquaresSubproblem(
            subproblem=self,
            equality_operator=equality_operator,
            equality_rhs_builder=equality_rhs_builder,
        )

    def solve(
        self,
        solver: LeastSquaresSolver,
        rhs: Any,
        *,
        preconditioner: Optional[Any] = None,
        equality_operator: Optional[Any] = None,
        equality_rhs: Optional[Any] = None,
        elimination_rcond: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the structured subproblem with the provided solver."""
        return solver.solve(
            self.problem,
            rhs,
            preconditioner=preconditioner,
            equality_operator=equality_operator,
            equality_rhs=equality_rhs,
            elimination_rcond=elimination_rcond,
            **kwargs,
        )


@dataclass(frozen=True)
class ConstrainedStructuredLeastSquaresSubproblem:
    """Structured subproblem with optional hard equalities attached."""

    subproblem: StructuredLeastSquaresSubproblem
    equality_operator: Optional[Any] = None
    equality_rhs_builder: Optional[Callable[[Any], Any]] = None

    @property
    def problem(self) -> LeastSquaresProblem:
        """Compatibility view of the underlying unconstrained problem."""
        return self.subproblem.problem

    @property
    def solution_size(self) -> int:
        """Return the flattened solution dimension."""
        return int(self.subproblem.solution_size)

    @property
    def num_data_terms(self) -> int:
        """Return the number of data-fit terms."""
        return int(self.subproblem.num_data_terms)

    def with_equality(
        self,
        *,
        equality_operator: Optional[Any] = None,
        equality_rhs_builder: Optional[Callable[[Any], Any]] = None,
    ) -> ConstrainedStructuredLeastSquaresSubproblem:
        """Return a copy with updated hard-equality settings."""
        return ConstrainedStructuredLeastSquaresSubproblem(
            subproblem=self.subproblem,
            equality_operator=self.equality_operator
            if equality_operator is None
            else equality_operator,
            equality_rhs_builder=(
                self.equality_rhs_builder if equality_rhs_builder is None else equality_rhs_builder
            ),
        )

    def solve(
        self,
        solver: LeastSquaresSolver,
        rhs: Any,
        *,
        preconditioner: Optional[Any] = None,
        equality_rhs_input: Optional[Any] = None,
        equality_rhs: Optional[Any] = None,
        elimination_rcond: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Solve the constrained structured subproblem."""
        resolved_equality_rhs = equality_rhs
        if resolved_equality_rhs is None and self.equality_rhs_builder is not None:
            resolved_equality_rhs = self.equality_rhs_builder(equality_rhs_input)
        elif resolved_equality_rhs is None:
            resolved_equality_rhs = equality_rhs_input

        return self.subproblem.solve(
            solver,
            rhs,
            preconditioner=preconditioner,
            equality_operator=self.equality_operator,
            equality_rhs=resolved_equality_rhs,
            elimination_rcond=elimination_rcond,
            **kwargs,
        )
