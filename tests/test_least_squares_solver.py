"""Tests for least-squares solver helpers."""

import numpy as np
import pytest

from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math import JAX_AVAILABLE, as_linear_map, set_backend, use_jax


def test_normal_pinv_solves_block_rhs():
    """Normal-equation pseudo-inverse supports reusable RHS maps."""
    A = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_uses_normal_equation_cutoff():
    """Normal pseudo-inverse applies cutoff after forming A* A."""
    A = np.diag([1.0, 1e-8])
    rhs = np.array([1.0, 1e-8])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 0.0]))


def test_normal_pinv_keeps_modes_above_normal_equation_cutoff():
    """Normal pseudo-inverse keeps modes above the A* A cutoff."""
    A = np.diag([1.0, 1e-6])
    rhs = np.array([1.0, 1e-6])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    solution = solver.solve(problem, rhs)

    np.testing.assert_allclose(solution, np.array([1.0, 1.0]))


def test_normal_pinv_does_not_use_direct_solve(monkeypatch):
    """Normal pseudo-inverse also used for full-rank systems."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)

    def fail_solve(*args, **kwargs):
        raise AssertionError("normal_pinv should apply a pseudo-inverse, not solve")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    solution = solver.solve(problem, rhs)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected)


def test_normal_pinv_response_solver_reuses_factorization(monkeypatch):
    """Reusable normal-pinv response solves cache dense factors."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
    solve_response = solver.build_response_solver(problem)

    def fail_dense_assembly():
        raise AssertionError("response solver should reuse cached dense factors")

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    A_H = A.T.conj()
    normal_pinv = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(solve_response(rhs_first), normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(solve_response(rhs_second), normal_pinv @ (A_H @ rhs_second))


def test_normal_pinv_solve_reuses_cached_pseudo_inverse(monkeypatch):
    """Repeated dense normal-pinv solves reuse the cached n^3 factor."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs_first = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])
    rhs_second = np.array([[0.0, 4.0], [2.5, -1.0], [1.5, 3.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=3)
    solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
    calls = 0
    original_pinv = np.linalg.pinv

    def counted_pinv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", counted_pinv)

    first = solver.solve(problem, rhs_first)
    second = solver.solve(problem, rhs_second)

    A_H = A.T.conj()
    normal_pinv = original_pinv(A_H @ A, rtol=solver.tolerance, hermitian=True)
    np.testing.assert_allclose(first, normal_pinv @ (A_H @ rhs_first))
    np.testing.assert_allclose(second, normal_pinv @ (A_H @ rhs_second))
    assert len(problem._dense_normal_pinv_cache) == 1
    assert calls == (0 if use_jax() else 1)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv"])
def test_dense_solvers_preserve_jax_output_when_backend_enabled(solver_name):
    """Dense solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_svd_solver_preserves_jax_output_when_backend_enabled():
    """SVD solver keeps JAX-facing assembly and output."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_matrix = problem.assemble_dense_system_matrix()
        assert "jax" in type(rhs_block).__module__
        assert "jax" in type(system_matrix).__module__
        solver = LeastSquaresSolver(solver="svd", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_least_squares_problem_follows_jax_operator_context_when_numpy_active():
    """JAX-backed operator terms should drive matrix-free assembly."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0]])

    try:
        set_backend("numpy")
        problem = LeastSquaresProblem(A=jnp.asarray(A), solution_shape=2, data_shapes=3)
        rhs_block, _, _ = problem.assemble_rhs_block(rhs)
        system_block = problem.get_system_linear_map().matmat(np.eye(2))
    finally:
        set_backend(previous_backend)

    assert "jax" in type(rhs_block).__module__
    assert "jax" in type(system_block).__module__
    np.testing.assert_allclose(np.asarray(system_block), A)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_normal_pinv_matches_numpy_hermitian_reference_when_jax_enabled():
    """JAX normal-pinv matches the hermitian reference."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="normal_pinv", tolerance=1e-13)
        solution = solver.solve(problem, rhs)
    finally:
        set_backend(previous_backend)

    A_H = A.T.conj()
    expected = np.linalg.pinv(A_H @ A, rtol=solver.tolerance, hermitian=True) @ (A_H @ rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solver_solves_block_rhs_with_base_preconditioner(solver_name):
    """Iterative block RHS solves reuse the base preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0, -1.0], [3.0, 1.0, 0.5], [0.5, -2.0, 4.0], [1.5, 0.0, 2.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12, preconditioner="jacobi")
    preconditioner = solver.build_preconditioner(problem)

    assert preconditioner.shape == (2, 2)
    solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_solvers_do_not_materialize_dense_system(monkeypatch, solver_name):
    """Iterative solves stay matrix-free for no dense preconditioner."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(solver=solver_name, tolerance=1e-12)

    def fail_dense_assembly():
        raise AssertionError("iterative solvers should not assemble dense systems")

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    solution = solver.solve(problem, rhs, maxiter=200)

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", ["lsmr", "cgls"])
def test_iterative_jacobi_preconditioner_does_not_materialize_dense_system(
    monkeypatch, solver_name
):
    """Jacobi-preconditioned iterative solves stay matrix-free."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    solver = LeastSquaresSolver(
        solver=solver_name,
        tolerance=1e-12,
        preconditioner="jacobi",
    )

    def fail_dense_assembly():
        raise AssertionError(
            "jacobi-preconditioned iterative solvers should not assemble dense systems"
        )

    monkeypatch.setattr(problem, "assemble_dense_system_matrix", fail_dense_assembly)

    preconditioner = solver.build_preconditioner(problem)
    solution = solver.solve(
        problem,
        rhs,
        preconditioner=preconditioner,
        maxiter=200,
    )

    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
@pytest.mark.parametrize("solver_name", ["cgls", "lsmr"])
@pytest.mark.parametrize("preconditioner_type", [None, "jacobi", "pinv"])
def test_iterative_solvers_preserve_jax_output_when_backend_enabled(
    solver_name, preconditioner_type
):
    """Iterative solvers preserve JAX output when JAX is active."""
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 1.0], [0.5, -2.0], [1.5, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=2, data_shapes=4)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(
            solver=solver_name, tolerance=1e-12, preconditioner=preconditioner_type
        )
        preconditioner = solver.build_preconditioner(problem)
        solution = solver.solve(problem, rhs, preconditioner=preconditioner, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_jax_lsmr_solves_underdetermined_block_rhs():
    """Internal JAX LSMR handles rectangular underdetermined systems."""
    A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, -1.0]])
    rhs = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    expected = np.linalg.lstsq(A, rhs, rcond=None)[0]
    problem = LeastSquaresProblem(A=A, solution_shape=4, data_shapes=3)
    previous_backend = use_jax()

    try:
        set_backend("jax")
        solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-12)
        solution = solver.solve(problem, rhs, maxiter=200)
    finally:
        set_backend(previous_backend)

    assert "jax" in type(solution).__module__
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan, np.array([1.0, 2.0])])
def test_regularization_weights_must_be_finite_non_negative_scalars(weight):
    """Invalid regularization weights fail before system assembly."""
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        LeastSquaresProblem(
            A=np.eye(2),
            solution_shape=2,
            data_shapes=2,
            regularization_matrices=np.eye(2),
            regularization_weights=weight,
        )


@pytest.mark.parametrize("solver_name", ["normal_solve", "normal_pinv", "svd"])
@pytest.mark.parametrize("entrypoint", ["solve", "build_response_solver"])
def test_dense_solvers_reject_explicit_preconditioners(solver_name, entrypoint):
    """Dense solvers reject explicitly supplied preconditioners."""
    problem = LeastSquaresProblem(A=np.eye(2), solution_shape=2, data_shapes=2)
    solver = LeastSquaresSolver(solver=solver_name)
    preconditioner = as_linear_map(np.eye(2))

    with pytest.raises(ValueError, match="does not accept a preconditioner"):
        if entrypoint == "solve":
            solver.solve(problem, np.ones(2), preconditioner=preconditioner)
        else:
            solver.build_response_solver(problem, preconditioner=preconditioner)
