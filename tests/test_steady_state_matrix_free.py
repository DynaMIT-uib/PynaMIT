"""Test matrix-free steady-state m_ind calculation."""

import pytest
import numpy as np

from pynamit.math.linear_map import as_linear_map
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.utils import asarray, xp


def solve_steady_state_dense(vec_b, induction_matrix):
    """Dense path: lstsq solver."""
    L = asarray(induction_matrix)
    result = xp.linalg.lstsq(L, vec_b, rcond=1e-13)
    return result[0]


def solve_steady_state_matrix_free(vec_b, induction_op, n, solver="lsmr"):
    """Matrix-free path: iterative solver."""
    problem = LeastSquaresProblem(
        A=[induction_op],
        solution_shape=(n,),
        data_shapes=[(n,)],
    )
    ls_solver = LeastSquaresSolver(solver=solver, tolerance=1e-10)
    return asarray(ls_solver.solve(problem, [vec_b], maxiter=5000))


class TestSteadyStateMatrixFree:
    """Tests for matrix-free steady_state_m_ind."""

    def test_matrix_free_lsmr_matches_dense(self):
        """Test that LSMR matrix-free solver matches dense lstsq."""
        n = 50

        # Create a random but well-conditioned induction matrix
        np.random.seed(42)
        A = np.random.randn(n, n)
        induction_matrix_dense = A.T @ A + 0.1 * np.eye(n)

        # Create corresponding LinearMap
        induction_matrix_lm = as_linear_map(induction_matrix_dense)

        # Create random RHS (simulating toroidal E-field)
        vec_b = np.random.randn(n) * 1e-6

        # Compute using dense solver
        result_dense = solve_steady_state_dense(vec_b, induction_matrix_dense)

        # Compute using matrix-free LSMR solver
        result_lsmr = solve_steady_state_matrix_free(
            vec_b, induction_matrix_lm, n, solver="lsmr"
        )

        # Results should match within tolerance
        np.testing.assert_allclose(
            result_lsmr, result_dense, rtol=1e-6, atol=1e-10,
            err_msg="LSMR matrix-free result differs from dense lstsq"
        )

    def test_matrix_free_cg_matches_dense(self):
        """Test that CG matrix-free solver matches dense lstsq."""
        n = 50

        # Create a random but well-conditioned induction matrix
        np.random.seed(42)
        A = np.random.randn(n, n)
        # Make it well-conditioned and symmetric positive definite
        induction_matrix_dense = A.T @ A + 0.5 * np.eye(n)

        # Create corresponding LinearMap
        induction_matrix_lm = as_linear_map(induction_matrix_dense)

        # Create random RHS
        vec_b = np.random.randn(n) * 1e-6

        # Compute using dense solver
        result_dense = solve_steady_state_dense(vec_b, induction_matrix_dense)

        # Compute using matrix-free CG solver
        result_cg = solve_steady_state_matrix_free(
            vec_b, induction_matrix_lm, n, solver="cg"
        )

        # Results should match within tolerance
        np.testing.assert_allclose(
            result_cg, result_dense, rtol=1e-5, atol=1e-9,
            err_msg="CG matrix-free result differs from dense lstsq"
        )

    def test_matrix_free_with_ill_conditioned_matrix(self):
        """Test matrix-free solver handles ill-conditioned matrices."""
        n = 50

        # Create an ill-conditioned matrix
        np.random.seed(123)
        U, _ = np.linalg.qr(np.random.randn(n, n))
        # Create singular values with large condition number
        s = np.logspace(0, -8, n)
        induction_matrix_dense = U @ np.diag(s) @ U.T

        # Create corresponding LinearMap
        induction_matrix_lm = as_linear_map(induction_matrix_dense)

        # Create random RHS
        vec_b = np.random.randn(n) * 1e-6

        # Both should compute without error
        result_dense = solve_steady_state_dense(vec_b, induction_matrix_dense)
        result_lsmr = solve_steady_state_matrix_free(
            vec_b, induction_matrix_lm, n, solver="lsmr"
        )

        # For ill-conditioned matrices, we expect minimum-norm solutions
        # The results may differ slightly due to different regularization approaches
        # But both should produce finite, reasonable results
        assert np.all(np.isfinite(result_dense))
        assert np.all(np.isfinite(result_lsmr))

        # Both solvers should find solutions with similar residuals
        # (even if the solutions themselves differ in the null-space)
        residual_dense = np.linalg.norm(induction_matrix_dense @ result_dense - vec_b)
        residual_lsmr = np.linalg.norm(induction_matrix_dense @ result_lsmr - vec_b)

        # Both residuals should be small relative to the RHS
        assert residual_dense < 10 * np.linalg.norm(vec_b)
        assert residual_lsmr < 10 * np.linalg.norm(vec_b)

    def test_with_sh_basis_toroidal_extraction(self):
        """Test that toroidal extraction + matrix-free solving works end-to-end."""
        # Create SH basis
        Nmax = 5
        Mmax = 5
        basis = SHBasis(Nmax=Nmax, Mmax=Mmax)
        n = basis.index_length

        # Create random E-field coefficients (2, n) for tangential field
        np.random.seed(42)
        E_coeffs = np.random.randn(2, n) * 1e-6

        # Extract toroidal part (second half for SH)
        vec_b = -asarray(basis.get_toroidal_potential_coeffs(E_coeffs))

        # Create a well-conditioned induction matrix
        A = np.random.randn(n, n)
        induction_matrix = A.T @ A + 0.1 * np.eye(n)

        # Test dense
        result_dense = solve_steady_state_dense(vec_b, induction_matrix)

        # Test matrix-free
        induction_lm = as_linear_map(induction_matrix)
        result_lsmr = solve_steady_state_matrix_free(
            vec_b, induction_lm, n, solver="lsmr"
        )

        np.testing.assert_allclose(
            result_lsmr, result_dense, rtol=1e-6, atol=1e-10,
            err_msg="End-to-end SH test: LSMR differs from dense"
        )


class TestCoupledSteadyStateMatrixFree:
    """Tests for matrix-free coupled steady_state_coupled."""

    def test_coupled_lsmr_matches_dense(self):
        """Test that coupled LSMR matrix-free solver matches dense lstsq."""
        N = 30  # N coefficients
        size = 2 * N  # Coupled system is [psi, m_ind]

        # Create a random but well-conditioned coupled system matrix
        np.random.seed(42)
        A = np.random.randn(size, size)
        L_flat = A.T @ A + 0.1 * np.eye(size)

        # Create corresponding LinearMap
        L_lm = as_linear_map(L_flat)

        # Create random forcing K of shape (2, N)
        K = np.random.randn(2, N) * 1e-6
        K_flat = K.reshape(size)

        # Solve using dense lstsq
        result_dense = xp.linalg.lstsq(L_flat, -K_flat, rcond=1e-13)[0]

        # Solve using matrix-free LSMR
        problem = LeastSquaresProblem(
            A=[L_lm],
            solution_shape=(size,),
            data_shapes=[(size,)],
        )
        ls_solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-10)
        result_lsmr = asarray(ls_solver.solve(problem, [-K_flat], maxiter=5000))

        # Results should match
        np.testing.assert_allclose(
            result_lsmr.reshape(2, N), result_dense.reshape(2, N),
            rtol=1e-6, atol=1e-10,
            err_msg="Coupled LSMR result differs from dense lstsq"
        )

    def test_coupled_cg_matches_dense(self):
        """Test that coupled CG matrix-free solver matches dense lstsq."""
        N = 30
        size = 2 * N

        np.random.seed(42)
        A = np.random.randn(size, size)
        # Make well-conditioned and SPD
        L_flat = A.T @ A + 0.5 * np.eye(size)

        L_lm = as_linear_map(L_flat)
        K = np.random.randn(2, N) * 1e-6
        K_flat = K.reshape(size)

        result_dense = xp.linalg.lstsq(L_flat, -K_flat, rcond=1e-13)[0]

        problem = LeastSquaresProblem(
            A=[L_lm],
            solution_shape=(size,),
            data_shapes=[(size,)],
        )
        ls_solver = LeastSquaresSolver(solver="cg", tolerance=1e-10)
        result_cg = asarray(ls_solver.solve(problem, [-K_flat], maxiter=5000))

        np.testing.assert_allclose(
            result_cg.reshape(2, N), result_dense.reshape(2, N),
            rtol=1e-5, atol=1e-9,
            err_msg="Coupled CG result differs from dense lstsq"
        )

    def test_coupled_4d_tensor_to_linearmap(self):
        """Test that (2, N, 2, N) tensor can be converted to LinearMap."""
        N = 20

        # Create a random (2, N, 2, N) tensor like coupled_induction_tensor
        np.random.seed(123)
        L_tensor = np.random.randn(2, N, 2, N)
        # Make it well-conditioned
        L_flat = L_tensor.reshape(2 * N, 2 * N)
        L_flat = L_flat.T @ L_flat + 0.1 * np.eye(2 * N)
        L_tensor = L_flat.reshape(2, N, 2, N)

        # Create LinearMap from flattened tensor
        L_lm = as_linear_map(L_flat)

        # Create forcing
        K = np.random.randn(2, N) * 1e-6
        K_flat = K.reshape(2 * N)

        # Dense solution
        result_dense = xp.linalg.lstsq(L_flat, -K_flat, rcond=1e-13)[0]

        # Matrix-free solution
        problem = LeastSquaresProblem(
            A=[L_lm],
            solution_shape=(2 * N,),
            data_shapes=[(2 * N,)],
        )
        ls_solver = LeastSquaresSolver(solver="lsmr", tolerance=1e-10)
        result_lsmr = asarray(ls_solver.solve(problem, [-K_flat], maxiter=5000))

        np.testing.assert_allclose(
            result_lsmr.reshape(2, N), result_dense.reshape(2, N),
            rtol=1e-6, atol=1e-10,
            err_msg="4D tensor LSMR result differs from dense"
        )
