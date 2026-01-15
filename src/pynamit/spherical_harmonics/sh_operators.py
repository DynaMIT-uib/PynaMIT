"""Spherical Harmonic Operators.

This module provides spectral space operators for spherical harmonics,
including gradient, curl, divergence, Laplacian, and product operators.
"""

from typing import TYPE_CHECKING, Any, Optional
import numpy as np

from pynamit.utils import xp

if TYPE_CHECKING:
    from pynamit.math.linear_map import LinearMap
    from pynamit.spherical_harmonics.sh_basis import SHBasis


def get_laplacian_factor(n: np.ndarray, r: float = 1.0) -> np.ndarray:
    """Compute the spherical harmonic Laplacian eigenvalues.

    Parameters
    ----------
    n : np.ndarray
        Degree array.
    r : float
        Radius at which to evaluate.

    Returns
    -------
    np.ndarray
        Laplacian factors: -n(n+1)/r^2
    """
    return -n * (n + 1) / r**2


def get_radial_shift_Ve_factor(n: np.ndarray, start: float, end: float) -> np.ndarray:
    """Compute factors to radially shift external potential coefficients.

    Parameters
    ----------
    n : np.ndarray
        Degree array.
    start : float
        Starting radius.
    end : float
        Ending radius.

    Returns
    -------
    np.ndarray
        Shift factors.
    """
    return (start / end) ** (1 - n)


def get_radial_shift_Vi_factor(n: np.ndarray, start: float, end: float) -> np.ndarray:
    """Compute factors to radially shift internal potential coefficients.

    Parameters
    ----------
    n : np.ndarray
        Degree array.
    start : float
        Starting radius.
    end : float
        Ending radius.

    Returns
    -------
    np.ndarray
        Shift factors.
    """
    return (start / end) ** (n + 2)


def get_coeffs_to_delta_V_factor(n: np.ndarray) -> np.ndarray:
    """Compute factors to convert coefficients to delta V at unit radius.

    Parameters
    ----------
    n : np.ndarray
        Degree array.

    Returns
    -------
    np.ndarray
        Conversion factors: 2n + 1
    """
    return 2 * n + 1


def build_laplacian_operator(basis: "SHBasis", r: float = 1.0) -> "LinearMap":
    """Build the Laplacian operator for a spherical harmonic basis.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    r : float
        Radius at which to evaluate.

    Returns
    -------
    LinearMap
        Diagonal linear map representing the Laplacian.
    """
    from pynamit.math.linear_map import diagonal_linear_map
    factors = get_laplacian_factor(basis.n, r)
    return diagonal_linear_map(factors)


def build_radial_shift_operator(
    basis: "SHBasis", start_r: float, end_r: float, kind: str = "external"
) -> "LinearMap":
    """Build the radial shift operator for potential coefficients.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    start_r : float
        Starting radius.
    end_r : float
        Ending radius.
    kind : str
        Either "external" or "internal" for potential type.

    Returns
    -------
    LinearMap
        Diagonal linear map for radial shifting.
    """
    from pynamit.math.linear_map import diagonal_linear_map
    if kind == "external":
        factors = get_radial_shift_Ve_factor(basis.n, start_r, end_r)
    else:
        factors = get_radial_shift_Vi_factor(basis.n, start_r, end_r)
    return diagonal_linear_map(factors)


def build_potential_scaling_operator(basis: "SHBasis") -> "LinearMap":
    """Build the operator for converting coefficients to surface potential.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.

    Returns
    -------
    LinearMap
        Diagonal linear map for potential scaling.
    """
    from pynamit.math.linear_map import diagonal_linear_map
    factors = get_coeffs_to_delta_V_factor(basis.n)
    return diagonal_linear_map(factors)


def build_gradient_operator(basis: "SHBasis", r: float = 1.0) -> "LinearMap":
    """Build the analytical gradient operator in spectral space.

    Maps scalar coefficients to Poloidal Vector coefficients.
    Applying this operator results in coefficients for the basis functions
    G_vsh[0] = -grad Y.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    r : float
        Radius at which to evaluate.

    Returns
    -------
    LinearMap
        Block linear map of shape (2L, L).
    """
    from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
    ident = diagonal_linear_map(xp.ones(basis.index_length) / r)
    zeros = diagonal_linear_map(xp.zeros(basis.index_length))
    return BlockLinearMap([[ident], [zeros]])


def build_curl_operator(basis: "SHBasis", r: float = 1.0) -> "LinearMap":
    """Build the analytical curl operator (r x grad) in spectral space.

    Maps scalar coefficients to Toroidal Vector coefficients.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    r : float
        Radius at which to evaluate.

    Returns
    -------
    LinearMap
        Block linear map of shape (2L, L).
    """
    from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
    zeros = diagonal_linear_map(xp.zeros(basis.index_length))
    ident = diagonal_linear_map(xp.ones(basis.index_length) / r)
    return BlockLinearMap([[zeros], [ident]])


def build_divergence_operator(basis: "SHBasis", r: float = 1.0) -> "LinearMap":
    """Build the analytical divergence operator in spectral space.

    Maps [Poloidal; Toroidal] vector coefficients to scalar coefficients.

    div(c_pol * (-grad Y) + c_tor * (rxgrad Y)) = c_pol * (-laplacian Y)

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    r : float
        Radius at which to evaluate.

    Returns
    -------
    LinearMap
        Block linear map of shape (L, 2L).
    """
    from pynamit.math.linear_map import diagonal_linear_map, BlockLinearMap
    factors = basis.n * (basis.n + 1) / r
    op_pol = diagonal_linear_map(factors)
    op_tor = diagonal_linear_map(xp.zeros(basis.index_length))
    return BlockLinearMap([[op_pol, op_tor]])


def build_product_operator(
    basis: "SHBasis",
    coeffs_a: np.ndarray,
    grid: Optional[Any] = None,
    method: str = "transform"
) -> "LinearMap":
    """Build product operator for SHBasis.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    coeffs_a : np.ndarray
        Coefficients of multiplier field.
    grid : Any, optional
        Grid for transform method.
    method : str
        'transform' (Grid-based evaluate/multiply/project) or
        'spectral' (Wigner-3j / Gaunt convolution).

    Returns
    -------
    LinearMap
        Product operator.
    """
    from pynamit.math.linear_map import as_linear_map, diagonal_linear_map

    if method == "spectral":
        from pynamit.spherical_harmonics.gaunt import GauntEngine
        engine = GauntEngine(basis)
        M = engine.get_interaction_matrix(coeffs_a)
        return as_linear_map(M)

    if grid is None:
        raise ValueError("SHBasis.get_product_operator requires a grid for the transform method.")

    # Transform multiplier coefficients to grid
    a_grid = basis.evaluate(coeffs_a, grid, vector_type="scalar")

    # Build the Transform-Product-Project operator: P @ diag(a) @ G
    G_mat = basis.get_G(grid)
    P_mat = grid.get_projection_matrix(basis)

    op_G = as_linear_map(G_mat)
    op_P = as_linear_map(P_mat)
    op_diag_a = diagonal_linear_map(a_grid.flatten())

    return op_P @ op_diag_a @ op_G


def build_vector_product_operator(basis: "SHBasis", tensor_sigma_coeffs: np.ndarray) -> "LinearMap":
    """Build interaction operator for a 2x2 conductance tensor and VSH vectors.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    tensor_sigma_coeffs : np.ndarray
        Tensor coefficients of shape (2, 2, L) matching SHBasis.

    Returns
    -------
    LinearMap
        Vector product operator.
    """
    from pynamit.spherical_harmonics.gaunt import GauntEngine
    from pynamit.math.linear_map import as_linear_map

    engine = GauntEngine(basis)

    # Project tensor components to quadrature grid
    Q = engine.quad_grid.size
    sigma_quad = np.empty((2, 2, Q))
    for i in range(2):
        for j in range(2):
            sigma_quad[i, j] = engine.G_scalar.T @ tensor_sigma_coeffs[i, j]

    # Compute Vector Interaction Matrix
    M = engine.get_vector_interaction_matrix(sigma_quad)
    return as_linear_map(M)


def get_regularization_matrix(
    basis: "SHBasis", scalar: bool = True, reg_lambda: Optional[float] = None
) -> Optional[np.ndarray]:
    """Get the regularization matrix for SHBasis.

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    scalar : bool
        If True, return scalar regularization. Otherwise vector.
    reg_lambda : float, optional
        Regularization parameter. If None, returns None.

    Returns
    -------
    np.ndarray or None
        Regularization matrix, or None if reg_lambda is None.
    """
    if reg_lambda is None:
        return None

    if scalar:
        return np.diag(basis.n)
    else:
        # Helmholtz/Vector regularization
        L_cf = np.stack(
            [
                np.diag(basis.n * (basis.n + 1) / (2 * basis.n + 1)),
                np.zeros((basis.index_length, basis.index_length)),
            ],
            axis=1,
        )
        L_df = np.stack(
            [
                np.zeros((basis.index_length, basis.index_length)),
                np.diag((basis.n + 1) / 2),
            ],
            axis=1,
        )
        return np.array([L_cf, L_df])
