"""Gauss-Legendre grid basis for exact spherical harmonic transforms.

This module provides a grid basis using Gauss-Legendre quadrature points
in colatitude and uniform points in longitude, enabling exact (to machine
precision) transforms between spherical harmonic coefficients and grid values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Any

import numpy as np

from pynamit.primitives.grid import Grid, GridBasis

if TYPE_CHECKING:
    from pynamit.spherical_harmonics.sh_basis import SHBasis
    from pynamit.math.linear_map import LinearMap

logger = logging.getLogger(__name__)


class GLBasis(GridBasis):
    """Grid basis using Gauss-Legendre quadrature points.

    Provides exact SH<->Grid transforms via quadrature weights. The grid
    uses Gauss-Legendre points in colatitude (theta) and uniform points
    in longitude (phi).

    Grid size: N_theta × N_phi = res × (2*res + 1)

    Parameters
    ----------
    Nmax : int
        Maximum spherical harmonic degree. Determines minimum grid resolution.
    resolution_factor : float, optional
        Multiplier for grid resolution. Default is 1.5, which provides a
        safety margin for accurate integration of products.

    Attributes
    ----------
    res : int
        Number of Gauss-Legendre points in theta.
    n_phi : int
        Number of uniform points in phi.
    weights : ndarray
        Quadrature weights for exact spherical integration.

    Notes
    -----
    The Gauss-Legendre quadrature with n points exactly integrates
    polynomials of degree up to 2n-1. For spherical harmonics of
    maximum degree Nmax, products involve terms up to degree 2*Nmax,
    so we need at least Nmax+1 points. The resolution_factor provides
    additional margin.

    The grid points cluster near the poles (theta=0 and theta=pi),
    which is less uniform than a cubed-sphere grid but enables
    exact spectral transforms.
    """

    def __init__(self, Nmax: int, resolution_factor: float = 1.5):
        """Initialize GL grid matched to SH truncation."""
        super().__init__()

        self.Nmax = Nmax
        self.res = int(resolution_factor * Nmax + 2)
        # Ensure res is even to avoid placing points on the magnetic equator
        if self.res % 2 != 0:
            self.res += 1
        self._kind = "GL"

        # Gauss-Legendre quadrature in theta (colatitude)
        # leggauss returns points x in [-1, 1] and weights w
        # For theta in [0, pi], we use theta = arccos(x)
        x, w_theta = np.polynomial.legendre.leggauss(self.res)
        self.theta_rad = np.arccos(x)  # GL points in [0, pi]
        self.w_theta = w_theta

        # Uniform sampling in phi (longitude)
        # 2*res + 1 points ensures adequate sampling for m up to Nmax
        self.n_phi = 2 * self.res + 1
        self.phi_rad = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        self.w_phi = 2 * np.pi / self.n_phi

        # Build grid (theta, phi meshgrid flattened)
        th_mesh, ph_mesh = np.meshgrid(self.theta_rad, self.phi_rad, indexing="ij")
        self._grid = Grid(
            theta=np.rad2deg(th_mesh.flatten()), phi=np.rad2deg(ph_mesh.flatten())
        )

        # Quadrature weights: W_i = w_theta[i] * w_phi
        # These weights satisfy: sum_i W_i * f(theta_i, phi_i) ≈ integral f dOmega
        self._weights = (
            np.tile(self.w_theta[:, None], (1, self.n_phi)) * self.w_phi
        ).flatten()
        
        # Attach weights to grid for integration purposes (used by ToroidalSystemMatrices)
        self._grid.weights = self._weights

        logger.info(
            f"GLBasis initialized: Nmax={Nmax}, res={self.res}, "
            f"grid_size={self.index_length}"
        )

    @property
    def grid(self) -> Grid:
        """Get the Gauss-Legendre grid."""
        return self._grid

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights for exact spherical integration."""
        return self._weights

    @property
    def index_length(self) -> int:
        """Total number of grid points."""
        return self.res * self.n_phi

    @property
    def kind(self) -> str:
        """Basis kind identifier."""
        return self._kind

    def get_analysis_matrix(self, sh_basis: "SHBasis") -> np.ndarray:
        """Get Grid -> SH transform matrix (exact via quadrature).

        Returns matrix A such that: sh_coeffs = A @ grid_values

        For Schmidt quasi-normalized spherical harmonics, the orthogonality
        relation includes a degree-dependent normalization factor. We use
        weighted least-squares:
            A = (G^T W G)^{-1} G^T W

        This ensures A @ G = I (identity roundtrip) for any normalization.

        Parameters
        ----------
        sh_basis : SHBasis
            The spherical harmonic basis to project onto.

        Returns
        -------
        A : ndarray
            Analysis matrix of shape (N_sh, N_grid).
        """
        # G: (N_grid, N_sh) - evaluation matrix (SH -> Grid)
        G = sh_basis.get_G(self.grid)
        if hasattr(G, "toarray"):
            G = G.toarray()

        # Weighted least-squares: A = (G^T W G)^{-1} G^T W
        # This works for any normalization convention
        GtW = G.T * self._weights  # (N_sh, N_grid), broadcast multiply
        GtWG = GtW @ G            # (N_sh, N_sh) - the mass matrix
        A = np.linalg.solve(GtWG, GtW)
        return A

    def get_synthesis_matrix(self, sh_basis: "SHBasis") -> np.ndarray:
        """Get SH -> Grid transform matrix (exact evaluation).

        Returns matrix G such that: grid_values = G @ sh_coeffs

        This is simply the evaluation of spherical harmonics at grid points.

        Parameters
        ----------
        sh_basis : SHBasis
            The spherical harmonic basis to evaluate.

        Returns
        -------
        G : ndarray
            Synthesis matrix of shape (N_grid, N_sh).
        """
        G = sh_basis.get_G(self.grid)
        if hasattr(G, "toarray"):
            G = G.toarray()
        return G

    def get_evaluation_matrix(
        self, grid: Any, derivative: Optional[str] = None
    ) -> np.ndarray:
        """Get evaluation matrix from this basis to a target grid.

        For GLBasis, this requires interpolation if the target grid
        differs from the GL grid.

        Parameters
        ----------
        grid : Grid
            Target grid for evaluation.
        derivative : str, optional
            Not supported for GL basis interpolation.

        Returns
        -------
        G : ndarray
            Evaluation matrix.
        """
        if derivative is not None:
            raise NotImplementedError(
                "Derivative evaluation not implemented for GLBasis. "
                "Use SH basis for derivative operations."
            )

        # If same grid, return identity
        if grid is self._grid or grid == self._grid:
            return np.eye(self.index_length)

        # Otherwise, need interpolation matrix
        # For now, raise an error - could implement via spherical interpolation
        raise NotImplementedError(
            "GLBasis.get_evaluation_matrix to different grid requires interpolation. "
            "Consider using evaluate() method instead."
        )

    # Alias for backwards compatibility
    get_G = get_evaluation_matrix

    def project_to_basis(
        self,
        input_values,
        input_grid,
        vector_type,
        target_grid,
        target_basis,
        **kwargs,
    ):
        """Project input data onto the target basis.

        For GLBasis, we interpolate from the input grid to the target grid
        and then fit to the target basis.

        Parameters
        ----------
        input_values : ndarray
            Input field values on input_grid.
        input_grid : Grid
            Grid where input values are defined.
        vector_type : str
            Type of field: 'scalar' or 'tangential'.
        target_grid : Grid
            Grid for fitting to target basis.
        target_basis : Basis
            Target basis for projection.
        **kwargs
            Additional arguments passed to target_basis.from_grid_values.

        Returns
        -------
        coeffs : ndarray
            Coefficients in target_basis.
        """
        from pynamit.primitives.grid.interpolation import create_interpolator

        if target_grid is None:
            raise ValueError("target_grid must be provided")

        interp = create_interpolator(input_grid.theta, input_grid.phi)

        if vector_type == "scalar":
            grid_values = interp.interpolate_scalar(
                input_values, target_grid.theta, target_grid.phi
            )
        elif vector_type == "tangential":
            # Input values are [u_theta, u_phi] = [-u_north, u_east]
            u_east = input_values[1]
            u_north = -input_values[0]
            u_r = np.zeros_like(u_north)

            u_east_int, u_north_int, _ = interp.interpolate_vector(
                u_east,
                u_north,
                u_r,
                target_grid.theta,
                target_grid.phi,
            )
            # Convert back to [u_theta, u_phi]
            grid_values = np.hstack((-u_north_int, u_east_int))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

        # Override weights since input weights are not valid on interpolated grid
        fit_kwargs = kwargs.copy()
        fit_kwargs["weights"] = None

        coeffs = target_basis.from_grid_values(
            grid_values,
            target_grid,
            vector_type,
            **fit_kwargs,
        )
        return coeffs
