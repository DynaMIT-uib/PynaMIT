"""Linear operators for ionospheric simulation.

This module provides specialized linear operators used in the
ionospheric electrodynamics simulation, including tensor operators
for resistivity/conductivity handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pynamit.utils import asarray, xp, to_numpy


class ResistivityTensorOperator:
    """Block-diagonal resistivity tensor operator.

    Wraps a 2x2 tensor field (2, 2, N) and applies it as a linear operator
    on flattened vector fields (2*N). This represents the resistivity tensor
    η that relates electric field E to current density J via E = η·J.

    The tensor is applied point-wise:
        [E_θ]   [η_θθ  η_θφ] [J_θ]
        [E_φ] = [η_φθ  η_φφ] [J_φ]

    Parameters
    ----------
    eta : np.ndarray
        Resistivity tensor of shape (2, 2, N) where N is the number of
        grid points. The first two dimensions are the tensor components
        (θθ, θφ, φθ, φφ).

    Attributes
    ----------
    shape : tuple
        Shape of the operator as (2*N, 2*N).
    dtype : dtype
        Data type of the tensor.
    """

    def __init__(self, eta: np.ndarray) -> None:
        """Initialize with resistivity tensor."""
        self.eta = asarray(eta)
        self.n = eta.shape[2]
        self.shape = (2 * self.n, 2 * self.n)
        self.dtype = eta.dtype

    def matvec(self, x: Any) -> Any:
        """Apply operator to a vector.

        Parameters
        ----------
        x : array-like
            Input vector of shape (2*N,).

        Returns
        -------
        array-like
            Output vector of shape (2*N,).
        """
        x_reshaped = asarray(x).reshape(2, self.n)
        # η is (2, 2, N), x is (2, N)
        # y_i = Σ_j η_ij * x_j  for each grid point
        y = xp.einsum("ijk,jk->ik", self.eta, x_reshaped)
        return y.reshape(-1)

    def rmatvec(self, y: Any) -> Any:
        """Apply adjoint operator to a vector.

        Parameters
        ----------
        y : array-like
            Input vector of shape (2*N,).

        Returns
        -------
        array-like
            Output vector of shape (2*N,).
        """
        y_reshaped = asarray(y).reshape(2, self.n)
        # Adjoint: transpose tensor in first two dims
        res = xp.einsum("jik,jk->ik", self.eta, y_reshaped)
        return res.reshape(-1)

    def matmat(self, X: Any) -> Any:
        """Apply operator to multiple vectors.

        Parameters
        ----------
        X : array-like
            Input matrix of shape (2*N, K).

        Returns
        -------
        array-like
            Output matrix of shape (2*N, K).
        """
        cols = X.shape[1]
        X_reshaped = asarray(X).reshape(2, self.n, cols)
        res = xp.einsum("ijk,jkl->ikl", self.eta, X_reshaped)
        return res.reshape(2 * self.n, cols)

    def rmatmat(self, Y: Any) -> Any:
        """Apply adjoint operator to multiple vectors.

        Parameters
        ----------
        Y : array-like
            Input matrix of shape (2*N, K).

        Returns
        -------
        array-like
            Output matrix of shape (2*N, K).
        """
        cols = Y.shape[1]
        Y_reshaped = asarray(Y).reshape(2, self.n, cols)
        res = xp.einsum("jik,jkl->ikl", self.eta, Y_reshaped)
        return res.reshape(2 * self.n, cols)

    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix representation.

        Returns a (2N, 2N) block-diagonal matrix where each 2x2 block
        corresponds to the tensor at one grid point.

        Returns
        -------
        np.ndarray
            Dense matrix of shape (2*N, 2*N).
        """
        eta_np = to_numpy(self.eta)
        d00 = np.diag(eta_np[0, 0])
        d01 = np.diag(eta_np[0, 1])
        d10 = np.diag(eta_np[1, 0])
        d11 = np.diag(eta_np[1, 1])
        return np.block([[d00, d01], [d10, d11]])

    def to_linear_map(self) -> "LinearMap":
        """Convert to LinearMap for operator composition.

        Returns
        -------
        LinearMap
            A LinearMap wrapping this operator.
        """
        from pynamit.math.linear_map import LinearMap

        return LinearMap(
            shape=self.shape,
            dtype=self.dtype,
            _matvec=self.matvec,
            _rmatvec=self.rmatvec,
            _matmat=self.matmat,
            _rmatmat=self.rmatmat,
            _to_dense=self.to_dense,
            source=self,
        )
