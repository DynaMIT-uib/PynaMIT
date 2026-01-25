"""Linear operators for ionospheric simulation.

This module provides specialized linear operators used in the
ionospheric electrodynamics simulation, including tensor operators
for resistivity/conductivity handling.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from pynamit.utils import asarray, xp, to_numpy
from pynamit.math.linear_map import LinearMap, as_linear_map


BlockOperator = Union[np.ndarray, LinearMap, None]


class BlockCoupledOperator:
    """Block 2x2 coupled system operator.

    Represents a block matrix structure:
        [[L_00, L_01],
         [L_10, L_11]]

    where each block L_ij operates on vectors of size N, and the full
    operator acts on vectors of size 2*N (representing [y0, y1]).

    This enables matrix-free operation when the blocks are LinearMaps,
    without materializing the full (2N, 2N) matrix.

    Parameters
    ----------
    L_00 : np.ndarray, LinearMap, or None
        Block (0,0) operator mapping y0 → output0. None means zero block.
    L_01 : np.ndarray, LinearMap, or None
        Block (0,1) operator mapping y1 → output0. None means zero block.
    L_10 : np.ndarray, LinearMap, or None
        Block (1,0) operator mapping y0 → output1. None means zero block.
    L_11 : np.ndarray, LinearMap, or None
        Block (1,1) operator mapping y1 → output1. None means zero block.
    n : int
        Size of each block (N). The full operator has shape (2N, 2N).

    Attributes
    ----------
    shape : tuple
        Shape of the operator as (2*N, 2*N).
    dtype : np.dtype
        Data type of the operator.
    """

    def __init__(
        self,
        L_00: BlockOperator,
        L_01: BlockOperator,
        L_10: BlockOperator,
        L_11: BlockOperator,
        n: int,
    ) -> None:
        """Initialize with block operators."""
        self.n = n
        self.shape = (2 * n, 2 * n)

        # Convert to LinearMaps where applicable
        self.blocks = [
            [self._wrap_block(L_00, n), self._wrap_block(L_01, n)],
            [self._wrap_block(L_10, n), self._wrap_block(L_11, n)],
        ]

        # Determine dtype from first non-None block
        self.dtype = np.float64
        for row in self.blocks:
            for block in row:
                if block is not None:
                    self.dtype = block.dtype
                    break

    @staticmethod
    def _wrap_block(block: BlockOperator, n: int) -> Optional[LinearMap]:
        """Wrap a block as a LinearMap, or return None for zero blocks."""
        if block is None:
            return None
        if isinstance(block, LinearMap):
            return block
        # Convert numpy array to LinearMap
        arr = asarray(block)
        if arr.shape != (n, n):
            raise ValueError(f"Block must have shape ({n}, {n}), got {arr.shape}")
        return as_linear_map(arr)

    def _apply_block(self, block: Optional[LinearMap], x: Any) -> Any:
        """Apply a block to a vector, returning zeros if block is None."""
        if block is None:
            return xp.zeros(self.n, dtype=self.dtype)
        return block.matvec(x)

    def _apply_block_T(self, block: Optional[LinearMap], y: Any) -> Any:
        """Apply adjoint of a block, returning zeros if block is None."""
        if block is None:
            return xp.zeros(self.n, dtype=self.dtype)
        return block.rmatvec(y)

    def matvec(self, x: Any) -> Any:
        """Apply operator to a vector.

        Parameters
        ----------
        x : array-like
            Input vector of shape (2*N,) or (2, N).

        Returns
        -------
        array-like
            Output vector of shape (2*N,).
        """
        x_arr = asarray(x).reshape(2, self.n)
        y0 = self._apply_block(self.blocks[0][0], x_arr[0])
        y0 = y0 + self._apply_block(self.blocks[0][1], x_arr[1])
        y1 = self._apply_block(self.blocks[1][0], x_arr[0])
        y1 = y1 + self._apply_block(self.blocks[1][1], x_arr[1])
        return xp.concatenate([y0, y1])

    def rmatvec(self, y: Any) -> Any:
        """Apply adjoint operator to a vector.

        Parameters
        ----------
        y : array-like
            Input vector of shape (2*N,) or (2, N).

        Returns
        -------
        array-like
            Output vector of shape (2*N,).
        """
        y_arr = asarray(y).reshape(2, self.n)
        # Adjoint of block matrix: transpose the block structure
        x0 = self._apply_block_T(self.blocks[0][0], y_arr[0])
        x0 = x0 + self._apply_block_T(self.blocks[1][0], y_arr[1])
        x1 = self._apply_block_T(self.blocks[0][1], y_arr[0])
        x1 = x1 + self._apply_block_T(self.blocks[1][1], y_arr[1])
        return xp.concatenate([x0, x1])

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
        X_arr = asarray(X)
        if X_arr.ndim == 1:
            return self.matvec(X_arr)
        # Apply column by column
        results = [self.matvec(X_arr[:, i]) for i in range(X_arr.shape[1])]
        return xp.stack(results, axis=1)

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
        Y_arr = asarray(Y)
        if Y_arr.ndim == 1:
            return self.rmatvec(Y_arr)
        results = [self.rmatvec(Y_arr[:, i]) for i in range(Y_arr.shape[1])]
        return xp.stack(results, axis=1)

    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix representation.

        Returns
        -------
        np.ndarray
            Dense matrix of shape (2*N, 2*N).
        """
        n = self.n
        result = np.zeros((2 * n, 2 * n), dtype=self.dtype)

        for i in range(2):
            for j in range(2):
                block = self.blocks[i][j]
                if block is not None:
                    result[i * n:(i + 1) * n, j * n:(j + 1) * n] = block.to_dense()

        return result

    def to_linear_map(self) -> LinearMap:
        """Convert to LinearMap for use with solvers.

        Returns
        -------
        LinearMap
            A LinearMap wrapping this operator.
        """
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
