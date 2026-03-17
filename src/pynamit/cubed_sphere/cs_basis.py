"""Cubed sphere basis module.

This module contains the CSBasis class for representing the cubed sphere
basis.
"""

from __future__ import annotations
from typing import Any, Tuple, Optional, Callable, Dict, TYPE_CHECKING
import functools
import importlib
import numpy as np
import os
from scipy.special import binom
from scipy.sparse import coo_matrix

from pynamit.cubed_sphere import diffutils
from pynamit.cubed_sphere import cs_math
from pynamit.primitives.grid import Grid, GridBasis, create_interpolator
from pynamit.primitives.grid.grid_utils import get_3D_determinants, constrain_values
from pynamit.utils import asarray

if TYPE_CHECKING:
    from pynamit.cubed_sphere.grid import CubedSphereGrid
    from pynamit.simulation.spatial import Geometry

d2r = np.pi / 180
datapath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


class CSBasis(GridBasis):
    """Class for representing cubed sphere bases.

    This module provides an implementation of the cubed sphere grid
    system following methods from Yin et al. (2017).
    """

    _GLOBAL_PROJECTION_CACHE: Dict[Tuple[int, int, int, str], Any] = {}

    def __init__(self, N: int):
        """Initialize the cubed sphere basis."""
        super().__init__()

        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N % 2 != 0:
            raise ValueError("Cubed sphere grid dimension must be even")

        self.N = N
        k, i, j = self.get_gridpoints(N)

        # Initialize grid points (flattened)
        self.arr_xi: np.ndarray = self.xi(i[:, :-1, :-1] + 0.5, N).flatten()
        self.arr_eta: np.ndarray = self.eta(j[:, :-1, :-1] + 0.5, N).flatten()
        self.arr_block: np.ndarray = k[:, :-1, :-1].flatten()

        # Convert to spherical coordinates using inherited method
        _, self.arr_theta, self.arr_phi = cs_math.cube2spherical(
            self.arr_xi, self.arr_eta, self.arr_block, deg=True
        )

        # Initialize Grid object (Essential for GridBasis compatibility)
        self.grid = Grid(theta=self.arr_theta, phi=self.arr_phi)
        # Attach weights for compatibility with numerical integrators (like SH path)
        self.grid.weights = self.unit_area

        # Initialize optimized interpolator
        from pynamit.primitives.grid.interpolation import CSInterpolator

        self._interpolator = CSInterpolator(N)
        self._mimetic_laplacian_cache: Dict[Tuple[int, float], np.ndarray] = {}
        self._mimetic_laplacian_pinv_cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    @property
    def kind(self) -> str:
        return "CS"

    def get_laplacian_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the Laplacian operator for CSBasis."""
        from pynamit.math.linear_map import as_linear_map

        return as_linear_map(self.laplacian(r))

    def get_gradient_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical gradient operator.

        Returns a LinearMap that maps scalar potential coefficients to
        vector field components: E = -grad(φ) = (-d_θ φ, -(1/sin θ) d_φ φ) / r
        """
        return self._get_grid_gradient_operator(self.grid, r=r)

    def get_curl_operator(self, r: float = 1.0) -> "LinearMap":
        """Get the analytical toroidal operator ``-r x grad``.

        Returns a LinearMap that maps scalar potential coefficients to
        vector field components: E = -r × grad(ψ) = ((1/sin θ) d_φ ψ, -d_θ ψ) / r
        """
        return self._get_grid_curl_operator(self.grid, r=r)

    def get_vector_curl_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the discrete radial curl operator on the grid."""
        from pynamit.math.linear_map import as_linear_map

        target_grid = grid if grid is not None else self.grid
        return as_linear_map(self._get_grid_curl(target_grid, r=1.0))

    def get_vector_divergence_operator(self, grid: Optional[Any] = None) -> "LinearMap":
        """Get the discrete divergence operator on the grid."""
        from pynamit.math.linear_map import as_linear_map

        target_grid = grid if grid is not None else self.grid
        return as_linear_map(self._get_grid_divergence(target_grid, r=1.0))

    def _extract_helmholtz_channel(self, coeffs: np.ndarray, channel: int) -> np.ndarray:
        """Extract a potential channel from canonical Helmholtz coefficients.

        CSBasis now follows the same canonical coefficient convention as SHBasis:
            coeffs[0] -> poloidal potential coefficients
            coeffs[1] -> toroidal potential coefficients
        """
        coeffs = asarray(coeffs)
        n = self.index_length

        if coeffs.ndim >= 2 and coeffs.shape[0] == 2:
            return coeffs[channel]

        if coeffs.ndim >= 1 and coeffs.shape[0] == 2 * n:
            if coeffs.ndim == 1:
                return coeffs.reshape(2, n)[channel]
            return coeffs.reshape((2, n) + coeffs.shape[1:])[channel]

        raise ValueError(
            "Full Helmholtz field must have leading size 2 or 2*Ncoeffs; "
            f"got shape {coeffs.shape}."
        )

    @staticmethod
    def _default_pinv_rcond(shape: Tuple[int, ...]) -> float:
        """Return deterministic pseudo-inverse cutoff based on machine precision."""
        dim_max = max(int(v) for v in shape) if len(shape) > 0 else 1
        return float(np.finfo(float).eps * max(dim_max, 1))

    @staticmethod
    def _pinv_symmetric(a: np.ndarray, rcond: float) -> np.ndarray:
        """Robust pseudoinverse for symmetric matrices using eigen decomposition."""
        a_np = np.asarray(a)
        if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
            return np.linalg.pinv(a_np, rcond=max(float(rcond), 0.0))

        a_sym = 0.5 * (a_np + a_np.T.conj())
        rcond = max(float(rcond), 0.0)
        try:
            eigvals, eigvecs = np.linalg.eigh(a_sym)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(a_sym, rcond=rcond)

        max_abs = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            return np.zeros_like(a_sym)
        cutoff = rcond * max_abs
        inv_eigs = np.where(np.abs(eigvals) > cutoff, 1.0 / eigvals, 0.0)
        return (eigvecs * inv_eigs) @ eigvecs.T.conj()

    def get_constant_mode_vector(self, n_coeff: Optional[int] = None) -> np.ndarray:
        """Return the canonical constant-mode vector in CS coefficient space."""
        n = self.index_length if n_coeff is None else int(n_coeff)
        return np.ones(n, dtype=float)

    def get_mean_zero_projector(self, n_coeff: Optional[int] = None) -> np.ndarray:
        """Return projector onto the mean-zero scalar subspace."""
        z = self.get_constant_mode_vector(n_coeff).reshape(-1)
        norm2 = float(np.dot(z, z))
        if norm2 <= 0.0:
            return np.eye(z.size, dtype=float)
        return np.eye(z.size, dtype=float) - np.outer(z, z) / norm2

    def get_helmholtz_gauge_constraint_matrix(self, n_coeff: Optional[int] = None) -> np.ndarray:
        """Return hard gauge rows for Helmholtz inversion.

        Enforces zero-mean poloidal and toroidal potentials independently.
        """
        n = self.index_length if n_coeff is None else int(n_coeff)
        z = self.get_constant_mode_vector(n).reshape(-1)
        z_norm = float(np.linalg.norm(z))
        if z_norm <= 0.0:
            z = np.zeros_like(z)
        else:
            z = z / z_norm
        C = np.zeros((2, 2 * n), dtype=float)
        C[0, :n] = z
        C[1, n:] = z
        return C

    def get_scalar_gauge_constraint_matrix(
        self, n_coeff: Optional[int] = None, mode: str = "pin_first"
    ) -> np.ndarray:
        """Return a single scalar gauge constraint row.

        Parameters
        ----------
        n_coeff : int, optional
            Number of coefficients; defaults to this basis size.
        mode : str
            Gauge row type:
            - ``"pin_first"``: first coefficient is fixed to zero.
            - ``"mean_zero"``: zero-mean constraint against constant mode.
        """
        n = self.index_length if n_coeff is None else int(n_coeff)
        if mode == "mean_zero":
            z = self.get_constant_mode_vector(n).astype(float, copy=False).reshape(1, -1)
            z_norm = float(np.linalg.norm(z))
            if z_norm > 0.0:
                z = z / z_norm
            return z

        row = np.zeros((1, n), dtype=float)
        row[0, 0] = 1.0
        return row

    def get_scalar_gauge_projector_for_operator(
        self, operator: np.ndarray, mode: str = "pin_first", rcond: Optional[float] = None
    ) -> np.ndarray:
        """Return projector that enforces scalar gauge while preserving operator image.

        Parameters
        ----------
        operator : np.ndarray
            Forward operator where gauge mode should lie in/near the null space.
        mode : str
            Gauge policy: ``"pin_first"`` or ``"mean_zero"``.
        rcond : float, optional
            Pseudoinverse cutoff for fallback null-space extraction.
        """
        A = np.asarray(operator)
        n = int(A.shape[1])
        I = np.eye(n, dtype=A.dtype)
        if mode in ("none", "", None):
            return I
        if mode == "mean_zero":
            return self.get_mean_zero_projector(n_coeff=n).astype(A.dtype, copy=False)

        # pin_first: subtract only along null-space direction(s) so Ax is preserved.
        z_const = (
            self.get_constant_mode_vector(n_coeff=n).astype(A.dtype, copy=False).reshape(-1, 1)
        )
        rel_const_null = np.linalg.norm(A @ z_const) / max(
            np.linalg.norm(A) * np.linalg.norm(z_const), 1e-30
        )
        if rel_const_null < 1e-6:
            null_basis = z_const
        else:
            _, s_vals, vh = np.linalg.svd(A, full_matrices=False)
            if s_vals.size == 0:
                return I
            if rcond is None:
                rcond = self._default_pinv_rcond(A.shape)
            cutoff = max(float(rcond), 0.0) * float(s_vals[0])
            null_mask = s_vals <= cutoff
            if not np.any(null_mask):
                return I
            null_basis = vh[null_mask].T

        pin_row = np.zeros((1, n), dtype=A.dtype)
        pin_row[0, 0] = 1.0
        pin_on_null = pin_row @ null_basis
        if np.linalg.norm(pin_on_null) <= 0:
            return I
        pin_on_null_pinv = np.linalg.pinv(pin_on_null, rcond=max(float(rcond or 0.0), 0.0))
        return I - (null_basis @ pin_on_null_pinv @ pin_row)

    def get_toroidal_potential_coeffs(
        self, coeffs: np.ndarray, grid: Optional[Any] = None
    ) -> np.ndarray:
        """Extract toroidal potential coefficients from Helmholtz coefficients."""
        return self._extract_helmholtz_channel(coeffs, channel=1)

    def get_poloidal_potential_coeffs(
        self, coeffs: np.ndarray, grid: Optional[Any] = None
    ) -> np.ndarray:
        """Extract poloidal potential coefficients from Helmholtz coefficients."""
        return self._extract_helmholtz_channel(coeffs, channel=0)

    def evaluate(self, coeffs: np.ndarray, grid: Any, vector_type: str = "scalar") -> np.ndarray:
        """Evaluate coefficients on a grid.

        For tangential vectors, CSBasis uses Helmholtz potentials as coefficients,
        consistent with SHBasis and the rest of the induction operators.
        """
        if vector_type == "scalar":
            return self.basis_to_grid(coeffs, grid, helmholtz=False)
        if vector_type == "tangential":
            if coeffs.ndim == 1:
                coeffs = coeffs.reshape(2, -1)
            return self.basis_to_grid(coeffs, grid, helmholtz=True)
        raise ValueError(f"Unknown vector_type: {vector_type}")

    def from_grid_values(
        self, values: np.ndarray, grid: Any, vector_type: str, **kwargs
    ) -> np.ndarray:
        """Project grid values to CS coefficients.

        For tangential vectors, returns Helmholtz potential coefficients.
        """
        weights = kwargs.get("weights")
        reg_lambda = kwargs.get("reg_lambda")
        pinv_rtol = kwargs.get("pinv_rtol", 1e-15)
        solver_type = kwargs.get("solver_type", "normal_eq")

        if vector_type == "scalar":
            return self.grid_to_basis(
                values,
                grid,
                helmholtz=False,
                weights=weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                solver_type=solver_type,
            )
        if vector_type == "tangential":
            return self.grid_to_basis(
                values,
                grid,
                helmholtz=True,
                weights=weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                solver_type=solver_type,
            )
        raise ValueError(f"Unknown vector_type: {vector_type}")

    def get_radial_shift_operator(
        self, start_r: float, end_r: float, kind: str = "external"
    ) -> "LinearMap":
        """Get the radial shift operator. Default to global SH-like scaling for now."""
        from pynamit.math.linear_map import diagonal_linear_map

        # Grid basis can follow same potential radial scaling as SH (physics-based)
        # We assume each point scales like a generic potential term.
        # This is strictly true only for SH, but a reasonable "grid" approximation
        # for potential fields if we don't have a better model.
        if kind == "external":
            # For Ve, usually n-dependent. If we don't have n, we assume a representative n=1?
            # Or we use a safe default of 1.0 if not supported.
            factor = start_r / end_r
        else:
            factor = (start_r / end_r) ** 2

        return diagonal_linear_map(np.ones(self.index_length) * factor)

    def get_potential_scaling_operator(self) -> "LinearMap":
        """Get potential-to-surface scaling operator for CS scalar coefficients.

        In CS basis the scalar coefficients are nodal values of the scalar
        potential on the grid, so no additional spectral degree scaling is
        applied here.
        """
        from pynamit.math.linear_map import diagonal_linear_map

        return diagonal_linear_map(np.ones(self.index_length))

    @property
    def size(self):
        """Number of grid points."""
        return self.index_length

    @property
    def theta(self):
        return self.arr_theta

    @property
    def phi(self):
        return self.arr_phi

    @functools.cached_property
    def g(self) -> np.ndarray:
        """Metric tensor."""
        return cs_math.get_metric_tensor(self.arr_xi, self.arr_eta)

    @functools.cached_property
    def sqrt_detg(self) -> np.ndarray:
        """Square root of determinant of the metric tensor."""
        return np.sqrt(get_3D_determinants(self.g))

    @functools.cached_property
    def unit_area(self) -> np.ndarray:
        """Area of each grid cell."""
        step = np.diff(self.xi(np.array([0, 1]), self.N))[0]
        return step**2 * self.sqrt_detg

    def get_gridpoints(self, N, flat=False):
        """Generate grid point indices for given resolution."""
        k, i, j = np.meshgrid(np.arange(6), np.arange(N + 1), np.arange(N + 1), indexing="ij")
        if flat:
            return k.flatten(), i.flatten(), j.flatten()
        else:
            return k, i, j

    def xi(self, i, N):
        """Calculate xi coordinate for grid index."""
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + i * np.pi / (2 * N)

    def eta(self, j, N):
        """Calculate eta coordinate for grid index."""
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N < 1:
            raise ValueError("N must be at least 1")
        return -np.pi / 4 + j * np.pi / (2 * N)

    # Methods get_delta through get_Q are inherited from GridBasis (removed here)

    def get_Diff(self, N, coordinate="xi", Ns=1, Ni=4, order=1):
        """Get scalar field differentiation matrix."""
        if coordinate not in ["xi", "eta", "both"]:
            raise ValueError(
                f'coordinate must be either "xi", "eta", or "both". Not {coordinate}.'
            )
        if Ns < order:
            raise ValueError("Ns must be >= order. You gave {} and {}".format(Ns, order))
        # if order != 1:
        #    raise NotImplementedError("Only first order differentiation is supported.")

        shape = (6, N, N)
        size = 6 * N * N
        h = self.xi(1, N) - self.xi(0, N)
        k, i, j = map(
            np.ravel, np.meshgrid(np.arange(6), np.arange(N), np.arange(N), indexing="ij")
        )

        stencil_points = np.hstack((np.r_[-Ns:0], np.r_[1 : Ns + 1]))
        Nsp = len(stencil_points)
        stencil_weight = diffutils.stencil(stencil_points, order=order, h=h)

        i_diff = np.hstack([i + _ for _ in stencil_points])
        j_diff = np.hstack([j + _ for _ in stencil_points])
        k_const, i_const, j_const = (np.tile(k, Nsp), np.tile(i, Nsp), np.tile(j, Nsp))
        weights = np.repeat(stencil_weight, size)

        rows = np.tile(np.ravel_multi_index((k, i, j), shape), Nsp)
        if coordinate in ["xi", "both"]:
            Dxi = self.get_interpolation_matrix(
                k_const, i_diff, j_const, N, Ni, rows=rows, weights=weights
            )
        if coordinate in ["eta", "both"]:
            Deta = self.get_interpolation_matrix(
                k_const, i_const, j_diff, N, Ni, rows=rows, weights=weights
            )

        if coordinate == "both":
            return (Dxi, Deta)
        if coordinate == "xi":
            return Dxi
        if coordinate == "eta":
            return Deta

    def get_interpolation_matrix(self, k, i, j, N, Ni, weights=None, rows=None):
        """Get matrix for grid to cubed sphere interpolation."""
        if Ni > N:
            raise ValueError("Ni must be <= N")
        k, i, j = map(np.ravel, [k, i, j])
        shape, size = (6, N, N), 6 * N**2
        if rows is None:
            rows = np.arange(k.size)
        if weights is None:
            weights = np.ones(k.size)
        weights = weights / Ni
        h = self.xi(1, N) - self.xi(0, N)
        cols = np.full(k.size, -1, dtype=np.int64)

        xi, eta = self.xi(i + 0.5, N), self.eta(j + 0.5, N)
        r, theta, phi = cs_math.cube2spherical(xi, eta, k, r=1.0, deg=True)
        new_xi, new_eta, new_k = cs_math.geo2cube(phi, 90 - theta)
        new_i, new_j = new_xi / h + (N - 1) / 2, new_eta / h + (N - 1) / 2

        assert np.all(
            (np.isclose(new_i - np.rint(new_i), 0) | np.isclose(new_j - np.rint(new_j), 0))
        )
        ii_integers = np.isclose(new_i - np.rint(new_i), 0) & np.isclose(new_j - np.rint(new_j), 0)
        cols[ii_integers] = np.ravel_multi_index(
            (
                new_k[ii_integers],
                np.rint(new_i[ii_integers]).astype(np.int64),
                np.rint(new_j[ii_integers]).astype(np.int64),
            ),
            shape,
        )

        i_is_float = ~np.isclose(np.rint(new_i) - new_i, 0)
        j_is_float = ~np.isclose(np.rint(new_j) - new_j, 0)
        assert sum(i_is_float & j_is_float) == 0
        j_floats = new_j[j_is_float].reshape((-1, 1))
        i_floats = new_i[i_is_float].reshape((-1, 1))

        interpolation_points = np.arange(Ni).reshape((1, -1))
        # Center interpolation stencil around the floating index.
        # For even Ni, using ``-Ni//2`` gives the expected symmetric
        # neighborhood around the floor/ceil pair; the previous ``-Ni//2-1``
        # introduced a one-cell left shift and degraded cross-face accuracy.
        j_interpolation_points = constrain_values(
            interpolation_points + np.int64(np.ceil(j_floats)) - Ni // 2, 0, N - 1, axis=1
        )
        i_interpolation_points = constrain_values(
            interpolation_points + np.int64(np.ceil(i_floats)) - Ni // 2, 0, N - 1, axis=1
        )

        j_distances = j_floats - j_interpolation_points
        i_distances = i_floats - i_interpolation_points
        w = (-1) ** interpolation_points * binom(Ni - 1, interpolation_points)
        w_i = w / i_distances / np.sum(w / i_distances, axis=1).reshape((-1, 1))
        w_j = w / j_distances / np.sum(w / j_distances, axis=1).reshape((-1, 1))

        stacked_weights = np.tile(weights, (Ni, 1)).T
        stacked_cols = np.tile(cols, (Ni, 1)).T
        stacked_rows = np.tile(rows, (Ni, 1)).T

        stacked_cols[i_is_float] = np.ravel_multi_index(
            (
                np.tile(new_k[i_is_float], (Ni, 1)).T,
                i_interpolation_points,
                np.rint(np.tile(new_j[i_is_float], (Ni, 1))).astype(np.int64).T,
            ),
            shape,
        )
        stacked_cols[j_is_float] = np.ravel_multi_index(
            (
                np.tile(new_k[j_is_float], (Ni, 1)).T,
                np.rint(np.tile(new_i[j_is_float], (Ni, 1))).astype(np.int64).T,
                j_interpolation_points,
            ),
            shape,
        )
        stacked_weights[i_is_float] = stacked_weights[i_is_float] * w_i * Ni
        stacked_weights[j_is_float] = stacked_weights[j_is_float] * w_j * Ni

        D = coo_matrix(
            (stacked_weights.flatten(), (stacked_rows.flatten(), stacked_cols.flatten())),
            shape=(rows.max() + 1, size),
        )
        D.count_nonzero()
        return D

    @staticmethod
    def _safe_sin_theta(theta_deg: np.ndarray) -> np.ndarray:
        """Return sin(theta) with pole-safe floor for metric scaling."""
        theta_rad = np.deg2rad(np.asarray(theta_deg).flatten())
        sin_th = np.sin(theta_rad)
        epsilon = 1e-10
        return np.where(np.abs(sin_th) < epsilon, epsilon, sin_th)

    def _is_grid_compatible(self, grid: Any) -> bool:
        """Check whether grid ordering matches this native CS basis ordering."""
        if grid is self:
            return True
        if hasattr(grid, "kind") and grid.kind == "CS" and getattr(grid, "N", -1) == self.N:
            return True
        if hasattr(grid, "theta") and hasattr(grid, "phi"):
            return grid == self.grid
        return False

    def _get_native_derivative_bundle(self) -> Dict[str, Any]:
        """Build native CS angular derivative operators and metric scalings."""
        bundle = self._cache.get("_native_derivative_bundle")
        if bundle is not None:
            return bundle

        import scipy.sparse

        Dxi, Deta = self.get_Diff(self.N, coordinate="both", Ns=1, Ni=4, order=1)
        dxi_dth, dxi_dph, deta_dth, deta_dph = cs_math.get_coordinate_derivatives(
            self.arr_xi, self.arr_eta, r=1.0, block=self.arr_block
        )

        D_theta = (
            scipy.sparse.diags(dxi_dth.flatten()) @ Dxi
            + scipy.sparse.diags(deta_dth.flatten()) @ Deta
        )
        D_phi_unscaled = (
            scipy.sparse.diags(dxi_dph.flatten()) @ Dxi
            + scipy.sparse.diags(deta_dph.flatten()) @ Deta
        )

        sin_th_safe = self._safe_sin_theta(self.arr_theta)
        sin_th = scipy.sparse.diags(sin_th_safe)
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        inv_sin2_th = scipy.sparse.diags(1.0 / (sin_th_safe**2))
        D_phi_scaled = inv_sin_th @ D_phi_unscaled

        bundle = {
            "D_theta": D_theta,
            "D_phi_unscaled": D_phi_unscaled,
            "D_phi_scaled": D_phi_scaled,
            "sin_th": sin_th,
            "inv_sin_th": inv_sin_th,
            "inv_sin2_th": inv_sin2_th,
        }
        self._cache["_native_derivative_bundle"] = bundle
        return bundle

    def _get_grid_derivative_bundle(self, grid: Any) -> Dict[str, Any]:
        """Return angular derivative bundle on arbitrary grid.

        Public phi-derivative semantics remain SH-compatible:
        ``D_phi_scaled`` represents ``(1/sin(theta)) * d/dphi``.
        ``D_phi_unscaled`` is used internally for Laplacian/div/curl assembly.
        """
        if self._is_grid_compatible(grid):
            return self._get_native_derivative_bundle()

        grid_key = getattr(grid, "hash", id(grid))
        grid_cache = self._cache.setdefault(grid_key, {})
        if "derivative_bundle" in grid_cache:
            return grid_cache["derivative_bundle"]

        import scipy.sparse

        native = self._get_native_derivative_bundle()
        interpolation_matrix = self._get_arbitrary_interpolation_matrix(grid, Ni=4)
        D_theta = interpolation_matrix @ native["D_theta"]
        D_phi_unscaled = interpolation_matrix @ native["D_phi_unscaled"]

        sin_th_safe = self._safe_sin_theta(grid.theta)
        sin_th = scipy.sparse.diags(sin_th_safe)
        inv_sin_th = scipy.sparse.diags(1.0 / sin_th_safe)
        inv_sin2_th = scipy.sparse.diags(1.0 / (sin_th_safe**2))
        D_phi_scaled = inv_sin_th @ D_phi_unscaled

        bundle = {
            "D_theta": D_theta,
            "D_phi_unscaled": D_phi_unscaled,
            "D_phi_scaled": D_phi_scaled,
            "sin_th": sin_th,
            "inv_sin_th": inv_sin_th,
            "inv_sin2_th": inv_sin2_th,
            "interpolation_matrix": interpolation_matrix,
        }
        grid_cache["derivative_bundle"] = bundle
        return bundle

    def get_evaluation_matrix(self, grid, derivative=None):
        """Get evaluation or differentiation matrix on the grid.

        Parameters
        ----------
        grid : object
            Grid object. Must match this basis's grid.
        derivative : str, optional
            'theta', 'phi', or None.

        Returns
        -------
        G : sparse matrix
            Evaluation or differentiation matrix.
        """
        import scipy.sparse

        target_grid = self if grid is None else grid
        is_compatible = self._is_grid_compatible(target_grid)

        if derivative is None:
            if is_compatible:
                return scipy.sparse.identity(6 * self.N * self.N, format="csr")

            grid_key = getattr(target_grid, "hash", id(target_grid))
            grid_cache = self._cache.setdefault(grid_key, {})
            if "interpolation_matrix" not in grid_cache:
                grid_cache["interpolation_matrix"] = self._get_arbitrary_interpolation_matrix(
                    target_grid, Ni=4
                )
            return grid_cache["interpolation_matrix"]

        if derivative not in ("theta", "phi"):
            raise ValueError(f"Unknown derivative: {derivative}")

        bundle = self._get_grid_derivative_bundle(target_grid)
        if derivative == "theta":
            return bundle["D_theta"]
        return bundle["D_phi_scaled"]

    def _get_arbitrary_interpolation_matrix(self, grid, Ni=4):
        """Construct interpolation matrix for arbitrary target points (2D Tensor Product)."""
        import scipy.sparse

        N = self.N
        # 1. Map target grid to CS coordinates
        xi_tgt, eta_tgt, k_tgt = cs_math.geo2cube(grid.phi, 90.0 - grid.theta)

        # 2. Fractional indices
        h_scaling = (2 * N) / np.pi
        i_tgt = (xi_tgt + np.pi / 4) * h_scaling
        j_tgt = (eta_tgt + np.pi / 4) * h_scaling

        # 3. Base integer indices (floor)
        i_base = np.floor(i_tgt).astype(int)
        j_base = np.floor(j_tgt).astype(int)

        # 4. Prepare Stencil Iteration
        # Stencil range relative to base: e.g. -1, 0, 1, 2 for Ni=4
        start = -(Ni // 2) + 1  # -1 for Ni=4
        stencil_offsets = np.arange(start, start + Ni)

        rows = []
        cols = []
        data = []

        n_targets = i_tgt.size
        target_indices = np.arange(n_targets)

        # Pre-calculate distances for weight computation
        # Lagrange weights for tensor product
        # w_total(di, dj) = w(di) * w(dj)

        # Loop over the 2D stencil (Ni x Ni)
        for di in stencil_offsets:
            for dj in stencil_offsets:
                # A. Identify Source Candidates (on the definition face k_tgt)
                i_cand = i_base + di
                j_cand = j_base + dj

                # B. Map Candidates to Valid Grid Nodes (Handle Face Crossing)
                # Convert (k_tgt, i_cand, j_cand) -> (xi_cand, eta_cand)
                # Note: This xi/eta might be outside [-pi/4, pi/4]
                xi_cand = -np.pi / 4 + i_cand * np.pi / (2 * N)
                eta_cand = -np.pi / 4 + j_cand * np.pi / (2 * N)

                # Map through sphere to canonical CS coordinates
                r, th, ph = cs_math.cube2spherical(xi_cand, eta_cand, k_tgt, r=1.0, deg=True)
                xi_prim, eta_prim, k_prim = cs_math.geo2cube(ph, 90.0 - th)

                # Convert back to indices on the new face
                i_prim = np.rint((xi_prim + np.pi / 4) * h_scaling).astype(int)
                j_prim = np.rint((eta_prim + np.pi / 4) * h_scaling).astype(int)

                # Clamp strict limits to avoid floating point noise issues at corners?
                # geo2cube should handle it, but indices must be 0..N
                i_prim = np.clip(i_prim, 0, N)
                j_prim = np.clip(j_prim, 0, N)

                # Flattened Column Indices
                # shape (6, N, N) -> (6, N+1, N+1)?
                # CSBasis typically N+1?
                # self.get_gridpoints(N) uses N+1.
                # get_gridpoints(flat=True) returns size 6*(N+1)*(N+1).
                # Wait, CSBasis index length?
                # line 49: k, i, j = self.get_gridpoints(N)
                # line 52: i[:, :-1, :-1].
                # It drops the last row/col?
                # This implies "Centered" or "Element-based"?
                # Yin et al method usually LGL nodes or similar.
                # Lines 52-54: i[:, :-1, :-1] + 0.5.
                # This implies CELL CENTRED basis? Or nodes at 0.5?
                # "xi = -pi/4 + (i+0.5)*..."
                # If basis is defined at cell centers, N is number of cells.

                # My `i_tgt` calculation used `i_tgt = (xi + pi/4) / h`.
                # If xi = -pi/4 + (i_idx + 0.5)*h
                # Then xi + pi/4 = (i_idx + 0.5)*h
                # i_tgt (from xi) = i_idx + 0.5.
                # So `i_tgt` is shifted by 0.5 relative to integer indices?

                # Let's check `xi` method:
                # `return -np.pi / 4 + i * np.pi / (2 * N)`
                # AND init uses `i[:, :-1, :-1] + 0.5`.
                # So the STORED grid is at half-indices. 0.5, 1.5, ...

                # To align `i_tgt` with integer grid storage indices `0, 1, 2...` (representing 0.5, 1.5...):
                # i_grid_idx = i_tgt - 0.5.
                # Let's adjust i_tgt in Step 2.

                # Re-check.
                # Basis stored at `indices` 0..N-1?
                # `self.arr_xi` derived from `i + 0.5`.
                # If I want to interpolate from values at `i+0.5`,
                # My coordinate `u` mapping to index `idx`:
                # `u = u0 + (idx + 0.5) * h`.
                # `u - u0 = idx*h + 0.5*h`
                # `(u-u0)/h = idx + 0.5`
                # `idx = (u-u0)/h - 0.5`.

                # So my `i_tgt` calculation (which was `(xi-xi0)/h`) yields `idx + 0.5`.
                # So I should subtract 0.5 to get `idx`.
                # Correct.

                # C. Compute Weights (Lagrange Interpolation for Cell-Centered Grid)
                # Nodes are at indices like 0.5, 1.5, etc.
                # i_tgt is in these units.
                # Relative to i_base, nodes are at (offset + 0.5)
                # Target is at delta = (i_tgt - i_base)

                # w_i(di) = Product_{m in stencil, m!=di} (delta - (m+0.5)) / ((di+0.5) - (m+0.5))
                #         = Product_{m!=di} (delta - m - 0.5) / (di - m)

                delta_i = i_tgt - i_base
                delta_j = j_tgt - j_base

                w_i = np.ones_like(delta_i)
                w_j = np.ones_like(delta_j)

                for m in stencil_offsets:
                    if m != di:
                        w_i *= (delta_i - m - 0.5) / (di - m)
                    if m != dj:
                        w_j *= (delta_j - m - 0.5) / (dj - m)

                weight = w_i * w_j

                # D. Store in Sparse Lists
                # flattened index for col: k, i, j
                # Clip to valid index range 0..N-1 just in case
                i_prim = np.clip(i_prim, 0, N - 1)
                j_prim = np.clip(j_prim, 0, N - 1)

                col_indices = np.ravel_multi_index((k_prim, i_prim, j_prim), (6, N, N))

                rows.append(target_indices)
                cols.append(col_indices)
                data.append(weight)

        # 5. Build Final Matrix
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        from scipy.sparse import coo_matrix

        return coo_matrix((data, (rows, cols)), shape=(n_targets, 6 * N * N))

    def laplacian(self, r=1.0):
        """Compute the Laplacian operator matrix on the sphere.

        Approximated using the strong form:
        nabla^2 V = (1/r^2 sin(theta)) * d/dtheta (sin(theta) dV/dtheta)
                  + (1/r^2 sin^2(theta)) * d^2V/dphi^2

        Returns
        -------
        L : scipy.sparse.spmatrix
            Sparse matrix representing the Laplacian operator.
        """
        bundle = self._get_native_derivative_bundle()

        # Strong-form scalar Laplacian:
        #   Δf = (1/sinθ) dθ(sinθ dθ f) + (1/sin²θ) d²φ f
        # Public phi derivative is scaled as (1/sinθ) d/dφ, but this term must
        # be assembled from the unscaled d/dφ operator to avoid extra metric
        # scaling in the discrete CS operator.
        term1 = bundle["inv_sin_th"] @ bundle["D_theta"] @ bundle["sin_th"] @ bundle["D_theta"]
        term2 = bundle["inv_sin2_th"] @ bundle["D_phi_unscaled"] @ bundle["D_phi_unscaled"]
        return (term1 + term2) / (r**2)

    def get_mimetic_laplacian_operator(
        self, grid: Optional[Any] = None, r: float = 1.0
    ) -> np.ndarray:
        """Return scalar Laplacian from discrete div/grad composition.

        Uses:
            ``Delta = -Div @ Grad``
        with the same CS vector operators used throughout the code path.
        """
        target_grid = self.grid if grid is None else grid
        grid_key = int(getattr(target_grid, "hash", id(target_grid)))
        key = (grid_key, float(r))
        cached = self._mimetic_laplacian_cache.get(key)
        if cached is not None:
            return cached

        div_op = np.asarray(self.get_vector_divergence_operator(target_grid).to_dense())
        grad_op = np.asarray(self._get_grid_gradient_operator(target_grid, r=r).to_dense())
        lap = -(div_op @ grad_op)
        lap = np.asarray(0.5 * (lap + lap.T))
        self._mimetic_laplacian_cache[key] = lap
        return lap

    def get_mimetic_laplacian_pinv(
        self, grid: Optional[Any] = None, r: float = 1.0, rcond: Optional[float] = None
    ) -> np.ndarray:
        """Return mean-zero gauge-fixed pseudoinverse of mimetic Laplacian."""
        lap = np.asarray(self.get_mimetic_laplacian_operator(grid=grid, r=r))
        if rcond is None:
            rcond = self._default_pinv_rcond(lap.shape)
        key = (
            int(
                getattr(
                    grid if grid is not None else self.grid,
                    "hash",
                    id(grid if grid is not None else self.grid),
                )
            ),
            float(r),
            float(max(rcond, 0.0)),
        )
        cached = self._mimetic_laplacian_pinv_cache.get(key)
        if cached is not None:
            return cached

        P = self.get_mean_zero_projector(n_coeff=lap.shape[0]).astype(lap.dtype, copy=False)
        lap_proj = P @ lap @ P
        lap_proj_pinv = self._pinv_symmetric(lap_proj, rcond=max(float(rcond), 0.0))
        lap_pinv = P @ lap_proj_pinv @ P
        self._mimetic_laplacian_pinv_cache[key] = lap_pinv
        return lap_pinv

    # Methods block, geo2cube, interpolate_scalar, interpolate_vector_components inherited

    def get_projected_coastlines(self, resolution="50m"):
        """Generate coastlines in projected coordinates."""
        coastlines = np.load(datapath + "coastlines_" + resolution + ".npz")
        for key in coastlines:
            lat, lon = coastlines[key]
            yield cs_math.geo2cube(lon, lat)

    def interpolate_to_self(self, values, theta, phi, vector_type="scalar"):
        """Interpolate values to this basis's grid."""
        if vector_type == "scalar":
            return self.interpolate_scalar(values, theta, phi, self.arr_theta, self.arr_phi)
        elif vector_type == "tangential":
            u_east = values[1]
            u_north = -values[0]
            u_r = np.zeros_like(u_north)
            u_e, u_n, _ = self.interpolate_vector_components(
                u_east, u_north, u_r, theta, phi, self.arr_theta, self.arr_phi
            )
            return np.vstack((-u_n, u_e))
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

    def interpolate_scalar(self, val, th_src, ph_src, th_tgt, ph_tgt):
        """Spherical interpolation for scalars."""
        # Use optimized interpolator if source matches this basis
        # Wrap inputs in Grid for comparison logic (leveraging fast caching)
        src_grid = Grid(theta=th_src, phi=ph_src)
        if src_grid == self.grid:
            return self._interpolator.interpolate_scalar(val, th_tgt, ph_tgt)

        # Fallback to generic interpolation
        interp = create_interpolator(th_src, ph_src)
        return interp.interpolate_scalar(val, th_tgt, ph_tgt)

    def interpolate_vector_components(self, u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt):
        """Interpolate vector components."""
        # Use optimized interpolator if source matches this basis
        src_grid = Grid(theta=th_src, phi=ph_src)
        if src_grid == self.grid:
            return self._interpolator.interpolate_vector(u_east, u_north, u_r, th_tgt, ph_tgt)

        # Fallback to generic interpolation
        interp = create_interpolator(th_src, ph_src)
        return interp.interpolate_vector(u_east, u_north, u_r, th_tgt, ph_tgt)

    def project_to_basis(
        self, input_values, input_grid, vector_type, target_grid, target_basis, **kwargs
    ):
        """Project input data onto the target basis."""
        from pynamit.primitives.projection_pipeline import interpolate_then_project_batch

        if target_grid is None:
            raise ValueError("target_grid must be provided")

        return interpolate_then_project_batch(
            input_values,
            input_grid=input_grid,
            vector_type=vector_type,
            target_grid=target_grid,
            target_basis=target_basis,
            scalar_interpolator=lambda values: self.interpolate_scalar(
                values, input_grid.theta, input_grid.phi, target_grid.theta, target_grid.phi
            ),
            vector_interpolator=lambda u_east, u_north, u_r: self.interpolate_vector_components(
                u_east,
                u_north,
                u_r,
                input_grid.theta,
                input_grid.phi,
                target_grid.theta,
                target_grid.phi,
            ),
            fit_kwargs=kwargs,
        )

    def construct_projection_matrix(self, grid) -> Any:
        """Construct the projection matrix mapping Grid Vector -> Scalar Potentials.

        For CSBasis in cs_dominant mode, the state variables (Phi, W) are
        scalar potentials. This matrix performs Helmholtz decomposition
        on the grid to extract them.

        Uses an exact equality-constrained least-squares solve with
        basis-level gauge constraints.
        """
        # 1. Check cache (mode-aware)
        grid_key = getattr(grid, "hash", id(grid))
        solver_kind = os.getenv("PYNAMIT_CS_PROJECTION_SOLVER", "normal_eq").strip().lower()
        if solver_kind not in {"normal_eq", "cgls", "lsmr", "svd"}:
            solver_kind = "normal_eq"
        mode_key = f"constrained:{solver_kind}"

        grid_cache = self._cache.setdefault(grid_key, {})
        by_mode = grid_cache.setdefault("projection_matrix_by_mode", {})
        if mode_key in by_mode:
            return by_mode[mode_key]

        global_key = (int(self.N), int(getattr(grid, "size", 0)), int(grid_key), mode_key)
        if global_key in CSBasis._GLOBAL_PROJECTION_CACHE:
            res = CSBasis._GLOBAL_PROJECTION_CACHE[global_key]
            by_mode[mode_key] = res
            grid_cache["projection_matrix"] = res
            return res

        # Get gradient operators
        G_th = self.get_evaluation_matrix(grid, derivative="theta")
        G_ph = self.get_evaluation_matrix(grid, derivative="phi")

        if hasattr(G_th, "toarray"):
            G_th = G_th.toarray()
        if hasattr(G_ph, "toarray"):
            G_ph = G_ph.toarray()

        n_grid, n_coeff = G_th.shape

        # Build forward Helmholtz mapping using the canonical tensor layout.
        # This keeps flattening/ordering identical to the legacy tensor_pinv path.
        G_grad = np.array([-G_th, -G_ph])
        G_rxgrad = np.array([G_ph, -G_th])
        G_helmholtz = np.stack([G_grad, G_rxgrad], axis=2)
        A = G_helmholtz.reshape(2 * n_grid, 2 * n_coeff)

        # Equality-constrained solve with hard gauge rows.
        if n_coeff <= 0:
            raise ValueError(f"Cannot construct CS Helmholtz projection with n_coeff={n_coeff}.")

        n_total = 2 * n_coeff
        m_total = 2 * n_grid

        from pynamit.math.least_squares_problem import LeastSquaresProblem
        from pynamit.math.least_squares_solver import LeastSquaresSolver

        C = self.get_helmholtz_gauge_constraint_matrix(n_coeff=n_coeff)
        problem = LeastSquaresProblem(
            A=[A], solution_shape=(n_total,), data_shapes=[(m_total,)], matrix_free=False
        )
        try:
            ls_solver = LeastSquaresSolver(solver=solver_kind, tolerance=1e-15)
            rhs = np.eye(m_total, dtype=float)
            sol = ls_solver.solve(
                problem,
                [rhs],
                equality_operator=C,
                equality_rhs=np.zeros((2, m_total), dtype=float),
                elimination_rcond=1e-15,
            )
            P_flat = np.asarray(sol).reshape(n_total, m_total)
            if P_flat.shape != (n_total, m_total):
                raise RuntimeError(
                    "Unexpected constrained projection shape: "
                    f"{P_flat.shape}, expected {(n_total, m_total)}."
                )
            res = P_flat.reshape(2, n_coeff, 2, n_grid)
        except Exception as exc:
            raise RuntimeError(
                "CS constrained Helmholtz projection solve failed. "
                "No unconstrained fallback is enabled."
            ) from exc

        # 2. Store in Cache
        by_mode[mode_key] = res
        grid_cache["projection_matrix"] = res
        CSBasis._GLOBAL_PROJECTION_CACHE[global_key] = res

        return res

    def _get_grid_divergence(self, grid: Any, r: float = 1.0) -> Any:
        """Get the discrete divergence operator matrix on the grid."""
        import scipy.sparse

        bundle = self._get_grid_derivative_bundle(grid)

        # Div = (1/r sin th) [ d_th (E_th sin th) + d_ph E_ph ]
        #
        # The phi block is assembled as
        #   (1/sin²θ) d/dphi [sinθ * E_phi]
        # which is algebraically equivalent to (1/sinθ) d/dphi(E_phi) when
        # sin(theta) is phi-independent. This form keeps the discrete CS
        # div/grad/curl operators consistent with the Laplacian assembly.
        D_th = bundle["inv_sin_th"] @ bundle["D_theta"] @ bundle["sin_th"]
        D_ph = bundle["inv_sin2_th"] @ bundle["D_phi_unscaled"] @ bundle["sin_th"]

        return scipy.sparse.hstack([D_th, D_ph]) / r

    def _get_grid_curl(self, grid: Any, r: float = 1.0) -> Any:
        """Get the discrete radial curl operator matrix on the grid."""
        import scipy.sparse

        bundle = self._get_grid_derivative_bundle(grid)

        # Curl_r = (1/r sin th) [ d_th (E_ph sin th) - d_ph E_th ]
        #
        # Use the same phi block form as divergence to preserve consistency with
        # the scalar Laplacian when composing curl(curl psi).
        C_th = -(bundle["inv_sin2_th"] @ bundle["D_phi_unscaled"] @ bundle["sin_th"])
        C_ph = bundle["inv_sin_th"] @ bundle["D_theta"] @ bundle["sin_th"]

        return scipy.sparse.hstack([C_th, C_ph]) / r

    def _get_grid_gradient_operator(self, grid: Any, r: float = 1.0) -> "LinearMap":
        """Get gradient operator mapping spectral potential to vector grid field.

        Returns a LinearMap that computes E = -grad(φ) = (-d_θ φ, -(1/sin θ) d_φ φ) / r
        The returned operator maps from scalar coefficients to stacked vector components.
        """
        from pynamit.math.linear_map import as_linear_map, block_linear_map

        bundle = self._get_grid_derivative_bundle(grid)

        # E = -grad(phi) = (-d_th phi, -1/sin_th * d_ph phi) / r
        # D_phi_scaled already includes 1/sin(theta) scaling.
        op_phi_th = as_linear_map(bundle["D_theta"]) * (-1.0 / r)
        op_phi_ph = as_linear_map(bundle["D_phi_scaled"]) * (-1.0 / r)

        return block_linear_map([[op_phi_th], [op_phi_ph]])

    def _get_grid_curl_operator(self, grid: Any, r: float = 1.0) -> "LinearMap":
        """Get curl operator mapping spectral potential to vector grid field.

        Returns a LinearMap that computes E = -r × grad(ψ) = ((1/sin θ) d_φ ψ, -d_θ ψ) / r
        The returned operator maps from scalar coefficients to stacked vector
        components. This is also the repo-wide df Helmholtz sign convention,
        opposite to the ``+r x grad`` df basis used in Laundal et al. (2025)
        Appendix C1 for generic tangential vector fields.
        """
        from pynamit.math.linear_map import as_linear_map, block_linear_map

        bundle = self._get_grid_derivative_bundle(grid)

        # E = -r x grad(psi) = (1/sin_th * d_ph psi, -d_th psi) / r
        # D_phi_scaled already includes 1/sin(theta) scaling.
        op_psi_th = as_linear_map(bundle["D_phi_scaled"]) * (1.0 / r)
        op_psi_ph = as_linear_map(bundle["D_theta"]) * (-1.0 / r)

        return block_linear_map([[op_psi_th], [op_psi_ph]])

    def get_extended_basis(self) -> "CSBasis":
        """Return a basis extended to include the monopole term.

        For CSBasis, the basis already includes all grid points.
        """
        return self

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Get vector basis evaluation matrix.

        For Cubed Sphere, we now use the Helmholtz decomposition as the vector
        representation: [Poloidal Potential; Toroidal Potential].

        This method returns the matrix G such that E_grid = G @ [phi_coeffs; psi_coeffs].
        G = [ G_pol, G_tor ] where G_pol maps potential to -grad(phi)
        and G_tor maps potential to -r x grad(psi).

        So, as in the SH path, the CS Helmholtz basis is ``[-grad, -r x grad]``.
        Relative to Laundal et al. (2025) Appendix C1, only the df channel is
        sign-flipped.

        Returns
        -------
        np.ndarray
            Canonical Helmholtz tensor with shape ``(2, N_grid, 2, N_coeffs)``.
        """
        from pynamit.math.linear_map import block_linear_map

        # Use our grid-based Helmholtz operators (at r=1.0 for the basis definition)
        G_pol = self._get_grid_gradient_operator(grid, r=1.0)
        G_tor = self._get_grid_curl_operator(grid, r=1.0)

        # Combine into [G_pol, G_tor]
        G_vec = block_linear_map([[G_pol, G_tor]])

        # Return canonical Helmholtz tensor (2, N_grid, 2, N_coeffs)
        G_dense = G_vec.to_dense()
        n_grid = grid.size if hasattr(grid, "size") else self.arr_theta.size
        return G_dense.reshape(2, n_grid, 2, self.index_length)
