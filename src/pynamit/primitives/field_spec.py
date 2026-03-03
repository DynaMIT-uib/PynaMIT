"""Structural field descriptors.

``FieldSpec`` describes the coefficient space for a field without carrying any
realized values. It is shared by runtime ``Field`` objects and persisted
``Timeseries`` schemas so both layers agree on basis family, field type, and
zero-mean intent.

For SH scalar/tangential fields, ``mean_free=True`` is realized by omitting the
``(n, m) = (0, 0)`` coefficient. For CS scalar fields, ``mean_free=True`` keeps
the full coefficient vector and relies on the CS mean-zero constraint/projector
machinery instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from functools import wraps
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """Structural description of a coefficient-backed field.

    Parameters
    ----------
    basis : Any
        Canonical basis family for the field coefficients.
    field_type : {"scalar", "tangential", "vector"}
        Kind of field represented by the coefficients.
    mean_free : bool, optional
        Whether the field is intended to be zero-mean.
        For SH scalar/tangential spaces this is represented by omitting the
        monopole coefficient. For CS scalar spaces it is represented by keeping
        the full coefficient vector and enforcing mean-zero through
        constraint/projector logic. This is structural metadata about the field
        space, not a solve or projection option.
    """

    basis: Any
    field_type: Literal["scalar", "tangential", "vector"] = "scalar"
    mean_free: bool = False

    @property
    def kind(self) -> str:
        """Return the underlying basis family identifier."""
        return str(self.basis.kind)

    @property
    def signature(self) -> tuple[Any, ...]:
        """Return a stable cache signature for this field specification."""
        basis_signature = getattr(self.basis, "signature", (id(self.basis),))
        return (basis_signature, self.field_type, bool(self.mean_free))

    @property
    def index_names(self) -> list[str]:
        """Return coefficient index names for this field space."""
        return list(self.basis.index_names)

    @property
    def index_arrays(self) -> list[np.ndarray]:
        """Return coefficient index arrays for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            if hasattr(self.basis, "scalar_index_arrays"):
                return list(self.basis.scalar_index_arrays(mean_free=self.mean_free))
        return list(self.basis.index_arrays)

    @property
    def index_length(self) -> int:
        """Return coefficient count for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            if hasattr(self.basis, "scalar_index_length"):
                return int(self.basis.scalar_index_length(mean_free=self.mean_free))
        return int(self.basis.index_length)

    @property
    def n(self) -> np.ndarray:
        """Return SH harmonic degrees for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "scalar_degrees"):
            return np.asarray(self.basis.scalar_degrees(mean_free=self.mean_free))
        return np.asarray(self.basis.n)

    @property
    def m(self) -> np.ndarray:
        """Return SH harmonic orders for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "scalar_orders"):
            return np.asarray(self.basis.scalar_orders(mean_free=self.mean_free))
        return np.asarray(self.basis.m)

    def __getattr__(self, name: str) -> Any:
        """Delegate unhandled attributes to the underlying basis family.

        For SH scalar/tangential spaces, missing callable attributes that
        accept a ``mean_free`` keyword automatically inherit this spec's
        zero-mean semantics unless the caller overrides them explicitly.
        """
        attr = getattr(self.basis, name)
        if not callable(attr):
            return attr
        if self.kind != "SH" or self.field_type not in ("scalar", "tangential"):
            return attr
        try:
            signature = inspect.signature(attr)
        except (TypeError, ValueError):
            return attr
        if "mean_free" not in signature.parameters:
            return attr

        @wraps(attr)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("mean_free", self.mean_free)
            return attr(*args, **kwargs)

        return wrapped

    def scalar_fields_are_mean_free_by_construction(self) -> bool:
        """Return whether zero-mean is built into the coefficient representation."""
        if self.kind == "SH" and self.field_type == "scalar":
            return bool(self.mean_free)
        if hasattr(self.basis, "scalar_fields_are_mean_free_by_construction"):
            return bool(self.basis.scalar_fields_are_mean_free_by_construction())
        return False

    def get_extended_basis(self) -> "FieldSpec":
        """Return the corresponding non-mean-free field space when applicable."""
        if self.kind == "SH":
            return FieldSpec(basis=self.basis, field_type=self.field_type, mean_free=False)
        if hasattr(self.basis, "get_extended_basis"):
            return FieldSpec(
                basis=self.basis.get_extended_basis(),
                field_type=self.field_type,
                mean_free=self.mean_free,
            )
        return self

    def get_evaluation_matrix(self, grid: Any, derivative: str | None = None) -> Any:
        """Return the scalar evaluation matrix for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_evaluation_matrix(
                grid,
                derivative=derivative,
                mean_free=self.mean_free,
            )
        return self.basis.get_evaluation_matrix(grid, derivative=derivative)

    def get_gradient_matrix(self, grid: Any) -> Any:
        """Return the gradient matrix for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_gradient_matrix(grid, mean_free=self.mean_free)
        return self.basis.get_gradient_matrix(grid)

    def get_curl_matrix(self, grid: Any) -> Any:
        """Return the curl matrix for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_curl_matrix(grid, mean_free=self.mean_free)
        return self.basis.get_curl_matrix(grid)

    def get_vector_basis_matrix(self, grid: Any) -> Any:
        """Return the Helmholtz vector basis matrix for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_vector_basis_matrix(grid, mean_free=self.mean_free)
        return self.basis.get_vector_basis_matrix(grid)

    def get_rxgrad_matrix(self, grid: Any) -> Any:
        """Return the rotated-gradient matrix for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_rxgrad_matrix(grid, mean_free=self.mean_free)
        return self.basis.get_rxgrad_matrix(grid)

    def get_laplacian_operator(self, r: float = 1.0) -> Any:
        """Return the Laplacian operator for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_laplacian_operator(r, mean_free=self.mean_free)
        return self.basis.get_laplacian_operator(r)

    def laplacian(self, r: float = 1.0) -> np.ndarray:
        """Return diagonal SH Laplacian factors for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "laplacian"):
            return np.asarray(self.basis.laplacian(r, mean_free=self.mean_free))
        return np.asarray(self.basis.laplacian(r))

    def get_radial_shift_operator(
        self,
        start_r: float,
        end_r: float,
        kind: str = "external",
    ) -> Any:
        """Return the radial shift operator for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_radial_shift_operator(
                start_r,
                end_r,
                kind=kind,
                mean_free=self.mean_free,
            )
        return self.basis.get_radial_shift_operator(start_r, end_r, kind=kind)

    def radial_shift_Ve(self, start: float, end: float) -> np.ndarray:
        """Return SH external radial-shift factors for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "radial_shift_Ve"):
            return np.asarray(self.basis.radial_shift_Ve(start, end, mean_free=self.mean_free))
        return np.asarray(self.basis.radial_shift_Ve(start, end))

    def radial_shift_Vi(self, start: float, end: float) -> np.ndarray:
        """Return SH internal radial-shift factors for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "radial_shift_Vi"):
            return np.asarray(self.basis.radial_shift_Vi(start, end, mean_free=self.mean_free))
        return np.asarray(self.basis.radial_shift_Vi(start, end))

    def get_potential_scaling_operator(self) -> Any:
        """Return the potential-scaling operator for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.get_potential_scaling_operator(mean_free=self.mean_free)
        return self.basis.get_potential_scaling_operator()

    def grid_to_basis_fast(
        self,
        data: Any,
        theta: np.ndarray,
        phi: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        reg_lambda: float | None = None,
        vector_type: str = "scalar",
    ) -> np.ndarray:
        """Project regular-grid data onto this field space using the fast path.

        For SH scalar/tangential spaces, ``mean_free=True`` is realized by using
        the corresponding reduced SH coefficient space rather than computing a
        full-space transform and trimming afterward.
        """
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            if hasattr(self.basis, "_with_mean_free"):
                sh_basis = self.basis._with_mean_free(bool(self.mean_free))
                return sh_basis.grid_to_basis_fast(
                    data,
                    theta,
                    phi=phi,
                    weights=weights,
                    reg_lambda=reg_lambda,
                    vector_type=vector_type,
                )
        return self.basis.grid_to_basis_fast(
            data,
            theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            vector_type=vector_type,
        )

    def get_regularization_matrix(
        self,
        scalar: bool = True,
        reg_lambda: float | None = None,
    ) -> Any:
        """Return the regularization matrix for this field space."""
        if self.kind == "SH" and hasattr(self.basis, "_with_mean_free"):
            return self.basis._with_mean_free(bool(self.mean_free)).get_regularization_matrix(
                scalar=scalar,
                reg_lambda=reg_lambda,
            )
        return self.basis.get_regularization_matrix(scalar=scalar, reg_lambda=reg_lambda)

    def construct_scalar_projection_matrix(self, grid: Any) -> Any:
        """Construct the scalar analysis matrix for this field space."""
        from pynamit.primitives.analysis import get_scalar_projection_matrix

        return get_scalar_projection_matrix(self, grid)

    def construct_projection_matrix(self, grid: Any) -> Any:
        """Construct the Helmholtz analysis matrix for this field space."""
        from pynamit.primitives.analysis import get_helmholtz_projection_matrix

        return get_helmholtz_projection_matrix(self, grid)

    def get_vector_divergence_operator(self, grid: Any = None) -> Any:
        """Return the vector divergence operator for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            from pynamit.spherical_harmonics import sh_operators

            return sh_operators.build_divergence_operator(self, r=1.0)
        return self.basis.get_vector_divergence_operator(grid)

    def get_vector_curl_operator(self, grid: Any = None) -> Any:
        """Return the vector curl operator for this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            from pynamit.spherical_harmonics import sh_operators

            return sh_operators.build_vector_curl_operator(self, r=1.0)
        return self.basis.get_vector_curl_operator(grid)

    def get_toroidal_potential_coeffs(self, coeffs: np.ndarray, grid: Any = None) -> np.ndarray:
        """Extract toroidal potential coefficients from vector coefficients."""
        coeffs = np.asarray(coeffs)
        n = int(self.index_length)
        if coeffs.shape[0] == 2:
            return coeffs[1]
        if coeffs.shape[0] == 2 * n:
            if coeffs.ndim == 1:
                return coeffs.reshape(2, n)[1]
            return coeffs.reshape(2, n, -1)[1].reshape((n,) + coeffs.shape[1:])
        return self.basis.get_toroidal_potential_coeffs(coeffs, grid=grid)

    def get_poloidal_potential_coeffs(self, coeffs: np.ndarray, grid: Any = None) -> np.ndarray:
        """Extract poloidal potential coefficients from vector coefficients."""
        coeffs = np.asarray(coeffs)
        n = int(self.index_length)
        if coeffs.shape[0] == 2:
            return coeffs[0]
        if coeffs.shape[0] == 2 * n:
            if coeffs.ndim == 1:
                return coeffs.reshape(2, n)[0]
            return coeffs.reshape(2, n, -1)[0].reshape((n,) + coeffs.shape[1:])
        return self.basis.get_poloidal_potential_coeffs(coeffs, grid=grid)

    def from_grid_values(
        self,
        values: np.ndarray,
        grid: Any,
        vector_type: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Project grid values into this field space."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            kwargs.setdefault("mean_free", self.mean_free)
        return self.basis.from_grid_values(values, grid, vector_type, **kwargs)

    def project_to_basis(
        self,
        input_values: np.ndarray,
        input_grid: Any,
        vector_type: str,
        target_grid: Any,
        target_basis: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Project grid values using this field space as the source representation."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            kwargs.setdefault("mean_free", self.mean_free)
        return self.basis.project_to_basis(
            input_values,
            input_grid,
            vector_type,
            target_grid,
            target_basis,
            **kwargs,
        )

    def evaluate(
        self,
        coeffs: np.ndarray,
        grid: Any,
        vector_type: str = "scalar",
    ) -> np.ndarray:
        """Evaluate coefficients in this field space on a grid."""
        if self.kind == "SH" and self.field_type in ("scalar", "tangential"):
            return self.basis.evaluate(
                coeffs,
                grid,
                vector_type=vector_type,
                mean_free=self.mean_free,
            )
        return self.basis.evaluate(coeffs, grid, vector_type=vector_type)

    def get_scaled_matrix(self, grid: Any, factor: Any) -> Any:
        """Return an evaluation matrix scaled by a row or column factor."""
        import scipy.sparse

        G = self.get_evaluation_matrix(grid)
        if np.isscalar(factor):
            return factor * G

        factor_arr = np.asarray(factor).ravel()
        rows, cols = G.shape
        is_sparse = scipy.sparse.issparse(G)
        if factor_arr.size == rows:
            if is_sparse:
                return scipy.sparse.diags(factor_arr) @ G
            return G * factor_arr.reshape(-1, 1)
        if factor_arr.size == cols:
            if is_sparse:
                return G @ scipy.sparse.diags(factor_arr)
            return G * factor_arr
        raise ValueError(
            f"Factor size {factor_arr.size} does not match G shape {G.shape} "
            "for either row or column scaling."
        )
