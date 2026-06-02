"""Geometry module.

This module contains the Geometry class, which encapsulates the spatial
grid, field transforms, magnetic field properties, and interhemispheric
mappings.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.math import as_linear_map
from pynamit.math.backend import block_until_ready, get_array_module, to_jax, to_numpy, use_jax
from pynamit.sphere import Grid
from pynamit.primitives.field_transform import FieldTransform, resolve_sqrt_weights
from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.math.tensor_operations import weighted_tensor_pinv
from pynamit.sphere import Basis, CSBasis, is_sh_basis

logger = logging.getLogger(__name__)


def _compact_operator_array(operator):
    """Return diagonal vector when possible, else dense matrix."""
    op = as_linear_map(operator)
    if op.shape[0] == op.shape[1]:
        try:
            return np.asarray(op.diagonal(backend="numpy")).copy()
        except ValueError:
            pass
    return np.asarray(op.dense(backend="numpy"))


def _diagonal_operator_values(operator):
    """Return diagonal values from a diagonal operator."""
    return np.asarray(as_linear_map(operator).diagonal(backend="numpy")).copy()


def _extended_scalar_basis_for_potential(basis, settings):
    """Return the scalar potential basis including the monopole."""
    del settings
    if is_sh_basis(basis):
        return basis.get_extended_basis()
    return basis


def _dense_operator_matrix(operator, input_length, output_length):
    """Return an explicit dense operator array."""
    op = as_linear_map(
        operator,
        input_shape=(input_length,),
        output_shape=(output_length,),
    )
    return np.asarray(op.dense(backend="numpy"))


class Geometry:
    """Encapsulates the geometric setup for the ionospheric simulation.

    This class manages grids, basis and field evaluators, geometric
    factors derived from the main magnetic field, and interhemispheric
    mappings. It provides a clean interface for the main State class to
    access pre-computed geometric quantities.
    """

    def __init__(
        self,
        basis: Basis,
        cs_basis: CSBasis,
        mainfield: Any,
        settings: Any,
        PFAC_matrix: Optional[xr.DataArray] = None,
        radial_continuation_basis: Optional[Basis] = None,
    ) -> None:
        """Initialize the geometric context."""
        self.basis = basis
        self.radial_continuation_basis = radial_continuation_basis or (
            basis if basis.supports_radial_potential_operators else None
        )
        self.settings = settings
        self.mainfield = mainfield
        self.cs_basis = cs_basis

        # Store relevant settings
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.connect_hemispheres = bool(settings.connect_hemispheres)
        self.latitude_boundary = settings.latitude_boundary
        self.ignore_PFAC = bool(settings.ignore_PFAC)
        self.FAC_integration_steps = settings.FAC_integration_steps
        self.area_weighted_least_squares = bool(
            getattr(settings, "area_weighted_least_squares", False)
        )

        # Initialize core geometric objects
        self._init_evaluators(cs_basis)
        self._init_constraint_mappings()

        # Caches for expensive properties
        self._bP: Optional[np.ndarray] = None
        self._bH: Optional[np.ndarray] = None
        self._bu: Optional[np.ndarray] = None

        # Allow pre-computed PFAC matrix
        if PFAC_matrix is not None:
            self._T_to_Ve = PFAC_matrix
        else:
            self._T_to_Ve: Optional[xr.DataArray] = None

        self._m_ind_to_gridded_JS = None
        self._m_imp_to_gridded_JS = None
        self._Br_to_gridded_JS = None

        if not self.basis.supports_surface_potential_operators:
            raise NotImplementedError(
                f"{type(self.basis).__name__} does not provide surface-potential operators."
            )

        self.surface_laplacian_operator = self.basis.get_surface_laplacian_operator(
            self.RI
        )
        self.helmholtz_curl_free_potential = (
            self.basis.get_helmholtz_curl_free_potential_matrix()
        )
        self.helmholtz_curl_free_potential_operator = (
            self.basis.get_helmholtz_curl_free_potential_operator()
        )
        self.helmholtz_divergence_free_potential = (
            self.basis.get_helmholtz_divergence_free_potential_matrix()
        )
        self.helmholtz_divergence_free_potential_operator = (
            self.basis.get_helmholtz_divergence_free_potential_operator()
        )
        self.m_imp_to_jr_operator = self.RI / mu0 * self.surface_laplacian_operator
        self.m_ind_to_Br_operator = -(self.RI**2) * self.surface_laplacian_operator
        self.m_imp_to_jr = _compact_operator_array(self.m_imp_to_jr_operator)
        self.E_df_to_d_m_ind_dt = 1.0 / self.RI
        self.m_ind_to_Br = _compact_operator_array(self.m_ind_to_Br_operator)
        if self.radial_continuation_basis is None:
            raise NotImplementedError(
                f"{type(self.basis).__name__} requires a radial Laplace continuation "
                "for sheet-current coupling."
            )

        self.horizontal_to_radial_continuation = (
            self._build_horizontal_to_radial_continuation()
        )
        self.radial_continuation_to_horizontal = (
            self._build_radial_continuation_to_horizontal()
        )
        self.radial_boundary_potential_discontinuity = _dense_operator_matrix(
            self.radial_continuation_basis.boundary_potential_discontinuity,
            self.radial_continuation_basis.index_length,
            self.radial_continuation_basis.index_length,
        )
        self.radial_Ve_to_JS = self._build_radial_potential_to_sheet_current()
        self.sheet_current_potential = self._build_sheet_current_potential()
        self.sheet_current_potential_operator = as_linear_map(
            self.sheet_current_potential,
            input_shape=(self.basis.index_length,),
            output_shape=(self.radial_continuation_basis.index_length,),
        )
        self.horizontal_potential_to_gridded_JS = np.tensordot(
            self.radial_Ve_to_JS,
            self.horizontal_to_radial_continuation,
            axes=([2], [0]),
        )

        self._helmholtz_analysis_matrix = None

    def tangential_to_helmholtz(self, vec: np.ndarray) -> np.ndarray:
        """Convert tangential vector field to Helmholtz coeffs."""
        coeffs = np.tensordot(self.helmholtz_analysis_matrix, vec, 2)
        projector = getattr(self.basis, "project_helmholtz_mean_free", None)
        return projector(coeffs) if callable(projector) else coeffs

    @property
    def helmholtz_analysis_matrix(self) -> np.ndarray:
        """Matrix mapping gridded vectors to Helmholtz coefficients."""
        if self._helmholtz_analysis_matrix is None:
            self._helmholtz_analysis_matrix = weighted_tensor_pinv(
                self.field_transform.helmholtz_coeffs_to_gridded_vector,
                sqrt_weights=self.grid_sqrt_weights(vector=True),
                n_leading_flattened=2,
            )
        return self._helmholtz_analysis_matrix

    def _init_evaluators(self, cs_basis: CSBasis) -> None:
        """Set up grid, field transforms, and field evaluators."""
        self.grid = Grid(
            theta=cs_basis.arr_theta,
            phi=cs_basis.arr_phi,
            area_weights=cs_basis.unit_area,
        )
        self.field_transform = FieldTransform(
            FieldSpace.from_basis(self.basis, field_type="scalar"),
            self.grid,
            area_weighted=self.area_weighted_least_squares,
        )
        self.field_transform_zero_added = FieldTransform(
            FieldSpace.from_basis(
                _extended_scalar_basis_for_potential(self.basis, self.settings),
                field_type="scalar",
            ),
            self.grid,
            area_weighted=self.area_weighted_least_squares,
        )
        if self.radial_continuation_basis is None:
            self.radial_continuation_evaluator = None
        elif self.radial_continuation_basis is self.basis:
            self.radial_continuation_evaluator = self.field_transform
        else:
            self.radial_continuation_evaluator = FieldTransform(
                FieldSpace.from_basis(
                    self.radial_continuation_basis, field_type="scalar"
                ),
                self.grid,
                area_weighted=self.area_weighted_least_squares,
            )
        self.b_evaluator = FieldEvaluator(self.mainfield, self.grid, self.RI)

        # Optional evaluators for the conjugate hemisphere
        self.cp_grid = self.cp_field_transform = self.cp_b_evaluator = None
        if self.connect_hemispheres:
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.RI, self.grid.theta, self.grid.phi
            )
            self.cp_grid = Grid(theta=cp_theta, phi=cp_phi)
            self.cp_field_transform = FieldTransform(
                FieldSpace.from_basis(self.basis, field_type="scalar"),
                self.cp_grid,
                area_weighted=self.area_weighted_least_squares,
            )
            self.cp_b_evaluator = FieldEvaluator(self.mainfield, self.cp_grid, self.RI)

    def grid_sqrt_weights(self, *, vector=False):
        """Return grid sqrt weights when area weighting is enabled."""
        return resolve_sqrt_weights(
            self.grid,
            area_weighted=self.area_weighted_least_squares,
            vector=vector,
        )

    def _build_horizontal_to_radial_continuation(self) -> np.ndarray:
        """Project horizontal scalar coefficients into radial space.

        The horizontal basis owns surface operators.  The radial
        continuation basis owns the regular/irregular Laplace solution
        used for boundary sheet-current coupling.  For the CS horizontal
        path this is a grid least-squares projection from CS nodal
        values to the SH continuation coefficients; for the SH path it
        is the identity.
        """
        if self.radial_continuation_basis.coefficients_are_compatible_with(self.basis):
            return np.eye(self.basis.index_length)
        radial_to_grid = self.radial_continuation_evaluator.scalar_coeffs_to_grid
        horizontal_to_grid = self.field_transform.scalar_coeffs_to_grid
        grid_to_radial = weighted_tensor_pinv(
            radial_to_grid,
            sqrt_weights=self.grid_sqrt_weights(),
            n_leading_flattened=1,
        )
        return np.asarray(grid_to_radial @ horizontal_to_grid)

    def _build_radial_continuation_to_horizontal(self) -> np.ndarray:
        """Project radial coefficients to horizontal space."""
        if self.radial_continuation_basis.coefficients_are_compatible_with(self.basis):
            return np.eye(self.basis.index_length)
        horizontal_to_grid = self.field_transform.scalar_coeffs_to_grid
        radial_to_grid = self.radial_continuation_evaluator.scalar_coeffs_to_grid
        grid_to_horizontal = weighted_tensor_pinv(
            horizontal_to_grid,
            sqrt_weights=self.grid_sqrt_weights(),
            n_leading_flattened=1,
        )
        return np.asarray(grid_to_horizontal @ radial_to_grid)

    def _build_radial_potential_to_sheet_current(self) -> np.ndarray:
        """Return radial potential to sheet current on grid."""
        return (1.0 / self.RI) * np.tensordot(
            self.radial_continuation_evaluator.scalar_coeffs_to_gridded_rhat_cross_gradient,
            (-self.RI / mu0) * self.radial_boundary_potential_discontinuity,
            axes=([2], [0]),
        )

    def _build_sheet_current_potential(self) -> np.ndarray:
        """Return horizontal scalar to radial boundary discontinuity."""
        return (
            self.radial_boundary_potential_discontinuity
            @ self.horizontal_to_radial_continuation
        )

    def _init_constraint_mappings(self) -> None:
        """Initialize geometric operators related to constraints."""
        kind = self.mainfield.kind
        if kind == "dipole":
            self.ll_mask = np.abs(self.grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                self.grid.lat, self.grid.lon, (self.RI - 6371e3) * 1e-3
            )
            self.ll_mask = np.abs(mlat) < self.latitude_boundary
        else:
            self.ll_mask = np.zeros(self.grid.size, dtype=bool)

        self.jr_coeffs_to_j_apex = np.asarray(
            self.b_evaluator.radial_to_apex.reshape((-1, 1))
            * self.field_transform.scalar_coeffs_to_grid
        ).copy()
        self.E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres:
            # Modify jr constraint for interhemispheric connection
            jr_coeffs_to_j_apex_cp = np.asarray(
                self.cp_b_evaluator.radial_to_apex.reshape((-1, 1))
                * self.cp_field_transform.scalar_coeffs_to_grid
            )
            self.jr_coeffs_to_j_apex = self.jr_coeffs_to_j_apex - (
                self.ll_mask.reshape((-1, 1)) * jr_coeffs_to_j_apex_cp
            )

            # Create E-field mapping difference operator for constraint
            E_coeffs_to_E_apex = np.einsum(
                "ijk,jklm->iklm",
                self.b_evaluator.horizontal_to_apex,
                np.asarray(self.field_transform.helmholtz_coeffs_to_gridded_vector),
                optimize=True,
            )
            E_coeffs_to_E_apex_cp = np.einsum(
                "ijk,jklm->iklm",
                self.cp_b_evaluator.horizontal_to_apex,
                np.asarray(self.cp_field_transform.helmholtz_coeffs_to_gridded_vector),
                optimize=True,
            )
            self.E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, self.ll_mask]
            )

    @property
    def bP(self) -> np.ndarray:
        """Pedersen geometric factor for conductance tensor."""
        if self._bP is None:
            b_th, b_ph, b_r = self.b_evaluator.btheta, self.b_evaluator.bphi, self.b_evaluator.br
            self._bP = np.array(
                [[b_ph**2 + b_r**2, -b_th * b_ph], [-b_th * b_ph, b_th**2 + b_r**2]]
            )
        return self._bP

    @property
    def bH(self) -> np.ndarray:
        """Hall geometric factor for conductance tensor."""
        if self._bH is None:
            br = self.b_evaluator.br
            self._bH = np.array([[np.zeros_like(br), br], [-br, np.zeros_like(br)]])
        return self._bH

    @property
    def bu(self) -> np.ndarray:
        """Geometric factor for u x B electric field."""
        if self._bu is None:
            Br = self.b_evaluator.Br
            self._bu = -np.array([[np.zeros_like(Br), Br], [-Br, np.zeros_like(Br)]])
        return self._bu

    @property
    def T_to_Ve(self) -> xr.DataArray:
        """Mapping external toroidal (T) to poloidal (Ve) potential."""
        if self._T_to_Ve is None:
            self._build_T_to_Ve()
        return self._T_to_Ve

    def _build_T_to_Ve(self) -> None:
        """Construct the T_to_Ve operator by integrating radially."""
        n = self.basis.index_length
        self._T_to_Ve = xr.DataArray(np.zeros((n, n)), dims=("i", "j"))
        if self.mainfield.kind == "radial" or self.ignore_PFAC:
            return
        if not self.radial_continuation_basis.supports_radial_potential_operators:
            raise NotImplementedError(
                "PFAC integration requires a radial-continuation basis with "
                "radial potential operators."
            )

        rk_steps = np.asarray(self.FAC_integration_steps)
        Delta_k = np.diff(rk_steps)
        rks = rk_steps[:-1] + 0.5 * Delta_k

        if np.any(rks < self.RI):
            raise ValueError(
                "All FAC integration steps must be outside the ionospheric boundary (RI)."
            )
        if self.RM is not None and np.any(rks > self.RM):
            raise ValueError(
                "All FAC integration steps must be inside the magnetospheric boundary (RM)."
            )

        JS_rk_to_radial_Ve_rk = weighted_tensor_pinv(
            self.radial_Ve_to_JS,
            sqrt_weights=self.grid_sqrt_weights(vector=True),
            n_leading_flattened=2,
            rtol=0,
        )
        m_imp_to_jr_coeffs = self.m_imp_to_jr

        for i, rk in enumerate(rks):
            logger.debug("PFAC integration step %d/%d (rk=%s)", i + 1, rks.size, rk)
            theta_mapped, phi_mapped = self.mainfield.map_coords(
                self.RI, rk, self.grid.theta, self.grid.phi
            )
            mapped_grid = Grid(theta=theta_mapped, phi=phi_mapped)
            rk_b_evaluator = FieldEvaluator(self.mainfield, self.grid, rk)
            mapped_b_evaluator = FieldEvaluator(self.mainfield, mapped_grid, self.RI)
            mapped_field_transform = FieldTransform(
                FieldSpace.from_basis(self.basis, field_type="scalar"),
                mapped_grid,
            )

            m_imp_to_jr_grid = mapped_field_transform.contract_scalar_coeffs_to_grid(
                m_imp_to_jr_coeffs
            )
            jr_to_JS_rk = np.array(
                [
                    rk_b_evaluator.Btheta / mapped_b_evaluator.Br,
                    rk_b_evaluator.Bphi / mapped_b_evaluator.Br,
                ]
            )
            m_imp_to_JS_rk = np.einsum("ij,jk->ijk", jr_to_JS_rk, m_imp_to_jr_grid, optimize=True)

            radial_Ve_rk_to_Ve = _diagonal_operator_values(
                self.radial_continuation_basis.external_potential_continuation(
                    rk, self.RI
                )
            ).reshape((-1, 1, 1))
            if self.RM is not None:
                radial_Ve_rk_to_Ve -= (
                    _diagonal_operator_values(
                        self.radial_continuation_basis.external_potential_continuation(
                            self.RM, self.RI
                        )
                    )
                    * _diagonal_operator_values(
                        self.radial_continuation_basis.internal_potential_continuation(
                            rk, self.RM
                        )
                    )
                ).reshape((-1, 1, 1))
                factor = -1.0 / (
                    1.0
                    - _diagonal_operator_values(
                        self.radial_continuation_basis.external_potential_continuation(
                            self.RM, self.RI
                        )
                    )
                    * _diagonal_operator_values(
                        self.radial_continuation_basis.internal_potential_continuation(
                            self.RI, self.RM
                        )
                    )
                )
            else:
                factor = -1.0

            JS_rk_to_radial_Ve = JS_rk_to_radial_Ve_rk * radial_Ve_rk_to_Ve
            if np.ndim(factor) == 0:
                JS_rk_to_radial_Ve *= factor
            else:
                JS_rk_to_radial_Ve *= np.asarray(factor).reshape((-1, 1, 1))
            JS_rk_to_Ve = np.tensordot(
                self.radial_continuation_to_horizontal,
                JS_rk_to_radial_Ve,
                axes=([1], [0]),
            )
            self._T_to_Ve += Delta_k[i] * np.tensordot(
                JS_rk_to_Ve, m_imp_to_JS_rk, axes=2
            )

    # ----- Source coefficients to gridded sheet current -----

    @property
    def m_imp_to_gridded_JS(self) -> np.ndarray:
        """Operator mapping m_imp to gridded sheet current."""
        if self._m_imp_to_gridded_JS is None:
            if use_jax():
                # Keep this on JAX. The result is consumed by JAX next,
                # so a NumPy/OpenBLAS handoff would only add risk.
                toroidal_to_gridded_JS = (
                    -to_jax(self.field_transform.scalar_coeffs_to_gridded_gradient)
                    / mu0
                )
                xp = get_array_module(toroidal_to_gridded_JS)
                PFAC_to_JS = xp.einsum(
                    "ijk,kl->ijl",
                    to_jax(self.horizontal_potential_to_gridded_JS),
                    to_jax(self.T_to_Ve.values),
                    optimize=True,
                )
                self._m_imp_to_gridded_JS = block_until_ready(
                    toroidal_to_gridded_JS + PFAC_to_JS
                )
            else:
                toroidal_to_gridded_JS = (
                    -to_numpy(self.field_transform.scalar_coeffs_to_gridded_gradient)
                    / mu0
                )
                self._m_imp_to_gridded_JS = toroidal_to_gridded_JS + np.tensordot(
                    to_numpy(self.horizontal_potential_to_gridded_JS),
                    to_numpy(self.T_to_Ve.values),
                    axes=([2], [0]),
                )
        return self._m_imp_to_gridded_JS

    @property
    def m_ind_to_gridded_JS(self) -> np.ndarray:
        """Operator mapping m_ind to gridded sheet current."""
        if self._m_ind_to_gridded_JS is None:
            m_ind_to_gridded_JS = self.horizontal_potential_to_gridded_JS.copy()
            if self.RM is not None:
                br_shift = _diagonal_operator_values(
                    self.radial_continuation_basis.external_potential_continuation(
                        self.RM, self.RI
                    )
                )
                vi_shift = _diagonal_operator_values(
                    self.radial_continuation_basis.internal_potential_continuation(
                        self.RI, self.RM
                    )
                )
                den = 1.0 - br_shift * vi_shift
                radial_m_ind_to_Br = _diagonal_operator_values(
                    -(self.RI**2)
                    * self.radial_continuation_basis.get_surface_laplacian_operator(
                        self.RI
                    )
                )
                br_to_radial_potential = (
                    (-br_shift / den / radial_m_ind_to_Br).reshape((-1, 1))
                    * self.horizontal_to_radial_continuation
                )
                self._Br_to_gridded_JS = np.tensordot(
                    self.radial_Ve_to_JS,
                    br_to_radial_potential,
                    axes=([2], [0]),
                )
                m_ind_to_radial_potential = (
                    (1.0 + (br_shift * vi_shift / den)).reshape((-1, 1))
                    * self.horizontal_to_radial_continuation
                )
                m_ind_to_gridded_JS = np.tensordot(
                    self.radial_Ve_to_JS,
                    m_ind_to_radial_potential,
                    axes=([2], [0]),
                )
            self._m_ind_to_gridded_JS = m_ind_to_gridded_JS
        return self._m_ind_to_gridded_JS

    @property
    def Br_to_gridded_JS(self) -> Optional[np.ndarray]:
        """Operator mapping boundary Br to gridded sheet current."""
        if self.RM is None:
            return None
        if self._Br_to_gridded_JS is None:
            _ = self.m_ind_to_gridded_JS
        return self._Br_to_gridded_JS
