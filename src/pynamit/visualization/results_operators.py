"""Explicit operator bundles for postprocessing and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pynamit.math.constants import mu0
from pynamit.math.linear_map import as_linear_map
from pynamit.primitives.grid import Grid
from pynamit.simulation.spatial.geometry_utils import to_dense


def _get_setting_attr(settings: Any, key: str, default: Any) -> Any:
    """Read a settings value from attrs or object attributes."""
    if hasattr(settings, "attrs") and key in settings.attrs:
        return settings.attrs[key]
    return getattr(settings, key, default)


@dataclass(frozen=True)
class PoloidalResultsOperators:
    """Explicit postprocessing operators on one target grid."""

    RI: float
    scalar_evaluation_matrix: np.ndarray
    m_ind_to_Br: np.ndarray
    m_imp_to_jr: np.ndarray
    m_ind_to_Jeq: np.ndarray
    T_to_Ve: Optional[np.ndarray]
    G_Ve_to_JS: np.ndarray
    G_B_tor_to_JS: np.ndarray
    G_m_ind_to_JS: np.ndarray
    G_m_imp_to_JS: np.ndarray
    G_Br_to_JS: np.ndarray

    @property
    def G_B_pol_to_JS(self) -> np.ndarray:
        """Compatibility alias for the poloidal JS operator."""
        return self.G_Ve_to_JS

    def evaluate_scalar_coefficients(
        self,
        coeffs: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Evaluate scalar coefficients directly on the target grid."""
        coeff_vec = float(scale) * np.asarray(coeffs).reshape(-1)
        return np.asarray(self.scalar_evaluation_matrix @ coeff_vec).reshape(-1)

    def evaluate_scalar_operator(
        self,
        operator: np.ndarray,
        coeffs: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``scalar_evaluation_matrix @ operator @ coeffs`` on the target grid."""
        coeff_vec = np.asarray(coeffs).reshape(-1)
        return np.asarray(self.scalar_evaluation_matrix @ (np.asarray(operator) @ coeff_vec)).reshape(-1)

    def evaluate_br(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate radial magnetic perturbation on the target grid."""
        return self.evaluate_scalar_operator(self.m_ind_to_Br, m_ind)

    def evaluate_jr(self, m_imp: np.ndarray) -> np.ndarray:
        """Evaluate radial current density on the target grid."""
        return self.evaluate_scalar_operator(self.m_imp_to_jr, m_imp)

    def evaluate_jeq(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate equivalent current function on the target grid."""
        return self.evaluate_scalar_operator(self.m_ind_to_Jeq, m_ind)

    @staticmethod
    def _evaluate_vector_operator(operator: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply a structured vector operator of shape ``(2, n_grid, n_coeff)``."""
        coeff_vec = np.asarray(coeffs).reshape(-1)
        return np.asarray(np.tensordot(np.asarray(operator), coeff_vec, axes=([2], [0])))

    def evaluate_js_from_m_ind(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate structured sheet current from poloidal induced coefficients."""
        return self._evaluate_vector_operator(self.G_m_ind_to_JS, m_ind)

    def evaluate_js_from_m_imp(self, m_imp: np.ndarray) -> np.ndarray:
        """Evaluate structured sheet current from toroidal/imposed coefficients."""
        return self._evaluate_vector_operator(self.G_m_imp_to_JS, m_imp)


def build_poloidal_results_operators(
    *,
    basis: Any,
    grid: Grid,
    RI: float,
    T_to_Ve: Optional[np.ndarray] = None,
    RM: Optional[float] = None,
) -> PoloidalResultsOperators:
    """Build explicit postprocessing operators on ``grid``.

    This bundle is intended for notebooks and visualization tools that want
    direct dense operators without re-deriving the underlying formulas.
    """
    L = int(basis.index_length)
    scalar_evaluation_matrix = np.asarray(to_dense(basis.get_evaluation_matrix(grid)), dtype=float)

    laplacian_operator = np.asarray(to_dense(basis.get_laplacian_operator(RI)), dtype=float)
    m_imp_to_jr = (float(RI) / mu0) * laplacian_operator
    m_ind_to_Br = -(float(RI) ** 2) * laplacian_operator

    potential_scaling = np.asarray(to_dense(basis.get_potential_scaling_operator()), dtype=float)
    m_ind_to_Jeq = (-float(RI) / mu0) * potential_scaling

    curl_operator = as_linear_map(basis.get_curl_matrix(grid))
    gradient_operator = as_linear_map(basis.get_gradient_matrix(grid))
    scaling_operator = as_linear_map(basis.get_potential_scaling_operator())

    G_Ve_to_JS = np.asarray(
        to_dense(((-1.0 / mu0) * (curl_operator @ scaling_operator))).reshape(2, -1, L),
        dtype=float,
    )
    G_B_tor_to_JS = np.asarray(
        to_dense(((-1.0 / mu0) * gradient_operator)).reshape(2, -1, L),
        dtype=float,
    )

    T_to_Ve_matrix = None if T_to_Ve is None else np.asarray(T_to_Ve, dtype=float)
    if T_to_Ve_matrix is None:
        G_m_imp_to_JS = G_B_tor_to_JS.copy()
    else:
        G_m_imp_to_JS = G_B_tor_to_JS + np.tensordot(G_Ve_to_JS, T_to_Ve_matrix, axes=([2], [0]))

    G_m_ind_to_JS = G_Ve_to_JS.copy()
    G_Br_to_JS = np.zeros_like(G_Ve_to_JS)

    RM_value = None if RM in (None, 0) else float(RM)
    if RM_value is not None:
        required = ("laplacian", "radial_shift_Ve", "radial_shift_Vi")
        if not all(hasattr(basis, attr) for attr in required):
            raise TypeError(
                "RM-corrected results operators require a basis that provides "
                "laplacian(), radial_shift_Ve(), and radial_shift_Vi()."
            )

        lap_factors = np.asarray(basis.laplacian(RI), dtype=float).reshape(-1)
        br_shift = np.asarray(basis.radial_shift_Ve(RM_value, RI), dtype=float).reshape(-1)
        vi_shift = np.asarray(basis.radial_shift_Vi(RI, RM_value), dtype=float).reshape(-1)
        den_safe = np.where(np.abs(1.0 - br_shift * vi_shift) < 1e-15, np.nan, 1.0 - br_shift * vi_shift)
        m_ind_to_Br_factors = -(float(RI) ** 2) * lap_factors
        br_scale = np.where(
            np.abs(m_ind_to_Br_factors) > 0.0,
            -(br_shift / den_safe) / m_ind_to_Br_factors,
            0.0,
        )
        ind_scale = 1.0 + (br_shift * vi_shift) / den_safe

        G_Br_to_JS = G_Ve_to_JS * br_scale[None, None, :]
        G_m_ind_to_JS = G_Ve_to_JS * ind_scale[None, None, :]

    return PoloidalResultsOperators(
        RI=float(RI),
        scalar_evaluation_matrix=scalar_evaluation_matrix,
        m_ind_to_Br=np.asarray(m_ind_to_Br, dtype=float),
        m_imp_to_jr=np.asarray(m_imp_to_jr, dtype=float),
        m_ind_to_Jeq=np.asarray(m_ind_to_Jeq, dtype=float),
        T_to_Ve=T_to_Ve_matrix,
        G_Ve_to_JS=G_Ve_to_JS,
        G_B_tor_to_JS=G_B_tor_to_JS,
        G_m_ind_to_JS=np.asarray(G_m_ind_to_JS, dtype=float),
        G_m_imp_to_JS=np.asarray(G_m_imp_to_JS, dtype=float),
        G_Br_to_JS=np.asarray(G_Br_to_JS, dtype=float),
    )


def build_poloidal_results_operators_from_settings(
    settings: Any,
    *,
    basis: Any,
    grid: Grid,
    T_to_Ve: Optional[np.ndarray] = None,
) -> PoloidalResultsOperators:
    """Build explicit postprocessing operators from a settings object or dataset."""
    RI = float(_get_setting_attr(settings, "RI", 0.0))
    if RI <= 0.0:
        raise ValueError("Settings must provide a positive RI value.")

    RM = _get_setting_attr(settings, "RM", None)
    return build_poloidal_results_operators(
        basis=basis,
        grid=grid,
        RI=RI,
        T_to_Ve=T_to_Ve,
        RM=RM,
    )


def build_poloidal_results_operators_from_simulation_data(
    simulation_data: Any,
    *,
    grid: Grid,
    basis: Any = None,
) -> PoloidalResultsOperators:
    """Build explicit postprocessing operators from a ``SimulationData`` object."""
    return build_poloidal_results_operators_from_settings(
        simulation_data.settings,
        basis=simulation_data.sh_basis_zero_removed if basis is None else basis,
        grid=grid,
        T_to_Ve=simulation_data.pfac_matrix,
    )
