"""Explicit operator bundles for postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pynamit.math.constants import mu0
from pynamit.math.linear_map import as_linear_map
from pynamit.primitives.basis import get_repo_df_helmholtz_sign
from pynamit.primitives.grid import Grid
from pynamit.simulation.spatial.geometry_utils import to_dense


@dataclass(frozen=True)
class PoloidalResultsOperators:
    """Explicit postprocessing operators on one target grid.

    The bundle intentionally keeps two different induced-current notions:

    - ``Jeq`` diagnostics: the magnetic equivalent-current function/vector
      implied by the induced poloidal field.
    - ``JS`` operators: the live current-like operators used in the runtime
      conductivity/PFAC chain.

    These should not be conflated; they use different conventions and serve
    different purposes.

    ``Jeq`` is a physical magnetic diagnostic defined by the poloidal field
    jump. ``JS`` is the live current-like quantity used inside the runtime
    conductivity/PFAC chain. The two agree only in special limits and should
    not be used interchangeably.
    """

    RI: float
    scalar_evaluation_matrix: np.ndarray
    m_ind_to_Br: np.ndarray
    m_imp_to_jr: np.ndarray
    m_ind_to_Jeq: np.ndarray
    G_m_ind_to_Jeq_vector: np.ndarray
    T_to_Ve: Optional[np.ndarray]
    G_Ve_to_JS: np.ndarray
    G_B_tor_to_JS: np.ndarray
    G_m_ind_to_JS: np.ndarray
    G_m_imp_to_JS: np.ndarray
    G_Br_to_JS: np.ndarray

    def evaluate_scalar_coefficients(
        self,
        coeffs: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Evaluate scalar coefficients directly on the target grid."""
        coeff_vec = float(scale) * np.asarray(coeffs).reshape(-1)
        return np.asarray(self.scalar_evaluation_matrix @ coeff_vec).reshape(-1)

    def _evaluate_scalar_operator(
        self,
        operator: np.ndarray,
        coeffs: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``scalar_evaluation_matrix @ operator @ coeffs`` on the target grid."""
        coeff_vec = np.asarray(coeffs).reshape(-1)
        return np.asarray(self.scalar_evaluation_matrix @ (np.asarray(operator) @ coeff_vec)).reshape(-1)

    def evaluate_br(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate radial magnetic perturbation on the target grid."""
        return self._evaluate_scalar_operator(self.m_ind_to_Br, m_ind)

    def evaluate_jr(self, m_imp: np.ndarray) -> np.ndarray:
        """Evaluate radial current density on the target grid."""
        return self._evaluate_scalar_operator(self.m_imp_to_jr, m_imp)

    def evaluate_jeq(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate equivalent current function on the target grid."""
        return self._evaluate_scalar_operator(self.m_ind_to_Jeq, m_ind)

    @staticmethod
    def _evaluate_vector_operator(operator: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply a structured vector operator of shape ``(2, n_grid, n_coeff)``."""
        coeff_vec = np.asarray(coeffs).reshape(-1)
        return np.asarray(np.tensordot(np.asarray(operator), coeff_vec, axes=([2], [0])))

    def evaluate_js_from_m_ind(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate the induced runtime ``JS``-like vector from ``m_ind``.

        This is the live induced-current operator used by the conductivity/PFAC
        chain. It is not the same object as the magnetic equivalent current
        diagnostic implied by ``Jeq``.
        """
        return self._evaluate_vector_operator(self.G_m_ind_to_JS, m_ind)

    def evaluate_js_from_m_imp(self, m_imp: np.ndarray) -> np.ndarray:
        """Evaluate structured sheet current from toroidal/imposed coefficients."""
        return self._evaluate_vector_operator(self.G_m_imp_to_JS, m_imp)

    def evaluate_js_from_br(self, br_coeffs: np.ndarray) -> np.ndarray:
        """Evaluate structured sheet current from radial magnetic coefficients."""
        return self._evaluate_vector_operator(self.G_Br_to_JS, br_coeffs)

    def evaluate_runtime_js(
        self,
        *,
        m_ind: Optional[np.ndarray] = None,
        m_imp: Optional[np.ndarray] = None,
        br_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the live runtime ``JS`` vector from any available contributions.

        This combines the same operator pieces used in the conductivity/PFAC
        chain:

        - induced-polodial contribution from ``m_ind``
        - imposed-toroidal contribution from ``m_imp``
        - optional magnetospheric radial-current contribution from ``Br``
        """
        if m_ind is None and m_imp is None and br_coeffs is None:
            raise ValueError("At least one runtime-current contribution must be provided.")

        js = np.zeros(self.G_m_ind_to_JS.shape[:2], dtype=float)
        if m_ind is not None:
            js = js + self.evaluate_js_from_m_ind(m_ind)
        if m_imp is not None:
            js = js + self.evaluate_js_from_m_imp(m_imp)
        if br_coeffs is not None:
            js = js + self.evaluate_js_from_br(br_coeffs)
        return js

    def evaluate_jeq_vector(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate the physical equivalent-current vector from ``m_ind``.

        By definition,
            ``J_eq = rhat x grad_S(Psi_eq)``
        where ``Psi_eq`` is the conventional equivalent-current function given
        by :meth:`evaluate_jeq`.
        """
        return self._evaluate_vector_operator(self.G_m_ind_to_Jeq_vector, m_ind)


def build_poloidal_results_operators(
    *,
    basis: Any,
    grid: Grid,
    RI: float,
    T_to_Ve: Optional[np.ndarray] = None,
    RM: Optional[float] = None,
    runtime_operators: Optional[dict[str, Any]] = None,
) -> PoloidalResultsOperators:
    """Build explicit postprocessing operators on ``grid``.

    This bundle is intended for notebooks and visualization tools that want
    direct dense operators without re-deriving the underlying formulas.
    """
    L = int(basis.index_length)
    scalar_evaluation_matrix = np.asarray(to_dense(basis.get_evaluation_matrix(grid)), dtype=float)

    def _runtime_array(name: str) -> Optional[np.ndarray]:
        if runtime_operators is None or name not in runtime_operators:
            return None
        value = runtime_operators[name]
        if value is None:
            return None
        return np.asarray(to_dense(value) if hasattr(value, "to_dense") else value, dtype=float)

    laplacian_operator = np.asarray(to_dense(basis.get_laplacian_operator(RI)), dtype=float)
    m_imp_to_jr = _runtime_array("m_imp_to_jr")
    if m_imp_to_jr is None:
        m_imp_to_jr = -(float(RI) / mu0) * laplacian_operator

    m_ind_to_Br = _runtime_array("m_ind_to_Br")
    if m_ind_to_Br is None:
        m_ind_to_Br = -(float(RI) ** 2) * laplacian_operator

    potential_scaling = np.asarray(to_dense(basis.get_potential_scaling_operator()), dtype=float)
    # For SH bases, ``m_ind`` matches the induced ``k_nm, l_nm`` coefficients,
    # so this gives the conventional equivalent-current function
    #   Psi_eq_nm = -(R/mu0) * (2n+1) * m_ind_nm.
    m_ind_to_Jeq = _runtime_array("m_ind_to_Jeq")
    if m_ind_to_Jeq is None:
        m_ind_to_Jeq = (-float(RI) / mu0) * potential_scaling

    curl_operator = as_linear_map(basis.get_curl_matrix(grid))
    gradient_operator = as_linear_map(basis.get_gradient_matrix(grid))
    scaling_operator = as_linear_map(basis.get_potential_scaling_operator())

    # ``get_curl_matrix()`` returns the repo df basis tensor
    # ``df_sign * (rhat x grad_Omega)``. The physical equivalent-current vector
    # is always the conventional
    #   J_eq = +rhat x grad_S(Psi_eq)
    # independent of the internal df-basis sign, so we divide out the repo df
    # sign here.
    G_m_ind_to_Jeq_vector = _runtime_array("G_m_ind_to_Jeq_vector")
    if G_m_ind_to_Jeq_vector is None:
        repo_df_sign = float(get_repo_df_helmholtz_sign())
        G_m_ind_to_Jeq_vector = np.asarray(
            to_dense(
                (
                    (1.0 / (repo_df_sign * float(RI)))
                    * (curl_operator @ as_linear_map(m_ind_to_Jeq))
                )
            ).reshape(2, -1, L),
            dtype=float,
        )

    G_Ve_to_JS = _runtime_array("G_Ve_to_JS")
    if G_Ve_to_JS is None:
        repo_df_sign = float(get_repo_df_helmholtz_sign())
        G_Ve_to_JS = np.asarray(
            to_dense(
                ((-1.0 / (repo_df_sign * mu0)) * (curl_operator @ scaling_operator))
            ).reshape(2, -1, L),
            dtype=float,
        )
    else:
        G_Ve_to_JS = G_Ve_to_JS.reshape(2, -1, L)

    G_B_tor_to_JS = _runtime_array("G_B_tor_to_JS")
    if G_B_tor_to_JS is None:
        G_B_tor_to_JS = np.asarray(
            to_dense(((-1.0 / mu0) * gradient_operator)).reshape(2, -1, L),
            dtype=float,
        )
    else:
        G_B_tor_to_JS = G_B_tor_to_JS.reshape(2, -1, L)

    T_to_Ve_matrix = None if T_to_Ve is None else np.asarray(T_to_Ve, dtype=float)
    G_m_imp_to_JS = _runtime_array("G_m_imp_to_JS")
    if G_m_imp_to_JS is None:
        if T_to_Ve_matrix is None:
            G_m_imp_to_JS = G_B_tor_to_JS.copy()
        else:
            G_m_imp_to_JS = G_B_tor_to_JS + np.tensordot(
                G_Ve_to_JS, T_to_Ve_matrix, axes=([2], [0])
            )
    else:
        G_m_imp_to_JS = G_m_imp_to_JS.reshape(2, -1, L)

    G_m_ind_to_JS = _runtime_array("G_m_ind_to_JS")
    if G_m_ind_to_JS is None:
        G_m_ind_to_JS = G_Ve_to_JS.copy()
    else:
        G_m_ind_to_JS = G_m_ind_to_JS.reshape(2, -1, L)

    G_Br_to_JS = _runtime_array("G_Br_to_JS")
    if G_Br_to_JS is None:
        G_Br_to_JS = np.zeros_like(G_Ve_to_JS)
    else:
        G_Br_to_JS = G_Br_to_JS.reshape(2, -1, L)

    RM_value = None if RM in (None, 0) else float(RM)
    if RM_value is not None and runtime_operators is None:
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
        G_m_ind_to_Jeq_vector=np.asarray(G_m_ind_to_Jeq_vector, dtype=float),
        T_to_Ve=T_to_Ve_matrix,
        G_Ve_to_JS=G_Ve_to_JS,
        G_B_tor_to_JS=G_B_tor_to_JS,
        G_m_ind_to_JS=np.asarray(G_m_ind_to_JS, dtype=float),
        G_m_imp_to_JS=np.asarray(G_m_imp_to_JS, dtype=float),
        G_Br_to_JS=np.asarray(G_Br_to_JS, dtype=float),
    )
