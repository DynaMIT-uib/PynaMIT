"""Simulation operator accessors.

This module keeps runtime operator composition separate from dense
matrix extraction.  Simulation code can use ``LinearMap`` objects
directly, while diagnostics and scripts can request dense matrices from
the same definitions.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from pynamit.math.linear_map import (
    DenseBackend,
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
)

if TYPE_CHECKING:
    from pynamit.simulation.state import State


class StateOperators:
    """Simulation model operator accessors."""

    def __init__(self, state: "State") -> None:
        """Bind accessors to one state."""
        self.state = state

    @property
    def E_coeffs_to_E_df(self) -> LinearMap:
        """Linear map extracting the divergence-free E potential."""
        return self.state.geometry.helmholtz_divergence_free_potential_operator

    @property
    def jr_to_m_imp(self) -> LinearMap:
        """Linear map from radial current to imposed potential."""
        if getattr(self.state, "_jr_to_m_imp_operator", None) is None:
            self.state._jr_to_m_imp_operator = as_linear_map(
                self._jr_to_m_imp_matrix,
                input_shape=(self.state.basis.index_length,),
                output_shape=(self.state.basis.index_length,),
            )
        return self.state._jr_to_m_imp_operator

    @property
    def E_direct_to_m_imp(self) -> Optional[LinearMap]:
        """Map direct E coefficients to imposed potential."""
        if getattr(self.state, "_E_direct_to_m_imp_operator", None) is None:
            matrix = self._E_direct_to_m_imp_matrix
            if matrix is None:
                return None
            self.state._E_direct_to_m_imp_operator = as_linear_map(
                matrix,
                input_shape=(2, self.state.basis.index_length),
                output_shape=(self.state.basis.index_length,),
            )
        return self.state._E_direct_to_m_imp_operator

    @property
    def _jr_to_m_imp_matrix(self) -> np.ndarray:
        """Dense response matrix from radial current to m_imp."""
        self.state._ensure_m_imp_response_matrices()
        return self.state._jr_to_m_imp_matrix

    @property
    def _E_direct_to_m_imp_matrix(self) -> Optional[np.ndarray]:
        """Dense IH response from direct E coefficients to m_imp."""
        self.state._ensure_m_imp_response_matrices()
        return self.state._E_direct_to_m_imp_matrix

    @property
    def direct_E_to_total_E(self) -> LinearMap:
        """Map direct E coefficients to total model E coefficients."""
        if getattr(self.state, "_direct_E_coeffs_to_total_E_coeffs_operator", None) is None:
            self.state._direct_E_coeffs_to_total_E_coeffs_operator = (
                self._create_direct_E_to_total_E()
            )
        return self.state._direct_E_coeffs_to_total_E_coeffs_operator

    @property
    def direct_E_to_E_df(self) -> LinearMap:
        """Map direct E coefficients to total E_df forcing."""
        if getattr(self.state, "_direct_E_coeffs_to_E_df_operator", None) is None:
            self.state._direct_E_coeffs_to_E_df_operator = (
                self.E_coeffs_to_E_df @ self.direct_E_to_total_E
            )
        return self.state._direct_E_coeffs_to_E_df_operator

    def _create_direct_E_to_total_E(self) -> LinearMap:
        """Construct direct-E to total-E map with m_imp feedback."""
        n = self.state.basis.index_length
        flat_E_size = 2 * n
        identity = diagonal_linear_map(
            np.ones(flat_E_size),
            input_shape=(2, n),
            output_shape=(2, n),
        )

        if not self.state.connect_hemispheres or self.state._E_map_constraint is None:
            return identity

        E_direct_to_m_imp = self.E_direct_to_m_imp
        if E_direct_to_m_imp is None:
            return identity

        m_imp_to_E = self.state.m_imp_to_E_coeffs
        if m_imp_to_E is None:
            raise RuntimeError("m_imp_to_E_coeffs is not available.")

        return identity + (m_imp_to_E @ E_direct_to_m_imp)

    def E_df(self, *, include_Br: bool = True) -> dict[str, LinearMap]:
        """Return named input/state to total E_df operators."""
        m_imp_to_E = self.state.m_imp_to_E_coeffs
        if m_imp_to_E is None:
            raise RuntimeError("m_imp_to_E_coeffs is not available.")

        operators = {
            "edf_from_u": self.direct_E_to_E_df @ self.state.u_coeffs_to_E_coeffs,
            "edf_from_jr": self.E_coeffs_to_E_df
            @ m_imp_to_E
            @ self.jr_to_m_imp,
            "edf_from_m_ind": self.state.m_ind_to_E_df_operator,
        }

        if include_Br:
            Br_to_E = self.state.Br_to_E_coeffs
            if Br_to_E is not None:
                operators["edf_from_Br"] = self.direct_E_to_E_df @ Br_to_E

        return operators

    def rates(self, *, include_Br: bool = True) -> dict[str, LinearMap]:
        """Return named input/state to d(m_ind)/dt operators."""
        scale = float(self.state.geometry.E_df_to_d_m_ind_dt)
        return {
            key.replace("edf_from_", "dt_m_ind_from_"): scale * operator
            for key, operator in self.E_df(include_Br=include_Br).items()
        }

    def E_df_dense(
        self, *, include_Br: bool = True, backend: DenseBackend | None = None
    ) -> dict[str, Any]:
        """Return E_df maps as dense arrays on the requested backend."""
        return {
            key: operator.dense(backend=backend)
            for key, operator in self.E_df(include_Br=include_Br).items()
        }

    def rates_dense(
        self, *, include_Br: bool = True, backend: DenseBackend | None = None
    ) -> dict[str, Any]:
        """Return d(m_ind)/dt maps as dense backend arrays."""
        return {
            key: operator.dense(backend=backend)
            for key, operator in self.rates(include_Br=include_Br).items()
        }

    def model(
        self, *, df_only: bool = False, include_Br: bool = True
    ) -> dict[str, LinearMap]:
        """Return simulation model maps."""
        if df_only:
            return self.E_df(include_Br=include_Br)
        return self.rates(include_Br=include_Br)

    def model_dense(
        self,
        *,
        df_only: bool = False,
        include_Br: bool = True,
        backend: DenseBackend | None = None,
    ) -> dict[str, Any]:
        """Return dense simulation model maps."""
        return {
            key: operator.dense(backend=backend)
            for key, operator in self.model(
                df_only=df_only, include_Br=include_Br
            ).items()
        }
