"""Simulation operator accessors.

This module keeps runtime operator composition separate from dense
matrix extraction.  Simulation code can use ``LinearMap`` objects
directly, while diagnostics and scripts can request dense matrices from
the same definitions.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from pynamit.math.backend import get_array_module
from pynamit.math.linear_map import DenseBackend, LinearMap, as_linear_map, dense_operator

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
        return as_linear_map(self._jr_to_m_imp_matrix)

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
        if self.state._direct_E_coeffs_to_total_E_coeffs_operator is None:
            self.state._direct_E_coeffs_to_total_E_coeffs_operator = (
                self._create_direct_E_to_total_E()
            )
        return self.state._direct_E_coeffs_to_total_E_coeffs_operator

    @property
    def direct_E_to_E_df(self) -> LinearMap:
        """Map direct E coefficients to total E_df forcing."""
        if self.state._direct_E_coeffs_to_E_df_operator is None:
            self.state._direct_E_coeffs_to_E_df_operator = (
                self.E_coeffs_to_E_df @ self.direct_E_to_total_E
            )
        return self.state._direct_E_coeffs_to_E_df_operator

    def _create_direct_E_to_total_E(self) -> LinearMap:
        """Construct direct-E to total-E map with m_imp feedback."""
        n = self.state.basis.index_length
        flat_E_size = 2 * n
        E_direct_to_m_imp = None
        m_imp_to_E = None

        if (
            self.state.connect_hemispheres
            and self.state.E_map_constraint_operator is not None
        ):
            E_direct_to_m_imp = self._E_direct_to_m_imp_matrix
            m_imp_to_E = self.state.m_imp_to_E_coeffs

        tensor_args = []
        if E_direct_to_m_imp is not None:
            tensor_args.append(E_direct_to_m_imp)
        if m_imp_to_E is not None:
            tensor_args.extend(m_imp_to_E.component_tensors)
        dtype = np.result_type(
            np.float64,
            getattr(E_direct_to_m_imp, "dtype", np.float64),
            getattr(m_imp_to_E, "dtype", np.float64),
        )

        def matmat(block: Any) -> Any:
            array_module = get_array_module(block, *tensor_args)
            block = array_module.asarray(block).reshape(flat_E_size, -1)
            total = block
            if E_direct_to_m_imp is not None:
                if m_imp_to_E is None:
                    raise RuntimeError("m_imp_to_E_coeffs is not available.")
                feedback = array_module.asarray(E_direct_to_m_imp).reshape(
                    n, flat_E_size
                )
                m_imp_block = feedback @ block
                total = total + m_imp_to_E.matmat(m_imp_block).reshape(
                    flat_E_size, -1
                )
            return total

        def rmatmat(block: Any) -> Any:
            array_module = get_array_module(block, *tensor_args)
            block = array_module.asarray(block).reshape(flat_E_size, -1)
            result = block
            if E_direct_to_m_imp is not None:
                if m_imp_to_E is None:
                    raise RuntimeError("m_imp_to_E_coeffs is not available.")
                feedback = array_module.asarray(E_direct_to_m_imp).reshape(
                    n, flat_E_size
                )
                m_imp_adjoint = m_imp_to_E.rmatmat(block).reshape(n, -1)
                result = result + feedback.T.conj() @ m_imp_adjoint
            return result

        def matvec(vec: Any) -> Any:
            array_module = get_array_module(vec, *tensor_args)
            return matmat(array_module.asarray(vec).reshape(flat_E_size, 1)).reshape(
                flat_E_size
            )

        def rmatvec(vec: Any) -> Any:
            array_module = get_array_module(vec, *tensor_args)
            return rmatmat(array_module.asarray(vec).reshape(flat_E_size, 1)).reshape(
                flat_E_size
            )

        return LinearMap(
            shape=(flat_E_size, flat_E_size),
            dtype=dtype,
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
        )

    @staticmethod
    def _dense_maps(
        operators: dict[str, LinearMap], *, backend: DenseBackend | Any = "active"
    ) -> dict[str, Any]:
        """Materialize named operators as dense arrays."""
        return {
            key: dense_operator(operator, backend=backend)
            for key, operator in operators.items()
        }

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

        if include_Br and self.state.Br_to_E_coeffs is not None:
            operators["edf_from_Br"] = (
                self.direct_E_to_E_df @ self.state.Br_to_E_coeffs
            )

        return operators

    def rates(self, *, include_Br: bool = True) -> dict[str, LinearMap]:
        """Return named input/state to d(m_ind)/dt operators."""
        scale = float(self.state.geometry.E_df_to_d_m_ind_dt)
        return {
            key.replace("edf_from_", "dt_m_ind_from_"): scale * operator
            for key, operator in self.E_df(include_Br=include_Br).items()
        }

    def E_df_dense(
        self, *, include_Br: bool = True, backend: DenseBackend | Any = "active"
    ) -> dict[str, Any]:
        """Return E_df maps as dense arrays on the requested backend."""
        return self._dense_maps(self.E_df(include_Br=include_Br), backend=backend)

    def rates_dense(
        self, *, include_Br: bool = True, backend: DenseBackend | Any = "active"
    ) -> dict[str, Any]:
        """Return d(m_ind)/dt maps as dense backend arrays."""
        return self._dense_maps(self.rates(include_Br=include_Br), backend=backend)

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
        backend: DenseBackend | Any = "active",
    ) -> dict[str, Any]:
        """Return dense simulation model maps."""
        return self._dense_maps(
            self.model(df_only=df_only, include_Br=include_Br), backend=backend
        )
