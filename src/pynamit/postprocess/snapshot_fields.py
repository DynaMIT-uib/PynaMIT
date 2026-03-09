"""Internal snapshot-field assembly and caching for notebook-style views."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class SnapshotFieldCache:
    """Assemble snapshot-view scalar fields and optionally cache them over time."""

    def __init__(
        self,
        *,
        datasets: dict[str, Any],
        plot_operators: Any,
        output_shape: tuple[int, int],
        conductance_basis: Any,
        settings: Any,
        evaluate_plot_potential: Callable[[np.ndarray], np.ndarray],
        evaluate_conductance_etaP_grid: Callable[[Any, int, Any, Any, Any], np.ndarray],
        cond_plot_grid: Any,
        state_m_ind_name: str,
        state_m_imp_name: str,
        state_phi_name: str,
        state_w_name: str,
        steady_m_ind_name: str,
        steady_m_imp_name: str,
        steady_phi_name: str,
        steady_w_name: str,
        br_input_name: Optional[str] = None,
    ) -> None:
        self.datasets = datasets
        self.plot_operators = plot_operators
        self.output_shape = tuple(int(v) for v in output_shape)
        self.conductance_basis = conductance_basis
        self.settings = settings
        self.evaluate_plot_potential = evaluate_plot_potential
        self.evaluate_conductance_etaP_grid = evaluate_conductance_etaP_grid
        self.cond_plot_grid = cond_plot_grid
        self.state_m_ind_name = state_m_ind_name
        self.state_m_imp_name = state_m_imp_name
        self.state_phi_name = state_phi_name
        self.state_w_name = state_w_name
        self.steady_m_ind_name = steady_m_ind_name
        self.steady_m_imp_name = steady_m_imp_name
        self.steady_phi_name = steady_phi_name
        self.steady_w_name = steady_w_name
        self.br_input_name = br_input_name
        self._precomputed_fields: Optional[dict[str, np.ndarray]] = None

    @property
    def n_time(self) -> int:
        return int(len(self.datasets["state"].time))

    @staticmethod
    def _select_sample_at_time(ds: Any, var_name: Optional[str], time_value: float) -> Optional[np.ndarray]:
        if ds is None or var_name is None:
            return None
        time_coord = ds.time.values
        idx = int(np.searchsorted(time_coord, time_value, side="right") - 1)
        idx = max(0, min(idx, len(time_coord) - 1))
        return ds[var_name].isel(time=idx).values

    def compute_fields_at_index(self, t_idx: int) -> dict[str, dict[str, np.ndarray]]:
        state_ds = self.datasets["state"]
        time_value = float(state_ds.time.isel(time=int(t_idx)).values)
        m_ind = state_ds[self.state_m_ind_name].isel(time=t_idx).values
        m_imp = state_ds[self.state_m_imp_name].isel(time=t_idx).values
        phi_state = state_ds[self.state_phi_name].isel(time=t_idx).values
        w_state = state_ds[self.state_w_name].isel(time=t_idx).values

        br_mag = None
        if self.datasets.get("Br_mag") is not None and self.br_input_name is not None:
            br_mag = self._select_sample_at_time(self.datasets["Br_mag"], self.br_input_name, time_value)

        m_ind_steady = self._select_sample_at_time(self.datasets["steady_state"], self.steady_m_ind_name, time_value)
        m_imp_steady = self._select_sample_at_time(self.datasets["steady_state"], self.steady_m_imp_name, time_value)
        phi_steady = self._select_sample_at_time(self.datasets["steady_state"], self.steady_phi_name, time_value)
        w_steady = self._select_sample_at_time(self.datasets["steady_state"], self.steady_w_name, time_value)

        conductance_idx = int(np.searchsorted(self.datasets["conductance"].time.values, time_value, side="right") - 1)
        conductance_idx = max(0, min(conductance_idx, len(self.datasets["conductance"].time) - 1))
        resistance = self.evaluate_conductance_etaP_grid(
            self.datasets["conductance"],
            conductance_idx,
            self.cond_plot_grid,
            self.conductance_basis,
            self.settings,
        )

        state_fields = {
            "Br": self.plot_operators.evaluate_br(m_ind),
            "jr": self.plot_operators.evaluate_jr(m_imp),
            "Jeq": self.plot_operators.evaluate_jeq(m_ind),
            "Phi": self.evaluate_plot_potential(phi_state),
            "W": self.evaluate_plot_potential(w_state),
        }
        steady_fields = {
            "Br": self.plot_operators.evaluate_br(m_ind_steady),
            "jr": self.plot_operators.evaluate_jr(m_imp_steady),
            "Jeq": self.plot_operators.evaluate_jeq(m_ind_steady),
            "Phi": self.evaluate_plot_potential(phi_steady),
            "W": self.evaluate_plot_potential(w_steady),
        }

        if br_mag is not None:
            js_ind_state = self.plot_operators.evaluate_js_from_m_ind(m_ind)
            js_imp_state = self.plot_operators.evaluate_js_from_m_imp(m_imp)
            js_br_state = self.plot_operators.evaluate_js_from_br(br_mag)
            js_state = js_ind_state + js_imp_state + js_br_state
            js_ind_steady = self.plot_operators.evaluate_js_from_m_ind(m_ind_steady)
            js_imp_steady = self.plot_operators.evaluate_js_from_m_imp(m_imp_steady)
            js_br_steady = self.plot_operators.evaluate_js_from_br(br_mag)
            js_steady = js_ind_steady + js_imp_steady + js_br_steady
            state_fields["joule"] = resistance * np.sum(js_state**2, axis=0)
            steady_fields["joule"] = resistance * np.sum(js_steady**2, axis=0)

        return {"state": state_fields, "steady": steady_fields}

    def precompute(self, *, progress_iter: Optional[Callable[[range], Any]] = None) -> dict[str, np.ndarray]:
        all_fields: dict[str, list[np.ndarray]] = {}
        index_iter = progress_iter(range(self.n_time)) if progress_iter is not None else range(self.n_time)
        for t_idx in index_iter:
            computed = self.compute_fields_at_index(int(t_idx))
            for state_type, fields in computed.items():
                for var, data in fields.items():
                    key = f"{var}_{state_type}"
                    all_fields.setdefault(key, []).append(np.asarray(data).reshape(self.output_shape))
        self._precomputed_fields = {
            key: np.stack(values, axis=0) for key, values in all_fields.items()
        }
        return self._precomputed_fields

    def get_current_data(self, idx: int, *, use_precomputed: bool = True) -> dict[str, np.ndarray]:
        idx = int(idx)
        if use_precomputed:
            if self._precomputed_fields is None:
                self.precompute()
            assert self._precomputed_fields is not None
            return {key: values[idx] for key, values in self._precomputed_fields.items()}

        computed = self.compute_fields_at_index(idx)
        return {
            f"{var}_{state_type}": np.asarray(data).reshape(self.output_shape)
            for state_type, fields in computed.items()
            for var, data in fields.items()
        }
