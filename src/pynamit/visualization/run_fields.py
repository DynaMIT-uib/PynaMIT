"""Saved coefficient-field evaluation for interactive visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pynamit.simulation.config import setting_value
from pynamit.sphere import SHBasis
from pynamit.visualization.artifacts import (
    artifact_path,
    load_dataarray_artifact,
    load_dataset_artifact,
    xarray_artifact_exists,
)
from pynamit.visualization.grid_evaluation import (
    build_JS_operators,
    build_evaluator,
    build_plot_grid,
    compute_conversion_factors,
    load_settings_and_basis,
    resistance_to_conductance,
)


def datetime_at_index(times, index, *, fallback_start_time=None):
    """Return one saved time value as a pandas timestamp."""
    values = np.asarray(times)
    if values.size == 0:
        raise ValueError("No time coordinates are available.")
    idx = int(max(0, min(int(index), values.size - 1)))
    value = values[idx]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Timestamp(value)
    if fallback_start_time is None:
        fallback_start_time = pd.Timestamp("1970-01-01")
    return pd.Timestamp(fallback_start_time) + pd.to_timedelta(float(value), unit="s")


def time_index_from_dataset(dataset, *, fallback_start_time=None):
    """Return dataset times as a ``DatetimeIndex``."""
    return pd.DatetimeIndex(
        [
            datetime_at_index(dataset.time.values, index, fallback_start_time=fallback_start_time)
            for index in range(len(dataset.time))
        ]
    )


def compute_state_comparison_fields_at_index(
    index, datasets, evaluator, conductance_evaluator, conversion, js_operators
):
    """Evaluate state and steady-state map fields at one saved index."""
    m_ind = datasets["state"].SH_m_ind.isel(time=index).values
    m_imp = datasets["state"].SH_m_imp.isel(time=index).values
    br_mag = datasets["Br_mag"].SH_Br.isel(time=index).values
    phi_coeffs = datasets["state"].SH_Phi.isel(time=index).values
    w_coeffs = datasets["state"].SH_W.isel(time=index).values

    resistance_coeffs = datasets["resistance"].SH_etaP.isel(time=index).values
    resistance = conductance_evaluator.G.dot(resistance_coeffs)
    e_potential_to_kv = float(conversion["RI"]) * 1e-3

    state_fields = {
        "Br": evaluator.G.dot(conversion["m_ind_to_Br"] * m_ind),
        "jr": evaluator.G.dot(conversion["m_imp_to_jr"] * m_imp),
        "Jeq": evaluator.G.dot(conversion["m_ind_to_Jeq"] * m_ind),
        "Phi": evaluator.G.dot(e_potential_to_kv * phi_coeffs),
        "W": evaluator.G.dot(e_potential_to_kv * w_coeffs),
    }

    js_state = (
        js_operators["G_m_ind_to_JS"].dot(m_ind)
        + js_operators["G_m_imp_to_JS"].dot(m_imp)
        + js_operators["G_Br_to_JS"].dot(br_mag)
    )
    state_fields["joule"] = resistance * np.sum(js_state**2, axis=0)

    result = {"state": state_fields}
    if "steady_state" in datasets:
        m_ind_steady = datasets["steady_state"].SH_m_ind.isel(time=index).values
        m_imp_steady = datasets["steady_state"].SH_m_imp.isel(time=index).values
        phi_coeffs_steady = datasets["steady_state"].SH_Phi.isel(time=index).values
        w_coeffs_steady = datasets["steady_state"].SH_W.isel(time=index).values
        steady_fields = {
            "Br": evaluator.G.dot(conversion["m_ind_to_Br"] * m_ind_steady),
            "jr": evaluator.G.dot(conversion["m_imp_to_jr"] * m_imp_steady),
            "Jeq": evaluator.G.dot(conversion["m_ind_to_Jeq"] * m_ind_steady),
            "Phi": evaluator.G.dot(e_potential_to_kv * phi_coeffs_steady),
            "W": evaluator.G.dot(e_potential_to_kv * w_coeffs_steady),
        }
        js_steady = (
            js_operators["G_m_ind_to_JS"].dot(m_ind_steady)
            + js_operators["G_m_imp_to_JS"].dot(m_imp_steady)
            + js_operators["G_Br_to_JS"].dot(br_mag)
        )
        steady_fields["joule"] = resistance * np.sum(js_steady**2, axis=0)
        result["steady"] = steady_fields
    return result


def compute_input_fields_at_index(
    index, datasets, evaluator, conductance_evaluator, wind_evaluator, scalar_shape, wind_shape
):
    """Evaluate projected input drivers at one saved index."""
    jr_coeffs = datasets["jr_input"].SH_jr.isel(time=index).values
    br_coeffs = datasets["Br_mag"].SH_Br.isel(time=index).values
    eta_p_coeffs = datasets["resistance"].SH_etaP.isel(time=index).values
    eta_h_coeffs = datasets["resistance"].SH_etaH.isel(time=index).values

    eta_p = conductance_evaluator.G.dot(eta_p_coeffs)
    eta_h = conductance_evaluator.G.dot(eta_h_coeffs)
    sigma_p, sigma_h = resistance_to_conductance(eta_p, eta_h)

    u_theta = np.full(wind_shape, np.nan, dtype=float)
    u_phi = np.full(wind_shape, np.nan, dtype=float)
    wind_ds = datasets.get("wind")
    if wind_ds is not None and "SH_u" in wind_ds:
        wind_coeffs = wind_ds.SH_u.isel(time=index).values
        u_theta_grid, u_phi_grid = wind_evaluator.synthesize_helmholtz(wind_coeffs)
        u_theta = u_theta_grid.reshape(wind_shape)
        u_phi = u_phi_grid.reshape(wind_shape)

    return {
        "jr": evaluator.G.dot(jr_coeffs).reshape(scalar_shape),
        "Br": evaluator.G.dot(br_coeffs).reshape(scalar_shape),
        "sigmaP": sigma_p.reshape(scalar_shape),
        "sigmaH": sigma_h.reshape(scalar_shape),
        "wind_theta": u_theta,
        "wind_phi": u_phi,
    }


@dataclass
class SavedCoefficientFieldView:
    """Field evaluator for saved PynaMIT coefficient artifacts."""

    run_directory: Path
    settings: object
    sh_basis: object
    conductance_sh_basis: object
    lat: np.ndarray
    lon: np.ndarray
    wind_lat: np.ndarray
    wind_lon: np.ndarray
    evaluator: object
    conductance_evaluator: object
    wind_evaluator: object
    conversion: dict
    js_operators: object
    datasets: dict[str, xr.Dataset]

    @classmethod
    def from_directory(
        cls, run_directory, *, nlat=60, nlon=100, wind_nlat=19, wind_nlon=37
    ) -> "SavedCoefficientFieldView":
        """Load artifacts needed by map and input-driver figures."""
        run_dir = Path(run_directory).expanduser()
        settings_path = artifact_path(run_dir, "settings")
        settings, sh_basis = load_settings_and_basis(settings_path)
        conductance_sh_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin=0)
        lat, lon, grid = build_plot_grid(nlat=nlat, nlon=nlon)
        wind_lat, wind_lon, wind_grid = build_plot_grid(
            nlat=wind_nlat, nlon=wind_nlon, lat_range=(-75.0, 75.0), lon_range=(-180.0, 180.0)
        )
        evaluator = build_evaluator(sh_basis, grid)
        conductance_evaluator = build_evaluator(conductance_sh_basis, grid)
        wind_evaluator = build_evaluator(sh_basis, wind_grid)
        conversion = compute_conversion_factors(settings, sh_basis)

        pfac_path = artifact_path(run_dir, "PFAC_matrix")
        t_to_ve = (
            load_dataarray_artifact(pfac_path).values
            if pfac_path and xarray_artifact_exists(pfac_path)
            else None
        )
        js_operators = build_JS_operators(settings, sh_basis, evaluator, T_to_Ve=t_to_ve)

        input_artifact_names = {
            "Br_mag": "Br",
            "jr_input": "jr",
            "resistance": "conductance",
        }
        missing = [
            name
            for name in input_artifact_names.values()
            if not xarray_artifact_exists(artifact_path(run_dir, name))
        ]
        if missing:
            raise ValueError(f"Missing saved artifact(s) in {run_dir}: {', '.join(missing)}")
        datasets = {
            key: load_dataset_artifact(artifact_path(run_dir, name))
            for key, name in input_artifact_names.items()
        }
        state_path = artifact_path(run_dir, "state")
        if state_path and xarray_artifact_exists(state_path):
            datasets["state"] = load_dataset_artifact(state_path)
        steady_path = artifact_path(run_dir, "steady_state")
        if steady_path and xarray_artifact_exists(steady_path):
            datasets["steady_state"] = load_dataset_artifact(steady_path)
        reference_dataset = datasets.get("state", datasets["Br_mag"])
        wind_path = artifact_path(run_dir, "u")
        datasets["wind"] = (
            load_dataset_artifact(wind_path)
            if wind_path and xarray_artifact_exists(wind_path)
            else xr.Dataset(coords={"time": reference_dataset.time})
        )

        min_len = min(len(dataset.time) for dataset in datasets.values())
        datasets = {
            key: dataset.isel(time=slice(None, min_len)) for key, dataset in datasets.items()
        }

        return cls(
            run_directory=run_dir,
            settings=settings,
            sh_basis=sh_basis,
            conductance_sh_basis=conductance_sh_basis,
            lat=lat,
            lon=lon,
            wind_lat=wind_lat,
            wind_lon=wind_lon,
            evaluator=evaluator,
            conductance_evaluator=conductance_evaluator,
            wind_evaluator=wind_evaluator,
            conversion=conversion,
            js_operators=js_operators,
            datasets=datasets,
        )

    @property
    def n_time(self):
        """Return the number of common saved time steps."""
        return len(self._time_dataset().time)

    @property
    def time_index(self):
        """Return saved times as datetimes."""
        return time_index_from_dataset(
            self._time_dataset(), fallback_start_time=self._fallback_start_time()
        )

    def timestamp_at_index(self, index):
        """Return one saved time as a timestamp."""
        return datetime_at_index(
            self._time_dataset().time.values,
            index,
            fallback_start_time=self._fallback_start_time(),
        )

    @property
    def has_output_state(self):
        """Return whether evolved output state is present."""
        return "state" in self.datasets

    def _time_dataset(self):
        """Return the dataset that defines display times."""
        if "state" in self.datasets:
            return self.datasets["state"]
        return self.datasets["Br_mag"]

    def _fallback_start_time(self):
        """Return the configured start time for numeric saved times."""
        return setting_value(self.settings, "t0", None)

    def state_comparison_fields(self, index):
        """Return flattened state/steady fields for one time index."""
        if "state" not in self.datasets:
            raise ValueError(
                "This directory contains projected inputs but no saved output state. "
                "Choose 'Input drivers' or run a simulation first."
            )
        return compute_state_comparison_fields_at_index(
            int(index),
            self.datasets,
            self.evaluator,
            self.conductance_evaluator,
            self.conversion,
            self.js_operators,
        )

    def state_comparison_grid_fields(self, index):
        """Return gridded state/steady fields for one time index."""
        fields = self.state_comparison_fields(index)
        return {
            f"{name}_{state_key}": values.reshape(self.lat.shape)
            for state_key, state_fields in fields.items()
            for name, values in state_fields.items()
        }

    def input_grid_fields(self, index):
        """Return projected input-driver fields."""
        return compute_input_fields_at_index(
            int(index),
            self.datasets,
            self.evaluator,
            self.conductance_evaluator,
            self.wind_evaluator,
            self.lat.shape,
            self.wind_lat.shape,
        )


__all__ = [
    "SavedCoefficientFieldView",
    "compute_input_fields_at_index",
    "compute_state_comparison_fields_at_index",
    "datetime_at_index",
    "time_index_from_dataset",
]
