"""Saved coefficient-field evaluation for interactive visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pynamit.math.constants import mu0
from pynamit.simulation.config import setting_value
from pynamit.simulation.geometry import Geometry
from pynamit.simulation.mainfield import mainfield_from_config
from pynamit.simulation.schema import build_simulation_schema
from pynamit.sphere import SHBasis
from pynamit.visualization.artifacts import (
    artifact_path,
    load_dataarray_artifact,
    load_dataset_artifact,
    xarray_artifact_exists,
)
from pynamit.visualization.grid_evaluation import (
    build_evaluator,
    build_plot_grid,
    load_settings_and_basis,
    resistance_to_conductance,
)

INPUT_ARTIFACT_KEYS = ("Br", "jr", "conductance", "u", "Q_eff", "E_source")
TANGENTIAL_INPUT_KEYS = ("u", "Q_eff", "E_source")


def _load_optional_dataset(run_dir, name):
    path = artifact_path(run_dir, name)
    if path and xarray_artifact_exists(path):
        return load_dataset_artifact(path)
    return None


def _time_datasets(datasets):
    return [
        dataset
        for dataset in datasets.values()
        if "time" in dataset.coords or "time" in dataset.dims
    ]


def _nan_field(shape):
    return np.full(shape, np.nan, dtype=float)


def _dataset_var_name(dataset, variable_name):
    """Return the stored xarray variable for one logical variable."""
    for candidate in (f"SH_{variable_name}", f"CS_{variable_name}", variable_name):
        if candidate in dataset:
            return candidate
    suffix = f"_{variable_name}"
    for name in dataset.data_vars:
        if str(name).endswith(suffix):
            return str(name)
    return None


def _required_dataset_values(dataset, variable_name, index):
    """Return stored coefficient values for one logical variable."""
    stored_name = _dataset_var_name(dataset, variable_name)
    if stored_name is None:
        available = ", ".join(str(name) for name in dataset.data_vars) or "none"
        raise KeyError(f"Dataset has no saved {variable_name!r} coefficients; found {available}.")
    return np.asarray(dataset[stored_name].isel(time=index).values)


def _apply_flat_operator(operator, coeffs):
    """Apply an operator and return a flat NumPy vector."""
    coeffs = np.asarray(coeffs)
    if hasattr(operator, "matvec"):
        values = operator.matvec(coeffs)
    elif hasattr(operator, "dot"):
        values = operator.dot(coeffs)
    else:
        values = operator @ coeffs
    return np.asarray(values).reshape(-1)


def _apply_js_operator(operator, coeffs):
    """Apply one gridded JS tensor to coefficient values."""
    return np.asarray(operator).dot(np.asarray(coeffs))


def _state_conversion_maps(settings, geometry, evaluator):
    """Return grid operators for saved state coefficients."""
    ri = float(setting_value(settings, "RI"))
    solid_evaluator = geometry.solid_transform_for(evaluator)
    return {
        "RI": ri,
        "m_ind_to_Br": evaluator.scalar_coeffs_to_grid_operator @ geometry.m_ind_to_Br_operator,
        "m_imp_to_jr": evaluator.scalar_coeffs_to_grid_operator @ geometry.m_imp_to_jr_operator,
        "m_ind_to_Jeq": (-ri / mu0)
        * (
            solid_evaluator.scalar_coeffs_to_grid_operator
            @ geometry.horizontal_to_boundary_potential_jump_factor_operator
        ),
    }


def _state_js_operators(geometry, evaluator):
    """Return gridded JS operators for saved state coefficients."""
    solid_evaluator = geometry.solid_transform_for(evaluator)
    return {
        "G_m_ind_to_JS": geometry.m_ind_to_gridded_JS(evaluator, solid_transform=solid_evaluator),
        "G_m_imp_to_JS": geometry.m_imp_to_gridded_JS(evaluator, solid_transform=solid_evaluator),
        "G_Br_to_JS": geometry.Br_to_gridded_JS(evaluator, solid_transform=solid_evaluator),
    }


def _state_fields_from_coefficients(m_ind, m_imp, phi_coeffs, w_coeffs, evaluator, conversion):
    """Evaluate flattened map fields from one state coefficient row."""
    e_potential_to_kv = float(conversion["RI"]) * 1e-3
    return {
        "Br": _apply_flat_operator(conversion["m_ind_to_Br"], m_ind),
        "jr": _apply_flat_operator(conversion["m_imp_to_jr"], m_imp),
        "Jeq": _apply_flat_operator(conversion["m_ind_to_Jeq"], m_ind),
        "Phi": _apply_flat_operator(
            evaluator.scalar_coeffs_to_grid_operator, e_potential_to_kv * phi_coeffs
        ),
        "W": _apply_flat_operator(
            evaluator.scalar_coeffs_to_grid_operator, e_potential_to_kv * w_coeffs
        ),
    }


def _input_scalar_grid(datasets, dataset_key, variable_name, index, evaluator, shape):
    dataset = datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape)
    stored_name = _dataset_var_name(dataset, variable_name)
    if stored_name is None:
        return _nan_field(shape)
    return evaluator.G.dot(dataset[stored_name].isel(time=index).values).reshape(shape)


def _input_tangential_grid(datasets, dataset_key, variable_name, index, evaluator, shape):
    dataset = datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape), _nan_field(shape)
    stored_name = _dataset_var_name(dataset, variable_name)
    if stored_name is None:
        return _nan_field(shape), _nan_field(shape)
    theta_grid, phi_grid = evaluator.synthesize_helmholtz(
        dataset[stored_name].isel(time=index).values
    )
    return theta_grid.reshape(shape), phi_grid.reshape(shape)


def _dataset_index_at_time(dataset, timestamp, *, fallback_start_time=None):
    """Return the latest dataset index at or before ``timestamp``."""
    times = time_index_from_dataset(dataset, fallback_start_time=fallback_start_time)
    if times.empty:
        raise ValueError("No time coordinates are available.")

    target = pd.Timestamp(timestamp)
    if target.tz is not None:
        target = target.tz_convert(None)
    position = int(np.searchsorted(times.asi8, target.value, side="right") - 1)
    return max(0, min(position, len(times) - 1))


def _input_scalar_grid_at_time(
    datasets, dataset_key, variable_name, timestamp, evaluator, shape, *, fallback_start_time=None
):
    dataset = datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape)
    stored_name = _dataset_var_name(dataset, variable_name)
    if stored_name is None:
        return _nan_field(shape)
    index = _dataset_index_at_time(dataset, timestamp, fallback_start_time=fallback_start_time)
    return evaluator.G.dot(dataset[stored_name].isel(time=index).values).reshape(shape)


def _input_tangential_grid_at_time(
    datasets, dataset_key, variable_name, timestamp, evaluator, shape, *, fallback_start_time=None
):
    dataset = datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape), _nan_field(shape)
    stored_name = _dataset_var_name(dataset, variable_name)
    if stored_name is None:
        return _nan_field(shape), _nan_field(shape)
    index = _dataset_index_at_time(dataset, timestamp, fallback_start_time=fallback_start_time)
    theta_grid, phi_grid = evaluator.synthesize_helmholtz(
        dataset[stored_name].isel(time=index).values
    )
    return theta_grid.reshape(shape), phi_grid.reshape(shape)


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
    index,
    datasets,
    evaluator,
    conductance_evaluator,
    conversion,
    js_operators,
    *,
    target_time=None,
    fallback_start_time=None,
):
    """Evaluate state and steady-state map fields at one saved index."""
    state_dataset = datasets["state"]
    state_index = int(index)
    if target_time is None:
        target_time = datetime_at_index(
            state_dataset.time.values, state_index, fallback_start_time=fallback_start_time
        )

    m_ind = _required_dataset_values(state_dataset, "m_ind", state_index)
    m_imp = _required_dataset_values(state_dataset, "m_imp", state_index)
    br_mag = None
    if "Br" in datasets:
        br_var = _dataset_var_name(datasets["Br"], "Br")
        if br_var is not None:
            br_index = _dataset_index_at_time(
                datasets["Br"], target_time, fallback_start_time=fallback_start_time
            )
            br_mag = datasets["Br"][br_var].isel(time=br_index).values
    phi_coeffs = _required_dataset_values(state_dataset, "Phi", state_index)
    w_coeffs = _required_dataset_values(state_dataset, "W", state_index)

    resistance = None
    if "conductance" in datasets:
        eta_p_var = _dataset_var_name(datasets["conductance"], "etaP")
        if eta_p_var is not None:
            conductance_index = _dataset_index_at_time(
                datasets["conductance"], target_time, fallback_start_time=fallback_start_time
            )
            resistance_coeffs = (
                datasets["conductance"][eta_p_var].isel(time=conductance_index).values
            )
            resistance = conductance_evaluator.G.dot(resistance_coeffs)
    state_fields = _state_fields_from_coefficients(
        m_ind, m_imp, phi_coeffs, w_coeffs, evaluator, conversion
    )

    js_state = _apply_js_operator(js_operators["G_m_ind_to_JS"], m_ind) + _apply_js_operator(
        js_operators["G_m_imp_to_JS"], m_imp
    )
    if br_mag is not None and js_operators.get("G_Br_to_JS") is not None:
        js_state += _apply_js_operator(js_operators["G_Br_to_JS"], br_mag)
    state_fields["joule"] = (
        resistance * np.sum(js_state**2, axis=0)
        if resistance is not None
        else np.full_like(state_fields["Br"], np.nan)
    )

    result = {"state": state_fields}
    if "steady_state" in datasets:
        steady_dataset = datasets["steady_state"]
        steady_index = _dataset_index_at_time(
            steady_dataset, target_time, fallback_start_time=fallback_start_time
        )
        m_ind_steady = _required_dataset_values(steady_dataset, "m_ind", steady_index)
        m_imp_steady = _required_dataset_values(steady_dataset, "m_imp", steady_index)
        phi_coeffs_steady = _required_dataset_values(steady_dataset, "Phi", steady_index)
        w_coeffs_steady = _required_dataset_values(steady_dataset, "W", steady_index)
        steady_fields = _state_fields_from_coefficients(
            m_ind_steady, m_imp_steady, phi_coeffs_steady, w_coeffs_steady, evaluator, conversion
        )
        js_steady = _apply_js_operator(
            js_operators["G_m_ind_to_JS"], m_ind_steady
        ) + _apply_js_operator(js_operators["G_m_imp_to_JS"], m_imp_steady)
        if br_mag is not None and js_operators.get("G_Br_to_JS") is not None:
            js_steady += _apply_js_operator(js_operators["G_Br_to_JS"], br_mag)
        steady_fields["joule"] = (
            resistance * np.sum(js_steady**2, axis=0)
            if resistance is not None
            else np.full_like(steady_fields["Br"], np.nan)
        )
        result["steady"] = steady_fields
    return result


def compute_input_fields_at_index(index, datasets, input_evaluators, scalar_shape, vector_shape):
    """Evaluate projected input drivers at one saved index."""
    jr = _input_scalar_grid(datasets, "jr", "jr", index, input_evaluators["jr"], scalar_shape)
    br = _input_scalar_grid(datasets, "Br", "Br", index, input_evaluators["Br"], scalar_shape)
    if "conductance" in datasets:
        eta_p = _input_scalar_grid(
            datasets, "conductance", "etaP", index, input_evaluators["conductance"], scalar_shape
        ).reshape(-1)
        eta_h = _input_scalar_grid(
            datasets, "conductance", "etaH", index, input_evaluators["conductance"], scalar_shape
        ).reshape(-1)
        sigma_p, sigma_h = resistance_to_conductance(eta_p, eta_h)
        sigma_p = sigma_p.reshape(scalar_shape)
        sigma_h = sigma_h.reshape(scalar_shape)
    else:
        sigma_p = _nan_field(scalar_shape)
        sigma_h = _nan_field(scalar_shape)

    tangential = {
        key: _input_tangential_grid(datasets, key, key, index, input_evaluators[key], vector_shape)
        for key in TANGENTIAL_INPUT_KEYS
    }

    return {
        "jr": jr,
        "Br": br,
        "sigmaP": sigma_p,
        "sigmaH": sigma_h,
        "wind_theta": tangential["u"][0],
        "wind_phi": tangential["u"][1],
        "Q_eff_theta": tangential["Q_eff"][0],
        "Q_eff_phi": tangential["Q_eff"][1],
        "E_source_theta": tangential["E_source"][0],
        "E_source_phi": tangential["E_source"][1],
    }


def compute_input_fields_at_time(
    timestamp, datasets, input_evaluators, scalar_shape, vector_shape, *, fallback_start_time=None
):
    """Evaluate projected input drivers at one physical time."""
    jr = _input_scalar_grid_at_time(
        datasets,
        "jr",
        "jr",
        timestamp,
        input_evaluators["jr"],
        scalar_shape,
        fallback_start_time=fallback_start_time,
    )
    br = _input_scalar_grid_at_time(
        datasets,
        "Br",
        "Br",
        timestamp,
        input_evaluators["Br"],
        scalar_shape,
        fallback_start_time=fallback_start_time,
    )
    if "conductance" in datasets:
        eta_p = _input_scalar_grid_at_time(
            datasets,
            "conductance",
            "etaP",
            timestamp,
            input_evaluators["conductance"],
            scalar_shape,
            fallback_start_time=fallback_start_time,
        ).reshape(-1)
        eta_h = _input_scalar_grid_at_time(
            datasets,
            "conductance",
            "etaH",
            timestamp,
            input_evaluators["conductance"],
            scalar_shape,
            fallback_start_time=fallback_start_time,
        ).reshape(-1)
        sigma_p, sigma_h = resistance_to_conductance(eta_p, eta_h)
        sigma_p = sigma_p.reshape(scalar_shape)
        sigma_h = sigma_h.reshape(scalar_shape)
    else:
        sigma_p = _nan_field(scalar_shape)
        sigma_h = _nan_field(scalar_shape)

    tangential = {
        key: _input_tangential_grid_at_time(
            datasets,
            key,
            key,
            timestamp,
            input_evaluators[key],
            vector_shape,
            fallback_start_time=fallback_start_time,
        )
        for key in TANGENTIAL_INPUT_KEYS
    }

    return {
        "jr": jr,
        "Br": br,
        "sigmaP": sigma_p,
        "sigmaH": sigma_h,
        "wind_theta": tangential["u"][0],
        "wind_phi": tangential["u"][1],
        "Q_eff_theta": tangential["Q_eff"][0],
        "Q_eff_phi": tangential["Q_eff"][1],
        "E_source_theta": tangential["E_source"][0],
        "E_source_phi": tangential["E_source"][1],
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
    input_evaluators: dict[str, object]
    conversion: dict
    js_operators: object
    datasets: dict[str, xr.Dataset]
    schema: object | None = None
    geometry: object | None = None

    @classmethod
    def from_directory(
        cls, run_directory, *, nlat=60, nlon=100, wind_nlat=19, wind_nlon=37
    ) -> "SavedCoefficientFieldView":
        """Load artifacts needed by map and input-driver figures."""
        run_dir = Path(run_directory).expanduser()
        settings_path = artifact_path(run_dir, "settings")
        settings, sh_basis = load_settings_and_basis(settings_path)
        schema = build_simulation_schema(settings)
        output_basis = schema.output_field_spaces["state"].representation
        conductance_basis = schema.input_field_spaces["conductance"].representation
        conductance_sh_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin=0)
        lat, lon, grid = build_plot_grid(nlat=nlat, nlon=nlon)
        wind_lat, wind_lon, wind_grid = build_plot_grid(
            nlat=wind_nlat, nlon=wind_nlon, lat_range=(-75.0, 75.0), lon_range=(-180.0, 180.0)
        )
        evaluator = build_evaluator(output_basis, grid)
        conductance_evaluator = build_evaluator(conductance_basis, grid)
        wind_evaluator = build_evaluator(schema.input_field_spaces["u"].representation, wind_grid)
        input_evaluators = {
            "jr": build_evaluator(schema.input_field_spaces["jr"].representation, grid),
            "Br": build_evaluator(schema.input_field_spaces["Br"].representation, grid),
            "conductance": build_evaluator(
                schema.input_field_spaces["conductance"].representation, grid
            ),
            "u": build_evaluator(schema.input_field_spaces["u"].representation, wind_grid),
            "Q_eff": build_evaluator(schema.input_field_spaces["Q_eff"].representation, wind_grid),
            "E_source": build_evaluator(
                schema.input_field_spaces["E_source"].representation, wind_grid
            ),
        }

        pfac_path = artifact_path(run_dir, "PFAC_matrix")
        pfac_matrix = (
            load_dataarray_artifact(pfac_path)
            if pfac_path and xarray_artifact_exists(pfac_path)
            else None
        )
        geometry = Geometry(
            schema.horizontal_basis,
            schema.cs_basis,
            mainfield_from_config(settings),
            settings,
            PFAC_matrix=pfac_matrix,
            solid_harmonics=schema.solid_harmonics,
        )
        conversion = _state_conversion_maps(settings, geometry, evaluator)
        js_operators = _state_js_operators(geometry, evaluator)

        datasets = {}
        for key in INPUT_ARTIFACT_KEYS:
            dataset = _load_optional_dataset(run_dir, key)
            if dataset is not None:
                datasets[key] = dataset
        state_path = artifact_path(run_dir, "state")
        if state_path and xarray_artifact_exists(state_path):
            datasets["state"] = load_dataset_artifact(state_path)
        steady_path = artifact_path(run_dir, "steady_state")
        if steady_path and xarray_artifact_exists(steady_path):
            datasets["steady_state"] = load_dataset_artifact(steady_path)
        time_datasets = _time_datasets(datasets)
        if not time_datasets:
            raise ValueError(
                f"No saved input or output time series exists in {run_dir}. "
                "Expected at least one of state, Br, jr, conductance, u, Q_eff, or E_source."
            )

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
            input_evaluators=input_evaluators,
            conversion=conversion,
            js_operators=js_operators,
            datasets=datasets,
            schema=schema,
            geometry=geometry,
        )

    @property
    def n_time(self):
        """Return the number of display time steps."""
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
        time_datasets = _time_datasets(self.datasets)
        if not time_datasets:
            raise ValueError("No saved time-dependent artifacts are available.")
        return time_datasets[0]

    @property
    def available_inputs(self):
        """Return projected input names available in this directory."""
        return tuple(key for key in INPUT_ARTIFACT_KEYS if key in self.datasets)

    def has_input(self, name):
        """Return whether one projected input artifact is available."""
        return name in self.available_inputs

    def dataset_values(self, dataset_key, variable_name):
        """Return stored values for one logical dataset variable."""
        dataset = self.datasets[dataset_key]
        stored_name = _dataset_var_name(dataset, variable_name)
        if stored_name is None:
            available = ", ".join(str(name) for name in dataset.data_vars) or "none"
            raise KeyError(
                f"{dataset_key!r} has no saved {variable_name!r} coefficients; found {available}."
            )
        return dataset[stored_name].values

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
        index = int(index)
        timestamp = self.timestamp_at_index(index)
        return compute_state_comparison_fields_at_index(
            index,
            self.datasets,
            self.evaluator,
            self.conductance_evaluator,
            self.conversion,
            self.js_operators,
            target_time=timestamp,
            fallback_start_time=self._fallback_start_time(),
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
        return self.input_grid_fields_at_time(self.timestamp_at_index(index))

    def input_grid_fields_at_time(self, timestamp):
        """Return input-driver fields selected by physical time."""
        return compute_input_fields_at_time(
            timestamp,
            self.datasets,
            self.input_evaluators,
            self.lat.shape,
            self.wind_lat.shape,
            fallback_start_time=self._fallback_start_time(),
        )


__all__ = [
    "SavedCoefficientFieldView",
    "compute_input_fields_at_index",
    "compute_input_fields_at_time",
    "compute_state_comparison_fields_at_index",
    "datetime_at_index",
    "time_index_from_dataset",
]
