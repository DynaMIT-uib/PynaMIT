"""Saved coefficient-field evaluation for interactive visualizations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.math.constants import mu0
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    joule_heating_from_current,
    pedersen_geometry_tensor,
)
from pynamit.visualization.field_maps import evaluate_conductance_values, evaluate_JS_from_maps
from pynamit.visualization.grid_evaluation import (
    build_evaluator,
    build_plot_grid,
    model_grid_for_geographic_display,
)
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.saved_run import SavedRunView

INPUT_ARTIFACT_KEYS = ("Br", "jr", "resistance", "u", "Q_eff", "E_source")
TANGENTIAL_INPUT_KEYS = ("u", "Q_eff", "E_source")
STATE_FIELD_NAMES = frozenset({"Br", "jr", "Jeq", "Phi", "W", "joule"})
_DISPLAY_COORDINATE_SYSTEMS = frozenset({"model", "geographic"})


def _normalize_display_coordinate_system(coordinate_system):
    """Return a supported map-display coordinate system."""
    normalized = str(coordinate_system).strip().lower()
    if normalized not in _DISPLAY_COORDINATE_SYSTEMS:
        raise ValueError(
            f"coordinate_system must be either 'model' or 'geographic'; got {coordinate_system!r}."
        )
    return normalized


def _normalize_state_field_names(field_names):
    """Return a validated immutable field selection."""
    if field_names is None:
        selected = STATE_FIELD_NAMES
    elif isinstance(field_names, str):
        selected = frozenset({field_names})
    else:
        selected = frozenset(field_names)
    unknown_fields = selected - STATE_FIELD_NAMES
    if unknown_fields:
        raise ValueError(f"Unknown state fields requested: {sorted(unknown_fields)}.")
    return selected


def _time_datasets(datasets):
    return [
        dataset
        for dataset in datasets.values()
        if "time" in dataset.coords or "time" in dataset.dims
    ]


def _nan_field(shape):
    return np.full(shape, np.nan, dtype=float)


def _build_input_evaluators(
    schema, datasets, scalar_grid, vector_grid, evaluator_cache=None, *, keys=INPUT_ARTIFACT_KEYS
):
    """Build input evaluators for the requested display grids."""
    evaluator_cache = {} if evaluator_cache is None else evaluator_cache
    evaluators = {}
    for key in keys:
        if key not in datasets:
            evaluators[key] = None
            continue
        target_grid = vector_grid if key in TANGENTIAL_INPUT_KEYS else scalar_grid
        representation = schema.input_field_spaces[key].representation
        cache_key = (
            getattr(representation, "signature", id(representation)),
            target_grid.signature,
        )
        if cache_key not in evaluator_cache:
            evaluator_cache[cache_key] = build_evaluator(representation, target_grid)
        evaluators[key] = evaluator_cache[cache_key]
    return evaluators


@dataclass
class _GeographicEvaluation:
    """Cached operators behind the fixed geographic display grid."""

    scalar_grid: object
    vector_grid: object
    state_evaluator: object | None
    input_evaluators: dict[str, object | None] = field(default_factory=dict)
    state_evaluation_context: dict[str, object] | None = None
    sheet_current_maps: dict[str, object] | None = None


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


def _state_evaluation_context(config, geometry, evaluator):
    """Return geometry and maps for evaluating saved state fields."""
    ri = float(config.RI)
    poloidal_evaluator = geometry.poloidal_transform_for(evaluator)
    return {
        "RI": ri,
        "m_ind_to_Br": (
            poloidal_evaluator.scalar_coeffs_to_grid_operator @ geometry.m_ind_to_Br_operator
        ),
        "m_imp_to_jr": evaluator.scalar_coeffs_to_grid_operator @ geometry.m_imp_to_jr_operator,
        "m_ind_to_Jeq": (-ri / mu0)
        * (
            poloidal_evaluator.scalar_coeffs_to_grid_operator
            @ geometry.poloidal_to_boundary_potential_jump_factor_operator
        ),
    }


def _sheet_current_maps(geometry, evaluator):
    """Return source-to-sheet-current maps on the plotting grid."""
    poloidal_evaluator = geometry.poloidal_transform_for(evaluator)
    return {
        "m_ind_to_JS": geometry.m_ind_to_gridded_JS(
            evaluator, poloidal_transform=poloidal_evaluator
        ),
        "m_imp_to_JS": geometry.m_imp_to_gridded_JS(
            evaluator, poloidal_transform=poloidal_evaluator
        ),
        "Br_to_JS": geometry.Br_to_gridded_JS(evaluator, poloidal_transform=poloidal_evaluator),
    }


def _state_fields_from_coefficients(
    m_ind, m_imp, phi_coeffs, w_coeffs, evaluator, state_evaluation_context, field_names
):
    """Evaluate flattened map fields from one state coefficient row."""
    fields = {}
    if "Br" in field_names:
        fields["Br"] = _apply_flat_operator(state_evaluation_context["m_ind_to_Br"], m_ind)
    if "jr" in field_names:
        fields["jr"] = _apply_flat_operator(state_evaluation_context["m_imp_to_jr"], m_imp)
    if "Jeq" in field_names:
        fields["Jeq"] = _apply_flat_operator(state_evaluation_context["m_ind_to_Jeq"], m_ind)
    radius_scale = float(state_evaluation_context["RI"]) * 1e-3
    if "Phi" in field_names:
        fields["Phi"] = _apply_flat_operator(
            evaluator.scalar_coeffs_to_grid_operator, radius_scale * phi_coeffs
        )
    if "W" in field_names:
        fields["W"] = _apply_flat_operator(
            evaluator.scalar_coeffs_to_grid_operator, radius_scale * w_coeffs
        )
    return fields


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
    return evaluator.scalar_coeffs_to_grid.dot(
        dataset[stored_name].isel(time=index).values
    ).reshape(shape)


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
    resistance_evaluator,
    state_evaluation_context,
    sheet_current_maps,
    *,
    target_time=None,
    fallback_start_time=None,
    field_names=None,
):
    """Evaluate state and steady-state map fields at one saved index."""
    field_names = _normalize_state_field_names(field_names)
    output_keys = [key for key in ("state", "steady_state") if key in datasets]
    if not output_keys:
        raise ValueError("No saved state or steady_state output is available.")
    reference_key = output_keys[0]
    reference_dataset = datasets[reference_key]
    if target_time is None:
        target_time = datetime_at_index(
            reference_dataset.time.values, int(index), fallback_start_time=fallback_start_time
        )

    br_mag = None
    if "joule" in field_names and "Br" in datasets:
        br_var = _dataset_var_name(datasets["Br"], "Br")
        if br_var is not None:
            br_index = _dataset_index_at_time(
                datasets["Br"], target_time, fallback_start_time=fallback_start_time
            )
            br_mag = datasets["Br"][br_var].isel(time=br_index).values
    etaP = None
    if "joule" in field_names and "resistance" in datasets:
        etaP_var = _dataset_var_name(datasets["resistance"], "etaP")
        if etaP_var is not None:
            resistance_index = _dataset_index_at_time(
                datasets["resistance"], target_time, fallback_start_time=fallback_start_time
            )
            etaP_coeffs = datasets["resistance"][etaP_var].isel(time=resistance_index).values
            etaP = resistance_evaluator.scalar_coeffs_to_grid.dot(etaP_coeffs)

    result = {}
    for dataset_key in output_keys:
        dataset = datasets[dataset_key]
        output_index = (
            int(index)
            if dataset_key == reference_key
            else _dataset_index_at_time(
                dataset, target_time, fallback_start_time=fallback_start_time
            )
        )
        m_ind = (
            _required_dataset_values(dataset, "m_ind", output_index)
            if field_names & {"Br", "Jeq", "joule"}
            else None
        )
        m_imp = (
            _required_dataset_values(dataset, "m_imp", output_index)
            if field_names & {"jr", "joule"}
            else None
        )
        fields = _state_fields_from_coefficients(
            m_ind,
            m_imp,
            (
                _required_dataset_values(dataset, "Phi", output_index)
                if "Phi" in field_names
                else None
            ),
            (_required_dataset_values(dataset, "W", output_index) if "W" in field_names else None),
            evaluator,
            state_evaluation_context,
            field_names,
        )
        if "joule" in field_names:
            if sheet_current_maps is None:
                raise ValueError("sheet_current_maps are required for Joule heating.")
            sheet_current = evaluate_JS_from_maps(
                m_imp,
                m_ind,
                m_imp_to_JS=sheet_current_maps["m_imp_to_JS"],
                m_ind_to_JS=sheet_current_maps["m_ind_to_JS"],
                Br=br_mag,
                Br_to_JS=sheet_current_maps["Br_to_JS"],
            )
            fields["joule"] = (
                joule_heating_from_current(
                    sheet_current, etaP, state_evaluation_context["pedersen_geometry"]
                )
                if etaP is not None
                else np.full(evaluator.grid.size, np.nan)
            )
        result["state" if dataset_key == "state" else "steady"] = fields
    return result


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
    if "resistance" in datasets:
        eta_p = _input_scalar_grid_at_time(
            datasets,
            "resistance",
            "etaP",
            timestamp,
            input_evaluators["resistance"],
            scalar_shape,
            fallback_start_time=fallback_start_time,
        ).reshape(-1)
        eta_h = _input_scalar_grid_at_time(
            datasets,
            "resistance",
            "etaH",
            timestamp,
            input_evaluators["resistance"],
            scalar_shape,
            fallback_start_time=fallback_start_time,
        ).reshape(-1)
        conductance = evaluate_conductance_values(eta_p, eta_h)
        sigma_p = conductance["SigmaP"].reshape(scalar_shape)
        sigma_h = conductance["SigmaH"].reshape(scalar_shape)
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

    run_view: SavedRunView
    lat: np.ndarray
    lon: np.ndarray
    wind_lat: np.ndarray
    wind_lon: np.ndarray
    state_evaluator: object | None
    input_evaluators: dict[str, object | None]
    state_evaluation_context: dict[str, object] | None
    sheet_current_maps: dict[str, object] | None
    _geographic_evaluation: _GeographicEvaluation | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_directory(
        cls, run_directory, *, nlat=60, nlon=100, wind_nlat=19, wind_nlon=37
    ) -> SavedCoefficientFieldView:
        """Load artifacts needed by map and input-driver figures."""
        run_view = SavedRunView.from_directory(
            run_directory, optional_datasets=INPUT_ARTIFACT_KEYS + ("state", "steady_state")
        )
        schema = run_view.schema
        has_output_state = any(key in run_view.datasets for key in ("state", "steady_state"))
        output_basis = schema.output_field_spaces["state"]["m_imp"].representation
        lat, lon, grid = build_plot_grid(nlat=nlat, nlon=nlon)
        wind_lat, wind_lon, wind_grid = build_plot_grid(
            nlat=wind_nlat, nlon=wind_nlon, lat_range=(-75.0, 75.0), lon_range=(-180.0, 180.0)
        )
        state_evaluator = build_evaluator(output_basis, grid) if has_output_state else None
        evaluator_cache = {}
        if state_evaluator is not None:
            evaluator_cache[
                (getattr(output_basis, "signature", id(output_basis)), grid.signature)
            ] = state_evaluator
        input_evaluators = _build_input_evaluators(
            schema, run_view.datasets, grid, wind_grid, evaluator_cache
        )

        time_datasets = _time_datasets(run_view.datasets)
        if not time_datasets:
            raise ValueError(
                "No saved input or output time series exists in "
                f"{run_view.artifact_store.directory}. "
                "Expected at least one of state, Br, jr, resistance, u, Q_eff, or E_source."
            )

        return cls(
            run_view=run_view,
            lat=lat,
            lon=lon,
            wind_lat=wind_lat,
            wind_lon=wind_lon,
            state_evaluator=state_evaluator,
            input_evaluators=input_evaluators,
            state_evaluation_context=None,
            sheet_current_maps=None,
        )

    def require_geometry(self):
        """Return the lazily constructed saved-run geometry."""
        if not self.has_output_state:
            raise ValueError("Saved-run geometry requires state or steady_state output.")
        return self.run_view.require_geometry()

    def _get_geographic_evaluation(self, event_time=None):
        """Return evaluators sampled on the geographic display grid.

        Saved coefficients live in the simulation's horizontal
        coordinate system. The model frame is Earth-fixed, so this
        model-to-geographic sampling geometry is immutable.
        """
        del event_time
        if self._geographic_evaluation is not None:
            return self._geographic_evaluation

        main_field = self.run_view.main_field
        scalar_grid = model_grid_for_geographic_display(main_field, self.lat, self.lon)
        vector_grid = model_grid_for_geographic_display(main_field, self.wind_lat, self.wind_lon)

        evaluation = _GeographicEvaluation(
            scalar_grid=scalar_grid, vector_grid=vector_grid, state_evaluator=None
        )
        self._geographic_evaluation = evaluation
        return evaluation

    def _geographic_state_evaluator(self, evaluation):
        """Return the lazy state evaluator for a geographic map."""
        if evaluation.state_evaluator is None:
            output_basis = self.run_view.schema.output_field_spaces["state"][
                "m_imp"
            ].representation
            evaluation.state_evaluator = build_evaluator(output_basis, evaluation.scalar_grid)
        return evaluation.state_evaluator

    def geographic_map_context(self, reference_time=None):
        """Return a geographic map centered on solar local noon."""
        if reference_time is None:
            reference_time = self.run_view.config.t0
        return MapCoordinateContext.geographic(pd.Timestamp(reference_time).to_pydatetime())

    def magnetic_map_context(self, reference_time=None):
        """Return a magnetic map centered on magnetic local noon."""
        if reference_time is None:
            reference_time = self.run_view.config.t0
        reference_time = pd.Timestamp(reference_time).to_pydatetime()
        return MapCoordinateContext.from_noon_longitude(
            self.run_view.main_field.magnetic_noon_longitude(reference_time),
            longitude_kind="magnetic",
            local_time_kind="magnetic",
            label="MLT",
            reference_time=reference_time,
        )

    def magnetic_plot_coordinates(self):
        """Return MAG coordinates of the regular model plotting grid."""
        main_field = self.run_view.main_field
        geographic_latitude, geographic_longitude = main_field.model_to_geo_coordinates(
            self.lat, self.lon
        )
        return main_field.geographic_to_magnetic_coordinates(
            geographic_latitude, geographic_longitude
        )

    def model_map_context(self, reference_time=None):
        """Return the model-coordinate local-time context."""
        if reference_time is None:
            reference_time = self.run_view.config.t0
        reference_time = pd.Timestamp(reference_time).to_pydatetime()
        main_field = self.run_view.main_field
        if main_field.horizontal_coordinate_system == "geographic":
            return MapCoordinateContext.geographic(reference_time)
        return MapCoordinateContext.from_noon_longitude(
            main_field.local_noon_longitude(reference_time),
            longitude_kind="magnetic",
            local_time_kind="magnetic",
            label="MLT",
            reference_time=reference_time,
        )

    def _geographic_input_evaluators(self, evaluation, *, keys=INPUT_ARTIFACT_KEYS):
        """Return input evaluators on the geographic map grid."""
        missing = tuple(key for key in keys if key not in evaluation.input_evaluators)
        if missing:
            evaluation.input_evaluators.update(
                _build_input_evaluators(
                    self.run_view.schema,
                    self.run_view.datasets,
                    evaluation.scalar_grid,
                    evaluation.vector_grid,
                    keys=missing,
                )
            )
        return evaluation.input_evaluators

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
        """Return whether any model-state output is present."""
        return "state" in self.run_view.datasets or "steady_state" in self.run_view.datasets

    def _time_dataset(self):
        """Return the dataset that defines display times."""
        if "state" in self.run_view.datasets:
            return self.run_view.datasets["state"]
        if "steady_state" in self.run_view.datasets:
            return self.run_view.datasets["steady_state"]
        time_datasets = _time_datasets(self.run_view.datasets)
        if not time_datasets:
            raise ValueError("No saved time-dependent artifacts are available.")
        return time_datasets[0]

    @property
    def available_inputs(self):
        """Return projected input names available in this directory."""
        return tuple(key for key in INPUT_ARTIFACT_KEYS if key in self.run_view.datasets)

    def dataset_values(self, dataset_key, variable_name):
        """Return stored values for one logical dataset variable."""
        dataset = self.run_view.datasets[dataset_key]
        stored_name = _dataset_var_name(dataset, variable_name)
        if stored_name is None:
            available = ", ".join(str(name) for name in dataset.data_vars) or "none"
            raise KeyError(
                f"{dataset_key!r} has no saved {variable_name!r} coefficients; found {available}."
            )
        return dataset[stored_name].values

    def _fallback_start_time(self):
        """Return the configured start time for numeric saved times."""
        return self.run_view.config.t0

    def state_comparison_fields(self, index, *, field_names=None, coordinate_system="model"):
        """Return flat fields in model or geographic coordinates."""
        field_names = _normalize_state_field_names(field_names)
        coordinate_system = _normalize_display_coordinate_system(coordinate_system)
        index = int(max(0, min(int(index), self.n_time - 1)))
        timestamp = self.timestamp_at_index(index)
        if not self.has_output_state:
            raise ValueError(
                "This directory contains projected inputs but no saved output state. "
                "Choose 'Input drivers' or run a simulation first."
            )
        if coordinate_system == "geographic":
            evaluation = self._get_geographic_evaluation(timestamp)
            evaluator = self._geographic_state_evaluator(evaluation)
            state_evaluation_context = evaluation.state_evaluation_context
            sheet_current_maps = evaluation.sheet_current_maps
        else:
            evaluation = None
            evaluator = self.state_evaluator
            state_evaluation_context = self.state_evaluation_context
            sheet_current_maps = self.sheet_current_maps

        if evaluator is None:
            raise RuntimeError("Saved state evaluation context is unavailable.")
        geometry = self.require_geometry()
        if state_evaluation_context is None:
            state_evaluation_context = _state_evaluation_context(
                self.run_view.config, geometry, evaluator
            )
        needs_joule = "joule" in field_names
        if needs_joule and sheet_current_maps is None:
            sheet_current_maps = _sheet_current_maps(geometry, evaluator)
        if needs_joule and "pedersen_geometry" not in state_evaluation_context:
            field = MagneticFieldEvaluation(
                geometry.main_field, evaluator.grid, self.run_view.config.RI
            )
            state_evaluation_context["pedersen_geometry"] = pedersen_geometry_tensor(
                field.unit_btheta, field.unit_bphi, field.unit_br
            )
        if evaluation is None:
            self.state_evaluation_context = state_evaluation_context
            self.sheet_current_maps = sheet_current_maps
        else:
            evaluation.state_evaluation_context = state_evaluation_context
            evaluation.sheet_current_maps = sheet_current_maps
        resistance_evaluator = self.input_evaluators["resistance"]
        if evaluation is not None:
            resistance_evaluator = (
                self._geographic_input_evaluators(evaluation, keys=("resistance",))["resistance"]
                if needs_joule
                else None
            )
        return compute_state_comparison_fields_at_index(
            index,
            self.run_view.datasets,
            evaluator,
            resistance_evaluator,
            state_evaluation_context,
            sheet_current_maps,
            target_time=timestamp,
            fallback_start_time=self._fallback_start_time(),
            field_names=field_names,
        )

    def state_comparison_grid_fields(self, index, *, field_names=None, coordinate_system="model"):
        """Return gridded fields in model or geographic coordinates."""
        fields = self.state_comparison_fields(
            index, field_names=field_names, coordinate_system=coordinate_system
        )
        return {
            f"{name}_{state_key}": values.reshape(self.lat.shape)
            for state_key, state_fields in fields.items()
            for name, values in state_fields.items()
        }

    def input_grid_fields(self, index, *, coordinate_system="model"):
        """Return input-driver fields in the requested coordinates."""
        return self.input_grid_fields_at_time(
            self.timestamp_at_index(index), coordinate_system=coordinate_system
        )

    def input_grid_fields_at_time(self, timestamp, *, coordinate_system="model"):
        """Return time-selected inputs in the requested coordinates."""
        coordinate_system = _normalize_display_coordinate_system(coordinate_system)
        evaluation = (
            self._get_geographic_evaluation(timestamp)
            if coordinate_system == "geographic"
            else None
        )
        input_evaluators = (
            self._geographic_input_evaluators(evaluation)
            if evaluation is not None
            else self.input_evaluators
        )
        fields = compute_input_fields_at_time(
            timestamp,
            self.run_view.datasets,
            input_evaluators,
            self.lat.shape,
            self.wind_lat.shape,
            fallback_start_time=self._fallback_start_time(),
        )
        if evaluation is None:
            return fields

        main_field = self.run_view.main_field
        for key in TANGENTIAL_INPUT_KEYS:
            theta_key = f"{key if key != 'u' else 'wind'}_theta"
            phi_key = f"{key if key != 'u' else 'wind'}_phi"
            theta = fields[theta_key]
            phi = fields[phi_key]
            if not np.any(np.isfinite(theta) & np.isfinite(phi)):
                continue
            _, _, east, north = main_field.model_to_geo_coordinates(
                evaluation.vector_grid.lat.reshape(self.wind_lat.shape),
                evaluation.vector_grid.lon.reshape(self.wind_lon.shape),
                phi,
                -theta,
            )
            fields[theta_key] = -np.asarray(north).reshape(self.wind_lat.shape)
            fields[phi_key] = np.asarray(east).reshape(self.wind_lat.shape)
        return fields


__all__ = [
    "SavedCoefficientFieldView",
    "compute_input_fields_at_time",
    "compute_state_comparison_fields_at_index",
    "datetime_at_index",
    "time_index_from_dataset",
]
