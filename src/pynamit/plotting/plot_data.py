"""Saved coefficient fields evaluated on plotting grids."""

from __future__ import annotations

import datetime as dt
import stat
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from kompe import SphericalTransform
from kompe.constants import MU0

from pynamit.coordinates import GEOCENTRIC_GEOGRAPHIC
from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.map_coordinates import MapCoordinateContext
from pynamit.results.evaluation import (
    build_plot_grid,
    evaluate_conductance_values,
    evaluate_sheet_current_from_operators,
    model_grid_for_geographic_display,
)
from pynamit.results.simulation_results import SimulationResults
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    joule_heating_from_current,
    pedersen_geometry_tensor,
)
from pynamit.simulation.schema import SIMULATION_ARTIFACT_NAMES
from pynamit.storage import ArtifactStore

INPUT_ARTIFACT_KEYS = ("boundary_Br", "boundary_jr", "conductance", "u", "Q_eff", "E_neutral_wind")
TANGENTIAL_INPUT_KEYS = ("u", "Q_eff", "E_neutral_wind")
OUTPUT_FIELD_NAMES = frozenset({"Br", "jr", "Jeq", "Phi", "W", "joule"})
_DISPLAY_COORDINATE_SYSTEMS = frozenset({"model", "geographic"})
_CACHE_ARTIFACTS = tuple(sorted(SIMULATION_ARTIFACT_NAMES))
_PLOT_DATA_CACHE: dict[tuple[str, int, int], tuple[tuple, PlotData]] = {}


def _normalize_display_coordinate_system(coordinate_system):
    """Return a supported map-display coordinate system."""
    normalized = str(coordinate_system).strip().lower()
    if normalized not in _DISPLAY_COORDINATE_SYSTEMS:
        raise ValueError(
            f"coordinate_system must be either 'model' or 'geographic'; got {coordinate_system!r}."
        )
    return normalized


def _normalize_output_field_names(field_names):
    """Return a validated immutable field selection."""
    if field_names is None:
        selected = OUTPUT_FIELD_NAMES
    elif isinstance(field_names, str):
        selected = frozenset({field_names})
    else:
        selected = frozenset(field_names)
    unknown_fields = selected - OUTPUT_FIELD_NAMES
    if unknown_fields:
        raise ValueError(f"Unknown output fields requested: {sorted(unknown_fields)}.")
    return selected


def _time_datasets(datasets):
    return [
        dataset
        for dataset in datasets.values()
        if "time" in dataset.coords or "time" in dataset.dims
    ]


def _nan_field(shape):
    return np.full(shape, np.nan, dtype=float)


def _build_input_transforms(
    schema, datasets, scalar_grid, vector_grid, transform_cache=None, *, keys=INPUT_ARTIFACT_KEYS
):
    """Build input transforms for the requested display grids."""
    transform_cache = {} if transform_cache is None else transform_cache
    transforms = {}
    for key in keys:
        if key not in datasets:
            transforms[key] = None
            continue
        target_grid = vector_grid if key in TANGENTIAL_INPUT_KEYS else scalar_grid
        basis = schema.input_field_spaces[key].basis
        cache_key = (basis.signature, target_grid.signature)
        if cache_key not in transform_cache:
            transform_cache[cache_key] = SphericalTransform(basis, target_grid)
        transforms[key] = transform_cache[cache_key]
    return transforms


@dataclass
class _GeographicEvaluation:
    """Cached operators behind the fixed geographic display grid."""

    scalar_grid: object
    vector_grid: object
    output_transform: SphericalTransform | None
    input_transforms: dict[str, SphericalTransform | None] = field(default_factory=dict)
    output_evaluation_context: dict[str, object] | None = None
    sheet_current_maps: dict[str, object] | None = None


def _required_dataset_values(results, dataset_key, variable_name, index):
    """Return stored coefficient values for one logical variable."""
    dataset = results.datasets[dataset_key]
    stored_name = results.data_var_name(dataset_key, variable_name)
    return np.asarray(dataset[stored_name].isel(time=index).values)


def _apply_flat_operator(operator, coeffs):
    """Apply an operator and return a flat NumPy vector."""
    return np.asarray(operator.matvec(np.asarray(coeffs))).reshape(-1)


def _output_evaluation_context(config, geometry, transform):
    """Return geometry and maps for evaluating saved output fields."""
    ri = float(config.RI)
    poloidal_transform = geometry.poloidal_transform_for(transform)
    return {
        "RI": ri,
        "induced_Br_to_Br": poloidal_transform.scalar_synthesis_operator,
        "boundary_jr_to_jr": transform.scalar_synthesis_operator,
        "induced_Br_to_Jeq": (-ri / MU0)
        * (
            poloidal_transform.scalar_synthesis_operator
            @ geometry.poloidal_to_boundary_potential_jump_factor_operator
            @ geometry.induced_Br_to_poloidal_potential_operator
        ),
    }


def _sheet_current_maps(geometry, transform):
    """Return source-to-sheet-current maps on the plotting grid."""
    poloidal_transform = geometry.poloidal_transform_for(transform)
    boundary_Br_operator = geometry.boundary_Br_to_gridded_JS_operator(
        transform, poloidal_transform=poloidal_transform
    )
    return {
        "induced_Br_to_JS": geometry.induced_Br_to_gridded_JS_operator(
            transform, poloidal_transform=poloidal_transform
        ),
        "boundary_jr_to_JS": geometry.boundary_jr_to_gridded_JS_operator(
            transform, poloidal_transform=poloidal_transform
        ),
        "boundary_Br_to_JS": boundary_Br_operator,
    }


def _output_fields_from_coefficients(
    induced_Br,
    boundary_jr,
    phi_coeffs,
    w_coeffs,
    transform,
    output_evaluation_context,
    field_names,
):
    """Evaluate flattened map fields from one output row."""
    fields = {}
    if "Br" in field_names:
        fields["Br"] = _apply_flat_operator(
            output_evaluation_context["induced_Br_to_Br"], induced_Br
        )
    if "jr" in field_names:
        fields["jr"] = _apply_flat_operator(
            output_evaluation_context["boundary_jr_to_jr"], boundary_jr
        )
    if "Jeq" in field_names:
        fields["Jeq"] = _apply_flat_operator(
            output_evaluation_context["induced_Br_to_Jeq"], induced_Br
        )
    radius_scale = float(output_evaluation_context["RI"]) * 1e-3
    if "Phi" in field_names:
        fields["Phi"] = _apply_flat_operator(
            transform.scalar_synthesis_operator, radius_scale * phi_coeffs
        )
    if "W" in field_names:
        fields["W"] = _apply_flat_operator(
            transform.scalar_synthesis_operator, radius_scale * w_coeffs
        )
    return fields


def _dataset_index_at_time(dataset, timestamp, *, start_time=None):
    """Return the latest dataset index at or before ``timestamp``."""
    times = time_index_from_dataset(dataset, start_time=start_time)
    if times.empty:
        raise ValueError("No time coordinates are available.")

    target = pd.Timestamp(timestamp)
    if target.tz is not None:
        target = target.tz_convert(None)
    position = int(times.searchsorted(target, side="right") - 1)
    return max(0, min(position, len(times) - 1))


def _input_scalar_grid_at_time(
    results, dataset_key, variable_name, timestamp, transform, shape, *, start_time=None
):
    dataset = results.datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape)
    stored_name = results.data_var_name(dataset_key, variable_name)
    index = _dataset_index_at_time(dataset, timestamp, start_time=start_time)
    return transform.scalar_synthesis_matrix.dot(
        dataset[stored_name].isel(time=index).values
    ).reshape(shape)


def _input_tangential_grid_at_time(
    results, dataset_key, variable_name, timestamp, transform, shape, *, start_time=None
):
    dataset = results.datasets.get(dataset_key)
    if dataset is None:
        return _nan_field(shape), _nan_field(shape)
    stored_name = results.data_var_name(dataset_key, variable_name)
    index = _dataset_index_at_time(dataset, timestamp, start_time=start_time)
    theta_grid, phi_grid = transform.synthesize_helmholtz(
        dataset[stored_name].isel(time=index).values
    )
    return theta_grid.reshape(shape), phi_grid.reshape(shape)


def datetime_at_index(times, index, *, start_time=None):
    """Return one saved time value as a pandas timestamp."""
    values = np.asarray(times)
    if values.size == 0:
        raise ValueError("No time coordinates are available.")
    idx = int(max(0, min(int(index), values.size - 1)))
    value = values[idx]
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.Timestamp(value)
    if start_time is None:
        raise ValueError("Numeric simulation times require the physical start_time.")
    return pd.Timestamp(start_time) + pd.to_timedelta(float(value), unit="s")


def time_index_from_dataset(dataset, *, start_time=None):
    """Return dataset times as a ``DatetimeIndex``."""
    return pd.DatetimeIndex(
        [
            datetime_at_index(dataset.time.values, index, start_time=start_time)
            for index in range(len(dataset.time))
        ]
    )


def compute_output_fields_at_index(
    index,
    results,
    transform,
    conductance_transform,
    output_evaluation_context,
    sheet_current_maps,
    *,
    target_time=None,
    start_time=None,
    field_names=None,
):
    """Evaluate dynamic and equilibrium fields at one saved index."""
    datasets = results.datasets
    field_names = _normalize_output_field_names(field_names)
    output_keys = [key for key in ("dynamic", "equilibrium") if key in datasets]
    if not output_keys:
        raise ValueError("No saved dynamic or equilibrium output is available.")
    reference_key = output_keys[0]
    reference_dataset = datasets[reference_key]
    if target_time is None:
        target_time = datetime_at_index(
            reference_dataset.time.values, int(index), start_time=start_time
        )

    boundary_Br = None
    if "joule" in field_names and "boundary_Br" in datasets:
        boundary_Br_var = results.data_var_name("boundary_Br", "boundary_Br")
        boundary_Br_index = _dataset_index_at_time(
            datasets["boundary_Br"], target_time, start_time=start_time
        )
        boundary_Br = datasets["boundary_Br"][boundary_Br_var].isel(time=boundary_Br_index).values
    etaP = None
    if "joule" in field_names and "conductance" in datasets:
        log_magnitude_var = results.data_var_name("conductance", "log_conductance_magnitude")
        log_ratio_var = results.data_var_name("conductance", "log_hall_to_pedersen_ratio")
        conductance_index = _dataset_index_at_time(
            datasets["conductance"], target_time, start_time=start_time
        )
        log_magnitude = conductance_transform.scalar_synthesis_matrix.dot(
            datasets["conductance"][log_magnitude_var].isel(time=conductance_index).values
        )
        log_ratio = conductance_transform.scalar_synthesis_matrix.dot(
            datasets["conductance"][log_ratio_var].isel(time=conductance_index).values
        )
        etaP = evaluate_conductance_values(log_magnitude, log_ratio)["etaP"]

    result = {}
    for dataset_key in output_keys:
        dataset = datasets[dataset_key]
        output_index = (
            int(index)
            if dataset_key == reference_key
            else _dataset_index_at_time(dataset, target_time, start_time=start_time)
        )
        induced_Br = (
            _required_dataset_values(results, dataset_key, "induced_Br", output_index)
            if field_names & {"Br", "Jeq", "joule"}
            else None
        )
        boundary_jr = (
            _required_dataset_values(results, dataset_key, "boundary_jr", output_index)
            if field_names & {"jr", "joule"}
            else None
        )
        fields = _output_fields_from_coefficients(
            induced_Br,
            boundary_jr,
            (
                _required_dataset_values(results, dataset_key, "Phi", output_index)
                if "Phi" in field_names
                else None
            ),
            (
                _required_dataset_values(results, dataset_key, "W", output_index)
                if "W" in field_names
                else None
            ),
            transform,
            output_evaluation_context,
            field_names,
        )
        if "joule" in field_names:
            if sheet_current_maps is None:
                raise ValueError("sheet_current_maps are required for Joule heating.")
            sheet_current = evaluate_sheet_current_from_operators(
                boundary_jr,
                induced_Br,
                boundary_jr_to_JS=sheet_current_maps["boundary_jr_to_JS"],
                induced_Br_to_JS=sheet_current_maps["induced_Br_to_JS"],
                boundary_Br=boundary_Br,
                boundary_Br_to_JS=sheet_current_maps["boundary_Br_to_JS"],
            )
            fields["joule"] = (
                joule_heating_from_current(
                    sheet_current, etaP, output_evaluation_context["pedersen_geometry"]
                )
                if etaP is not None
                else np.full(transform.grid.size, np.nan)
            )
        result[dataset_key] = fields
    return result


def compute_input_fields_at_time(
    timestamp, results, input_transforms, scalar_shape, vector_shape, *, start_time=None
):
    """Evaluate projected input drivers at one physical time."""
    datasets = results.datasets
    boundary_jr = _input_scalar_grid_at_time(
        results,
        "boundary_jr",
        "boundary_jr",
        timestamp,
        input_transforms["boundary_jr"],
        scalar_shape,
        start_time=start_time,
    )
    boundary_Br = _input_scalar_grid_at_time(
        results,
        "boundary_Br",
        "boundary_Br",
        timestamp,
        input_transforms["boundary_Br"],
        scalar_shape,
        start_time=start_time,
    )
    if "conductance" in datasets:
        log_magnitude = _input_scalar_grid_at_time(
            results,
            "conductance",
            "log_conductance_magnitude",
            timestamp,
            input_transforms["conductance"],
            scalar_shape,
            start_time=start_time,
        ).reshape(-1)
        log_ratio = _input_scalar_grid_at_time(
            results,
            "conductance",
            "log_hall_to_pedersen_ratio",
            timestamp,
            input_transforms["conductance"],
            scalar_shape,
            start_time=start_time,
        ).reshape(-1)
        conductance = evaluate_conductance_values(log_magnitude, log_ratio)
        sigma_p = conductance["SigmaP"].reshape(scalar_shape)
        sigma_h = conductance["SigmaH"].reshape(scalar_shape)
    else:
        sigma_p = _nan_field(scalar_shape)
        sigma_h = _nan_field(scalar_shape)

    tangential = {
        key: _input_tangential_grid_at_time(
            results,
            key,
            key,
            timestamp,
            input_transforms[key],
            vector_shape,
            start_time=start_time,
        )
        for key in TANGENTIAL_INPUT_KEYS
    }

    return {
        "jr": boundary_jr,
        "Br": boundary_Br,
        "sigmaP": sigma_p,
        "sigmaH": sigma_h,
        "wind_theta": tangential["u"][0],
        "wind_phi": tangential["u"][1],
        "Q_eff_theta": tangential["Q_eff"][0],
        "Q_eff_phi": tangential["Q_eff"][1],
        "E_neutral_wind_theta": tangential["E_neutral_wind"][0],
        "E_neutral_wind_phi": tangential["E_neutral_wind"][1],
    }


@dataclass
class PlotData:
    """Evaluate stored coefficients on plotting grids."""

    results: SimulationResults
    lat: np.ndarray
    lon: np.ndarray
    wind_lat: np.ndarray
    wind_lon: np.ndarray
    output_transform: SphericalTransform | None
    input_transforms: dict[str, SphericalTransform | None]
    output_evaluation_context: dict[str, object] | None
    sheet_current_maps: dict[str, object] | None
    _geographic_evaluation: _GeographicEvaluation | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_directory(
        cls, simulation_directory, *, nlat=60, nlon=100, wind_nlat=19, wind_nlon=37
    ) -> PlotData:
        """Load artifacts needed by map and input-driver figures."""
        results = SimulationResults.from_directory(
            simulation_directory,
            optional_datasets=INPUT_ARTIFACT_KEYS + ("dynamic", "equilibrium"),
        )
        schema = results.schema
        has_model_output = any(key in results.datasets for key in ("dynamic", "equilibrium"))
        output_basis = schema.output_field_spaces["dynamic"]["boundary_jr"].basis
        lat, lon, grid = build_plot_grid(nlat=nlat, nlon=nlon)
        wind_lat, wind_lon, wind_grid = build_plot_grid(
            nlat=wind_nlat, nlon=wind_nlon, lat_range=(-75.0, 75.0), lon_range=(-180.0, 180.0)
        )
        output_transform = SphericalTransform(output_basis, grid) if has_model_output else None
        transform_cache = {}
        if output_transform is not None:
            transform_cache[(output_basis.signature, grid.signature)] = output_transform
        input_transforms = _build_input_transforms(
            schema, results.datasets, grid, wind_grid, transform_cache
        )

        time_datasets = _time_datasets(results.datasets)
        if not time_datasets:
            raise ValueError(
                "No saved input or output time series exists in "
                f"{results.artifact_store.directory}. "
                "Expected at least one of dynamic, boundary_Br, boundary_jr, "
                "conductance, u, Q_eff, or E_neutral_wind."
            )

        return cls(
            results=results,
            lat=lat,
            lon=lon,
            wind_lat=wind_lat,
            wind_lon=wind_lon,
            output_transform=output_transform,
            input_transforms=input_transforms,
            output_evaluation_context=None,
            sheet_current_maps=None,
        )

    def load_geometry(self):
        """Load and return the saved simulation geometry."""
        if not self.has_model_output:
            raise ValueError("Saved simulation geometry requires dynamic or equilibrium output.")
        return self.results.load_geometry()

    def __repr__(self):
        """Summarize evaluated fields without printing their arrays."""
        inputs = ", ".join(self.available_inputs) or "none"
        return (
            f"PlotData(n_time={self.n_time}, inputs=[{inputs}], "
            f"has_model_output={self.has_model_output}, "
            f"simulation_directory={self.results.simulation_directory!r})"
        )

    def _get_geographic_evaluation(self, event_time=None):
        """Return evaluators sampled on the geographic display grid.

        Saved coefficients live in the simulation's horizontal
        coordinate system. Its orientation is fixed by the persisted
        main-field epoch, so this sampling geometry is immutable.
        """
        del event_time
        if self._geographic_evaluation is not None:
            return self._geographic_evaluation

        main_field = self.results.main_field
        scalar_grid = model_grid_for_geographic_display(main_field, self.lat, self.lon)
        vector_grid = model_grid_for_geographic_display(main_field, self.wind_lat, self.wind_lon)

        evaluation = _GeographicEvaluation(
            scalar_grid=scalar_grid, vector_grid=vector_grid, output_transform=None
        )
        self._geographic_evaluation = evaluation
        return evaluation

    def _geographic_output_transform(self, evaluation):
        """Return the lazy output transform for a geographic map."""
        if evaluation.output_transform is None:
            output_basis = self.results.schema.output_field_spaces["dynamic"]["boundary_jr"].basis
            evaluation.output_transform = SphericalTransform(output_basis, evaluation.scalar_grid)
        return evaluation.output_transform

    def geographic_map_context(self, reference_time=None):
        """Return a geographic map centered on mean-solar local noon."""
        if reference_time is None:
            reference_time = self.results.config.t0
        return MapCoordinateContext.geographic(pd.Timestamp(reference_time).to_pydatetime())

    def magnetic_map_context(self, reference_time=None):
        """Return a magnetic map centered on magnetic local noon."""
        if reference_time is None:
            reference_time = self.results.config.t0
        reference_time = pd.Timestamp(reference_time).to_pydatetime()
        return MapCoordinateContext.from_noon_longitude(
            self.results.main_field.magnetic_noon_longitude(reference_time),
            longitude_kind="magnetic",
            local_time_kind="magnetic",
            label="MLT",
            reference_time=reference_time,
        )

    def magnetic_plot_coordinates(self):
        """Return MAG coordinates of the regular model plotting grid."""
        main_field = self.results.main_field
        geographic_latitude, geographic_longitude = main_field.model_to_geo_coordinates(
            self.lat, self.lon
        )
        return main_field.geographic_to_magnetic_coordinates(
            geographic_latitude, geographic_longitude
        )

    def model_map_context(self, reference_time=None):
        """Return the model-coordinate local-time context."""
        if reference_time is None:
            reference_time = self.results.config.t0
        reference_time = pd.Timestamp(reference_time).to_pydatetime()
        main_field = self.results.main_field
        if main_field.horizontal_coordinate_system == GEOCENTRIC_GEOGRAPHIC:
            return MapCoordinateContext.geographic(reference_time)
        return MapCoordinateContext.from_noon_longitude(
            main_field.local_noon_longitude(reference_time),
            longitude_kind="magnetic",
            local_time_kind="magnetic",
            label="MLT",
            reference_time=reference_time,
        )

    def _geographic_input_transforms(self, evaluation, *, keys=INPUT_ARTIFACT_KEYS):
        """Return input transforms on the geographic map grid."""
        missing = tuple(key for key in keys if key not in evaluation.input_transforms)
        if missing:
            evaluation.input_transforms.update(
                _build_input_transforms(
                    self.results.schema,
                    self.results.datasets,
                    evaluation.scalar_grid,
                    evaluation.vector_grid,
                    keys=missing,
                )
            )
        return evaluation.input_transforms

    @property
    def n_time(self):
        """Return the number of display time steps."""
        return len(self._time_dataset().time)

    @property
    def time_index(self):
        """Return saved times as datetimes."""
        return time_index_from_dataset(self._time_dataset(), start_time=self._start_time())

    def timestamp_at_index(self, index):
        """Return one saved time as a timestamp."""
        return datetime_at_index(
            self._time_dataset().time.values, index, start_time=self._start_time()
        )

    @property
    def has_model_output(self):
        """Return whether any model output is present."""
        return "dynamic" in self.results.datasets or "equilibrium" in self.results.datasets

    def _time_dataset(self):
        """Return the dataset that defines display times."""
        if "dynamic" in self.results.datasets:
            return self.results.datasets["dynamic"]
        if "equilibrium" in self.results.datasets:
            return self.results.datasets["equilibrium"]
        time_datasets = _time_datasets(self.results.datasets)
        if not time_datasets:
            raise ValueError("No saved time-dependent artifacts are available.")
        return time_datasets[0]

    @property
    def available_inputs(self):
        """Return projected input names available in this directory."""
        return tuple(key for key in INPUT_ARTIFACT_KEYS if key in self.results.datasets)

    def dataset_values(self, dataset_key, variable_name):
        """Return stored values for one logical dataset variable."""
        dataset = self.results.datasets[dataset_key]
        stored_name = self.results.data_var_name(dataset_key, variable_name)
        return dataset[stored_name].values

    def _start_time(self):
        """Return the configured start time for numeric saved times."""
        return self.results.config.t0

    def output_fields(self, index, *, field_names=None, coordinate_system="model"):
        """Return flat output fields in the requested coordinates."""
        field_names = _normalize_output_field_names(field_names)
        coordinate_system = _normalize_display_coordinate_system(coordinate_system)
        index = int(max(0, min(int(index), self.n_time - 1)))
        timestamp = self.timestamp_at_index(index)
        if not self.has_model_output:
            raise ValueError(
                "This directory contains projected inputs but no saved model output. "
                "Choose 'Input drivers' or run a simulation first."
            )
        if coordinate_system == "geographic":
            evaluation = self._get_geographic_evaluation(timestamp)
            transform = self._geographic_output_transform(evaluation)
            output_evaluation_context = evaluation.output_evaluation_context
            sheet_current_maps = evaluation.sheet_current_maps
        else:
            evaluation = None
            transform = self.output_transform
            output_evaluation_context = self.output_evaluation_context
            sheet_current_maps = self.sheet_current_maps

        if transform is None:
            raise RuntimeError("Saved output evaluation context is unavailable.")
        geometry = self.load_geometry()
        if output_evaluation_context is None:
            output_evaluation_context = _output_evaluation_context(
                self.results.config, geometry, transform
            )
        needs_joule = "joule" in field_names
        if needs_joule and sheet_current_maps is None:
            sheet_current_maps = _sheet_current_maps(geometry, transform)
        if needs_joule and "pedersen_geometry" not in output_evaluation_context:
            field = MagneticFieldEvaluation(
                geometry.main_field, transform.grid, self.results.config.RI
            )
            output_evaluation_context["pedersen_geometry"] = pedersen_geometry_tensor(
                field.unit_btheta, field.unit_bphi, field.unit_br
            )
        if evaluation is None:
            self.output_evaluation_context = output_evaluation_context
            self.sheet_current_maps = sheet_current_maps
        else:
            evaluation.output_evaluation_context = output_evaluation_context
            evaluation.sheet_current_maps = sheet_current_maps
        conductance_transform = self.input_transforms.get("conductance") if needs_joule else None
        if evaluation is not None:
            conductance_transform = (
                self._geographic_input_transforms(evaluation, keys=("conductance",))["conductance"]
                if needs_joule
                else None
            )
        return compute_output_fields_at_index(
            index,
            self.results,
            transform,
            conductance_transform,
            output_evaluation_context,
            sheet_current_maps,
            target_time=timestamp,
            start_time=self._start_time(),
            field_names=field_names,
        )

    def output_plot_data(self, index, *, field_names=None, coordinate_system="model"):
        """Return gridded output fields in the requested coordinates."""
        fields = self.output_fields(
            index, field_names=field_names, coordinate_system=coordinate_system
        )
        return {
            f"{name}_{output_key}": values.reshape(self.lat.shape)
            for output_key, output_fields in fields.items()
            for name, values in output_fields.items()
        }

    def input_plot_data(self, index, *, coordinate_system="model"):
        """Return input-driver fields in the requested coordinates."""
        return self.input_plot_data_at_time(
            self.timestamp_at_index(index), coordinate_system=coordinate_system
        )

    def input_plot_data_at_time(self, timestamp, *, coordinate_system="model"):
        """Return time-selected inputs in the requested coordinates."""
        coordinate_system = _normalize_display_coordinate_system(coordinate_system)
        evaluation = (
            self._get_geographic_evaluation(timestamp)
            if coordinate_system == "geographic"
            else None
        )
        input_transforms = (
            self._geographic_input_transforms(evaluation)
            if evaluation is not None
            else self.input_transforms
        )
        fields = compute_input_fields_at_time(
            timestamp,
            self.results,
            input_transforms,
            self.lat.shape,
            self.wind_lat.shape,
            start_time=self._start_time(),
        )
        if evaluation is None:
            return fields

        main_field = self.results.main_field
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
                east=phi,
                north=-theta,
            )
            fields[theta_key] = -np.asarray(north).reshape(self.wind_lat.shape)
            fields[phi_key] = np.asarray(east).reshape(self.wind_lat.shape)
        return fields


def format_figure_time(timestamp):
    """Return a compact title-friendly timestamp label."""
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        if isinstance(timestamp, (int, float)):
            return str(dt.timedelta(seconds=float(timestamp)))
        return str(timestamp)


def _coerce_figure_settings(settings):
    """Return a :class:`FigureSettings` instance."""
    if isinstance(settings, FigureSettings):
        return settings
    return FigureSettings.from_dict(settings)


def clear_plot_data_cache():
    """Clear cached plotting data."""
    _PLOT_DATA_CACHE.clear()


def _path_fingerprint(path):
    """Return a change fingerprint for one file or directory tree."""
    try:
        path_stat = path.stat()
    except OSError:
        return None
    if not path.is_dir():
        return ("file", path_stat.st_mtime_ns, path_stat.st_size)

    latest_mtime = path_stat.st_mtime_ns
    entry_count = 0
    total_file_size = 0
    for child in path.rglob("*"):
        try:
            child_stat = child.stat()
        except OSError:
            continue
        entry_count += 1
        latest_mtime = max(latest_mtime, child_stat.st_mtime_ns)
        if stat.S_ISREG(child_stat.st_mode):
            total_file_size += child_stat.st_size
    return ("tree", latest_mtime, entry_count, total_file_size)


def _artifact_fingerprint(simulation_directory):
    directory = Path(simulation_directory).expanduser()
    artifacts = ArtifactStore(directory)
    fingerprint = []
    for name in _CACHE_ARTIFACTS:
        path = artifacts.existing_artifact_path(name)
        if path is not None:
            fingerprint.append((name, str(path), _path_fingerprint(path)))
    return tuple(fingerprint)


def get_plot_data(settings):
    """Return cached plotting data for one simulation directory."""
    settings = _coerce_figure_settings(settings)
    simulation_directory = str(Path(settings.simulation_directory).expanduser().resolve())
    key = (simulation_directory, 60, 100)
    fingerprint = _artifact_fingerprint(simulation_directory)
    cached = _PLOT_DATA_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    plot_data = PlotData.from_directory(simulation_directory)
    _PLOT_DATA_CACHE[key] = (fingerprint, plot_data)
    return plot_data


__all__ = [
    "PlotData",
    "clear_plot_data_cache",
    "compute_input_fields_at_time",
    "compute_output_fields_at_index",
    "datetime_at_index",
    "format_figure_time",
    "get_plot_data",
    "time_index_from_dataset",
]
