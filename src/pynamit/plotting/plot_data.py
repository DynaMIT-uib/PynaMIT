"""Saved coefficient fields evaluated on plotting grids."""

from __future__ import annotations

import datetime as dt
import stat
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from kompe import SphericalGrid, SphericalTransform
from kompe.cache import BoundedCache

from pynamit.coordinates import GEOCENTRIC_GEOGRAPHIC
from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.map_coordinates import (
    MapCoordinateContext,
    model_grid_from_geographic,
    regular_geographic_grid,
)
from pynamit.results.input_fields import evaluate_projected_input
from pynamit.results.output_fields import (
    build_ground_magnetic_field_operators,
    build_output_evaluation_operators,
    build_sheet_current_operators,
    evaluate_output_coefficients,
)
from pynamit.results.simulation_results import SimulationResults
from pynamit.results.time_series import datetime_at_index, time_index_from_dataset
from pynamit.simulation.electrodynamics.ionospheric_closure import pedersen_geometry_tensor
from pynamit.simulation.schema import SIMULATION_ARTIFACT_NAMES
from pynamit.storage import ArtifactStore

INPUT_ARTIFACT_KEYS = ("boundary_Br", "boundary_jr", "conductance", "u", "Q_eff", "E_neutral_wind")
TANGENTIAL_INPUT_KEYS = ("u", "Q_eff", "E_neutral_wind")
OUTPUT_FIELD_NAMES = frozenset({"Br", "jr", "Jeq", "Phi", "W", "joule"})
_DISPLAY_OUTPUT_TO_PHYSICAL = {
    "Br": "induced_Br",
    "jr": "boundary_jr",
    "Jeq": "equivalent_current_function",
    "Phi": "Phi",
    "W": "W",
    "joule": "joule_heating",
}
_DISPLAY_COORDINATE_SYSTEMS = frozenset({"model", "geographic"})
_CACHE_ARTIFACTS = tuple(sorted(SIMULATION_ARTIFACT_NAMES))
_PLOT_DATA_CACHE = BoundedCache(4)


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
    sheet_current_operators: dict[str, object] | None = None


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


def evaluate_output_fields_at_index(
    index,
    results,
    transform,
    conductance_transform,
    output_evaluation_context,
    sheet_current_operators,
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
            reference_dataset.time.values, index, start_time=start_time
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
        simulation_time = (pd.Timestamp(target_time) - pd.Timestamp(start_time)).total_seconds()
        conductance = evaluate_projected_input(
            results, "conductance", simulation_time, transform=conductance_transform
        )
        etaP = conductance["etaP"]

    physical_fields = {_DISPLAY_OUTPUT_TO_PHYSICAL[name] for name in field_names}
    if etaP is None:
        physical_fields.discard("joule_heating")
    required_coefficients = set(physical_fields & {"Phi", "W"})
    if physical_fields & {"induced_Br", "equivalent_current_function", "joule_heating"}:
        required_coefficients.add("induced_Br")
    if physical_fields & {"boundary_jr", "joule_heating"}:
        required_coefficients.add("boundary_jr")

    result = {}
    for dataset_key in output_keys:
        dataset = datasets[dataset_key]
        output_index = (
            index
            if dataset_key == reference_key
            else _dataset_index_at_time(dataset, target_time, start_time=start_time)
        )
        coefficients = {
            name: np.asarray(
                dataset[results.data_var_name(dataset_key, name)].isel(time=output_index).values
            )
            for name in required_coefficients
        }
        evaluated = evaluate_output_coefficients(
            coefficients,
            transform,
            field_names=physical_fields,
            operators=output_evaluation_context,
            sheet_current_operators=sheet_current_operators,
            boundary_Br=boundary_Br,
            etaP=etaP,
            pedersen_geometry=output_evaluation_context.get("pedersen_geometry"),
        )
        fields = {
            name: np.asarray(evaluated[physical_name]).reshape(-1)
            for name, physical_name in _DISPLAY_OUTPUT_TO_PHYSICAL.items()
            if name in field_names and physical_name in evaluated
        }
        for name in field_names & {"Phi", "W"}:
            fields[name] = fields[name] * 1e-3
        if "joule" in field_names and etaP is None:
            fields["joule"] = np.full(transform.grid.size, np.nan)
        result[dataset_key] = fields
    return result


def evaluate_input_fields_at_time(
    timestamp, results, input_transforms, scalar_shape, vector_shape, *, start_time=None
):
    """Evaluate projected input drivers at one physical time."""
    datasets = results.datasets
    simulation_time = (pd.Timestamp(timestamp) - pd.Timestamp(start_time)).total_seconds()

    def evaluate(key):
        if key not in datasets:
            return {}
        times = np.asarray(datasets[key].time.values, dtype=float)
        if not np.any(times <= simulation_time):
            return {}
        return evaluate_projected_input(
            results,
            key,
            simulation_time,
            transform=input_transforms[key],
            include_derived=key == "conductance",
        )

    boundary_jr = evaluate("boundary_jr")
    boundary_Br = evaluate("boundary_Br")
    conductance = evaluate("conductance")
    tangential = {key: evaluate(key) for key in TANGENTIAL_INPUT_KEYS}

    def field(values, name, shape):
        if name not in values:
            return np.full(shape, np.nan, dtype=float)
        return np.asarray(values[name]).reshape(shape)

    return {
        "jr": field(boundary_jr, "boundary_jr", scalar_shape),
        "Br": field(boundary_Br, "boundary_Br", scalar_shape),
        "SigmaP": field(conductance, "SigmaP", scalar_shape),
        "SigmaH": field(conductance, "SigmaH", scalar_shape),
        "wind_theta": field(tangential["u"], "u_theta", vector_shape),
        "wind_phi": field(tangential["u"], "u_phi", vector_shape),
        "Q_eff_theta": field(tangential["Q_eff"], "Q_eff_theta", vector_shape),
        "Q_eff_phi": field(tangential["Q_eff"], "Q_eff_phi", vector_shape),
        "E_neutral_wind_theta": field(
            tangential["E_neutral_wind"], "E_neutral_wind_theta", vector_shape
        ),
        "E_neutral_wind_phi": field(
            tangential["E_neutral_wind"], "E_neutral_wind_phi", vector_shape
        ),
    }


@dataclass
class PlotData:
    """Evaluate saved coefficients for figures and requested sites."""

    results: SimulationResults
    lat: np.ndarray
    lon: np.ndarray
    wind_lat: np.ndarray
    wind_lon: np.ndarray
    output_transform: SphericalTransform | None
    input_transforms: dict[str, SphericalTransform | None]
    output_evaluation_context: dict[str, object] | None
    sheet_current_operators: dict[str, object] | None
    _geographic_evaluation: _GeographicEvaluation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _ground_magnetic_field_cache: BoundedCache = field(
        default_factory=lambda: BoundedCache(16), init=False, repr=False, compare=False
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
        lat, lon, grid = regular_geographic_grid(nlat=nlat, nlon=nlon)
        wind_lat, wind_lon, wind_grid = regular_geographic_grid(
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
            sheet_current_operators=None,
        )

    @property
    def geometry(self):
        """Return the saved simulation geometry."""
        if not self.has_model_output:
            raise ValueError("Saved simulation geometry requires dynamic or equilibrium output.")
        return self.results.geometry

    def __repr__(self):
        """Summarize evaluated fields without printing their arrays."""
        inputs = ", ".join(self.available_inputs) or "none"
        return (
            f"PlotData(n_time={self.n_time}, inputs=[{inputs}], "
            f"has_model_output={self.has_model_output}, "
            f"simulation_directory={self.results.simulation_directory!r})"
        )

    def _get_geographic_evaluation(self):
        """Return evaluators sampled on the geographic display grid.

        Saved coefficients live in the simulation's horizontal
        coordinate system. Its orientation is fixed by the persisted
        main-field epoch, so this sampling geometry is immutable.
        """
        if self._geographic_evaluation is not None:
            return self._geographic_evaluation

        main_field = self.results.main_field
        scalar_grid = model_grid_from_geographic(main_field, self.lat, self.lon)
        vector_grid = model_grid_from_geographic(main_field, self.wind_lat, self.wind_lon)

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
        return time_index_from_dataset(self._time_dataset(), start_time=self.results.config.t0)

    def timestamp_at_index(self, index):
        """Return one saved time as a timestamp."""
        return datetime_at_index(
            self._time_dataset().time.values, index, start_time=self.results.config.t0
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

    def ground_magnetic_fields(self, site_lat, site_lon):
        """Return induced ground magnetic fields at geographic sites.

        ``site_lat`` and ``site_lon`` are geocentric geographic degrees.
        The returned mapping separates dynamic and, when available,
        equilibrium fields into ``radial`` and ``tangential`` arrays in
        tesla. Radial arrays have shape ``(site, time)``. Tangential
        arrays have shape ``(component, site, time)``. Their component
        order is ``(theta, phi)``, equivalent to ``(south, east)``.
        """
        if "dynamic" not in self.results.datasets:
            raise ValueError("Ground magnetic fields require saved dynamic output.")
        lat = np.asarray(site_lat, dtype=float).reshape(-1)
        lon = np.asarray(site_lon, dtype=float).reshape(-1)
        if lat.size != lon.size:
            raise ValueError("site_lat and site_lon must have the same length.")

        key = (tuple(np.round(lat, 8).tolist()), tuple(np.round(lon, 8).tolist()))
        cached = self._ground_magnetic_field_cache.get(key)
        if cached is not None:
            return cached

        grid = SphericalGrid(lat=lat, lon=lon)
        operators = build_ground_magnetic_field_operators(self.geometry, grid)

        def evaluate(coefficients):
            radial = np.asarray(operators["radial"].matmat(coefficients)).reshape(grid.size, -1)
            tangential = np.asarray(operators["tangential"].matmat(coefficients)).reshape(
                2, grid.size, -1
            )
            radial.setflags(write=False)
            tangential.setflags(write=False)
            return {"radial": radial, "tangential": tangential}

        induced_Br = self.dataset_values("dynamic", "induced_Br").T
        fields = {"dynamic": evaluate(induced_Br)}
        if "equilibrium" in self.results.datasets:
            equilibrium_induced_Br = self.dataset_values("equilibrium", "induced_Br").T
            fields["equilibrium"] = evaluate(equilibrium_induced_Br)

        self._ground_magnetic_field_cache.store(key, fields)
        return fields

    def output_fields(self, index, *, field_names=None, coordinate_system="model"):
        """Return flat output fields in the requested coordinates."""
        field_names = _normalize_output_field_names(field_names)
        coordinate_system = _normalize_display_coordinate_system(coordinate_system)
        timestamp = self.timestamp_at_index(index)
        if not self.has_model_output:
            raise ValueError(
                "This directory contains projected inputs but no saved model output. "
                "Choose 'Input drivers' or run a simulation first."
            )
        if coordinate_system == "geographic":
            evaluation = self._get_geographic_evaluation()
            transform = self._geographic_output_transform(evaluation)
            output_evaluation_context = evaluation.output_evaluation_context
            sheet_current_operators = evaluation.sheet_current_operators
        else:
            evaluation = None
            transform = self.output_transform
            output_evaluation_context = self.output_evaluation_context
            sheet_current_operators = self.sheet_current_operators

        if transform is None:
            raise RuntimeError("Saved output evaluation context is unavailable.")
        geometry = self.geometry
        if output_evaluation_context is None:
            output_evaluation_context = build_output_evaluation_operators(geometry, transform)
        needs_joule = "joule" in field_names
        if needs_joule and sheet_current_operators is None:
            sheet_current_operators = build_sheet_current_operators(geometry, transform)
        if needs_joule and "pedersen_geometry" not in output_evaluation_context:
            unit_br, unit_btheta, unit_bphi = geometry.main_field.unit_vector(
                transform.grid, self.results.config.RI
            )
            output_evaluation_context["pedersen_geometry"] = pedersen_geometry_tensor(
                unit_btheta, unit_bphi, unit_br
            )
        if evaluation is None:
            self.output_evaluation_context = output_evaluation_context
            self.sheet_current_operators = sheet_current_operators
        else:
            evaluation.output_evaluation_context = output_evaluation_context
            evaluation.sheet_current_operators = sheet_current_operators
        conductance_transform = self.input_transforms.get("conductance") if needs_joule else None
        if evaluation is not None:
            conductance_transform = (
                self._geographic_input_transforms(evaluation, keys=("conductance",))["conductance"]
                if needs_joule
                else None
            )
        return evaluate_output_fields_at_index(
            index,
            self.results,
            transform,
            conductance_transform,
            output_evaluation_context,
            sheet_current_operators,
            target_time=timestamp,
            start_time=self.results.config.t0,
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
            self._get_geographic_evaluation() if coordinate_system == "geographic" else None
        )
        input_transforms = (
            self._geographic_input_transforms(evaluation)
            if evaluation is not None
            else self.input_transforms
        )
        fields = evaluate_input_fields_at_time(
            timestamp,
            self.results,
            input_transforms,
            self.lat.shape,
            self.wind_lat.shape,
            start_time=self.results.config.t0,
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
    _PLOT_DATA_CACHE.store(key, (fingerprint, plot_data))
    return plot_data


__all__ = [
    "PlotData",
    "clear_plot_data_cache",
    "evaluate_input_fields_at_time",
    "evaluate_output_fields_at_index",
    "format_figure_time",
    "get_plot_data",
]
