"""Read-only access to persisted PynaMIT simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from kompe.cache import PersistentArrayCache

from pynamit.geomagnetism import MainField
from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION, SimulationConfig
from pynamit.simulation.geometry import SimulationGeometry, build_main_field
from pynamit.simulation.schema import (
    INPUT_DATASET_KEYS,
    INPUT_VARIABLE_ATTRS,
    OUTPUT_DATASET_KEYS,
    OUTPUT_VARIABLE_ATTRS,
    SimulationSchema,
    build_simulation_schema,
)
from pynamit.storage import ArtifactStore, FieldTimeSeries


@dataclass
class SimulationResults:
    """Saved inputs, outputs, and geometry for one PynaMIT simulation.

    Unlike :class:`pynamit.Simulation`, this object is read-only and
    does not construct a response model or time-evolution state.
    Input and output datasets are loaded lazily. This object does not
    write simulation artifacts; returned xarray values remain ordinary
    inspectable, editable Python objects.
    """

    artifact_store: ArtifactStore
    settings: xr.Dataset
    config: SimulationConfig
    schema: SimulationSchema
    main_field: MainField
    operator_cache: PersistentArrayCache | None = None
    boundary_jr_to_gap_Br_matrix: xr.DataArray | None = None
    _geometry: SimulationGeometry | None = field(default=None, init=False, repr=False)
    _input_series: FieldTimeSeries | None = field(default=None, init=False, repr=False)
    _output_series: FieldTimeSeries | None = field(default=None, init=False, repr=False)
    _other_datasets: dict[str, xr.Dataset] = field(default_factory=dict, init=False, repr=False)
    _loaded_streams: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_directory(
        cls,
        simulation_directory,
        *,
        required_datasets=(),
        optional_datasets=(),
        require_boundary_jr_to_gap_Br_matrix=False,
        build_geometry=False,
        artifact_storage="auto",
        operator_cache_directory=None,
        print_info=False,
    ) -> SimulationResults:
        """Open saved results without constructing a live simulation."""
        artifact_store = ArtifactStore(
            simulation_directory, preferred_dataset_storage=artifact_storage
        )
        settings = artifact_store.load_dataset("settings", print_info=print_info)
        if settings is None:
            raise ValueError(f"No saved 'settings' dataset exists at {artifact_store.directory!r}")
        stored_version = settings.attrs.get("simulation_schema_version")
        if stored_version != SIMULATION_SCHEMA_VERSION:
            raise ValueError(
                "Saved simulation uses schema "
                f"{stored_version!r}; expected {SIMULATION_SCHEMA_VERSION}."
            )
        config = SimulationConfig.from_settings(settings)
        operator_cache = (
            None
            if operator_cache_directory is None
            else PersistentArrayCache(operator_cache_directory)
        )
        results = cls(
            artifact_store=artifact_store,
            settings=settings,
            config=config,
            schema=build_simulation_schema(config, operator_cache=operator_cache),
            main_field=build_main_field(config),
            operator_cache=operator_cache,
        )

        for key in required_datasets:
            results._load_requested_dataset(key, required=True, print_info=print_info)
        for key in optional_datasets:
            results._load_requested_dataset(key, required=False, print_info=print_info)

        if require_boundary_jr_to_gap_Br_matrix or build_geometry:
            results.boundary_jr_to_gap_Br_matrix = artifact_store.load_dataarray(
                "gap_Br_response", print_info=print_info
            )
            if (
                require_boundary_jr_to_gap_Br_matrix
                and results.boundary_jr_to_gap_Br_matrix is None
            ):
                raise ValueError(
                    f"No saved 'gap_Br_response' data array exists at {simulation_directory!r}"
                )
        if build_geometry:
            _ = results.geometry
        return results

    def _load_requested_dataset(self, key, *, required, print_info):
        """Load one requested artifact through canonical storage."""
        if key == "settings":
            return self.settings
        if key in INPUT_DATASET_KEYS:
            self.load_input_series(key)
        elif key in OUTPUT_DATASET_KEYS:
            self.load_output_series(key)
        else:
            dataset = self.artifact_store.load_dataset(key, print_info=print_info)
            if dataset is not None:
                self._other_datasets[key] = dataset
        dataset = self.datasets.get(key)
        if required and dataset is None:
            raise ValueError(
                f"No saved {key!r} dataset exists at {self.artifact_store.directory!r}"
            )
        return dataset

    @property
    def datasets(self) -> dict[str, xr.Dataset]:
        """Return a catalog snapshot of already loaded datasets.

        The time-series objects own their datasets; this catalog never
        loads files or maintains a second set of dataset references.
        """
        datasets = {"settings": self.settings, **self._other_datasets}
        for series in (self._input_series, self._output_series):
            if series is not None:
                datasets.update(series.datasets)
        return datasets

    @property
    def simulation_directory(self) -> str:
        """Return the directory containing this simulation."""
        return self.artifact_store.directory

    @property
    def inputs(self) -> dict[str, xr.Dataset]:
        """Return all persisted input datasets, loading them once."""
        return self.load_input_series().datasets

    @property
    def outputs(self) -> dict[str, xr.Dataset]:
        """Return all persisted output datasets, loading them once."""
        return self.load_output_series().datasets

    @property
    def times(self) -> np.ndarray:
        """Return the sorted saved input and output times."""
        arrays = [
            np.asarray(dataset.time.values)
            for dataset in (*self.inputs.values(), *self.outputs.values())
            if "time" in dataset.coords
        ]
        return np.unique(np.concatenate(arrays)) if arrays else np.array([])

    @property
    def geometry(self) -> SimulationGeometry:
        """Return the lazily constructed saved-simulation geometry."""
        if self._geometry is None:
            if self.boundary_jr_to_gap_Br_matrix is None:
                self.boundary_jr_to_gap_Br_matrix = self.artifact_store.load_dataarray(
                    "gap_Br_response"
                )
            self._geometry = SimulationGeometry(
                horizontal_basis=self.schema.horizontal_basis,
                cs_basis=self.schema.cs_basis,
                main_field=self.main_field,
                config=self.config,
                boundary_jr_to_gap_Br_matrix=self.boundary_jr_to_gap_Br_matrix,
                solid_harmonics=self.schema.solid_harmonics,
                operator_cache=self.operator_cache,
            )
        return self._geometry

    def load_input_series(self, *keys) -> FieldTimeSeries:
        """Load named input streams, or all inputs by default."""
        if self._input_series is None:
            self._input_series = FieldTimeSeries(
                self.schema.input_field_spaces,
                self.schema.input_variables,
                variable_attrs=INPUT_VARIABLE_ATTRS,
                time_origin=self.config.t0,
            )
        self._load_streams(self._input_series, keys or INPUT_DATASET_KEYS)
        return self._input_series

    def load_output_series(self, *keys) -> FieldTimeSeries:
        """Load named output streams, or all outputs by default."""
        if self._output_series is None:
            self._output_series = FieldTimeSeries(
                self.schema.output_field_spaces,
                self.schema.output_variables,
                variable_attrs=OUTPUT_VARIABLE_ATTRS,
                time_origin=self.config.t0,
            )
        self._load_streams(self._output_series, keys or OUTPUT_DATASET_KEYS)
        return self._output_series

    def _load_streams(self, series, keys):
        """Read each requested stream at most once, even when absent."""
        for key in keys:
            if key not in series.variables:
                raise KeyError(
                    f"Unknown stream {key!r}; expected one of {tuple(series.variables)}."
                )
            if key not in self._loaded_streams:
                series.load(key, self.artifact_store)
                self._loaded_streams.add(key)

    def data_var_name(self, dataset_key, variable_name):
        """Return a physical variable's schema-defined xarray name."""
        if dataset_key in INPUT_DATASET_KEYS:
            series = self.load_input_series(dataset_key)
        elif dataset_key in OUTPUT_DATASET_KEYS:
            series = self.load_output_series(dataset_key)
        else:
            raise KeyError(f"{dataset_key!r} is not a coefficient time series.")
        if variable_name not in series.variables[dataset_key]:
            raise KeyError(
                f"{dataset_key!r} has no schema variable {variable_name!r}; "
                f"expected one of {series.variables[dataset_key]}."
            )
        return series.get_data_var_name(dataset_key, variable_name)

    def __repr__(self):
        """Summarize the simulation without triggering lazy loads."""
        loaded = ", ".join(key for key in self.datasets if key != "settings") or "none"
        return (
            f"SimulationResults(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, loaded=[{loaded}], "
            f"simulation_directory={self.simulation_directory!r})"
        )


__all__ = ["SimulationResults"]
