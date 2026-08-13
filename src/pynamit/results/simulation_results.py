"""Read-only access to persisted PynaMIT simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from pynamit.geomagnetism import MainField
from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION, SimulationConfig
from pynamit.simulation.geometry import SimulationGeometry, build_main_field
from pynamit.simulation.schema import (
    INPUT_DATASET_KEYS,
    OUTPUT_DATASET_KEYS,
    SimulationSchema,
    build_simulation_schema,
)
from pynamit.storage import ArrayCache, ArtifactStore, FieldTimeSeries


@dataclass
class SimulationResults:
    """Saved inputs, outputs, and geometry for one PynaMIT simulation.

    Unlike :class:`pynamit.Simulation`, this object is read-only and
    does not construct a response model or evolution runner. Input and
    output datasets are loaded lazily when their properties are used.
    """

    artifact_store: ArtifactStore
    datasets: dict[str, xr.Dataset]
    config: SimulationConfig
    schema: SimulationSchema
    main_field: MainField
    operator_cache: ArrayCache | None = None
    gap_Br_response: xr.DataArray | None = None
    geometry: SimulationGeometry | None = None
    _input_series: FieldTimeSeries | None = field(default=None, init=False, repr=False)
    _output_series: FieldTimeSeries | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_directory(
        cls,
        simulation_directory,
        *,
        required_datasets=(),
        optional_datasets=(),
        require_gap_Br_response=False,
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
        datasets = {"settings": settings}
        stored_version = datasets["settings"].attrs.get("simulation_schema_version")
        if stored_version != SIMULATION_SCHEMA_VERSION:
            raise ValueError(
                "Saved simulation uses schema "
                f"{stored_version!r}; expected {SIMULATION_SCHEMA_VERSION}."
            )
        config = SimulationConfig.from_settings(datasets["settings"])
        operator_cache = (
            None if operator_cache_directory is None else ArrayCache(operator_cache_directory)
        )
        results = cls(
            artifact_store=artifact_store,
            datasets=datasets,
            config=config,
            schema=build_simulation_schema(config, operator_cache=operator_cache),
            main_field=build_main_field(config),
            operator_cache=operator_cache,
        )

        for key in required_datasets:
            results._load_requested_dataset(key, required=True, print_info=print_info)
        for key in optional_datasets:
            results._load_requested_dataset(key, required=False, print_info=print_info)

        if require_gap_Br_response or build_geometry:
            results.gap_Br_response = artifact_store.load_dataarray(
                "gap_Br_response", print_info=print_info
            )
            if require_gap_Br_response and results.gap_Br_response is None:
                raise ValueError(
                    f"No saved 'gap_Br_response' data array exists at {simulation_directory!r}"
                )
        if build_geometry:
            results.load_geometry()
        return results

    def _load_requested_dataset(self, key, *, required, print_info):
        """Load one requested artifact through canonical storage."""
        if key == "settings":
            return self.datasets["settings"]
        if key in INPUT_DATASET_KEYS:
            self.load_input_series()
        elif key in OUTPUT_DATASET_KEYS:
            self.load_output_series()
        else:
            dataset = self.artifact_store.load_dataset(key, print_info=print_info)
            if dataset is not None:
                self.datasets[key] = dataset
        dataset = self.datasets.get(key)
        if required and dataset is None:
            raise ValueError(
                f"No saved {key!r} dataset exists at {self.artifact_store.directory!r}"
            )
        return dataset

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

    def load_geometry(self) -> SimulationGeometry:
        """Load and return the geometry for this saved simulation."""
        if self.geometry is None:
            if self.gap_Br_response is None:
                self.gap_Br_response = self.artifact_store.load_dataarray("gap_Br_response")
            self.geometry = SimulationGeometry(
                horizontal_basis=self.schema.horizontal_basis,
                cs_basis=self.schema.cs_basis,
                main_field=self.main_field,
                config=self.config,
                gap_Br_response_matrix=self.gap_Br_response,
                solid_harmonics=self.schema.solid_harmonics,
                operator_cache=self.operator_cache,
            )
        return self.geometry

    def load_input_series(self) -> FieldTimeSeries:
        """Load all persisted input time series for this simulation."""
        if self._input_series is None:
            self._input_series = FieldTimeSeries(
                self.schema.input_field_spaces, self.schema.input_variables
            )
            self._input_series.load_all(self.artifact_store)
            self.datasets.update(self._input_series.datasets)
        return self._input_series

    def load_output_series(self) -> FieldTimeSeries:
        """Load all persisted output time series for this simulation."""
        if self._output_series is None:
            self._output_series = FieldTimeSeries(
                self.schema.output_field_spaces, self.schema.output_variables
            )
            self._output_series.load_all(self.artifact_store)
            self.datasets.update(self._output_series.datasets)
        return self._output_series

    def data_var_name(self, dataset_key, variable_name):
        """Return a physical variable's schema-defined xarray name."""
        if dataset_key in INPUT_DATASET_KEYS:
            series = self.load_input_series()
        elif dataset_key in OUTPUT_DATASET_KEYS:
            series = self.load_output_series()
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
