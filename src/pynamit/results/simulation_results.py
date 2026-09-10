"""Live and saved histories with reusable numerical geometry."""

from __future__ import annotations

import numpy as np
import xarray as xr
from kompe.cache import PersistentArrayCache

from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION
from pynamit.simulation.geometry import SimulationGeometry
from pynamit.simulation.input_preparation import InputPreparation
from pynamit.simulation.schema import INPUT_DATASET_KEYS, OUTPUT_DATASET_KEYS
from pynamit.storage import FieldTimeSeries


class SimulationResults:
    """Inputs, outputs, and geometry for one PynaMIT simulation.

    ``simulation.results`` and ``from_directory(...)`` have the same
    evaluation interface. ``inputs`` is the input preparation; this
    object adds output histories and restart artifacts, not a response
    model or time-evolution state. Saved streams are loaded
    lazily; returned xarray datasets are editable. ``save`` explicitly
    persists the current histories and enables subsequent live writes.
    """

    def __init__(self, inputs: InputPreparation):
        self.inputs = inputs
        self.config = inputs.config
        self._geometry = inputs.geometry
        self.schema = inputs.schema
        self.artifact_store = inputs.artifact_store
        self.artifact_storage = inputs.artifact_storage
        self._output_series = self.schema.create_output_series(time_origin=self.config.t0)
        self._other_datasets = {}
        self._loaded_streams = set()
        self._gap_response_loaded = self.artifact_store is None
        self._gap_response = None

    @classmethod
    def open(
        cls,
        settings=None,
        *,
        simulation_directory=None,
        geometry=None,
        artifact_storage="auto",
        operator_cache=None,
        print_info=False,
    ):
        """Open editable histories using explicit or saved settings."""
        inputs = InputPreparation.__new__(InputPreparation)
        inputs._open_input_preparation(
            settings,
            directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache=operator_cache,
            geometry=geometry,
            print_info=print_info,
        )
        return cls(inputs)

    @classmethod
    def from_directory(
        cls,
        simulation_directory,
        *,
        required_datasets=(),
        optional_datasets=(),
        require_boundary_jr_to_gap_Br_matrix=False,
        artifact_storage="auto",
        operator_cache_directory=None,
        print_info=False,
    ) -> SimulationResults:
        """Open saved results without constructing a live simulation."""
        operator_cache = (
            None
            if operator_cache_directory is None
            else PersistentArrayCache(operator_cache_directory)
        )
        results = cls.open(
            simulation_directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache=operator_cache,
            print_info=print_info,
        )

        for key in required_datasets:
            results._load_requested_dataset(key, required=True, print_info=print_info)
        for key in optional_datasets:
            results._load_requested_dataset(key, required=False, print_info=print_info)

        if require_boundary_jr_to_gap_Br_matrix and results.boundary_jr_to_gap_Br_matrix is None:
            raise ValueError(
                f"No saved 'gap_Br_response' data array exists at {simulation_directory!r}"
            )
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
    def settings(self):
        """Return the canonical persisted simulation description."""
        return self.config.to_dataset()

    @property
    def main_field(self):
        """Return the geometry's background magnetic field."""
        return self._geometry.main_field

    @property
    def model_grid(self):
        """Return the integration and input-projection grid."""
        return self._geometry.model_grid

    @property
    def operator_cache(self):
        """Return the geometry's optional persistent array cache."""
        return self._geometry.operator_cache

    @property
    def datasets(self) -> dict[str, xr.Dataset]:
        """Return a catalog snapshot of already loaded datasets.

        The time-series objects own their datasets; this catalog never
        loads files or maintains a second set of dataset references.
        """
        return {
            "settings": self.settings,
            **self._other_datasets,
            **self.inputs._input_series.datasets,
            **self._output_series.datasets,
        }

    @property
    def simulation_directory(self):
        """Return the directory containing this simulation."""
        return None if self.artifact_store is None else self.artifact_store.directory

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
        """Return geometry with lazily built physical operators."""
        self._load_gap_response()
        return self._geometry

    @property
    def input_series(self):
        """Return input streams, loading saved datasets once."""
        return self.inputs.input_series

    @property
    def output_series(self):
        """Return output streams, loading saved datasets once."""
        return self.load_output_series()

    def load_input_series(self, *keys) -> FieldTimeSeries:
        """Load named input streams, or all inputs by default."""
        return self.inputs.load_input_series(*keys)

    def load_output_series(self, *keys) -> FieldTimeSeries:
        """Load named output streams, or all outputs by default."""
        series = self._output_series
        for key in keys or OUTPUT_DATASET_KEYS:
            if key not in series.variables:
                raise KeyError(
                    f"Unknown stream {key!r}; expected one of {tuple(series.variables)}."
                )
            if key not in self._loaded_streams:
                if self.artifact_store is not None and key not in series.datasets:
                    series.load(key, self.artifact_store)
                self._loaded_streams.add(key)
        return series

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

    def _load_gap_response(self):
        if not self._gap_response_loaded:
            self._gap_response = self.artifact_store.load_dataarray("gap_Br_response")
            if self._gap_response is not None:
                self._geometry._init_boundary_jr_to_gap_Br_matrix(self._gap_response.values)
            self._gap_response_loaded = True

    @property
    def boundary_jr_to_gap_Br_matrix(self):
        """Return the stored gap-field response, loading it once."""
        self._load_gap_response()
        return self._gap_response

    def save(self, directory=None, *, artifact_storage=None):
        """Save histories and bind this directory for later writes."""
        original_directory = self.simulation_directory
        # Read unloaded streams before rebinding the original store.
        outputs = self.output_series
        self._load_gap_response()
        self.inputs._save_input_package(
            self.simulation_directory if directory is None else directory,
            self.artifact_storage if artifact_storage is None else artifact_storage,
            existing_directory=original_directory,
        )
        self.artifact_store = self.inputs.artifact_store
        self.artifact_storage = self.inputs.artifact_storage
        for key in outputs.datasets:
            outputs.save(key, self.artifact_store, incremental=False)
        if self._gap_response is not None:
            self.artifact_store.save_dataarray(self._gap_response, "gap_Br_response")
        if original_directory is not None and original_directory != self.simulation_directory:
            # Rebind lazy Zarr arrays to the new store too.
            for key in outputs.datasets:
                outputs.load(key, self.artifact_store)
            if self._gap_response is not None:
                self._gap_response = self.artifact_store.load_dataarray("gap_Br_response")

    def save_boundary_jr_to_gap_Br_matrix_if_missing(self, matrix, *, print_info=False):
        """Persist the physical gap-field response for restart."""
        if self.boundary_jr_to_gap_Br_matrix is not None:
            return
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"gap_Br_response must be two-dimensional; got shape {matrix.shape}.")
        self._gap_response = xr.DataArray(
            matrix,
            dims=("poloidal_i", "surface_i"),
            name="gap_Br_response",
            attrs={
                "input_quantity": "boundary_jr_at_RI",
                "output_quantity": "unshielded_gap_Br_at_RI",
                "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
            },
        )
        if self.artifact_store is not None:
            self.artifact_store.save_dataarray(
                self._gap_response, "gap_Br_response", print_info=print_info
            )


__all__ = ["SimulationResults"]
