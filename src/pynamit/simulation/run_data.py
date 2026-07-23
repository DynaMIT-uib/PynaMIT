"""Writable data and persistence context for one simulation run.

``RunData`` owns runtime and restart plumbing: artifact storage,
schema, and loaded input/output field series.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.schema import SimulationSchema, build_simulation_schema
from pynamit.storage import ArtifactStore, FieldTimeSeries


@dataclass
class RunData:
    """Own one run's configuration, schema, time series, and storage."""

    artifact_store: ArtifactStore
    config: SimulationConfig
    schema: SimulationSchema
    input_series: FieldTimeSeries
    output_series: FieldTimeSeries
    settings_saved: bool = False
    pfac_matrix: xr.DataArray | None = None

    @classmethod
    def open(
        cls,
        settings: Any,
        *,
        run_directory=None,
        artifact_storage="auto",
        operator_cache=None,
        print_info=False,
    ) -> "RunData":
        """Open or create a persisted run context."""
        config = (
            settings
            if isinstance(settings, SimulationConfig)
            else SimulationConfig.from_settings(settings)
        )
        settings_dataset = config.to_dataset()

        if run_directory is None:
            run_directory = ArtifactStore.create_temporary_directory()

        artifact_store = ArtifactStore(
            directory=run_directory, preferred_dataset_storage=artifact_storage
        )

        stored_settings = artifact_store.load_dataset("settings", print_info=print_info)
        if stored_settings is not None:
            normalized_stored_settings = SimulationConfig.from_settings(
                stored_settings
            ).to_dataset()
            if not settings_dataset.identical(normalized_stored_settings):
                raise ValueError(
                    "Mismatch between Simulation object arguments and settings on file."
                )

        pfac_matrix = artifact_store.load_dataarray("PFAC_matrix", print_info=print_info)
        schema = build_simulation_schema(config, operator_cache=operator_cache)

        input_series = FieldTimeSeries(schema.input_field_spaces, schema.input_variables)
        input_series.load_all(artifact_store)

        output_series = FieldTimeSeries(schema.output_field_spaces, schema.output_variables)
        output_series.load_all(artifact_store)

        return cls(
            artifact_store=artifact_store,
            config=config,
            schema=schema,
            input_series=input_series,
            output_series=output_series,
            settings_saved=stored_settings is not None,
            pfac_matrix=pfac_matrix,
        )

    @property
    def run_directory(self):
        """Return the resolved run directory."""
        return self.artifact_store.directory

    def save_settings_if_missing(self, *, print_info=False):
        """Persist settings when this is a new run directory."""
        if self.settings_saved:
            return
        self.artifact_store.save_dataset(
            self.config.to_dataset(), "settings", print_info=print_info
        )
        self.settings_saved = True

    def save_pfac_matrix_if_missing(self, pfac_matrix, *, print_info=False):
        """Persist the PFAC sidecar for a new run directory."""
        if self.pfac_matrix is not None:
            return
        matrix = np.asarray(pfac_matrix)
        if matrix.ndim != 2:
            raise ValueError(f"pfac_matrix must be two-dimensional; got shape {matrix.shape}.")
        dataarray = xr.DataArray(matrix, dims=("poloidal_i", "surface_i"), name="PFAC_matrix")
        self.artifact_store.save_dataarray(dataarray, "PFAC_matrix", print_info=print_info)
        self.pfac_matrix = dataarray

    def save_input_dataset(self, key, *, print_info=False):
        """Persist one loaded input time series."""
        self.input_series.save(key, self.artifact_store, print_info=print_info)

    def add_output_entry(self, key, data, *, time):
        """Append one output entry to a loaded output time series."""
        self.output_series.add_entry(key, data, time=time)

    def save_output_dataset(self, key, *, print_info=False):
        """Persist one loaded output time series."""
        self.output_series.save(key, self.artifact_store, print_info=print_info)
