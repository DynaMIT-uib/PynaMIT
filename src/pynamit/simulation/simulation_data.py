"""Writable data and persistence context for one simulation.

``SimulationData`` owns runtime and restart plumbing: artifact storage,
schema, and loaded input/output field series.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION, SimulationConfig
from pynamit.simulation.schema import (
    INPUT_VARIABLE_ATTRS,
    OUTPUT_VARIABLE_ATTRS,
    SimulationSchema,
    build_simulation_schema,
)
from pynamit.storage import ArtifactStore, FieldTimeSeries


@dataclass
class SimulationData:
    """Own a simulation's configuration, series, and storage."""

    artifact_store: ArtifactStore
    config: SimulationConfig
    schema: SimulationSchema
    input_series: FieldTimeSeries
    output_series: FieldTimeSeries
    settings_saved: bool = False
    gap_Br_response: xr.DataArray | None = None

    def __repr__(self):
        """Summarize persisted data without printing xarray datasets."""
        inputs = ", ".join(sorted(self.input_series.datasets)) or "none"
        outputs = ", ".join(sorted(self.output_series.datasets)) or "none"
        return (
            f"SimulationData(inputs=[{inputs}], outputs=[{outputs}], "
            f"simulation_directory={self.simulation_directory!r})"
        )

    @classmethod
    def open(
        cls,
        settings: Any,
        *,
        simulation_directory=None,
        artifact_storage="auto",
        operator_cache=None,
        print_info=False,
    ) -> "SimulationData":
        """Open or create persisted simulation data."""
        config = (
            settings
            if isinstance(settings, SimulationConfig)
            else SimulationConfig.from_settings(settings)
        )
        settings_dataset = config.to_dataset()

        if simulation_directory is None:
            simulation_directory = ArtifactStore.create_temporary_directory()

        artifact_store = ArtifactStore(
            directory=simulation_directory, preferred_dataset_storage=artifact_storage
        )

        stored_settings = artifact_store.load_dataset("settings", print_info=print_info)
        if stored_settings is not None:
            stored_version = stored_settings.attrs.get("simulation_schema_version")
            if stored_version != SIMULATION_SCHEMA_VERSION:
                raise ValueError(
                    "Simulation directory uses schema "
                    f"{stored_version!r}; expected {SIMULATION_SCHEMA_VERSION}. "
                    "Create a new simulation directory for the physical magnetic-variable schema."
                )
            normalized_stored_settings = SimulationConfig.from_settings(
                stored_settings
            ).to_dataset()
            if not settings_dataset.identical(normalized_stored_settings):
                raise ValueError(
                    "Mismatch between Simulation object arguments and settings on file."
                )

        gap_Br_response = artifact_store.load_dataarray("gap_Br_response", print_info=print_info)
        schema = build_simulation_schema(config, operator_cache=operator_cache)

        input_series = FieldTimeSeries(
            schema.input_field_spaces,
            schema.input_variables,
            variable_attrs=INPUT_VARIABLE_ATTRS,
            time_origin=config.t0,
        )
        input_series.load_all(artifact_store)

        output_series = FieldTimeSeries(
            schema.output_field_spaces,
            schema.output_variables,
            variable_attrs=OUTPUT_VARIABLE_ATTRS,
            time_origin=config.t0,
        )
        output_series.load_all(artifact_store)

        return cls(
            artifact_store=artifact_store,
            config=config,
            schema=schema,
            input_series=input_series,
            output_series=output_series,
            settings_saved=stored_settings is not None,
            gap_Br_response=gap_Br_response,
        )

    @property
    def simulation_directory(self):
        """Return the resolved simulation directory."""
        return self.artifact_store.directory

    def save_settings_if_missing(self, *, print_info=False):
        """Persist settings when this is a new simulation directory."""
        if self.settings_saved:
            return
        self.artifact_store.save_dataset(
            self.config.to_dataset(), "settings", print_info=print_info
        )
        self.settings_saved = True

    def save_gap_Br_response_if_missing(self, matrix, *, print_info=False):
        """Persist the physical gap-field response for restart."""
        if self.gap_Br_response is not None:
            return
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"gap_Br_response must be two-dimensional; got shape {matrix.shape}.")
        dataarray = xr.DataArray(
            matrix,
            dims=("poloidal_i", "surface_i"),
            name="gap_Br_response",
            attrs={
                "input_quantity": "boundary_jr_at_RI",
                "output_quantity": "unshielded_gap_Br_at_RI",
                "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
            },
        )
        self.artifact_store.save_dataarray(dataarray, "gap_Br_response", print_info=print_info)
        self.gap_Br_response = dataarray
