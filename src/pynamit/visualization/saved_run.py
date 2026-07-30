"""Saved-run view shared by visualization frontends."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from pynamit.geomagnetism import MainField
from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION, SimulationConfig
from pynamit.simulation.geometry import SimulationGeometry, build_main_field
from pynamit.simulation.schema import SimulationSchema, build_simulation_schema
from pynamit.storage import ArrayCache, ArtifactStore, FieldTimeSeries


@dataclass
class SavedRunView:
    """Loaded simulation artifacts and derived visualization objects."""

    artifact_store: ArtifactStore
    datasets: dict[str, xr.Dataset]
    config: SimulationConfig
    schema: SimulationSchema
    main_field: MainField
    operator_cache: ArrayCache | None = None
    gap_Br_response: xr.DataArray | None = None
    geometry: SimulationGeometry | None = None

    @classmethod
    def from_directory(
        cls,
        run_directory,
        *,
        required_datasets=(),
        optional_datasets=(),
        require_gap_Br_response=False,
        build_geometry=False,
        artifact_storage="auto",
        operator_cache_directory=None,
        print_info=False,
    ) -> SavedRunView:
        """Load one saved run for visualization."""
        artifact_store = ArtifactStore(run_directory, preferred_dataset_storage=artifact_storage)
        datasets = {"settings": cls._load_required_dataset(artifact_store, "settings", print_info)}
        for key in required_datasets:
            if key == "settings":
                continue
            datasets[key] = cls._load_required_dataset(artifact_store, key, print_info)
        for key in optional_datasets:
            if key in datasets:
                continue
            dataset = artifact_store.load_dataset(key, print_info=print_info)
            if dataset is not None:
                datasets[key] = dataset

        stored_version = datasets["settings"].attrs.get("simulation_schema_version")
        if stored_version != SIMULATION_SCHEMA_VERSION:
            raise ValueError(
                "Saved run uses simulation schema "
                f"{stored_version!r}; expected {SIMULATION_SCHEMA_VERSION}."
            )
        config = SimulationConfig.from_settings(datasets["settings"])
        operator_cache = (
            None if operator_cache_directory is None else ArrayCache(operator_cache_directory)
        )
        schema = build_simulation_schema(config, operator_cache=operator_cache)
        main_field = build_main_field(config)

        gap_Br_response = None
        if require_gap_Br_response or build_geometry:
            gap_Br_response = artifact_store.load_dataarray(
                "gap_Br_response", print_info=print_info
            )
            if require_gap_Br_response and gap_Br_response is None:
                raise ValueError(
                    f"No saved 'gap_Br_response' data array exists at {run_directory!r}"
                )

        view = cls(
            artifact_store=artifact_store,
            datasets=datasets,
            config=config,
            schema=schema,
            main_field=main_field,
            operator_cache=operator_cache,
            gap_Br_response=gap_Br_response,
        )
        if build_geometry:
            view.require_geometry()
        return view

    @staticmethod
    def _load_required_dataset(artifact_store: ArtifactStore, key: str, print_info: bool):
        """Load a required saved dataset."""
        dataset = artifact_store.load_dataset(key, print_info=print_info)
        if dataset is None:
            raise ValueError(f"No saved {key!r} dataset exists at {artifact_store.directory!r}")
        return dataset

    def require_geometry(self) -> SimulationGeometry:
        """Return the lazily constructed geometry for this saved run."""
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
        """Load all persisted input time series for this run."""
        series = FieldTimeSeries(self.schema.input_field_spaces, self.schema.input_variables)
        series.load_all(self.artifact_store)
        return series

    def load_output_series(self) -> FieldTimeSeries:
        """Load all persisted output time series for this run."""
        series = FieldTimeSeries(self.schema.output_field_spaces, self.schema.output_variables)
        series.load_all(self.artifact_store)
        return series


__all__ = ["SavedRunView"]
