"""Saved-run view shared by visualization frontends."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.geometry import Geometry
from pynamit.simulation.mainfield import Mainfield
from pynamit.simulation.schema import SimulationSchema, build_simulation_schema


@dataclass
class SavedRunView:
    """Loaded simulation artifacts and derived visualization objects."""

    io: IO
    datasets: dict[str, xr.Dataset]
    config: SimulationConfig
    schema: SimulationSchema
    mainfield: Mainfield
    pfac_matrix: xr.DataArray | None = None
    geometry: Geometry | None = None

    @classmethod
    def from_directory(
        cls,
        run_directory,
        *,
        required_datasets=(),
        optional_datasets=(),
        require_pfac_matrix=False,
        build_geometry=False,
        artifact_storage="auto",
        print_info=False,
    ) -> "SavedRunView":
        """Load one saved run for visualization."""
        io = IO(run_directory, preferred_dataset_storage=artifact_storage)
        datasets = {"settings": cls._load_required_dataset(io, "settings", print_info)}
        for key in required_datasets:
            if key == "settings":
                continue
            datasets[key] = cls._load_required_dataset(io, key, print_info)
        for key in optional_datasets:
            if key in datasets:
                continue
            dataset = io.load_dataset(key, print_info=print_info)
            if dataset is not None:
                datasets[key] = dataset

        config = SimulationConfig.from_settings(datasets["settings"])
        schema = build_simulation_schema(config)
        mainfield = Mainfield(
            kind=config.mainfield_kind,
            epoch=config.mainfield_epoch,
            hI=(config.RI - RE) * 1e-3,
            B0=config.mainfield_B0,
        )

        pfac_matrix = None
        if require_pfac_matrix or build_geometry:
            pfac_matrix = io.load_dataarray("PFAC_matrix", print_info=print_info)
            if pfac_matrix is None:
                raise ValueError(
                    f"No saved 'PFAC_matrix' data array exists at {run_directory!r}"
                )

        geometry = None
        if build_geometry:
            geometry = Geometry(
                basis=schema.horizontal_basis,
                cs_basis=schema.cs_basis,
                mainfield=mainfield,
                settings=config,
                PFAC_matrix=pfac_matrix,
                solid_harmonics=schema.solid_harmonics,
            )

        return cls(
            io=io,
            datasets=datasets,
            config=config,
            schema=schema,
            mainfield=mainfield,
            pfac_matrix=pfac_matrix,
            geometry=geometry,
        )

    @staticmethod
    def _load_required_dataset(io: IO, key: str, print_info: bool):
        """Load a required saved dataset."""
        dataset = io.load_dataset(key, print_info=print_info)
        if dataset is None:
            raise ValueError(f"No saved {key!r} dataset exists at {io.run_directory!r}")
        return dataset

    @property
    def settings(self):
        """Return the saved settings dataset."""
        return self.datasets["settings"]

    @property
    def run_directory(self):
        """Return the resolved run directory."""
        return self.io.run_directory

    @property
    def RI(self):
        """Return the ionospheric radius."""
        return self.config.RI

    def load_input_timeseries(self) -> Timeseries:
        """Load all persisted input time series for this run."""
        timeseries = Timeseries(
            self.schema.input_field_spaces,
            self.schema.input_vars,
            area_weighted_least_squares=self.config.area_weighted_least_squares,
        )
        timeseries.load_all(self.io)
        return timeseries

    def load_output_timeseries(self) -> Timeseries:
        """Load all persisted output time series for this run."""
        timeseries = Timeseries(
            self.schema.output_field_spaces,
            self.schema.output_vars,
            area_weighted_least_squares=self.config.area_weighted_least_squares,
        )
        timeseries.load_all(self.io)
        return timeseries


__all__ = ["SavedRunView"]
