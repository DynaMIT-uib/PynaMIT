"""Saved-run data container for Dynamics.

``SimulationData`` owns the persisted-run plumbing that is shared by
runtime simulation, restart, and saved-result inspection: IO, storage
schema, and loaded input/output time-series datasets.
"""

from dataclasses import dataclass
from typing import Any

import xarray as xr

from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.simulation.schema import (
    SimulationSchema,
    build_simulation_schema,
    normalize_horizontal_basis_kind,
    setting_value,
)


def _resolve_setting_override(
    settings,
    name,
    explicit,
    *,
    default,
    normalize=lambda value: value,
):
    """Return a setting, rejecting explicit values that conflict."""
    stored = setting_value(settings, name, None)
    if explicit is None:
        value = default if stored is None else stored
        return normalize(value)

    normalized_explicit = normalize(explicit)
    if stored is not None and normalized_explicit != normalize(stored):
        raise ValueError(f"{name} argument does not match settings.")
    return normalized_explicit


@dataclass
class SimulationData:
    """Persisted-run context for one simulation."""

    io: IO
    settings: xr.Dataset
    schema: SimulationSchema
    input_timeseries: Timeseries
    output_timeseries: Timeseries
    settings_on_file: xr.Dataset | None = None
    pfac_matrix: Any = None
    uses_temporary_run_directory: bool = False
    settings_loaded_from_file: bool = False
    pfac_matrix_loaded_from_file: bool = False

    @classmethod
    def create(
        cls,
        settings: xr.Dataset,
        *,
        run_directory=None,
        artifact_storage="auto",
        horizontal_basis_kind=None,
        area_weighted_least_squares=None,
        print_info=False,
    ) -> "SimulationData":
        """Create persisted-run context and load saved artifacts."""
        uses_temporary_run_directory = run_directory is None
        if uses_temporary_run_directory:
            run_directory = IO.build_temporary_run_directory()

        io = IO(
            run_directory=run_directory,
            preferred_dataset_storage=artifact_storage,
        )

        settings_on_file = io.load_dataset("settings", print_info=print_info)
        if settings_on_file is not None and not settings.identical(settings_on_file):
            raise ValueError("Mismatch between Dynamics object arguments and settings on file.")

        pfac_matrix = io.load_dataarray("PFAC_matrix", print_info=print_info)
        horizontal_basis_kind = _resolve_setting_override(
            settings,
            "horizontal_basis_kind",
            horizontal_basis_kind,
            default="SH",
            normalize=normalize_horizontal_basis_kind,
        )
        area_weighted_least_squares = _resolve_setting_override(
            settings,
            "area_weighted_least_squares",
            area_weighted_least_squares,
            default=False,
            normalize=bool,
        )
        schema = build_simulation_schema(settings, horizontal_basis_kind)

        input_timeseries = Timeseries(
            schema.input_field_spaces,
            schema.input_vars,
            area_weighted_least_squares=bool(area_weighted_least_squares),
        )
        input_timeseries.load_all(io)

        output_timeseries = Timeseries(
            schema.output_field_spaces,
            schema.output_vars,
            area_weighted_least_squares=bool(area_weighted_least_squares),
        )
        output_timeseries.load_all(io)

        return cls(
            io=io,
            settings=settings,
            schema=schema,
            input_timeseries=input_timeseries,
            output_timeseries=output_timeseries,
            settings_on_file=settings_on_file,
            pfac_matrix=pfac_matrix,
            uses_temporary_run_directory=uses_temporary_run_directory,
            settings_loaded_from_file=settings_on_file is not None,
            pfac_matrix_loaded_from_file=pfac_matrix is not None,
        )

    @property
    def run_directory(self):
        """Return the resolved run directory."""
        return self.io.run_directory

    @property
    def settings_from_file(self):
        """Return whether settings were loaded from this run dir."""
        return self.settings_loaded_from_file

    @property
    def pfac_matrix_from_file(self):
        """Return whether PFAC was loaded from this run dir."""
        return self.pfac_matrix_loaded_from_file

    def save_settings_if_missing(self, *, print_info=False):
        """Persist settings when this is a new run directory."""
        if self.settings_on_file is not None:
            return
        self.io.save_dataset(self.settings, "settings", print_info=print_info)
        self.settings_on_file = self.settings

    def save_pfac_matrix_if_missing(self, pfac_matrix, *, print_info=False):
        """Persist the PFAC sidecar for a new run directory."""
        if self.pfac_matrix is not None:
            return
        self.io.save_dataarray(pfac_matrix, "PFAC_matrix", print_info=print_info)
        self.pfac_matrix = pfac_matrix

    def save_input_dataset(self, key, *, print_info=False):
        """Persist one loaded input time series."""
        self.input_timeseries.save(key, self.io, print_info=print_info)

    def add_output_entry(self, key, data, *, time):
        """Append one output entry to a loaded output time series."""
        self.output_timeseries.add_entry(key, data, time=time)

    def save_output_dataset(self, key, *, print_info=False):
        """Persist one loaded output time series."""
        self.output_timeseries.save(key, self.io, print_info=print_info)
