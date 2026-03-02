"""Saved simulation data helpers.

This module defines the shared persisted-run schema used by both the live
simulation writer and saved-results readers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.primitives.io import IO
from pynamit.primitives.mainfield import Mainfield
from pynamit.primitives.timeseries import Timeseries
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.simulation.input import conductance_timeseries_vars_for_mode
from pynamit.simulation.settings import DynamicsSettings


def build_simulation_bases(settings: DynamicsSettings) -> tuple[CSBasis, SHBasis, SHBasis]:
    """Build the standard CS/SH bases implied by saved settings."""
    cs_basis = CSBasis(int(settings.Ncs))
    sh_basis = SHBasis(int(settings.Nmax), int(settings.Mmax), Nmin=0)
    sh_basis_zero_removed = SHBasis(int(settings.Nmax), int(settings.Mmax))
    return cs_basis, sh_basis, sh_basis_zero_removed


def create_input_timeseries(
    settings: DynamicsSettings,
    *,
    sh_basis: SHBasis,
    sh_basis_zero_removed: SHBasis,
) -> tuple[Timeseries, dict[str, dict[str, str]], dict[str, Any]]:
    """Create the canonical input timeseries schema for a simulation."""
    input_variables = {
        "jr": {"jr": "scalar"},
        "Br": {"Br": "scalar"},
        "conductance": conductance_timeseries_vars_for_mode(
            settings.conductance_interpolation_mode
        ),
        "u": {"u": "tangential"},
    }
    input_storage_bases = {
        "jr": sh_basis_zero_removed,
        "Br": sh_basis_zero_removed,
        "conductance": sh_basis,
        "u": sh_basis_zero_removed,
    }
    return (
        Timeseries(input_storage_bases, input_variables),
        input_variables,
        input_storage_bases,
    )


def create_output_timeseries(
    settings: DynamicsSettings,
    *,
    cs_basis: CSBasis,
    sh_basis_zero_removed: SHBasis,
) -> tuple[Timeseries, dict[str, dict[str, str]], dict[str, Any], Any]:
    """Create the canonical output timeseries schema for a simulation."""
    output_variables = {
        "state": {
            "m_ind": "scalar",
            "psi": "scalar",
            "m_imp": "scalar",
            "Phi": "scalar",
            "W": "scalar",
        },
        "steady_state": {
            "m_ind": "scalar",
            "psi": "scalar",
            "m_imp": "scalar",
            "Phi": "scalar",
            "W": "scalar",
        },
    }
    solution_basis = cs_basis if settings.solution_basis_kind == "CS" else sh_basis_zero_removed
    output_storage_bases = {
        "state": solution_basis,
        "steady_state": solution_basis,
    }
    return (
        Timeseries(output_storage_bases, output_variables),
        output_variables,
        output_storage_bases,
        solution_basis,
    )


class SimulationData:
    """Persisted simulation package.

    This object owns the saved-run context needed to interpret archived inputs
    and outputs: normalized settings, sidecar operators, and the input/output
    timeseries datasets. It is used on the write side by ``Dynamics`` and on the
    read side by visualization and notebooks.
    """

    def __init__(
        self,
        *,
        filename_prefix: str | None,
        io: IO,
        settings_dataset: Any,
        settings: DynamicsSettings,
        mainfield: Mainfield,
        cs_basis: CSBasis,
        sh_basis: SHBasis,
        sh_basis_zero_removed: SHBasis,
        input_timeseries: Timeseries,
        input_variables: dict[str, dict[str, str]],
        input_storage_bases: dict[str, Any],
        output_timeseries: Timeseries,
        output_variables: dict[str, dict[str, str]],
        output_storage_bases: dict[str, Any],
        solution_basis: Any,
        pfac_matrix: Optional[np.ndarray],
        settings_from_file: bool,
        pfac_from_file: bool,
    ) -> None:
        self.filename_prefix = filename_prefix
        self.io = io
        self.settings_dataset = settings_dataset
        self.settings = settings
        self.mainfield = mainfield
        self.cs_basis = cs_basis
        self.sh_basis = sh_basis
        self.sh_basis_zero_removed = sh_basis_zero_removed
        self.input_timeseries = input_timeseries
        self.input_variables = input_variables
        self.input_storage_bases = input_storage_bases
        self.output_timeseries = output_timeseries
        self.output_variables = output_variables
        self.output_storage_bases = output_storage_bases
        self.solution_basis = solution_basis
        self.pfac_matrix = None if pfac_matrix is None else np.asarray(pfac_matrix)
        self.settings_from_file = bool(settings_from_file)
        self.pfac_from_file = bool(pfac_from_file)

    @classmethod
    def create(
        cls,
        filename_prefix: str | Path | None,
        settings: DynamicsSettings,
        *,
        load_existing: bool = True,
        print_info: bool = False,
        require_saved_settings: bool = False,
    ) -> "SimulationData":
        """Create a persisted-run container from settings and optional saved files."""
        prefix = None if filename_prefix is None else str(filename_prefix)
        io = IO(prefix)

        settings_dataset = None
        if load_existing:
            settings_dataset = io.load_dataset("settings", print_info=print_info)
        settings_from_file = settings_dataset is not None
        if settings_dataset is not None:
            if not settings.to_dataset().identical(settings_dataset):
                raise ValueError(
                    "Mismatch between requested settings and saved settings on file."
                )
            effective_settings = DynamicsSettings.from_dataset(settings_dataset, settings)
        else:
            if require_saved_settings:
                raise ValueError(
                    f"Settings dataset not found for filename prefix {prefix!r}."
                )
            effective_settings = settings
            settings_dataset = effective_settings.to_dataset()
        if prefix is not None:
            effective_settings.filename_prefix = prefix

        cs_basis, sh_basis, sh_basis_zero_removed = build_simulation_bases(effective_settings)

        (
            input_timeseries,
            input_variables,
            input_storage_bases,
        ) = create_input_timeseries(
            effective_settings,
            sh_basis=sh_basis,
            sh_basis_zero_removed=sh_basis_zero_removed,
        )
        if load_existing:
            input_timeseries.load_all(io)
            cls._prune_missing_variables(input_timeseries)

        (
            output_timeseries,
            output_variables,
            output_storage_bases,
            solution_basis,
        ) = create_output_timeseries(
            effective_settings,
            cs_basis=cs_basis,
            sh_basis_zero_removed=sh_basis_zero_removed,
        )
        if load_existing:
            output_timeseries.load_all(io)
            cls._prune_missing_variables(output_timeseries)

        pfac_dataarray = io.load_dataarray("PFAC_matrix", print_info=print_info) if load_existing else None
        pfac_from_file = pfac_dataarray is not None
        pfac_matrix = None if pfac_dataarray is None else np.asarray(pfac_dataarray.values)

        mainfield = Mainfield(
            kind=effective_settings.mainfield_kind,
            epoch=int(effective_settings.mainfield_epoch),
            hI=(float(effective_settings.RI) - RE) * 1e-3,
            B0=effective_settings.mainfield_B0,
        )

        return cls(
            filename_prefix=prefix,
            io=io,
            settings_dataset=settings_dataset,
            settings=effective_settings,
            mainfield=mainfield,
            cs_basis=cs_basis,
            sh_basis=sh_basis,
            sh_basis_zero_removed=sh_basis_zero_removed,
            input_timeseries=input_timeseries,
            input_variables=input_variables,
            input_storage_bases=input_storage_bases,
            output_timeseries=output_timeseries,
            output_variables=output_variables,
            output_storage_bases=output_storage_bases,
            solution_basis=solution_basis,
            pfac_matrix=pfac_matrix,
            settings_from_file=settings_from_file,
            pfac_from_file=pfac_from_file,
        )

    @classmethod
    def from_prefix(cls, filename_prefix: str | Path) -> "SimulationData":
        """Load a saved simulation package from one filename prefix."""
        prefix = str(filename_prefix)
        io = IO(prefix)
        settings_dataset = io.load_dataset("settings")
        if settings_dataset is None:
            raise ValueError(f"Settings dataset not found for filename prefix {prefix!r}.")
        settings = DynamicsSettings.from_dataset(
            settings_dataset,
            DynamicsSettings(filename_prefix=prefix),
        )
        return cls.create(
            prefix,
            settings,
            load_existing=True,
            require_saved_settings=True,
        )

    @property
    def datasets(self) -> dict[str, Any]:
        """Return loaded datasets keyed by saved-run name."""
        merged = {"settings": self.settings_dataset}
        merged.update(self.input_timeseries.datasets)
        merged.update(self.output_timeseries.datasets)
        return merged

    def has_dataset(self, key: str) -> bool:
        """Return whether a named dataset is available."""
        return key == "settings" or key in self.input_timeseries.datasets or key in self.output_timeseries.datasets

    def has_input_dataset(self, key: str) -> bool:
        """Return whether an input dataset is available."""
        return key in self.input_timeseries.datasets

    def has_output_dataset(self, key: str) -> bool:
        """Return whether an output dataset is available."""
        return key in self.output_timeseries.datasets

    def get_dataset(self, key: str) -> Any:
        """Return a raw xarray dataset by name."""
        if key == "settings":
            return self.settings_dataset
        if key in self.input_timeseries.datasets:
            return self.input_timeseries.datasets[key]
        if key in self.output_timeseries.datasets:
            return self.output_timeseries.datasets[key]
        raise KeyError(f"Dataset {key!r} is not available.")

    def get_input_entry(
        self,
        key: str,
        time: float,
        interpolation: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        """Return one saved input entry at ``time``."""
        if key not in self.input_timeseries.datasets:
            return None
        return self.input_timeseries.get_entry(key, time, interpolation=interpolation)

    def get_input_entry_with_derivative(
        self,
        key: str,
        time: float,
        interpolation: bool = False,
    ) -> tuple[Optional[dict[str, np.ndarray]], Optional[dict[str, np.ndarray]]]:
        """Return one saved input entry and its derivative at ``time``."""
        if key not in self.input_timeseries.datasets:
            return None, None
        return self.input_timeseries.get_entry_with_derivative(
            key,
            time,
            interpolation=interpolation,
        )

    def get_output_entry(
        self,
        key: str,
        time: float,
        interpolation: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        """Return one saved output entry at ``time``."""
        if key not in self.output_timeseries.datasets:
            return None
        return self.output_timeseries.get_entry(key, time, interpolation=interpolation)

    def get_state_entry(
        self,
        time: float,
        *,
        steady_state: bool = False,
        interpolation: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        """Return one saved state entry at ``time``."""
        key = "steady_state" if steady_state else "state"
        return self.get_output_entry(key, time, interpolation=interpolation)

    def get_latest_output_time(self, key: str = "state") -> float:
        """Return the latest saved time for one output dataset."""
        if key not in self.output_timeseries.datasets:
            raise KeyError(f"Output dataset {key!r} is not available.")
        return float(np.max(self.output_timeseries.datasets[key].time.values))

    def get_storage_basis(self, key: str) -> Any:
        """Return the storage basis for a saved input/output dataset."""
        if key in self.input_timeseries.storage_bases:
            return self.input_timeseries.storage_bases[key]
        if key in self.output_timeseries.storage_bases:
            return self.output_timeseries.storage_bases[key]
        raise KeyError(f"No storage basis is registered for dataset {key!r}.")

    def save_settings(self, *, print_info: bool = False) -> None:
        """Persist normalized settings to disk."""
        self.settings_dataset = self.settings.to_dataset()
        self.io.save_dataset(self.settings_dataset, "settings", print_info=print_info)
        self.settings_from_file = True

    def save_pfac_matrix(
        self,
        pfac_matrix: Any,
        *,
        print_info: bool = False,
    ) -> None:
        """Persist the PFAC sidecar matrix to disk and cache it on the object."""
        self.pfac_matrix = np.asarray(pfac_matrix)
        self.io.save_dataarray(
            xr.DataArray(self.pfac_matrix),
            "PFAC_matrix",
            print_info=print_info,
        )
        self.pfac_from_file = True

    def add_output_entry(
        self,
        key: str,
        data: dict[str, Any],
        *,
        time: float,
    ) -> None:
        """Append one output entry to the persisted output timeseries."""
        self.output_timeseries.add_entry(key, data, time=time)

    def save_output_dataset(self, key: str, *, print_info: bool = False) -> None:
        """Persist one output dataset to disk."""
        del print_info
        self.output_timeseries.save(key, self.io)

    @staticmethod
    def _prune_missing_variables(timeseries: Timeseries) -> None:
        """Trim variable maps to the variables actually present on disk.

        Saved datasets can legitimately contain an older/newer variable subset than
        the schema implied by the current settings. In that case we infer the
        available variables directly from the stored dataset names so read-side
        access keeps working for archived results.
        """
        for key, dataset in timeseries.datasets.items():
            prefix = f"{timeseries.storage_bases[key].kind}_"
            known_variables = dict(timeseries.variables[key])
            data_var_names = [name for name in dataset.data_vars if name.startswith(prefix)]

            present = {
                var: var_type
                for var, var_type in known_variables.items()
                if f"{prefix}{var}" in data_var_names
            }
            if present:
                timeseries.variables[key] = present
                continue

            inferred = {}
            for name in data_var_names:
                var = name[len(prefix) :]
                inferred[var] = known_variables.get(var, "scalar")
            if inferred:
                timeseries.variables[key] = inferred
