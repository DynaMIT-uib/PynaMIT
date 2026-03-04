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
from pynamit.primitives.field_spec import FieldSpec
from pynamit.primitives.io import IO
from pynamit.primitives.mainfield import Mainfield
from pynamit.primitives.timeseries import Timeseries
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.simulation.input import conductance_timeseries_vars_for_mode
from pynamit.simulation.settings import DynamicsSettings
from pynamit.simulation.spatial.geometry_utils import get_radial_shift_diagonal, to_dense

def _build_simulation_bases(settings: DynamicsSettings) -> tuple[CSBasis, SHBasis]:
    """Build the standard CS basis and canonical full SH basis."""
    cs_basis = CSBasis(int(settings.Ncs))
    sh_basis = SHBasis(int(settings.Nmax), int(settings.Mmax), mean_free=False)
    return cs_basis, sh_basis


def _create_input_timeseries(
    settings: DynamicsSettings,
    *,
    sh_basis: SHBasis,
) -> tuple[Timeseries, dict[str, dict[str, str]]]:
    """Create the canonical input timeseries schema for a simulation."""
    input_variables = {
        "jr": {"jr": "scalar"},
        "Br": {"Br": "scalar"},
        "conductance": conductance_timeseries_vars_for_mode(
            settings.conductance_interpolation_mode
        ),
        "u": {"u": "tangential"},
    }
    input_storage_specs = {
        "jr": FieldSpec(basis=sh_basis, field_type="scalar", mean_free=True),
        "Br": FieldSpec(basis=sh_basis, field_type="scalar", mean_free=True),
        "conductance": FieldSpec(basis=sh_basis, field_type="scalar", mean_free=False),
        "u": FieldSpec(basis=sh_basis, field_type="tangential", mean_free=True),
    }
    timeseries = Timeseries(input_storage_specs, input_variables)
    return timeseries, input_variables


def _create_output_timeseries(
    settings: DynamicsSettings,
    *,
    cs_basis: CSBasis,
    sh_basis: SHBasis,
) -> tuple[Timeseries, dict[str, dict[str, str]], FieldSpec]:
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
    solution_spec = FieldSpec(
        basis=cs_basis if settings.solution_basis_kind == "CS" else sh_basis,
        field_type="scalar",
        # State-side scalar quantities are conceptually zero-mean in both SH and
        # CS representations. SH realizes this by omitting the monopole;
        # CS realizes it through mean-zero gauge/constraint handling.
        mean_free=True,
    )
    output_storage_specs = {
        "state": solution_spec,
        "steady_state": solution_spec,
    }
    timeseries = Timeseries(output_storage_specs, output_variables)
    return timeseries, output_variables, solution_spec


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
        run_directory: str | None,
        io: IO,
        settings_dataset: Any,
        settings: DynamicsSettings,
        mainfield: Mainfield,
        cs_basis: CSBasis,
        sh_basis: SHBasis,
        input_timeseries: Timeseries,
        input_variables: dict[str, dict[str, str]],
        output_timeseries: Timeseries,
        output_variables: dict[str, dict[str, str]],
        solution_spec: FieldSpec,
        pfac_matrix: Optional[np.ndarray],
        settings_from_file: bool,
        pfac_from_file: bool,
    ) -> None:
        self.run_directory = run_directory
        self.io = io
        self.settings_dataset = settings_dataset
        self.settings = settings
        self.mainfield = mainfield
        self.cs_basis = cs_basis
        self.sh_basis = sh_basis
        self.input_timeseries = input_timeseries
        self.input_variables = input_variables
        self.output_timeseries = output_timeseries
        self.output_variables = output_variables
        self.solution_spec = solution_spec
        self.pfac_matrix = None if pfac_matrix is None else np.asarray(pfac_matrix)
        self.settings_from_file = bool(settings_from_file)
        self.pfac_from_file = bool(pfac_from_file)
        self._poloidal_results_operators_cache: dict[tuple[Any, ...], Any] = {}

    @classmethod
    def create(
        cls,
        run_directory: str | Path | None,
        settings: DynamicsSettings,
        *,
        load_existing: bool = True,
        print_info: bool = False,
        require_saved_settings: bool = False,
        io: IO | None = None,
        settings_dataset: xr.Dataset | None = None,
    ) -> "SimulationData":
        """Create a persisted-run container from settings and optional saved files."""
        resolved_run_directory = (
            None if run_directory is None else IO.build_run_directory(run_directory)
        )
        io = IO(resolved_run_directory) if io is None else io

        if settings_dataset is None and load_existing:
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
                    f"Settings dataset not found for run directory {resolved_run_directory!r}."
                )
            effective_settings = settings
            settings_dataset = effective_settings.to_dataset()
        if resolved_run_directory is not None:
            effective_settings.run_directory = resolved_run_directory

        cs_basis, sh_basis = _build_simulation_bases(effective_settings)

        (
            input_timeseries,
            input_variables,
        ) = _create_input_timeseries(
            effective_settings,
            sh_basis=sh_basis,
        )
        if load_existing:
            input_timeseries.load_all(io)
            cls._prune_missing_variables(input_timeseries)

        (
            output_timeseries,
            output_variables,
            solution_spec,
        ) = _create_output_timeseries(
            effective_settings,
            cs_basis=cs_basis,
            sh_basis=sh_basis,
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
            run_directory=resolved_run_directory,
            io=io,
            settings_dataset=settings_dataset,
            settings=effective_settings,
            mainfield=mainfield,
            cs_basis=cs_basis,
            sh_basis=sh_basis,
            input_timeseries=input_timeseries,
            input_variables=input_variables,
            output_timeseries=output_timeseries,
            output_variables=output_variables,
            solution_spec=solution_spec,
            pfac_matrix=pfac_matrix,
            settings_from_file=settings_from_file,
            pfac_from_file=pfac_from_file,
        )

    @classmethod
    def from_directory(cls, run_directory: str | Path) -> "SimulationData":
        """Load a saved simulation package from one run directory."""
        resolved_run_directory = IO.discover_run_directory(run_directory)
        io = IO(resolved_run_directory)
        settings_dataset = io.load_dataset("settings")
        if settings_dataset is None:
            raise ValueError(f"Settings dataset not found for run directory {resolved_run_directory!r}.")
        settings = DynamicsSettings.from_dataset(
            settings_dataset,
            DynamicsSettings(run_directory=resolved_run_directory),
        )
        return cls.create(
            resolved_run_directory,
            settings,
            load_existing=True,
            require_saved_settings=True,
            io=io,
            settings_dataset=settings_dataset,
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

    def get_latest_output_time(self, key: str = "state") -> float:
        """Return the latest saved time for one output dataset."""
        if key not in self.output_timeseries.datasets:
            raise KeyError(f"Output dataset {key!r} is not available.")
        return float(np.max(self.output_timeseries.datasets[key].time.values))

    def get_storage_spec(self, key: str) -> FieldSpec:
        """Return the storage specification for a saved input/output dataset."""
        if key in self.input_timeseries.storage_specs:
            return self.input_timeseries.get_storage_spec(key)
        if key in self.output_timeseries.storage_specs:
            return self.output_timeseries.get_storage_spec(key)
        raise KeyError(f"No storage basis is registered for dataset {key!r}.")

    def get_data_var_name(self, key: str, var: str) -> str:
        """Return the stored xarray variable name for one saved series variable."""
        if key in self.input_timeseries.storage_specs:
            return self.input_timeseries.get_data_var_name(key, var)
        if key in self.output_timeseries.storage_specs:
            return self.output_timeseries.get_data_var_name(key, var)
        raise KeyError(f"No saved series {key!r} is registered for variable lookup.")

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

    def save_input_dataset(self, key: str, *, print_info: bool = False) -> None:
        """Persist one input dataset to disk."""
        self.input_timeseries.save(key, self.io, print_info=print_info)

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
        self.output_timeseries.save(key, self.io, print_info=print_info)

    def get_poloidal_results_operators(
        self,
        *,
        grid: Any,
        basis: Any = None,
    ) -> Any:
        """Build explicit postprocessing operators for one target grid."""
        from pynamit.postprocess.results_operators import build_poloidal_results_operators

        target_basis = self.solution_spec if basis is None else basis
        cache_key = (
            getattr(grid, "hash", id(grid)),
            getattr(target_basis, "signature", id(target_basis)),
        )
        cached = self._poloidal_results_operators_cache.get(cache_key)
        if cached is not None:
            return cached

        t_to_ve = self._get_locked_pfac_operator()
        if t_to_ve is not None and int(target_basis.index_length) != int(t_to_ve.shape[0]):
            raise ValueError(
                "Saved PFAC operator is stored in the simulation solution basis. "
                f"Requested basis length {int(target_basis.index_length)} does not match "
                f"stored PFAC shape {t_to_ve.shape}."
            )

        bundle = build_poloidal_results_operators(
            basis=target_basis,
            grid=grid,
            RI=float(self.settings.RI),
            T_to_Ve=t_to_ve,
            RM=self.settings.RM,
        )
        self._poloidal_results_operators_cache[cache_key] = bundle
        return bundle

    def _get_locked_pfac_operator(self) -> Optional[np.ndarray]:
        """Return the saved PFAC operator with imposed RM closure applied."""
        if self.pfac_matrix is None:
            return None
        return self._apply_imposed_toroidal_poloidal_lock(
            np.asarray(self.pfac_matrix, dtype=float),
            solution_space=self.solution_spec,
        )

    def _apply_imposed_toroidal_poloidal_lock(
        self,
        operator: np.ndarray,
        *,
        solution_space: Any,
    ) -> np.ndarray:
        """Apply the imposed toroidal-poloidal RM closure in solution space."""
        rm = None if self.settings.RM in (None, 0) else float(self.settings.RM)
        if rm is None:
            return np.asarray(operator, dtype=float)

        closure_basis = self._get_pfac_closure_basis(solution_space)
        br_rm_to_ri_shift = get_radial_shift_diagonal(
            closure_basis,
            rm,
            float(self.settings.RI),
            kind="external",
        )
        br_ri_to_rm_shift = get_radial_shift_diagonal(
            closure_basis,
            float(self.settings.RI),
            rm,
            kind="internal",
        )
        roundtrip_denominator = 1.0 - (br_rm_to_ri_shift * br_ri_to_rm_shift)
        roundtrip_operator = np.diag(np.asarray(roundtrip_denominator, dtype=float))

        if closure_basis is solution_space:
            roundtrip_vec = np.asarray(roundtrip_denominator, dtype=float).reshape(-1)
            tol = max(float(np.finfo(float).eps * max(roundtrip_vec.size, 1)), 1e-15)
            inv_roundtrip_vec = np.zeros_like(roundtrip_vec)
            keep = np.abs(roundtrip_vec) > tol
            inv_roundtrip_vec[keep] = 1.0 / roundtrip_vec[keep]
            roundtrip_inv = np.diag(inv_roundtrip_vec)
        else:
            grid = getattr(solution_space, "grid", None)
            if grid is None:
                raise ValueError(
                    "Cannot project saved PFAC RM closure into solution space without "
                    "a solution-basis grid."
                )

            solution_to_closure = np.asarray(
                to_dense(closure_basis.construct_scalar_projection_matrix(grid))
                @ to_dense(solution_space.get_evaluation_matrix(grid)),
                dtype=float,
            )
            closure_to_solution = np.asarray(
                to_dense(solution_space.construct_scalar_projection_matrix(grid))
                @ to_dense(closure_basis.get_evaluation_matrix(grid)),
                dtype=float,
            )
            roundtrip_solution = closure_to_solution @ roundtrip_operator @ solution_to_closure
            rcond = float(np.finfo(float).eps * max(roundtrip_solution.shape))
            roundtrip_inv = np.linalg.pinv(roundtrip_solution, rcond=max(rcond, 1e-15))

        if roundtrip_inv.shape[1] != operator.shape[0]:
            raise ValueError(
                "RM closure operator shape mismatch for saved PFAC matrix: "
                f"{roundtrip_inv.shape} cannot left-multiply {operator.shape}."
            )
        return np.asarray(roundtrip_inv @ np.asarray(operator, dtype=float))

    def _get_pfac_closure_basis(self, solution_space: Any) -> Any:
        """Return the closure basis used for PFAC/radial coupling semantics."""
        mode = getattr(self.settings.simulation_mode, "value", self.settings.simulation_mode)
        if getattr(solution_space, "kind", "") in ("CS", "GRID") and mode == "cs_dominant":
            return FieldSpec(basis=self.sh_basis, field_type="scalar", mean_free=True)
        return solution_space

    @staticmethod
    def _prune_missing_variables(timeseries: Timeseries) -> None:
        """Trim variable maps to the variables actually present on disk.

        Saved datasets can legitimately contain an older/newer variable subset than
        the schema implied by the current settings. In that case we infer the
        available variables directly from the stored dataset names so read-side
        access keeps working for archived results.
        """
        for key, dataset in timeseries.datasets.items():
            prefix = f"{timeseries.get_storage_spec(key).kind}_"
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
