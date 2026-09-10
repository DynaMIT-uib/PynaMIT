"""Configure and evolve a coupled PynaMIT simulation."""

from dataclasses import replace

import numpy as np
from kompe.constants import EARTH_RADIUS_M
from kompe.math import get_array_module

from pynamit.results.simulation_results import SimulationResults
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import induction
from pynamit.simulation.evolution import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DEFAULT_SAMPLES_PER_WRITE,
    _positive_integer,
    _TimeEvolution,
)
from pynamit.simulation.input_manifest import validate_prepared_input_compatibility
from pynamit.simulation.input_preparation import InputPreparation
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.storage.field_time_series import FieldTimeSeries, time_roundoff


class Simulation:
    """Configure, drive, evolve, and persist one coupled MIT simulation.

    ``inputs`` is an :class:`InputPreparation`: set or inspect forcing
    there, then evolve the simulation. Without a directory, inputs and
    outputs stay in memory until ``save(directory)`` is called.

    ``current_time`` and ``induced_Br`` are the live dynamic state,
    independent of recorded results. The field is None initially.
    Use ``set_state`` for an initial condition and ``restore_state``
    for a recorded checkpoint. Opening a saved simulation restores
    its latest dynamic checkpoint once, not on each continuation.
    """

    def __init__(
        self,
        simulation_directory=None,
        *,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=EARTH_RADIUS_M + 110.0e3,
        RM=None,
        magnetic_boundary_shielding=False,
        main_field_kind="dipole",
        main_field_epoch=None,
        main_field_B0=None,
        fac_integration_radii=None,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=False,
        interhemispheric_coupling_latitude=50,
        interhemispheric_electric_field_weight=1e-5,
        boundary_jr_remapping=None,
        boundary_Br_remapping=None,
        conductance_basis=None,
        u_remapping=None,
        Q_eff_remapping=None,
        E_neutral_wind_remapping=None,
        t0="2020-01-01 00:00:00",
        save_equilibria=True,
        integrator="euler",
        least_squares_solver=None,
        least_squares_tolerance=1e-15,
        least_squares_preconditioner=None,
        reuse_preconditioner=False,
        toroidal_potential_regularization_lambda=0.0,
        artifact_storage="auto",
        operator_cache_directory=None,
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
    ):
        """Initialize a coupled MIT simulation.

        Parameters
        ----------
        simulation_directory : path-like, optional
            Directory for persisted settings, inputs, and outputs.
        Nmax, Mmax, Ncs : int, optional
            Spherical-harmonic truncation and cubed-sphere resolution.
        RI, RM : float, optional
            Ionospheric and magnetospheric radii in meters.
        magnetic_boundary_shielding : bool, optional
            Impose a shielding condition at ``RM``.
        main_field_kind, main_field_epoch, main_field_B0 : optional
            Main-field model, decimal-year epoch, and optional field
            magnitude override.
        fac_integration_radii : array-like, optional
            Radii used to integrate the FAC poloidal field.
        enable_pfac_coupling : bool, optional
            Include the FAC poloidal field in the coupled response.
        enable_interhemispheric_coupling : bool, optional
            Couple conjugate current and electric-field solutions.
        interhemispheric_coupling_latitude : float, optional
            Absolute latitude bounding the coupled low-latitude region.
        interhemispheric_electric_field_weight : float, optional
            Relative least-squares weight of the conjugate constraint.
        boundary_jr_remapping, boundary_Br_remapping,
        conductance_basis, u_remapping,
        Q_eff_remapping, E_neutral_wind_remapping :
            Sample remapping choices, except ``conductance_basis`` which
            selects conductance storage. The simulation schema
            determines the coefficient spaces. Each projection route
            has the default derived by :class:`SimulationConfig`.
        t0 : str, optional
            Physical start time.
        save_equilibria : bool, optional
            Save instantaneous induction equilibria by default.
        integrator : str, optional
            Integrator used for ``induced_Br`` evolution.
        least_squares_solver, least_squares_preconditioner : optional
            Shared algorithm and preconditioner for input and response
            fits. Defaults are SH ``normal_pinv`` and CS ``lsmr``.
        least_squares_tolerance : float, optional
            Shared fit tolerance; see ``kompe.math.LeastSquaresSolver``
            for its algorithm-specific meaning. Fixed geometric inverse
            operators are independent of these fit settings.
        reuse_preconditioner : bool, optional
            Reuse a compatible iterative-solver preconditioner.
        toroidal_potential_regularization_lambda : float, optional
            Regularization strength for the toroidal-potential solve.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred storage backend for newly saved artifacts.
        operator_cache_directory : path-like, optional
            Shared cache for deterministic numerical operators.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Horizontal surface basis.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for projections without explicit
            weights.
        """
        config = SimulationConfig(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            magnetic_boundary_shielding=magnetic_boundary_shielding,
            interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
            enable_pfac_coupling=enable_pfac_coupling,
            enable_interhemispheric_coupling=enable_interhemispheric_coupling,
            fac_integration_radii=fac_integration_radii,
            interhemispheric_electric_field_weight=interhemispheric_electric_field_weight,
            main_field_kind=main_field_kind,
            main_field_epoch=main_field_epoch,
            main_field_B0=main_field_B0,
            boundary_jr_remapping=boundary_jr_remapping,
            boundary_Br_remapping=boundary_Br_remapping,
            conductance_basis=conductance_basis,
            u_remapping=u_remapping,
            Q_eff_remapping=Q_eff_remapping,
            E_neutral_wind_remapping=E_neutral_wind_remapping,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            t0=t0,
            save_equilibria=save_equilibria,
            integrator=integrator,
            least_squares_solver=least_squares_solver,
            least_squares_tolerance=least_squares_tolerance,
            least_squares_preconditioner=least_squares_preconditioner,
            reuse_preconditioner=reuse_preconditioner,
            toroidal_potential_regularization_lambda=toroidal_potential_regularization_lambda,
        )
        self.inputs = InputPreparation.from_config(
            config,
            input_directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
        )
        self._open_simulation_runtime()

    def _open_simulation_runtime(self):
        """Initialize output and evolution state."""
        self.results = SimulationResults(self.inputs)
        self.config = self.inputs.config
        self.main_field = self.inputs.main_field
        self.model_grid = self.inputs.model_grid
        self.operator_cache = self.inputs.operator_cache
        self._response = None
        self.outputs = self.results.output_series.datasets
        self.current_time = 0.0
        self.induced_Br = None
        self._time_evolution = _TimeEvolution(self)
        if "dynamic" in self.outputs:
            self.restore_state()

    @property
    def geometry(self):
        """Return the shared, lazily constructed physical geometry."""
        return self.results.geometry

    def set_state(self, induced_Br, *, time=None):
        """Set induced-Br coefficients and their time in seconds.

        Own a backend array, independent of the caller and recorded
        results. Omitted time means the current live time. No output is
        written; the next evolution starts here, not at a saved row.
        Use a new simulation to branch before existing dynamic output.
        """
        time = FieldTimeSeries._time_value(self.current_time if time is None else time)
        if time < 0:
            raise ValueError("State time must be non-negative.")
        dynamic = self.outputs.get("dynamic")
        if dynamic is not None and time < float(dynamic.time.values[-1]) - time_roundoff(time):
            raise ValueError(
                "State time precedes recorded dynamic output; use a new simulation to branch."
            )
        space = self.results.schema.output_field_spaces["dynamic"]["induced_Br"]
        values = space.validate_coefficients(induced_Br)
        if values.shape != space.shape:
            raise ValueError(f"A live state must have shape {space.shape}, without batch axes.")
        xp = get_array_module(values)
        if not bool(xp.all(xp.isfinite(values))):
            raise ValueError("induced_Br must contain finite coefficients.")
        self.induced_Br = xp.array(values, copy=True)
        self.current_time = time

    def restore_state(self, results=None, *, time=None):
        """Restore an exact dynamic checkpoint, latest by default.

        ``results`` defaults to this simulation's recorded results.
        Pass another SimulationResults to start an independent branch.
        Coefficient space, radius, frame, and time origin must agree.
        Never interpolate or borrow a nearby checkpoint.
        """
        results = self.results if results is None else results
        series = results.output_series
        dynamic = series.datasets.get("dynamic")
        if dynamic is None:
            raise ValueError("No dynamic checkpoint is available to restore.")
        source_space = series.get_field_space("dynamic", "induced_Br")
        target_space = self.results.schema.output_field_spaces["dynamic"]["induced_Br"]
        if source_space.signature != target_space.signature or any(
            getattr(results.config, name) != getattr(self.config, name)
            for name in ("RI", "horizontal_coordinate_system", "t0")
        ):
            raise ValueError(
                "Checkpoint coordinates, time origin, or coefficient space do not match."
            )
        if (
            self.config.main_field_kind == "dipole"
            and results.config.main_field_epoch != self.config.main_field_epoch
        ):
            raise ValueError("Checkpoint dipole coordinate epochs do not match.")
        time = (
            float(dynamic.time.values[-1]) if time is None else FieldTimeSeries._time_value(time)
        )
        matches = np.flatnonzero(np.abs(dynamic.time.values - time) <= time_roundoff(time))
        if not matches.size:
            raise ValueError(f"No dynamic checkpoint exists at t={time:g} s.")
        time = float(dynamic.time.values[matches[-1]])
        self.set_state(
            series.get_entry("dynamic", time, variables=("induced_Br",))["induced_Br"], time=time
        )

    @property
    def simulation_directory(self):
        """Return the directory, or None for an in-memory simulation."""
        return self.results.simulation_directory

    def save(self, directory=None, *, artifact_storage=None):
        """Save recorded histories and enable automatic later writes.

        To record an edited or newly set live state first, call
        ``record_state()``. Saving histories does not
        change their coefficients to match an unrecorded live state.
        """
        self.results.save(directory, artifact_storage=artifact_storage)

    def output_coefficients(self):
        """Evaluate live output without recording it.

        Return physical induced Br, boundary current, and electric
        potentials. Unlike sampling ``results``, this sees an edited
        live field even when no output has been recorded at this time.
        """
        if self.induced_Br is None:
            raise ValueError("Initialize the live state before evaluating its output.")
        series = self.results.input_series
        evolution = self._time_evolution
        _, _, entries = next(
            series.iter_intervals(
                self.current_time,
                self.current_time,
                previous=evolution._input_values if series is evolution._input_series else None,
            )
        )
        response, E, jr = evolution._response_and_forcing(entries)
        return response.output_coefficients(self.induced_Br, E, jr)

    def equilibrium_coefficients(self, time=None, *, interpolation=False):
        """Calculate equilibria without changing state or histories.

        ``time`` is seconds after t0, defaulting to the live time.
        A 1-D query retains order and repetitions as a trailing
        time axis. Times need not follow the live trajectory. Inputs are
        held at their timestamps unless interpolation is requested.
        Missing optional forcing is zero before onset; conductance is
        required. Consecutive cases with equal conductance share one
        batched closure solve.
        """
        time = self.current_time if time is None else time
        series = self.results.input_series
        conductance = series.get_entry("conductance", time, interpolation)
        times = np.atleast_1d(np.asarray(time, dtype=float))
        if np.any(times < 0):
            raise ValueError("Equilibrium times must be non-negative.")
        if conductance is None:
            raise ValueError("Conductance is required at every requested equilibrium time.")
        scalar = np.ndim(time) == 0
        if scalar:
            conductance = {name: values[..., None] for name, values in conductance.items()}
        same = np.ones(times.size - 1, dtype=bool)
        for values in conductance.values():
            same &= np.all(values[..., 1:] == values[..., :-1], axis=tuple(range(values.ndim - 1)))
        boundaries = np.r_[0, np.flatnonzero(~same) + 1, times.size]
        batches = []
        response = self._response
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            response = ElectrodynamicResponse.from_conductance(
                self.geometry,
                self.config,
                series.get_field_space("conductance"),
                {name: values[..., start] for name, values in conductance.items()},
                previous=response,
            )
            forcing = {}
            for key in series.datasets:
                if key != "conductance":
                    forcing.update(
                        series.get_entry(key, times[start:stop], interpolation, fill_value=0)
                    )
            E, jr = response.solve_noninductive_response(**forcing)
            Br = induction.equilibrium_induced_Br(response, E)
            fields = response.output_coefficients(Br, E, jr)
            xp = get_array_module(Br)
            # Without forcing, the zero solution has no batch axis.
            batches.append(
                {
                    name: xp.broadcast_to(
                        value[..., None] if value.ndim == 1 else value,
                        value.shape[:1] + (stop - start,),
                    )
                    for name, value in fields.items()
                }
            )
        self._response = response
        fields = (
            batches[0]
            if len(batches) == 1
            else {
                name: xp.concatenate([batch[name] for batch in batches], axis=-1)
                for name in batches[0]
            }
        )
        return {name: value[..., 0] for name, value in fields.items()} if scalar else fields

    def record_state(self, *, save=True):
        """Record the live dynamic state without advancing it.

        Return the recorded coefficient dictionary. If a directory is
        bound, ``save=True`` also writes the dynamic stream. Equilibrium
        diagnostics are independent; use ``sample_equilibria`` for them.
        """
        fields = self.output_coefficients()
        self.results.output_series.add_entry("dynamic", fields, self.current_time)
        if save:
            self._save_output("dynamic")
        return fields

    def sample_equilibria(
        self,
        times,
        *,
        interpolation=False,
        samples_per_write=DEFAULT_SAMPLES_PER_WRITE,
        save=True,
        quiet=False,
    ):
        """Record independent equilibria, including earlier times.

        Evaluate bounded batches without changing the live
        trajectory. ``save`` writes to an already bound directory. No
        integrator, step size, or dynamic checkpoint is involved.
        """
        times = np.asarray(times)
        if (
            times.ndim != 1
            or not times.size
            or times.dtype.kind not in "fiu"
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
            or np.any(times[1:] <= times[:-1] + time_roundoff(times[1:]))
        ):
            raise ValueError("times must be a nonempty, increasing array of non-negative times.")
        size = _positive_integer(samples_per_write, name="samples_per_write")
        for start in range(0, times.size, size):
            batch_times = times[start : start + size]
            fields = self.equilibrium_coefficients(batch_times, interpolation=interpolation)
            xp = get_array_module(*fields.values())
            self.results.output_series.add_entries(
                "equilibrium",
                {name: xp.moveaxis(value, -1, 0) for name, value in fields.items()},
                batch_times,
            )
            if save:
                self._save_output("equilibrium")
            if not quiet:
                print(f"Equilibrium at t = {batch_times[-1]:g} / {times[-1]:g} s", flush=True)

    def _save_output(self, *keys):
        """Persist outputs and their magnetic reconstruction data."""
        if self.geometry.main_field.kind != "radial" and self.config.enable_pfac_coupling:
            self.results.save_boundary_jr_to_gap_Br_matrix_if_missing(
                self.geometry.boundary_jr_to_gap_Br_matrix
            )
        if self.results.artifact_store is not None:
            # A recorded trajectory must carry its current forcing, even
            # after a separate input export to another directory.
            if self.inputs.artifact_store is not self.results.artifact_store:
                for key in self.inputs:
                    self.inputs.input_series.save(key, self.results.artifact_store)
            for key in keys:
                self.results.output_series.save(key, self.results.artifact_store)

    def __repr__(self):
        """Summarize the live simulation for interactive sessions."""
        inputs = ", ".join(sorted(self.inputs)) or "none"
        outputs = ", ".join(sorted(self.outputs)) or "none"
        return (
            f"Simulation(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, current_time={float(self.current_time):g}, "
            f"inputs=[{inputs}], outputs=[{outputs}], "
            f"simulation_directory={self.simulation_directory!r})"
        )

    @property
    def response(self):
        """Return the fixed response for conductance at current time."""
        return self.response_at_time(self.current_time)

    def response_at_time(self, time, *, interpolation=False):
        """Return a fixed response without changing simulation time."""
        series = self.results.input_series
        self._response = ElectrodynamicResponse.from_conductance(
            self.geometry,
            self.config,
            series.get_field_space("conductance"),
            series.get_entry("conductance", time, interpolation),
            previous=self._response,
        )
        return self._response

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        simulation_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        geometry=None,
    ):
        """Construct a simulation from a normalized configuration."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation.from_config requires a SimulationConfig.")
        simulation = cls.__new__(cls)
        simulation.inputs = InputPreparation.from_config(
            config,
            input_directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            geometry=geometry,
        )
        simulation._open_simulation_runtime()
        return simulation

    @classmethod
    def from_directory(
        cls, simulation_directory, *, artifact_storage="auto", operator_cache_directory=None
    ):
        """Reopen a trajectory with its saved physical configuration."""
        simulation = cls.__new__(cls)
        simulation.inputs = InputPreparation.from_directory(
            simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
        )
        simulation._open_simulation_runtime()
        return simulation

    @classmethod
    def from_geometry(
        cls, geometry, *, simulation_directory=None, artifact_storage="auto", **settings
    ):
        """Reuse geometry with independent inputs and outputs."""
        simulation = cls.__new__(cls)
        simulation.inputs = InputPreparation.from_geometry(
            geometry,
            input_directory=simulation_directory,
            artifact_storage=artifact_storage,
            **settings,
        )
        simulation._open_simulation_runtime()
        return simulation

    @classmethod
    def from_inputs(
        cls,
        inputs: InputPreparation,
        *,
        simulation_directory=None,
        artifact_storage="auto",
        **settings,
    ):
        """Copy prepared inputs into an independent simulation.

        Inherit the preparation's configuration and apply explicit
        simulation-setting overrides. Coefficient spaces, physical
        coordinates and time origin must remain compatible. Source
        datasets are loaded and copied once; later edits or deletion
        of the source package cannot change this trajectory.
        """
        if not isinstance(inputs, InputPreparation):
            raise TypeError("Simulation.from_inputs requires an InputPreparation.")
        config = replace(inputs.config, **settings)
        validate_prepared_input_compatibility(inputs.config, config, input_datasets=tuple(inputs))
        simulation = cls.from_config(
            config,
            simulation_directory=simulation_directory,
            artifact_storage=artifact_storage,
            geometry=inputs.geometry,
        )
        if simulation.inputs or simulation.outputs:
            raise ValueError("Use an empty destination when constructing from prepared inputs.")
        simulation.results.input_series.datasets.update(
            {key: dataset.compute().copy(deep=True) for key, dataset in inputs.items()}
        )
        if simulation.results.artifact_store is not None:
            simulation.save()
        return simulation

    def evolve_to_time(
        self,
        t,
        *,
        dt=None,
        output_interval=None,
        output_times=None,
        samples_per_write=DEFAULT_SAMPLES_PER_WRITE,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        quiet=False,
        initialize_from_equilibrium=True,
        sample_equilibrium=None,
    ):
        """Evolve the inductive solution to ``t`` seconds after ``t0``.

        ``output_interval`` is a spacing in seconds, anchored at t0
        (default 0.1 s). Or, ``output_times`` gives increasing
        seconds after t0; it does not implicitly add the initial state.
        The final time is always retained as a restart checkpoint.

        Euler's ``dt`` (default 0.5 ms) is independent of output times.
        Omit dt for exponential or adaptive integration. SciPy methods
        control physical induced-Br coefficients with ``rtol`` (1e-3)
        and ``atol`` (1e-12 tesla). The internal coordinate conversion
        preserves these bounds. These are not least-squares tolerances.

        ``sample_equilibrium`` also records independent diagnostics.
        Use ``sample_equilibria(times)`` for equilibrium-only output.

        ``samples_per_write`` controls batching of xarray updates and
        optional disk writes. Completed samples are also exposed on
        return or interruption. Input changes occur at their timestamps;
        input values remain held constant between records.
        """
        return self._time_evolution.evolve_to_time(
            t,
            dt=dt,
            output_interval=output_interval,
            output_times=output_times,
            samples_per_write=samples_per_write,
            rtol=rtol,
            atol=atol,
            quiet=quiet,
            initialize_from_equilibrium=initialize_from_equilibrium,
            sample_equilibrium=sample_equilibrium,
        )


__all__ = ["Simulation"]
