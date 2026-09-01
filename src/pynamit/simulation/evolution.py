"""Time evolution, output sampling, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from kompe.math import get_array_module

from pynamit.simulation.electrodynamics import induction
from pynamit.storage.field_time_series import TIME_TOLERANCE_SECONDS

if TYPE_CHECKING:
    from pynamit.simulation.simulation import Simulation

DEFAULT_DT_SECONDS = 5e-4
DEFAULT_STEPS_PER_SAMPLE = 200
DEFAULT_SAMPLES_PER_WRITE = 10


def _positive_integer(value, *, name):
    """Return a positive integer without silent truncation."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= 1.")
    integer = int(value)
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return integer


def _boolean_option(value, *, name):
    """Return a boolean without accepting arbitrary truthy values."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean value.")
    return bool(value)


@dataclass(frozen=True)
class _EvolutionOptions:
    """Validated options for one evolution request."""

    target_time: float
    dt: np.float64
    steps_per_sample: int
    samples_per_write: int
    quiet: bool
    initialize_from_equilibrium: bool
    run_dynamic: bool
    run_equilibrium: bool

    @classmethod
    def from_values(
        cls,
        config,
        *,
        t,
        dt,
        steps_per_sample,
        samples_per_write,
        quiet,
        initialize_from_equilibrium,
        run_dynamic,
        run_equilibrium,
    ):
        """Normalize and validate evolution arguments."""
        if isinstance(t, (bool, np.bool_)):
            raise ValueError("t must be a finite, non-negative simulation time.")
        if isinstance(dt, (bool, np.bool_)):
            raise ValueError("dt must be finite and greater than zero.")
        target_time = float(t)
        dt = np.float64(dt)
        if not np.isfinite(target_time) or target_time < 0.0:
            raise ValueError("t must be a finite, non-negative simulation time.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero.")

        run_dynamic = _boolean_option(run_dynamic, name="run_dynamic")
        if run_equilibrium is None:
            run_equilibrium = config.save_equilibria
        else:
            run_equilibrium = _boolean_option(run_equilibrium, name="run_equilibrium")

        if not run_dynamic and not run_equilibrium:
            raise ValueError("At least one of run_dynamic or run_equilibrium must be True.")

        steps_per_sample = _positive_integer(steps_per_sample, name="steps_per_sample")
        samples_per_write = _positive_integer(samples_per_write, name="samples_per_write")

        return cls(
            target_time=target_time,
            dt=dt,
            steps_per_sample=steps_per_sample,
            samples_per_write=samples_per_write,
            quiet=_boolean_option(quiet, name="quiet"),
            initialize_from_equilibrium=_boolean_option(
                initialize_from_equilibrium, name="initialize_from_equilibrium"
            ),
            run_dynamic=run_dynamic,
            run_equilibrium=run_equilibrium,
        )

    @property
    def step_increment(self) -> int:
        """Return loop-step increment for enabled evolution modes."""
        return 1 if self.run_dynamic else self.steps_per_sample

    @property
    def save_step_interval(self) -> int:
        """Return the step interval between persisted samples."""
        return self.steps_per_sample * self.samples_per_write


class _TimeEvolution:
    """Advance one simulation and retain its exponential propagator."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self._cached_exponential_conductance_fingerprint = None
        self._cached_exponential_dt = None
        self._cached_exponential_propagator = None

    def evolve_to_time(
        self,
        t,
        dt=DEFAULT_DT_SECONDS,
        steps_per_sample=DEFAULT_STEPS_PER_SAMPLE,
        samples_per_write=DEFAULT_SAMPLES_PER_WRITE,
        quiet=False,
        initialize_from_equilibrium=True,
        run_dynamic=True,
        run_equilibrium=None,
    ) -> None:
        """Evolve the associated simulation to a target time."""
        options = _EvolutionOptions.from_values(
            self.simulation.config,
            t=t,
            dt=dt,
            steps_per_sample=steps_per_sample,
            samples_per_write=samples_per_write,
            quiet=quiet,
            initialize_from_equilibrium=initialize_from_equilibrium,
            run_dynamic=run_dynamic,
            run_equilibrium=run_equilibrium,
        )
        dynamic_induced_Br = self._initialize_induced_Br(options)

        if self._saved_outputs_reach_target(options):
            if not options.quiet:
                print(
                    f"Saved output already reaches t = {options.target_time:.2f} s; "
                    "nothing to evolve.",
                    flush=True,
                )
            return

        self._require_forward_checkpoint(options)
        if (
            self.simulation.geometry.main_field.kind != "radial"
            and self.simulation.config.enable_pfac_coupling
        ):
            self.simulation.data.save_boundary_jr_to_gap_Br_matrix_if_missing(
                self.simulation.geometry.boundary_jr_to_gap_Br_matrix, print_info=not options.quiet
            )
        self._evolution_loop(options, dynamic_induced_Br)

    def impose_equilibrium(self, *, time=None, interpolation=True, save=True, quiet=False):
        """Solve and optionally save the instantaneous equilibrium."""
        if time is not None:
            if isinstance(time, (bool, np.bool_)):
                raise ValueError("time must be a finite, non-negative simulation time.")
            imposed_time = float(time)
            if not np.isfinite(imposed_time) or imposed_time < 0.0:
                raise ValueError("time must be a finite, non-negative simulation time.")
            if imposed_time < float(self.simulation.current_time) - TIME_TOLERANCE_SECONDS:
                raise ValueError(
                    f"Cannot impose equilibrium at {imposed_time:g} s before the active "
                    f"checkpoint at {float(self.simulation.current_time):g} s. Start from an "
                    "earlier simulation directory to create a new trajectory."
                )
            self.simulation.current_time = np.float64(imposed_time)

        response = self.simulation.response
        response.activate_inputs_at_time(
            self.simulation.data.input_series,
            self.simulation.current_time,
            interpolation=interpolation,
        )
        E_coeffs_noninductive, boundary_jr_noninductive = response.solve_noninductive_response()
        equilibrium_induced_Br = induction.equilibrium_induced_Br(response, E_coeffs_noninductive)

        if (
            save
            and self.simulation.geometry.main_field.kind != "radial"
            and self.simulation.config.enable_pfac_coupling
        ):
            self.simulation.data.save_boundary_jr_to_gap_Br_matrix_if_missing(
                self.simulation.geometry.boundary_jr_to_gap_Br_matrix, print_info=not quiet
            )
        self._record_output_snapshot(
            "dynamic", equilibrium_induced_Br, E_coeffs_noninductive, boundary_jr_noninductive
        )
        if self.simulation.config.save_equilibria:
            self._record_output_snapshot(
                "equilibrium",
                equilibrium_induced_Br,
                E_coeffs_noninductive,
                boundary_jr_noninductive,
            )

        if save:
            self.simulation.data.output_series.save("dynamic", self.simulation.data.artifact_store)
            if self.simulation.config.save_equilibria:
                self.simulation.data.output_series.save(
                    "equilibrium", self.simulation.data.artifact_store
                )

        if not quiet:
            persisted = " and persisted" if save else ""
            current_time = float(self.simulation.current_time)
            print(f"Imposed{persisted} equilibrium at t = {current_time:.2f} s")

        return equilibrium_induced_Br

    def _require_forward_checkpoint(self, options: _EvolutionOptions) -> None:
        """Reject backfill from a later checkpoint."""
        if float(self.simulation.current_time) <= options.target_time + TIME_TOLERANCE_SECONDS:
            return
        raise ValueError(
            f"Target time {options.target_time:g} s precedes the active checkpoint at "
            f"{float(self.simulation.current_time):g} s, and not all requested outputs already "
            "reach the target. Start from an earlier simulation directory to backfill outputs."
        )

    def _initialize_induced_Br(self, options: _EvolutionOptions):
        """Return initial inductive coefficients."""
        output_datasets = self.simulation.data.output_series.datasets
        if options.run_dynamic and "dynamic" in output_datasets:
            return self._resume_induced_Br(options)
        if options.run_dynamic:
            return self._initialize_new_induced_Br(options)
        if "equilibrium" in output_datasets:
            self.simulation.current_time = np.max(output_datasets["equilibrium"].time.values)
        else:
            self.simulation.current_time = np.float64(0)
        return None

    def _resume_induced_Br(self, options: _EvolutionOptions):
        """Resume induced_Br coefficients from the transient output."""
        if not options.quiet:
            print("Resuming dynamic induced Br from saved output.", flush=True)
        transient_output = self.simulation.data.output_series.datasets["dynamic"]
        self.simulation.current_time = np.max(transient_output.time.values)
        dynamic_induced_Br = self.simulation.data.output_series.get_entry(
            "dynamic", self.simulation.current_time, interpolation=False
        )["induced_Br"]
        return get_array_module().asarray(dynamic_induced_Br)

    def _initialize_new_induced_Br(self, options: _EvolutionOptions):
        """Initialize induced Br from equilibrium or zero."""
        if options.initialize_from_equilibrium:
            if not options.quiet:
                print("Initializing dynamic induced Br from equilibrium.", flush=True)
            self.simulation.response.activate_inputs_at_time(
                self.simulation.data.input_series, self.simulation.current_time
            )
            E_coeffs_noninductive, _ = self.simulation.response.solve_noninductive_response()
            return induction.equilibrium_induced_Br(
                self.simulation.response, E_coeffs_noninductive
            )

        if not options.quiet:
            print("Initializing dynamic induced Br from zero.", flush=True)
        self.simulation.current_time = np.float64(0)
        return get_array_module().zeros(
            self.simulation.data.schema.output_field_spaces["dynamic"]["induced_Br"].index_length
        )

    def _saved_outputs_reach_target(self, options: _EvolutionOptions) -> bool:
        """Return whether requested outputs reach target."""
        requested_outputs = []
        if options.run_dynamic:
            requested_outputs.append("dynamic")
        if options.run_equilibrium:
            requested_outputs.append("equilibrium")
        return bool(requested_outputs) and all(
            self._output_dataset_reaches(dataset_key, options.target_time)
            for dataset_key in requested_outputs
        )

    def _output_dataset_reaches(self, dataset_key: str, target_time: float) -> bool:
        """Return whether one saved output reaches target time."""
        dataset = self.simulation.data.output_series.datasets.get(dataset_key)
        if dataset is None or "time" not in dataset:
            return False
        return float(np.max(dataset.time.values)) >= float(target_time) - TIME_TOLERANCE_SECONDS

    def _evolution_loop(self, options: _EvolutionOptions, dynamic_induced_Br) -> None:
        """Advance through the configured evolution loop."""
        step = 0
        total_steps_estimate = self._total_steps_estimate(options)

        while True:
            remaining_time = options.target_time - float(self.simulation.current_time)
            if 0.0 <= remaining_time <= TIME_TOLERANCE_SECONDS:
                self.simulation.current_time = np.float64(options.target_time)

            self._report_progress(step, total_steps_estimate, options)
            self.simulation.response.activate_inputs_at_time(
                self.simulation.data.input_series, self.simulation.current_time
            )

            E_coeffs_noninductive, boundary_jr_noninductive = (
                self.simulation.response.solve_noninductive_response()
            )
            is_final_step = (
                float(self.simulation.current_time) >= options.target_time - TIME_TOLERANCE_SECONDS
            )
            is_sample_step = is_final_step or step % options.steps_per_sample == 0
            should_save_sample = is_final_step or (
                is_sample_step and step % options.save_step_interval == 0
            )
            equilibrium_induced_Br = self._equilibrium_for_step(
                options, is_sample_step, is_final_step, E_coeffs_noninductive
            )

            if is_sample_step:
                self._sample_outputs(
                    options,
                    dynamic_induced_Br,
                    equilibrium_induced_Br,
                    E_coeffs_noninductive,
                    boundary_jr_noninductive,
                )
                if should_save_sample:
                    self._save_sample_outputs(options)

            if is_final_step:
                break

            step_duration = min(
                float(options.dt) * options.step_increment,
                options.target_time - float(self.simulation.current_time),
            )
            next_time = float(self.simulation.current_time) + step_duration

            if options.run_dynamic:
                dynamic_induced_Br = induction.evolve_induced_Br(
                    self.simulation.response,
                    dynamic_induced_Br,
                    step_duration,
                    E_coeffs_noninductive,
                    equilibrium_induced_Br,
                    poloidal_potential_propagator=(
                        self._exponential_propagator_for_step(step_duration)
                    ),
                )
            self.simulation.current_time = np.float64(next_time)
            step += options.step_increment

    def _total_steps_estimate(self, options: _EvolutionOptions) -> int:
        """Return approximate loop steps for progress output."""
        return max(
            1,
            int(
                np.ceil(
                    max(options.target_time - float(self.simulation.current_time), 0.0)
                    / max(float(options.dt), TIME_TOLERANCE_SECONDS)
                )
            ),
        )

    def _report_progress(
        self, step: int, total_steps_estimate: int, options: _EvolutionOptions
    ) -> None:
        """Print progress at the configured interval."""
        if options.quiet or not (step == 0 or step % options.save_step_interval == 0):
            return
        print(
            f"Evolution step {step}/{total_steps_estimate} "
            f"at t = {float(self.simulation.current_time):.2f} s",
            flush=True,
        )

    def _equilibrium_for_step(self, options, is_sample_step, is_final_step, E_coeffs_noninductive):
        """Return equilibrium coefficients when needed."""
        needs_equilibrium = (
            options.run_dynamic
            and self.simulation.config.integrator == "exponential"
            and not is_final_step
        ) or (options.run_equilibrium and is_sample_step)

        if not needs_equilibrium:
            return None

        return induction.equilibrium_induced_Br(self.simulation.response, E_coeffs_noninductive)

    def _exponential_propagator_for_step(self, dt):
        """Return the cached propagator for this closure and step."""
        if self.simulation.config.integrator != "exponential":
            return None

        conductance_fingerprint = self.simulation.response.conductance_fingerprint
        dt = float(dt)
        same_closure = conductance_fingerprint == self._cached_exponential_conductance_fingerprint
        if not same_closure or dt != self._cached_exponential_dt:
            self._cached_exponential_conductance_fingerprint = conductance_fingerprint
            self._cached_exponential_dt = dt
            feedback_matrix = self.simulation.response.induced_poloidal_potential_feedback_matrix
            self._cached_exponential_propagator = (
                induction.poloidal_potential_exponential_propagator(
                    self.simulation.response, dt, feedback_matrix=feedback_matrix
                )
            )
        return self._cached_exponential_propagator

    def _sample_outputs(
        self,
        options: _EvolutionOptions,
        dynamic_induced_Br,
        equilibrium_induced_Br,
        E_coeffs_noninductive,
        boundary_jr_noninductive,
    ) -> None:
        """Add enabled outputs for the current loop time."""
        if options.run_dynamic:
            self._record_output_snapshot(
                "dynamic", dynamic_induced_Br, E_coeffs_noninductive, boundary_jr_noninductive
            )
        if options.run_equilibrium:
            self._record_output_snapshot(
                "equilibrium",
                equilibrium_induced_Br,
                E_coeffs_noninductive,
                boundary_jr_noninductive,
            )

    def _record_output_snapshot(
        self, key, induced_Br, E_coeffs_noninductive, boundary_jr_noninductive
    ):
        """Append a complete model response to one output stream."""
        response = self.simulation.response
        E_coeffs_induced, boundary_jr_induced = response.solve_induced_response(induced_Br)

        E_coeffs = self.simulation.geometry.horizontal_basis.project_helmholtz_mean_free(
            E_coeffs_noninductive + E_coeffs_induced
        )
        # Each term is obtained from the same gauge-fixed toroidal
        # potential through the discrete surface Laplacian. Preserve
        # that exact range element so a saved physical current can
        # reconstruct the private potential without loss.
        boundary_jr = boundary_jr_noninductive + boundary_jr_induced

        output_data = {
            "induced_Br": induced_Br,
            "boundary_jr": boundary_jr,
            "Phi": self.simulation.geometry.helmholtz_curl_free_potential_operator.matvec(
                E_coeffs
            ),
            "W": self.simulation.geometry.helmholtz_divergence_free_potential_operator.matvec(
                E_coeffs
            ),
        }
        self.simulation.data.output_series.add_entry(
            key, output_data, self.simulation.current_time
        )

    def _save_sample_outputs(self, options: _EvolutionOptions) -> None:
        """Persist enabled output datasets for the current sample."""
        saved_outputs = []
        if options.run_dynamic:
            self.simulation.data.output_series.save("dynamic", self.simulation.data.artifact_store)
            saved_outputs.append("dynamic")

        if options.run_equilibrium:
            self.simulation.data.output_series.save(
                "equilibrium", self.simulation.data.artifact_store
            )
            saved_outputs.append("equilibrium")

        if not options.quiet and saved_outputs:
            print(
                f"Saved {' and '.join(saved_outputs)} at "
                f"t = {float(self.simulation.current_time):.2f} s",
                flush=True,
            )
