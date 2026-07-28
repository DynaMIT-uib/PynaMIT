"""Simulation execution, sampling, and persistence orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pynamit.math.backend import to_jax, to_numpy, use_jax
from pynamit.simulation.electrodynamics import induction
from pynamit.storage.field_time_series import TIME_TOLERANCE_SECONDS

if TYPE_CHECKING:
    from pynamit.simulation.api import Simulation


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


def _maxrss_label():
    """Return a compact max-RSS label when the platform exposes it."""
    try:
        import resource
        import sys

        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return ""
    if not np.isfinite(maxrss) or maxrss <= 0.0:
        return ""
    mib = maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else maxrss / 1024.0
    return f", max RSS ~{mib:.0f} MiB"


@dataclass(frozen=True)
class _EvolutionOptions:
    """Validated options for one evolution run."""

    target_time: float
    dt: np.float64
    sampling_step_interval: int
    saving_sample_interval: int
    quiet: bool
    steady_state_initialization: bool
    run_inductive: bool
    run_steady_state: bool

    @classmethod
    def from_values(
        cls,
        config,
        *,
        t,
        dt,
        sampling_step_interval,
        saving_sample_interval,
        quiet,
        steady_state_initialization,
        run_inductive,
        run_steady_state,
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

        run_inductive = _boolean_option(run_inductive, name="run_inductive")
        if run_steady_state is None:
            run_steady_state = config.save_steady_states
        else:
            run_steady_state = _boolean_option(run_steady_state, name="run_steady_state")

        if not run_inductive and not run_steady_state:
            raise ValueError("At least one of run_inductive or run_steady_state must be True.")

        sampling_step_interval = _positive_integer(
            sampling_step_interval, name="sampling_step_interval"
        )
        saving_sample_interval = _positive_integer(
            saving_sample_interval, name="saving_sample_interval"
        )

        return cls(
            target_time=target_time,
            dt=dt,
            sampling_step_interval=sampling_step_interval,
            saving_sample_interval=saving_sample_interval,
            quiet=_boolean_option(quiet, name="quiet"),
            steady_state_initialization=_boolean_option(
                steady_state_initialization, name="steady_state_initialization"
            ),
            run_inductive=run_inductive,
            run_steady_state=run_steady_state,
        )

    @property
    def step_increment(self) -> int:
        """Return loop-step increment for enabled evolution modes."""
        return 1 if self.run_inductive else self.sampling_step_interval

    @property
    def save_step_interval(self) -> int:
        """Return the step interval between persisted samples."""
        return self.sampling_step_interval * self.saving_sample_interval


class SimulationRunner:
    """Coordinate simulation execution, sampling, and persistence."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self._cached_exponential_operator = None
        self._cached_exponential_conductance_fingerprint = None
        self._cached_exponential_dt = None
        self._cached_exponential_propagator = None

    @staticmethod
    def normalize_evolution_options(config, **kwargs) -> _EvolutionOptions:
        """Validate options without constructing a simulation."""
        return _EvolutionOptions.from_values(config, **kwargs)

    def evolve_to_time(
        self,
        t,
        dt=5e-4,
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        steady_state_initialization=True,
        run_inductive=True,
        run_steady_state=None,
    ) -> None:
        """Evolve the associated simulation to a target time."""
        options = self.normalize_evolution_options(
            self.simulation.config,
            t=t,
            dt=dt,
            sampling_step_interval=sampling_step_interval,
            saving_sample_interval=saving_sample_interval,
            quiet=quiet,
            steady_state_initialization=steady_state_initialization,
            run_inductive=run_inductive,
            run_steady_state=run_steady_state,
        )
        inductive_m_ind = self._initialize_run_state(options)

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
            self.simulation.run_data.save_pfac_matrix_if_missing(
                self.simulation.geometry.pfac_coupling_matrix, print_info=not options.quiet
            )
        self._run_loop(options, inductive_m_ind)

    def impose_steady_state(self, *, time=None, interpolation=True, save=True, quiet=False):
        """Solve and optionally persist the steady state at one time."""
        if time is not None:
            if isinstance(time, (bool, np.bool_)):
                raise ValueError("time must be a finite, non-negative simulation time.")
            imposed_time = float(time)
            if not np.isfinite(imposed_time) or imposed_time < 0.0:
                raise ValueError("time must be a finite, non-negative simulation time.")
            if imposed_time < float(self.simulation.current_time) - TIME_TOLERANCE_SECONDS:
                raise ValueError(
                    f"Cannot impose a state at {imposed_time:g} s before the active "
                    f"checkpoint at {float(self.simulation.current_time):g} s. Start from an "
                    "earlier run directory to create a new trajectory."
                )
            self.simulation.current_time = np.float64(imposed_time)

        response = self.simulation.response
        response.activate_inputs_at_time(
            self.simulation.run_data.input_series,
            self.simulation.current_time,
            interpolation=interpolation,
        )
        E_coeffs_noninductive, m_imp_noninductive = response.calculate_noninductive_response()
        steady_state_m_ind = induction.steady_state_m_ind(response, E_coeffs_noninductive)

        if (
            save
            and self.simulation.geometry.main_field.kind != "radial"
            and self.simulation.config.enable_pfac_coupling
        ):
            self.simulation.run_data.save_pfac_matrix_if_missing(
                self.simulation.geometry.pfac_coupling_matrix, print_info=not quiet
            )
        self._record_output_state(
            "state", steady_state_m_ind, E_coeffs_noninductive, m_imp_noninductive
        )
        if self.simulation.config.save_steady_states:
            self._record_output_state(
                "steady_state", steady_state_m_ind, E_coeffs_noninductive, m_imp_noninductive
            )

        if save:
            self.simulation.run_data.save_output_dataset("state")
            if self.simulation.config.save_steady_states:
                self.simulation.run_data.save_output_dataset("steady_state")

        if not quiet:
            persisted = " and persisted" if save else ""
            current_time = float(self.simulation.current_time)
            print(f"Imposed{persisted} steady state at t = {current_time:.2f} s")

        return steady_state_m_ind

    def _require_forward_checkpoint(self, options: _EvolutionOptions) -> None:
        """Reject backfill from a later checkpoint."""
        if float(self.simulation.current_time) <= options.target_time + TIME_TOLERANCE_SECONDS:
            return
        raise ValueError(
            f"Target time {options.target_time:g} s precedes the active checkpoint at "
            f"{float(self.simulation.current_time):g} s, and not all requested outputs already "
            "reach the target. Start from an earlier run directory to backfill outputs."
        )

    def _initialize_run_state(self, options: _EvolutionOptions):
        """Return initial inductive coefficients."""
        output_datasets = self.simulation.run_data.output_series.datasets
        if options.run_inductive and "state" in output_datasets:
            return self._resume_inductive_state(options)
        if options.run_inductive:
            return self._new_inductive_state(options)
        if "steady_state" in output_datasets:
            self.simulation.current_time = np.max(output_datasets["steady_state"].time.values)
        else:
            self.simulation.current_time = np.float64(0)
        return None

    def _resume_inductive_state(self, options: _EvolutionOptions):
        """Resume inductive coefficients from saved state output."""
        if not options.quiet:
            print("Resuming inductive state from saved output.", flush=True)
        state_dataset = self.simulation.run_data.output_series.datasets["state"]
        self.simulation.current_time = np.max(state_dataset.time.values)
        inductive_m_ind = self.simulation.run_data.output_series.get_entry(
            "state", self.simulation.current_time, interpolation=False
        )["m_ind"]
        return to_jax(inductive_m_ind) if use_jax() else inductive_m_ind

    def _new_inductive_state(self, options: _EvolutionOptions):
        """Build a new inductive state from steady state or zero."""
        if options.steady_state_initialization:
            if not options.quiet:
                print("Initializing inductive state from steady state.", flush=True)
            self.simulation.response.activate_inputs_at_time(
                self.simulation.run_data.input_series, self.simulation.current_time
            )
            E_coeffs_noninductive, _ = self.simulation.response.calculate_noninductive_response()
            return induction.steady_state_m_ind(self.simulation.response, E_coeffs_noninductive)

        if not options.quiet:
            print("Initializing inductive state from zero.", flush=True)
        self.simulation.current_time = np.float64(0)
        zeros = np.zeros(
            self.simulation.run_data.schema.output_field_spaces["state"]["m_ind"].index_length
        )
        return to_jax(zeros) if use_jax() else zeros

    def _saved_outputs_reach_target(self, options: _EvolutionOptions) -> bool:
        """Return whether requested outputs reach target."""
        requested_outputs = []
        if options.run_inductive:
            requested_outputs.append("state")
        if options.run_steady_state:
            requested_outputs.append("steady_state")
        return bool(requested_outputs) and all(
            self._output_dataset_reaches(dataset_key, options.target_time)
            for dataset_key in requested_outputs
        )

    def _output_dataset_reaches(self, dataset_key: str, target_time: float) -> bool:
        """Return whether one saved output reaches target time."""
        dataset = self.simulation.run_data.output_series.datasets.get(dataset_key)
        if dataset is None or "time" not in dataset:
            return False
        return float(np.max(dataset.time.values)) >= float(target_time) - TIME_TOLERANCE_SECONDS

    def _run_loop(self, options: _EvolutionOptions, inductive_m_ind) -> None:
        """Run the configured evolution loop."""
        step = 0
        total_steps_estimate = self._total_steps_estimate(options)

        while True:
            remaining_time = options.target_time - float(self.simulation.current_time)
            if 0.0 <= remaining_time <= TIME_TOLERANCE_SECONDS:
                self.simulation.current_time = np.float64(options.target_time)

            self._report_progress(step, total_steps_estimate, options)
            self.simulation.response.activate_inputs_at_time(
                self.simulation.run_data.input_series, self.simulation.current_time
            )

            E_coeffs_noninductive, m_imp_noninductive = (
                self.simulation.response.calculate_noninductive_response()
            )
            is_final_step = (
                float(self.simulation.current_time) >= options.target_time - TIME_TOLERANCE_SECONDS
            )
            is_sample_step = is_final_step or step % options.sampling_step_interval == 0
            should_save_sample = is_final_step or (
                is_sample_step and step % options.save_step_interval == 0
            )
            steady_state_m_ind = self._steady_state_for_step(
                options, is_sample_step, is_final_step, E_coeffs_noninductive
            )

            if is_sample_step:
                self._sample_outputs(
                    options,
                    inductive_m_ind,
                    steady_state_m_ind,
                    E_coeffs_noninductive,
                    m_imp_noninductive,
                )
                if should_save_sample:
                    self._save_sample_outputs(options)

            if is_final_step:
                if not options.quiet:
                    print("\n\n")
                break

            step_duration = min(
                float(options.dt) * options.step_increment,
                options.target_time - float(self.simulation.current_time),
            )
            next_time = float(self.simulation.current_time) + step_duration

            if options.run_inductive:
                if not options.quiet and self.simulation.config.integrator == "exponential":
                    print("  Applying dense exponential induction step.", flush=True)
                inductive_m_ind = induction.evolve_m_ind(
                    self.simulation.response,
                    inductive_m_ind,
                    step_duration,
                    E_coeffs_noninductive,
                    steady_state_m_ind,
                    propagator=self._exponential_propagator_for_step(step_duration),
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
            f"at t = {float(self.simulation.current_time):.2f} s{_maxrss_label()}",
            flush=True,
        )

    def _steady_state_for_step(
        self, options, is_sample_step, is_final_step, E_coeffs_noninductive
    ):
        """Return steady-state coefficients when needed."""
        needs_steady_state = (
            options.run_inductive
            and self.simulation.config.integrator == "exponential"
            and not is_final_step
        ) or (options.run_steady_state and is_sample_step)

        if not needs_steady_state:
            return None

        if not options.quiet and self.simulation.config.integrator == "exponential":
            print("  Solving steady state required by exponential integrator.", flush=True)
        return induction.steady_state_m_ind(self.simulation.response, E_coeffs_noninductive)

    def _exponential_propagator_for_step(self, dt):
        """Return the cached propagator for this closure and step."""
        if self.simulation.config.integrator != "exponential":
            return None

        operator = self.simulation.response.m_ind_feedback_matrix
        conductance_fingerprint = getattr(
            self.simulation.response, "conductance_fingerprint", None
        )
        dt = float(dt)
        same_closure = (
            conductance_fingerprint == self._cached_exponential_conductance_fingerprint
            if conductance_fingerprint is not None
            else operator is self._cached_exponential_operator
        )
        if not same_closure or dt != self._cached_exponential_dt:
            self._cached_exponential_operator = operator
            self._cached_exponential_conductance_fingerprint = conductance_fingerprint
            self._cached_exponential_dt = dt
            self._cached_exponential_propagator = induction.exponential_propagator(
                self.simulation.response, dt, m_ind_feedback_matrix=operator
            )
        return self._cached_exponential_propagator

    def _sample_outputs(
        self,
        options: _EvolutionOptions,
        inductive_m_ind,
        steady_state_m_ind,
        E_coeffs_noninductive,
        m_imp_noninductive,
    ) -> None:
        """Add enabled outputs for the current loop time."""
        if options.run_inductive:
            self._record_output_state(
                "state", inductive_m_ind, E_coeffs_noninductive, m_imp_noninductive
            )
        if options.run_steady_state:
            self._record_output_state(
                "steady_state", steady_state_m_ind, E_coeffs_noninductive, m_imp_noninductive
            )

    def _record_output_state(self, key, m_ind, E_coeffs_noninductive, m_imp_noninductive):
        """Append a complete model response to one output stream."""
        response = self.simulation.response
        E_coeffs_inductive, m_imp_inductive = response.calculate_inductive_response(m_ind)

        E_coeffs = response.project_helmholtz_mean_free(E_coeffs_noninductive + E_coeffs_inductive)
        m_imp = response.project_surface_scalar_mean_free(m_imp_noninductive + m_imp_inductive)

        state_data = {
            "m_ind": to_numpy(m_ind),
            "m_imp": to_numpy(m_imp),
            "Phi": to_numpy(
                self.simulation.geometry.helmholtz_curl_free_potential_operator.matvec(E_coeffs)
            ),
            "W": to_numpy(
                self.simulation.geometry.helmholtz_divergence_free_potential_operator.matvec(
                    E_coeffs
                )
            ),
        }
        self.simulation.run_data.add_output_entry(
            key, state_data, time=self.simulation.current_time
        )

    def _save_sample_outputs(self, options: _EvolutionOptions) -> None:
        """Persist enabled output datasets for the current sample."""
        saved_outputs = []
        if options.run_inductive:
            self.simulation.run_data.save_output_dataset("state")
            saved_outputs.append("state")

        if options.run_steady_state:
            self.simulation.run_data.save_output_dataset("steady_state")
            saved_outputs.append("steady state")

        if not options.quiet and saved_outputs:
            print(
                "Saved {} at t = {:.2f} s{}".format(
                    " and ".join(saved_outputs),
                    float(self.simulation.current_time),
                    _maxrss_label(),
                ),
                flush=True,
            )
