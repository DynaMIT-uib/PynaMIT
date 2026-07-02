"""Evolution loop orchestration for simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pynamit.math.backend import to_jax, use_jax

_FLOAT_ERROR_MARGIN = 1e-6


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
        run_inductive = bool(run_inductive)
        if run_steady_state is None:
            run_steady_state = config.save_steady_states
        else:
            run_steady_state = bool(run_steady_state)

        if not run_inductive and not run_steady_state:
            raise ValueError("At least one of run_inductive or run_steady_state must be True.")

        sampling_step_interval = int(sampling_step_interval)
        saving_sample_interval = int(saving_sample_interval)
        if sampling_step_interval < 1:
            raise ValueError("sampling_step_interval must be >= 1.")
        if saving_sample_interval < 1:
            raise ValueError("saving_sample_interval must be >= 1.")

        return cls(
            target_time=float(t),
            dt=np.float64(dt),
            sampling_step_interval=sampling_step_interval,
            saving_sample_interval=saving_sample_interval,
            quiet=bool(quiet),
            steady_state_initialization=bool(steady_state_initialization),
            run_inductive=run_inductive,
            run_steady_state=run_steady_state,
        )

    @property
    def step_increment(self) -> int:
        """Return loop-step increment for enabled evolution modes."""
        return 1 if self.run_inductive else self.sampling_step_interval

    @property
    def report_step_interval(self) -> int:
        """Return loop-step interval for progress reports and saves."""
        return self.sampling_step_interval * self.saving_sample_interval


class EvolutionRunner:
    """Coordinate time evolution, sampling, and output persistence."""

    def __init__(self, owner):
        self.owner = owner

    @property
    def state(self):
        """Return the owner's current state object."""
        return self.owner.state

    @property
    def current_time(self):
        """Return the owner's current simulation time."""
        return self.owner.current_time

    @current_time.setter
    def current_time(self, value) -> None:
        self.owner.current_time = value

    def evolve_to_time(
        self,
        t,
        dt=np.float64(5e-4),
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        steady_state_initialization=True,
        run_inductive=True,
        run_steady_state=None,
    ) -> None:
        """Evolve the owner to a target time."""
        options = _EvolutionOptions.from_values(
            self.owner.config,
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

        self._run_loop(options, inductive_m_ind)

    def _initialize_run_state(self, options: _EvolutionOptions):
        """Return initial inductive coefficients."""
        output_datasets = self.owner.output_timeseries.datasets
        if options.run_inductive and "state" in output_datasets:
            return self._resume_inductive_state(options)
        if options.run_inductive:
            return self._new_inductive_state(options)
        if "steady_state" in output_datasets:
            self.current_time = np.max(output_datasets["steady_state"].time.values)
        else:
            self.current_time = np.float64(0)
        return None

    def _resume_inductive_state(self, options: _EvolutionOptions):
        """Resume inductive coefficients from saved state output."""
        if not options.quiet:
            print("Resuming inductive state from saved output.", flush=True)
        state_dataset = self.owner.output_timeseries.datasets["state"]
        self.current_time = np.max(state_dataset.time.values)
        inductive_m_ind = self.owner.output_timeseries.get_entry(
            "state", self.current_time, interpolation=False
        )["m_ind"]
        inductive_m_ind = to_jax(inductive_m_ind) if use_jax() else inductive_m_ind
        return self.state.project_scalar_mean_free(inductive_m_ind)

    def _new_inductive_state(self, options: _EvolutionOptions):
        """Build a new inductive state from steady state or zero."""
        if options.steady_state_initialization:
            if not options.quiet:
                print("Initializing inductive state from steady state.", flush=True)
            self.state.update(self.owner.input_timeseries, self.current_time)
            E_coeffs_noind, _ = self.state.calculate_noind_coeffs()
            return self.state.steady_state_m_ind(E_coeffs_noind)

        if not options.quiet:
            print("Initializing inductive state from zero.", flush=True)
        self.current_time = np.float64(0)
        zeros = np.zeros(self.owner.output_field_spaces["state"].index_length)
        zeros = to_jax(zeros) if use_jax() else zeros
        return self.state.project_scalar_mean_free(zeros)

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
        dataset = self.owner.output_timeseries.datasets.get(dataset_key)
        if dataset is None or "time" not in dataset:
            return False
        return float(np.max(dataset.time.values)) >= float(target_time) - _FLOAT_ERROR_MARGIN

    def _run_loop(self, options: _EvolutionOptions, inductive_m_ind) -> None:
        """Run the configured evolution loop."""
        step = 0
        total_steps_estimate = self._total_steps_estimate(options)

        while True:
            self._report_progress(step, total_steps_estimate, options)
            self.state.update(self.owner.input_timeseries, self.current_time)

            E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
            is_sample_step = step % options.sampling_step_interval == 0
            should_save_sample = is_sample_step and step % options.report_step_interval == 0
            steady_state_m_ind = self._steady_state_for_step(
                options, is_sample_step, E_coeffs_noind
            )

            if is_sample_step:
                self._sample_outputs(
                    options, inductive_m_ind, steady_state_m_ind, E_coeffs_noind, m_imp_noind
                )
                if should_save_sample:
                    self._save_sample_outputs(options)

            next_time = self.current_time + options.dt * options.step_increment

            if next_time > options.target_time + _FLOAT_ERROR_MARGIN:
                if not options.quiet:
                    print("\n\n")
                break

            if options.run_inductive:
                if not options.quiet and self.owner.config.integrator == "exponential":
                    print("  Applying dense exponential induction step.", flush=True)
                inductive_m_ind = self.state.evolve_m_ind(
                    inductive_m_ind, options.dt, E_coeffs_noind, steady_state_m_ind
                )
            self.current_time = next_time
            step += options.step_increment

    def _total_steps_estimate(self, options: _EvolutionOptions) -> int:
        """Return approximate loop steps for progress output."""
        return max(
            1,
            int(
                np.ceil(
                    max(options.target_time - float(self.current_time), 0.0)
                    / max(float(options.dt) * options.step_increment, _FLOAT_ERROR_MARGIN)
                )
            ),
        )

    def _report_progress(
        self, step: int, total_steps_estimate: int, options: _EvolutionOptions
    ) -> None:
        """Print progress at the configured interval."""
        if options.quiet or not (step == 0 or step % options.report_step_interval == 0):
            return
        print(
            f"Evolution step {step}/{total_steps_estimate} "
            f"at t = {float(self.current_time):.2f} s{_maxrss_label()}",
            flush=True,
        )

    def _steady_state_for_step(self, options, is_sample_step, E_coeffs_noind):
        """Return steady-state coefficients when needed."""
        needs_steady_state = (
            options.run_inductive and self.owner.config.integrator == "exponential"
        ) or (options.run_steady_state and is_sample_step)

        if not needs_steady_state:
            return None

        if not options.quiet and self.owner.config.integrator == "exponential":
            print("  Solving steady state required by exponential integrator.", flush=True)
        return self.state.steady_state_m_ind(E_coeffs_noind)

    def _sample_outputs(
        self,
        options: _EvolutionOptions,
        inductive_m_ind,
        steady_state_m_ind,
        E_coeffs_noind,
        m_imp_noind,
    ) -> None:
        """Add enabled outputs for the current loop time."""
        if options.run_inductive:
            self.owner.add_state_to_timeseries(
                "state", inductive_m_ind, E_coeffs_noind, m_imp_noind
            )
        if options.run_steady_state:
            self.owner.add_state_to_timeseries(
                "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind
            )

    def _save_sample_outputs(self, options: _EvolutionOptions) -> None:
        """Persist enabled output datasets for the current sample."""
        saved_outputs = []
        if options.run_inductive:
            self.owner.data.save_output_dataset("state")
            saved_outputs.append("state")

        if options.run_steady_state:
            self.owner.data.save_output_dataset("steady_state")
            saved_outputs.append("steady state")

        if not options.quiet and saved_outputs:
            print(
                "Saved {} at t = {:.2f} s{}".format(
                    " and ".join(saved_outputs), float(self.current_time), _maxrss_label()
                ),
                flush=True,
            )


__all__ = ["EvolutionRunner"]
