"""Time evolution, output sampling, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
from kompe.math import get_array_module

from pynamit.simulation.electrodynamics import induction
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.storage.field_time_series import time_roundoff

if TYPE_CHECKING:
    from pynamit.simulation.simulation import Simulation

DEFAULT_DT_SECONDS = induction.DEFAULT_DT_SECONDS
DEFAULT_OUTPUT_INTERVAL_SECONDS = 0.1
DEFAULT_RTOL = induction.DEFAULT_RTOL
DEFAULT_ATOL = induction.DEFAULT_ATOL
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
    """Validated output times and integration controls."""

    target_time: float
    dt: float | None
    output_interval: float | None
    output_times: tuple[float, ...] | None
    samples_per_write: int
    rtol: float
    atol: float
    quiet: bool
    initialize_from_equilibrium: bool
    sample_equilibrium: bool

    @classmethod
    def from_values(
        cls,
        config,
        *,
        t,
        dt,
        output_interval,
        output_times,
        samples_per_write,
        rtol,
        atol,
        quiet,
        initialize_from_equilibrium,
        sample_equilibrium,
    ):
        """Normalize public options before numerical work."""
        if isinstance(t, (bool, np.bool_)):
            raise ValueError("t must be a finite, non-negative simulation time.")
        target_time = float(t)
        if not np.isfinite(target_time) or target_time < 0:
            raise ValueError("t must be a finite, non-negative simulation time.")
        if config.integrator == "euler":
            dt = DEFAULT_DT_SECONDS if dt is None else dt
            if isinstance(dt, (bool, np.bool_)) or not np.isfinite(dt) or dt <= 0:
                raise ValueError("dt must be finite and greater than zero.")
            dt = float(dt)
        elif dt is not None:
            raise ValueError("dt controls Euler stepping; omit it for other integrators.")
        for name, tolerance in (("rtol", rtol), ("atol", atol)):
            if (
                isinstance(tolerance, (bool, np.bool_))
                or not np.isfinite(tolerance)
                or tolerance <= 0
            ):
                raise ValueError(f"{name} must be finite and greater than zero.")

        if output_times is not None:
            if output_interval is not None:
                raise ValueError("Specify output_interval or output_times, not both.")
            times = np.asarray(output_times)
            if times.dtype.kind == "b":
                raise ValueError("output_times must contain numerical times, not booleans.")
            times = np.asarray(times, dtype=float)
            if (
                times.ndim != 1
                or not np.all(np.isfinite(times))
                or np.any(times < 0)
                or np.any(times > target_time)
                or np.any(np.diff(times) <= time_roundoff(times[1:]))
            ):
                raise ValueError("output_times must be increasing times between zero and t.")
            output_times = tuple(map(float, times))
        else:
            output_interval = (
                DEFAULT_OUTPUT_INTERVAL_SECONDS if output_interval is None else output_interval
            )
            if (
                isinstance(output_interval, (bool, np.bool_))
                or not np.isfinite(output_interval)
                or output_interval <= 0
            ):
                raise ValueError("output_interval must be finite and greater than zero.")
            output_interval = float(output_interval)

        sample_equilibrium = (
            config.save_equilibria
            if sample_equilibrium is None
            else _boolean_option(sample_equilibrium, name="sample_equilibrium")
        )
        return cls(
            target_time=target_time,
            dt=dt,
            output_interval=output_interval,
            output_times=output_times,
            samples_per_write=_positive_integer(samples_per_write, name="samples_per_write"),
            rtol=float(rtol),
            atol=float(atol),
            quiet=_boolean_option(quiet, name="quiet"),
            initialize_from_equilibrium=_boolean_option(
                initialize_from_equilibrium, name="initialize_from_equilibrium"
            ),
            sample_equilibrium=sample_equilibrium,
        )


class _TimeEvolution:
    """Schedule forcing, evolution, and output for one simulation."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self._input_series = None
        self._input_values = {}
        self._forcing_values = {}
        self._forcing_response = None
        self._noninductive_response = None
        self._equilibrium_induced_Br = None
        self._stepper_setup = None
        self._stepper_forcing = None
        self._stepper = None
        self._output_samples = {}

    def _response_and_forcing(self, entries):
        """Update the physical response from already selected inputs."""
        simulation = self.simulation
        series = simulation.results.input_series
        previous = self._input_values if series is self._input_series else {}
        response = self._forcing_response
        if (
            response is None
            or "conductance" not in previous
            or entries["conductance"] is not previous["conductance"]
        ):
            response = ElectrodynamicResponse.from_conductance(
                simulation.geometry,
                simulation.config,
                series.get_field_space("conductance"),
                entries["conductance"],
                previous=simulation._response,
            )
        changed = response is not self._forcing_response
        forcing_values = self._forcing_values.copy() if series is self._input_series else {}
        for key, variables in series.variables.items():
            if key == "conductance":
                continue
            values = entries[key]
            if key in previous and values is previous[key]:
                continue
            changed = True
            if values is None:
                for name in variables:
                    forcing_values.pop(name, None)
                continue
            # Storage already owns, shapes, and gauges these fields.
            xp = get_array_module()
            forcing_values.update({name: xp.asarray(values[name]) for name in variables})
        self._input_series = series
        self._input_values = entries
        self._forcing_values = forcing_values
        self._forcing_response = simulation._response = response
        if changed:
            self._noninductive_response = None
            self._equilibrium_induced_Br = None
        if self._noninductive_response is None:
            self._noninductive_response = response.solve_noninductive_response(
                **self._forcing_values
            )
        return response, *self._noninductive_response

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
    ) -> None:
        """Evolve on input intervals and sample output."""
        simulation = self.simulation
        config = simulation.config
        output_series = simulation.results.output_series
        options = _EvolutionOptions.from_values(
            config,
            t=t,
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
        target_time = options.target_time
        requested_outputs = ["dynamic"] + (["equilibrium"] if options.sample_equilibrium else [])
        dynamic_Br = simulation.induced_Br
        start_time = float(simulation.current_time)
        if options.output_times is None:
            interval = options.output_interval
            first = max(0, int(np.ceil(start_time / interval)))
            last = int(np.floor(target_time / interval))
            times = np.arange(first, last + 1, dtype=float) * interval
        else:
            times = np.asarray(options.output_times, dtype=float)
            for time in times[times < start_time - time_roundoff(start_time)]:
                for key in requested_outputs:
                    dataset = output_series.datasets.get(key)
                    if dataset is None or not np.any(
                        np.abs(dataset.time.values - time) <= time_roundoff(time)
                    ):
                        raise ValueError(
                            f"Cannot add output at {time:g} s before the active checkpoint "
                            f"at {start_time:g} s; start from an earlier trajectory."
                        )
            times = times[times >= start_time - time_roundoff(start_time)]
        # The final state is always retained as the restart checkpoint.
        # Explicit output_times do not implicitly add an initial sample.
        times = times[times < target_time - time_roundoff(target_time)]
        times = np.append(times, target_time)
        if target_time < start_time - time_roundoff(start_time) and all(
            key in output_series.datasets
            and float(output_series.datasets[key].time.values[-1])
            >= target_time - time_roundoff(target_time)
            for key in requested_outputs
        ):
            if not options.quiet:
                print(f"Saved output already reaches t = {target_time:.2f} s; nothing to evolve.")
            return
        if start_time > target_time + time_roundoff(target_time):
            raise ValueError(
                f"Target time {target_time:g} s precedes the active checkpoint at "
                f"{start_time:g} s. Start from an earlier trajectory to backfill outputs."
            )
        series = simulation.results.input_series
        intervals = series.iter_intervals(
            start_time,
            target_time,
            previous=self._input_values if series is self._input_series else None,
        )
        output_index = 0
        recorded = 0
        pending = 0

        def record(time, induced_Br, response, E, jr):
            """Sample solutions and flush complete batches."""
            nonlocal recorded, pending
            sample_times = np.atleast_1d(time)
            self._record_output_snapshot("dynamic", induced_Br, E, jr, response, time=time)
            if options.sample_equilibrium:
                if self._equilibrium_induced_Br is None:
                    self._equilibrium_induced_Br = induction.equilibrium_induced_Br(response, E)
                self._record_output_snapshot(
                    "equilibrium", self._equilibrium_induced_Br, E, jr, response, time=time
                )
            recorded += sample_times.size
            pending += sample_times.size
            if (
                recorded == sample_times.size
                or pending >= options.samples_per_write
                or sample_times[-1] == target_time
            ):
                self._flush_output_samples()
                pending = 0
                simulation._save_output(*requested_outputs)
                if not options.quiet:
                    print(
                        f"Output at t = {sample_times[-1]:.6g} / {target_time:.6g} s", flush=True
                    )

        try:
            for left, right, entries in intervals:
                response, E, jr = self._response_and_forcing(entries)
                if dynamic_Br is None:
                    if options.initialize_from_equilibrium:
                        if not options.quiet:
                            print("Initializing dynamic induced Br from equilibrium.", flush=True)
                        self._equilibrium_induced_Br = induction.equilibrium_induced_Br(
                            response, E
                        )
                        dynamic_Br = get_array_module(self._equilibrium_induced_Br).array(
                            self._equilibrium_induced_Br, copy=True
                        )
                    else:
                        if not options.quiet:
                            print("Initializing dynamic induced Br from zero.", flush=True)
                        dynamic_Br = get_array_module().zeros(
                            simulation.results.schema.output_field_spaces["dynamic"][
                                "induced_Br"
                            ].shape
                        )
                    simulation.induced_Br = dynamic_Br
                # Br is continuous across forcing changes; algebraic
                # fields at a change use the new inputs.
                while output_index < times.size and times[output_index] <= left + time_roundoff(
                    left
                ):
                    record(float(times[output_index]), dynamic_Br, response, E, jr)
                    output_index += 1
                if right <= left:
                    continue
                stop = int(np.searchsorted(times, right - time_roundoff(right), side="left"))
                interior = times[output_index:stop]
                evaluation_times = np.append(interior, right)
                setup = (response, options.dt, options.rtol, options.atol)
                if setup != self._stepper_setup or E is not self._stepper_forcing:
                    self._stepper = induction.build_induction_stepper(
                        response, E, dt=options.dt, rtol=options.rtol, atol=options.atol
                    )
                    self._stepper_setup = setup
                    self._stepper_forcing = E
                samples = self._stepper(
                    dynamic_Br,
                    evaluation_times - left,
                    output_interval=options.output_interval,
                    batch_size=options.samples_per_write,
                )
                index = 0
                for states in samples:
                    count = states.shape[-1]
                    batch_times = evaluation_times[index : index + count]
                    dynamic_Br = states[..., -1]
                    simulation.induced_Br = dynamic_Br
                    simulation.current_time = float(batch_times[-1])
                    sampled_count = min(count, interior.size - index)
                    if sampled_count > 0:
                        record(
                            batch_times[:sampled_count],
                            states[..., :sampled_count],
                            response,
                            E,
                            jr,
                        )
                        output_index += sampled_count
                    index += count
        finally:
            # Flush completed samples without hiding interruptions.
            self._flush_output_samples()

    def _record_output_snapshot(
        self, key, induced_Br, E_coeffs_noninductive, boundary_jr_noninductive, response, *, time
    ):
        """Retain shaped sample blocks and their fixed response."""
        times = np.atleast_1d(time)
        xp = get_array_module(induced_Br, E_coeffs_noninductive, boundary_jr_noninductive)
        Br = induced_Br[..., None] if induced_Br.ndim == 1 else induced_Br
        E = (
            E_coeffs_noninductive[..., None]
            if E_coeffs_noninductive.ndim == 2
            else E_coeffs_noninductive
        )
        jr = (
            boundary_jr_noninductive[..., None]
            if boundary_jr_noninductive.ndim == 1
            else boundary_jr_noninductive
        )
        self._output_samples.setdefault(key, []).append(
            (
                times,
                response,
                xp.broadcast_to(Br, Br.shape[:-1] + (times.size,)),
                xp.broadcast_to(E, E.shape[:-1] + (times.size,)),
                xp.broadcast_to(jr, jr.shape[:-1] + (times.size,)),
            )
        )

    def _flush_output_samples(self):
        """Evaluate batches, then transfer each stream to xarray."""
        for key, samples in self._output_samples.items():
            batches = []
            # Batch states sharing a conductance response, retaining
            # each sample's own noninductive forcing.
            for response, group in groupby(samples, key=lambda sample: sample[1]):
                _, _, Br_samples, E_samples, jr_samples = zip(*group, strict=True)
                xp = get_array_module(Br_samples[0], E_samples[0], jr_samples[0])
                induced_Br = xp.concatenate(Br_samples, axis=-1)
                fields = response.output_coefficients(
                    induced_Br,
                    xp.concatenate(E_samples, axis=-1),
                    xp.concatenate(jr_samples, axis=-1),
                )
                batches.append(
                    {name: xp.moveaxis(values, -1, 0) for name, values in fields.items()}
                )
            rows = batches[0]
            if len(batches) > 1:
                rows = {name: xp.concatenate([batch[name] for batch in batches]) for name in rows}
            times = np.concatenate([sample[0] for sample in samples])
            self.simulation.results.output_series.add_entries(key, rows, times)
        self._output_samples.clear()
