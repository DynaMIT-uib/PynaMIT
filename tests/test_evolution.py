"""Focused tests for evolution scheduling and propagator reuse."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from kompe.math import as_linear_map, set_backend, use_jax

from pynamit.simulation.electrodynamics import induction
from pynamit.simulation.runner import SimulationRunner


class _FakeResponse:
    def __init__(self):
        self.induced_poloidal_potential_feedback_matrix = np.eye(1)
        self.conductance_fingerprint = "initial"
        self.geometry = SimpleNamespace(main_field=SimpleNamespace(kind="radial"))

    @staticmethod
    def activate_inputs_at_time(_input_series, _time):
        return None

    @staticmethod
    def calculate_noninductive_response():
        return np.zeros((2, 1)), np.zeros(1)


class _FakeSimulation:
    def __init__(self, *, integrator="euler"):
        self.config = SimpleNamespace(
            integrator=integrator, save_equilibria=False, enable_pfac_coupling=False
        )
        self.response = _FakeResponse()
        self.response.config = self.config
        self.geometry = self.response.geometry
        self.current_time = np.float64(0.0)
        self.recorded = []
        self.saved = []

        def save_output(key, _store):
            self.saved.append((key, float(self.current_time)))

        output_series = SimpleNamespace(datasets={}, save=save_output)
        self.data = SimpleNamespace(
            input_series=SimpleNamespace(),
            output_series=output_series,
            artifact_store=SimpleNamespace(),
            schema=SimpleNamespace(
                output_field_spaces={"dynamic": {"induced_Br": SimpleNamespace(index_length=1)}}
            ),
        )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_evolution_rejects_invalid_time_step(dt):
    """Invalid time steps fail before entering the run loop."""
    simulation = _FakeSimulation()
    with pytest.raises(ValueError, match="dt must be finite and greater than zero"):
        SimulationRunner(simulation).evolve_to_time(1.0, dt=dt, quiet=True)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_evolution_rejects_invalid_sample_intervals(value):
    """Sample controls must be positive integers without truncation."""
    simulation = _FakeSimulation()
    with pytest.raises(ValueError, match="sampling_step_interval"):
        SimulationRunner(simulation).evolve_to_time(1.0, sampling_step_interval=value, quiet=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t": True}, "t"),
        ({"t": 1.0, "dt": False}, "dt"),
        ({"t": 1.0, "quiet": "false"}, "quiet"),
        ({"t": 1.0, "run_dynamic": 1}, "run_dynamic"),
        ({"t": 1.0, "run_equilibrium": "true"}, "run_equilibrium"),
    ],
)
def test_evolution_rejects_ambiguous_runtime_option_types(kwargs, match):
    """Reject options with ambiguous truth meaning."""
    simulation = _FakeSimulation()

    with pytest.raises(ValueError, match=match):
        SimulationRunner(simulation).evolve_to_time(**kwargs)


def test_evolution_rejects_backfill_from_later_checkpoint():
    """A later checkpoint cannot generate earlier missing output."""
    simulation = _FakeSimulation()
    simulation.data.output_series = SimpleNamespace(
        datasets={"dynamic": xr.Dataset(coords={"time": [10.0]})},
        get_entry=lambda *_args, **_kwargs: {"induced_Br": np.zeros(1)},
    )

    with pytest.raises(ValueError, match="precedes the active checkpoint"):
        SimulationRunner(simulation).evolve_to_time(5.0, run_equilibrium=True, quiet=True)


def test_evolution_records_and_saves_exact_off_grid_target(monkeypatch):
    """An off-grid target is still a final checkpoint."""
    simulation = _FakeSimulation()

    def advance_by_dt(
        _response, induced_Br, dt, _forcing, _equilibrium, poloidal_potential_propagator=None
    ):
        return induced_Br + dt

    monkeypatch.setattr(induction, "evolve_induced_Br", advance_by_dt)

    runner = SimulationRunner(simulation)
    monkeypatch.setattr(
        runner,
        "_record_output_snapshot",
        lambda key, *_values: simulation.recorded.append((key, float(simulation.current_time))),
    )
    runner.evolve_to_time(
        0.25,
        dt=0.1,
        sampling_step_interval=10,
        write_sample_interval=10,
        initialize_from_equilibrium=False,
        run_equilibrium=False,
        quiet=True,
    )

    assert simulation.current_time == np.float64(0.25)
    assert simulation.recorded == [("dynamic", 0.0), ("dynamic", 0.25)]
    assert simulation.saved == [("dynamic", 0.0), ("dynamic", 0.25)]


def test_exponential_propagator_reused_until_conductance_or_dt_changes(monkeypatch):
    """Cache exponentials for one conductance field and dt."""
    simulation = _FakeSimulation(integrator="exponential")
    runner = SimulationRunner(simulation)
    calls = []

    def build_propagator(_response, dt, *, feedback_matrix):
        calls.append((feedback_matrix, dt))
        return np.array([[len(calls)]], dtype=float)

    monkeypatch.setattr(induction, "poloidal_potential_exponential_propagator", build_propagator)

    first = runner._exponential_propagator_for_step(0.1)
    second = runner._exponential_propagator_for_step(0.1)
    third = runner._exponential_propagator_for_step(0.05)
    simulation.response.induced_poloidal_potential_feedback_matrix = np.eye(1) * 2.0
    simulation.response.conductance_fingerprint = "changed"
    fourth = runner._exponential_propagator_for_step(0.05)

    assert first is second
    assert len(calls) == 3
    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(third, [[2.0]])
    np.testing.assert_allclose(fourth, [[3.0]])


def test_exponential_propagator_identity_tracks_active_resistance(monkeypatch):
    """Equivalent closures reuse a propagator by exact resistance."""
    simulation = _FakeSimulation(integrator="exponential")
    simulation.response.conductance_fingerprint = "first"
    runner = SimulationRunner(simulation)
    calls = []

    def build_propagator(_response, _dt, *, feedback_matrix):
        calls.append(feedback_matrix)
        return np.array([[len(calls)]], dtype=float)

    monkeypatch.setattr(induction, "poloidal_potential_exponential_propagator", build_propagator)

    first = runner._exponential_propagator_for_step(0.1)
    simulation.response.induced_poloidal_potential_feedback_matrix = np.eye(1) * 2.0
    equivalent = runner._exponential_propagator_for_step(0.1)
    simulation.response.conductance_fingerprint = "second"
    changed = runner._exponential_propagator_for_step(0.1)

    assert equivalent is first
    assert changed is not first
    assert len(calls) == 2


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exponential_step_returns_to_explicit_jax_backend(backend, data_source):
    """Return once to the input backend after the SciPy handoff."""
    import jax.numpy as jnp

    identity = as_linear_map(jnp.eye(2))
    response = SimpleNamespace(
        config=SimpleNamespace(integrator="exponential"),
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=2.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
        ),
        induced_poloidal_potential_feedback_matrix=jnp.asarray([[-1.0, 0.25], [0.0, -2.0]]),
    )

    previous_backend = use_jax()
    try:
        set_backend("numpy")
        propagator = induction.poloidal_potential_exponential_propagator(response, 0.1)
        evolved = induction.evolve_induced_Br(
            response,
            jnp.asarray([1.0, 2.0]),
            0.1,
            jnp.zeros((2, 1)),
            equilibrium=jnp.zeros(2),
            poloidal_potential_propagator=propagator,
        )
    finally:
        set_backend(previous_backend)

    assert "jax" in type(propagator).__module__
    assert "jax" in type(evolved).__module__
