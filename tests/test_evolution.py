"""Focused tests for evolution scheduling and propagator reuse."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pynamit.simulation.electrodynamics import induction
from pynamit.simulation.runner import SimulationRunner


class _FakeResponse:
    def __init__(self):
        self.m_ind_to_E_df_matrix = np.eye(1)
        self.geometry = SimpleNamespace(main_field=SimpleNamespace(kind="radial"))

    @staticmethod
    def project_scalar_mean_free(values):
        return np.asarray(values)

    @staticmethod
    def activate_inputs_at_time(_input_series, _time):
        return None

    @staticmethod
    def calculate_noninductive_response():
        return np.zeros((2, 1)), np.zeros(1)


class _FakeSimulation:
    def __init__(self, *, integrator="euler"):
        self.config = SimpleNamespace(
            integrator=integrator, save_steady_states=False, enable_pfac_coupling=False
        )
        self.response = _FakeResponse()
        self.response.config = self.config
        self.geometry = self.response.geometry
        self.current_time = np.float64(0.0)
        self.recorded = []
        self.saved = []
        self.run_data = SimpleNamespace(
            input_series=SimpleNamespace(),
            output_series=SimpleNamespace(datasets={}),
            schema=SimpleNamespace(output_field_spaces={"state": SimpleNamespace(index_length=1)}),
            save_output_dataset=lambda key: self.saved.append((key, float(self.current_time))),
        )

    def _record_output_state(self, key, *_values):
        self.recorded.append((key, float(self.current_time)))


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
        ({"t": 1.0, "run_inductive": 1}, "run_inductive"),
        ({"t": 1.0, "run_steady_state": "true"}, "run_steady_state"),
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
    simulation.run_data.output_series = SimpleNamespace(
        datasets={"state": xr.Dataset(coords={"time": [10.0]})},
        get_entry=lambda *_args, **_kwargs: {"m_ind": np.zeros(1)},
    )

    with pytest.raises(ValueError, match="precedes the active checkpoint"):
        SimulationRunner(simulation).evolve_to_time(5.0, run_steady_state=True, quiet=True)


def test_evolution_records_and_saves_exact_off_grid_target(monkeypatch):
    """An off-grid target is still a final checkpoint."""
    simulation = _FakeSimulation()
    monkeypatch.setattr(
        induction,
        "evolve_m_ind",
        lambda _response, m_ind, dt, _forcing, _steady, propagator=None: m_ind + dt,
    )

    runner = SimulationRunner(simulation)
    monkeypatch.setattr(
        runner,
        "_record_output_state",
        lambda key, *_values: simulation.recorded.append((key, float(simulation.current_time))),
    )
    runner.evolve_to_time(
        0.25,
        dt=0.1,
        sampling_step_interval=10,
        saving_sample_interval=10,
        steady_state_initialization=False,
        run_steady_state=False,
        quiet=True,
    )

    assert simulation.current_time == np.float64(0.25)
    assert simulation.recorded == [("state", 0.0), ("state", 0.25)]
    assert simulation.saved == [("state", 0.0), ("state", 0.25)]


def test_exponential_propagator_reused_until_operator_or_dt_changes(monkeypatch):
    """Cache exponentials for one closure operator and dt."""
    simulation = _FakeSimulation(integrator="exponential")
    runner = SimulationRunner(simulation)
    calls = []

    def build_propagator(_response, dt, *, m_ind_to_E_df_matrix):
        calls.append((m_ind_to_E_df_matrix, dt))
        return np.array([[len(calls)]], dtype=float)

    monkeypatch.setattr(induction, "exponential_propagator", build_propagator)

    first = runner._exponential_propagator_for_step(0.1)
    second = runner._exponential_propagator_for_step(0.1)
    third = runner._exponential_propagator_for_step(0.05)
    simulation.response.m_ind_to_E_df_matrix = np.eye(1) * 2.0
    fourth = runner._exponential_propagator_for_step(0.05)

    assert first is second
    assert len(calls) == 3
    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(third, [[2.0]])
    np.testing.assert_allclose(fourth, [[3.0]])
