"""Live state, explicit checkpoints, and independent diagnostics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from kompe.constants import MU0
from kompe.math import get_array_module

from pynamit import Simulation


@pytest.fixture
def simulation():
    """Use uniform conductance for analytic harmonic relaxation."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        integrator="exponential",
        save_equilibria=False,
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=np.full(grid.size, 5.0), hall=np.full(grid.size, 3.0), grid=grid
    )
    return simulation


def _initial_and_rate(simulation):
    degree = simulation.geometry.poloidal_basis.n
    initial = get_array_module().linspace(1e-9, 2e-9, degree.size)
    rate = (5.0 / 34.0) * (2 * degree + 1) / (MU0 * simulation.config.RI)
    return initial, rate


def test_continuation_ignores_edited_and_deleted_recordings(simulation, monkeypatch):
    """Only the live field determines the next Faraday step."""
    initial, rate = _initial_and_rate(simulation)
    simulation.set_state(initial)
    simulation.evolve_to_time(0.1, quiet=True)
    live = simulation.induced_Br
    simulation.outputs["dynamic"]["SH_induced_Br"].values[:] = 9.0
    assert simulation.induced_Br is live

    def unexpected_read(*args, **kwargs):
        pytest.fail("Continuation must not load its state from recorded output.")

    monkeypatch.setattr(simulation.results.output_series, "get_entry", unexpected_read)
    simulation.evolve_to_time(0.2, quiet=True)
    simulation.outputs.clear()
    simulation.evolve_to_time(0.3, quiet=True)
    np.testing.assert_allclose(simulation.induced_Br, initial * np.exp(-0.3 * rate), rtol=1e-11)
    assert simulation.current_time == 0.3


def test_setting_state_owns_its_array_and_respects_the_requested_time(simulation):
    """An explicit state starts at its own time without recording."""
    initial, rate = _initial_and_rate(simulation)
    simulation.set_state(initial, time=0.2)
    assert simulation.induced_Br is not initial
    expected = np.asarray(initial).copy()
    if isinstance(initial, np.ndarray):
        initial[:] = 0
    np.testing.assert_array_equal(simulation.induced_Br, expected)
    assert not simulation.outputs
    simulation.evolve_to_time(0.3, quiet=True)
    np.testing.assert_allclose(simulation.induced_Br, expected * np.exp(-0.1 * rate), rtol=1e-11)


@pytest.mark.parametrize("time", [-1, np.nan, np.inf, True, [0]])
def test_invalid_state_time_leaves_live_state_unchanged(simulation, time):
    """Reject an invalid clock before replacing either state value."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial, time=0.1)
    live = simulation.induced_Br
    with pytest.raises(ValueError):
        simulation.set_state(initial, time=time)
    assert simulation.current_time == 0.1
    assert simulation.induced_Br is live


@pytest.mark.parametrize("invalid", ["batch", "shape", "nan"])
def test_invalid_state_field_leaves_live_state_unchanged(simulation, invalid):
    """Accept one finite state, not an ensemble or bad shape."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial)
    live = simulation.induced_Br
    values = np.asarray(initial).copy()
    if invalid == "batch":
        values = values[:, None]
    elif invalid == "shape":
        values = values[:-1]
    else:
        values[0] = np.nan
    with pytest.raises(ValueError):
        simulation.set_state(values, time=1)
    assert simulation.current_time == 0
    assert simulation.induced_Br is live


def test_restore_is_explicit_exact_and_can_start_an_independent_branch(simulation):
    """Restore exact checkpoints without changing another branch."""
    initial, rate = _initial_and_rate(simulation)
    simulation.set_state(initial)
    simulation.evolve_to_time(0.2, quiet=True)
    history = simulation.outputs["dynamic"].copy(deep=True)
    with pytest.raises(ValueError, match="new simulation"):
        simulation.restore_state(time=0.1)
    with pytest.raises(ValueError, match="No dynamic checkpoint"):
        simulation.restore_state(time=0.15)
    branch = Simulation.from_inputs(simulation.inputs)
    branch.restore_state(simulation.results, time=0.1)
    assert branch.current_time == 0.1
    assert not branch.outputs
    branch.evolve_to_time(0.3, quiet=True)
    np.testing.assert_allclose(branch.induced_Br, initial * np.exp(-0.3 * rate), rtol=1e-11)
    assert simulation.outputs["dynamic"].identical(history)

    simulation.outputs["dynamic"]["SH_induced_Br"].values[-1] *= 2
    simulation.restore_state()
    np.testing.assert_allclose(
        simulation.induced_Br, 2 * initial * np.exp(-0.2 * rate), rtol=1e-11
    )
    simulation.outputs["dynamic"]["SH_induced_Br"].values[-1] *= 3
    np.testing.assert_allclose(
        simulation.induced_Br, 2 * initial * np.exp(-0.2 * rate), rtol=1e-11
    )


@pytest.mark.parametrize(
    "changes", [{"RI": 7e6}, {"t0": "2021-01-01"}, {"main_field_kind": "dipole"}]
)
def test_restore_checks_physical_coordinates(simulation, changes):
    """Do not reinterpret matching arrays in different coordinates."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial)
    simulation.evolve_to_time(0, quiet=True)
    source = SimpleNamespace(
        config=replace(simulation.config, fac_integration_radii=None, **changes),
        output_series=simulation.results.output_series,
    )
    live = simulation.induced_Br
    with pytest.raises(ValueError, match="do not match"):
        simulation.restore_state(source)
    assert simulation.induced_Br is live


def test_equilibrium_sampling_does_not_advance_or_initialize_live_state(simulation):
    """A diagnostic history is independent of the live trajectory."""
    assert simulation.induced_Br is None
    with pytest.raises(ValueError, match="No dynamic checkpoint"):
        simulation.restore_state()
    simulation.sample_equilibria([0, 0.1], quiet=True)
    assert simulation.current_time == 0
    assert simulation.induced_Br is None
    initial, rate = _initial_and_rate(simulation)
    simulation.set_state(initial)
    live = simulation.induced_Br
    simulation.sample_equilibria([0.2, 0.3], quiet=True)
    assert simulation.current_time == 0
    assert simulation.induced_Br is live
    simulation.evolve_to_time(0.2, quiet=True)
    np.testing.assert_allclose(simulation.induced_Br, initial * np.exp(-0.2 * rate), rtol=1e-11)
    assert simulation.outputs["equilibrium"].time.values[-1] == 0.3


def test_live_field_does_not_alias_cached_equilibrium(simulation):
    """Editing a NumPy state cannot change a reused diagnostic."""
    initial, _ = _initial_and_rate(simulation)
    simulation.evolve_to_time(0, sample_equilibrium=True, quiet=True)
    if isinstance(simulation.induced_Br, np.ndarray):
        simulation.induced_Br[:] = initial
    else:
        simulation.induced_Br = initial
    simulation.evolve_to_time(0.1, sample_equilibrium=True, quiet=True)
    np.testing.assert_array_equal(simulation.outputs["equilibrium"]["SH_induced_Br"], 0)
    assert np.linalg.norm(simulation.induced_Br) > 0


def test_failed_equilibrium_imposition_does_not_move_the_state(simulation, monkeypatch):
    """Changing live time is committed only after a successful solve."""
    from pynamit.simulation.electrodynamics import induction

    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial, time=0.1)
    live = simulation.induced_Br

    def fail(*args):
        raise RuntimeError("Equilibrium solve failed.")

    monkeypatch.setattr(induction, "equilibrium_induced_Br", fail)
    with pytest.raises(RuntimeError, match="Equilibrium solve failed"):
        simulation.set_state(
            simulation.equilibrium_coefficients(time=0.2, interpolation=True)["induced_Br"],
            time=0.2,
        )
        simulation.record_state(save=True)
    assert simulation.current_time == 0.1
    assert simulation.induced_Br is live


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_save_preserves_recordings_until_live_edits_are_recorded(simulation, tmp_path, storage):
    """Saving a history and recording a changed state are distinct."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial)
    simulation.evolve_to_time(0.1, quiet=True)
    recorded = simulation.induced_Br.copy()
    simulation.set_state(2 * recorded)
    simulation.save(tmp_path, artifact_storage=storage)
    reopened = Simulation.from_directory(tmp_path)
    np.testing.assert_array_equal(reopened.induced_Br, recorded)
    np.testing.assert_array_equal(simulation.induced_Br, 2 * recorded)
    simulation.record_state()
    reopened = Simulation.from_directory(tmp_path)
    np.testing.assert_array_equal(reopened.induced_Br, 2 * recorded)


def test_interruption_retains_live_state_beyond_last_recorded_sample(simulation, monkeypatch):
    """A failed update cannot send continuation back to a saved row."""
    initial, rate = _initial_and_rate(simulation)
    simulation.set_state(initial)
    space = simulation.results.schema.input_field_spaces["boundary_jr"]
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(space.shape), time=0.15)
    evolution = simulation._time_evolution
    update = evolution._response_and_forcing
    fail = True

    def fail_once(entries):
        nonlocal fail
        if entries["boundary_jr"] is not None and fail:
            fail = False
            raise RuntimeError("Interrupted at the new input.")
        return update(entries)

    monkeypatch.setattr(evolution, "_response_and_forcing", fail_once)
    with pytest.raises(RuntimeError, match="Interrupted"):
        simulation.evolve_to_time(0.4, output_times=[0, 0.4], quiet=True)
    assert simulation.current_time == 0.15
    np.testing.assert_allclose(simulation.induced_Br, initial * np.exp(-0.15 * rate), rtol=1e-11)
    np.testing.assert_array_equal(simulation.outputs["dynamic"].time, [0])
    simulation.evolve_to_time(0.4, output_times=[0.4], quiet=True)
    np.testing.assert_allclose(simulation.induced_Br, initial * np.exp(-0.4 * rate), rtol=1e-11)


def test_evolution_never_resamples_resolved_input_rows(simulation, monkeypatch):
    """Use selected interval arrays without a second time lookup."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial)

    def unexpected_selection(*args, **kwargs):
        pytest.fail("The interval iterator has already selected these rows.")

    monkeypatch.setattr(simulation.results.input_series, "get_entry", unexpected_selection)
    simulation.evolve_to_time(0.2, quiet=True)
    simulation.evolve_to_time(0.3, quiet=True)


def test_live_output_evaluation_and_explicit_recording(simulation):
    """Distinguish live evaluation from inspecting recorded samples."""
    initial, _ = _initial_and_rate(simulation)
    with pytest.raises(ValueError, match="Initialize"):
        simulation.output_coefficients()
    simulation.set_state(initial)
    live_output = simulation.output_coefficients()
    assert not simulation.outputs
    simulation.evolve_to_time(0, quiet=True)
    stored = simulation.results.output_series.get_entry("dynamic", 0)
    for name, values in live_output.items():
        np.testing.assert_allclose(values, stored[name], atol=1e-18)
    simulation.set_state(2 * initial)
    np.testing.assert_array_equal(simulation.output_coefficients()["induced_Br"], 2 * initial)
    np.testing.assert_array_equal(
        simulation.results.output_series.get_entry("dynamic", 0)["induced_Br"], initial
    )
    simulation.evolve_to_time(0, quiet=True)
    np.testing.assert_array_equal(
        simulation.results.output_series.get_entry("dynamic", 0)["induced_Br"], 2 * initial
    )


def test_new_input_series_cannot_inherit_missing_conductance(simulation):
    """A new package must not inherit the previous one's fields."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial)
    simulation.output_coefficients()
    empty_series = simulation.results.schema.create_input_series(time_origin=simulation.config.t0)
    simulation.inputs.input_series = empty_series
    with pytest.raises(RuntimeError, match="conductance"):
        simulation.output_coefficients()


def test_current_time_roundoff_is_not_backward_evolution(simulation):
    """Different arithmetic can represent the same float64 instant."""
    initial, _ = _initial_and_rate(simulation)
    simulation.set_state(initial, time=3 * 0.1)
    simulation.evolve_to_time(0.3, output_times=[0.3], quiet=True)
    np.testing.assert_array_equal(simulation.induced_Br, initial)
    np.testing.assert_array_equal(simulation.outputs["dynamic"].time, [0.3])
