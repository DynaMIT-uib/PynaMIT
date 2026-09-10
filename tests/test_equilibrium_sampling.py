"""Independent equilibrium queries, batching, and explicit recording."""

import numpy as np
import pytest
from kompe.math import get_array_module

from pynamit import Simulation
from pynamit.simulation.electrodynamics.induction import induced_Br_time_derivative


@pytest.fixture
def simulation():
    """Uniform conductance and known time-dependent SH forcing."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        save_equilibria=False,
    )
    grid = simulation.model_grid
    for time, pedersen in ((0, 5.0), (0.3, 8.0)):
        simulation.inputs.set_conductance(
            pedersen=np.full(grid.size, pedersen),
            hall=np.full(grid.size, 3.0),
            grid=grid,
            time=time,
        )
    n = simulation.geometry.horizontal_basis.coefficient_count
    E = np.stack([np.linspace(0.1, 0.2, n), np.linspace(0.3, 0.4, n)]) * 1e-3
    simulation.inputs.set_coefficients("E_neutral_wind", E, time=0.1)
    simulation.inputs.set_coefficients("E_neutral_wind", 2 * E, time=0.2)
    return simulation


@pytest.mark.parametrize("interpolation", [False, True])
def test_equilibria_preserve_query_order_and_do_not_modify_state(simulation, interpolation):
    """Batch queries retain forcing and conductance changes."""
    xp = get_array_module()
    simulation.set_state(
        xp.ones(simulation.geometry.poloidal_basis.coefficient_count) * 1e-9, time=1.0
    )
    state = simulation.induced_Br
    times = [0.4, 0.15, 0.0, 0.2, 0.1, 0.15, 0.3]
    batch = simulation.equilibrium_coefficients(times, interpolation=interpolation)
    for i, time in enumerate(times):
        single = simulation.equilibrium_coefficients(time, interpolation=interpolation)
        for name in single:
            np.testing.assert_allclose(batch[name][..., i], single[name], rtol=1e-11, atol=1e-18)
        E = simulation.results.input_series.get_entry(
            "E_neutral_wind", time, interpolation, fill_value=0
        )["E_neutral_wind"]
        response = simulation.response_at_time(time, interpolation=interpolation)
        E, _ = response.solve_noninductive_response(E_neutral_wind=E)
        # These uniform-conductance SH solutions are stationary.
        rate = induced_Br_time_derivative(response, single["induced_Br"], E)
        np.testing.assert_allclose(rate, 0, atol=1e-20)
    assert simulation.current_time == 1.0
    assert simulation.induced_Br is state
    assert not simulation.outputs
    np.testing.assert_array_equal(batch["induced_Br"][..., 2], 0)


def test_equilibrium_sampling_can_backfill_without_a_trajectory(simulation):
    """Equilibrium histories have no dynamic checkpoint restriction."""
    simulation.sample_equilibria([0.2, 0.4], samples_per_write=1, quiet=True)
    assert simulation.induced_Br is None and simulation.current_time == 0
    simulation.sample_equilibria([0.0, 0.1, 0.3], samples_per_write=2, quiet=True)
    np.testing.assert_allclose(simulation.outputs["equilibrium"].time, [0, 0.1, 0.2, 0.3, 0.4])
    assert "dynamic" not in simulation.outputs


def test_fixed_conductance_equilibria_share_one_forcing_solve(simulation, monkeypatch):
    """Repeated conductance is one multi-right-hand-side solve."""
    response = simulation.response_at_time(0)
    solve = response.solve_noninductive_response
    shapes = []

    def counted(**forcing):
        shapes.append(forcing["E_neutral_wind"].shape)
        return solve(**forcing)

    monkeypatch.setattr(response, "solve_noninductive_response", counted)
    simulation.equilibrium_coefficients([0, 0.1, 0.15, 0.2])
    assert shapes == [(2, simulation.geometry.horizontal_basis.coefficient_count, 4)]


@pytest.mark.parametrize("times", [[], [[0]], [-1], [np.nan], [True]])
def test_invalid_equilibrium_queries(simulation, times):
    """Reject ambiguous or nonphysical time queries."""
    with pytest.raises(ValueError):
        simulation.equilibrium_coefficients(times)
    with pytest.raises(ValueError):
        simulation.sample_equilibria(times, quiet=True)


def test_record_state_neither_initializes_nor_evolves(simulation, monkeypatch):
    """Recording requires an explicit state and never integrates."""
    with pytest.raises(ValueError, match="Initialize"):
        simulation.record_state()
    Br = simulation.equilibrium_coefficients(0.2)["induced_Br"]
    simulation.set_state(Br, time=0.2)
    monkeypatch.setattr(
        simulation._time_evolution,
        "evolve_to_time",
        lambda *args, **kwargs: pytest.fail("Recording must not evolve."),
    )
    fields = simulation.record_state(save=False)
    assert simulation.current_time == 0.2
    np.testing.assert_array_equal(fields["induced_Br"], Br)
    assert set(simulation.outputs) == {"dynamic"}
