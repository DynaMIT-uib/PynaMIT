"""Batched physical fields preserve causal input changes."""

import numpy as np
import pytest
from kompe import SphericalGrid
from kompe.math import get_array_module

from pynamit import Simulation, SimulationResults
from pynamit.results import OutputEvaluation, evaluate_projected_input, evaluate_simulation_output


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
@pytest.mark.parametrize("interpolation", [False, True])
def test_result_time_blocks_match_single_queries_and_live_edits(
    tmp_path, horizontal_basis_kind, interpolation
):
    """Live/saved batches preserve fields and coefficient edits."""
    simulation = Simulation(
        Nmax=2,
        Mmax=1,
        Ncs=4,
        RM=4 * 6381e3,
        horizontal_basis_kind=horizontal_basis_kind,
        enable_pfac_coupling=False,
    )
    results = simulation.results
    times = np.array([0.0, 0.3, 1.0, 2.0])
    rng = np.random.default_rng(72)
    coefficients = {
        name: rng.normal(size=(4,) + space.shape) * 1e-8
        for name, space in results.schema.output_field_spaces["dynamic"].items()
    }
    results.output_series.add_entries("dynamic", coefficients, times)
    space = results.schema.input_field_spaces["conductance"]
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": rng.normal(size=(4,) + space.shape) * 0.01,
            "log_hall_to_pedersen_ratio": rng.normal(size=(4,) + space.shape) * 0.01,
        },
        time=times,
    )
    space = results.schema.input_field_spaces["boundary_Br"]
    simulation.inputs.set_coefficients("boundary_Br", np.full(space.shape, 1e-7), time=0.3)
    space = results.schema.input_field_spaces["u"]
    simulation.inputs.set_coefficients("u", rng.normal(size=(4,) + space.shape), time=times)
    grid = SphericalGrid(lat=[[30.0], [60.0]], lon=[[0.0, 90.0, 180.0]])
    queries = np.array([2.0, 0.1, 3 * 0.1, 0.3, 1.5, 0.0, 1.5])
    results.save(tmp_path, artifact_storage="netcdf")
    for source in (results, SimulationResults.from_directory(tmp_path)):
        evaluation = OutputEvaluation(source.geometry, grid)
        batch = evaluate_simulation_output(
            source, queries, evaluation=evaluation, interpolation=interpolation
        )
        singles = [
            evaluate_simulation_output(
                source, t, evaluation=evaluation, interpolation=interpolation
            )
            for t in queries
        ]
        for name, values in batch.items():
            assert values.shape == grid.shape + (queries.size,)
            assert isinstance(values, get_array_module().ndarray)
            np.testing.assert_allclose(
                values, np.stack([item[name] for item in singles], axis=-1), rtol=1e-11, atol=1e-13
            )
        for key in ("u", "conductance", "boundary_Br"):
            available = queries[queries >= (0.3 if key == "boundary_Br" else 0.0)]
            block = evaluate_projected_input(
                source, key, available, grid=grid, interpolation=interpolation
            )
            rows = [
                evaluate_projected_input(source, key, t, grid=grid, interpolation=interpolation)
                for t in available
            ]
            for name, values in block.items():
                np.testing.assert_allclose(
                    values, np.stack([row[name] for row in rows], axis=-1), rtol=1e-11, atol=1e-12
                )
        # Direct evaluation also retains multiple batch axes.
        entry = source.output_series.get_entry("dynamic", queries[:4], interpolation)
        entry = {name: value.reshape(value.shape[:-1] + (2, 2)) for name, value in entry.items()}
        boundary = source.input_series.get_entry(
            "boundary_Br", queries[:4], interpolation, fill_value=0.0
        )["boundary_Br"]
        etaP = evaluate_projected_input(
            source, "conductance", queries[:4], grid=grid, interpolation=interpolation
        )["etaP"]
        direct = evaluation.evaluate(
            entry, boundary_Br=boundary.reshape(-1, 2, 2), etaP=etaP.reshape(grid.size, 2, 2)
        )
        for name, values in direct.items():
            np.testing.assert_allclose(
                values,
                np.asarray(batch[name])[..., :4].reshape(grid.size, 2, 2),
                rtol=1e-11,
                atol=1e-13,
            )
        # Cached maps never cache coefficient samples.
        dataset = source.output_series.datasets["dynamic"]
        name = source.output_series.get_data_var_name("dynamic", "induced_Br")
        dataset[name].values[-1] *= 2
        edited = evaluate_simulation_output(
            source, [2.0], evaluation=evaluation, field_names={"induced_Br"}
        )
        np.testing.assert_allclose(
            edited["induced_Br"][..., 0], 2 * np.asarray(batch["induced_Br"])[..., 0], atol=1e-13
        )


def test_missing_conductance_in_a_time_block_is_explicit():
    """Required conductance is never borrowed from a future sample."""
    simulation = Simulation(Nmax=2, Mmax=1, Ncs=4, enable_pfac_coupling=False)
    results = simulation.results
    coefficients = {
        name: np.zeros((2,) + space.shape)
        for name, space in results.schema.output_field_spaces["dynamic"].items()
    }
    results.output_series.add_entries("dynamic", coefficients, [0.0, 1.0])
    space = results.schema.input_field_spaces["conductance"]
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(space.shape),
            "log_hall_to_pedersen_ratio": np.zeros(space.shape),
        },
        time=1.0,
    )
    with pytest.raises(ValueError, match="No conductance"):
        evaluate_simulation_output(results, [0.0, 1.0], field_names={"joule_heating"})
    assert "joule_heating" not in evaluate_simulation_output(results, [0.0, 1.0])
    with pytest.raises(ValueError, match="No 'dynamic' output"):
        evaluate_simulation_output(results, [-1.0, 1.0])
