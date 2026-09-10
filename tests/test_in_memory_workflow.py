"""In-memory experiments, persistence, and batched evolution."""

import numpy as np
import pytest
import xarray as xr
from kompe.math import LeastSquaresSolver

from pynamit import InputPreparation, Simulation, SimulationConfig
from pynamit.plotting import PlotData
from pynamit.results import SimulationResults
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.storage import ArtifactStore


@pytest.mark.parametrize("algorithm", LeastSquaresSolver.VALID_SOLVERS)
@pytest.mark.parametrize("kind", ["SH", "CS"])
def test_one_fit_policy_covers_input_preparation_and_evolution(algorithm, kind, monkeypatch):
    """One algorithm and tolerance cover the complete workflow."""
    preparation = InputPreparation(
        Nmax=2,
        Mmax=1,
        Ncs=4,
        main_field_kind="radial",
        horizontal_basis_kind=kind,
        least_squares_solver=algorithm,
        least_squares_tolerance=1e-11,
        least_squares_preconditioner="jacobi" if algorithm in {"lsmr", "cgls"} else None,
    )
    grid = preparation.model_grid
    transform = preparation.geometry.horizontal_transform
    potentials = np.random.default_rng(6).normal(size=(2, transform.basis.coefficient_count))
    wind = np.asarray(transform.synthesize_helmholtz(potentials))
    calls = []
    solve = LeastSquaresSolver.solve

    def record(self, problem, rhs, **kwargs):
        calls.append((self.method, self.tolerance))
        return solve(self, problem, rhs, **kwargs)

    monkeypatch.setattr(LeastSquaresSolver, "solve", record)
    preparation.set_conductance(
        pedersen=np.full(grid.size, 2.0), hall=np.ones(grid.size), grid=grid
    )
    preparation.set_boundary_jr(boundary_jr=1e-6 * np.cos(np.deg2rad(grid.theta)), grid=grid)
    preparation.set_Q_eff_from_neutral_wind(wind[0], wind[1], grid=grid)
    assert calls and set(calls) == {(algorithm, 1e-11)}

    simulation = Simulation.from_inputs(preparation)
    assert simulation.geometry is preparation.geometry
    assert simulation.config.least_squares_solver == algorithm
    assert simulation.response._toroidal_potential_solver.tolerance == 1e-11
    simulation.evolve_to_time(0.0002, dt=0.0002, quiet=True, initialize_from_equilibrium=False)
    assert np.all(np.isfinite(simulation.outputs["dynamic"]["SH_induced_Br"]))


def _prepare_inputs(*, horizontal_basis_kind="SH", integrator="euler", changing=False):
    preparation = InputPreparation.from_config(
        SimulationConfig(
            Nmax=2,
            Mmax=1,
            Ncs=4,
            horizontal_basis_kind=horizontal_basis_kind,
            main_field_kind="radial",
            enable_pfac_coupling=False,
            integrator=integrator,
            least_squares_solver="normal_pinv",
        )
    )
    spaces = preparation.schema.input_field_spaces
    shape = spaces["conductance"].shape
    times = [0.0, 0.0012] if changing else [0.0]
    preparation.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.stack(
                [np.full(shape, i * 0.1) for i in range(len(times))]
            ),
            "log_hall_to_pedersen_ratio": np.zeros((len(times), *shape)),
        },
        time=times,
    )
    pattern = np.linspace(-1e-6, 1e-6, spaces["boundary_jr"].size)
    times = [0.0, 0.0007, 0.0014] if changing else [0.0]
    preparation.set_coefficients(
        "boundary_jr", np.stack([(i + 1) * pattern for i in range(len(times))]), time=times
    )
    return preparation


def test_in_memory_workflow_never_constructs_artifact_storage(monkeypatch):
    """Preparation and evolution do not require a directory."""

    def unexpected_storage(*args, **kwargs):
        pytest.fail("In-memory work must not construct artifact storage.")

    monkeypatch.setattr(ArtifactStore, "__init__", unexpected_storage)
    preparation = _prepare_inputs()
    simulation = Simulation.from_inputs(preparation)
    simulation.evolve_to_time(0.001, dt=0.0002, quiet=True, initialize_from_equilibrium=False)

    assert preparation.input_directory is None
    assert simulation.simulation_directory is None
    assert not isinstance(simulation, InputPreparation)
    assert isinstance(simulation.inputs, InputPreparation)
    assert simulation.outputs["dynamic"].time.values[-1] == 0.001
    assert np.linalg.norm(simulation.outputs["dynamic"]["SH_induced_Br"].values[-1]) > 0


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
@pytest.mark.parametrize("integrator", ["euler", "exponential"])
def test_sample_batches_preserve_every_output_row(horizontal_basis_kind, integrator, monkeypatch):
    """Batching preserves output at forcing and conductance changes."""
    preparation = _prepare_inputs(
        horizontal_basis_kind=horizontal_basis_kind, integrator=integrator, changing=True
    )
    simulation = Simulation.from_inputs(preparation)
    reference = Simulation.from_inputs(preparation)
    series = simulation.results.output_series
    add_entries = series.add_entries
    batches = []

    def record_batch(key, values, times):
        batches.append((key, len(times)))
        return add_entries(key, values, times)

    monkeypatch.setattr(series, "add_entries", record_batch)
    # One write interval crosses both the current change at 0.0007 and
    # the conductance change at 0.0012. Fields use the new inputs.
    times = [0.0, 0.0005, 0.0007, 0.0012, 0.0014, 0.0018, 0.002]
    options = dict(
        dt=0.0002 if integrator == "euler" else None,
        output_times=times,
        initialize_from_equilibrium=False,
        sample_equilibrium=True,
        quiet=True,
    )
    simulation.evolve_to_time(times[-1], samples_per_write=4, **options)
    reference.evolve_to_time(times[-1], samples_per_write=1, **options)
    for key in ("dynamic", "equilibrium"):
        assert [size for stream, size in batches if stream == key] == [1, 4, 2]
        np.testing.assert_array_equal(simulation.outputs[key].time, times)
        for name, values in simulation.outputs[key].data_vars.items():
            # Near-zero potentials may differ by cancellation roundoff
            # between matrix-vector and matrix-matrix evaluation.
            atol = 1e-12 if name.endswith(("_Phi", "_W")) else 1e-22
            np.testing.assert_allclose(values, reference.outputs[key][name], rtol=1e-10, atol=atol)
    assert not simulation._time_evolution._output_samples


def test_interrupted_evolution_retains_pending_samples_and_resumes(monkeypatch):
    """Ctrl-C preserves completed samples between write intervals."""
    from pynamit.simulation.electrodynamics import induction

    preparation = _prepare_inputs()
    simulation = Simulation.from_inputs(preparation)
    reference = Simulation.from_inputs(preparation)
    build_stepper = induction.build_induction_stepper
    calls = 0

    def interruptible_stepper(*args, **kwargs):
        advance = build_stepper(*args, **kwargs)

        def step(*args, **kwargs):
            nonlocal calls
            kwargs["batch_size"] = 2
            for state in advance(*args, **kwargs):
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt
                yield state

        return step

    monkeypatch.setattr(induction, "build_induction_stepper", interruptible_stepper)
    options = dict(
        dt=0.0002,
        output_interval=0.0002,
        samples_per_write=10,
        initialize_from_equilibrium=False,
        quiet=True,
    )
    with pytest.raises(KeyboardInterrupt):
        simulation.evolve_to_time(0.001, **options)
    np.testing.assert_allclose(
        simulation.outputs["dynamic"].time, [0.0, 0.0002, 0.0004, 0.0006, 0.0008]
    )
    assert not simulation._time_evolution._output_samples
    simulation.evolve_to_time(0.001, **options)
    reference.evolve_to_time(0.001, **options)
    xr.testing.assert_allclose(simulation.outputs["dynamic"], reference.outputs["dynamic"])


def test_from_inputs_owns_an_independent_snapshot():
    """One preparation can seed independent simulation experiments."""
    preparation = _prepare_inputs()
    first = Simulation.from_inputs(preparation)
    second = Simulation.from_inputs(preparation, integrator="exponential")
    before = second.inputs["boundary_jr"].copy(deep=True)
    shape = preparation.schema.input_field_spaces["boundary_jr"].shape
    preparation.set_coefficients("boundary_jr", np.zeros(shape))
    xr.testing.assert_identical(first.inputs["boundary_jr"], before)
    first.inputs["boundary_jr"]["SH_boundary_jr"].values[:] = 7.0
    xr.testing.assert_identical(second.inputs["boundary_jr"], before)
    assert second.config.integrator == "exponential"
    assert first.config.integrator == preparation.config.integrator == "euler"


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_memory_save_reopen_and_restart_preserves_trajectory(tmp_path, storage):
    """Reopened evolution matches the uninterrupted trajectory."""
    preparation = _prepare_inputs(changing=True)
    simulation = Simulation.from_inputs(preparation)
    reference = Simulation.from_inputs(preparation)
    options = dict(dt=0.0002, output_interval=0.001, quiet=True, initialize_from_equilibrium=False)
    simulation.evolve_to_time(0.001, **options)
    simulation.save(tmp_path, artifact_storage=storage)
    reopened = Simulation.from_directory(tmp_path, artifact_storage=storage)
    for key in simulation.outputs:
        xr.testing.assert_allclose(reopened.outputs[key], simulation.outputs[key])
    reopened.evolve_to_time(0.0021, **options)
    reference.evolve_to_time(0.0021, **options)
    saved = SimulationResults.from_directory(tmp_path, artifact_storage=storage)
    for key in reference.outputs:
        xr.testing.assert_allclose(
            saved.outputs[key], reference.outputs[key], rtol=1e-10, atol=1e-14
        )
    assert preparation.input_directory is None


def test_preparation_save_and_explicit_input_time(tmp_path):
    """Prepared inputs have timestamps, not a simulation clock."""
    preparation = _prepare_inputs()
    preparation.save(tmp_path, artifact_storage="netcdf")
    preparation.write_manifest(source="test")
    loaded = InputPreparation.from_directory(tmp_path)
    for key in preparation:
        xr.testing.assert_allclose(loaded[key], preparation[key])
    simulation = Simulation.from_inputs(loaded)
    simulation.evolve_to_time(0.001, dt=0.0002, quiet=True, initialize_from_equilibrium=False)
    shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(shape))
    assert simulation.inputs["boundary_jr"].time.values[-1] == 0.0
    simulation.inputs.set_coefficients(
        "boundary_jr", np.zeros(shape), time=simulation.current_time
    )
    assert simulation.inputs["boundary_jr"].time.values[-1] == simulation.current_time
    assert loaded["boundary_jr"].time.size == 1


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_input_export_does_not_relocate_a_live_trajectory(tmp_path, storage):
    """Input export preserves later checkpoints and relocation."""
    simulation = Simulation.from_inputs(_prepare_inputs())
    simulation.set_state(np.zeros(simulation.geometry.poloidal_basis.coefficient_count))
    simulation.record_state()
    original = tmp_path / "trajectory"
    exported = tmp_path / "inputs"
    relocated = tmp_path / "relocated"
    simulation.save(original, artifact_storage=storage)
    simulation.inputs.save(exported)
    assert simulation.simulation_directory == str(original.resolve())
    assert simulation.inputs.input_directory == str(exported.resolve())
    assert not ArtifactStore(exported).get_dataset_storage_kinds("dynamic")
    simulation.inputs.set_coefficients(
        "boundary_jr",
        np.zeros(simulation.inputs.schema.input_field_spaces["boundary_jr"].shape),
        time=0.001,
    )
    simulation.evolve_to_time(0.001, dt=0.0002, quiet=True)
    saved = SimulationResults.from_directory(original)
    xr.testing.assert_allclose(saved.inputs["boundary_jr"], simulation.inputs["boundary_jr"])
    xr.testing.assert_allclose(saved.outputs["dynamic"], simulation.outputs["dynamic"])
    # Saving via the live result view relocates both sets of histories.
    simulation.results.save(relocated)
    assert simulation.inputs.artifact_store is simulation.results.artifact_store
    assert (
        simulation.inputs.input_directory
        == simulation.simulation_directory
        == str(relocated.resolve())
    )
    simulation.evolve_to_time(0.002, dt=0.0002, quiet=True)
    reopened = Simulation.from_directory(relocated)
    assert reopened.current_time == simulation.current_time
    np.testing.assert_allclose(reopened.induced_Br, simulation.induced_Br)


def test_saved_results_load_only_requested_inputs_and_never_build_a_projector(
    tmp_path, monkeypatch
):
    """Reading results does not initialize input-fitting machinery."""
    from pynamit.results import evaluate_projected_input
    from pynamit.simulation.input_projection import _InputProjector

    preparation = _prepare_inputs()
    preparation.save(tmp_path, artifact_storage="netcdf")
    loaded = []
    load = ArtifactStore.load_dataset

    def record_load(self, name, *args, **kwargs):
        loaded.append(name)
        return load(self, name, *args, **kwargs)

    def unexpected_projection(*args, **kwargs):
        pytest.fail("Reading input coefficients must not build a projector.")

    monkeypatch.setattr(ArtifactStore, "load_dataset", record_load)
    monkeypatch.setattr(_InputProjector, "__init__", unexpected_projection)
    results = SimulationResults.from_directory(tmp_path)
    assert loaded == ["settings"]
    evaluate_projected_input(results, "boundary_jr", 0.0)
    assert loaded == ["settings", "boundary_jr"]
    evaluate_projected_input(results.inputs, "boundary_jr", 0.0)
    assert loaded == ["settings", "boundary_jr"]


def test_from_inputs_rejects_incompatible_geometry_before_writing(tmp_path):
    """Copying must not reinterpret a coefficient space."""
    preparation = _prepare_inputs()
    destination = tmp_path / "incompatible"
    with pytest.raises(ValueError, match="Nmax"):
        Simulation.from_inputs(preparation, Nmax=3, simulation_directory=destination)
    assert not destination.exists()


def test_save_rejects_different_settings_without_rebinding(tmp_path):
    """A save must not silently overwrite an unrelated trajectory."""
    simulation = Simulation.from_inputs(_prepare_inputs())
    other = Simulation(tmp_path, Nmax=3, Mmax=1, Ncs=4, artifact_storage="netcdf")
    with pytest.raises(ValueError, match="different simulation settings"):
        simulation.save(tmp_path)
    assert simulation.simulation_directory is None
    assert (
        Simulation.from_directory(tmp_path)
        .config.to_dataset()
        .identical(other.config.to_dataset())
    )


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_explicit_save_includes_live_dataset_edits(tmp_path, storage):
    """Explicit saving includes edits made directly in IPython."""
    simulation = Simulation.from_inputs(_prepare_inputs())
    simulation.save(tmp_path, artifact_storage=storage)
    simulation.inputs["boundary_jr"]["SH_boundary_jr"].values *= 2
    simulation.save()
    reopened = Simulation.from_directory(tmp_path, artifact_storage=storage)
    xr.testing.assert_identical(reopened.inputs["boundary_jr"], simulation.inputs["boundary_jr"])


def test_save_does_not_mix_independent_trajectories(tmp_path):
    """Matching settings do not identify a single trajectory."""
    inputs = _prepare_inputs()
    first = Simulation.from_inputs(inputs)
    first.save(tmp_path)
    second = Simulation.from_inputs(inputs)
    with pytest.raises(ValueError, match="already contains a simulation"):
        second.save(tmp_path)
    assert second.simulation_directory is None


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_memory_retains_storage_preference_until_save(tmp_path, storage):
    """The constructor's storage choice applies when first saved."""
    simulation = Simulation.from_inputs(_prepare_inputs(), artifact_storage=storage)
    simulation.save(tmp_path)
    assert simulation.results.artifact_store.get_dataset_storage_kind("boundary_jr") == storage


def test_failed_forcing_update_does_not_reuse_an_old_solution(monkeypatch):
    """Retry a numerical failure with the newly selected forcing."""
    simulation = Simulation.from_inputs(_prepare_inputs())
    evolution = simulation._time_evolution
    series = simulation.results.input_series
    response, _, before = evolution._response_and_forcing(next(series.iter_intervals(0, 0))[2])
    simulation.inputs["boundary_jr"]["SH_boundary_jr"].values *= 2
    solve = response.solve_noninductive_response
    calls = []

    def fail_once(**forcing):
        calls.append(forcing)
        if len(calls) == 1:
            raise RuntimeError("Numerical solve failed.")
        return solve(**forcing)

    monkeypatch.setattr(response, "solve_noninductive_response", fail_once)
    with pytest.raises(RuntimeError, match="Numerical solve failed"):
        evolution._response_and_forcing(next(series.iter_intervals(0, 0))[2])
    _, _, after = evolution._response_and_forcing(next(series.iter_intervals(0, 0))[2])
    assert len(calls) == 2
    np.testing.assert_allclose(after, 2 * before)


@pytest.mark.parametrize("basis", ["SH", "CS"])
@pytest.mark.parametrize("integrator", ["euler", "exponential"])
def test_block_steps_match_single_steps_with_off_grid_input_changes(basis, integrator):
    """Blocks preserve the dt grid and physical output fields."""
    preparation = _prepare_inputs(
        horizontal_basis_kind=basis, integrator=integrator, changing=True
    )
    single = Simulation.from_inputs(preparation)
    blocked = Simulation.from_inputs(preparation)
    options = dict(
        dt=0.0002 if integrator == "euler" else None, quiet=True, initialize_from_equilibrium=False
    )
    single.evolve_to_time(0.0021, output_interval=0.0002, **options)
    blocked.evolve_to_time(0.0021, output_interval=0.001, **options)
    for key, actual in blocked.outputs.items():
        expected = single.outputs[key].sel(time=actual.time, method="nearest")
        xr.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(blocked.outputs["dynamic"].time, [0, 0.001, 0.002, 0.0021])


def test_unchanged_forcing_is_solved_once_across_blocks(monkeypatch):
    """Unchanged forcing does not repeat a noninductive solve."""
    simulation = Simulation.from_inputs(_prepare_inputs())
    solve = ElectrodynamicResponse.solve_noninductive_response
    calls = []

    def counted_solve(response, **forcing):
        calls.append(response)
        return solve(response, **forcing)

    monkeypatch.setattr(ElectrodynamicResponse, "solve_noninductive_response", counted_solve)
    options = dict(
        dt=0.0002, output_interval=0.0004, quiet=True, initialize_from_equilibrium=False
    )
    simulation.evolve_to_time(0.001, **options)
    simulation.evolve_to_time(0.002, **options)
    assert len(calls) == 1
    simulation.inputs["boundary_jr"]["SH_boundary_jr"].values *= 2
    simulation.evolve_to_time(0.003, **options)
    assert len(calls) == 2
    assert calls[0] is calls[1]


def test_interpolated_equilibrium_uses_one_selected_closure():
    """Output synthesis retains the interpolated conductance."""
    preparation = _prepare_inputs(changing=True)
    simulation = Simulation.from_inputs(preparation)
    midpoint = 0.0006
    interpolated = InputPreparation.from_config(preparation.config)
    for key in preparation:
        entry = preparation.input_series.get_entry(key, midpoint, interpolation=True)
        interpolated.input_series.add_entry(key, entry, time=midpoint)
    reference = Simulation.from_inputs(interpolated)
    simulation.sample_equilibria([midpoint], interpolation=True, quiet=True)
    reference.sample_equilibria([midpoint], quiet=True)
    for key in simulation.outputs:
        xr.testing.assert_allclose(simulation.outputs[key], reference.outputs[key])
    assert simulation.induced_Br is reference.induced_Br is None


@pytest.mark.parametrize("coordinates", ["model", "geographic"])
def test_live_plotting_sees_added_streams_samples_and_coefficient_edits(coordinates, monkeypatch):
    """Live plot views cache only numerical operators."""
    import pynamit.plotting.plot_data as plot_module

    simulation = Simulation.from_inputs(_prepare_inputs())
    view = PlotData.from_results(simulation.results, nlat=5, nlon=8, wind_nlat=3, wind_nlon=4)
    assert view.results is simulation.results
    assert view.results.inputs is simulation.inputs
    assert not view.has_model_output
    before_inputs = view.input_plot_data(0, coordinate_system=coordinates)
    assert np.all(np.isnan(before_inputs["wind_theta"]))
    simulation.inputs.set_coefficients(
        "u", np.ones(simulation.results.schema.input_field_spaces["u"].shape)
    )
    after_inputs = view.input_plot_data(0, coordinate_system=coordinates)
    assert np.all(np.isfinite(after_inputs["wind_theta"]))

    options = dict(dt=0.0002, output_interval=0.001, quiet=True, initialize_from_equilibrium=False)
    simulation.evolve_to_time(0.001, **options)
    assert view.n_time == 2
    assert np.all(
        np.isfinite(view.output_plot_data(-1, coordinate_system=coordinates)["Br_dynamic"])
    )
    ground = view.ground_magnetic_fields([60], [10])

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("Unchanged site geometry must reuse its operators")

    monkeypatch.setattr(plot_module, "build_ground_magnetic_field_operators", unexpected_rebuild)
    simulation.outputs["dynamic"]["SH_induced_Br"].values *= 2
    updated = view.ground_magnetic_fields([60], [10])
    np.testing.assert_allclose(updated["dynamic"]["radial"], 2 * ground["dynamic"]["radial"])
    simulation.evolve_to_time(0.002, **options)
    assert view.n_time == 3
    assert view.ground_magnetic_fields([60], [10])["dynamic"]["radial"].shape[-1] == 3


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_saved_results_relocation_retains_unloaded_streams(tmp_path, storage):
    """A copied saved result no longer depends on the original store."""
    original = tmp_path / "original"
    destination = tmp_path / "copy"
    simulation = Simulation.from_inputs(_prepare_inputs())
    simulation.evolve_to_time(0.001, dt=0.0002, quiet=True)
    simulation.save(original, artifact_storage=storage)
    results = SimulationResults.from_directory(original)
    assert set(results.datasets) == {"settings"}
    results.save(destination)
    original.rename(tmp_path / "moved_original")
    for key, expected in simulation.outputs.items():
        xr.testing.assert_allclose(results.outputs[key], expected)
    for key, expected in simulation.inputs.items():
        xr.testing.assert_allclose(results.inputs[key], expected)
    results.save()
