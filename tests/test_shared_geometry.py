"""Reusable numerical objects with independent experiment histories."""

import numpy as np
import pytest
from kompe import GlobalCSBasis, SHBasis
from kompe.basis import BasisSubset
from kompe.cache import PersistentArrayCache
from kompe.math import backend_context

from pynamit import InputPreparation, Simulation, SimulationConfig, SimulationGeometry
from pynamit.results import OutputEvaluation, SimulationResults, evaluate_projected_input


@pytest.mark.parametrize("horizontal", ["SH", "CS"])
@pytest.mark.parametrize("mean_free", [False, True])
def test_existing_bases_define_the_geometry_and_saved_spaces(tmp_path, horizontal, mean_free):
    """Both construction paths define identical field spaces."""
    sh = SHBasis(
        2, 1, mean_free=mean_free, operator_cache=PersistentArrayCache(tmp_path / "cache")
    )
    cs = GlobalCSBasis(4)
    geometry = SimulationGeometry.from_bases(
        sh,
        cs,
        horizontal_basis=cs if horizontal == "CS" else None,
        main_field_kind="radial",
        enable_pfac_coupling=False,
    )
    simulation = Simulation.from_geometry(geometry)
    reference = Simulation.from_config(simulation.config)
    assert simulation.geometry is geometry
    assert geometry.cs_basis is cs
    assert geometry.poloidal_basis is sh.with_mean_free(True)
    assert simulation.operator_cache is sh.operator_cache
    assert "surface_to_poloidal_operator" not in geometry.__dict__
    for key, space in simulation.results.schema.input_field_spaces.items():
        assert space.signature == reference.results.schema.input_field_spaces[key].signature
    coeffs = np.linspace(-1e-6, 1e-6, geometry.horizontal_basis.coefficient_count)
    for run in (simulation, reference):
        run.inputs.set_coefficients("boundary_jr", coeffs)
    np.testing.assert_allclose(
        evaluate_projected_input(simulation.results, "boundary_jr", 0)["boundary_jr"],
        evaluate_projected_input(reference.results, "boundary_jr", 0)["boundary_jr"],
    )
    simulation.save(tmp_path / "saved", artifact_storage="netcdf")
    reopened = SimulationResults.from_directory(tmp_path / "saved")
    for key, space in simulation.results.schema.input_field_spaces.items():
        assert space.signature == reopened.schema.input_field_spaces[key].signature


def test_shared_geometry_reuses_operators_not_histories_or_run_controls():
    """An integrator change leaves sibling experiments untouched."""
    geometry = SimulationGeometry.from_config(SimulationConfig(Nmax=2, Mmax=1, Ncs=4))
    preparation = InputPreparation.from_geometry(geometry)
    preparation.set_coefficients(
        "boundary_jr", np.ones(geometry.horizontal_basis.coefficient_count)
    )
    first = Simulation.from_inputs(preparation, integrator="exponential")
    second = Simulation.from_inputs(first.inputs, save_equilibria=False)
    assert first.geometry is second.geometry is preparation.geometry
    assert (
        first.geometry.horizontal_transform
        is preparation._input_projector.projection_transform("boundary_jr")
    )
    operator = first.geometry.surface_to_poloidal_operator
    operator.to_array()
    assert second.geometry.surface_to_poloidal_operator is operator
    assert second.config.integrator == "exponential"
    assert first.config.save_equilibria and not second.config.save_equilibria
    assert second.results is not first.results
    first.inputs["boundary_jr"]["SH_boundary_jr"].values[:] = 5
    np.testing.assert_allclose(second.inputs["boundary_jr"]["SH_boundary_jr"], 1)


@pytest.mark.parametrize("setting", ["RI", "main_field_B0", "area_weighted_least_squares"])
def test_changed_physics_reuses_bases_but_not_physical_operators(setting):
    """Changed assumptions start fresh physical caches."""
    geometry = SimulationGeometry.from_config(
        SimulationConfig(Nmax=2, Mmax=1, Ncs=4, main_field_kind="radial", main_field_B0=4e-5)
    )
    original = geometry.surface_laplacian_operator
    value = {"RI": geometry.RI * 1.1, "main_field_B0": 5e-5, "area_weighted_least_squares": True}[
        setting
    ]
    config = SimulationConfig.from_geometry(
        geometry, fac_integration_radii=None, **{setting: value}
    )
    changed = geometry.with_config(config)
    assert changed is not geometry
    assert changed.sh_basis is geometry.sh_basis
    assert changed.cs_basis is geometry.cs_basis
    assert "surface_laplacian_operator" not in changed.__dict__
    reference = SimulationGeometry.from_config(config)
    np.testing.assert_allclose(
        changed.surface_laplacian_operator.to_matrix(),
        reference.surface_laplacian_operator.to_matrix(),
    )
    assert geometry.surface_laplacian_operator is original
    assert (changed.horizontal_transform is geometry.horizontal_transform) == (
        setting != "area_weighted_least_squares"
    )


def test_basis_objects_reject_conflicting_storage_descriptions():
    """The on-disk metadata cannot reinterpret supplied coefficients."""
    sh, cs = SHBasis(2, 1), GlobalCSBasis(4)
    with pytest.raises(ValueError, match="determine"):
        SimulationGeometry.from_bases(sh, cs, Nmax=3)
    geometry = SimulationGeometry.from_bases(sh, cs)
    with pytest.raises(ValueError, match="Ncs"):
        Simulation.from_geometry(geometry, Ncs=8)
    geometry = SimulationGeometry.from_bases(SHBasis(2, 1, schmidt_quasi_normalized=False), cs)
    experiment = Simulation.from_geometry(geometry)
    with pytest.raises(ValueError, match="Schmidt"):
        experiment.config.validate_geometry_for_storage(geometry)


@pytest.mark.parametrize("variant", ["unnormalized_SH", "permuted_horizontal_SH"])
def test_alternative_in_memory_bases_preserve_physical_evolution(tmp_path, variant):
    """File-format limitations do not constrain the equations."""
    cs = GlobalCSBasis(4)
    sh = SHBasis(3, 2, schmidt_quasi_normalized=variant != "unnormalized_SH")
    horizontal = sh.with_mean_free(True)
    if variant == "permuted_horizontal_SH":
        horizontal = BasisSubset(horizontal, np.arange(horizontal.coefficient_count)[::-1])
    geometry = SimulationGeometry.from_bases(
        sh, cs, horizontal_basis=horizontal, main_field_kind="radial", enable_pfac_coupling=False
    )
    experiment = Simulation.from_geometry(geometry, integrator="exponential")
    reference = Simulation.from_config(experiment.config)
    grid = cs.native_grid
    theta = np.deg2rad(grid.theta)
    phi = np.deg2rad(grid.phi)
    current = 1e-7 * np.sin(theta) * np.cos(phi)
    Br = 1e-9 * np.cos(theta)
    for simulation in (experiment, reference):
        simulation.inputs.set_conductance(
            pedersen=5.0 + np.zeros(grid.size), hall=8.0 + np.zeros(grid.size), grid=grid
        )
        simulation.inputs.set_boundary_jr(current, grid=grid)
        initial = simulation.geometry.horizontal_transform.with_basis(
            simulation.geometry.poloidal_basis
        ).analyze_scalar(Br)
        simulation.set_state(initial)
        simulation.evolve_to_time(0.002, output_interval=0.001, quiet=True)
    fields = OutputEvaluation(geometry).evaluate(experiment.output_coefficients())
    expected = OutputEvaluation(reference.geometry).evaluate(reference.output_coefficients())
    for name in fields:
        np.testing.assert_allclose(fields[name], expected[name], rtol=2e-10, atol=1e-15)
    # No partial artifact should be written for an unsupported recipe.
    destination = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="file format"):
        experiment.save(destination)
    assert not destination.exists()
    assert experiment.simulation_directory is None


def test_geometry_does_not_inherit_experiment_controls():
    """One spatial context can seed independent experiment policies."""
    source = SimulationConfig(Nmax=2, Mmax=1, Ncs=4, integrator="exponential", t0="2001-05-12")
    geometry = SimulationGeometry.from_config(source)
    first = Simulation.from_geometry(geometry, t0=source.t0, integrator="RK45")
    second = Simulation.from_geometry(geometry, t0="2001-05-13", integrator="euler")
    assert first.geometry is second.geometry is geometry
    assert first.config.t0 != second.config.t0
    assert (
        first.config.main_field_epoch == second.config.main_field_epoch == source.main_field_epoch
    )
    assert first.config.integrator == "RK45"
    assert second.config.integrator == "euler"
    settings = geometry.physical_settings
    settings["RI"] *= 1.1
    assert SimulationConfig.from_geometry(geometry).RI == source.RI
    with pytest.raises(ValueError, match="experiment controls"):
        SimulationGeometry.from_bases(SHBasis(2, 1), GlobalCSBasis(4), integrator="euler")


@pytest.mark.requires_jax
def test_geometry_reuse_across_backends_retains_bases_not_array_caches():
    """The new backend gets fresh physical and transform operators."""
    import jax

    with backend_context("numpy"):
        geometry = SimulationGeometry.from_config(SimulationConfig(Nmax=2, Mmax=1, Ncs=4))
        original = geometry.induced_Br_to_poloidal_potential_operator.to_matrix()
        _ = geometry.horizontal_transform.scalar_synthesis_array
    with backend_context("jax"):
        changed = geometry.with_config(SimulationConfig.from_geometry(geometry))
        assert changed.sh_basis is geometry.sh_basis
        assert changed.horizontal_transform is not geometry.horizontal_transform
        values = changed.induced_Br_to_poloidal_potential_operator.to_matrix()
        assert isinstance(values, jax.Array)
        assert isinstance(changed.horizontal_transform.scalar_synthesis_array, jax.Array)
        np.testing.assert_array_equal(values, original)


@pytest.mark.requires_jax
def test_lazy_physical_operators_do_not_capture_stepper_tracers():
    """Independent JIT kernels may safely share a fresh geometry."""
    import jax

    from pynamit.simulation.electrodynamics import induction

    with backend_context("jax"):
        simulation = Simulation(
            Nmax=2, Mmax=1, Ncs=4, main_field_kind="radial", enable_pfac_coupling=False
        )
        shape = simulation.results.schema.input_field_spaces["conductance"].shape
        simulation.inputs.set_coefficients(
            "conductance",
            {
                "log_conductance_magnitude": np.zeros(shape),
                "log_hall_to_pedersen_ratio": np.zeros(shape),
            },
        )
        response = simulation.response
        xp = jax.numpy
        values = xp.ones(simulation.geometry.poloidal_basis.coefficient_count) * 1e-9
        forcing = xp.zeros((2, simulation.geometry.horizontal_basis.coefficient_count))
        step = induction.build_induction_stepper(response, forcing, dt=1e-4)
        actual = next(step(values, [2e-4]))[..., 0]
        single = jax.jit(
            lambda values: induction.evolve_induced_Br(response, values, 1e-4, forcing, dt=1e-4)
        )
        expected = single(single(values))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-22)
