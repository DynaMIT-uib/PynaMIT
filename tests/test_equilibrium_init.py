"""Equilibrium initialization test module."""

import numpy as np
import pytest

from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


def test_equilibrium_init():
    """Test simulation with equilibrium initialization."""
    # Arrange.
    # HWM winds are rotated from geographic into dipole coordinates.
    expected_coeff_norm = 1.3348885792521104e-08
    expected_coeff_max = 4.556847574222238e-09
    expected_coeff_min = -5.482384396922763e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        equilibrium_initialization=True,
        boundary_jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    # Assert.
    coeff_array = magnetic_potential_coordinate_array(simulation)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    # pyHWM uses single precision, relax tolerances for wind tests.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)


def test_impose_equilibrium_at_current_time(tmp_path, monkeypatch):
    """Imposed equilibrium should overwrite the live dynamic output."""
    monkeypatch.chdir(tmp_path)

    simulation = run_pynamit(
        final_time=0.0,
        dt=1e-2,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        equilibrium_initialization=False,
        boundary_jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    equilibrium_induced_Br = simulation.impose_equilibrium(quiet=True)

    dynamic_entry = simulation.run_data.output_series.get_entry("dynamic", simulation.current_time)
    equilibrium_entry = simulation.run_data.output_series.get_entry(
        "equilibrium", simulation.current_time
    )

    np.testing.assert_allclose(
        np.asarray(dynamic_entry["induced_Br"]), np.asarray(equilibrium_induced_Br)
    )
    np.testing.assert_allclose(
        np.asarray(equilibrium_entry["induced_Br"]), np.asarray(equilibrium_induced_Br)
    )


def test_impose_equilibrium_updates_memory_without_persisting(tmp_path):
    """The save option controls disk persistence, not live output."""
    simulation = run_pynamit(
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        equilibrium_initialization=False,
        run_directory=tmp_path / "run",
        artifact_storage="netcdf",
    )
    simulation.run_data.artifact_store.remove_artifact("dynamic")
    simulation.run_data.artifact_store.remove_artifact("equilibrium")
    simulation.run_data.output_series.datasets.clear()

    equilibrium_induced_Br = simulation.impose_equilibrium(time=0.0, save=False, quiet=True)

    dynamic_entry = simulation.run_data.output_series.get_entry("dynamic", 0.0)
    np.testing.assert_allclose(dynamic_entry["induced_Br"], equilibrium_induced_Br)
    assert simulation.run_data.artifact_store.get_dataset_storage_kind("dynamic") is None
    assert simulation.run_data.artifact_store.get_dataset_storage_kind("equilibrium") is None


def test_impose_equilibrium_rejects_an_earlier_trajectory_time(tmp_path):
    """Imposition cannot leave later checkpoints on another branch."""
    simulation = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        saving_sample_interval=1,
        run_directory=tmp_path / "run",
        artifact_storage="netcdf",
    )

    with pytest.raises(ValueError, match="before the active checkpoint"):
        simulation.impose_equilibrium(time=0.0, quiet=True)


def test_impose_equilibrium_matches_equilibrium_initialization(tmp_path, monkeypatch):
    """Explicit equilibrium should match initialized equilibrium."""
    common_kwargs = dict(
        final_time=0.0,
        dt=1e-2,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        boundary_jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    init_dir = tmp_path / "initialized"
    init_dir.mkdir()
    monkeypatch.chdir(init_dir)
    initialized = run_pynamit(**common_kwargs, equilibrium_initialization=True)

    imposed_dir = tmp_path / "imposed"
    imposed_dir.mkdir()
    monkeypatch.chdir(imposed_dir)
    imposed = run_pynamit(**common_kwargs, equilibrium_initialization=False)
    imposed.impose_equilibrium(quiet=True)

    initialized_entry = initialized.run_data.output_series.get_entry(
        "dynamic", initialized.current_time
    )
    imposed_entry = imposed.run_data.output_series.get_entry("dynamic", imposed.current_time)

    for key in ("induced_Br", "boundary_jr", "Phi", "W"):
        # Phi and W vanish at exact equilibrium, so independently
        # composed operator paths need an absolute roundoff floor.
        np.testing.assert_allclose(
            np.asarray(imposed_entry[key]), np.asarray(initialized_entry[key]), atol=1e-15
        )


def test_evolve_to_time_can_run_equilibrium_without_dynamic_output(tmp_path):
    """Equilibrium output can run without dynamic evolution."""
    simulation = run_pynamit(
        final_time=0.1,
        dt=0.05,
        saving_sample_interval=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        use_wind=False,
        equilibrium_initialization=False,
        run_dynamic=False,
        run_equilibrium=True,
        run_directory=str(tmp_path / "equilibrium-only"),
        artifact_storage="netcdf",
    )

    assert "dynamic" not in simulation.run_data.output_series.datasets
    assert "equilibrium" in simulation.run_data.output_series.datasets
    np.testing.assert_allclose(
        simulation.run_data.output_series.datasets["equilibrium"].time.values, [0.0, 0.05, 0.1]
    )
    assert not (tmp_path / "equilibrium-only" / "dynamic.ncdf").exists()
    assert (tmp_path / "equilibrium-only" / "equilibrium.ncdf").is_file()

    reloaded = Simulation.from_directory(
        tmp_path / "equilibrium-only", artifact_storage="netcdf", backend="numpy"
    )
    assert reloaded.current_time == pytest.approx(0.1)


def test_evolve_to_time_can_run_dynamic_output_without_equilibrium(tmp_path):
    """Dynamic output can run without equilibrium output."""
    simulation = run_pynamit(
        final_time=0.1,
        dt=0.05,
        saving_sample_interval=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        use_wind=False,
        equilibrium_initialization=False,
        run_dynamic=True,
        run_equilibrium=False,
        run_directory=str(tmp_path / "inductive-only"),
        artifact_storage="netcdf",
    )

    assert "dynamic" in simulation.run_data.output_series.datasets
    assert "equilibrium" not in simulation.run_data.output_series.datasets
    np.testing.assert_allclose(
        simulation.run_data.output_series.datasets["dynamic"].time.values, [0.0, 0.05, 0.1]
    )
    assert (tmp_path / "inductive-only" / "dynamic.ncdf").is_file()
    assert not (tmp_path / "inductive-only" / "equilibrium.ncdf").exists()


def test_evolve_to_time_split_modes_match_combined_numerically(tmp_path, rel_tol):
    """Separate dynamic and equilibrium runs match a combined run."""
    common_kwargs = dict(
        final_time=0.1,
        dt=0.05,
        saving_sample_interval=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        use_wind=False,
        equilibrium_initialization=False,
        multi_data=True,
        artifact_storage="netcdf",
    )

    combined = run_pynamit(
        **common_kwargs,
        run_dynamic=True,
        run_equilibrium=True,
        run_directory=str(tmp_path / "combined"),
    )
    inductive = run_pynamit(
        **common_kwargs,
        run_dynamic=True,
        run_equilibrium=False,
        run_directory=str(tmp_path / "inductive"),
    )
    equilibrium = run_pynamit(
        **common_kwargs,
        run_dynamic=False,
        run_equilibrium=True,
        run_directory=str(tmp_path / "equilibrium"),
    )

    dynamic_combined = combined.run_data.output_series.datasets["dynamic"]
    dynamic_split = inductive.run_data.output_series.datasets["dynamic"]
    equilibrium_combined = combined.run_data.output_series.datasets["equilibrium"]
    equilibrium_split = equilibrium.run_data.output_series.datasets["equilibrium"]

    np.testing.assert_allclose(dynamic_split.time.values, dynamic_combined.time.values)
    np.testing.assert_allclose(equilibrium_split.time.values, equilibrium_combined.time.values)

    for variable in ("SH_induced_Br", "SH_boundary_jr", "SH_Phi", "SH_W"):
        np.testing.assert_allclose(
            dynamic_split[variable].values,
            dynamic_combined[variable].values,
            rtol=rel_tol,
            atol=0.0,
        )
        np.testing.assert_allclose(
            equilibrium_split[variable].values,
            equilibrium_combined[variable].values,
            rtol=rel_tol,
            atol=0.0,
        )

    assert np.linalg.norm(dynamic_combined["SH_induced_Br"].values[-1]) > 0.0
    assert np.linalg.norm(equilibrium_combined["SH_induced_Br"].values[-1]) > 0.0
