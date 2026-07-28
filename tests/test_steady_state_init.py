"""Steady state initialization test module."""

import numpy as np
import pytest

from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.standard import run_pynamit


def test_steady_state_init():
    """Test simulation with steady state initialization."""
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
        steady_state_initialization=True,
        jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    # Assert.
    coeff_array = np.hstack(
        (
            simulation.run_data.output_series.datasets["state"]["SH_m_ind"].values[-1],
            simulation.run_data.output_series.datasets["state"]["SH_m_imp"].values[-1],
        )
    )

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


def test_impose_steady_state_at_current_time(tmp_path, monkeypatch):
    """Imposed steady state should overwrite the live state."""
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
        steady_state_initialization=False,
        jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    steady_state_m_ind = simulation.impose_steady_state(quiet=True)

    state_entry = simulation.run_data.output_series.get_entry("state", simulation.current_time)
    steady_entry = simulation.run_data.output_series.get_entry(
        "steady_state", simulation.current_time
    )

    np.testing.assert_allclose(np.asarray(state_entry["m_ind"]), np.asarray(steady_state_m_ind))
    np.testing.assert_allclose(np.asarray(steady_entry["m_ind"]), np.asarray(steady_state_m_ind))


def test_impose_steady_state_updates_memory_without_persisting(tmp_path):
    """The save option controls disk persistence, not live state."""
    simulation = run_pynamit(
        final_time=0.0,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        steady_state_initialization=False,
        run_directory=tmp_path / "run",
        artifact_storage="netcdf",
    )
    simulation.run_data.artifact_store.remove_artifact("state")
    simulation.run_data.artifact_store.remove_artifact("steady_state")
    simulation.run_data.output_series.datasets.clear()

    steady_state_m_ind = simulation.impose_steady_state(time=0.0, save=False, quiet=True)

    state_entry = simulation.run_data.output_series.get_entry("state", 0.0)
    np.testing.assert_allclose(state_entry["m_ind"], steady_state_m_ind)
    assert simulation.run_data.artifact_store.get_dataset_storage_kind("state") is None
    assert simulation.run_data.artifact_store.get_dataset_storage_kind("steady_state") is None


def test_impose_steady_state_rejects_an_earlier_trajectory_time(tmp_path):
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
        simulation.impose_steady_state(time=0.0, quiet=True)


def test_impose_steady_state_matches_steady_state_initialization(tmp_path, monkeypatch):
    """Explicit steady state should match initialized steady state."""
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
        jr_projection_basis="SH",
        conductance_projection_basis="SH",
        u_projection_basis="SH",
    )

    init_dir = tmp_path / "initialized"
    init_dir.mkdir()
    monkeypatch.chdir(init_dir)
    initialized = run_pynamit(**common_kwargs, steady_state_initialization=True)

    imposed_dir = tmp_path / "imposed"
    imposed_dir.mkdir()
    monkeypatch.chdir(imposed_dir)
    imposed = run_pynamit(**common_kwargs, steady_state_initialization=False)
    imposed.impose_steady_state(quiet=True)

    initialized_entry = initialized.run_data.output_series.get_entry(
        "state", initialized.current_time
    )
    imposed_entry = imposed.run_data.output_series.get_entry("state", imposed.current_time)

    for key in ("m_ind", "m_imp", "Phi", "W"):
        # Phi and W vanish at exact equilibrium, so independently
        # composed operator paths need an absolute roundoff floor.
        np.testing.assert_allclose(
            np.asarray(imposed_entry[key]), np.asarray(initialized_entry[key]), atol=1e-15
        )


def test_evolve_to_time_can_run_steady_state_without_inductive_state(tmp_path):
    """Steady-state output can run without inductive evolution."""
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
        steady_state_initialization=False,
        run_inductive=False,
        run_steady_state=True,
        run_directory=str(tmp_path / "steady-only"),
        artifact_storage="netcdf",
    )

    assert "state" not in simulation.run_data.output_series.datasets
    assert "steady_state" in simulation.run_data.output_series.datasets
    np.testing.assert_allclose(
        simulation.run_data.output_series.datasets["steady_state"].time.values, [0.0, 0.05, 0.1]
    )
    assert not (tmp_path / "steady-only" / "state.ncdf").exists()
    assert (tmp_path / "steady-only" / "steady_state.ncdf").is_file()

    reloaded = Simulation.from_directory(
        tmp_path / "steady-only", artifact_storage="netcdf", backend="numpy"
    )
    assert reloaded.current_time == pytest.approx(0.1)


def test_evolve_to_time_can_run_inductive_state_without_steady_state(tmp_path):
    """Inductive output can run without steady-state output."""
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
        steady_state_initialization=False,
        run_inductive=True,
        run_steady_state=False,
        run_directory=str(tmp_path / "inductive-only"),
        artifact_storage="netcdf",
    )

    assert "state" in simulation.run_data.output_series.datasets
    assert "steady_state" not in simulation.run_data.output_series.datasets
    np.testing.assert_allclose(
        simulation.run_data.output_series.datasets["state"].time.values, [0.0, 0.05, 0.1]
    )
    assert (tmp_path / "inductive-only" / "state.ncdf").is_file()
    assert not (tmp_path / "inductive-only" / "steady_state.ncdf").exists()


def test_evolve_to_time_split_modes_match_combined_numerically(tmp_path, rel_tol):
    """Separate inductive and steady-state runs match a combined run."""
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
        steady_state_initialization=False,
        multi_data=True,
        artifact_storage="netcdf",
    )

    combined = run_pynamit(
        **common_kwargs,
        run_inductive=True,
        run_steady_state=True,
        run_directory=str(tmp_path / "combined"),
    )
    inductive = run_pynamit(
        **common_kwargs,
        run_inductive=True,
        run_steady_state=False,
        run_directory=str(tmp_path / "inductive"),
    )
    steady = run_pynamit(
        **common_kwargs,
        run_inductive=False,
        run_steady_state=True,
        run_directory=str(tmp_path / "steady"),
    )

    state_combined = combined.run_data.output_series.datasets["state"]
    state_inductive = inductive.run_data.output_series.datasets["state"]
    steady_combined = combined.run_data.output_series.datasets["steady_state"]
    steady_split = steady.run_data.output_series.datasets["steady_state"]

    np.testing.assert_allclose(state_inductive.time.values, state_combined.time.values)
    np.testing.assert_allclose(steady_split.time.values, steady_combined.time.values)

    for variable in ("SH_m_ind", "SH_m_imp", "SH_Phi", "SH_W"):
        np.testing.assert_allclose(
            state_inductive[variable].values,
            state_combined[variable].values,
            rtol=rel_tol,
            atol=0.0,
        )
        np.testing.assert_allclose(
            steady_split[variable].values, steady_combined[variable].values, rtol=rel_tol, atol=0.0
        )

    assert np.linalg.norm(state_combined["SH_m_ind"].values[-1]) > 0.0
    assert np.linalg.norm(steady_combined["SH_m_ind"].values[-1]) > 0.0
