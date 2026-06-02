"""Steady state initialization test module."""

import os
import tempfile
import pytest

from pynamit.default_run import run_pynamit
import numpy as np


def test_steady_state_init():
    """Test simulation with steady state initialization."""
    # Arrange.
    expected_coeff_norm = 1.3120048541771941e-08
    expected_coeff_max = 1.7170964863338117e-09
    expected_coeff_min = -4.858577603591746e-09
    expected_n_coeffs = 228

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        use_wind=True,
        steady_state_initialization=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
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

    dynamics = run_pynamit(
        final_time=0.0,
        dt=1e-2,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        fig_directory=str(tmp_path),
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        use_wind=True,
        steady_state_initialization=False,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
    )

    steady_state_m_ind = dynamics.impose_steady_state(quiet=True)

    state_entry = dynamics.output_timeseries.get_entry("state", dynamics.current_time)
    steady_entry = dynamics.output_timeseries.get_entry("steady_state", dynamics.current_time)

    np.testing.assert_allclose(np.asarray(state_entry["m_ind"]), np.asarray(steady_state_m_ind))
    np.testing.assert_allclose(np.asarray(steady_entry["m_ind"]), np.asarray(steady_state_m_ind))


def test_impose_steady_state_matches_steady_state_initialization(tmp_path, monkeypatch):
    """Explicit steady state should match initialized steady state."""
    common_kwargs = dict(
        final_time=0.0,
        dt=1e-2,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        use_wind=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
    )

    init_dir = tmp_path / "initialized"
    init_dir.mkdir()
    monkeypatch.chdir(init_dir)
    initialized = run_pynamit(
        **common_kwargs, fig_directory=str(init_dir), steady_state_initialization=True
    )

    imposed_dir = tmp_path / "imposed"
    imposed_dir.mkdir()
    monkeypatch.chdir(imposed_dir)
    imposed = run_pynamit(
        **common_kwargs, fig_directory=str(imposed_dir), steady_state_initialization=False
    )
    imposed.impose_steady_state(quiet=True)

    initialized_entry = initialized.output_timeseries.get_entry("state", initialized.current_time)
    imposed_entry = imposed.output_timeseries.get_entry("state", imposed.current_time)

    for key in ("m_ind", "m_imp", "Phi", "W"):
        np.testing.assert_allclose(
            np.asarray(imposed_entry[key]), np.asarray(initialized_entry[key])
        )


def test_evolve_to_time_can_run_steady_state_without_inductive_state(tmp_path):
    """Steady-state output can run without inductive evolution."""
    dynamics = run_pynamit(
        final_time=0.1,
        dt=0.05,
        plotsteps=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
        use_wind=False,
        steady_state_initialization=False,
        run_inductive=False,
        run_steady_state=True,
        run_directory=str(tmp_path / "steady-only"),
        artifact_storage="netcdf",
    )

    assert "state" not in dynamics.output_timeseries.datasets
    assert "steady_state" in dynamics.output_timeseries.datasets
    np.testing.assert_allclose(
        dynamics.output_timeseries.datasets["steady_state"].time.values,
        [0.0, 0.05, 0.1],
    )
    assert not (tmp_path / "steady-only" / "state.ncdf").exists()
    assert (tmp_path / "steady-only" / "steady_state.ncdf").is_file()


def test_evolve_to_time_can_run_inductive_state_without_steady_state(tmp_path):
    """Inductive output can run without steady-state output."""
    dynamics = run_pynamit(
        final_time=0.1,
        dt=0.05,
        plotsteps=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
        use_wind=False,
        steady_state_initialization=False,
        run_inductive=True,
        run_steady_state=False,
        run_directory=str(tmp_path / "inductive-only"),
        artifact_storage="netcdf",
    )

    assert "state" in dynamics.output_timeseries.datasets
    assert "steady_state" not in dynamics.output_timeseries.datasets
    np.testing.assert_allclose(
        dynamics.output_timeseries.datasets["state"].time.values,
        [0.0, 0.05, 0.1],
    )
    assert (tmp_path / "inductive-only" / "state.ncdf").is_file()
    assert not (tmp_path / "inductive-only" / "steady_state.ncdf").exists()


def test_evolve_to_time_split_modes_match_combined_numerically(tmp_path, rel_tol):
    """Separate inductive and steady-state runs match a combined run."""
    common_kwargs = dict(
        final_time=0.1,
        dt=0.05,
        plotsteps=1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
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

    state_combined = combined.output_timeseries.datasets["state"]
    state_inductive = inductive.output_timeseries.datasets["state"]
    steady_combined = combined.output_timeseries.datasets["steady_state"]
    steady_split = steady.output_timeseries.datasets["steady_state"]

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
            steady_split[variable].values,
            steady_combined[variable].values,
            rtol=rel_tol,
            atol=0.0,
        )

    assert np.linalg.norm(state_combined["SH_m_ind"].values[-1]) > 0.0
    assert np.linalg.norm(steady_combined["SH_m_ind"].values[-1]) > 0.0
