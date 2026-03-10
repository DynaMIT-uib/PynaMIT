from __future__ import annotations

import numpy as np
import xarray as xr

from pynamit.primitives.io import IO
from pynamit.simulation.data import SimulationData
from pynamit.simulation.settings import DynamicsSettings
from pynamit.primitives.grid import Grid


def test_simulation_data_loads_saved_inputs_and_outputs(tmp_path):
    run_dir = tmp_path / "results_case"
    settings = DynamicsSettings(
        run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6, t0="2001-05-12 21:45:00"
    )
    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    io = IO(str(run_dir))
    io.save_dataset(settings.to_dataset(), "settings")

    sh_full_basis = simulation_data.sh_basis
    input_timeseries = simulation_data.input_timeseries
    output_timeseries = simulation_data.output_timeseries
    solution_spec = simulation_data.solution_spec

    n_scalar = sh_full_basis.scalar_index_length(mean_free=True)
    n_conductance = sh_full_basis.index_length
    n_solution = solution_spec.index_length

    input_timeseries.add_entry("jr", {"jr": np.arange(n_scalar, dtype=float)}, time=0.0)
    input_timeseries.add_entry("jr", {"jr": np.arange(n_scalar, dtype=float) + 10.0}, time=2.0)
    input_timeseries.add_entry(
        "conductance",
        {"etaP": np.full(n_conductance, 2.0), "etaH": np.full(n_conductance, 0.5)},
        time=0.0,
    )
    input_timeseries.add_entry(
        "u", {"u": np.arange(2 * n_scalar, dtype=float).reshape(2, n_scalar)}, time=0.0
    )
    input_timeseries.save("jr", io)
    input_timeseries.save("conductance", io)
    input_timeseries.save("u", io)

    output_timeseries.add_entry(
        "state",
        {
            "m_ind": np.full(n_solution, 1.0),
            "m_imp": np.full(n_solution, 2.0),
            "Phi": np.full(n_solution, 3.0),
            "W": np.full(n_solution, 4.0),
        },
        time=0.0,
    )
    output_timeseries.add_entry(
        "state",
        {
            "m_ind": np.full(n_solution, 5.0),
            "m_imp": np.full(n_solution, 6.0),
            "Phi": np.full(n_solution, 7.0),
            "W": np.full(n_solution, 8.0),
        },
        time=2.0,
    )
    output_timeseries.save("state", io)

    io.save_dataarray(xr.DataArray(np.eye(n_solution)), "PFAC_matrix")

    simulation_data = SimulationData.from_directory(run_dir)

    assert simulation_data.settings.Nmax == settings.Nmax
    assert simulation_data.mainfield.kind == settings.mainfield_kind
    assert simulation_data.pfac_matrix.shape == (n_solution, n_solution)

    state_entry = simulation_data.get_output_entry("state", 1.0)
    assert state_entry is not None
    assert "psi" not in state_entry
    np.testing.assert_allclose(state_entry["m_imp"], np.full(n_solution, 2.0))

    jr_entry = simulation_data.get_input_entry("jr", 1.0)
    assert jr_entry is not None
    np.testing.assert_allclose(jr_entry["jr"], np.arange(n_scalar, dtype=float))

    jr_interp, jr_derivative = simulation_data.get_input_entry_with_derivative(
        "jr", 1.0, interpolation=True
    )
    assert jr_interp is not None and jr_derivative is not None
    np.testing.assert_allclose(jr_interp["jr"], np.arange(n_scalar, dtype=float) + 5.0)
    np.testing.assert_allclose(jr_derivative["jr"], np.full(n_scalar, 5.0))

    conductance_entry = simulation_data.get_input_entry("conductance", 0.0)
    assert conductance_entry is not None
    assert set(conductance_entry) == {"etaP", "etaH"}
    conductance_spec = simulation_data.get_storage_spec("conductance")
    jr_spec = simulation_data.get_storage_spec("jr")
    assert conductance_spec.basis.kind == "SH"
    assert conductance_spec.mean_free is False
    assert jr_spec.mean_free is True
    assert simulation_data.has_dataset("state")

    simulation_data_from_dir = SimulationData.from_directory(run_dir)
    assert simulation_data_from_dir.settings.Nmax == settings.Nmax
    np.testing.assert_allclose(simulation_data_from_dir.pfac_matrix, simulation_data.pfac_matrix)


def test_simulation_data_create_saves_sidecars(tmp_path):
    run_dir = tmp_path / "runtime_case"
    settings = DynamicsSettings(
        run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6, t0="2001-05-12 21:45:00"
    )

    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    simulation_data.save_settings()
    simulation_data.save_pfac_matrix(np.eye(simulation_data.solution_spec.index_length))

    settings_storage = simulation_data.io.get_dataset_storage_kind("settings")
    pfac_storage = simulation_data.io.get_dataset_storage_kind("PFAC_matrix")
    settings_suffix = ".zarr" if settings_storage == "zarr" else ".ncdf"
    pfac_suffix = ".zarr" if pfac_storage == "zarr" else ".ncdf"
    assert (run_dir / f"settings{settings_suffix}").exists()
    assert (run_dir / f"PFAC_matrix{pfac_suffix}").exists()

    reloaded = SimulationData.from_directory(run_dir)
    assert reloaded.settings.Nmax == settings.Nmax
    assert reloaded.io.get_dataset_storage_kind("settings") == settings_storage
    assert reloaded.io.get_dataset_storage_kind("PFAC_matrix") == pfac_storage
    np.testing.assert_allclose(
        reloaded.pfac_matrix, np.eye(simulation_data.solution_spec.index_length)
    )


def test_simulation_data_output_helpers_round_trip_entries(tmp_path):
    run_dir = tmp_path / "output_helpers"
    settings = DynamicsSettings(run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6)

    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    n_solution = simulation_data.solution_spec.index_length

    assert not simulation_data.has_dataset("state")

    simulation_data.add_output_entry(
        "state",
        {
            "m_ind": np.full(n_solution, 1.0),
            "m_imp": np.full(n_solution, 2.0),
            "Phi": np.full(n_solution, 3.0),
            "W": np.full(n_solution, 4.0),
        },
        time=0.0,
    )
    simulation_data.add_output_entry(
        "state",
        {
            "m_ind": np.full(n_solution, 5.0),
            "m_imp": np.full(n_solution, 6.0),
            "Phi": np.full(n_solution, 7.0),
            "W": np.full(n_solution, 8.0),
        },
        time=2.0,
    )
    simulation_data.save_output_dataset("state")

    reloaded = SimulationData.create(run_dir, settings, load_existing=True)
    assert reloaded.has_dataset("state")
    assert reloaded.get_latest_output_time("state") == 2.0
    state_entry = reloaded.get_output_entry("state", 2.0)
    assert state_entry is not None
    np.testing.assert_allclose(state_entry["m_imp"], np.full(n_solution, 6.0))


def test_simulation_data_input_helpers_round_trip_dataset(tmp_path):
    run_dir = tmp_path / "input_helpers"
    settings = DynamicsSettings(run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6)

    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    n_scalar = simulation_data.sh_basis.scalar_index_length(mean_free=True)

    simulation_data.input_timeseries.add_entry(
        "jr", {"jr": np.arange(n_scalar, dtype=float)}, time=0.0
    )
    simulation_data.save_input_dataset("jr")

    reloaded = SimulationData.create(run_dir, settings, load_existing=True)
    assert reloaded.has_dataset("jr")
    jr_entry = reloaded.get_input_entry("jr", 0.0)
    assert jr_entry is not None
    np.testing.assert_allclose(jr_entry["jr"], np.arange(n_scalar, dtype=float))


def test_simulation_data_builds_results_operator_bundle(tmp_path):
    run_dir = tmp_path / "results_ops"
    settings = DynamicsSettings(run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6)

    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    bundle = simulation_data.get_poloidal_results_operators(
        grid=Grid(lat=np.array([60.0]), lon=np.array([0.0]))
    )
    bundle_cached = simulation_data.get_poloidal_results_operators(
        grid=Grid(lat=np.array([60.0]), lon=np.array([0.0]))
    )

    assert bundle.m_ind_to_Br.shape[0] == simulation_data.sh_basis.scalar_index_length(
        mean_free=True
    )
    assert bundle.scalar_evaluation_matrix.shape[0] == 1
    assert bundle is bundle_cached
