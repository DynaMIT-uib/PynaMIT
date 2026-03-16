"""Dipole, PFAC and exponential test."""

import pytest

from pynamit.simulation.runner import run_pynamit
import numpy as np
from pynamit.math.integration import ExponentialIntegrator
from pynamit.simulation.settings import IntegratorKind, MainfieldKind, SimulationMode


def test_2d_dipole_pfac_exp():
    """Test 2D simulation with dipole, PFAC and exponential."""
    # Arrange.
    expected_coeff_norm = 1.1342052545869683e-08
    expected_coeff_max = 8.006258968163613e-10
    expected_coeff_min = -5.063807785683825e-09
    expected_n_coeffs = 240

    # Act.
    dynamics = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=10,
        Ncs=20,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        integrator=IntegratorKind.EXPONENTIAL,
        steady_state_initialization=False,
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

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)


def test_legacy_exponential_freezes_forcing_between_steps(monkeypatch):
    """Legacy exponential stepping should evolve around the frozen steady state."""
    original_step = ExponentialIntegrator.step
    seen_calls = []

    def recording_step(self, y, dt, **kwargs):
        seen_calls.append(
            {
                "forcing_is_none": kwargs.get("forcing") is None,
                "has_steady_state": kwargs.get("steady_state") is not None,
            }
        )
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(ExponentialIntegrator, "step", recording_step)

    run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=5,
        Mmax=5,
        Ncs=10,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        integrator=IntegratorKind.EXPONENTIAL,
        steady_state_initialization=False,
    )

    assert seen_calls
    assert seen_calls[0]["forcing_is_none"] is True
    assert seen_calls[0]["has_steady_state"] is True


@pytest.mark.parametrize(
    "simulation_mode", [SimulationMode.PURE_SPECTRAL, SimulationMode.CS_DOMINANT]
)
def test_legacy_exponential_steady_state_matches_affine_step(tmp_path, simulation_mode):
    """Legacy frozen-step steady-state form should match the affine form.

    This is the key distinction from the coupled full-induction case: the
    gauge-constrained legacy `m_ind` steady state is consistent with the exact
    stepped reduced operator, including `CS_DOMINANT`.
    """
    run_directory = tmp_path / f"legacy_exp_consistency_{simulation_mode.value}"
    dynamics = run_pynamit(
        run_directory=str(run_directory),
        final_time=0.0,
        dt=0.1,
        Nmax=6,
        Mmax=4,
        Ncs=10,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        integrator=IntegratorKind.EXPONENTIAL,
        steady_state_initialization=False,
        simulation_mode=simulation_mode,
        benchmark_mode=True,
    )

    state = dynamics.state
    E_coeffs_noind, _ = state.calculate_noind_coeffs()
    _, steady_state_m_ind = state.solve_steady_state_model_variables(
        E_coeffs_noind, update_state=False
    )

    scale = state.poloidal_matrices.E_df_to_d_m_ind_dt
    full_linear_operator = np.asarray(scale * state.m_ind_to_E_df_matrix, dtype=float)
    reduced_system = state.get_m_ind_reduced_system(linear_operator=full_linear_operator)
    linear_operator = reduced_system.reduced_operator
    assert linear_operator is not None

    forcing = reduced_system.reduce_vector(
        np.asarray(
            scale
            * state.poloidal_matrices.solution_space.get_toroidal_potential_coeffs(E_coeffs_noind),
            dtype=float,
        ).reshape(-1)
    )
    steady_state = reduced_system.reduce_vector(
        np.asarray(steady_state_m_ind, dtype=float).reshape(-1)
    )

    residual = np.linalg.norm(linear_operator.matvec(steady_state) + forcing) / max(
        np.linalg.norm(forcing), 1e-30
    )

    y0 = np.random.default_rng(0).standard_normal(steady_state.shape)
    integrator = ExponentialIntegrator()
    affine_step = np.asarray(
        integrator.step(
            y=y0,
            dt=0.1,
            linear_operator=linear_operator,
            forcing=forcing,
            affine_expm_mode="dense",
        )
    )
    centered_step = np.asarray(
        integrator.step(
            y=y0,
            dt=0.1,
            linear_operator=linear_operator,
            steady_state=steady_state,
            affine_expm_mode="dense",
        )
    )

    assert residual < 1e-7
    assert np.linalg.norm(affine_step - centered_step) < 1e-11


def test_legacy_exponential_cs_dominant_uses_reduced_coordinates(tmp_path, monkeypatch):
    """Legacy CS-dominant exponential stepping should evolve in reduced coordinates."""
    from pynamit.simulation.state import State

    original_step = ExponentialIntegrator.step
    original_system = State.get_m_ind_reduced_system
    seen: dict[str, int | list[int]] = {}

    def recording_system(self, *args, **kwargs):
        system = original_system(self, *args, **kwargs)
        seen["n_total"] = system.n_total
        seen["n_reduced"] = system.n_reduced
        return system

    def recording_step(self, y, dt, **kwargs):
        seen.setdefault("step_sizes", []).append(int(np.asarray(y).size))
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(State, "get_m_ind_reduced_system", recording_system)
    monkeypatch.setattr(ExponentialIntegrator, "step", recording_step)

    run_pynamit(
        run_directory=str(tmp_path / "legacy_exp_reduced_cs"),
        final_time=0.1,
        dt=0.1,
        Nmax=5,
        Mmax=4,
        Ncs=10,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        integrator=IntegratorKind.EXPONENTIAL,
        steady_state_initialization=False,
        simulation_mode=SimulationMode.CS_DOMINANT,
    )

    assert seen["step_sizes"]
    assert seen["n_reduced"] < seen["n_total"]
    assert seen["step_sizes"][0] == seen["n_reduced"]


def test_legacy_exponential_cs_dominant_uses_frozen_steady_state(monkeypatch):
    """Legacy CS-dominant exponential stepping should still use the centered form."""
    original_step = ExponentialIntegrator.step
    seen_calls = []

    def recording_step(self, y, dt, **kwargs):
        seen_calls.append(
            {
                "forcing_is_none": kwargs.get("forcing") is None,
                "has_steady_state": kwargs.get("steady_state") is not None,
            }
        )
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(ExponentialIntegrator, "step", recording_step)

    run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=5,
        Mmax=4,
        Ncs=10,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        integrator=IntegratorKind.EXPONENTIAL,
        steady_state_initialization=False,
        simulation_mode=SimulationMode.CS_DOMINANT,
    )

    assert seen_calls
    assert seen_calls[0]["forcing_is_none"] is True
    assert seen_calls[0]["has_steady_state"] is True
