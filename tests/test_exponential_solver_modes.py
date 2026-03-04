"""Regression tests for explicit exponential solver mode combinations."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.simulation.dynamics import SimulationMode
from pynamit.utils import JAX_AVAILABLE


def _run_full_induction_exponential_mode(tmp_path, *, dense_full_operators: bool, exponential_solver: str):
    from pynamit.simulation.runner import run_pynamit

    sim = run_pynamit(
        run_directory=str(tmp_path / f"exp_mode_{exponential_solver}_{'dense' if dense_full_operators else 'mf'}"),
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=5,
        Mmax=2,
        Ncs=6,
        dynamics_mode="full_induction",
        simulation_mode=SimulationMode.PURE_SPECTRAL.value,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        multi_data=False,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator="exponential",
        dense_full_operators=dense_full_operators,
        exponential_solver=exponential_solver,
    )

    ds = sim.io.load_dataset("state")
    psi = ds["SH_psi"].values[-1]
    m_ind = ds["SH_m_ind"].values[-1]
    return float(np.linalg.norm(psi)), float(np.linalg.norm(m_ind))


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize(
    "label,dense_full_operators,exponential_solver,expected_psi_norm,expected_mind_norm",
    [
        (
            "dense_expm",
            True,
            "expm",
            5.048226712788428e-10,
            1.6874470337361023e-09,
        ),
        (
            "dense_expm_multiply",
            True,
            "expm_multiply",
            5.048226712788427e-10,
            1.6874470337361019e-09,
        ),
        (
            "matrixfree_expm_multiply",
            False,
            "expm_multiply",
            5.048226712788428e-10,
            1.6874470337361023e-09,
        ),
    ],
)
def test_full_induction_exponential_solver_modes(
    tmp_path,
    data_source,
    label,
    dense_full_operators,
    exponential_solver,
    expected_psi_norm,
    expected_mind_norm,
):
    if label == "matrixfree_expm_multiply":
        with pytest.warns(UserWarning, match="Trace of LinearOperator not available"):
            psi_norm, mind_norm = _run_full_induction_exponential_mode(
                tmp_path,
                dense_full_operators=dense_full_operators,
                exponential_solver=exponential_solver,
            )
    else:
        psi_norm, mind_norm = _run_full_induction_exponential_mode(
            tmp_path,
            dense_full_operators=dense_full_operators,
            exponential_solver=exponential_solver,
        )

    assert psi_norm == pytest.approx(expected_psi_norm, rel=1e-10, abs=0.0)
    assert mind_norm == pytest.approx(expected_mind_norm, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exponential_solver_requires_dense_operators_for_expm(tmp_path):
    from pynamit.simulation.runner import run_pynamit

    with pytest.raises(
        ValueError,
        match="full_induction'.*exponential_solver='expm'.*dense_full_operators=True",
    ):
        run_pynamit(
            run_directory=str(tmp_path / "exp_invalid_combo"),
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=3,
            Mmax=1,
            Ncs=4,
            dynamics_mode="full_induction",
            simulation_mode=SimulationMode.PURE_SPECTRAL.value,
            least_squares_solver="svd",
            integrator="exponential",
            dense_full_operators=False,
            exponential_solver="expm",
        )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_expm_multiply_not_supported_for_jax_backend(tmp_path):
    from pynamit.simulation.runner import run_pynamit

    with pytest.raises(NotImplementedError, match="expm_multiply"):
        run_pynamit(
            run_directory=str(tmp_path / "exp_jax_expm_multiply"),
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=5,
            Mmax=2,
            Ncs=6,
            dynamics_mode="full_induction",
            simulation_mode=SimulationMode.PURE_SPECTRAL.value,
            least_squares_solver="svd",
            integrator="exponential",
            dense_full_operators=False,
            exponential_solver="expm_multiply",
        )
