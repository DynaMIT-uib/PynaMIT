from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from pynamit.math.integration import ExponentialIntegrator, ScipySolveIVPIntegrator


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exponential_action_path_supplies_trace_for_small_linear_operator() -> None:

    integrator = ExponentialIntegrator()
    dense = np.array([[-2.0, 0.0], [0.0, -1.0]], dtype=float)

    op = LinearOperator(
        shape=dense.shape, matvec=lambda v: dense @ v, rmatvec=lambda v: dense.T @ v, dtype=float
    )
    y0 = np.array([1.0, -2.0], dtype=float)
    forcing = np.array([0.5, 1.0], dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result_action = integrator.step(
            y=y0,
            dt=0.25,
            linear_operator=op,
            forcing=forcing,
            affine_expm_mode="action",
            trace_dim_limit=8,
        )

    result_dense = integrator.step(
        y=y0, dt=0.25, linear_operator=dense, forcing=forcing, affine_expm_mode="dense"
    )

    assert result_action == pytest.approx(result_dense)
    assert not any("Trace of LinearOperator not available" in str(w.message) for w in caught)


def test_scipy_solve_ivp_integrator_raises_on_failure(monkeypatch) -> None:
    integrator = ScipySolveIVPIntegrator(method="DOP853")

    def failing_solve_ivp(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            success=False, status=-1, message="synthetic failure", y=np.zeros((2, 0), dtype=float)
        )

    monkeypatch.setattr("pynamit.math.integration.solve_ivp", failing_solve_ivp)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        integrator.step(
            y=np.array([1.0, 2.0], dtype=float), dt=1.0, rates_func=lambda y, t: -y + t
        )
