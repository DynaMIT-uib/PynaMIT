"""Regression tests for CS differential operators."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.cubed_sphere.cs_basis import CSBasis


def _rel_rms(err: np.ndarray, ref: np.ndarray) -> float:
    err = np.asarray(err)
    ref = np.asarray(ref)
    denom = np.sqrt(np.mean(ref**2))
    if denom <= 1e-14:
        return np.sqrt(np.mean(err**2))
    return np.sqrt(np.mean(err**2)) / denom


def _roughly_non_increasing(values: list[float], rel_slack: float = 0.10) -> bool:
    return all(values[i + 1] <= values[i] * (1.0 + rel_slack) for i in range(len(values) - 1))


def _harmonic_case(name: str, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, ...]:
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    c2p = np.cos(2.0 * phi)
    s2p = np.sin(2.0 * phi)

    if name == "l1m0":
        # f = Y_1,0 (up to normalization)
        f = ct
        d_theta = -st
        d_phi_scaled = np.zeros_like(theta)
        lap = -2.0 * f
        return f, d_theta, d_phi_scaled, lap

    if name == "l1m1":
        # f = Y_1,1 (real, up to normalization)
        f = st * cp
        d_theta = ct * cp
        d_phi_scaled = -sp
        lap = -2.0 * f
        return f, d_theta, d_phi_scaled, lap

    if name == "l2m0":
        # f = Y_2,0 (up to normalization)
        f = 0.5 * (3.0 * ct * ct - 1.0)
        d_theta = -3.0 * ct * st
        d_phi_scaled = np.zeros_like(theta)
        lap = -6.0 * f
        return f, d_theta, d_phi_scaled, lap

    if name == "l2m2":
        # f = Y_2,2 (real, up to normalization)
        f = st * st * c2p
        d_theta = 2.0 * st * ct * c2p
        d_phi_scaled = -2.0 * st * s2p
        lap = -6.0 * f
        return f, d_theta, d_phi_scaled, lap

    raise ValueError(f"Unknown harmonic case: {name}")


@pytest.mark.parametrize("name", ["l1m1", "l2m2"])
def test_cs_derivative_converges_for_nonzonal_harmonics(name: str) -> None:
    n_values = [10, 20, 30, 40]
    err_theta: list[float] = []
    err_phi_scaled: list[float] = []

    for n in n_values:
        basis = CSBasis(n)
        theta = np.deg2rad(basis.theta)
        phi = np.deg2rad(basis.phi)
        f, d_theta_true, d_phi_scaled_true, _ = _harmonic_case(name, theta, phi)

        d_theta_num = np.asarray(basis.get_evaluation_matrix(basis, derivative="theta") @ f)
        d_phi_scaled_num = np.asarray(basis.get_evaluation_matrix(basis, derivative="phi") @ f)

        err_theta.append(_rel_rms(d_theta_num - d_theta_true, d_theta_true))
        err_phi_scaled.append(_rel_rms(d_phi_scaled_num - d_phi_scaled_true, d_phi_scaled_true))

    assert _roughly_non_increasing(err_theta)
    assert _roughly_non_increasing(err_phi_scaled)
    assert err_theta[-1] < 2e-3
    assert err_phi_scaled[-1] < 2e-3


def test_cs_laplacian_harmonic_convergence() -> None:
    n_values = [10, 20, 30, 40]
    cases = ["l1m0", "l1m1", "l2m0", "l2m2"]
    errors: dict[str, list[float]] = {name: [] for name in cases}

    for n in n_values:
        basis = CSBasis(n)
        lap = basis.laplacian(r=1.0)
        theta = np.deg2rad(basis.theta)
        phi = np.deg2rad(basis.phi)

        for name in cases:
            f, _, _, lap_true = _harmonic_case(name, theta, phi)
            lap_num = np.asarray(lap @ f)
            errors[name].append(_rel_rms(lap_num - lap_true, lap_true))

    for name in cases:
        assert _roughly_non_increasing(errors[name])

    # m!=0 regressions that previously plateaued must now converge to low error.
    assert errors["l1m1"][0] > 5.0 * errors["l1m1"][-1]
    assert errors["l2m2"][0] > 5.0 * errors["l2m2"][-1]
    assert errors["l1m1"][-1] < 3e-3
    assert errors["l2m2"][-1] < 3e-3

    # m=0 accuracy should remain strong.
    assert errors["l1m0"][-1] < 3e-3
    assert errors["l2m0"][-1] < 5e-3


@pytest.mark.parametrize("n", [8, 10])
def test_cs_operator_identities_match_laplacian(n: int) -> None:
    """Ensure internal CS operator algebra remains coherent."""
    basis = CSBasis(n)
    n_coeff = basis.index_length
    rng = np.random.default_rng(0)
    state = rng.standard_normal(n_coeff)

    div_op = basis.get_vector_divergence_operator(basis.grid).to_dense()
    grad_op = basis.get_gradient_operator(r=1.0).to_dense()
    curl_vec_op = basis.get_vector_curl_operator(basis.grid).to_dense()
    curl_op = basis.get_curl_operator(r=1.0).to_dense()
    lap = basis.laplacian(r=1.0)

    lhs_div_grad = div_op @ (grad_op @ state)
    lhs_curl_curl = curl_vec_op @ (curl_op @ state)
    rhs = -np.asarray(lap @ state)

    rel_div = _rel_rms(lhs_div_grad - rhs, rhs)
    rel_curl = _rel_rms(lhs_curl_curl - rhs, rhs)

    assert rel_div < 1e-12, f"Div(Grad) identity mismatch: {rel_div:.3e}"
    assert rel_curl < 1e-12, f"Curl(CurlOp) identity mismatch: {rel_curl:.3e}"
