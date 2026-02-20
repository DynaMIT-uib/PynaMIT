"""Regression checks for HL/LL concentration split in full-induction mode."""

from __future__ import annotations

import numpy as np

from pynamit.simulation.dynamics import Dynamics, SimulationMode


def _relative_rank(mat: np.ndarray, rtol: float = 1e-10) -> int:
    """Compute numerical rank using a relative SVD cutoff."""
    svals = np.linalg.svd(mat, compute_uv=False)
    if svals.size == 0:
        return 0
    return int(np.sum(svals > (rtol * float(svals[0]))))


def _concentration_fraction(Q: np.ndarray, A_region: np.ndarray, A_total: np.ndarray) -> float:
    """Return Tr(Q^T A_region Q) / Tr(Q^T A_total Q)."""
    num = float(np.trace(Q.T @ A_region @ Q))
    den = float(np.trace(Q.T @ A_total @ Q))
    if den <= 0.0:
        return 0.0
    return num / den


def _assert_split_regression_properties(
    *,
    tmp_path,
    simulation_mode: SimulationMode,
    expected_hl_modes: int,
    expected_ll_modes: int,
    hl_frac_min: float,
    ll_frac_min: float,
) -> None:
    """Lock core HL/LL split properties used by hard full-induction constraints."""
    dynamics = Dynamics(
        filename_prefix=str(tmp_path / f"split_regression_{simulation_mode.value}"),
        Nmax=10,
        Mmax=5,
        Ncs=10,
        dynamics_mode="full_induction",
        simulation_mode=simulation_mode,
        ignore_PFAC=False,
        connect_hemispheres=True,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        least_squares_solver="svd",
    )

    state = dynamics.state
    ll_mask = np.asarray(state.geometry.ll_mask, dtype=bool).reshape(-1)
    n_coeffs = int(state.geometry.jr_map_sim.shape[1])
    Q_hl, Q_ll = state._build_hl_ll_subspaces(n_coeffs=n_coeffs, ll_mask=ll_mask)

    # Stable mode counts for the reference setup (Nmax=10, Mmax=5, lat boundary=50 deg).
    assert Q_hl.shape == (n_coeffs, expected_hl_modes)
    assert Q_ll.shape == (n_coeffs, expected_ll_modes)
    assert (n_coeffs - Q_hl.shape[1] - Q_ll.shape[1]) > 0

    # Concentration modes are orthonormal in the apex Gram metric.
    weights = np.asarray(state.geometry.grid.weights).reshape(-1)
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    M_total, M_hl = state._build_apex_metric_pair(ll_mask=ll_mask, weights=weights, n_coeffs=n_coeffs)
    assert M_total is not None
    assert M_hl is not None
    M_total = np.asarray(M_total)
    np.testing.assert_allclose(Q_hl.T @ M_total @ Q_hl, np.eye(Q_hl.shape[1]), atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(Q_ll.T @ M_total @ Q_ll, np.eye(Q_ll.shape[1]), atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(Q_hl.T @ M_total @ Q_ll, 0.0, atol=1e-8, rtol=0.0)

    # HL and LL spans should still be linearly independent in Euclidean space.
    Q_cat = np.hstack([Q_hl, Q_ll])
    assert _relative_rank(Q_cat, rtol=1e-10) == Q_cat.shape[1]

    # Check concentration quality in the same metric used to construct the split.
    M_ll = 0.5 * ((M_total - M_hl) + (M_total - M_hl).T)

    hl_frac = _concentration_fraction(Q_hl, M_hl, M_total)
    ll_frac = _concentration_fraction(Q_ll, M_ll, M_total)
    assert hl_frac > hl_frac_min
    assert ll_frac > ll_frac_min

    bundle = state.induction_constraint_bundle_hard
    assert bundle is not None
    # HL residual lock is intentionally disabled; only LL hard constraints remain.
    assert bundle["C_hl"].shape[0] == 0
    assert bundle["C_ll"].shape[0] > 0
    assert bundle["C_total"].shape[1] == n_coeffs

    # LL mismatch constraints should suppress HL-only vectors.
    rng = np.random.default_rng(0)
    if Q_hl.shape[1] > 0:
        z_hl = rng.standard_normal(Q_hl.shape[1])
        x_hl = Q_hl @ z_hl
        hl_misfit = np.linalg.norm(bundle["C_ll"] @ x_hl)
        assert hl_misfit < 1e-6 * np.linalg.norm(x_hl)


def test_hl_ll_split_regression_properties_pure_spectral(tmp_path) -> None:
    _assert_split_regression_properties(
        tmp_path=tmp_path,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        expected_hl_modes=21,
        expected_ll_modes=64,
        hl_frac_min=0.75,
        ll_frac_min=0.95,
    )


def test_hl_ll_split_regression_properties_spectral_transform_cs(tmp_path) -> None:
    _assert_split_regression_properties(
        tmp_path=tmp_path,
        simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_hl_modes=24,
        expected_ll_modes=64,
        hl_frac_min=0.75,
        ll_frac_min=0.95,
    )
