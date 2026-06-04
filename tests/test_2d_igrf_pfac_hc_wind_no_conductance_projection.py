"""IGRF, PFAC, HC, and wind test without conductance projection."""

import numpy as np

from pynamit.default_run import run_pynamit


def _final_state_coefficients(dynamics):
    """Return concatenated final magnetic state coefficients."""
    return np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
        )
    )


def test_2d_igrf_pfac_hc_wind_no_conductance_projection():
    """Grid-stored conductance stays close to the baseline."""
    common_kwargs = dict(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind="igrf",
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        use_wind=True,
        steady_state_initialization=False,
    )

    projected = run_pynamit(**common_kwargs)
    direct_grid = run_pynamit(project_conductance=False, **common_kwargs)

    projected_coeffs = _final_state_coefficients(projected)
    direct_grid_coeffs = _final_state_coefficients(direct_grid)
    relative_difference = np.linalg.norm(direct_grid_coeffs - projected_coeffs) / np.linalg.norm(
        projected_coeffs
    )

    assert "CS_etaP" in direct_grid.input_timeseries.datasets["conductance"]
    assert "CS_etaH" in direct_grid.input_timeseries.datasets["conductance"]
    assert direct_grid.input_field_spaces["conductance"].representation is direct_grid.cs_basis
    assert direct_grid_coeffs.shape == projected_coeffs.shape
    assert relative_difference < 0.25
