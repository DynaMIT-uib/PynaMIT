"""IGRF, PFAC, HC, and wind test with CS conductance storage."""

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


def test_2d_igrf_pfac_hc_wind_cs_conductance_basis():
    """CS-stored conductance stays close to the SH baseline."""
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

    sh_conductance = run_pynamit(**common_kwargs)
    cs_conductance = run_pynamit(conductance_projection_basis="CS", **common_kwargs)

    sh_coeffs = _final_state_coefficients(sh_conductance)
    cs_coeffs = _final_state_coefficients(cs_conductance)
    relative_difference = np.linalg.norm(cs_coeffs - sh_coeffs) / np.linalg.norm(sh_coeffs)

    assert "CS_etaP" in cs_conductance.input_timeseries.datasets["conductance"]
    assert "CS_etaH" in cs_conductance.input_timeseries.datasets["conductance"]
    assert (
        cs_conductance.input_field_spaces["conductance"].representation is cs_conductance.cs_basis
    )
    assert cs_coeffs.shape == sh_coeffs.shape
    assert relative_difference < 0.25
