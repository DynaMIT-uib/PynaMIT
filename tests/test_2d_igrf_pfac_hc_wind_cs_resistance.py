"""IGRF, PFAC, HC, and wind test with CS resistance storage."""

import numpy as np

from pynamit.simulation.workflows.standard import run_pynamit


def _final_state_coefficients(simulation):
    """Return concatenated final magnetic state coefficients."""
    return np.hstack(
        (
            simulation.run_data.output_series.datasets["state"]["SH_m_ind"].values[-1],
            simulation.run_data.output_series.datasets["state"]["SH_m_imp"].values[-1],
        )
    )


def test_2d_igrf_pfac_hc_wind_cs_resistance_basis():
    """CS-stored resistance stays close to the SH baseline."""
    common_kwargs = dict(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="igrf",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        steady_state_initialization=False,
    )

    sh_resistance = run_pynamit(**common_kwargs)
    cs_resistance = run_pynamit(resistance_projection_basis="CS", **common_kwargs)

    sh_coeffs = _final_state_coefficients(sh_resistance)
    cs_coeffs = _final_state_coefficients(cs_resistance)
    relative_difference = np.linalg.norm(cs_coeffs - sh_coeffs) / np.linalg.norm(sh_coeffs)

    assert "CS_etaP" in cs_resistance.run_data.input_series.datasets["resistance"]
    assert "CS_etaH" in cs_resistance.run_data.input_series.datasets["resistance"]
    assert (
        cs_resistance.run_data.schema.input_field_spaces["resistance"].representation
        is cs_resistance.run_data.schema.cs_basis
    )
    assert cs_coeffs.shape == sh_coeffs.shape
    assert relative_difference < 0.25
