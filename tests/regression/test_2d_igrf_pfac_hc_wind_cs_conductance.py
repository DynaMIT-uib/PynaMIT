"""IGRF, PFAC, HC, and wind test with CS resistance storage."""

import numpy as np
from tests import magnetic_potential_coordinate_array
from tests.example_scenario import run_example


def _final_magnetic_coordinates(simulation):
    """Return concatenated final private magnetic coordinates."""
    return magnetic_potential_coordinate_array(simulation)


def test_2d_igrf_pfac_hc_wind_cs_resistance_basis():
    """CS-stored conductance stays close to the SH baseline."""
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
        initialize_from_equilibrium=False,
    )

    sh_resistance = run_example(**common_kwargs)
    cs_resistance = run_example(conductance_projection_basis="CS", **common_kwargs)

    sh_coeffs = _final_magnetic_coordinates(sh_resistance)
    cs_coeffs = _final_magnetic_coordinates(cs_resistance)
    relative_difference = np.linalg.norm(cs_coeffs - sh_coeffs) / np.linalg.norm(sh_coeffs)

    assert (
        "CS_log_conductance_magnitude" in cs_resistance.data.input_series.datasets["conductance"]
    )
    assert (
        "CS_log_hall_to_pedersen_ratio" in cs_resistance.data.input_series.datasets["conductance"]
    )
    assert (
        cs_resistance.data.schema.input_field_spaces["conductance"].basis
        is cs_resistance.data.schema.cs_basis
    )
    assert cs_coeffs.shape == sh_coeffs.shape
    assert relative_difference < 0.25
