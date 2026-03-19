"""End-to-end simulation test using CSBasis."""

import pytest
import numpy as np

from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import (
    IntegratorKind,
    MainfieldKind,
    SimulationMode,
    SolutionBasisKind,
)


@pytest.mark.wind
def test_cs_basis_simulation_dop853(pynamit_approx, data_source):
    """Test full simulation with CSBasis as solution basis.

    Uses a balanced resolution (Nmax=14, Mmax=12, Ncs=8) where SH modes (218)
    exceed half the CS coefficients (384), ensuring the m_imp least-squares
    problem is near-full rank (383/384). Achieves numpy/JAX consistency ~1e-12.

    Note: Unbalanced configs (e.g., Ncs=18 with Nmax=10) create large null spaces
    because FAC physics flows through the SH bottleneck, leaving most CS directions
    unconstrained.
    """
    # Expected values averaged between numpy and JAX backends.
    # Updated 2026-03-19 after aligning CS tangential fields with the active
    # Helmholtz-potential representation and using the projected CS toroidal
    # electric potential in the induction forcing path. These values are
    # invariant under the repo cf/df sign choice.
    expected_coeff_norm = 3.5652303104310223e-07
    expected_coeff_max = 1.095561287096179e-07
    expected_coeff_min = -1.0754867720509506e-07
    expected_n_coeffs = 768

    # Using Northern Apex Constraints improved values (slightly higher energy due to different constraint)
    # The previous legacy values were ~8.01e-7.

    # We use small integration time for speed.
    final_time = 0.003
    dt = 0.001

    # Act.
    # Run simulation with solution_basis_kind=SolutionBasisKind.CS
    # Balanced resolution: SH modes (218) sufficient for near-full rank
    dynamics = run_pynamit(
        final_time=final_time,
        dt=dt,
        Nmax=14,
        Mmax=12,
        Ncs=8,
        mainfield_kind=MainfieldKind.IGRF,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=False,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        integrator=IntegratorKind.EULER,
        multi_data=True,
        simulation_mode=SimulationMode.CS_DOMINANT,
        northern_hemisphere_apex_constraints=True,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["CS_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["CS_m_imp"].values[-1],
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

    # Uses standard tolerance: 1e-9 for fallback, 1e-5 for native
    assert actual_coeff_norm == pynamit_approx(expected_coeff_norm)
    assert actual_coeff_max == pynamit_approx(expected_coeff_max)
    assert actual_coeff_min == pynamit_approx(expected_coeff_min)
    assert actual_n_coeffs == expected_n_coeffs
