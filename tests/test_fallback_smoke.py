import os
import tempfile

import numpy as np
import pytest

from pynamit.default_run import run_pynamit


def test_run_with_fallback_inputs(data_source):
    if data_source != "fallback":
        pytest.skip("Fallback smoke test only runs with fallback inputs.")

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    os.makedirs(temp_dir, exist_ok=True)

    dynamics = run_pynamit(
        final_time=0.05,
        dt=1e-2,
        Nmax=6,
        Mmax=6,
        Ncs=12,
        mainfield_kind="dipole",
        fig_directory=temp_dir,
        wind=True,
        steady_state_initialization=False,
    )

    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
        )
    )

    assert np.isfinite(np.linalg.norm(coeff_array))
