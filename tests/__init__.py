"""Test suite package and shared helpers for PynaMIT."""

from __future__ import annotations

import numpy as np

DETERMINISTIC_REGRESSION_RTOL = 1e-10
SINGLE_PRECISION_REGRESSION_RTOL = 1e-5


def magnetic_potential_coordinate_array(simulation, output="dynamic"):
    """Return the former numerical coordinates for regression comparisons.

    The simulation now persists the physical ``induced_Br`` and
    ``boundary_jr`` fields. Historical numerical regression values were
    recorded in the private poloidal- and toroidal-potential coordinates,
    so this helper applies the exact inverse maps before comparing them.
    """
    dataset = simulation.run_data.output_series.datasets[output]
    induced_poloidal_potential = (
        simulation.geometry.induced_Br_to_poloidal_potential_operator.matvec(
            dataset["SH_induced_Br"].values[-1]
        )
    )
    boundary_jr_name = simulation.run_data.output_series.get_data_var_name(
        output, "boundary_jr"
    )
    toroidal_potential = (
        simulation.geometry.boundary_jr_to_toroidal_potential_operator.matvec(
            dataset[boundary_jr_name].values[-1]
        )
    )
    return np.hstack((induced_poloidal_potential, toroidal_potential))
