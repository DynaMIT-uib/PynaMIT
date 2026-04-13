"""Shared magnetospheric boundary operator bundles.

These helpers expose the finite-``R_M`` magnetic boundary data in one place so
the runtime and theory note can refer to the same outer-boundary pair:

- toroidal magnetic trace ``psi^+(R_M)``
- poloidal magnetic scalar reconstructed from ``B_r(R_M)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MagneticRMBoundaryOperators:
    """Finite-``R_M`` magnetic boundary operators in solution coefficient space.

    The preferred explicit outer magnetic data are:

    - ``psi^+(R_M)`` on the toroidal branch
    - a poloidal magnetic scalar at ``R_M`` reconstructed from ``B_r(R_M)``

    The bundle therefore exposes both the raw ``B_r(R_M)`` operators and the
    deterministic ``B_r(R_M) <-> M(R_M)`` transforms used to recover the
    boundary magnetic scalar.
    """

    alpha_to_boundary_psi_rm: np.ndarray
    magnetic_potential_rm_to_br_rm: np.ndarray
    br_rm_to_magnetic_potential_rm: np.ndarray
    m_ind_to_br_rm_open: np.ndarray
    m_ind_to_br_rm_effective: np.ndarray
    m_ind_to_br_rm_shielding: np.ndarray
    m_ind_to_magnetic_potential_rm_open: np.ndarray
    m_ind_to_magnetic_potential_rm_effective: np.ndarray
    m_ind_to_magnetic_potential_rm_shielding: np.ndarray
