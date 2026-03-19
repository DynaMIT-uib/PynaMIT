"""Ground magnetic response operators for postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pynamit.math.constants import RE
from pynamit.primitives.grid import Grid


@dataclass(frozen=True)
class GroundMagneticResponseOperators:
    """Explicit operators mapping induced coefficients to ground magnetic response."""

    radial_matrix: np.ndarray
    horizontal_matrix: np.ndarray

    def evaluate_radial(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate ground radial magnetic perturbation."""
        coeff_vec = np.asarray(m_ind).reshape(-1)
        return np.asarray(self.radial_matrix @ coeff_vec)

    def evaluate_horizontal(self, m_ind: np.ndarray) -> np.ndarray:
        """Evaluate ground tangential magnetic perturbation ``(theta, phi)``."""
        coeff_vec = np.asarray(m_ind).reshape(-1)
        return np.asarray(np.tensordot(self.horizontal_matrix, coeff_vec, axes=([2], [0])))


def build_ground_magnetic_response_operators(
    *,
    state_spec: Any,
    ground_grid: Grid,
    ionosphere_radius: float,
    ground_radius: float = RE,
) -> GroundMagneticResponseOperators:
    """Build explicit operators mapping ``m_ind`` to ground magnetic response.

    ``state_spec`` is expected to describe the solution scalar space used for
    poloidal induced coefficients.
    """
    ve_to_ground = np.asarray(
        state_spec.radial_shift_Ve(ionosphere_radius, ground_radius),
        dtype=float,
    ).reshape(-1)
    m_ind_to_br = np.asarray(
        -(float(ionosphere_radius) ** 2) * state_spec.laplacian(ionosphere_radius),
        dtype=float,
    ).reshape(-1)
    radial_matrix = np.asarray(
        state_spec.get_evaluation_matrix(ground_grid),
        dtype=float,
    ) * (ve_to_ground * m_ind_to_br)[None, :]

    # ``m_ind`` follows the induced ``k_nm, l_nm`` convention. Below the
    # ionosphere the field is ``B = -grad(V^e)`` with
    # ``V^e_nm = -(n+1) * (r/R)^n * m_ind_nm`` up to the common SH basis
    # normalization. Differentiating the extra minus sign in ``V^e`` against
    # the physical ``B = -grad(V^e)`` leaves the tangential field with the
    # positive modal factor ``+(n+1)`` here.
    horizontal_scale = np.asarray(state_spec.n, dtype=float).reshape(-1) + 1.0
    horizontal_scale = horizontal_scale * ve_to_ground
    horizontal_matrix = np.asarray(
        state_spec.get_gradient_matrix(ground_grid),
        dtype=float,
    ) * horizontal_scale[None, None, :]

    return GroundMagneticResponseOperators(
        radial_matrix=radial_matrix,
        horizontal_matrix=horizontal_matrix,
    )
