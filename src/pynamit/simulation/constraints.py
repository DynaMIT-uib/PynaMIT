"""Apex coordinate and constraint mapping module.

This module handles transformations between geographic and magnetic apex
coordinates, and constructs operators for interhemispheric constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.field import Field
    from pynamit.primitives.grid import Grid


logger = logging.getLogger(__name__)


@dataclass
class ConstraintMappings:
    """Container for constraint-related operators.

    Attributes
    ----------
    jr_map_spectral : np.ndarray
        Operator mapping radial current to apex current (spectral basis).
    jr_map_sim : np.ndarray
        Operator mapping radial current to apex current (simulation basis).
    E_coeffs_to_E_apex_ll_diff : np.ndarray or None
        E-field difference operator for low-latitude interhemispheric constraint.
    ll_mask : np.ndarray
        Boolean mask for low-latitude points.
    """

    jr_map_spectral: np.ndarray
    jr_map_sim: np.ndarray
    E_coeffs_to_E_apex_ll_diff: Optional[np.ndarray]
    ll_mask: np.ndarray


class ApexMapper:
    """Handles apex coordinate transformations and constraint mappings.

    This class encapsulates the geometric transformations needed to map
    between geographic and magnetic apex coordinate systems, and constructs
    operators for enforcing interhemispheric constraints.

    Parameters
    ----------
    mainfield : Any
        The main magnetic field model.
    basis : Basis
        The spectral basis for physics computations.
    latitude_boundary : float
        Latitude boundary for low-latitude region (degrees).
    connect_hemispheres : bool
        Whether to connect hemispheres via constraints.
    """

    def __init__(
        self,
        mainfield: Any,
        basis: "Basis",
        latitude_boundary: float,
        connect_hemispheres: bool,
    ) -> None:
        self.mainfield = mainfield
        self.basis = basis
        self.latitude_boundary = latitude_boundary
        self.connect_hemispheres = connect_hemispheres

    def get_transformation_matrices(
        self, dfield: "Field"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute transformation matrices for apex coordinates.

        Parameters
        ----------
        dfield : Field
            Discretized magnetic field.

        Returns
        -------
        radial_to_apex : np.ndarray
            Transformation from radial to apex coordinates (N,).
        horizontal_to_apex : np.ndarray
            Transformation from horizontal to apex coordinates (2, 2, N).
        """
        # Get basis vectors
        r_eval = dfield.r_loc if dfield.r_loc is not None else 1.0
        bv = dfield.basis_vectors(r_eval, dfield.grid.theta, dfield.grid.phi)
        d3 = bv[2]  # (3, N) e.g. [d3r, d3th, d3ph]
        e1 = bv[3]
        e2 = bv[4]

        # Get unit vector components
        mag = dfield.magnitude
        br, btheta, bphi = (
            dfield.vec.r / mag,
            dfield.vec.theta / mag,
            dfield.vec.phi / mag,
        )

        # Ensure flattened vectors
        br, btheta, bphi = br.flatten(), btheta.flatten(), bphi.flatten()
        ones = np.ones_like(br)
        zeros = np.zeros_like(br)

        # Radial to apex mapping
        # Regularize division by br to handle magnetic equator (where br→0)
        EPS = 1e-10
        br_safe = np.where(np.abs(br) < EPS, np.sign(br + EPS) * EPS, br)
        ratio_theta = btheta / br_safe
        ratio_phi = bphi / br_safe

        # Transformation: proj_radial = d3 . r_hat_sph
        rad_to_fp = np.stack([ones, ratio_theta, ratio_phi], axis=0)
        fp_to_apex = np.array(d3)
        radial_to_apex = np.sum(fp_to_apex * rad_to_fp, axis=0)

        # Horizontal to apex mapping
        c1 = np.stack([-ratio_theta, ones, zeros], axis=0)
        c2 = np.stack([-ratio_phi, zeros, ones], axis=0)
        horiz_to_fo = np.stack([c1, c2], axis=1)  # (3, 2, N)

        # Field-orthogonal basis to apex
        fo_to_apex = np.stack([e1, e2], axis=0)  # (2, 3, N)

        # Compose mapping: (2, 3, N) @ (3, 2, N) -> (2, 2, N)
        horizontal_to_apex = np.einsum(
            "ikn,kjn->ijn", fo_to_apex, horiz_to_fo, optimize=True
        )

        return radial_to_apex, horizontal_to_apex

    def create_apex_operators(
        self, field: "Field", grid: "Grid"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create current and field mapping operators for a given field/grid pair.

        Parameters
        ----------
        field : Field
            Discretized magnetic field.
        grid : Grid
            Spatial grid.

        Returns
        -------
        jr_op : np.ndarray
            Operator mapping radial current coefficients to apex current.
        E_op : np.ndarray
            Operator mapping E-field coefficients to apex E-field.
        """
        radial_to_apex, horizontal_to_apex = self.get_transformation_matrices(field)

        # jr_coeffs_to_j_apex
        jr_op = self.basis.get_scaled_matrix(grid, radial_to_apex)

        # E_coeffs_to_E_apex
        E_op = np.einsum(
            "ijk,jklm->iklm",
            horizontal_to_apex,
            self.basis.get_vector_basis_matrix(grid),
            optimize=True,
        )
        return jr_op, E_op

    def build_constraint_mappings(
        self,
        grid: "Grid",
        b_field: "Field",
        RI: float,
        cp_grid: Optional["Grid"] = None,
        cp_b_field: Optional["Field"] = None,
        input_adapter: Optional[np.ndarray] = None,
    ) -> ConstraintMappings:
        """Build all constraint-related operators.

        Parameters
        ----------
        grid : Grid
            Main hemisphere grid.
        b_field : Field
            Main hemisphere magnetic field.
        RI : float
            Ionosphere radius.
        cp_grid : Grid, optional
            Conjugate hemisphere grid.
        cp_b_field : Field, optional
            Conjugate hemisphere magnetic field.
        input_adapter : np.ndarray, optional
            Adapter for hybrid basis transformation.

        Returns
        -------
        ConstraintMappings
            Container with all constraint operators.
        """
        # Compute low-latitude mask
        ll_mask = self._compute_ll_mask(grid, RI)

        # Main hemisphere operators
        jr_map_spectral, E_coeffs_to_E_apex = self.create_apex_operators(b_field, grid)

        E_coeffs_to_E_apex_ll_diff = None

        if self.connect_hemispheres and cp_grid is not None and cp_b_field is not None:
            # Conjugate hemisphere operators
            jr_coeffs_to_j_apex_cp, E_coeffs_to_E_apex_cp = self.create_apex_operators(
                cp_b_field, cp_grid
            )

            # Apply correction in spectral domain
            jr_map_spectral[ll_mask] -= jr_coeffs_to_j_apex_cp[ll_mask]

            # Create E-field mapping difference operator for constraint
            E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, ll_mask]
            )

            if input_adapter is not None:
                # Pre-compose with analysis matrix (SH->Grid)
                logger.info("Pre-composing hybrid constraint operator (SH->Grid)...")
                E_coeffs_to_E_apex_ll_diff = np.tensordot(
                    E_coeffs_to_E_apex_ll_diff, input_adapter, axes=([-1], [0])
                )

        # Compute simulation operator
        if input_adapter is not None:
            jr_map_sim = jr_map_spectral @ input_adapter
        else:
            jr_map_sim = jr_map_spectral

        return ConstraintMappings(
            jr_map_spectral=jr_map_spectral,
            jr_map_sim=jr_map_sim,
            E_coeffs_to_E_apex_ll_diff=E_coeffs_to_E_apex_ll_diff,
            ll_mask=ll_mask,
        )

    def _compute_ll_mask(self, grid: "Grid", RI: float) -> np.ndarray:
        """Compute low-latitude mask based on mainfield type.

        Parameters
        ----------
        grid : Grid
            Spatial grid.
        RI : float
            Ionosphere radius.

        Returns
        -------
        np.ndarray
            Boolean mask for low-latitude points.
        """
        kind = self.mainfield.kind
        if kind == "dipole":
            return np.abs(grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                grid.lat, grid.lon, (RI - 6371e3) * 1e-3
            )
            return np.abs(mlat) < self.latitude_boundary
        else:
            return np.zeros(grid.size, dtype=bool)
