"""Apex coordinate and constraint mapping module.

This module handles transformations between geographic and magnetic apex
coordinates, and constructs operators for interhemispheric constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pynamit.simulation.spatial.geometry_utils import canonicalize_vector_basis_matrix
from pynamit.simulation.settings import DynamicsMode, MainfieldKind

if TYPE_CHECKING:
    from pynamit.primitives.basis import Basis
    from pynamit.primitives.field import Field
    from pynamit.primitives.grid import Grid


logger = logging.getLogger(__name__)




@dataclass
class ConstraintOperator:
    """Applies canonical interhemispheric constraint tensors."""
    tensor: Any

    def __init__(self, tensor: Any):
        if isinstance(tensor, ConstraintOperator):
            self.tensor = tensor.tensor
        else:
            self.tensor = tensor

    def apply(self, coeffs: np.ndarray) -> np.ndarray:
        """Apply the operator to input coefficients."""
        from pynamit.utils import xp

        # Ensure tensor is compatible with backend (e.g. JAX)
        t_in = xp.asarray(self.tensor)
        coeffs = xp.asarray(coeffs)

        # Canonical constraint tensor: (2, n_mask, 2, n_coeffs).
        if t_in.ndim != 4:
            raise ValueError(
                f"Constraint tensor must be rank-4 canonical (2, n_mask, 2, n_coeffs), got {t_in.shape}."
            )

        # Input can be (2*n_coeffs,), (2*n_coeffs, batch), (2, n_coeffs), or (2, n_coeffs, batch).
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(2, -1)
        elif coeffs.ndim == 2 and coeffs.shape[0] != 2:
            coeffs = coeffs.reshape(2, -1, coeffs.shape[1])

        return -xp.tensordot(t_in, coeffs, axes=([2, 3], [0, 1]))


# Register ConstraintOperator as a JAX PyTree node if JAX is available
try:
    import jax
    jax.tree_util.register_pytree_node(
        ConstraintOperator,
        lambda c: ((c.tensor,), None), # Flatten: children=(tensor,), aux=None
        lambda aux, children: ConstraintOperator(children[0]) # Unflatten
    )
except ImportError:
    pass

@dataclass
class ConstraintMappings:
    """Container for constraint-related operators.

    Attributes
    ----------
    constraint_scalar_map_spectral : np.ndarray
        Operator mapping coefficients to the constraint scalar (spectral basis).
        In ``full_induction`` this is direct alpha sample-space mapping.
    constraint_scalar_map_sim : np.ndarray
        Operator mapping coefficients to the constraint scalar (simulation basis).
    constraint_scalar_map_reference_spectral : np.ndarray
        Constraint-scalar map before LL conjugate-mismatch subtraction
        (spectral basis).
    constraint_scalar_map_reference_sim : np.ndarray
        Constraint-scalar map before LL conjugate-mismatch subtraction
        (simulation basis).
    E_coeffs_to_E_apex_ll_diff : ConstraintOperator or None
        E-field difference operator for low-latitude interhemispheric constraint.
    ll_mask : np.ndarray
        Boolean mask for low-latitude points.
    """

    constraint_scalar_map_spectral: np.ndarray
    constraint_scalar_map_sim: np.ndarray
    constraint_scalar_map_reference_spectral: np.ndarray
    constraint_scalar_map_reference_sim: np.ndarray
    E_coeffs_to_E_apex_ll_diff: Optional[ConstraintOperator]
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
        northern_hemisphere_apex_constraints: bool = False,
        dynamics_mode: DynamicsMode | str = DynamicsMode.LEGACY,
    ) -> None:
        self.mainfield = mainfield
        self.basis = basis
        self.latitude_boundary = latitude_boundary
        self.connect_hemispheres = connect_hemispheres
        self.northern_hemisphere_apex_constraints = northern_hemisphere_apex_constraints
        self.dynamics_mode = DynamicsMode(str(dynamics_mode))

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
        constraint_scalar_op : np.ndarray
            Operator mapping coefficients to the configured constraint scalar:
            - ``full_induction``: direct ``alpha`` sample-space mismatch map
              (no ``Br`` reweighting inside the hard-constraint rows).
            - other modes: legacy apex-current mapping.
        E_op : np.ndarray
            Operator mapping E-field coefficients to apex E-field.
        """
        _, horizontal_to_apex = self.get_transformation_matrices(field)

        # Full-induction constraints use direct alpha sample-space mismatch.
        # Legacy path keeps the historical d3/apex mapping.
        if self.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            radial_to_apex = np.ones(grid.size, dtype=float)
        else:
            radial_to_apex, _ = self.get_transformation_matrices(field)
        constraint_scalar_op = self.basis.get_scaled_matrix(grid, radial_to_apex)

        # E_coeffs_to_E_apex
        # horizontal_to_apex: (2, 2, N_grid) [i, j, k]
        # vector_basis: canonical (Comp, N_grid, PotentialType, Coeffs) [j, k, p, q]
        G_in = canonicalize_vector_basis_matrix(
            self.basis.get_vector_basis_matrix(grid),
            basis_index_length=self.basis.index_length,
        )

        if G_in.shape[0] != horizontal_to_apex.shape[1]:
            raise ValueError(
                "Apex mapping component mismatch: "
                f"vector basis has {G_in.shape[0]} components, "
                f"apex map expects {horizontal_to_apex.shape[1]}."
            )

        E_op = np.einsum("ijk,jkpq->ikpq", horizontal_to_apex, G_in, optimize=True)
        return constraint_scalar_op, E_op

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
            Radius of the shell where operators are evaluated (m).
        cp_grid : Grid, optional
            Conjugate hemisphere grid on the same shell.
        cp_b_field : Field, optional
            Conjugate hemisphere magnetic field on the same shell.
        input_adapter : np.ndarray, optional
            Adapter for hybrid basis transformation.

        Returns
        -------
        ConstraintMappings
            Container with all constraint operators.

        Notes
        -----
        In `northern_hemisphere_apex_constraints=True` mode, the returned `ll_mask`
        still refers to the *simulation grid* (used for jr-map modification), while
        the E-field constraint operator is defined on a separate custom northern-apex
        constraint grid and is already "pre-sliced" to that grid.
        """
        import scipy.sparse as sp

        def _subtract_rows_inplace(lhs, rhs, row_mask):
            """Subtract rhs[row_mask] from lhs[row_mask], preserving sparse format."""
            if lhs is None or rhs is None:
                return lhs
            if sp.issparse(lhs):
                lhs_lil = lhs.tolil()
                lhs_lil[row_mask] -= rhs[row_mask]
                return lhs_lil.tocsr()
            lhs[row_mask] -= rhs[row_mask]
            return lhs

        # Main (simulation-grid) apex operators
        constraint_scalar_map_spectral, E_coeffs_to_E_apex = self.create_apex_operators(
            b_field, grid
        )

        # Preserve the non-mismatch apex current operator (before LL conjugate subtraction)
        if sp.issparse(constraint_scalar_map_spectral):
            constraint_scalar_map_reference_spectral = constraint_scalar_map_spectral.copy()
        else:
            constraint_scalar_map_reference_spectral = np.ascontiguousarray(
                np.array(constraint_scalar_map_spectral, copy=True)
            )

        # Defaults
        E_coeffs_to_E_apex_ll_diff = None
        ll_mask = np.zeros(grid.size, dtype=bool)

        # Optional conjugate operators (only if both are provided)
        have_cp = (cp_grid is not None) and (cp_b_field is not None)

        if self.connect_hemispheres:
            if self.northern_hemisphere_apex_constraints:
                # --- New mode: define E-constraint on a custom northern-apex grid ---
                g_north, g_south = self._create_northern_apex_grids_with_RI(grid, RI)

                b_north = self.mainfield.discretize(g_north, RI)
                b_south = self.mainfield.discretize(g_south, RI)

                _, E_op_north = self.create_apex_operators(b_north, g_north)
                _, E_op_south = self.create_apex_operators(b_south, g_south)

                # Constraint operator lives on the custom constraint grid (already "sliced")
                E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(E_op_north - E_op_south)

                # Simulation-grid LL mask is still used for jr-map physics modification
                ll_mask = self._compute_ll_mask(grid, RI)

                if have_cp:
                    cp_constraint_scalar_map, _ = self.create_apex_operators(cp_b_field, cp_grid)
                    constraint_scalar_map_spectral = _subtract_rows_inplace(
                        constraint_scalar_map_spectral, cp_constraint_scalar_map, ll_mask
                    )

            else:
                # --- Legacy mode: constraints defined on simulation-grid low latitudes ---
                ll_mask = self._compute_ll_mask(grid, RI)

                if have_cp:
                    cp_constraint_scalar_map, E_cp = self.create_apex_operators(
                        cp_b_field, cp_grid
                    )

                    # Modify apex current map on low-lat simulation rows
                    constraint_scalar_map_spectral = _subtract_rows_inplace(
                        constraint_scalar_map_spectral, cp_constraint_scalar_map, ll_mask
                    )

                    # Constraint operator is the LL-sliced E-field mismatch on the simulation grid
                    E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                        (E_coeffs_to_E_apex - E_cp)[:, ll_mask]
                    )

        # Optional hybrid pre-composition (SH -> analysis/grid basis)
        if E_coeffs_to_E_apex_ll_diff is not None and input_adapter is not None:
            logger.info("Pre-composing hybrid constraint operator (SH->Grid)...")
            E_coeffs_to_E_apex_ll_diff = np.tensordot(
                E_coeffs_to_E_apex_ll_diff, input_adapter, axes=([-1], [0])
            )

        # Wrap constraint operator
        E_op_obj = (
            ConstraintOperator(E_coeffs_to_E_apex_ll_diff)
            if E_coeffs_to_E_apex_ll_diff is not None
            else None
        )

        # Simulation-space operators (possibly adapted)
        if input_adapter is not None:
            constraint_scalar_map_sim = constraint_scalar_map_spectral @ input_adapter
            constraint_scalar_map_reference_sim = (
                constraint_scalar_map_reference_spectral @ input_adapter
            )
        else:
            constraint_scalar_map_sim = constraint_scalar_map_spectral
            constraint_scalar_map_reference_sim = constraint_scalar_map_reference_spectral

        return ConstraintMappings(
            constraint_scalar_map_spectral=constraint_scalar_map_spectral,
            constraint_scalar_map_sim=constraint_scalar_map_sim,
            constraint_scalar_map_reference_spectral=constraint_scalar_map_reference_spectral,
            constraint_scalar_map_reference_sim=constraint_scalar_map_reference_sim,
            E_coeffs_to_E_apex_ll_diff=E_op_obj,
            ll_mask=ll_mask,
        )
    
    def _create_northern_apex_grids_with_RI(self, grid_ref: "Grid", RI: float) -> tuple["Grid", "Grid"]:
        """Create Northern and conjugate (Southern) grids from an Apex-like mesh at a given RI.

        The mesh is built uniformly in magnetic/apex-like coordinates and mapped to geographic
        coordinates on the simulation shell RI.

        - IGRF/Apex: true Apex mesh (MLat, MLon) -> apex2geo(...)
        - Dipole:    dipole-consistent "apex-like" mesh defined on a reference shell, then
                     mapped with Mainfield.map_coords(...)

        The southern grid is obtained from Mainfield.conjugate_coordinates(...), which keeps
        conjugate logic centralized in the mainfield implementation.
        """
        from pynamit.primitives.grid import Grid
        from pynamit.math.constants import RE

        # Resolution heuristic from reference grid size (rough square side length)
        n_side = max(1, int(np.sqrt(grid_ref.size)))
        n_lat = max(10, n_side // 2)
        n_lon = max(20, n_side)

        h_sim_km = (RI - RE) / 1000.0  # apexpy uses km altitude

        def _uniform_mesh(lat_min: float, lat_max: float, nlat: int, nlon: int):
            """Flattened mesh with lon in [0, 360)."""
            lats = np.linspace(lat_min, lat_max, nlat)
            lons = np.linspace(0.0, 360.0, nlon, endpoint=False)
            lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
            return lat_mesh.ravel(), lon_mesh.ravel()

        def _min_reachable_lat_deg(r_ref_m: float, r_dest_m: float, buffer_deg: float = 0.1) -> float:
            """Conservative minimum |magnetic latitude| so field lines from r_ref can reach r_dest.

            Dipole-like estimate:
                cos^2(lambda) = r_ref / r_dest
            Only relevant when r_dest > r_ref.
            """
            if r_dest_m <= r_ref_m:
                return 0.1
            val = np.clip(r_ref_m / r_dest_m, 0.0, 1.0)
            return float(np.degrees(np.arccos(np.sqrt(val))) + buffer_deg)

        apx = getattr(self.mainfield, "apx", None)

        # --- IGRF/Apex case: true Apex-uniform northern mesh -------------------------
        if apx is not None:
            # Some low-|MLat| field lines may not reach the requested altitude.
            # Compute a conservative minimum Apex latitude when h_sim > apex ref height.
            r_ref_apex = RE + float(apx.refh) * 1000.0
            min_mlat = _min_reachable_lat_deg(r_ref_apex, RI)

            mlat_max = max(min_mlat, float(self.latitude_boundary))
            mlat_flat, mlon_flat = _uniform_mesh(min_mlat, mlat_max, n_lat, n_lon)

            # North footpoints are +MLat
            glat_n, glon_n, _ = apx.apex2geo(mlat_flat, mlon_flat, h_sim_km)
            grid_north = Grid(lat=glat_n, lon=glon_n)

        # --- Dipole case: exact dipole-consistent Apex-like construction -------------
        elif self.mainfield.kind == MainfieldKind.DIPOLE:
            # Define a reference shell (analog of Apex refh) on which the mesh is uniform in
            # "magnetic latitude" and longitude. Then map along dipole field lines to RI.
            #
            # If you have a preferred project-wide value, expose it as an attribute and override:
            #   self.dipole_apex_refh_km = ...
            dipole_refh_km = float(getattr(self, "dipole_apex_refh_km", 110.0))
            r_ref = RE + dipole_refh_km * 1000.0

            # Ensure field lines from the reference shell can reach RI
            min_mlat = _min_reachable_lat_deg(r_ref, RI)
            mlat_max = max(min_mlat, float(self.latitude_boundary))

            # Uniform dipole "magnetic" mesh on the reference shell
            mlat_flat, mlon_flat = _uniform_mesh(min_mlat, mlat_max, n_lat, n_lon)
            theta_ref = 90.0 - mlat_flat
            phi_ref = mlon_flat

            # Map north grid from reference shell to the requested shell RI
            theta_n, phi_n = self.mainfield.map_coords(
                RI, np.full_like(theta_ref, r_ref, dtype=float), theta_ref, phi_ref
            )
            grid_north = Grid(theta=theta_n, phi=phi_n)

        # --- Fallback (e.g. radial): no meaningful conjugate/Apex construction -------
        else:
            return Grid(lat=[10.0], lon=[0.0]), Grid(lat=[-10.0], lon=[0.0])

        # Use Mainfield implementation for conjugate mapping (IGRF and dipole)
        cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
            RI, grid_north.theta, grid_north.phi
        )
        grid_south = Grid(theta=cp_theta, phi=cp_phi)

        return grid_north, grid_south

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
        from pynamit.math.constants import RE
        kind = self.mainfield.kind
        if kind == MainfieldKind.DIPOLE:
            return np.abs(grid.lat) < self.latitude_boundary
        elif kind == MainfieldKind.IGRF:
            mlat, _ = self.mainfield.apx.geo2apex(
                grid.lat, grid.lon, (RI - RE) * 1e-3
            )
            return np.abs(mlat) < self.latitude_boundary
        else:
            return np.zeros(grid.size, dtype=bool)
