"""Apex coordinate and constraint mapping module.

This module handles transformations between geographic and magnetic apex
coordinates, and constructs operators for interhemispheric constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pynamit.simulation.geometry_utils import canonicalize_vector_basis_matrix

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
    jr_map_spectral : np.ndarray
        Operator mapping radial current to apex current (spectral basis).
    jr_map_sim : np.ndarray
        Operator mapping radial current to apex current (simulation basis).
    jr_map_apex_spectral : np.ndarray
        Apex-current operator before LL conjugate-mismatch subtraction (spectral basis).
    jr_map_apex_sim : np.ndarray
        Apex-current operator before LL conjugate-mismatch subtraction (simulation basis).
    E_coeffs_to_E_apex_ll_diff : ConstraintOperator or None
        E-field difference operator for low-latitude interhemispheric constraint.
    ll_mask : np.ndarray
        Boolean mask for low-latitude points.
    """

    jr_map_spectral: np.ndarray
    jr_map_sim: np.ndarray
    jr_map_apex_spectral: np.ndarray
    jr_map_apex_sim: np.ndarray
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
    ) -> None:
        self.mainfield = mainfield
        self.basis = basis
        self.latitude_boundary = latitude_boundary
        self.connect_hemispheres = connect_hemispheres
        self.northern_hemisphere_apex_constraints = northern_hemisphere_apex_constraints

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
        # Main Mapping (always needed for simulation physics)
        jr_map_spectral, E_coeffs_to_E_apex = self.create_apex_operators(b_field, grid)

        # Preserve non-mismatch apex operator for concentration-mode splitting.
        import scipy.sparse
        if scipy.sparse.issparse(jr_map_spectral):
            jr_map_apex_spectral = jr_map_spectral.copy()
        else:
            jr_map_apex_spectral = np.ascontiguousarray(np.array(jr_map_spectral, copy=True))
        
        # Initialize constraint container
        E_coeffs_to_E_apex_ll_diff = None
        ll_mask = None
        
        # ---- Constraint Construction Logic ----
        
        if self.connect_hemispheres:
            
            if self.northern_hemisphere_apex_constraints:
                # === NEW MODE: Northern Apex Constraints ===
                # 1. Generate Custom Grids
                # We need to pass RI to helper (refactor helper to take RI)
                # Helper update: Passing RI dynamically
                g_north, g_south = self._create_northern_apex_grids_with_RI(grid, RI)
                
                # 2. Evaluate Field on these grids
                b_north = self.mainfield.discretize(g_north, RI)
                b_south = self.mainfield.discretize(g_south, RI)
                
                # 3. Create Operators
                # We only need E-field operators for the constraint.
                # jr is not mapped FROM this grid, so we ignore the jr_op return.
                _, E_op_north = self.create_apex_operators(b_north, g_north)
                _, E_op_south = self.create_apex_operators(b_south, g_south)
                
                # 4. Difference Operator
                # The grids are constructed to be index-matched (point i in N corresponds to point i in S).
                # So we can directly subtract the matrices.
                E_diff_raw = E_op_north - E_op_south
                
                # 5. Mask
                # The entire grid is the constraint region.
                # Mask is all-True.
                ll_mask = np.ones(g_north.size, dtype=bool)
                
                # 6. Store
                # Shape: (2, N_points, Coeffs)
                # To match expected shape (2, Mask, 2, L) or (2, Mask, L)?
                # Legacy code handles shape variations.
                # Let's provide (2, N_points, Coeffs) directly.
                # But wait, `E_op` from create_apex_operators is (2, N, Coeffs) [Vectors(2) x Grid(N) x Coeffs]
                # Actually `create_apex_operators` returns:
                # E_op = einsum("ijk,jkpq->ikpq") -> (2, Grid, PolTor, Coeffs) [Rank 4]
                # OR (2, Grid, Coeffs) [Rank 3] depending on basis.
                
                # If Rank 4 SH: (2, Grid, 2, L).
                # Subtracting works fine.
                
                # Important: Apply LL Mask? 
                # Since mask is all-True, we just keep the whole thing.
                # But for consistency with downstream which might slice by mask:
                E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(E_diff_raw)
                
                # Legacy mask for simulation grid interactions (e.g. subtracting CP contribution)?
                # "jr_map_spectral[ll_mask] -= ..." logic modifies the MAIN simulation operator.
                # This logic assumes `ll_mask` refers to the SIMULATION grid.
                # BUT we just defined `ll_mask` for our CUSTOM grid!
                # Conflict: The legacy logic MIXES the "Simulation Physics Modification" (jr subtraction)
                # with the "Constraint Definition" (E_diff).
                
                # CRITICAL FIX: Separating Constraint vs Physics modification.
                
                # The "jr_map_spectral" modification puts the interhemispheric current continuity 
                # into the main linear operator (implicit handling).
                # This requires `jr_map_spectral` (on SIM GRID) to be modified.
                # Does the "Northern Constraint" mode imply we STOP doing the implicit jr modification on the sim grid?
                # The prompt says: "constraints are instead defined on Northern low latitude points... double counting...".
                # This implies we REPLACE the old constraint mechanism.
                # The old mechanism had TWO parts:
                # A. Enforcing E_north = E_south (Constraint Equation)
                # B. Enforcing J_r_north + J_r_south = 0 (Implicitly or explicitly?)
                
                # Actually, `jr_map_spectral[ll_mask] -= jr_coeffs_to_j_apex_cp[ll_mask]`
                # This line modifies the operator that computes "Mapped Apex Current".
                # J_apex = J_r_north - J_r_south.
                # This is the quantity that enters the solver as "Source".
                # So physically, we define J_apex = divergence of current systems.
                # If we move the *Constraints* (E-field match) to a new grid,
                # do we still map currents from the Simulation Grid?
                # Yes, the Physics is still solved on the Simulation Grid.
                # The "Constraint" is an additional equation block added to the Least Squares system.
                
                # So:
                # 1. We still need `ll_mask` for the SIMULATION GRID to know where to subtract J_south?
                # Actually, if we use Northern Apex Constraints, we might not need the `jr` modification on the sim grid?
                # No, the `jr` modification is part of the DEFINITION of the current system at low latitudes.
                # We likely still want `jr_map_sim` to represent "Dual Hemisphere Current" at low latitudes.
                
                # Let's calculate the SIMULATION GRID mask as usual for the purpose of `jr` mapping.
                ll_mask_sim = self._compute_ll_mask(grid, RI)
                ll_mask = ll_mask_sim # Return this so other things work
                
                # Modify jr_map_spectral on simulation grid (Legacy behavior maintained for Physics Map)
                if cp_grid is not None and cp_b_field is not None:
                     jr_cp, _ = self.create_apex_operators(cp_b_field, cp_grid)

                     if scipy.sparse.issparse(jr_map_spectral):
                        jr_lil = jr_map_spectral.tolil()
                        jr_lil[ll_mask_sim] -= jr_cp[ll_mask_sim]
                        jr_map_spectral = jr_lil.tocsr()
                     else:
                        jr_map_spectral[ll_mask_sim] -= jr_cp[ll_mask_sim]
                
                # NOW: We assign the CUSTOM E-field difference operator.
                # This operator does NOT live on the simulation grid.
                # It lives on the Custom Constraint Grid.
                # So its 2nd dimension (Grid points) will be N_custom, not N_sim_masked.
                # The Solver doesn't care about spatial correspondence, just row correspondence.
                # As long as `E_coeffs_to_E_apex_ll_diff` provides rows that are added to the LS matrix, it's fine.
                
                # So we simply override the constraint operator with our custom one.
                # Note: We must ensure it is shaped (2, N_points, ...) so that flatten matches 2*N_points rows.
                
                # Masking logic in `state.py`:
                # "E_coeffs_to_E_apex_ll_diff[:, ll_mask]"
                # It tries to SLICE the operator using `ll_mask`.
                # If we provide a pre-sliced/pre-ready operator, we must trick `state.py` or wrap it.
                # If we attach a ConstraintOperator that claims to be ready, `state.py` might still try to slice it?
                
                # Let's check `state.py`:
                # "op_obj = self.geometry.E_coeffs_to_E_apex_ll_diff"
                # "if hasattr(op_obj, "tensor"): outer_t = ...Op wrapped..."
                # "return op_outer @ op_inner"
                # It does NOT re-slice in `E_map_constraint_operator` property!
                # BUT, in `_solve_for_m_imp`:
                # "E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff" (Raw array or wrapped)
                # Wait, where is the slicing?
                # In `build_constraint_mappings` (THIS function), the LEGACY code did:
                # "E_op_diff = (E - E_cp)[:, ll_mask]"
                # So the object returned by this function is ALREADY SLICED.
                
                # So for our New Mode:
                # We return `E_diff_raw` (which acts as the "Sliced" operator because it's defined on the constraint set).
                # We do NOT slice it with `ll_mask_sim` (shapes don't match).
                E_coeffs_to_E_apex_ll_diff = E_diff_raw
                
            else:
                # === LEGACY MODE ===
                # Compute low-latitude mask on SIMULATION Grid
                ll_mask = self._compute_ll_mask(grid, RI)
                
                if cp_grid is not None and cp_b_field is not None:
                    # Conjugate hemisphere operators
                    jr_coeffs_to_j_apex_cp, E_coeffs_to_E_apex_cp = self.create_apex_operators(
                        cp_b_field, cp_grid
                    )
        
                    # Apply correction in spectral domain
                    if scipy.sparse.issparse(jr_map_spectral):
                        jr_map_spectral_lil = jr_map_spectral.tolil()
                        jr_map_spectral_lil[ll_mask] -= jr_coeffs_to_j_apex_cp[ll_mask]
                        jr_map_spectral = jr_map_spectral_lil.tocsr()
                    else:
                        jr_map_spectral[ll_mask] -= jr_coeffs_to_j_apex_cp[ll_mask]
        
                    # Create E-field mapping difference operator for constraint (SLICED)
                    E_coeffs_to_E_apex_ll_diff = np.ascontiguousarray(
                        (E_coeffs_to_E_apex - E_coeffs_to_E_apex_cp)[:, ll_mask]
                    )

        # Common Processing
        if E_coeffs_to_E_apex_ll_diff is not None:
             if input_adapter is not None:
                 # Pre-compose with analysis matrix (SH->Grid)
                 logger.info("Pre-composing hybrid constraint operator (SH->Grid)...")
                 E_coeffs_to_E_apex_ll_diff = np.tensordot(
                     E_coeffs_to_E_apex_ll_diff, input_adapter, axes=([-1], [0])
                 )
        
        # Wrap in ConstraintOperator
        E_op_obj = None
        if E_coeffs_to_E_apex_ll_diff is not None:
             E_op_obj = ConstraintOperator(E_coeffs_to_E_apex_ll_diff)

        # Compute simulation operator
        if input_adapter is not None:
            jr_map_sim = jr_map_spectral @ input_adapter
            jr_map_apex_sim = jr_map_apex_spectral @ input_adapter
        else:
            jr_map_sim = jr_map_spectral
            jr_map_apex_sim = jr_map_apex_spectral

        return ConstraintMappings(
            jr_map_spectral=jr_map_spectral,
            jr_map_sim=jr_map_sim,
            jr_map_apex_spectral=jr_map_apex_spectral,
            jr_map_apex_sim=jr_map_apex_sim,
            E_coeffs_to_E_apex_ll_diff=E_op_obj,
            ll_mask=ll_mask, # Note: This corresponds to Sim Grid for Physics map usage
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
        elif self.mainfield.kind == "dipole":
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
        if kind == "dipole":
            return np.abs(grid.lat) < self.latitude_boundary
        elif kind == "igrf":
            mlat, _ = self.mainfield.apx.geo2apex(
                grid.lat, grid.lon, (RI - RE) * 1e-3
            )
            return np.abs(mlat) < self.latitude_boundary
        else:
            return np.zeros(grid.size, dtype=bool)
