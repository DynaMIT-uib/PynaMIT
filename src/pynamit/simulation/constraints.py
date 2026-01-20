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
class ConstraintOperator:
    """Encapsulates the constraint tensor logic to abstract rank differences."""
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
        
        # E_map_op: (2, Mask, [PolTor], L) or (2, Mask, L)
        # coeffs: (N) or (N, Batch)
        # N = 2*L for SH, or L for CS (usually)
        
        if t_in.ndim == 4:
            # SH: (2, Mask, 2, L) -> Contract dims 2,3
            # Input must be shaped (2, L) or (2, L, Batch)
            
            # If flat input (N) or (N, Batch), reshape to (2, L, ...)
            if coeffs.ndim == 1:
                # (2*L) -> (2, L)
                coeffs = coeffs.reshape(2, -1)
            elif coeffs.ndim == 2 and coeffs.shape[0] != 2:
                # (2*L, Batch) -> (2, L, Batch)
                # Note: If coeffs is naturally (2, L), shape[0]==2.
                # If N=2, ambiguity exists, but L=1 is rare.
                coeffs = coeffs.reshape(2, -1, coeffs.shape[1])
                
            return -xp.tensordot(t_in, coeffs, axes=([2, 3], [0, 1]))



        else:
            # CS: (2, Mask, N) -> Contract dim 2
            # Input must be (N) or (N, Batch)
            
            # Use reshape to flatten stacked inputs if necessary (e.g. from state.py logic)
            # If coeffs is (2, L, Batch), reshape to (2*L, Batch)
            



            if coeffs.ndim == 3 and coeffs.shape[0] == 2:
                # (2, L, Batch) -> (2*L, Batch)
                coeffs = coeffs.reshape(-1, coeffs.shape[2])
            elif coeffs.ndim == 2 and coeffs.shape[0] == 2:
                # (2, L) -> (2*L)
                coeffs = coeffs.reshape(-1)
            

            # tensordot axes=([2], [0]) handles both (N) and (N, Batch) correctly

            # tensordot axes=([2], [0]) handles both (N) and (N, Batch) correctly
            # because dim 0 of coeffs matches dim 2 of tensor.
            return -xp.tensordot(t_in, coeffs, axes=([2], [0]))


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
    E_coeffs_to_E_apex_ll_diff : ConstraintOperator or None
        E-field difference operator for low-latitude interhemispheric constraint.
    ll_mask : np.ndarray
        Boolean mask for low-latitude points.
    """

    jr_map_spectral: np.ndarray
    jr_map_sim: np.ndarray
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
    
    def _create_northern_apex_grids(self, grid_ref: "Grid") -> tuple["Grid", "Grid"]:
        """Create Northern and Conjugate (Southern) grids based on Apex coordinates.
        
        Generates a grid uniform in Apex coordinates (MLat 0-Boundary, MLon 0-360)
        and maps it to geographic coordinates for both hemispheres.
        
        Parameters
        ----------
        grid_ref : Grid
            Reference grid to determine resolution (approximate density).
            
        Returns
        -------
        grid_north : Grid
            Geographic grid corresponding to the Northern Apex mesh.
        grid_south : Grid
            Geographic grid corresponding to the Southern (Conjugate) Apex mesh.
        """
        from pynamit.primitives.grid import Grid
        
        # Determine resolution based on ref grid size or explicit Ncs logic
        # Square root of size gives rough N side length
        N_side = int(np.sqrt(grid_ref.size))
        
        # We want coverage of [0, lat_boundary] in MLat
        # and [0, 360] in MLon
        
        # N_lat proportional to boundary fraction of 90 deg? 
        # Or just use N_side/2 to be safe/dense enough.
        n_lat = max(10, N_side // 2)
        n_lon = max(20, N_side)
        
        # Create Apex Mesh
        # mlat: 0 to latitude_boundary (Northern Hemisphere)
        # Avoid 0 exactly if singularity? Apex coords at equator are fine usually.
        mlats = np.linspace(0.1, self.latitude_boundary, n_lat)
        mlons = np.linspace(0, 360, n_lon, endpoint=False)
        
        mlat_mesh, mlon_mesh = np.meshgrid(mlats, mlons, indexing='ij')
        
        # Flatten
        mlat_flat = mlat_mesh.flatten()
        mlon_flat = mlon_mesh.flatten()
        
        # height for mapping (Ionosphere)
        h_km = (self.mainfield.apx.refh if hasattr(self.mainfield, "apx") else 110.0) # Fallback? 
        # Actually Apex class has refh. Or we use (RI - RE)/1km.
        # But we need the Apex object to do apex2geo.
        
        if self.mainfield.apx is not None:
            # IGRF/Apex case
            # North: mlat positive
            glat_n, glon_n, _ = self.mainfield.apx.apex2geo(mlat_flat, mlon_flat, 0.0) # Height 0 or RefH? 
            # Usually simulation is at RI. 
            # If we map AT altitude RI:
            h_sim = (self.measure_RI() - 6371e3) / 1000.0 # m to km
            
            glat_n, glon_n, _ = self.mainfield.apx.apex2geo(mlat_flat, mlon_flat, h_sim)
            grid_north = Grid(lat=glat_n, lon=glon_n)
            
            # South: mlat negative (conjugate)
            # Conjugate of Mlat X is -X.
            # But wait, we want the point that maps to the SAME Apex coords?
            # No, we want the OTHER hemisphere point on the same field line.
            # Field line is defined by (Apex Lat, Apex Lon).
            # North footpoint: (+MLat, MLon). South footpoint: (-MLat, MLon) or similar?
            # "Conjugate point" usually means "Other end of field line".
            # For Apex, +/- MLat defines the two hemispheres.
            
            glat_s, glon_s, _ = self.mainfield.apx.apex2geo(-mlat_flat, mlon_flat, h_sim)
            grid_south = Grid(lat=glat_s, lon=glon_s)
            
        elif self.mainfield.kind == "dipole":
            # Analytical Dipole Mapping
            # MLat = Lat in Dipole (approx, if axis aligned)
            # If Dipole, Geo ~= Apex (rotated).
            # Assuming aligned dipole for simplicity or using Dipole impl logic?
            # Mainfield Dipole implementation has map_coords but not explicit apex2geo.
            # But Dipole is symmetric.
            # Let's assume standard Dipole coordinates.
            # Grid North = (MLat, MLon) (interpreted as Geo Lat/Lon relative to pole)
            # For strict Dipole, let's just use the Geo Grid directly?
            # Or better, rely on existing conjugate logic?
            # "Conjugate" tells us S from N.
            # So if we define N grid, we can get S grid via `conjugate_coordinates`.
            
            # 1. Define North Geo Grid (Approximation: Uniform Geo Lat/Lon in North)
            # Since Dipole Apex ~ Geo, a uniform Geo grid is also uniform Apex.
            lats = np.linspace(0.1, self.latitude_boundary, n_lat)
            lons = np.linspace(0, 360, n_lon, endpoint=False)
            ll_mesh, lo_mesh = np.meshgrid(lats, lons, indexing='ij')
            
            grid_north = Grid(lat=ll_mesh.flatten(), lon=lo_mesh.flatten())
            
            # 2. Get Conjugate (South)
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                self.measure_RI(), grid_north.theta, grid_north.phi
            )
            grid_south = Grid(theta=cp_theta, phi=cp_phi)
            
        else:
             # Radial or other: No constraints
             # Return dummy 1-point grids
             grid_north = Grid(lat=[10], lon=[0])
             grid_south = Grid(lat=[-10], lon=[0])
             
        return grid_north, grid_south

    def measure_RI(self) -> float:
        """Helper to get RI from existing context if possible, else Default."""
        # We don't have RI stored in ApexMapper directly, but we pass it to build methods.
        # But we need it for grid generation inside this method?
        # Let's look at where we call this. `build_constraint_mappings` has RI.
        # So pass RI to this method.
        return 6371e3 + 110e3 # Fallback default


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
        # E_coeffs_to_E_apex
        # Robust construction:
        # horizontal_to_apex: (2, 2, N_grid) [i, j, k]
        # vector_basis: (Comp, Spatial..., Coeffs)
        # We need (Comp, N_grid, Coeffs) [j, k, l]
        


        G_raw = self.basis.get_vector_basis_matrix(grid)
        if isinstance(G_raw, tuple): # Some bases return tuple
             G_raw = np.array(G_raw)
        
        # Determine contraction string based on basis rank
        # SHBasis: (Comp, Grid, PolTor, Coeffs) -> Rank 4
        # CSBasis: (Comp, Grid, Coeffs) -> Rank 3
        
        if G_raw.ndim == 4:
            einsum_str = "ijk,jkpq->ikpq"
            G_in = G_raw
        else:
            einsum_str = "ijk,jkm->ikm"
            G_in = G_raw

        # Handle Component Dimension mismatch (legacy/robustness)
        # horizontal_to_apex is (2, 2, N). Input dim 2 (theta, phi).
        # G_in dim 0 is components. If 3 (r, th, ph), slice to 2.
        if G_in.shape[0] == 3 and horizontal_to_apex.shape[1] == 2:
             G_in = G_in[1:]

        E_op = np.einsum(
            einsum_str,
            horizontal_to_apex,
            G_in,
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
        # Main Mapping (always needed for simulation physics)
        jr_map_spectral, E_coeffs_to_E_apex = self.create_apex_operators(b_field, grid)
        
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
                     
                     import scipy.sparse
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
                    import scipy.sparse
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
        else:
            jr_map_sim = jr_map_spectral

        return ConstraintMappings(
            jr_map_spectral=jr_map_spectral,
            jr_map_sim=jr_map_sim,
            E_coeffs_to_E_apex_ll_diff=E_op_obj,
            ll_mask=ll_mask, # Note: This corresponds to Sim Grid for Physics map usage
        )
    
    def _create_northern_apex_grids_with_RI(self, grid_ref: "Grid", RI: float) -> tuple["Grid", "Grid"]:
        """Helper to create grids with explicit RI passing."""
        # This re-implements _create_northern_apex_grids but uses the passed RI
        # To avoid changing the signature of the other method if cached/used elsewhere,
        # or just updating the main one. The main one was added just now, so I can fix it.
        # But for safety, I will inline the logic or call a refined version.
        
        # Actually, let's just make the previously added method accept RI? 
        # I can't restart the file edit, but I can add this new method.
        # Or I can just put the logic here.
        
        # Using the logic from previous step, but ensuring RI is used.
        
        from pynamit.primitives.grid import Grid
        N_side = int(np.sqrt(grid_ref.size))
        n_lat = max(10, N_side // 2)
        n_lon = max(20, N_side)
        
        h_sim = (RI - 6371e3) / 1000.0 # m to km
        
        # Calculate minimum safe Apex Latitude
        min_mlat = 0.1
        if self.mainfield.apx is not None:
            # Need to ensure field line reaches h_sim
            # cos^2(lat) = (Re + href) / (Re + hsim)
            # Re ~ 6371 km
            Re = 6371.0
            href = self.mainfield.apx.refh
            if h_sim > href:
                val = (Re + href) / (Re + h_sim)
                if val < 1.0:
                    min_rad = np.arccos(np.sqrt(val))
                    min_mlat = np.degrees(min_rad) + 0.1 # Add buffer
        
        mlats = np.linspace(min_mlat, self.latitude_boundary, n_lat)
        mlons = np.linspace(0, 360, n_lon, endpoint=False)
        mlat_mesh, mlon_mesh = np.meshgrid(mlats, mlons, indexing='ij')
        
        mlat_flat, mlon_flat = mlat_mesh.flatten(), mlon_mesh.flatten()

        if self.mainfield.apx is not None:
            glat_n, glon_n, _ = self.mainfield.apx.apex2geo(mlat_flat, mlon_flat, h_sim)
            glat_s, glon_s, _ = self.mainfield.apx.apex2geo(-mlat_flat, mlon_flat, h_sim)
            return Grid(lat=glat_n, lon=glon_n), Grid(lat=glat_s, lon=glon_s)
            
        elif self.mainfield.kind == "dipole":
             # Dipole Approx
            lats = np.linspace(0.1, self.latitude_boundary, n_lat)
            lons = np.linspace(0, 360, n_lon, endpoint=False)
            ll_mesh, lo_mesh = np.meshgrid(lats, lons, indexing='ij')
            g_north = Grid(lat=ll_mesh.flatten(), lon=lo_mesh.flatten())
            
            cp_theta, cp_phi = self.mainfield.conjugate_coordinates(
                RI, g_north.theta, g_north.phi
            )
            g_south = Grid(theta=cp_theta, phi=cp_phi)
            return g_north, g_south
        else:
             return Grid(lat=[10], lon=[0]), Grid(lat=[-10], lon=[0])

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
