
import numpy as np
import pytest
from pynamit.simulation.settings import DynamicsSettings, SimulationMode
from pynamit.simulation.spatial import Geometry
from pynamit.primitives.mainfield import Mainfield
from pynamit.primitives.basis import Basis
# Need concrete basis (SH or CS)
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.cubed_sphere.cs_basis import CSBasis


class TestNorthernConstraints:
    
    def test_constraint_operator_shape_difference(self):
        """Verify that enabling the option changes the constraint operator,
        specifically its shape (due to new grid usage).
        """
        # Setup
        settings = DynamicsSettings(
             Nmax=5,
             Mmax=5,
             Ncs=10, 
             connect_hemispheres=True,
             northern_hemisphere_apex_constraints=False # Baseline
        )
        
        mainfield = Mainfield(kind="igrf", epoch=2020)
        basis = SHBasis(settings.Nmax, settings.Mmax)
        grid_basis = CSBasis(settings.Ncs) # dummy grid
        
        # 1. Baseline (Legacy)
        geo_legacy = Geometry(basis, grid_basis, mainfield, settings)
        op_legacy = geo_legacy.E_coeffs_to_E_apex_ll_diff
        
        # Expectation: Shape depends on number of simulation grid points in low-lat
        assert op_legacy is not None
        rows_legacy = op_legacy.tensor.shape[1] # (2, Mask, 2, L) or similar
        
        # 2. New Mode
        settings.northern_hemisphere_apex_constraints = True
        geo_new = Geometry(basis, grid_basis, mainfield, settings)
        op_new = geo_new.E_coeffs_to_E_apex_ll_diff
        
        assert op_new is not None
        rows_new = op_new.tensor.shape[1]
        
        print(f"Legacy Rows: {rows_legacy}, New Mode Rows: {rows_new}")
        
        # New mode uses a grid derived from Sqrt(Size).
        # Should be different from Legacy (which depends on CS grid distribution).
        assert rows_legacy != rows_new
        assert rows_legacy == 455
        assert rows_new == 288
        
    def test_run_simulation_step(self):
        """Smoke test running one step with the new constraints."""
        from pynamit.simulation.runner import run_pynamit
        
        # Use simple settings for speed
        # run_pynamit returns a Dynamics object
        dynamics = run_pynamit(
            Nmax=3, Ncs=6,
            connect_hemispheres=True,
            northern_hemisphere_apex_constraints=True,
            final_time=5e-4, # Run one step
            mainfield_kind='igrf'
        )
        
        assert dynamics is not None
        assert dynamics.state.geometry.northern_hemisphere_apex_constraints == True
        # Check if constraints are active
        assert dynamics.state.E_map_constraint_operator is not None
        
        # Regression Value Check
        # Retrieve state from output timeseries
        state_data = dynamics.output_timeseries.get_entry("state", 5e-4) # final_time
        if state_data and "m_imp" in state_data:
             m_imp = state_data["m_imp"]
             m_imp_norm = np.linalg.norm(m_imp)
             # print(f"DEBUG: m_imp_norm = {m_imp_norm}")
             # Expected value with IGRF + Northern Apex Constraints
             expected_norm = 7.114965741852009e-08
             assert np.isclose(m_imp_norm, expected_norm, rtol=1e-10)
        else:
             assert False, "State m_imp missing from output timeseries."
