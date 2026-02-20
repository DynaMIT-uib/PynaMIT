"""Test Br induction path with CSBasis."""

import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import SimulationMode

def test_br_induction_cs():
    """Test that Br induction works with CSBasis (no size mismatch)."""
    # Use CS_DOMINANT mode to force CSBasis as solution basis
    # But input Br is always SH coefficients
    try:
        dynamics = run_pynamit(
            final_time=0.0, # Just initialization and operator setup
            Ncs=10,
            Nmax=5,
            Mmax=5,
            mainfield_kind="dipole",
            RM=2.0, # Enable magnetosphere for coupling factors
            simulation_mode=SimulationMode.CS_DOMINANT,
            ignore_PFAC=False, # Needed for induction
            vector_Br=True, # Ensure Br is handled
        )
        
        # Access the Br_to_E_coeffs operator, which triggers the bug
        op = dynamics.state.Br_to_E_coeffs
        print(f"DEBUG: Br_to_E_coeffs op: {op}")
        if op is not None:
             print(f"DEBUG: op shape: {op.shape}")
        n_cs = dynamics.state.solution_basis.index_length
        n_sh = dynamics.state.basis.index_length
        print(f"DEBUG: n_cs: {n_cs}, n_sh: {n_sh}")
        assert op is not None
        
        # Test applying it to some mock Br coefficients (spectral size)
        n_sh = dynamics.state.basis.index_length
        br_coeffs = np.zeros(n_sh)
        br_coeffs[0] = 1.0 # Monopole Br (just for test)
        
        e_coeffs = op.matvec(br_coeffs)
        assert e_coeffs is not None
        # Output should be in solution basis vector space (2*n_cs)
        n_cs = dynamics.state.solution_basis.index_length
        assert e_coeffs.size == 2 * n_cs
        
    except ValueError as e:
        if "size mismatch" in str(e).lower() or "not aligned" in str(e).lower():
            pytest.fail(f"Br induction path failed with size mismatch: {e}")
        else:
            raise e
    except Exception as e:
        # Check for the specific diagonal_linear_map/matmul error
        if "matmul" in str(e).lower() or "shape" in str(e).lower():
             pytest.fail(f"Br induction path failed with dimension error: {e}")
        else:
             raise e

if __name__ == "__main__":
    test_br_induction_cs()
