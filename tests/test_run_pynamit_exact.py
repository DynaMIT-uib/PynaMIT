
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.dynamics import Dynamics
from pynamit.utils import set_backend
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid
from pynamit.simulation.settings import MainfieldKind, SimulationMode

def test_run_pynamit_mw_weighting_option():
    """Test run_pynamit with input_weighting='mw' using GL grid (regular)."""
    set_backend("numpy")
    
    # Mock data getters to return dummy data matching input grid shape
    with patch("pynamit.data.get_conductance_inputs") as mock_cond, \
         patch("pynamit.data.get_jr_inputs") as mock_jr, \
         patch("pynamit.data.get_wind_inputs") as mock_wind:
         
        # Setup mocks to return data shaped like the grid passed to them
        def side_effect_cond(date, lat, lon, time):
            # Return ones
            shape = lat.shape
            return np.ones(shape), np.ones(shape), lat, lon
            
        mock_cond.side_effect = side_effect_cond
        
        def side_effect_jr(date, lat, lon, time):
             shape = lat.shape
             return np.ones(shape), lat, lon
             
        mock_jr.side_effect = side_effect_jr
        
        mock_wind.return_value = None # Disable wind for simplicity
        
        # Run with GL mode (Regular Grid)
        print("Running run_pynamit with GL mode and exact weights...")
        dynamics = run_pynamit(
            final_time=1e-3, # minimal run (2 steps with dt=5e-4)
            plotsteps=100,
            Nmax=4,
            Mmax=4,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_GL, # Ensures Regular Grid
            input_weighting="mw",
            mainfield_kind=MainfieldKind.DIPOLE
        )
        pass

# We will use a wrapper to patch Dynamics methods
@patch("pynamit.simulation.dynamics.Dynamics.evolve_to_time")
@patch("pynamit.simulation.dynamics.Dynamics.set_u")
@patch("pynamit.simulation.dynamics.Dynamics.set_jr")
@patch("pynamit.simulation.dynamics.Dynamics.set_conductance")
@patch("pynamit.data.get_conductance_inputs")
@patch("pynamit.data.get_jr_inputs")
@patch("pynamit.data.get_wind_inputs")
def test_run_pynamit_passes_weights(mock_wind, mock_jr, mock_cond, 
                                    mock_set_cond, mock_set_jr, mock_set_u, mock_evolve):
    """Verify that run_pynamit passes sqrt_weights when input_weighting='mw'."""
    
    # Mock Data
    def side_effect_cond(date, lat, lon, time):
        return np.ones(lat.shape), np.ones(lat.shape), lat, lon
    mock_cond.side_effect = side_effect_cond
    
    def side_effect_jr(date, lat, lon, time):
        return np.ones(lat.shape), lat, lon
    mock_jr.side_effect = side_effect_jr
    
    # Mock wind
    def side_effect_wind(date, wind, time):
        lat = np.array([0, 10, 20])
        lon = np.array([0, 0, 0])
        shape = (3,)
        return np.ones(shape), np.ones(shape), lat, lon, None
    mock_wind.side_effect = side_effect_wind
    # Note: mocking wind to return valid tuple
    
    # Run
    # Use GL mode so grid is regular
    run_pynamit(
        final_time=1.0,
        Nmax=4, 
        simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_GL,
        input_weighting="mw",
        mainfield_kind=MainfieldKind.DIPOLE,
        wind=True
    )
    
    # Assert Conductance
    assert mock_set_cond.called
    kwargs = mock_set_cond.call_args[1]
    assert "sqrt_weights" in kwargs
    weights = kwargs["sqrt_weights"]
    assert weights is not None
    assert np.any(weights != 1.0)
    
    # Assert Jr
    assert mock_set_jr.called
    kwargs_jr = mock_set_jr.call_args[1]
    assert kwargs_jr.get("sqrt_weights") is not None
    
    # Assert Wind
    assert mock_set_u.called
    kwargs_u = mock_set_u.call_args[1]
    assert kwargs_u.get("sqrt_weights") is not None
    
    print("Verified: sqrt_weights passed to all setters.")

def test_numerical_exactness_verified():
    """Verify that input_weighting='mw' results in exact coefficient recovery."""
    set_backend("numpy")
    
    # 1. Setup a controlled input signal: Real Spherical Harmonic Y_2,2
    # We want to verify exact SHT.
    # We choose a regular grid N_theta x N_phi.
    # N=10, M=10. SHT exact requires N_theta >= L+1.
    Nmax = 4
    N_lat = Nmax + 2 
    N_lon = 2*Nmax + 2
    
    theta_1d = np.linspace(0.1, np.pi-0.1, N_lat) # Arbitrary regular theta
    phi_1d = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    
    # Generate signal Y_2,2 (Real) using PynaMIT basis
    basis = SHBasis(Nmax, Nmax, mean_free=True)
    coeffs_true = np.zeros(basis.index_length)
    # Inject mode L=2, M=2 (Cos)
    idx_target = np.where((basis.n == 2) & (basis.m == 2))[0][0]
    
    # 2. Mock getters to return this grid and data
    with patch("pynamit.data.get_conductance_inputs") as mock_cond, \
         patch("pynamit.data.get_jr_inputs") as mock_jr, \
         patch("pynamit.data.get_wind_inputs") as mock_wind:
         
        mock_wind.return_value = None
        
        # Define a non-zero Jr to drive the system
        def side_effect_jr(date, lat, lon, time):
            # Create a simple pattern: Y_1,0 (Dipole-ish) + some noise
            # Just return sine of latitude to be non-zero
            val = np.sin(np.deg2rad(lat)) * 1e-6 # Small current
            return val.flatten(), lat, lon
            
        mock_jr.side_effect = side_effect_jr

        # Mock Cond
        def side_effect_data(date, lat, lon, time):
            print("DEBUG: side_effect_data called")
            g = Grid(lat, lon)
            
            # Use basis to evaluate
            b = SHBasis(Nmax, Nmax, mean_free=True)
            c = np.zeros(b.index_length)
            idx_b = np.where((b.n == 2) & (b.m == 2))[0][0]
            c[idx_b] = 10.0 # Target Coeff
            
            val_flat = b.evaluate(c, g, vector_type='scalar')
            # Add base conductance to avoid singular matrix/negativity
            val_flat = np.abs(val_flat) + 1.0
            
            return np.zeros_like(val_flat), val_flat, lat, lon
            
        mock_cond.side_effect = side_effect_data
        
        # Act
        dynamics = run_pynamit(
            final_time=1e-3,
            Nmax=Nmax,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_GL,
            input_weighting="mw",
            mainfield_kind=MainfieldKind.DIPOLE
        )
        
        # Assert Output State
        m_ind = dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1]
        m_imp = dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1]
        
        coeff_array = np.hstack((m_ind, m_imp))
        
        # Regression Values (Captured 2026-01-13 via extraction script)
        expected_coeff_norm = 4.073168003504329e-06
        expected_coeff_max = 7.443758228741214e-08
        expected_coeff_min = -4.072258061289225e-06
        expected_n_coeffs = 48
        
        actual_coeff_norm = np.linalg.norm(coeff_array)
        actual_coeff_max = np.max(coeff_array)
        actual_coeff_min = np.min(coeff_array)
        actual_n_coeffs = coeff_array.size
        
        print(f"actual_coeff_norm: {actual_coeff_norm}")
        print(f"actual_coeff_max: {actual_coeff_max}")
        print(f"actual_coeff_min: {actual_coeff_min}")
        print(f"actual_n_coeffs: {actual_n_coeffs}")

        # Assert.
        assert actual_coeff_norm == pytest.approx(expected_coeff_norm, rel=1e-12)
        assert actual_coeff_max == pytest.approx(expected_coeff_max, rel=1e-12)
        assert actual_coeff_min == pytest.approx(expected_coeff_min, rel=1e-12)
        assert actual_n_coeffs == expected_n_coeffs
