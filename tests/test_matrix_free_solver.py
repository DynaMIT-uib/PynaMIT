"""Tests for matrix-free iterative solver paths in full induction mode."""

import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, MainfieldKind, SimulationMode

@pytest.mark.parametrize("solver", ["lsmr", "cgls"])
def test_full_induction_iterative_solvers(solver):
    """Verify that full induction runs successfully with iterative solvers.
    
    This ensures the matrix-free path (State.steady_state_coupled with LinearMap)
    is correctly exercised and converges, unlike test_dynamic_solver which forces SVD.
    """
    # Run a short simulation
    sim = run_pynamit(
        final_time=0.01,
        dt=0.01,
        Nmax=8,
        Mmax=4,
        Ncs=16,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        # Use simple spectral mode for speed
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        connect_hemispheres=True,
        # Explicitly request iterative solver
        least_squares_solver=solver,
        # Ensure we don't carry over persistent state issues
        steady_state_initialization=True,
    )
    
    # Check that output exists and is finite
    state_ds = sim.io.load_dataset("state")
    assert state_ds is not None
    
    m_ind = state_ds["SH_m_ind"].values[-1]
    psi = state_ds["SH_psi"].values[-1]
    
    assert np.all(np.isfinite(m_ind))
    assert np.all(np.isfinite(psi))
    
    # Check norm is reasonable (not zero, not explosive)
    nm_mind = np.linalg.norm(m_ind)
    nm_psi = np.linalg.norm(psi)
    
    print(f"Solver {solver}: |m_ind|={nm_mind:.4e}, |psi|={nm_psi:.4e}")
    
    # Basic sanity bounds (based on typical IGRF responses)
    assert nm_mind > 1e-15
    assert nm_mind < 1e-3
    # psi can be zero if no toroidal driving force exists
    assert nm_psi < 1e-1


@pytest.mark.filterwarnings("error:LSMR may not have converged.*:RuntimeWarning")
@pytest.mark.filterwarnings("error:Projected Tikhonov CG may not have converged.*:RuntimeWarning")
def test_full_induction_coupled_column_scale_cache(tmp_path):
    """Coupled steady-state iterative solve caches/reuses column scales and clears on invalidate.

    The conductance-update path clears this cache via ``State._invalidate_caches()``.
    """
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        plotsteps=1,
        Nmax=8,
        Mmax=4,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        connect_hemispheres=True,
        multi_data=True,
        least_squares_solver="lsmr",
        least_squares_preconditioner=None,
        dense_full_operators=False,
        benchmark_mode=True,
        run_directory=str(tmp_path / "coupled_scale_cache"),
    )

    st = sim.state
    cache = getattr(st, "_coupled_steady_state_column_scale_cache")
    n = st.solution_space.index_length
    key = (bool(st.apply_psi_gauge), int(n))
    cache_key = key
    if cache_key not in cache:
        for suffix in ("projected_tikhonov", "projected_gmres"):
            candidate = (bool(st.apply_psi_gauge), int(n), suffix)
            if candidate in cache:
                cache_key = candidate
                break

    assert cache_key in cache
    col_scale_0 = np.asarray(cache[cache_key])
    assert col_scale_0.shape == (2 * n,)
    assert np.all(np.isfinite(col_scale_0))
    assert np.all(col_scale_0 > 0)
    first_id = id(cache[cache_key])

    # Repeat the same coupled steady-state solve and verify the cached scale is reused.
    forcing = st.build_coupled_forcing(np.zeros((2, n), dtype=float))
    _ = st._solve_linear_steady_state(
        linear_operator=None,
        forcing=forcing,
        solution_shape=(2, n),
        solver="lsmr",
        preconditioner=None,
        use_pinning=st.apply_psi_gauge,
    )
    cache_after = getattr(st, "_coupled_steady_state_column_scale_cache")
    assert cache_key in cache_after
    assert id(cache_after[cache_key]) == first_id

    # Conductance updates clear this via State._invalidate_caches(); test the clear directly.
    st._invalidate_caches()
    assert getattr(st, "_coupled_steady_state_column_scale_cache") == {}
