"""Time Integration Module.

This module implements the Method of Lines pattern, decoupling time integration
algorithms from physics rate calculations. It defines an abstract `Integrator`
interface and concrete implementations for common schemes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union
import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
from pynamit.utils import asarray, xp, to_numpy

class Integrator(ABC):
    """Abstract Base Class for Time Integrators."""

    @abstractmethod
    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        linear_operator: Optional[np.ndarray] = None,
        steady_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Perform a single time step.

        y(t + dt) = Step(y(t), dt, ...)

        Parameters
        ----------
        y : np.ndarray
            Current state vector.
        dt : float
            Time step size.
        rates_func : Callable, optional
            Function calculating dy/dt = f(y, t). Required for explicit methods.
        linear_operator : np.ndarray, optional
            System matrix L for linear systems dy/dt = L y + K.
            Required for Exponential methods.
        steady_state : np.ndarray, optional
            Steady state solution y_ss = -L^-1 K.
            Used for efficient exponential integration: y(t+dt) = y_ss + exp(L dt)(y(t) - y_ss).
        """
        pass


class EulerIntegrator(Integrator):
    """Explicit Euler Integrator.
    
    y_{n+1} = y_n + dt * f(y_n, t_n)
    """

    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        if rates_func is None:
            raise ValueError("rates_func is required for Euler integration.")
            
        rates = rates_func(y, 0.0) # Assuming autonomous for now or t passed in kwargs
        return asarray(y) + dt * asarray(rates)


class ExponentialIntegrator(Integrator):
    """Exponential Time Differencing / Exponential Euler Integrator.
    
    Exact for linear systems with constant coefficients:
    dy/dt = L y + K
    
    Solution:
    y(t+dt) = exp(L dt) * y(t) + L^{-1} * (exp(L dt) - I) * K
    
    Alternative form using steady state y_ss = -L^{-1} K:
    y(t+dt) = y_ss + exp(L dt) * (y(t) - y_ss)
    """

    def step(
        self,
        y: np.ndarray,
        dt: float,
        linear_operator: Optional[np.ndarray] = None,
        steady_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if linear_operator is None:
            raise ValueError("linear_operator is required for Exponential integration.")
        
        # Calculate propagator exp(L * dt)
        # Note: For large sparse matrices, dense expm is expensive.
        # This assumes the system size is manageable (spectral coefficients).
        if hasattr(linear_operator, "toarray"):
            L_dense = linear_operator.toarray()
        else:
            L_dense = asarray(linear_operator)

        # Handle JAX/Numpy dispatch for expm if needed, or stick to scipy
        # For now, using scipy.linalg.expm on CPU (standard numpy)
        # If input is JAX array, we might need jax.scipy.linalg.expm
        import scipy.linalg
        
        # Ensure we are working with standard numpy for scipy.linalg
        # (Unless we add JAX logic here)
        L_host = np.array(L_dense)
        propagator = scipy.linalg.expm(L_host * dt)
        propagator = asarray(propagator) # Move back to device/backend if needed

        if steady_state is not None:
             # Form: y_next = y_ss + P @ (y - y_ss)
             diff = asarray(y) - asarray(steady_state)
             decayed = propagator @ diff
             return asarray(steady_state) + decayed
        else:
             # Standard ETD1 form without pre-calculated steady state?
             # Requires K (forcing vector).
             # y(t+dt) = P y + (P - I) L^-1 K
             # This path is less numerically stable if L is singular.
             # We prioritize the steady_state form as established in PoloidalSystemMatrices.
             raise ValueError("steady_state argument is currently required for Exponential integration.")


class ScipySolveIVPIntegrator(Integrator):
    """Wrapper around scipy.integrate.solve_ivp."""
    
    def __init__(self, method: str = "RK45", **solver_options):
        self.method = method
        self.solver_options = solver_options

    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        if rates_func is None:
             raise ValueError("rates_func is required for Scipy integration.")

        # Scipy requires numpy arrays
        y_np = to_numpy(y) # Ensure start state is CPU numpy
        
        def rhs_wrapper(t, y_curr):
            # y_curr comes from scipy as numpy array
            # rates_func might expect/return backend array (if JAX)
            # But rates_func in State wraps compute_rates which handles backend conversion.
            # Ideally compute_rates accepts whatever.
            # But State.evolve methods wrap it.
            # Let's assume rates_func returns something we need to ensure is numpy for scipy
             
            # Convert input to backend if needed?
            # State.rates_func currently: 
            #   return self.poloidal_matrices.compute_rates(m_ind=y, ...)
            # poloidal.compute_rates does: backend_m_ind = asarray(m_ind)
            # So passing numpy y is fine.
            
            # The output of rates_func is backend array (e.g. JAX array).
            # We must convert to numpy for scipy.
            rates = rates_func(y_curr, t)
            return to_numpy(rates)

        sol = solve_ivp(
            fun=rhs_wrapper,
            t_span=(0, dt),
            y0=y_np,
            method=self.method,
            t_eval=[dt],
            dense_output=False,
            **self.solver_options
        )
        
        # Return state on backend
        return asarray(sol.y[:, -1])
