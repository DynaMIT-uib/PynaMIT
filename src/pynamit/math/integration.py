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
        forcing: Optional[np.ndarray] = None,
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
        y_host = np.array(y)

        max_step_scale = kwargs.get("max_step_scale", None)
        max_substeps = int(kwargs.get("max_substeps", 512))
        if max_step_scale is not None and float(max_step_scale) > 0.0:
            spectral_scale = float(np.linalg.norm(L_host, ord=np.inf) * abs(float(dt)))
            n_substeps = max(1, int(np.ceil(spectral_scale / float(max_step_scale))))
            n_substeps = min(n_substeps, max_substeps)
        else:
            n_substeps = 1
        forcing_arr = None if forcing is None else np.array(forcing).reshape(-1)

        if forcing_arr is not None:
             # Affine linear system:
             #   y' = L y + K
             # Exact one-step with pure expm via augmented matrix:
             #   d/dt [y; 1] = [[L, K], [0, 0]] [y; 1]
             # This remains valid even when L is singular / no exact steady state exists.
             n = int(y_host.size)
             if forcing_arr.size != n:
                 raise ValueError(
                     "forcing size mismatch in ExponentialIntegrator: "
                     f"got {forcing_arr.size}, expected {n}."
                 )
             if n_substeps == 1:
                 aug = np.zeros((n + 1, n + 1), dtype=L_host.dtype)
                 aug[:n, :n] = L_host
                 aug[:n, n] = forcing_arr
                 propagator = asarray(scipy.linalg.expm(aug * float(dt)))
                 y_aug = np.concatenate([y_host, np.array([1.0], dtype=y_host.dtype)])
                 y_next_aug = propagator @ y_aug
                 return asarray(y_next_aug[:n])

             dt_sub = float(dt) / float(n_substeps)
             aug = np.zeros((n + 1, n + 1), dtype=L_host.dtype)
             aug[:n, :n] = L_host
             aug[:n, n] = forcing_arr
             propagator = asarray(scipy.linalg.expm(aug * dt_sub))
             y_aug = np.concatenate([y_host, np.array([1.0], dtype=y_host.dtype)])
             for _ in range(n_substeps):
                 y_aug = propagator @ y_aug
             return asarray(y_aug[:n])

        if steady_state is not None:
             # Form: y_next = y_ss + P @ (y - y_ss)
             y_ss = asarray(steady_state)
             if n_substeps == 1:
                 propagator = asarray(scipy.linalg.expm(L_host * float(dt)))
                 diff = asarray(y) - y_ss
                 decayed = propagator @ diff
                 return y_ss + decayed

             dt_sub = float(dt) / float(n_substeps)
             propagator = asarray(scipy.linalg.expm(L_host * dt_sub))
             y_curr = asarray(y_host)
             for _ in range(n_substeps):
                 diff = y_curr - y_ss
                 y_curr = y_ss + propagator @ diff
             return asarray(y_curr)
        else:
             raise ValueError(
                 "Exponential integration requires either forcing (affine form) "
                 "or steady_state."
             )


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
