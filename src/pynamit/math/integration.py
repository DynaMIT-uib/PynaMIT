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
from scipy.sparse.linalg import expm_multiply, LinearOperator as ScipyLinearOperator
from scipy.integrate import solve_ivp
from pynamit.utils import asarray, xp, to_numpy, use_jax

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

    @staticmethod
    def _compute_substeps(L_host: np.ndarray, dt: float, kwargs: dict[str, Any]) -> int:
        """Return number of affine/exponential substeps for stiff dense systems."""
        max_step_scale = kwargs.get("max_step_scale", None)
        max_substeps = int(kwargs.get("max_substeps", 512))
        if max_step_scale is not None and float(max_step_scale) > 0.0:
            spectral_scale = float(np.linalg.norm(L_host, ord=np.inf) * abs(float(dt)))
            n_substeps = max(1, int(np.ceil(spectral_scale / float(max_step_scale))))
            return min(n_substeps, max_substeps)
        return 1

    @staticmethod
    def _should_use_affine_expm_action(n: int, kwargs: dict[str, Any]) -> bool:
        """Choose a low-memory affine exponential step implementation.

        The action method is mathematically equivalent to the `phi1` formulation
        for affine systems, but is implemented via `expm_multiply` on the
        augmented operator to keep the code path compact and avoid building a
        dense matrix exponential for large systems.
        """
        mode = str(kwargs.get("affine_expm_mode", "auto")).lower()
        if mode not in ("auto", "dense", "action"):
            raise ValueError(
                "affine_expm_mode must be one of {'auto', 'dense', 'action'}, "
                f"got {mode!r}."
            )
        if mode == "dense":
            return False
        if mode == "action":
            if use_jax():
                raise NotImplementedError(
                    "Exponential affine solver 'expm_multiply' is not supported with the JAX backend. "
                    "Use exponential_solver='expm' or a non-exponential integrator."
                )
            return True
        threshold = int(kwargs.get("affine_action_dim_threshold", 512))
        return int(n) >= max(1, threshold)

    @staticmethod
    def _affine_step_dense_augmented(
        *,
        L_host: np.ndarray,
        y_host: np.ndarray,
        forcing_arr: np.ndarray,
        dt: float,
        n_substeps: int,
    ) -> np.ndarray:
        """Exact affine step using dense matrix exponential on augmented system."""
        n = int(y_host.size)
        aug = np.zeros((n + 1, n + 1), dtype=L_host.dtype)
        aug[:n, :n] = L_host
        aug[:n, n] = forcing_arr

        if n_substeps == 1:
            propagator = asarray(scipy.linalg.expm(aug * float(dt)))
            y_aug = np.concatenate([y_host, np.array([1.0], dtype=y_host.dtype)])
            y_next_aug = propagator @ y_aug
            return asarray(y_next_aug[:n])

        dt_sub = float(dt) / float(n_substeps)
        propagator = asarray(scipy.linalg.expm(aug * dt_sub))
        y_aug = np.concatenate([y_host, np.array([1.0], dtype=y_host.dtype)])
        for _ in range(n_substeps):
            y_aug = propagator @ y_aug
        return asarray(y_aug[:n])

    @staticmethod
    def _affine_step_expm_multiply_augmented(
        *,
        linear_operator: Any,
        y_host: np.ndarray,
        forcing_arr: np.ndarray,
        dt: float,
        n_substeps: int,
    ) -> np.ndarray:
        """Exact affine step via `expm_multiply` on the augmented operator.

        This is equivalent to the `expm + phi1` affine update, but avoids forming
        the dense matrix exponential (which can dominate memory use).
        """
        n = int(y_host.size)

        if hasattr(linear_operator, "as_linear_operator"):
            base_op = linear_operator.as_linear_operator()
        elif isinstance(linear_operator, ScipyLinearOperator):
            base_op = linear_operator
        else:
            base_op = None

        if base_op is None:
            L_host = np.asarray(linear_operator)
            aug = np.zeros((n + 1, n + 1), dtype=L_host.dtype)
            aug[:n, :n] = L_host
            aug[:n, n] = forcing_arr
            augmented_op: Any = aug
        else:
            forcing_np = np.asarray(forcing_arr)
            dtype = np.result_type(getattr(base_op, "dtype", np.float64), forcing_np.dtype, np.float64)

            def aug_matvec(v: np.ndarray) -> np.ndarray:
                v_arr = np.asarray(v, dtype=dtype).reshape(n + 1)
                top = np.asarray(base_op.matvec(v_arr[:n]), dtype=dtype).reshape(n)
                top = top + forcing_np * v_arr[n]
                return np.concatenate([top, np.zeros(1, dtype=dtype)])

            def aug_rmatvec(v: np.ndarray) -> np.ndarray:
                v_arr = np.asarray(v, dtype=dtype).reshape(n + 1)
                if hasattr(base_op, "rmatvec"):
                    top = np.asarray(base_op.rmatvec(v_arr[:n]), dtype=dtype).reshape(n)
                else:
                    raise TypeError("matrix-free expm_multiply requires rmatvec on the base operator.")
                bottom = np.array([np.dot(forcing_np.conj(), v_arr[:n])], dtype=dtype)
                return np.concatenate([top, bottom])

            def aug_matmat(V: np.ndarray) -> np.ndarray:
                V_arr = np.asarray(V, dtype=dtype).reshape(n + 1, -1)
                if hasattr(base_op, "matmat"):
                    top = np.asarray(base_op.matmat(V_arr[:n]), dtype=dtype)
                else:
                    top = np.column_stack(
                        [np.asarray(base_op.matvec(V_arr[:n, j]), dtype=dtype) for j in range(V_arr.shape[1])]
                    )
                top = top + forcing_np[:, None] * V_arr[n : n + 1, :]
                bottom = np.zeros((1, V_arr.shape[1]), dtype=dtype)
                return np.vstack([top, bottom])

            def aug_rmatmat(V: np.ndarray) -> np.ndarray:
                V_arr = np.asarray(V, dtype=dtype).reshape(n + 1, -1)
                if hasattr(base_op, "rmatmat"):
                    top = np.asarray(base_op.rmatmat(V_arr[:n]), dtype=dtype)
                elif hasattr(base_op, "rmatvec"):
                    top = np.column_stack(
                        [np.asarray(base_op.rmatvec(V_arr[:n, j]), dtype=dtype) for j in range(V_arr.shape[1])]
                    )
                else:
                    raise TypeError("matrix-free expm_multiply requires rmatmat or rmatvec on the base operator.")
                bottom = np.sum(forcing_np[:, None].conj() * V_arr[:n, :], axis=0, keepdims=True)
                return np.vstack([top, bottom])

            augmented_op = ScipyLinearOperator(
                shape=(n + 1, n + 1),
                matvec=aug_matvec,
                rmatvec=aug_rmatvec,
                matmat=aug_matmat,
                rmatmat=aug_rmatmat,
                dtype=dtype,
            )

        y_aug = np.concatenate([y_host, np.array([1.0], dtype=y_host.dtype)])
        if n_substeps == 1:
            y_next_aug = expm_multiply(augmented_op * float(dt), y_aug)
            return asarray(y_next_aug[:n])

        dt_sub = float(dt) / float(n_substeps)
        for _ in range(n_substeps):
            y_aug = expm_multiply(augmented_op * dt_sub, y_aug)
        return asarray(y_aug[:n])

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
        y_host = np.array(y)
        forcing_arr = None if forcing is None else np.array(forcing).reshape(-1)

        if forcing_arr is not None:
             # Affine linear system:
             #   y' = L y + K
             # We evaluate the exact affine step on the augmented operator. The
             # low-memory action path is equivalent to an `expm + phi1` update.
             n = int(y_host.size)
             if forcing_arr.size != n:
                 raise ValueError(
                     "forcing size mismatch in ExponentialIntegrator: "
                     f"got {forcing_arr.size}, expected {n}."
                 )
             if self._should_use_affine_expm_action(n, kwargs):
                 return self._affine_step_expm_multiply_augmented(
                     linear_operator=linear_operator,
                     y_host=y_host,
                     forcing_arr=forcing_arr,
                     dt=float(dt),
                     n_substeps=self._compute_substeps(
                         np.array(asarray(linear_operator.toarray() if hasattr(linear_operator, "toarray") else linear_operator)),
                         dt,
                         kwargs,
                     )
                     if not hasattr(linear_operator, "matvec")
                     else 1,
                 )
             if hasattr(linear_operator, "toarray"):
                 L_dense = linear_operator.toarray()
             else:
                 L_dense = asarray(linear_operator)
             L_host = np.array(L_dense)
             n_substeps = self._compute_substeps(L_host, dt, kwargs)
             return self._affine_step_dense_augmented(
                 L_host=L_host,
                 y_host=y_host,
                 forcing_arr=forcing_arr,
                 dt=float(dt),
                 n_substeps=n_substeps,
             )

        if steady_state is not None:
             if hasattr(linear_operator, "toarray"):
                 L_dense = linear_operator.toarray()
             else:
                 L_dense = asarray(linear_operator)
             L_host = np.array(L_dense)
             n_substeps = self._compute_substeps(L_host, dt, kwargs)
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
