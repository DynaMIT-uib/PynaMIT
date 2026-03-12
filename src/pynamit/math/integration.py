"""Time integration helpers for linear and nonlinear evolution problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import expm_multiply

from pynamit.utils import asarray, to_numpy, use_jax


class Integrator(ABC):
    """Abstract base class for one-step time integrators."""

    @abstractmethod
    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        linear_operator: Optional[np.ndarray] = None,
        steady_state: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Advance the state by one step."""


class EulerIntegrator(Integrator):
    """Explicit Euler integrator."""

    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        if rates_func is None:
            raise ValueError("rates_func is required for Euler integration.")

        rates = rates_func(y, 0.0)
        return asarray(y) + float(dt) * asarray(rates)


class ExponentialIntegrator(Integrator):
    """Exact one-step integrator for frozen linear systems."""

    @staticmethod
    def _compute_substeps(L_host: np.ndarray, dt: float, kwargs: dict[str, Any]) -> int:
        """Return the requested number of exponential substeps."""
        max_step_scale = kwargs.get("max_step_scale")
        max_substeps = int(kwargs.get("max_substeps", 512))
        if max_step_scale is not None and float(max_step_scale) > 0.0:
            spectral_scale = float(np.linalg.norm(L_host, ord=np.inf) * abs(float(dt)))
            n_substeps = max(1, int(np.ceil(spectral_scale / float(max_step_scale))))
            return min(n_substeps, max_substeps)
        return 1

    @staticmethod
    def _reshape_dense_square_operator(dense: np.ndarray) -> np.ndarray:
        """Flatten square operator tensors to a 2-D matrix."""
        dense_arr = np.asarray(dense, dtype=float)
        if dense_arr.ndim == 2:
            return dense_arr
        n_flat = int(np.sqrt(dense_arr.size))
        if n_flat * n_flat != int(dense_arr.size):
            raise ValueError(
                "Dense operator could not be reshaped to a square matrix: "
                f"shape={dense_arr.shape}, size={dense_arr.size}."
            )
        return dense_arr.reshape(n_flat, n_flat)

    @staticmethod
    def _should_use_affine_expm_action(n: int, kwargs: dict[str, Any]) -> bool:
        """Choose between dense `expm` and action-based `expm_multiply`."""
        mode = str(kwargs.get("affine_expm_mode", "auto")).lower()
        if mode not in ("auto", "dense", "action"):
            raise ValueError(
                f"affine_expm_mode must be one of {{'auto', 'dense', 'action'}}, got {mode!r}."
            )
        if mode == "dense":
            return False
        if mode == "action":
            if use_jax():
                raise NotImplementedError(
                    "Exponential affine solver 'expm_multiply' is not supported with the JAX "
                    "backend. Use exponential_solver='expm' or a non-exponential integrator."
                )
            return True
        threshold = int(kwargs.get("affine_action_dim_threshold", 512))
        return int(n) >= max(1, threshold)

    @staticmethod
    def _try_get_dense_operator(linear_operator: Any) -> Optional[np.ndarray]:
        """Return a dense numpy operator when one is cheaply available."""
        if hasattr(linear_operator, "toarray"):
            return ExponentialIntegrator._reshape_dense_square_operator(linear_operator.toarray())
        if hasattr(linear_operator, "to_dense"):
            try:
                return ExponentialIntegrator._reshape_dense_square_operator(
                    linear_operator.to_dense()
                )
            except Exception:
                return None
        if hasattr(linear_operator, "matvec") or isinstance(linear_operator, ScipyLinearOperator):
            return None
        return ExponentialIntegrator._reshape_dense_square_operator(linear_operator)

    @classmethod
    def _require_dense_operator(cls, linear_operator: Any) -> np.ndarray:
        """Return a dense operator or raise when the branch requires one."""
        dense = cls._try_get_dense_operator(linear_operator)
        if dense is None:
            raise TypeError("Dense exponential stepping requires a dense linear operator.")
        return dense

    @classmethod
    def _get_action_operator(cls, linear_operator: Any) -> Any:
        """Return an operator suitable for `expm_multiply`."""
        dense = cls._try_get_dense_operator(linear_operator)
        if dense is not None:
            return dense
        if hasattr(linear_operator, "as_linear_operator"):
            return linear_operator.as_linear_operator()
        if isinstance(linear_operator, ScipyLinearOperator):
            return linear_operator
        return np.asarray(linear_operator, dtype=float)

    @classmethod
    def _get_substeps_for_operator(
        cls, linear_operator: Any, dt: float, kwargs: dict[str, Any]
    ) -> int:
        """Return substeps based on a dense operator when available."""
        dense = cls._try_get_dense_operator(linear_operator)
        if dense is None:
            return 1
        return cls._compute_substeps(dense, dt, kwargs)

    @classmethod
    def _maybe_trace_of_operator(
        cls, linear_operator: Any, *, trace_dim_limit: int
    ) -> Optional[float]:
        """Return an exact operator trace when it is cheap enough to compute."""
        dense = cls._try_get_dense_operator(linear_operator)
        if dense is not None:
            return float(np.trace(dense))

        if hasattr(linear_operator, "as_linear_operator"):
            base_op = linear_operator.as_linear_operator()
        elif isinstance(linear_operator, ScipyLinearOperator):
            base_op = linear_operator
        else:
            return float(np.trace(np.asarray(linear_operator, dtype=float)))

        if base_op.shape[0] != base_op.shape[1] or int(base_op.shape[0]) > int(trace_dim_limit):
            return None

        n = int(base_op.shape[0])
        eye = np.eye(n, dtype=np.result_type(getattr(base_op, "dtype", np.float64), np.float64))
        if hasattr(base_op, "matmat"):
            applied = np.asarray(base_op.matmat(eye), dtype=float)
        else:
            applied = np.column_stack(
                [np.asarray(base_op.matvec(eye[:, j]), dtype=float) for j in range(n)]
            )
        return float(np.trace(applied))

    @staticmethod
    def _expm_multiply_with_optional_trace(
        *, operator: Any, y: np.ndarray, dt: float, trace: Optional[float]
    ) -> np.ndarray:
        """Apply `exp(operator * dt)` with an explicit trace when available."""
        kwargs: dict[str, Any] = {}
        if trace is not None:
            kwargs["traceA"] = float(dt) * float(trace)
        return np.asarray(expm_multiply(operator * float(dt), y, **kwargs), dtype=float)

    @classmethod
    def _affine_step_dense_augmented(
        cls,
        *,
        L_host: np.ndarray,
        y_host: np.ndarray,
        forcing_arr: np.ndarray,
        dt: float,
        n_substeps: int,
    ) -> np.ndarray:
        """Exact affine step using a dense augmented exponential."""
        n = int(y_host.size)
        aug = np.zeros((n + 1, n + 1), dtype=L_host.dtype)
        aug[:n, :n] = L_host
        aug[:n, n] = forcing_arr

        if n_substeps == 1:
            propagator = np.asarray(scipy.linalg.expm(aug * float(dt)), dtype=float)
            y_aug = np.concatenate([y_host, np.array([1.0], dtype=float)])
            return asarray((propagator @ y_aug)[:n])

        dt_sub = float(dt) / float(n_substeps)
        propagator = np.asarray(scipy.linalg.expm(aug * dt_sub), dtype=float)
        y_aug = np.concatenate([y_host, np.array([1.0], dtype=float)])
        for _ in range(n_substeps):
            y_aug = propagator @ y_aug
        return asarray(y_aug[:n])

    @classmethod
    def _affine_step_expm_multiply_augmented(
        cls,
        *,
        linear_operator: Any,
        y_host: np.ndarray,
        forcing_arr: np.ndarray,
        dt: float,
        n_substeps: int,
        trace_dim_limit: int,
    ) -> np.ndarray:
        """Exact affine step via `expm_multiply` on the augmented system."""
        n = int(y_host.size)
        dense = cls._try_get_dense_operator(linear_operator)
        trace_base = cls._maybe_trace_of_operator(linear_operator, trace_dim_limit=trace_dim_limit)

        if dense is not None:
            aug = np.zeros((n + 1, n + 1), dtype=dense.dtype)
            aug[:n, :n] = dense
            aug[:n, n] = forcing_arr
            augmented_op: Any = aug
            trace_aug = float(np.trace(aug))
        else:
            base_op = cls._get_action_operator(linear_operator)
            forcing_np = np.asarray(forcing_arr, dtype=float)
            dtype = np.result_type(
                getattr(base_op, "dtype", np.float64), forcing_np.dtype, np.float64
            )

            def aug_matvec(v: np.ndarray) -> np.ndarray:
                v_arr = np.asarray(v, dtype=dtype).reshape(n + 1)
                top = np.asarray(base_op.matvec(v_arr[:n]), dtype=dtype).reshape(n)
                top = top + forcing_np * v_arr[n]
                return np.concatenate([top, np.zeros(1, dtype=dtype)])

            def aug_rmatvec(v: np.ndarray) -> np.ndarray:
                v_arr = np.asarray(v, dtype=dtype).reshape(n + 1)
                if not hasattr(base_op, "rmatvec"):
                    raise TypeError(
                        "matrix-free expm_multiply requires rmatvec on the base operator."
                    )
                top = np.asarray(base_op.rmatvec(v_arr[:n]), dtype=dtype).reshape(n)
                bottom = np.array([np.dot(forcing_np.conj(), v_arr[:n])], dtype=dtype)
                return np.concatenate([top, bottom])

            def aug_matmat(V: np.ndarray) -> np.ndarray:
                V_arr = np.asarray(V, dtype=dtype).reshape(n + 1, -1)
                if hasattr(base_op, "matmat"):
                    top = np.asarray(base_op.matmat(V_arr[:n]), dtype=dtype)
                else:
                    top = np.column_stack(
                        [
                            np.asarray(base_op.matvec(V_arr[:n, j]), dtype=dtype)
                            for j in range(V_arr.shape[1])
                        ]
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
                        [
                            np.asarray(base_op.rmatvec(V_arr[:n, j]), dtype=dtype)
                            for j in range(V_arr.shape[1])
                        ]
                    )
                else:
                    raise TypeError(
                        "matrix-free expm_multiply requires rmatmat or rmatvec on the base operator."
                    )
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
            trace_aug = trace_base

        y_aug = np.concatenate([y_host, np.array([1.0], dtype=float)])
        if n_substeps == 1:
            y_next_aug = cls._expm_multiply_with_optional_trace(
                operator=augmented_op, y=y_aug, dt=float(dt), trace=trace_aug
            )
            return asarray(y_next_aug[:n])

        dt_sub = float(dt) / float(n_substeps)
        for _ in range(n_substeps):
            y_aug = cls._expm_multiply_with_optional_trace(
                operator=augmented_op, y=y_aug, dt=dt_sub, trace=trace_aug
            )
        return asarray(y_aug[:n])

    @classmethod
    def _homogeneous_step_expm_multiply(
        cls,
        *,
        linear_operator: Any,
        y_host: np.ndarray,
        dt: float,
        n_substeps: int,
        trace_dim_limit: int,
    ) -> np.ndarray:
        """Apply `exp(L dt)` to `y` without forming a dense exponential."""
        base_op = cls._get_action_operator(linear_operator)
        trace = cls._maybe_trace_of_operator(linear_operator, trace_dim_limit=trace_dim_limit)

        if n_substeps == 1:
            return asarray(
                cls._expm_multiply_with_optional_trace(
                    operator=base_op, y=y_host, dt=float(dt), trace=trace
                )
            )

        dt_sub = float(dt) / float(n_substeps)
        y_curr = np.asarray(y_host, dtype=float)
        for _ in range(n_substeps):
            y_curr = cls._expm_multiply_with_optional_trace(
                operator=base_op, y=y_curr, dt=dt_sub, trace=trace
            )
        return asarray(y_curr)

    def step(
        self,
        y: np.ndarray,
        dt: float,
        linear_operator: Optional[np.ndarray] = None,
        steady_state: Optional[np.ndarray] = None,
        forcing: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Advance one frozen linear step."""
        if linear_operator is None:
            raise ValueError("linear_operator is required for Exponential integration.")

        y_host = np.asarray(y, dtype=float).reshape(-1)
        forcing_arr = None if forcing is None else np.asarray(forcing, dtype=float).reshape(-1)
        trace_dim_limit = int(kwargs.get("trace_dim_limit", 256))

        if forcing_arr is not None:
            n = int(y_host.size)
            if forcing_arr.size != n:
                raise ValueError(
                    "forcing size mismatch in ExponentialIntegrator: "
                    f"got {forcing_arr.size}, expected {n}."
                )
            use_action = self._should_use_affine_expm_action(n, kwargs)
            n_substeps = self._get_substeps_for_operator(linear_operator, dt, kwargs)
            if use_action:
                return self._affine_step_expm_multiply_augmented(
                    linear_operator=linear_operator,
                    y_host=y_host,
                    forcing_arr=forcing_arr,
                    dt=float(dt),
                    n_substeps=n_substeps,
                    trace_dim_limit=trace_dim_limit,
                )

            L_host = self._require_dense_operator(linear_operator)
            return self._affine_step_dense_augmented(
                L_host=L_host,
                y_host=y_host,
                forcing_arr=forcing_arr,
                dt=float(dt),
                n_substeps=n_substeps,
            )

        if steady_state is None:
            raise ValueError(
                "Exponential integration requires either forcing (affine form) or steady_state."
            )

        y_ss = np.asarray(steady_state, dtype=float).reshape(y_host.shape)
        diff = y_host - y_ss
        use_action = self._should_use_affine_expm_action(int(diff.size), kwargs)
        n_substeps = self._get_substeps_for_operator(linear_operator, dt, kwargs)
        if use_action:
            decayed = self._homogeneous_step_expm_multiply(
                linear_operator=linear_operator,
                y_host=diff,
                dt=float(dt),
                n_substeps=n_substeps,
                trace_dim_limit=trace_dim_limit,
            )
            return asarray(y_ss + decayed)

        L_host = self._require_dense_operator(linear_operator)
        if n_substeps == 1:
            propagator = np.asarray(scipy.linalg.expm(L_host * float(dt)), dtype=float)
            return asarray(y_ss + propagator @ diff)

        dt_sub = float(dt) / float(n_substeps)
        propagator = np.asarray(scipy.linalg.expm(L_host * dt_sub), dtype=float)
        diff_curr = np.asarray(diff, dtype=float)
        for _ in range(n_substeps):
            diff_curr = propagator @ diff_curr
        return asarray(y_ss + diff_curr)


class ScipySolveIVPIntegrator(Integrator):
    """Thin wrapper around `scipy.integrate.solve_ivp`."""

    def __init__(self, method: str = "RK45", **solver_options: Any):
        self.method = method
        self.solver_options = solver_options

    def step(
        self,
        y: np.ndarray,
        dt: float,
        rates_func: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        if rates_func is None:
            raise ValueError("rates_func is required for Scipy integration.")

        y_np = to_numpy(y)

        def rhs_wrapper(t: float, y_curr: np.ndarray) -> np.ndarray:
            return to_numpy(rates_func(y_curr, t))

        sol = solve_ivp(
            fun=rhs_wrapper,
            t_span=(0.0, float(dt)),
            y0=y_np,
            method=self.method,
            t_eval=[float(dt)],
            dense_output=False,
            **self.solver_options,
        )
        if not sol.success:
            raise RuntimeError(
                f"solve_ivp integrator '{self.method}' failed with status {sol.status}: "
                f"{sol.message}"
            )
        if sol.y.shape[1] == 0:
            raise RuntimeError(
                f"solve_ivp integrator '{self.method}' returned no solution sample at t={dt}."
            )
        return asarray(sol.y[:, -1])
