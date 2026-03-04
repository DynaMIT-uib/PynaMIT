"""Runtime diagnostics helpers for the simulation state."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateDiagnostics:
    """Own runtime diagnostics that are derived from live state operators."""

    def __init__(self, state: Any) -> None:
        self._state = state
        self._coupled_stability_warned_keys: set[Tuple[Any, ...]] = set()

    def reset_stability_warnings(self) -> None:
        """Forget previously emitted coupled-stability warnings."""
        self._coupled_stability_warned_keys.clear()

    def analyze_coupled_stability(
        self,
        l_flat: np.ndarray,
        *,
        label: str,
        unstable_tol: float = 1e-10,
    ) -> Dict[str, float]:
        """Analyze coupled-operator spectrum and warn on unstable modes."""
        arr = np.asarray(l_flat, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    "Coupled stability analysis requires a square matrix, "
                    f"got {arr.shape}."
                )

        eigvals = np.linalg.eigvals(arr)
        real = np.real(eigvals)
        max_real = float(np.max(real)) if real.size > 0 else 0.0
        min_real = float(np.min(real)) if real.size > 0 else 0.0
        n_pos = int(np.sum(real > float(unstable_tol)))
        n_total = int(real.size)

        report = {
            "max_real": max_real,
            "min_real": min_real,
            "positive_real_count": float(n_pos),
            "n_eigs": float(n_total),
        }

        if max_real > float(unstable_tol):
            key = (
                label,
                arr.shape[0],
                round(max_real, 9),
                round(min_real, 9),
                n_pos,
            )
            if key not in self._coupled_stability_warned_keys:
                msg = (
                    "Coupled full-induction operator has unstable eigenmodes "
                    f"(label={label}, max Re(lambda)={max_real:.3e}, "
                    f"positive modes={n_pos}/{n_total}). "
                    "Explicit Euler integration is expected to be unstable for this operator."
                )
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._coupled_stability_warned_keys.add(key)
        return report

    def get_coupled_stability_report(
        self,
        *,
        source: Literal["dense", "sparse", "auto"] = "dense",
        use_pinning: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Return spectral stability report for the coupled full-induction operator."""
        st = self._state
        if use_pinning is None:
            use_pinning = st.apply_psi_gauge
        l_flat = np.asarray(
            st.get_coupled_induction_matrix(
                source=source,
                flatten=True,
                use_pinning=use_pinning,
            )
        )
        return self.analyze_coupled_stability(
            l_flat,
            label=f"{source}:pinning={int(bool(use_pinning))}",
        )
