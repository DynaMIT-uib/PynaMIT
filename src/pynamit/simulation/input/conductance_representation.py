"""Helpers for conductance storage/interpolation representations.

This module centralizes conversions between the supported conductance storage
representations:

- legacy eta-space storage: ``etaP``, ``etaH``
- conductivity storage: ``SigmaP``, ``SigmaH``
- log-conductivity storage: ``logSigmaP``, ``logSigmaH``
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np


def conductance_timeseries_vars_for_mode(mode: str) -> dict[str, str]:
    """Return conductance variable names stored in a Timeseries for ``mode``."""
    if mode == "legacy_eta_linear":
        return {"etaP": "scalar", "etaH": "scalar"}
    if mode == "sigma_linear":
        return {"SigmaP": "scalar", "SigmaH": "scalar"}
    if mode == "sigma_log":
        return {"logSigmaP": "scalar", "logSigmaH": "scalar"}
    raise ValueError(
        "Invalid conductance_interpolation_mode="
        f"{mode!r}. Valid modes: 'legacy_eta_linear', 'sigma_linear', 'sigma_log'."
    )


def sigma_to_eta(
    sigmaP: np.ndarray,
    sigmaH: np.ndarray,
    *,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert conductance to resistivity representation with denominator floor."""
    sigma_floor_safe = max(float(sigma_floor), np.finfo(float).tiny)
    sigmaP_arr = np.asarray(sigmaP, dtype=float)
    sigmaH_arr = np.asarray(sigmaH, dtype=float)
    denom = sigmaP_arr * sigmaP_arr + sigmaH_arr * sigmaH_arr + sigma_floor_safe * sigma_floor_safe
    return sigmaP_arr / denom, sigmaH_arr / denom


def eta_to_sigma(
    etaP: np.ndarray,
    etaH: np.ndarray,
    *,
    valid_denom_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert resistivity representation back to conductance (best-effort)."""
    etaP_arr = np.asarray(etaP, dtype=float)
    etaH_arr = np.asarray(etaH, dtype=float)
    den = etaP_arr * etaP_arr + etaH_arr * etaH_arr
    sigmaP = np.full_like(etaP_arr, np.nan)
    sigmaH = np.full_like(etaH_arr, np.nan)
    valid = den > float(valid_denom_floor)
    if np.any(valid):
        sigmaP[valid] = etaP_arr[valid] / den[valid]
        sigmaH[valid] = etaH_arr[valid] / den[valid]
    return sigmaP, sigmaH


def encode_conductance_input_for_storage(
    *,
    Hall: np.ndarray,
    Pedersen: np.ndarray,
    mode: str,
    sigma_floor: float,
    logger: Optional[logging.Logger] = None,
) -> dict[str, np.ndarray]:
    """Encode conductance arrays for the configured storage/interpolation mode."""
    hall = np.asarray(Hall, dtype=float)
    ped = np.asarray(Pedersen, dtype=float)

    if mode == "legacy_eta_linear":
        den = hall * hall + ped * ped
        return {
            "etaP": ped / den,
            "etaH": hall / den,
        }

    if np.any(ped < 0.0):
        if logger is not None:
            logger.warning(
                "Negative Pedersen conductance supplied; clipping to nonnegative "
                "for conductance_interpolation_mode=%s.",
                mode,
            )
        ped = np.maximum(ped, 0.0)
    if np.any(hall < 0.0):
        if logger is not None:
            logger.warning(
                "Negative Hall conductance supplied; clipping to nonnegative "
                "for conductance_interpolation_mode=%s (Hall sign is geometry-driven).",
                mode,
            )
        hall = np.maximum(hall, 0.0)

    if mode == "sigma_linear":
        return {"SigmaP": ped, "SigmaH": hall}
    if mode == "sigma_log":
        floor = max(float(sigma_floor), np.finfo(float).tiny)
        return {
            "logSigmaP": np.log(ped + floor),
            "logSigmaH": np.log(hall + floor),
        }

    raise ValueError(
        "Invalid conductance_interpolation_mode="
        f"{mode!r}. Valid modes: 'legacy_eta_linear', 'sigma_linear', 'sigma_log'."
    )


def decode_conductance_representation_to_grids(
    *,
    data: dict[str, np.ndarray],
    eval_scalar_coeffs_to_grid: Callable[[np.ndarray], np.ndarray],
    sigma_floor: float,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a conductance representation to ``(SigmaP, SigmaH, etaP, etaH)`` on a grid.

    ``eval_scalar_coeffs_to_grid`` should evaluate scalar coefficients to the target grid.
    """
    sigma_floor_safe = max(float(sigma_floor), np.finfo(float).tiny)

    if "etaP" in data and "etaH" in data:
        etaP_grid = np.asarray(eval_scalar_coeffs_to_grid(data["etaP"]), dtype=float)
        etaH_grid = np.asarray(eval_scalar_coeffs_to_grid(data["etaH"]), dtype=float)
        sigmaP_grid, sigmaH_grid = eta_to_sigma(etaP_grid, etaH_grid)
        return sigmaP_grid, sigmaH_grid, etaP_grid, etaH_grid

    if "SigmaP" in data and "SigmaH" in data:
        sigmaP_grid = np.asarray(eval_scalar_coeffs_to_grid(data["SigmaP"]), dtype=float)
        sigmaH_grid = np.asarray(eval_scalar_coeffs_to_grid(data["SigmaH"]), dtype=float)
    elif "logSigmaP" in data and "logSigmaH" in data:
        log_sigmaP_grid = np.asarray(eval_scalar_coeffs_to_grid(data["logSigmaP"]), dtype=float)
        log_sigmaH_grid = np.asarray(eval_scalar_coeffs_to_grid(data["logSigmaH"]), dtype=float)
        sigmaP_grid = np.exp(log_sigmaP_grid) - sigma_floor_safe
        sigmaH_grid = np.exp(log_sigmaH_grid) - sigma_floor_safe
    else:
        raise KeyError(
            "Unsupported conductance representation. Expected "
            "('etaP','etaH'), ('SigmaP','SigmaH'), or ('logSigmaP','logSigmaH')."
        )

    if np.any(sigmaP_grid < 0.0):
        if logger is not None:
            logger.warning(
                "Negative Pedersen conductance encountered after interpolation; "
                "clipping to nonnegative."
            )
        sigmaP_grid = np.maximum(sigmaP_grid, 0.0)
    if np.any(sigmaH_grid < 0.0):
        if logger is not None:
            logger.warning(
                "Negative Hall conductance encountered after interpolation; "
                "clipping to nonnegative (Hall sign is geometry-driven)."
            )
        sigmaH_grid = np.maximum(sigmaH_grid, 0.0)

    etaP_grid, etaH_grid = sigma_to_eta(sigmaP_grid, sigmaH_grid, sigma_floor=sigma_floor_safe)
    return sigmaP_grid, sigmaH_grid, etaP_grid, etaH_grid
