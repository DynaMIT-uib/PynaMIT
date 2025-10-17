"""
Utility helpers for working with optional JAX acceleration.

This module centralises the logic needed to switch between NumPy and JAX
implementations at runtime. The behaviour is controlled through the
``PYNAMIT_USE_JAX`` environment variable (accepted values: ``1``, ``true``,
``yes``, ``on``) and can also be overridden programmatically via
``use_jax(True/False)``.

The helpers exposed here are intentionally lightweight wrappers that
preserve backwards compatibility with the existing NumPy-based code while
allowing simulation modules to opt in to JAX-backed array operations where
appropriate.
"""

from __future__ import annotations

import os
import types
from functools import wraps
from typing import Any, Iterable, Optional, Union

import numpy as _np

JAX_AVAILABLE = False
_jax_namespace: Optional[types.ModuleType] = None
_jax_array_type: tuple[type, ...] = ()
_jax_device_put = None
_jax_jit = None
_jax_vmap = None
_jax_to_numpy = None

try:  # pragma: no cover - we fall back gracefully when JAX is absent
    import jax
    import jax.numpy as _jnp

    from jax import jit as _jax_jit  # type: ignore[assignment]
    from jax import vmap as _jax_vmap  # type: ignore[assignment]

    _jax_namespace = _jnp
    _jax_device_put = jax.device_put
    _jax_to_numpy = jax.numpy.asarray

    try:
        # Newer versions expose the Array type publicly
        from jax import Array as _JaxArray  # type: ignore[attr-defined]

        _jax_array_type = (_JaxArray,)
    except Exception:  # pragma: no cover - fallback for older JAX versions
        from jax.interpreters.xla import DeviceArray as _DeviceArray  # type: ignore

        _jax_array_type = (_DeviceArray,)

    JAX_AVAILABLE = True

    # Enable double precision if available; ignore errors on older versions.
    try:  # pragma: no cover - configuration helper
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
except Exception:  # pragma: no cover - JAX is optional
    _jax_namespace = None


def _env_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


_USE_JAX = JAX_AVAILABLE and _env_flag(os.environ.get("PYNAMIT_USE_JAX"))


def use_jax(flag: Optional[bool] = None) -> bool:
    """
    Query or set whether JAX should be used.

    Parameters
    ----------
    flag
        Optional boolean to force enabling/disabling JAX at runtime.

    Returns
    -------
    bool
        ``True`` if JAX is available and enabled, ``False`` otherwise.
    """
    global _USE_JAX
    if flag is not None:
        if flag and not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed; cannot enable JAX backend.")
        _USE_JAX = bool(flag)
    return bool(_USE_JAX and JAX_AVAILABLE)


def set_backend(backend: Union[str, bool, None]) -> str:
    """
    Set the active array backend.

    Parameters
    ----------
    backend
        ``"numpy"``/``False`` selects NumPy, ``"jax"``/``True`` selects JAX,
        and ``"auto"``/``None`` keeps the current choice. The function raises if
        JAX is requested but not installed.

    Returns
    -------
    str
        Name of the backend that ended up being active (``"numpy"`` or ``"jax"``).
    """
    if isinstance(backend, bool):
        target = backend
    elif backend is None:
        target = use_jax()
    elif isinstance(backend, str):
        normalized = backend.strip().lower()
        if normalized in {"jax"}:
            target = True
        elif normalized in {"numpy", "np"}:
            target = False
        elif normalized in {"auto", ""}:
            target = use_jax()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'numpy', 'jax', or 'auto'.")
    else:
        raise TypeError("backend must be a string, boolean, or None.")

    use_jax(target)
    os.environ["PYNAMIT_USE_JAX"] = "1" if target else "0"
    return "jax" if target else "numpy"


def get_array_module(*arrays: Any) -> types.ModuleType:
    """
    Return the active array module (NumPy or JAX NumPy).

    If explicit ``arrays`` are provided, the module is inferred from the
    first JAX array. This allows mixed inputs to favour the JAX backend
    when any argument is already a JAX Array.
    """
    if arrays and JAX_AVAILABLE:
        for array in arrays:
            if isinstance(array, _jax_array_type):
                return _jax_namespace  # type: ignore[return-value]
    return _jax_namespace if use_jax() else _np  # type: ignore[return-value]


def to_jax(array: Any) -> Any:
    """Convert ``array`` to a JAX device array when JAX is enabled."""
    if not use_jax():
        return array
    if JAX_AVAILABLE and _jax_device_put is not None:
        return _jax_device_put(array)
    return array


def to_numpy(array: Any) -> Any:
    """Convert ``array`` to a NumPy ``ndarray`` irrespective of backend."""
    if JAX_AVAILABLE and isinstance(array, _jax_array_type):
        return _np.asarray(array)
    return _np.asarray(array)


def asarray(array: Any, dtype: Any = None) -> Any:
    """Backend-aware ``asarray`` helper."""
    module = get_array_module(array)
    return module.asarray(array, dtype=dtype) if dtype is not None else module.asarray(array)


def _identity_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def jit(func=None, *jit_args, **jit_kwargs):
    """Wrapper around ``jax.jit`` that no-ops when JAX is disabled."""
    if not JAX_AVAILABLE:
        if func is None:
            return _identity_decorator
        return func

    def decorator(fn):
        if use_jax():
            return _jax_jit(fn, *jit_args, **jit_kwargs)
        return fn

    if func is None:
        return decorator
    return decorator(func)


def vmap(func=None, *vmap_args, **vmap_kwargs):
    """Wrapper around ``jax.vmap`` that requires the JAX backend."""
    if not (JAX_AVAILABLE and use_jax()):
        raise RuntimeError("JAX vmap requested but the JAX backend is not enabled.")

    if func is None:
        return lambda fn: _jax_vmap(fn, *vmap_args, **vmap_kwargs)

    return _jax_vmap(func, *vmap_args, **vmap_kwargs)


class _ArrayModuleProxy(types.SimpleNamespace):
    """Proxy that forwards attribute access to the active array module."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - thin delegation
        module = get_array_module()
        return getattr(module, item)

    def __dir__(self) -> Iterable[str]:  # pragma: no cover - delegation only
        return dir(get_array_module())

    def __repr__(self) -> str:  # pragma: no cover - representational
        module = get_array_module()
        return f"<ArrayModuleProxy for {module.__name__}>"


xp = _ArrayModuleProxy()

__all__ = [
    "JAX_AVAILABLE",
    "xp",
    "use_jax",
    "set_backend",
    "get_array_module",
    "to_jax",
    "to_numpy",
    "asarray",
    "jit",
    "vmap",
]
