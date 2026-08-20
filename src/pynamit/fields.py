"""Coefficient-space descriptors and realized field values."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from kompe import ScalarBasis, SurfaceDifferentialBasis
from kompe.math import get_array_module


@dataclass(frozen=True)
class FieldSpace:
    """Describe the coefficient space for one field.

    ``FieldSpace`` deliberately carries no values and does not evaluate
    fields on grids. It is the structural metadata shared by time-series
    storage: the coefficient basis, whether the field is scalar
    or tangential, and whether stored values should satisfy a mean-free
    gauge.
    """

    basis: ScalarBasis
    field_type: str = "scalar"
    mean_free: bool | None = None

    def __post_init__(self):
        """Validate field-space metadata."""
        if self.field_type not in {"scalar", "tangential"}:
            raise ValueError("field_type must be either 'scalar' or 'tangential'.")
        if not isinstance(self.basis, ScalarBasis):
            raise TypeError("FieldSpace basis must be a Kompe ScalarBasis.")
        self.basis.validate_metadata()
        if self.mean_free is None:
            mean_free = (
                self.basis.omits_constant_mode()
                if isinstance(self.basis, SurfaceDifferentialBasis)
                else False
            )
        else:
            mean_free = bool(self.mean_free)
        object.__setattr__(self, "mean_free", mean_free)
        if (self.field_type == "tangential" or self.mean_free) and not isinstance(
            self.basis, SurfaceDifferentialBasis
        ):
            raise TypeError(
                f"A {self.field_type} FieldSpace with mean_free={self.mean_free} "
                "requires a SurfaceDifferentialBasis."
            )

    @property
    def kind(self):
        """Return the underlying basis-family identifier."""
        return self.basis.kind

    @property
    def index_names(self):
        """Return coefficient index names."""
        return self.basis.index_names

    @property
    def index_arrays(self):
        """Return per-coefficient index arrays."""
        return self.basis.index_arrays

    @property
    def index_length(self):
        """Return scalar coefficient count."""
        return int(self.basis.index_length)

    @property
    def component_count(self):
        """Return coefficient component count for this field type."""
        return 2 if self.field_type == "tangential" else 1

    @property
    def coefficient_length(self):
        """Return flattened coefficient count for one variable."""
        return self.component_count * self.index_length

    @property
    def coefficient_shape(self):
        """Return canonical coefficient array shape for one variable."""
        if self.field_type == "scalar":
            return (self.index_length,)
        return (self.component_count, self.index_length)

    @property
    def signature(self):
        """Return coefficient-layout and storage-policy identity."""
        return (
            self.basis.coefficient_space_signature,
            self.field_type,
            bool(self.mean_free),
        )

    def multiindex_arrays(self):
        """Return arrays for xarray's coefficient axis."""
        if self.field_type == "scalar":
            return list(self.index_arrays)
        return [np.tile(values, self.component_count) for values in self.index_arrays]

    def project_mean_free(self, coeffs, *, name="coefficients"):
        """Apply this space's mean-free coefficient policy."""
        array = self.validate_coefficients(coeffs, name=name)
        if not self.mean_free:
            return array

        if self.field_type == "scalar":
            return self.basis.project_scalar_mean_free(array)
        return self.basis.project_helmholtz_mean_free(array)

    def validate_coefficients(self, coeffs, *, name="coefficients"):
        """Return coefficients as an array after length validation."""
        xp = get_array_module(coeffs)
        array = xp.asarray(coeffs)
        if array.size != self.coefficient_length:
            raise ValueError(
                f"{name} has {array.size} coefficients, expected "
                f"{self.coefficient_length} for {self.field_type} "
                f"{self.kind} field space."
            )
        return array.reshape(self.coefficient_shape)


class FieldCoefficients:
    """Realized coefficient values in a :class:`FieldSpace`.

    The container owns its values, validates their shape, and applies
    the field space's gauge policy. Sampling and projection remain the
    coefficient basis's responsibility.
    """

    def __init__(self, field_space: FieldSpace, coeffs: Any, *, name: str | None = None):
        """Initialize owned field coefficients."""
        if not isinstance(field_space, FieldSpace):
            raise TypeError("FieldCoefficients requires a FieldSpace.")
        self.field_space = field_space
        field_name = name or f"{self.__class__.__name__}.array"
        # JAX on CPU may share a NumPy buffer instead of copying it.
        if isinstance(coeffs, np.ndarray) and get_array_module(coeffs) is not np:
            coeffs = np.array(coeffs, copy=True)
        array = self.field_space.project_mean_free(coeffs, name=field_name)
        if isinstance(array, np.ndarray):
            array = np.array(array, copy=True)
            array.setflags(write=False)
        self._array = array

    @property
    def array(self):
        """Return coefficients in canonical shaped form."""
        return self._array

    def __repr__(self):
        """Summarize coefficients without printing the full array."""
        return (
            f"FieldCoefficients(field_space={self.field_space!r}, "
            f"shape={self.array.shape}, dtype={self.array.dtype})"
        )

    def to_vector(self):
        """Return coefficients as a flat operator-compatible vector."""
        return self.array.reshape(-1)

    def __array__(self, dtype=None):
        """Return coefficients for NumPy coercion."""
        return np.asarray(self.array, dtype=dtype)


__all__ = ["FieldCoefficients", "FieldSpace"]
