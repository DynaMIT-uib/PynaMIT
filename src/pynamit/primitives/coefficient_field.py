"""Coefficient-backed field values."""

from typing import Any

import numpy as np

from pynamit.primitives.field_space import FieldSpace


class CoefficientField:
    """Realized field coefficients in a ``FieldSpace``.

    This is the value-carrying counterpart to ``FieldSpace``. It validates
    coefficient length and applies the field-space mean-free projector, but it
    deliberately does not own grid projection or grid evaluation machinery.
    """

    def __init__(
        self,
        field_space: FieldSpace,
        coeffs: Any,
        *,
        name: str | None = None,
    ):
        """Initialize a coefficient-backed field."""
        if not isinstance(field_space, FieldSpace):
            raise TypeError("CoefficientField requires a FieldSpace.")
        self.field_space = field_space
        self.coeffs = self.field_space.validate_coefficients(
            self.field_space.project_mean_free(coeffs),
            name=name or f"{self.__class__.__name__}.coeffs",
        )

    @property
    def kind(self):
        """Return the underlying basis family identifier."""
        return self.field_space.kind

    @property
    def basis(self):
        """Return the storage basis for this coefficient field."""
        return self.field_space.basis

    @property
    def field_type(self):
        """Return whether coefficients represent a scalar or tangential field."""
        return self.field_space.field_type

    @property
    def mean_free(self):
        """Return whether the field space enforces zero-mean intent."""
        return self.field_space.mean_free

    @property
    def coefficient_length(self):
        """Return the flattened coefficient count."""
        return self.field_space.coefficient_length

    @property
    def signature(self):
        """Return the structural field-space signature."""
        return self.field_space.signature

    def __array__(self, dtype=None):
        """Return coefficients for NumPy coercion."""
        return np.asarray(self.coeffs, dtype=dtype)
