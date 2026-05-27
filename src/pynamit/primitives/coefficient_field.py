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
        field_space_or_basis: Any,
        coeffs: Any,
        field_type: str | None = None,
        *,
        name: str | None = None,
    ):
        """Initialize a coefficient-backed field."""
        self.field_space = self._normalize_field_space(field_space_or_basis, field_type)
        self.basis = self.field_space.basis
        self.field_type = self.field_space.field_type
        self.mean_free = self.field_space.mean_free
        self.coeffs = self.field_space.validate_coefficients(
            self.field_space.project_mean_free(coeffs),
            name=name or f"{self.__class__.__name__}.coeffs",
        )

    @staticmethod
    def _normalize_field_space(field_space_or_basis: Any, field_type: str | None = None):
        """Return a ``FieldSpace`` from either a field space or a raw basis."""
        if isinstance(field_space_or_basis, FieldSpace):
            effective_field_type = (
                field_space_or_basis.field_type if field_type is None else field_type
            )
        else:
            effective_field_type = "scalar" if field_type is None else field_type
        return FieldSpace.from_basis(field_space_or_basis, field_type=effective_field_type)

    @property
    def kind(self):
        """Return the underlying basis family identifier."""
        return self.field_space.kind

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
