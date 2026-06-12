"""Field coefficient values."""

from typing import Any

import numpy as np

from pynamit.primitives.field_space import FieldSpace


class FieldCoefficients:
    """Realized field coefficients in a ``FieldSpace``.

    This is the value-carrying counterpart to ``FieldSpace``. It checks
    coefficient length, stores the canonical coefficient shape, and
    applies the field-space coefficient policy. It does not own grid
    projection or grid evaluation.
    """

    def __init__(
        self,
        field_space: FieldSpace,
        coeffs: Any,
        *,
        name: str | None = None,
    ):
        """Initialize field coefficients."""
        if not isinstance(field_space, FieldSpace):
            raise TypeError("FieldCoefficients requires a FieldSpace.")
        self.field_space = field_space
        field_name = name or f"{self.__class__.__name__}.coeffs"
        coeffs = self.field_space.validate_coefficients(coeffs, name=field_name)
        self.coeffs = self.field_space.validate_coefficients(
            self.field_space.project_mean_free(coeffs),
            name=field_name,
        )

    @property
    def kind(self):
        """Return the underlying representation family identifier."""
        return self.field_space.kind

    @property
    def representation(self):
        """Return the storage representation."""
        return self.field_space.representation

    @property
    def field_type(self):
        """Return scalar or tangential field type."""
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
    def coefficient_shape(self):
        """Return the canonical coefficient array shape."""
        return self.field_space.coefficient_shape

    @property
    def signature(self):
        """Return the structural field-space signature."""
        return self.field_space.signature

    @property
    def array(self):
        """Return coefficients in canonical shaped form."""
        return self.coeffs

    def to_vector(self):
        """Return coefficients as a flat operator-compatible vector."""
        return np.asarray(self.coeffs).reshape(-1)

    def __array__(self, dtype=None):
        """Return coefficients for NumPy coercion."""
        return np.asarray(self.array, dtype=dtype)
