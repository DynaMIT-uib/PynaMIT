"""Coefficient-space descriptors for fields."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldSpace:
    """Describe the coefficient space for one field.

    ``FieldSpace`` deliberately carries no values and does not evaluate
    fields on grids. It is the structural metadata shared by time-series
    storage and coefficient transforms: the basis, whether the field is
    scalar or tangential, and whether stored coefficients should satisfy
    a mean-free gauge.
    """

    basis: Any
    field_type: str = "scalar"
    mean_free: bool = False

    def __post_init__(self):
        """Validate field-space metadata."""
        if self.field_type not in {"scalar", "tangential"}:
            raise ValueError("field_type must be either 'scalar' or 'tangential'.")
        if not hasattr(self.basis, "validate_metadata"):
            raise TypeError("FieldSpace basis must expose basis metadata.")
        self.basis.validate_metadata()

    @classmethod
    def from_basis(cls, basis, field_type="scalar", mean_free=None):
        """Construct a field space from an existing basis."""
        if isinstance(basis, cls):
            if basis.field_type != field_type:
                raise ValueError(
                    f"FieldSpace already has field_type={basis.field_type!r}, "
                    f"not {field_type!r}."
                )
            if mean_free is not None and basis.mean_free != bool(mean_free):
                raise ValueError(
                    f"FieldSpace already has mean_free={basis.mean_free!r}, "
                    f"not {bool(mean_free)!r}."
                )
            return basis
        if mean_free is None:
            mean_free = bool(getattr(basis, "mean_free", False))
        return cls(basis=basis, field_type=field_type, mean_free=bool(mean_free))

    @property
    def kind(self):
        """Return the underlying basis family identifier."""
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
    def signature(self):
        """Return a stable-ish cache signature for this field space."""
        basis_signature = getattr(self.basis, "coefficient_space_signature", None)
        if basis_signature is None:
            basis_signature = getattr(self.basis, "signature", id(self.basis))
        return (basis_signature, self.field_type, bool(self.mean_free))

    def multiindex_arrays(self):
        """Return arrays for xarray's coefficient axis."""
        if self.field_type == "scalar":
            return list(self.index_arrays)
        return [
            np.tile(values, self.component_count) for values in self.index_arrays
        ]

    def project_mean_free(self, coeffs):
        """Apply this space's mean-free coefficient policy."""
        array = np.asarray(coeffs)
        if not self.mean_free:
            return array

        if self.field_type == "scalar":
            projector = getattr(self.basis, "project_scalar_mean_free", None)
        else:
            projector = getattr(self.basis, "project_helmholtz_mean_free", None)

        return projector(array) if callable(projector) else array

    def validate_coefficients(self, coeffs, *, name="coefficients"):
        """Return coefficients as an array after length validation."""
        array = np.asarray(coeffs)
        if array.size != self.coefficient_length:
            raise ValueError(
                f"{name} has {array.size} coefficients, expected "
                f"{self.coefficient_length} for {self.field_type} "
                f"{self.kind} field space."
            )
        return array
