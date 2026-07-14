"""Background geomagnetic fields and local magnetic geometry."""

from .field_evaluation import MagneticFieldEvaluation
from .main_field import MainField, decimal_year, is_dipole_kind, normalize_main_field_kind


__all__ = [
    "MagneticFieldEvaluation",
    "MainField",
    "decimal_year",
    "is_dipole_kind",
    "normalize_main_field_kind",
]
