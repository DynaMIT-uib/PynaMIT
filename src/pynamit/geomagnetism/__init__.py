"""Background geomagnetic fields and local magnetic geometry."""

from .main_field import (
    MainField,
    decimal_year,
    horizontal_coordinate_system_for_kind,
    is_dipole_kind,
    normalize_main_field_kind,
)

__all__ = [
    "MainField",
    "decimal_year",
    "horizontal_coordinate_system_for_kind",
    "is_dipole_kind",
    "normalize_main_field_kind",
]
