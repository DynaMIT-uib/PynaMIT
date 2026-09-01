"""Background geomagnetic fields and local magnetic geometry."""

from .main_field import (
    MainField,
    horizontal_coordinate_system_for_kind,
    is_dipole_kind,
    normalize_main_field_kind,
)

__all__ = [
    "MainField",
    "horizontal_coordinate_system_for_kind",
    "is_dipole_kind",
    "normalize_main_field_kind",
]
