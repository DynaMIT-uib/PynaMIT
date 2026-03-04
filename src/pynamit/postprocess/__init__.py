"""Postprocessing helpers shared by visualization and analysis code."""

from .grid_evaluation import (
    decode_conductance_dataset_to_grids,
    decode_conductance_entry_to_grids,
    evaluate_scalar_coeffs_to_grid,
    evaluate_tangential_coeffs_to_grid_components,
    get_scalar_grid_evaluation_matrix,
    get_tangential_grid_component_matrices,
    load_netcdf_dataarray,
    load_netcdf_dataset,
)
from .ground_response import (
    GroundMagneticResponseOperators,
    build_ground_magnetic_response_operators,
)
from .results_operators import (
    PoloidalResultsOperators,
    build_poloidal_results_operators,
)

__all__ = [
    "GroundMagneticResponseOperators",
    "build_ground_magnetic_response_operators",
    "PoloidalResultsOperators",
    "build_poloidal_results_operators",
    "decode_conductance_dataset_to_grids",
    "decode_conductance_entry_to_grids",
    "evaluate_scalar_coeffs_to_grid",
    "evaluate_tangential_coeffs_to_grid_components",
    "get_scalar_grid_evaluation_matrix",
    "get_tangential_grid_component_matrices",
    "load_netcdf_dataarray",
    "load_netcdf_dataset",
]
