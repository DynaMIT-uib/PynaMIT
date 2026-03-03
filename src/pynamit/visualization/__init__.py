"""Visualization tools for PynaMIT.

Optional plotting dependencies such as ``cartopy`` are not required for the
operator/grid-evaluation helpers. Import those plotting entry points only when
their dependencies are available.
"""

from pynamit.postprocess import (
    PoloidalResultsOperators,
    build_poloidal_results_operators,
    decode_conductance_dataset_to_grids,
    decode_conductance_entry_to_grids,
    evaluate_scalar_coeffs_to_grid,
    evaluate_tangential_coeffs_to_grid_components,
    get_scalar_grid_evaluation_matrix,
    get_tangential_grid_component_matrices,
    load_netcdf_dataarray,
    load_netcdf_dataset,
)

__all__ = [
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

try:
    from .plots import debugplot, globalplot
except ModuleNotFoundError:
    pass
else:
    __all__.extend(["debugplot", "globalplot"])

try:
    from .pynameye import PynamEye
except ModuleNotFoundError:
    pass
else:
    __all__.append("PynamEye")

try:
    from .input_vs_interpolated import plot_input_vs_interpolated
except ModuleNotFoundError:
    pass
else:
    __all__.append("plot_input_vs_interpolated")
